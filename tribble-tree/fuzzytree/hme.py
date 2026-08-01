"""Hierarchical mixture of fuzzy experts (HME).

This composes *multiple sub-FIS into one tree*. Internal nodes are fuzzy **gating**
sub-systems that softly route a sample toward their children; leaves are full TSK
**expert** sub-FIS. The final output is the gated blend over all leaves:

    y_hat(x) = sum_leaf  g_leaf(x) * expert_leaf(x)          (regression)
    p(c|x)   = sum_leaf  g_leaf(x) * expert_leaf.proba(c|x)  (classification)

where ``g_leaf(x)`` is the product of the (per-node, partition-of-unity) gate
weights along the root->leaf path, so the leaf gates sum to 1 and the blend is
convex. This is the classic Hierarchical Mixture of Experts (Jordan & Jacobs,
1994) with *fuzzy* gates and *fuzzy* experts.

Inferring & building the tree
-----------------------------
The gate structure -- which variable routes at each node and into how many fuzzy
regimes -- is inferred by the *same* criterion-driven recursive partitioning that
``builder.build_tree`` already performs (variance reduction for regression,
ambiguity/​info-gain for classification), with full user override via
``VariablePlan`` (pin a routing variable to a node, fix a per-level order, or
exclude variables). We then:

    1. reuse ``build_tree`` to get the gate topology (a ``FuzzyTreeNode`` tree),
    2. turn each node's linguistic terms into a partition-of-unity gate,
    3. compute each training sample's leaf responsibilities (path gate product),
    4. assign each sample to its argmax-responsibility leaf and fit that leaf's
       expert sub-FIS on the assigned subset (hard-assignment / Viterbi build),
    5. blend experts by the soft gates at predict time.

Because the gate structure is exactly the fuzzy-decision-tree structure, the plain
``FuzzyRegressionTree`` is the special case of this model where every expert is a
single TSK consequent instead of a full sub-FIS.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from tribblefis.gauss_math import calculate_gaussian_correlation, take_top_features
from tribblefis.gaussian_classifier import MixtureOfGaussiansFuzzyClassifier
from tribblefis.gaussian_regressor import MixtureOfGaussiansFuzzyRegressor
from tribblefis.regression import partition_output

from .builder import build_tree
from .firing import DEFAULT_T_NORM
from .plan import VariablePlan

_EPS = 1e-12


# --------------------------------------------------------------------------
# Fallback "experts" for leaves with too few / single-class training samples
# --------------------------------------------------------------------------
class _ConstantRegressor:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, X):
        return np.full(len(X), self.value)


class _ConstantClassifier:
    def __init__(self, proba: np.ndarray):
        self.proba_ = np.asarray(proba, dtype=float)

    def predict_proba_aligned(self, X):
        return np.tile(self.proba_, (len(X), 1))


class _AlignedClassifier:
    """Wrap a fitted classifier so ``predict_proba`` columns line up with the
    model's global class order (a leaf subset may miss some classes)."""

    def __init__(self, clf, global_classes):
        self.clf = clf
        self.global_classes = list(global_classes)
        local = list(clf.classes_)
        self._cols = [local.index(c) if c in local else None for c in self.global_classes]

    def predict_proba_aligned(self, X):
        p = self.clf.predict_proba(X)
        out = np.zeros((len(X), len(self.global_classes)))
        for gi, ci in enumerate(self._cols):
            if ci is not None:
                out[:, gi] = p[:, ci]
        return out


# --------------------------------------------------------------------------
# Gate responsibilities
# --------------------------------------------------------------------------
def compute_responsibilities(root, X_df: pd.DataFrame, n_leaves: int) -> np.ndarray:
    """(n_samples, n_leaves) leaf responsibilities = product of partition-of-unity
    gate weights along each root->leaf path. Rows sum to 1.

    The product here is deliberate and is NOT a configurable t-norm. Each node
    normalises its gates to sum to 1, and a product of such factors along a path
    keeps the leaf responsibilities summing to 1 -- that partition of unity is
    precisely what makes this a mixture of experts rather than an arbitrary
    weighting. min, Lukasiewicz or Hamacher would each break the normalisation
    and the leaves would stop being mixture weights. So unlike the fuzzy tree,
    the HME gate is not a sweepable axis; it is fixed by the model's semantics.
    """
    n = len(X_df)
    R = np.zeros((n, n_leaves), dtype=float)

    def recurse(node, w: np.ndarray) -> None:
        if node.is_leaf:
            R[:, node.leaf_id] = w
            return
        col = X_df[node.split_var].to_numpy(dtype=float)
        M = np.column_stack([mf.evaluate(col) for _, mf in node.terms])  # (n, K)
        totals = M.sum(axis=1)
        gates = np.zeros_like(M)
        ok = totals > _EPS
        gates[ok] = M[ok] / totals[ok, None]
        # A sample that fires no gate (only possible with degenerate terms) is
        # routed uniformly rather than dropped.
        gates[~ok] = 1.0 / M.shape[1]
        for k, child in enumerate(node.children):
            recurse(child, w * gates[:, k])

    recurse(root, np.ones(n, dtype=float))
    return R


def _leaves(root):
    return list(root.iter_leaves())


# --------------------------------------------------------------------------
# Base estimator
# --------------------------------------------------------------------------
class _BaseHierarchicalExperts(BaseEstimator):
    """Shared structure inference, gate building, and expert assignment."""

    def __init__(
        self,
        variable_plan=None,
        criterion="variance",
        top_n=-1,
        top_p=0.95,
        max_depth=3,
        n_gate_terms=2,
        min_soft_count=20.0,
        min_gain=1e-3,
        max_leaves=32,
        n_score_buckets=3,
        min_expert_samples=30,
        responsibility_threshold=0.2,
        expert_kwargs=None,
        random_state=42,
    ):
        self.variable_plan = variable_plan
        self.criterion = criterion
        self.top_n = top_n
        self.top_p = top_p
        self.max_depth = max_depth
        self.n_gate_terms = n_gate_terms
        self.min_soft_count = min_soft_count
        self.min_gain = min_gain
        self.max_leaves = max_leaves
        self.n_score_buckets = n_score_buckets
        self.min_expert_samples = min_expert_samples
        self.responsibility_threshold = responsibility_threshold
        self.expert_kwargs = expert_kwargs
        self.random_state = random_state

    def _expert_training_index(self, R, leaf_id):
        """Samples used to train a leaf's expert: those whose responsibility for
        this leaf exceeds the threshold, plus any sample this leaf wins outright.
        Overlapping (soft-inclusion) training sets give each expert enough data
        and cover the gate boundaries, so the soft blend interpolates cleanly --
        far better than a hard argmax partition that starves boundary regions."""
        r = R[:, leaf_id]
        mask = r >= self.responsibility_threshold
        mask |= R.argmax(axis=1) == leaf_id
        return np.where(mask)[0]

    def _resolve_plan(self) -> VariablePlan:
        if self.variable_plan is not None:
            return self.variable_plan
        return VariablePlan(
            criterion=self.criterion,
            max_depth=self.max_depth,
            default_n_terms=self.n_gate_terms,
            max_terms_per_var=max(self.n_gate_terms, 2),
        )

    def _prepare(self, X):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
            X_df = X.reset_index(drop=True).copy()
        else:
            self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)
        return X_df

    def _build_gate_structure(self, X_df, y_value, y_bucket):
        """Select routing (gate) features and infer the gate tree via build_tree.

        Gates route on the top differentiating features; experts (fit later) use
        the full feature set via their own sub-FIS selection.
        """
        self.feature_differentiators_ = calculate_gaussian_correlation(
            X_df, pd.Series(y_bucket)
        )
        _, self.gate_features_ = take_top_features(
            self.feature_differentiators_, top_p=self.top_p, top_n=self.top_n
        )
        plan = self._resolve_plan()
        root, n_leaves = build_tree(
            X_df[self.gate_features_],
            y_value,
            y_bucket,
            plan,
            t_norm_name=DEFAULT_T_NORM,
            min_soft_count=self.min_soft_count,
            min_gain=self.min_gain,
            max_leaves=self.max_leaves,
        )
        return root, n_leaves

    def _expert_kwargs(self) -> dict:
        return dict(self.expert_kwargs) if self.expert_kwargs else {}


# --------------------------------------------------------------------------
# Regression HME
# --------------------------------------------------------------------------
class HierarchicalFuzzyExpertsRegressor(_BaseHierarchicalExperts, RegressorMixin):
    """Hierarchical mixture of fuzzy experts for regression.

    Internal nodes softly route on a fuzzy gate variable; each leaf is a
    ``MixtureOfGaussiansFuzzyRegressor`` sub-FIS. The output is the gate-weighted
    blend of the leaf experts.
    """

    def fit(self, X, y):
        X_df = self._prepare(X)
        y_value = np.asarray(y, dtype=float).flatten()

        y_part, _ = partition_output(self.n_score_buckets, pd.Series(y_value, name="y_value"))
        y_bucket = y_part["y_bucket"].to_numpy()

        self.tree_, self.n_leaves_ = self._build_gate_structure(X_df, y_value, y_bucket)

        R = compute_responsibilities(self.tree_, X_df[self.gate_features_], self.n_leaves_)

        base_kwargs = {"n_output_buckets": 3, "tsk_order": "1st", "random_state": self.random_state}
        base_kwargs.update(self._expert_kwargs())

        self.experts_ = {}
        self.leaf_info_ = {}
        for leaf in _leaves(self.tree_):
            idx = self._expert_training_index(R, leaf.leaf_id)
            if len(idx) >= self.min_expert_samples and np.ptp(y_value[idx]) > _EPS:
                expert = MixtureOfGaussiansFuzzyRegressor(**base_kwargs)
                expert.fit(X_df.iloc[idx], y_value[idx])
                feats = list(getattr(expert, "top_features_", []))
                kind = "sub-FIS"
            else:
                # Fallback: responsibility-weighted mean (robust for tiny leaves).
                w = R[:, leaf.leaf_id]
                mean = float((w * y_value).sum() / w.sum()) if w.sum() > _EPS else float(y_value.mean())
                expert = _ConstantRegressor(mean)
                feats = []
                kind = "constant"
            self.experts_[leaf.leaf_id] = expert
            self.leaf_info_[leaf.leaf_id] = {"n_train": int(len(idx)), "features": feats, "kind": kind}

        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        R = compute_responsibilities(self.tree_, X[self.gate_features_], self.n_leaves_)
        y_pred = np.zeros(len(X))
        for leaf_id, expert in self.experts_.items():
            y_pred += R[:, leaf_id] * expert.predict(X)
        return y_pred


# --------------------------------------------------------------------------
# Classification HME
# --------------------------------------------------------------------------
class HierarchicalFuzzyExpertsClassifier(_BaseHierarchicalExperts, ClassifierMixin):
    """Hierarchical mixture of fuzzy experts for classification.

    Internal nodes softly route on a fuzzy gate variable; each leaf is a
    ``MixtureOfGaussiansFuzzyClassifier`` sub-FIS. Class probabilities are the
    gate-weighted blend of the leaf experts' probabilities.
    """

    def __init__(self, *args, criterion="ambiguity", **kwargs):
        super().__init__(*args, criterion=criterion, **kwargs)

    def fit(self, X, y):
        X_df = self._prepare(X)
        y_arr = np.asarray(y).flatten()
        self.classes_ = np.unique(y_arr)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[c] for c in y_arr])

        # Feature ranking + gate structure use the true class labels.
        self.tree_, self.n_leaves_ = self._build_gate_structure(
            X_df, y_idx.astype(float), y_idx
        )

        R = compute_responsibilities(self.tree_, X_df[self.gate_features_], self.n_leaves_)
        n_classes = len(self.classes_)

        base_kwargs = {"top_n": 5, "random_state": self.random_state}
        base_kwargs.update(self._expert_kwargs())

        self.experts_ = {}
        self.leaf_info_ = {}
        for leaf in _leaves(self.tree_):
            idx = self._expert_training_index(R, leaf.leaf_id)
            present = np.unique(y_idx[idx]) if len(idx) else np.array([], dtype=int)
            if len(idx) >= self.min_expert_samples and len(present) >= 2:
                clf = MixtureOfGaussiansFuzzyClassifier(**base_kwargs)
                clf.fit(X_df.iloc[idx], self.classes_[y_idx[idx]])
                expert = _AlignedClassifier(clf, self.classes_)
                feats = list(getattr(clf, "top_features_", []))
                kind = "sub-FIS"
            else:
                # Fallback: responsibility-weighted class distribution.
                w = R[:, leaf.leaf_id]
                proba = np.array(
                    [w[y_idx == c].sum() for c in range(n_classes)], dtype=float
                )
                proba = proba / proba.sum() if proba.sum() > _EPS else np.full(n_classes, 1.0 / n_classes)
                expert = _ConstantClassifier(proba)
                feats = []
                kind = "constant"
            self.experts_[leaf.leaf_id] = expert
            self.leaf_info_[leaf.leaf_id] = {"n_train": int(len(idx)), "features": feats, "kind": kind}

        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        R = compute_responsibilities(self.tree_, X[self.gate_features_], self.n_leaves_)
        proba = np.zeros((len(X), len(self.classes_)))
        for leaf_id, expert in self.experts_.items():
            proba += R[:, leaf_id, None] * expert.predict_proba_aligned(X)
        row = proba.sum(axis=1, keepdims=True)
        row[row <= _EPS] = 1.0
        return proba / row

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
