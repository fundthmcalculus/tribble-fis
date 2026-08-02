"""Binary space partitioning tree of Gaussian fuzzy classifiers.

This module implements candidate #2 from the iris_v2 improvement goal:
recursively split any region whose local model is not accurate enough into two
smaller regions, each with its own fuzzy classifier, up to a bounded depth.

How it works
------------
Each node fits a :class:`CalibratedGaussianFuzzyClassifier` on the rows routed to
it and measures its own resubstitution accuracy:

* If the node is accurate enough (``>= accuracy_threshold``), pure, too small
  (``< min_samples_split``), or at ``max_depth``, it becomes a **leaf** holding
  that fuzzy classifier.
* Otherwise the node picks the single **axis-aligned split** ``feature <= tau``
  that most reduces class impurity (Gini information gain -- the CART criterion)
  and recurses on the two sides.

An axis-aligned threshold keeps every split human-readable
("petal_length_cm <= 4.9"), so the whole tree reads as a short list of
interpretable rules whose leaves are compact fuzzy rule bases -- a fuzzy
model tree.  Prediction routes a sample down the thresholds to its leaf and
returns that leaf's fuzzy prediction.

References
----------
* Breiman, Friedman, Olshen & Stone, *Classification and Regression Trees*
  (1984) -- the Gini-gain axis-aligned split used at each node.
* Quinlan, "Learning with continuous classes" (1992) / model trees -- trees
  whose leaves hold a model rather than a constant.
* Kohavi, "Scaling Up the Accuracy of Naive-Bayes Classifiers: a
  Decision-Tree Hybrid" (NBTree, 1996) -- the direct ancestor of this design:
  a decision tree with naive-Bayes leaves, split only where it helps.

Caveat
------
Splitting reduces *model bias* in regions where one Gaussian rule base cannot
fit the local shape.  On iris_v2 the per-class densities are already close to
Gaussian and the classes simply overlap, so extra splits mostly chase noise;
the tree therefore matches the ~0.826 Bayes ceiling rather than beating it.  The
``accuracy_threshold`` and ``min_gain`` guards keep it from over-splitting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from .calibrated_fuzzy_classifier import CalibratedGaussianFuzzyClassifier


class _Node:
    __slots__ = ("model", "feature", "threshold", "left", "right", "depth", "n", "leaf")

    def __init__(self):
        self.model = None
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.depth = 0
        self.n = 0
        self.leaf = True


def _gini(y_codes: np.ndarray, n_classes: int) -> float:
    if len(y_codes) == 0:
        return 0.0
    counts = np.bincount(y_codes, minlength=n_classes)
    p = counts / counts.sum()
    return 1.0 - float(np.sum(p * p))


class BSPFuzzyTreeClassifier(BaseEstimator, ClassifierMixin):
    """A depth-bounded binary partition tree with fuzzy-classifier leaves.

    Args:
        accuracy_threshold: A node that already classifies its own rows at least
            this well is kept as a leaf (no split).
        max_depth: Hard cap on recursion depth (the goal suggested 10).
        min_samples_split: A node with fewer rows than this becomes a leaf.
        min_gain: Minimum Gini information gain required to accept a split;
            prevents splitting on noise.
        n_split_quantiles: Number of candidate thresholds probed per feature
            (feature quantiles), trading split precision for fit speed.
        n_gaussians, top_p: Passed through to each leaf's fuzzy classifier.
        random_state: Seed for the leaf classifiers.
    """

    def __init__(
        self,
        accuracy_threshold=0.90,
        max_depth=10,
        min_samples_split=200,
        min_gain=1e-3,
        n_split_quantiles=16,
        n_gaussians=1,
        top_p=1.0,
        random_state=42,
    ):
        self.accuracy_threshold = accuracy_threshold
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.n_split_quantiles = n_split_quantiles
        self.n_gaussians = n_gaussians
        self.top_p = top_p
        self.random_state = random_state

    def _make_leaf_model(self):
        return CalibratedGaussianFuzzyClassifier(
            n_gaussians=self.n_gaussians, top_p=self.top_p, random_state=self.random_state
        )

    def _best_split(self, X_df, y_codes):
        """Best axis-aligned (feature, threshold, gain) by Gini information gain."""
        n = len(y_codes)
        parent = _gini(y_codes, self.n_classes_)
        best = (None, None, 0.0)
        for feat in self.feature_names_in_:
            col = X_df[feat].values
            qs = np.quantile(col, np.linspace(0.05, 0.95, self.n_split_quantiles))
            for tau in np.unique(qs):
                left = col <= tau
                nl = int(left.sum())
                if nl == 0 or nl == n:
                    continue
                gl = _gini(y_codes[left], self.n_classes_)
                gr = _gini(y_codes[~left], self.n_classes_)
                gain = parent - (nl / n) * gl - ((n - nl) / n) * gr
                if gain > best[2]:
                    best = (feat, float(tau), gain)
        return best

    def _build(self, X_df, y_series, depth) -> _Node:
        node = _Node()
        node.depth = depth
        node.n = len(y_series)

        model = self._make_leaf_model().fit(X_df, y_series)
        node.model = model

        y_codes = self._encode(y_series.values)
        # Stop conditions: pure, too small, at max depth.
        if (node.n < self.min_samples_split
                or depth >= self.max_depth
                or y_series.nunique() < 2):
            return node

        # Accurate enough already? Keep as a leaf.
        acc = float(np.mean(model.predict(X_df) == y_series.values))
        if acc >= self.accuracy_threshold:
            return node

        feat, tau, gain = self._best_split(X_df, y_codes)
        if feat is None or gain < self.min_gain:
            return node  # no useful split -> stay a leaf

        left_mask = X_df[feat].values <= tau
        # Guard against degenerate partitions.
        if left_mask.all() or (~left_mask).all():
            return node

        node.leaf = False
        node.feature = feat
        node.threshold = tau
        node.model = None  # internal nodes route; they don't classify
        node.left = self._build(
            X_df[left_mask].reset_index(drop=True),
            y_series[left_mask].reset_index(drop=True), depth + 1)
        node.right = self._build(
            X_df[~left_mask].reset_index(drop=True),
            y_series[~left_mask].reset_index(drop=True), depth + 1)
        return node

    def _encode(self, labels):
        return np.array([self._cls_to_code[l] for l in labels], dtype=int)

    def fit(self, X, y):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y_series = y if isinstance(y, pd.Series) else pd.Series(np.asarray(y))
        X_df = X_df.reset_index(drop=True)
        y_series = y_series.reset_index(drop=True)

        self.feature_names_in_ = X_df.columns.tolist()
        self.classes_ = np.unique(y_series.values)
        self.n_classes_ = len(self.classes_)
        self._cls_to_code = {c: i for i, c in enumerate(self.classes_)}

        self.root_ = self._build(X_df, y_series, depth=0)
        self.is_fitted_ = True
        return self

    def _route(self, X_df):
        """Return a leaf-node array, one entry per row of X_df."""
        leaves = np.empty(len(X_df), dtype=object)
        idx_all = np.arange(len(X_df))

        def descend(node, idx):
            if node.leaf:
                leaves[idx] = node
                return
            go_left = X_df[node.feature].values[idx] <= node.threshold
            descend(node.left, idx[go_left])
            descend(node.right, idx[~go_left])

        descend(self.root_, idx_all)
        return leaves

    def _as_df(self, X):
        return X.reset_index(drop=True) if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_in_)

    def predict(self, X):
        check_is_fitted(self)
        X_df = self._as_df(X)
        leaves = self._route(X_df)
        preds = np.empty(len(X_df), dtype=object)
        for node in {id(l): l for l in leaves}.values():
            mask = np.array([leaves[i] is node for i in range(len(X_df))])
            preds[mask] = node.model.predict(X_df[mask])
        return preds

    @property
    def n_leaves_(self) -> int:
        def count(node):
            return 1 if node.leaf else count(node.left) + count(node.right)
        check_is_fitted(self)
        return count(self.root_)

    @property
    def max_depth_reached_(self) -> int:
        def d(node):
            return node.depth if node.leaf else max(d(node.left), d(node.right))
        check_is_fitted(self)
        return d(self.root_)

    def describe(self) -> str:
        """Human-readable dump of the split structure."""
        check_is_fitted(self)
        lines = []

        def walk(node, prefix):
            if node.leaf:
                lines.append(f"{prefix}leaf(n={node.n}, classes={list(node.model.classes_)})")
            else:
                lines.append(f"{prefix}if {node.feature} <= {node.threshold:.4g}:")
                walk(node.left, prefix + "  ")
                lines.append(f"{prefix}else:")
                walk(node.right, prefix + "  ")

        walk(self.root_, "")
        return "\n".join(lines)
