"""Deconstruct a flat TSK fuzzy model into a user-specified hierarchical tree.

This is a different way to build a hierarchical FIS than `hme.py`'s
top-down `HierarchicalFuzzyExpertsRegressor`. HME infers (or is pinned to) a
gate topology and fits a *brand new* sub-FIS from scratch on each leaf's row
subset. Here instead:

    1. Fit one flat `TribbleRegressor` on *all* features (no row subsetting).
    2. Take a user-supplied `TopologyNode` tree (see `topology.py`).
    3. For each leaf, slice the flat model's already-fitted antecedents down
       to just that leaf's own features -- since every feature model shares
       the same bucket/label keys, this slice *is* the projection of every
       flat rule's antecedent onto the leaf's features. No re-clustering, no
       antecedent refitting.
    4. Re-solve each leaf's own consequent (closed-form ridge, the same
       solver `TribbleRegressor` itself uses) against a leaf target (the
       root target by default, or a caller-supplied per-node target -- see
       `leaf_targets`).
    5. Combine each branch node's children outputs into its own output via a
       fitted affine combiner (`a . children + a0`), solved with the exact
       same ridge machinery in its degenerate single-rule form.

Leaf/branch consequent order (and the *values being combined*) is
independent of the flat model's own `tsk_order` -- the flat fit only
supplies the antecedent structure.

`DeconstructedHierarchicalClassifier` is the classification counterpart:
structurally identical, but a `TribbleClassifier` rule's "consequent" already
IS its class label, so there is no leaf-level consequent to re-solve at all
-- slicing the flat model's antecedents to a leaf's own features already
gives that leaf's per-class probability directly (exactly what
`TribbleClassifier.predict_proba` does over the *full* feature set). Only
the branch combiner needs fitting, once per class; because its output is a
probability, each branch's raw per-class scores are clipped to [0, 1] and
renormalized before being handed to its own parent -- the classification
analogue of the regressor's unconstrained affine combine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from tribblefis.gauss_data import AnomalyParameters, GaussianMixtureModel, resolve_norm_pair
from tribblefis.gauss_math import tsk_firing_strengths
from tribblefis.gaussian_classifier import TribbleClassifier
from tribblefis.gaussian_regressor import TribbleRegressor
from tribblefis.regression import (
    apply_tsk_consequents,
    predict_tsk,
    solve_tsk_consequents,
    solve_tsk_consequents_from_firing,
)

from .hme import _ConstantClassifier, _ConstantRegressor
from .topology import TopologyNode, parse_topology

_ORDER = "1st"


def _as_target(values, index) -> pd.Series:
    return pd.Series(np.asarray(values, dtype=float).reshape(-1), index=index, name="y_value")


class DeconstructedHierarchicalRegressor(BaseEstimator, RegressorMixin):
    """Build a hierarchical FIS by deconstructing one flat `TribbleRegressor`.

    Parameters
    ----------
    flat_regressor_kwargs : dict, optional
        Passed to the internal `TribbleRegressor` used to fit the flat
        antecedent structure (e.g. ``{"n_output_buckets": 5}``).
    l2_reg : float
        Ridge penalty used for every leaf and branch consequent solve.
    order : str
        TSK consequent order ("0th", "1st", "2nd", ...) used for every leaf
        and branch consequent solve. Independent of the flat model's own
        `tsk_order`.
    """

    def __init__(self, flat_regressor_kwargs=None, l2_reg: float = 1e-6, order: str = _ORDER):
        self.flat_regressor_kwargs = flat_regressor_kwargs
        self.l2_reg = l2_reg
        self.order = order

    def fit(self, X, y, topology: dict[str, list[str]], leaf_targets: dict[str, object] | None = None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X = X.reset_index(drop=True)
        y = _as_target(y, X.index)
        leaf_targets = leaf_targets or {}

        self.flat_ = TribbleRegressor(**(self.flat_regressor_kwargs or {}))
        self.flat_.fit(X, y)
        self.norms_ = resolve_norm_pair(
            self.flat_.norm_conorm, self.flat_.t_norm, self.flat_.t_conorm,
            self.flat_.allow_mixed_norms,
        )

        self.root_ = parse_topology(topology, list(X.columns))
        self.node_state_: dict[str, dict] = {}
        self._fit_node(self.root_, X, y, leaf_targets)
        self.is_fitted_ = True
        return self

    def _node_target(self, node: TopologyNode, X: pd.DataFrame, y: pd.Series, leaf_targets: dict) -> pd.Series:
        raw = leaf_targets.get(node.name, y)
        return _as_target(raw, X.index)

    def _fit_node(self, node: TopologyNode, X: pd.DataFrame, y: pd.Series, leaf_targets: dict) -> np.ndarray:
        target = self._node_target(node, X, y, leaf_targets)

        if node.is_leaf:
            return self._fit_leaf(node, X, target)

        child_outputs = {
            child.name: self._fit_node(child, X, y, leaf_targets) for child in node.children
        }
        return self._fit_branch(node, child_outputs, target)

    def _leaf_gmm(self, node: TopologyNode) -> tuple[GaussianMixtureModel, list[str]]:
        feature_models = {
            f: self.flat_.model_.feature_models[f]
            for f in node.own_features
            if f in self.flat_.model_.feature_models
        }
        return GaussianMixtureModel(feature_models=feature_models), list(feature_models.keys())

    def _fit_leaf(self, node: TopologyNode, X: pd.DataFrame, target: pd.Series) -> np.ndarray:
        gmm, top_n_todo = self._leaf_gmm(node)
        if not top_n_todo:
            value = float(target.mean())
            self.node_state_[node.name] = {"kind": "constant", "model": _ConstantRegressor(value)}
            return np.full(len(X), value)

        y_train = pd.DataFrame({"y_value": target.values})
        corr_terms, y_bucket_mean = solve_tsk_consequents(
            X, gmm, top_n_todo, None, y_train,
            n_output_buckets=gmm.n_rules,
            order=self.order, l2_reg=self.l2_reg, basis="raw",
            pin_extremes=False, norms=self.norms_, verbose=False,
        )
        self.node_state_[node.name] = {
            "kind": "leaf", "gmm": gmm, "top_n_todo": top_n_todo,
            "corr_terms": corr_terms, "y_bucket_mean": y_bucket_mean,
        }
        return predict_tsk(
            X, gmm, top_n_todo, y_bucket_mean, corr_terms,
            order=self.order, basis="raw", norms=self.norms_,
        )

    def _fit_branch(self, node: TopologyNode, child_outputs: dict[str, np.ndarray], target: pd.Series) -> np.ndarray:
        children_names = list(child_outputs.keys())
        children_df = pd.DataFrame(child_outputs)
        firing = np.ones((len(children_df), 1))
        y_train = pd.DataFrame({"y_value": target.values})
        corr_terms, y_bucket_mean = solve_tsk_consequents_from_firing(
            firing, [0], children_df, children_names, None, y_train,
            order=self.order, l2_reg=self.l2_reg, basis="raw",
            pin_extremes=False, verbose=False,
        )
        self.node_state_[node.name] = {
            "kind": "branch", "children": children_names,
            "corr_terms": corr_terms, "y_bucket_mean": y_bucket_mean,
        }
        return apply_tsk_consequents(
            children_df, children_names, firing, [0], y_bucket_mean, corr_terms,
            order=self.order, basis="raw",
        )

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.flat_.feature_names_in_)
        X = X.reset_index(drop=True)
        return self._predict_node(self.root_, X)

    def _predict_node(self, node: TopologyNode, X: pd.DataFrame) -> np.ndarray:
        state = self.node_state_[node.name]
        if state["kind"] == "constant":
            return state["model"].predict(X)
        if state["kind"] == "leaf":
            return predict_tsk(
                X, state["gmm"], state["top_n_todo"], state["y_bucket_mean"], state["corr_terms"],
                order=self.order, basis="raw", norms=self.norms_,
            )

        child_outputs = {child.name: self._predict_node(child, X) for child in node.children}
        children_df = pd.DataFrame(child_outputs)
        firing = np.ones((len(children_df), 1))
        return apply_tsk_consequents(
            children_df, state["children"], firing, [0], state["y_bucket_mean"], state["corr_terms"],
            order=self.order, basis="raw",
        )


class DeconstructedHierarchicalClassifier(BaseEstimator, ClassifierMixin):
    """Classification counterpart of `DeconstructedHierarchicalRegressor`.

    Fits one flat `TribbleClassifier` on all features, then deconstructs it
    into a user-supplied topology exactly like the regressor -- except a
    `TribbleClassifier` rule's "consequent" already IS its class label, so a
    leaf has nothing of its own left to fit: slicing the flat model's
    antecedents down to the leaf's own features already gives that leaf's
    per-class probability directly, the same computation
    `TribbleClassifier.predict_proba` does over the full feature set. There
    is therefore no `leaf_targets` parameter here (unlike the regressor) --
    nothing about a leaf's own output is separately fit, so there is nothing
    to redirect at fit time.

    Only branch combiners are fit: one ridge solve per class (the same
    degenerate single-rule solve the regressor uses), regressing that
    class's one-hot indicator on the children's per-class probabilities.
    That solve is unconstrained, so its raw output can land outside [0, 1]
    or break the per-row sum-to-1 invariant a probability vector must have.
    Each branch's raw per-class scores are therefore clipped to [0, 1] and
    renormalized (falling back to a uniform distribution on an all-zero row,
    the same convention `TribbleClassifier.predict_proba` already uses)
    before being handed to its own parent.
    """

    def __init__(self, flat_classifier_kwargs=None, l2_reg: float = 1e-6, order: str = _ORDER):
        self.flat_classifier_kwargs = flat_classifier_kwargs
        self.l2_reg = l2_reg
        self.order = order

    def fit(self, X, y, topology: dict[str, list[str]]):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X = X.reset_index(drop=True)
        y_arr = np.asarray(y).reshape(-1)

        self.flat_ = TribbleClassifier(**(self.flat_classifier_kwargs or {}))
        self.flat_.fit(X, y_arr)
        self.classes_ = self.flat_.classes_
        self.anomaly_params_ = AnomalyParameters(
            include_anomaly=False,
            norm_conorm=self.flat_.norm_conorm, t_norm=self.flat_.t_norm,
            t_conorm=self.flat_.t_conorm, allow_mixed_norms=self.flat_.allow_mixed_norms,
        )

        self.root_ = parse_topology(topology, list(X.columns))
        self.node_state_: dict[str, dict] = {}
        self._fit_node(self.root_, X, y_arr)
        self.is_fitted_ = True
        return self

    def _class_prior(self, y_arr: np.ndarray) -> np.ndarray:
        counts = np.array([np.sum(y_arr == c) for c in self.classes_], dtype=float)
        total = counts.sum()
        if total <= 0:
            return np.full(len(self.classes_), 1.0 / len(self.classes_))
        return counts / total

    def _leaf_gmm(self, node: TopologyNode) -> tuple[GaussianMixtureModel, list[str]]:
        feature_models = {
            f: self.flat_.model_.feature_models[f]
            for f in node.own_features
            if f in self.flat_.model_.feature_models
        }
        return GaussianMixtureModel(feature_models=feature_models), list(feature_models.keys())

    def _leaf_proba(self, X: pd.DataFrame, gmm: GaussianMixtureModel) -> np.ndarray:
        firing, labels = tsk_firing_strengths(X, gmm, anomaly_details=self.anomaly_params_)
        row_sums = firing.sum(axis=1, keepdims=True)
        proba = np.zeros_like(firing)
        nonzero = row_sums.flatten() > 0
        proba[nonzero] = firing[nonzero] / row_sums[nonzero]
        proba[~nonzero] = 1.0 / len(labels)

        label_to_idx = {label: i for i, label in enumerate(labels)}
        out = np.zeros((len(X), len(self.classes_)))
        for i, cls in enumerate(self.classes_):
            if cls in label_to_idx:
                out[:, i] = proba[:, label_to_idx[cls]]
        return out

    @staticmethod
    def _bound_and_normalize(raw: np.ndarray) -> np.ndarray:
        """Clip an unconstrained branch-combiner output into a valid probability
        row: non-negative, upper-bounded at 1, and summing to 1."""
        clipped = np.clip(raw, 0.0, 1.0)
        row_sums = clipped.sum(axis=1, keepdims=True)
        out = np.zeros_like(clipped)
        nonzero = row_sums.flatten() > 0
        out[nonzero] = clipped[nonzero] / row_sums[nonzero]
        out[~nonzero] = 1.0 / clipped.shape[1]
        return out

    def _fit_node(self, node: TopologyNode, X: pd.DataFrame, y_arr: np.ndarray) -> np.ndarray:
        if node.is_leaf:
            return self._fit_leaf(node, X, y_arr)
        children_names = [c.name for c in node.children]
        child_outputs = [self._fit_node(c, X, y_arr) for c in node.children]
        return self._fit_branch(node, children_names, child_outputs, y_arr)

    def _fit_leaf(self, node: TopologyNode, X: pd.DataFrame, y_arr: np.ndarray) -> np.ndarray:
        gmm, top_n_todo = self._leaf_gmm(node)
        if not top_n_todo:
            proba = self._class_prior(y_arr)
            self.node_state_[node.name] = {"kind": "constant", "model": _ConstantClassifier(proba)}
            return np.tile(proba, (len(X), 1))
        self.node_state_[node.name] = {"kind": "leaf", "gmm": gmm, "top_n_todo": top_n_todo}
        return self._leaf_proba(X, gmm)

    def _fit_branch(
        self, node: TopologyNode, children_names: list[str], child_outputs: list[np.ndarray], y_arr: np.ndarray,
    ) -> np.ndarray:
        n = len(y_arr)
        firing = np.ones((n, 1))
        per_class: dict = {}
        raw = np.zeros((n, len(self.classes_)))
        for ci, cls in enumerate(self.classes_):
            children_df = pd.DataFrame(
                {name: out[:, ci] for name, out in zip(children_names, child_outputs)}
            )
            y_train = pd.DataFrame({"y_value": (y_arr == cls).astype(float)})
            corr_terms, y_bucket_mean = solve_tsk_consequents_from_firing(
                firing, [0], children_df, children_names, None, y_train,
                order=self.order, l2_reg=self.l2_reg, basis="raw",
                pin_extremes=False, verbose=False,
            )
            per_class[cls] = (corr_terms, y_bucket_mean)
            raw[:, ci] = apply_tsk_consequents(
                children_df, children_names, firing, [0], y_bucket_mean, corr_terms,
                order=self.order, basis="raw",
            )
        self.node_state_[node.name] = {"kind": "branch", "children": children_names, "per_class": per_class}
        return self._bound_and_normalize(raw)

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.flat_.feature_names_in_)
        X = X.reset_index(drop=True)
        return self._predict_node(self.root_, X)

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def _predict_node(self, node: TopologyNode, X: pd.DataFrame) -> np.ndarray:
        state = self.node_state_[node.name]
        if state["kind"] == "constant":
            return state["model"].predict_proba_aligned(X)
        if state["kind"] == "leaf":
            return self._leaf_proba(X, state["gmm"])

        child_outputs = [self._predict_node(child, X) for child in node.children]
        n = len(X)
        firing = np.ones((n, 1))
        raw = np.zeros((n, len(self.classes_)))
        for ci, cls in enumerate(self.classes_):
            children_df = pd.DataFrame(
                {name: out[:, ci] for name, out in zip(state["children"], child_outputs)}
            )
            corr_terms, y_bucket_mean = state["per_class"][cls]
            raw[:, ci] = apply_tsk_consequents(
                children_df, state["children"], firing, [0], y_bucket_mean, corr_terms,
                order=self.order, basis="raw",
            )
        return self._bound_and_normalize(raw)
