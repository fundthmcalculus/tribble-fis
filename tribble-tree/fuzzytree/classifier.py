"""Sklearn-style fuzzy classification tree.

Mirrors ``tribblefis.gaussian_classifier.TribbleClassifier`` at the
API level but produces a hierarchical soft tree. Splits default to the Yuan-Shaw
classification-ambiguity criterion; each leaf stores a fuzzy-weighted class
distribution, and prediction is the firing-weighted vote across leaves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_X_y

from tribblefis.gauss_math import calculate_gaussian_correlation, take_top_features
from tribblefis.regression import _normalize_firing_strengths

from .builder import build_tree
from .firing import DEFAULT_T_NORM, compute_leaf_firing
from .plan import VariablePlan
from .prune import prune_tree
from .splitter import get_criterion


class FuzzyClassificationTree(BaseEstimator, ClassifierMixin):
    """Hierarchical fuzzy (soft) classification tree with class-vote leaves.

    ``ccp_alpha`` (default 0, disabled) applies the same post-hoc split-gain
    pruning as ``FuzzyRegressionTree`` -- see ``fuzzytree.prune`` -- collapsing
    any node whose own split gain (under this tree's ``criterion``) falls
    below ``ccp_alpha``.
    """

    def __init__(
        self,
        variable_plan: VariablePlan | None = None,
        criterion: str = "ambiguity",
        top_n: int = -1,
        top_p: float = 0.95,
        max_depth: int = 3,
        n_terms: int = 3,
        min_soft_count: float = 5.0,
        min_gain: float = 1e-3,
        max_leaves: int = 64,
        t_norm: str = DEFAULT_T_NORM,
        term_style: str = "trapezoid",
        ccp_alpha: float = 0.0,
        random_state: int = 42,
    ):
        self.variable_plan = variable_plan
        self.criterion = criterion
        self.top_n = top_n
        self.top_p = top_p
        self.max_depth = max_depth
        self.n_terms = n_terms
        self.min_soft_count = min_soft_count
        self.min_gain = min_gain
        self.max_leaves = max_leaves
        self.t_norm = t_norm
        self.term_style = term_style
        self.ccp_alpha = ccp_alpha
        self.random_state = random_state

    def _resolve_plan(self) -> VariablePlan:
        if self.variable_plan is not None:
            return self.variable_plan
        return VariablePlan(
            criterion=self.criterion,
            max_depth=self.max_depth,
            default_n_terms=self.n_terms,
            max_terms_per_var=max(self.n_terms, 2),
            term_style=self.term_style,
        )

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
        else:
            self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        y_arr = np.asarray(y).flatten()
        X_array, y_arr = check_X_y(X, y_arr, multi_output=False)
        X_df = pd.DataFrame(X_array, columns=self.feature_names_in_).reset_index(drop=True)

        self.classes_ = np.unique(y_arr)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[c] for c in y_arr])
        y_series = pd.Series(y_arr, name="label")

        # Feature preselection using the true labels.
        self.feature_differentiators_ = calculate_gaussian_correlation(X_df, y_series)
        self.top_n_actual_, self.top_features_ = take_top_features(
            self.feature_differentiators_, top_p=self.top_p, top_n=self.top_n
        )

        X_top = X_df[self.top_features_]
        plan = self._resolve_plan()
        # For classification, y_value carries class indices (used only if the
        # criterion falls back to variance); y_bucket carries the true classes.
        self.tree_, self.n_leaves_ = build_tree(
            X_top,
            y_idx.astype(float),
            y_idx,
            plan,
            min_soft_count=self.min_soft_count,
            min_gain=self.min_gain,
            max_leaves=self.max_leaves,
        )

        if self.ccp_alpha > 0:
            self.tree_, self.n_leaves_ = prune_tree(
                self.tree_,
                X_top,
                get_criterion(plan.criterion),
                y_idx.astype(float),
                y_idx,
                self.ccp_alpha,
                t_norm_name=self.t_norm,
            )

        leaf_firing = compute_leaf_firing(self.tree_, X_top, self.n_leaves_, self.t_norm)
        # Per-leaf fuzzy class mass -> normalised class distribution.
        n_classes = len(self.classes_)
        mass = np.zeros((self.n_leaves_, n_classes))
        for c in range(n_classes):
            mass[:, c] = leaf_firing[y_idx == c].sum(axis=0)
        row_tot = mass.sum(axis=1, keepdims=True)
        # Uniform fallback for a leaf that captured no mass of any class.
        safe = row_tot[:, 0] > 1e-9
        self.leaf_class_dist_ = np.full((self.n_leaves_, n_classes), 1.0 / n_classes)
        self.leaf_class_dist_[safe] = mass[safe] / row_tot[safe]

        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        X_top = X[self.top_features_]
        leaf_firing = compute_leaf_firing(self.tree_, X_top, self.n_leaves_, self.t_norm)
        norm_fs = _normalize_firing_strengths(leaf_firing)
        proba = norm_fs @ self.leaf_class_dist_
        row_sums = proba.sum(axis=1, keepdims=True)
        # Rows where no leaf fired -> fall back to the global class distribution.
        empty = row_sums[:, 0] <= 1e-9
        if empty.any():
            proba[empty] = self.leaf_class_dist_.mean(axis=0)
            row_sums = proba.sum(axis=1, keepdims=True)
        return proba / row_sums

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
