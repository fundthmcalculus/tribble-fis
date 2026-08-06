"""Sklearn-style fuzzy regression tree estimator (and a MIMO wrapper).

``FuzzyRegressionTree`` mirrors the parameters and fit/predict flow of
``tribblefis.gaussian_regressor.TribbleRegressor`` so it is a
drop-in alternative, but produces a *hierarchical* model: each internal node
splits on one input variable, and each leaf holds a local TSK consequent solved by
the shared closed-form ridge least squares.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, check_X_y

from tribblefis.gauss_math import (
    calculate_gaussian_correlation,
    take_top_features,
)
from tribblefis.regression import partition_output

from .builder import build_tree
from .firing import DEFAULT_T_NORM, compute_leaf_firing
from .plan import VariablePlan
from .prune import prune_tree
from .solve import predict_leaves, solve_leaf_consequents
from .splitter import get_criterion


class FuzzyRegressionTree(BaseEstimator, RegressorMixin):
    """Hierarchical fuzzy (soft) regression tree with local TSK leaves.

    Args:
        variable_plan: Optional ``VariablePlan`` giving explicit structure (level
            ordering, node pins, criterion, caps). When provided, its structural
            fields take precedence over the scalar params below.
        criterion: Auto split criterion ('variance', 'ambiguity', 'info_gain',
            'differentiation') when no plan is given. 'variance' is recommended.
        tsk_order: Leaf model order ('0th'..'3rd', 'full-2nd'). '0th' gives the
            most readable constant leaves; '1st'+ trade readability for accuracy.
        consequent_basis: 'raw' or 'orthogonal' (Legendre) consequent basis.
        l2_reg: Ridge penalty on leaf correction terms (intercepts unpenalised).
        top_n / top_p: Feature preselection by differentiation score (as in the
            flat regressor). The tree splits among, and leaves regress on, these.
        max_depth / n_terms: Tree depth cap and linguistic terms per split.
        min_soft_count / min_gain / max_leaves: Stopping/pruning guards.
        t_norm: Fuzzy AND for path-weight propagation ('probability' = product,
            'min/max', 'luk', 'hamacher'). Used consistently at fit and predict.
        n_score_buckets: Output quantile buckets used only to form pseudo-classes
            for the 'ambiguity'/'info_gain'/'differentiation' criteria.
        term_style: 'trapezoid' (default, nameable bands) or 'gaussian'.
        ccp_alpha: Post-hoc split-gain pruning threshold (see
            ``fuzzytree.prune``): collapses any node whose own split gain
            (under ``criterion``) falls below this value. 0 (default) disables
            pruning entirely, reproducing the tree exactly as
            ``max_depth``/``max_leaves``/``min_gain`` built it. Only useful set
            higher than ``min_gain``, since ``build_tree`` never creates a
            split scoring below ``min_gain`` in the first place.
    """

    def __init__(
        self,
        variable_plan: VariablePlan | None = None,
        criterion: str = "variance",
        tsk_order: str = "0th",
        consequent_basis: str = "raw",
        l2_reg: float = 1e-6,
        top_n: int = -1,
        top_p: float = 0.95,
        max_depth: int = 3,
        n_terms: int = 3,
        min_soft_count: float = 5.0,
        min_gain: float = 1e-3,
        max_leaves: int = 64,
        t_norm: str = DEFAULT_T_NORM,
        n_score_buckets: int = 3,
        term_style: str = "trapezoid",
        ccp_alpha: float = 0.0,
        random_state: int = 42,
    ):
        self.variable_plan = variable_plan
        self.criterion = criterion
        self.tsk_order = tsk_order
        self.consequent_basis = consequent_basis
        self.l2_reg = l2_reg
        self.top_n = top_n
        self.top_p = top_p
        self.max_depth = max_depth
        self.n_terms = n_terms
        self.min_soft_count = min_soft_count
        self.min_gain = min_gain
        self.max_leaves = max_leaves
        self.t_norm = t_norm
        self.n_score_buckets = n_score_buckets
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

        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_array = y.values.flatten()
        else:
            y_array = np.asarray(y).flatten()

        X_array, y_array = check_X_y(X, y_array, multi_output=False, y_numeric=True)
        X_df = pd.DataFrame(X_array, columns=self.feature_names_in_)
        y_series = pd.Series(y_array, name="y_value")

        # Feature preselection by differentiation score (pseudo-classes from y).
        y_part, _ = partition_output(self.n_score_buckets, y_series)
        self.feature_differentiators_ = calculate_gaussian_correlation(
            X_df, y_part["y_bucket"]
        )
        self.top_n_actual_, self.top_features_ = take_top_features(
            self.feature_differentiators_, top_p=self.top_p, top_n=self.top_n
        )

        X_top = X_df[self.top_features_]
        y_value = y_series.to_numpy(dtype=float)
        y_bucket = y_part["y_bucket"].to_numpy()

        plan = self._resolve_plan()
        self.tree_, self.n_leaves_ = build_tree(
            X_top,
            y_value,
            y_bucket,
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
                y_value,
                y_bucket,
                self.ccp_alpha,
                t_norm_name=self.t_norm,
            )

        leaf_firing = compute_leaf_firing(self.tree_, X_top, self.n_leaves_, self.t_norm)
        self.corr_terms_, self.leaf_mean_ = solve_leaf_consequents(
            X_top.to_numpy(),
            y_value,
            leaf_firing,
            order=self.tsk_order,
            basis=self.consequent_basis,
            l2_reg=self.l2_reg,
        )

        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)

        X_top = X_df[self.top_features_]
        leaf_firing = compute_leaf_firing(self.tree_, X_top, self.n_leaves_, self.t_norm)
        return predict_leaves(
            X_top.to_numpy(),
            leaf_firing,
            self.leaf_mean_,
            self.corr_terms_,
            order=self.tsk_order,
            basis=self.consequent_basis,
        )


class MimoFuzzyTreeRegressor(BaseEstimator, RegressorMixin):
    """Fit one ``FuzzyRegressionTree`` per output column (mirrors
    ``tribblefis.gaussian_regressor.MimoGaussianPredictor``)."""

    def __init__(
        self,
        variable_plan: VariablePlan | None = None,
        criterion: str = "variance",
        tsk_order: str = "0th",
        consequent_basis: str = "raw",
        l2_reg: float = 1e-6,
        top_n: int = -1,
        top_p: float = 0.95,
        max_depth: int = 3,
        n_terms: int = 3,
        min_soft_count: float = 5.0,
        min_gain: float = 1e-3,
        max_leaves: int = 64,
        t_norm: str = DEFAULT_T_NORM,
        n_score_buckets: int = 3,
        term_style: str = "trapezoid",
        ccp_alpha: float = 0.0,
        random_state: int = 42,
    ):
        self.variable_plan = variable_plan
        self.criterion = criterion
        self.tsk_order = tsk_order
        self.consequent_basis = consequent_basis
        self.l2_reg = l2_reg
        self.top_n = top_n
        self.top_p = top_p
        self.max_depth = max_depth
        self.n_terms = n_terms
        self.min_soft_count = min_soft_count
        self.min_gain = min_gain
        self.max_leaves = max_leaves
        self.t_norm = t_norm
        self.n_score_buckets = n_score_buckets
        self.term_style = term_style
        self.ccp_alpha = ccp_alpha
        self.random_state = random_state

    def _make_tree(self) -> FuzzyRegressionTree:
        return FuzzyRegressionTree(
            variable_plan=self.variable_plan,
            criterion=self.criterion,
            tsk_order=self.tsk_order,
            consequent_basis=self.consequent_basis,
            l2_reg=self.l2_reg,
            top_n=self.top_n,
            top_p=self.top_p,
            max_depth=self.max_depth,
            n_terms=self.n_terms,
            min_soft_count=self.min_soft_count,
            min_gain=self.min_gain,
            max_leaves=self.max_leaves,
            t_norm=self.t_norm,
            n_score_buckets=self.n_score_buckets,
            term_style=self.term_style,
            ccp_alpha=self.ccp_alpha,
            random_state=self.random_state,
        )

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        if isinstance(y, pd.DataFrame):
            self.output_names_ = y.columns.tolist()
        elif isinstance(y, pd.Series):
            self.output_names_ = [y.name or "output_0"]
            y = y.to_frame()
        else:
            y = np.asarray(y)
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            self.output_names_ = [f"output_{i}" for i in range(y.shape[1])]
            y = pd.DataFrame(y, columns=self.output_names_)

        self.regressors_ = {}
        for name in self.output_names_:
            tree = self._make_tree()
            tree.fit(X, y[name])
            self.regressors_[name] = tree
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        preds = {name: reg.predict(X) for name, reg in self.regressors_.items()}
        return pd.DataFrame(preds, index=X.index)
