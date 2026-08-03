import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted

from .gauss_data import DefaultNormCornorm, NormPair, resolve_norm_pair
from .gauss_math import (
    calculate_gaussian_correlation,
    take_top_features,
    create_gaussian_membership_dict,
)
from .regression import (
    partition_output,
    solve_tsk_consequents,
    predict_tsk,
)


class MixtureOfGaussiansFuzzyRegressor(BaseEstimator, RegressorMixin):
    """
    Gaussian Mixture Regressor using TSK (Takagi-Sugeno-Kang) fuzzy inference.
    Handles continuous output prediction with multiple TSK orders.
    """

    def __init__(
        self,
        top_n=-1,
        top_p=0.95,
        n_gaussians=0,
        n_output_buckets=2,
        tsk_order="1st",
        optimize_coefficients=True,
        consequent_basis="raw",
        l2_reg=1e-6,
        pin_extremes=True,
        norm_conorm=DefaultNormCornorm,
        t_norm=None,
        t_conorm=None,
        allow_mixed_norms=False,
        random_state=42,
        max_samples=None,
    ):
        """
        Initialize the MixtureOfGaussiansFuzzyRegressor.

        Args:
            top_n: Number of top features to select based on differentiation score.
                   If > 0, top_p is ignored.
            top_p: Per-feature score threshold, not cumulative coverage: a feature
                   is kept when its own normalized differentiation score is
                   >= (1 - top_p). Ignored if top_n > 0.
            n_gaussians: Number of Gaussians per feature per label (0 for automatic).
            n_output_buckets: Number of output buckets for partitioning y during training.
            tsk_order: TSK polynomial order ('0th', '1st', '2nd', '3rd', 'full-2nd').
            optimize_coefficients: Retained for API compatibility. Consequents are
                always solved in closed form (the exact firing-weighted ridge
                least-squares optimum), which supersedes the former per-bucket LS
                initialization plus L-BFGS refinement.
            consequent_basis: 'raw' monomials or 'orthogonal' (Legendre) basis for
                the consequent polynomial. Orthogonal is better conditioned at
                higher orders.
            l2_reg: Ridge penalty on the correction coefficients (constants are not
                penalized). 0 disables regularization.
            pin_extremes: If True (default), the first and last bucket means are pinned
                to the observed min and max of the target, ensuring the model's output
                range exactly matches the training range.
            norm_conorm: Fuzzy operator family used to combine memberships -- the
                t-norm for the rule AND and its De Morgan dual conorm for the
                per-feature OR. Previously the regressor had no way to express
                this and was fixed at the "min/max" default.
            t_norm, t_conorm: Advanced. Override one half of the pair. Leaving both
                None takes both operators from `norm_conorm`, which keeps the pair
                De Morgan-consistent.
            allow_mixed_norms: Advanced. Required to opt in to a t-norm and t-conorm
                from different families, which are not De Morgan duals.
            random_state: Seed for reproducibility.
            max_samples: Cap on the rows used per (feature, label-bucket) when
                fitting Gaussian memberships. ``None`` -- the default -- uses
                every row; pass an int to bound fit time on large datasets.
        """
        self.is_fitted_ = False
        self.model_ = None
        self.top_features_ = None
        self.top_n_actual_ = None
        self.feature_differentiators_ = None
        self.feature_names_in_ = []
        self.y_bucket_mean_ = None
        self.corr_terms_ = None
        self.n_rules_ = None

        self.top_n = top_n
        self.top_p = top_p
        self.n_gaussians = n_gaussians
        self.n_output_buckets = n_output_buckets
        self.tsk_order = tsk_order
        self.optimize_coefficients = optimize_coefficients
        self.consequent_basis = consequent_basis
        self.l2_reg = l2_reg
        self.pin_extremes = pin_extremes
        self.norm_conorm = norm_conorm
        self.t_norm = t_norm
        self.t_conorm = t_conorm
        self.allow_mixed_norms = allow_mixed_norms
        self.random_state = random_state
        self.max_samples = max_samples

    def _norms(self) -> NormPair:
        """Resolved (t-norm, t-conorm) for this estimator.

        Derived on demand rather than cached in __init__: sklearn requires that
        __init__ only assign its arguments unmodified, so that get_params /
        set_params round-trips and clone() reproduces the estimator exactly.
        """
        return resolve_norm_pair(
            self.norm_conorm, self.t_norm, self.t_conorm, self.allow_mixed_norms
        )

    def fit(self, X, y):
        """
        Fit the Gaussian Mixture regression model.

        Args:
            X: Training features (n_samples, n_features)
            y: Target values (n_samples,) for single-output or (n_samples, n_outputs) for multi-output
        """
        # Convert X to DataFrame if needed
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
        else:
            self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        # Validate and convert y
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_array = y.values.flatten()
        else:
            y_array = np.asarray(y).flatten()

        # Standard sklearn validation
        X_array, y_array = check_X_y(X, y_array, multi_output=False, y_numeric=True)

        # Store as DataFrame for internal functions
        X_df = pd.DataFrame(X_array, columns=self.feature_names_in_)
        y_series = pd.Series(y_array, name="y_value")

        # Partition output into buckets
        y_partitioned, self.y_bucket_mean_ = partition_output(self.n_output_buckets, y_series)

        # Calculate feature differentiators
        self.feature_differentiators_ = calculate_gaussian_correlation(X_df, y_partitioned["y_bucket"])

        # Select top features
        self.top_n_actual_, self.top_features_ = take_top_features(
            self.feature_differentiators_, top_p=self.top_p, top_n=self.top_n
        )

        # Create Gaussian membership model
        self.model_ = create_gaussian_membership_dict(
            X_df, y_partitioned["y_bucket"], top_n_var_names=self.top_features_, n_gaussians=self.n_gaussians,
            max_samples=self.max_samples, random_state=self.random_state,
        )

        self.n_rules_ = self.model_.n_rules

        # Solve TSK consequents in closed form: for fixed firing strengths the
        # output is linear in the coefficients, so a single ridge least-squares
        # solve yields the exact firing-weighted optimum.
        self.corr_terms_, self.y_bucket_mean_ = solve_tsk_consequents(
            X_df, self.model_, self.top_features_,
            self.y_bucket_mean_, y_partitioned,
            n_output_buckets=self.n_output_buckets,
            order=self.tsk_order, l2_reg=self.l2_reg, basis=self.consequent_basis,
            pin_extremes=self.pin_extremes,
            norms=self._norms(),
            verbose=False,
        )

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Predict continuous values for X.

        Args:
            X: Input features (n_samples, n_features)

        Returns:
            Predicted values (n_samples,)
        """
        check_is_fitted(self)

        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)

        # Shared prediction path: identical firing-strength normalization and
        # feature basis as the solver, so fit and predict cannot diverge.
        return predict_tsk(
            X_df, self.model_, self.top_features_,
            self.y_bucket_mean_, self.corr_terms_,
            order=self.tsk_order, basis=self.consequent_basis,
            norms=self._norms(),
        )


class MimoGaussianPredictor(BaseEstimator, RegressorMixin):
    """
    Multi-input multi-output wrapper around MixtureOfGaussiansFuzzyRegressor.

    Fits one independent regressor per output column, enabling simultaneous
    prediction of multiple outputs from the same input features.
    """

    def __init__(
        self,
        top_n=-1,
        top_p=0.95,
        n_gaussians=0,
        n_output_buckets=15,
        tsk_order="1st",
        optimize_coefficients=True,
        random_state=42,
        max_samples=None,
    ):
        self.top_n = top_n
        self.top_p = top_p
        self.n_gaussians = n_gaussians
        self.n_output_buckets = n_output_buckets
        self.tsk_order = tsk_order
        self.optimize_coefficients = optimize_coefficients
        self.random_state = random_state
        self.max_samples = max_samples

    def _make_regressor(self):
        return MixtureOfGaussiansFuzzyRegressor(
            top_n=self.top_n,
            top_p=self.top_p,
            n_gaussians=self.n_gaussians,
            n_output_buckets=self.n_output_buckets,
            tsk_order=self.tsk_order,
            optimize_coefficients=self.optimize_coefficients,
            random_state=self.random_state,
            max_samples=self.max_samples,
        )

    def fit(self, X, y):
        """
        Fit one regressor per output column.

        Args:
            X: Training features (n_samples, n_features)
            y: Target values (n_samples, n_outputs) or DataFrame
        """
        if isinstance(y, pd.DataFrame):
            self.output_names_ = y.columns.tolist()
            y_array = y.values
        elif isinstance(y, pd.Series):
            self.output_names_ = [y.name or "output_0"]
            y_array = y.values.reshape(-1, 1)
        else:
            y_array = np.asarray(y)
            if y_array.ndim == 1:
                y_array = y_array.reshape(-1, 1)
            self.output_names_ = [f"output_{i}" for i in range(y_array.shape[1])]

        self.regressors_ = {}
        for i, name in enumerate(self.output_names_):
            print(f"  Fitting regressor for output '{name}' ({i+1}/{len(self.output_names_)})")
            reg = self._make_regressor()
            reg.fit(X, y_array[:, i])
            self.regressors_[name] = reg

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Predict all outputs for X.

        Args:
            X: Input features (n_samples, n_features)

        Returns:
            DataFrame with one column per output (n_samples, n_outputs)
        """
        check_is_fitted(self)
        preds = {name: reg.predict(X) for name, reg in self.regressors_.items()}
        return pd.DataFrame(preds, columns=self.output_names_)
