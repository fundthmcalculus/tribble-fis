import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted

from .gauss_data import DefaultNormCornorm, NormPair, resolve_norm_pair
from .gauss_math import (
    calculate_gaussian_correlation,
    take_top_features,
    create_gaussian_membership_dict,
    LogTransformMixin,
)
from .regression import (
    partition_output,
    solve_tsk_consequents,
    predict_tsk,
)


class MixtureOfGaussiansFuzzyRegressor(LogTransformMixin, BaseEstimator, RegressorMixin):
    """
    Gaussian Mixture Regressor using TSK (Takagi-Sugeno-Kang) fuzzy inference.
    Handles continuous output prediction with multiple TSK orders.
    """

    def __init__(
        self,
        top_n=-1,
        top_p=0.95,
        n_gaussians=0,
        log_transform=False,
        n_output_buckets=2,
        tsk_order="1st",
        optimize_coefficients=True,
        consequent_basis="raw",
        l2_reg=0.0,
        pin_extremes=True,
        norm_conorm=DefaultNormCornorm,
        t_norm=None,
        t_conorm=None,
        allow_mixed_norms=False,
        bucket_strategy="uniform",
        max_rules=8,
        bucket_r2_threshold=0.9,
        min_bucket_samples=20,
        adaptive_split_method="median",
        guard_stalled_splits=True,
        random_state=42,
    ):
        """
        Initialize the MixtureOfGaussiansFuzzyRegressor.

        Args:
            top_n: Number of top features to select based on differentiation score.
                   If > 0, top_p is ignored.
            top_p: Percentage of cumulative differentiation score to cover.
            n_gaussians: Number of Gaussians per feature per label (0 for automatic).
            log_transform: Whether to automatically apply log-transformation to features.
            n_output_buckets: Number of output buckets for partitioning y during training
                   when bucket_strategy='uniform'. When bucket_strategy='adaptive', this
                   value is used only to bucket y for feature-relevance ranking
                   (calculate_gaussian_correlation) -- the rule structure actually fit
                   comes from the adaptive growth loop instead.
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
            bucket_strategy: 'uniform' (default) partitions y into n_output_buckets
                equal-frequency rules via qcut, fixed for the whole fit. 'adaptive'
                instead grows the partition from a single rule, repeatedly splitting
                the worst-fitting rule (lowest local R^2) until every rule clears
                bucket_r2_threshold or max_rules is reached -- see
                adaptive_partition.grow_adaptive_partition.
            max_rules: Rule-count ceiling for bucket_strategy='adaptive'. Ignored
                otherwise.
            bucket_r2_threshold: A rule stops being a split candidate once its local
                R^2 (against its own y-mean) meets this threshold. Ignored unless
                bucket_strategy='adaptive'.
            min_bucket_samples: A rule with fewer than this many training rows is
                never split further, regardless of its R^2. Ignored unless
                bucket_strategy='adaptive'.
            adaptive_split_method: 'median' bisects the chosen rule at its median y
                (equal-frequency children). 'sse' scans every possible split and
                picks the one minimizing the two children's combined SSE against
                their own means (CART-style). Ignored unless bucket_strategy='adaptive'.
            guard_stalled_splits: If True (default), a split whose resulting SSE
                over the parent rule's own rows doesn't improve blocks its children
                from being split again -- prevents endlessly re-splitting a region
                that isn't actually getting better. Ignored unless
                bucket_strategy='adaptive'.
            random_state: Seed for reproducibility.
        """
        self.is_fitted_ = False
        self.model_ = None
        self.top_features_ = None
        self.top_n_actual_ = None
        self.feature_differentiators_ = None
        self.feature_names_in_ = []
        self.log_transformed_features_ = {}
        self.y_bucket_mean_ = None
        self.corr_terms_ = None
        self.n_rules_ = None
        self.partition_edges_ = None
        self.partition_history_ = None

        self.top_n = top_n
        self.top_p = top_p
        self.n_gaussians = n_gaussians
        self.log_transform = log_transform
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
        self.bucket_strategy = bucket_strategy
        self.max_rules = max_rules
        self.bucket_r2_threshold = bucket_r2_threshold
        self.min_bucket_samples = min_bucket_samples
        self.adaptive_split_method = adaptive_split_method
        self.guard_stalled_splits = guard_stalled_splits
        self.random_state = random_state

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

        # Apply log-transformation if requested
        X_df = self._apply_log_transform(X_df)

        # Partition output into buckets
        y_partitioned, self.y_bucket_mean_ = partition_output(self.n_output_buckets, y_series)

        # Calculate feature differentiators
        self.feature_differentiators_ = calculate_gaussian_correlation(X_df, y_partitioned["y_bucket"])

        # Select top features
        self.top_n_actual_, self.top_features_ = take_top_features(
            self.feature_differentiators_, top_p=self.top_p, top_n=self.top_n
        )

        if self.bucket_strategy == "adaptive":
            # Imported lazily: adaptive_partition is experimental and not
            # always present (see ADAPTIVE_PARTITIONING_FINDINGS.md) -- the
            # default "uniform" strategy must not require it to be installed.
            from .adaptive_partition import grow_adaptive_partition

            # The uniform partition above was only used for feature-relevance
            # ranking; the rule structure itself is grown from a single rule.
            result = grow_adaptive_partition(
                X_df, y_series, self.top_features_, n_gaussians=self.n_gaussians,
                tsk_order=self.tsk_order, l2_reg=self.l2_reg, basis=self.consequent_basis,
                pin_extremes=self.pin_extremes, norms=self._norms(),
                max_rules=self.max_rules, r2_threshold=self.bucket_r2_threshold,
                min_bucket_samples=self.min_bucket_samples,
                guard_stalled_splits=self.guard_stalled_splits,
                split_method=self.adaptive_split_method, verbose=False,
            )
            self.model_ = result.model
            self.y_bucket_mean_ = result.y_bucket_mean
            self.corr_terms_ = result.corr_terms
            self.partition_edges_ = result.edges
            self.partition_history_ = result.history
        else:
            # Create Gaussian membership model
            self.model_ = create_gaussian_membership_dict(
                X_df, y_partitioned["y_bucket"], top_n_var_names=self.top_features_, n_gaussians=self.n_gaussians
            )

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

        self.n_rules_ = self.model_.n_rules

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

        X_df = self._apply_log_transform(X_df)

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
        log_transform=False,
        n_output_buckets=15,
        tsk_order="1st",
        optimize_coefficients=True,
        random_state=42,
    ):
        self.top_n = top_n
        self.top_p = top_p
        self.n_gaussians = n_gaussians
        self.log_transform = log_transform
        self.n_output_buckets = n_output_buckets
        self.tsk_order = tsk_order
        self.optimize_coefficients = optimize_coefficients
        self.random_state = random_state

    def _make_regressor(self):
        return MixtureOfGaussiansFuzzyRegressor(
            top_n=self.top_n,
            top_p=self.top_p,
            n_gaussians=self.n_gaussians,
            log_transform=self.log_transform,
            n_output_buckets=self.n_output_buckets,
            tsk_order=self.tsk_order,
            optimize_coefficients=self.optimize_coefficients,
            random_state=self.random_state,
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
