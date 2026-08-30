import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted

from .gauss_data import DefaultNormCornorm, NormPair, resolve_norm_pair
from .gauss_math import (
    calculate_gaussian_correlation,
    calculate_interaction_scores,
    take_top_features,
    take_top_interactions,
    rescue_interacting_features,
    create_gaussian_membership_dict,
)
from .regression import (
    partition_output,
    solve_tsk_consequents,
    select_interaction_terms,
    select_consequent_hyperparams,
    predict_tsk,
    compute_rbf_centers,
)

# Candidates tried by tsk_order="auto" (regression.select_consequent_hyperparams's
# CV screen). Includes "0th" -- unlike that helper's own default -- because auto's
# whole point is a safety net against small-sample overfitting (issue #120), and
# 0th is the floor every candidate above it must beat.
_AUTO_ORDER_CANDIDATES = ("0th", "1st", "2nd", "full-2nd", "3rd")


class TribbleRegressor(BaseEstimator, RegressorMixin):
    """
    Gaussian Mixture Regressor using TSK (Takagi-Sugeno-Kang) fuzzy inference.
    Handles continuous output prediction with multiple TSK orders.
    """

    def __init__(
        self,
        top_n=-1,
        top_p=0.95,
        correlation_threshold=0.85,
        n_gaussians=0,
        n_output_buckets=2,
        output_partition="uniform",
        tsk_order="1st",
        optimize_coefficients=True,
        consequent_basis="raw",
        l2_reg=1e-6,
        pin_extremes=False,
        norm_conorm=DefaultNormCornorm,
        t_norm=None,
        t_conorm=None,
        allow_mixed_norms=False,
        member_function="gaussian",
        trapz_method="fast",
        trapz_width_reg=0.0,
        random_state=42,
        max_samples=None,
        detect_interactions=False,
        interaction_top_p=0.95,
        select_interactions=False,
        auto_order_candidates=_AUTO_ORDER_CANDIDATES,
        rbf_n_centers=3,
        rbf_gamma=1.0,
        rbf_radius=None,
        firing_exponent=1.0,
    ):
        """
        Initialize the TribbleRegressor.

        Args:
            top_n: Number of top features to select based on differentiation score.
                   If > 0, top_p is ignored.
            top_p: Per-feature score threshold, not cumulative coverage: a feature
                   is kept when its own normalized differentiation score is
                   >= (1 - top_p). Ignored if top_n > 0.
            correlation_threshold: When top_n > 0, drop a candidate feature whose
                   absolute correlation with an already-selected feature is >=
                   this value, pulling in the next-best feature instead. Set to
                   <= 0.0 or >= 1.0 to disable.
            n_gaussians: Number of Gaussians per feature per label (0 for automatic).
                **No effect when `member_function="trap"` and the default
                `trapz_method="fast"`**: that path is a histogram fitter with no
                component-count argument at all -- it emits one trapezoid per
                merged contiguous non-empty bin region, so the count comes from
                the data. Honoured by "gaussian", by "triangular", and by
                `trapz_method="em"`. This asymmetry is why a `[trap]` and a
                `[triangular]` run at the same `n_gaussians` are not the same
                amount of work; see issue #213 and
                tests/test_member_function_component_counts.py.
            n_output_buckets: Number of output buckets for partitioning y during training.
            output_partition: "uniform" for equal-width buckets (default), or
                "quantile" for equal-frequency buckets with pinned extreme centroids,
                which is what this estimator shipped with before.
            tsk_order: TSK polynomial order ('0th', '1st', '2nd', '3rd', 'full-2nd'),
                or 'auto' to pick one per fit. A full-2nd consequent fits
                1 + 2*n_features + C(n_features, 2) coefficients per rule, which
                overfits catastrophically once training rows per rule undercut
                that count by roughly 5x (issue #120 -- e.g. diabetes-scale data
                went from R²=0.44 at order 1 to R²=-0.05 at full-2nd). 'auto'
                runs `regression.select_consequent_hyperparams`'s k-fold CV over
                `auto_order_candidates` (basis and l2_reg held at whatever this
                estimator was constructed with) and fits the winner; the choice
                is exposed as `tsk_order_` and the full CV result as
                `consequent_selection_`. Costs one CV sweep at fit time (cheap:
                each candidate is a single linear solve per fold). Note CV always
                scores at firing_exponent=1.0 regardless of this estimator's
                `firing_exponent`, since `select_consequent_hyperparams` does not
                thread that parameter through.
            optimize_coefficients: Retained for API compatibility. Consequents are
                always solved in closed form (the exact firing-weighted ridge
                least-squares optimum), which supersedes the former per-bucket LS
                initialization plus L-BFGS refinement.
            consequent_basis: 'raw' monomials, 'orthogonal' (Legendre) basis, or
                'gaussian-rbf' (Gaussian RBF basis). Orthogonal is better conditioned
                at higher orders. RBF provides nonlinear basis functions.
            l2_reg: Ridge penalty on the correction coefficients (constants are not
                penalized). 0 disables regularization.
            pin_extremes: If False (default), the first and last bucket means are not pinned
                to the observed extremes. If True, they are pinned to the observed min and max
                of the target, ensuring the model's output range exactly matches the training range.
            norm_conorm: Fuzzy operator family used to combine memberships -- the
                t-norm for the rule AND and its De Morgan dual conorm for the
                per-feature OR. Previously the regressor had no way to express
                this and was fixed at the "min/max" default.
            t_norm, t_conorm: Advanced. Override one half of the pair. Leaving both
                None takes both operators from `norm_conorm`, which keeps the pair
                De Morgan-consistent.
            allow_mixed_norms: Advanced. Required to opt in to a t-norm and t-conorm
                from different families, which are not De Morgan duals.
            member_function: "gaussian" (default), "trap" (EM or histogram-fit
                trapezoid), or "triangular" (special case of trap) -- same
                semantics as `TribbleClassifier`'s parameter of the same name.
            trapz_method: "fast" (histogram-based) or "em" (EM algorithm).
                **Ignored for "triangular"**, which has no histogram path and
                always uses EM -- so "triangular" also pays EM's fit cost
                (measured 0.97s against "trap"'s 0.02s on the #213 fixture).
            trapz_width_reg: EM-only support-width regularization (see
                tribblefis.trapz_math and issue #163). 0.0 (default) is pure
                maximum-likelihood, which collapses trapezoid support onto the
                data mode -- a poor antecedent partition. Values > 0 (~1.0)
                reward wider support. No effect for trapz_method="fast".
                Ignored for "triangular" and "gaussian".
            random_state: Seed for reproducibility.
            max_samples: Cap on the rows used per (feature, label-bucket) when
                fitting Gaussian memberships. ``None`` -- the default -- uses
                every row; pass an int to bound fit time on large datasets.
            detect_interactions: If True, score every candidate feature pair
                for interaction "lift" beyond either feature alone
                (`gauss_math.calculate_interaction_scores`) *before*
                `top_features_` is finalized, and union in any feature that
                is individually below the `top_p`/`top_n` threshold but
                participates in a kept pair. Without this, a feature that
                only matters jointly with another can be dropped before the
                model -- or `tsk_order='full-2nd'`'s cross terms -- ever see
                it. See `docs/interaction-detection.md`.
            interaction_top_p: Per-pair lift threshold for
                `gauss_math.take_top_interactions`, same semantics as `top_p`.
                Only used when `detect_interactions=True`.
            select_interactions: If True (and `tsk_order='full-2nd'`),
                additionally screen the detected candidate pairs with
                `regression.select_interaction_terms`'s LassoCV -- a final
                sparsity pass over the shortlist rather than the dense
                all-`n_choose_2`-pairs default `full-2nd` otherwise uses. Has
                no effect for any other `tsk_order` (a warning is raised).
            auto_order_candidates: Orders tried by `tsk_order='auto'`, in the
                order CV screens them (first entry wins ties). Ignored otherwise.
            rbf_n_centers: For 'gaussian-rbf' basis, number of centers per feature
                (produces n_features * rbf_n_centers total centers).
            rbf_gamma: Shape parameter for Gaussian RBF evaluations. Larger values
                create narrower, more localized RBFs (default 1.0).
            rbf_radius: Compact support radius for RBF basis. RBFs are exactly zero
                outside this radius. If None (default), RBFs have infinite support.
                Typical values: 0.5-1.0 in normalized feature space.
            firing_exponent: Blend-concentration exponent applied to the rule
                firing strengths before normalization, in the solve and at
                predict time alike. 1.0 (default) is the shipped TSK weighting
                (an exact no-op); >1 concentrates the blend toward the strongest
                rule, <1 flattens it toward a uniform average. See
                `regression.apply_firing_exponent`.
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
        self.interaction_pairs_ = None
        self.cross_pairs_ = None
        self.rbf_centers_ = None
        self.tsk_order_ = None
        self.consequent_selection_ = None

        self.top_n = top_n
        self.top_p = top_p
        self.correlation_threshold = correlation_threshold
        self.n_gaussians = n_gaussians
        self.n_output_buckets = n_output_buckets
        # Equal-WIDTH output buckets by default. "quantile" restores the previous
        # equal-frequency behaviour, including the pinned extreme centroids that
        # shipped with it. See `regression.partition_output` for why uniform is the
        # default and for what to do when the target is badly skewed.
        self.output_partition = output_partition
        self.tsk_order = tsk_order
        self.optimize_coefficients = optimize_coefficients
        self.consequent_basis = consequent_basis
        self.l2_reg = l2_reg
        self.pin_extremes = pin_extremes
        self.norm_conorm = norm_conorm
        self.t_norm = t_norm
        self.t_conorm = t_conorm
        self.allow_mixed_norms = allow_mixed_norms
        self.member_function = member_function
        self.trapz_method = trapz_method
        self.trapz_width_reg = trapz_width_reg
        self.random_state = random_state
        self.max_samples = max_samples
        self.detect_interactions = detect_interactions
        self.interaction_top_p = interaction_top_p
        self.select_interactions = select_interactions
        self.auto_order_candidates = auto_order_candidates
        self.rbf_n_centers = rbf_n_centers
        self.rbf_gamma = rbf_gamma
        self.rbf_radius = rbf_radius
        self.firing_exponent = firing_exponent

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
        y_partitioned, self.y_bucket_mean_ = partition_output(
            self.n_output_buckets, y_series, method=self.output_partition)

        # Calculate feature differentiators
        # When detect_interactions=True, we need all features to find interactions,
        # so only pass top_n when interaction detection is disabled
        top_n_for_correlation = -1 if self.detect_interactions else self.top_n
        self.feature_differentiators_ = calculate_gaussian_correlation(
            X_df, y_partitioned["y_bucket"], top_n=top_n_for_correlation,
            correlation_threshold=self.correlation_threshold,
        )

        # Select top features
        self.top_n_actual_, self.top_features_ = take_top_features(
            self.feature_differentiators_, top_p=self.top_p, top_n=self.top_n
        )

        # Detect candidate interacting pairs *before* the feature list is
        # frozen: a feature that is individually below `top_p`/`top_n` but
        # jointly informative with another would otherwise never reach the
        # model, cross terms or not (see docs/interaction-detection.md).
        self.interaction_pairs_ = []
        self.cross_pairs_ = None
        if self.detect_interactions:
            interaction_scores = calculate_interaction_scores(
                X_df, y_partitioned["y_bucket"], self.feature_differentiators_,
            )
            self.interaction_pairs_ = take_top_interactions(
                interaction_scores, top_p=self.interaction_top_p
            )
            self.top_features_ = rescue_interacting_features(
                self.top_features_, self.feature_differentiators_, self.interaction_pairs_
            )
            self.top_n_actual_ = len(self.top_features_)

            if self.tsk_order in ("full-2nd", "auto"):
                # "auto" may or may not resolve to full-2nd, but cross_pairs_ is
                # only ever consumed when the resolved order is full-2nd
                # (build_consequent_features ignores it otherwise), so it is
                # harmless to prepare unconditionally here.
                # cross_pairs is index-space into the *final* (post-rescue)
                # top_features_ order -- build_consequent_features/
                # solve_tsk_consequents/predict_tsk all key on that ordering.
                name_to_idx = {name: i for i, name in enumerate(self.top_features_)}
                candidate_idx_pairs = sorted(
                    (name_to_idx[fi], name_to_idx[fj]) if name_to_idx[fi] < name_to_idx[fj]
                    else (name_to_idx[fj], name_to_idx[fi])
                    for fi, fj in self.interaction_pairs_
                )
                if self.select_interactions:
                    self.cross_pairs_ = select_interaction_terms(
                        X_df, self.top_features_, y_partitioned, self.y_bucket_mean_,
                        random_state=self.random_state, candidate_pairs=candidate_idx_pairs,
                    )
                else:
                    self.cross_pairs_ = candidate_idx_pairs
            elif self.select_interactions:
                warnings.warn(
                    "select_interactions=True has no effect unless tsk_order='full-2nd' "
                    "(cross_pairs is only consumed by that order); detected interactions "
                    "still rescued qualifying features into top_features_.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # Create membership model (Gaussian, trapezoid, or triangular)
        if self.member_function == "gaussian":
            self.model_ = create_gaussian_membership_dict(
                X_df, y_partitioned["y_bucket"], top_n_var_names=self.top_features_, n_gaussians=self.n_gaussians,
                max_samples=self.max_samples, random_state=self.random_state,
            )
        elif self.member_function == "trap":
            if self.trapz_method == "fast":
                from .trapz_math_fast import create_trapz_membership_dict_fast
                self.model_ = create_trapz_membership_dict_fast(
                    X_df, y_partitioned["y_bucket"], top_n_var_names=self.top_features_
                )
            elif self.trapz_method == "em":
                from .trapz_math import create_trapz_membership_dict
                self.model_ = create_trapz_membership_dict(
                    X_df, y_partitioned["y_bucket"], top_n_var_names=self.top_features_, n_trapezoids=self.n_gaussians,
                    max_samples=self.max_samples, random_state=self.random_state,
                    width_reg=self.trapz_width_reg,
                )
            else:
                raise ValueError(f"Unknown trapz_method: {self.trapz_method}")
        elif self.member_function == "triangular":
            # Triangle is degenerate trapezoid; reuse EM code with shape="triangle".
            from .trapz_math import create_trapz_membership_dict
            self.model_ = create_trapz_membership_dict(
                X_df, y_partitioned["y_bucket"], top_n_var_names=self.top_features_, n_trapezoids=self.n_gaussians,
                max_samples=self.max_samples, random_state=self.random_state, shape="triangle",
                width_reg=self.trapz_width_reg,
            )
        else:
            raise ValueError(f"Unknown member_function: {self.member_function}")

        self.n_rules_ = self.model_.n_rules

        # TODO(#85): self.model_ is a GaussianMixtureModel built the same way as
        # TribbleClassifier's (same create_gaussian_membership_dict call), so it
        # carries the same antecedent redundancy TribbleClassifier.deduplicate()/
        # to_simple_model() now expose -- but there is no regression-side
        # equivalent yet. Unlike the classifier, dropping the *consequent*
        # coefficients tied to a removed membership function is not obviously
        # safe here (solve_tsk_consequents/predict_tsk key rule outputs off the
        # firing-strength columns this model produces), so wiring dedup in
        # requires a regression-side deployable path analogous to
        # SimpleGaussianClassifierModel/simple_gaussian_predict -- e.g. driving
        # a deduplicated GaussianMixtureModel through regression.predict_tsk --
        # not just calling remove_duplicate_membership_fcns() here.

        # Compute RBF centers if using Gaussian RBF basis
        if self.consequent_basis == "gaussian-rbf":
            X_features = X_df[self.top_features_].to_numpy()
            self.rbf_centers_ = compute_rbf_centers(X_features, n_centers=self.rbf_n_centers)
        else:
            self.rbf_centers_ = None

        # Resolve tsk_order="auto" to a concrete order via k-fold CV over
        # auto_order_candidates, before the (single) closed-form consequent
        # solve below -- see issue #120: full-2nd overfits catastrophically
        # once rows/coeff drops below ~5, and this is the automatic guard
        # against picking it blind on small data. self.tsk_order itself is
        # left untouched (sklearn's clone()/get_params() require __init__
        # args to round-trip unmodified); the resolved order lives in
        # tsk_order_, which predict() also reads.
        self.tsk_order_ = self.tsk_order
        self.consequent_selection_ = None
        if self.tsk_order == "auto":
            self.consequent_selection_ = select_consequent_hyperparams(
                X_df, self.model_, self.top_features_,
                self.y_bucket_mean_, y_partitioned,
                n_output_buckets=self.n_output_buckets,
                candidate_orders=self.auto_order_candidates,
                candidate_bases=(self.consequent_basis,),
                candidate_l2=(self.l2_reg,),
                pin_extremes=self.pin_extremes,
                random_state=self.random_state,
                rbf_n_centers=self.rbf_n_centers, rbf_gamma=self.rbf_gamma,
                rbf_radius=self.rbf_radius,
            )
            self.tsk_order_ = self.consequent_selection_["order"]

        # Solve TSK consequents in closed form: for fixed firing strengths the
        # output is linear in the coefficients, so a single ridge least-squares
        # solve yields the exact firing-weighted optimum.
        self.corr_terms_, self.y_bucket_mean_ = solve_tsk_consequents(
            X_df, self.model_, self.top_features_,
            self.y_bucket_mean_, y_partitioned,
            n_output_buckets=self.n_output_buckets,
            order=self.tsk_order_, l2_reg=self.l2_reg, basis=self.consequent_basis,
            pin_extremes=self.pin_extremes,
            norms=self._norms(),
            cross_pairs=self.cross_pairs_,
            verbose=False,
            rbf_centers=self.rbf_centers_, rbf_gamma=self.rbf_gamma,
            rbf_radius=self.rbf_radius,
            firing_exponent=self.firing_exponent,
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
            order=self.tsk_order_, basis=self.consequent_basis,
            norms=self._norms(),
            cross_pairs=self.cross_pairs_,
            rbf_centers=self.rbf_centers_, rbf_gamma=self.rbf_gamma,
            rbf_radius=self.rbf_radius,
            firing_exponent=self.firing_exponent,
        )


class MimoGaussianPredictor(BaseEstimator, RegressorMixin):
    """
    Multi-input multi-output wrapper around TribbleRegressor.

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
        return TribbleRegressor(
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
