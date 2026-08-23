"""General Type-2 Fuzzy Regressor (alpha-plane representation)."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y, check_is_fitted

from .gauss_data import (
    GT2FeatureModel,
    GT2LabelModel,
    GT2GaussianMixtureModel,
    DefaultNormCornorm,
    resolve_norm_pair,
    widen_membership,
    to_gt2_membership,
)
from .gaussian_regressor import TribbleRegressor
from .gt2_kernel import gt2_rule_firing, gt2_karnik_mendel_tsk, alpha_weighted_average
from .gt2_refine import refine_gt2_regressor_antecedents
from .regression import (
    rule_consequent_values,
    _normalize_firing_strengths,
    conformal_calibration_margin,
)


class GT2TribbleRegressor(BaseEstimator, RegressorMixin):
    """General Type-2 Fuzzy regressor via alpha-plane decomposition.

    Converts a Type-1 TSK regressor to GT2 the same way
    `IT2TribbleRegressor` converts to IT2, plus a `principal_mf` per
    membership carrying the original (un-widened) Type-1 sigma (see
    `GT2TribbleClassifier._convert_to_gt2`). Prediction decomposes each
    membership's implied triangular secondary grade into `n_alpha_planes`
    ordinary IT2 sets, runs today's Karnik-Mendel search
    (`it2_kernel.karnik_mendel_tsk`) unchanged once per plane
    (`gt2_kernel.gt2_rule_firing` + `gt2_karnik_mendel_tsk`), and combines
    with an alpha-weighted average -- see `docs/gt2-evaluation.md` for the
    survey this implements.

    Parameters
    ----------
    top_n, top_p, n_gaussians, n_output_buckets, norm_conorm, member_function,
    trapz_method, random_state, max_samples : as in `IT2TribbleRegressor` --
        the Type-1 base model this converts is fit identically.

    uncertainty_width : float, default=0.5
        Controls the footprint of uncertainty, exactly as in
        `IT2TribbleRegressor`.

    n_alpha_planes : int, default=5
        Number of alpha-planes to combine (see `gt2_kernel.default_alpha_levels`).
        Cost is linear in this (measured in `docs/gt2-evaluation.md`); 5-10 is
        the recommended range for this library's typical model sizes.

    km_iterations : int | None, default=10
        Karnik-Mendel iterations for *each* alpha-plane's own type reduction.
        `None`/`0` skips the switch-point search per plane for a faster,
        approximate per-plane interval (plain firing-strength-weighted
        average, no cross-rule optimization -- mirrors
        `IT2TribbleRegressor`'s own fast path exactly, just run once per
        plane before the alpha-combination).

    refine_gt2 : bool, default=False
        If True, refines the GT2 upper/lower/principal Gaussian antecedents
        after conversion (`gt2_refine.refine_gt2_regressor_antecedents`)
        against held-out MSE, re-solving TSK consequents in closed form for
        every candidate -- the GT2 analogue of `IT2TribbleRegressor`'s
        `refine_it2`.

    refine_gt2_n_sweeps : int, default=3
        Number of coordinate-descent sweeps for `refine_gt2`.

    refine_gt2_km_iterations : int | None, default=None
        Karnik-Mendel iterations (per plane) used for the *loss* evaluated
        during `refine_gt2`'s search. `None` falls back to `km_iterations`
        (or 15 if that is also `None`/`0`).

    refine_gt2_n_folds : int, default=3
        Cross-validation folds for `refine_gt2`'s held-out MSE objective.

    conformal_calibration, conformal_alpha, conformal_calibration_frac :
        as in `IT2TribbleRegressor` -- fixes #149's regression coverage
        finding by padding `predict_intervals`'s output with an additive
        split-conformal margin, since the raw interval's coverage plateaus
        well under any target no matter how `uncertainty_width` is tuned.

    Attributes
    ----------
    model_ : GT2GaussianMixtureModel
        The fitted GT2 model.

    conformal_margin_ : float | None
        Additive margin from `conformal_calibration`, or `None` if it was
        never enabled.

    y_bucket_mean_, corr_terms_ : ndarray
        Rule consequent coefficients currently in use for prediction -- the
        base Type-1 regressor's own fit, unless `refine_gt2=True`.

    y_min_, y_max_ : float
        Target range from training.

    feature_names_in_ : list
        Feature names from X during fit.
    """

    def __init__(
        self,
        top_n=-1,
        top_p=0.95,
        n_gaussians=0,
        n_output_buckets=2,
        uncertainty_width=0.5,
        n_alpha_planes=5,
        km_iterations=10,
        norm_conorm=DefaultNormCornorm,
        member_function="gaussian",
        trapz_method="fast",
        refine_gt2=False,
        refine_gt2_n_sweeps=3,
        refine_gt2_km_iterations=None,
        refine_gt2_n_folds=3,
        conformal_calibration=False,
        conformal_alpha=0.1,
        conformal_calibration_frac=0.2,
        random_state=42,
        max_samples=None,
    ):
        self.top_n = top_n
        self.top_p = top_p
        self.n_gaussians = n_gaussians
        self.n_output_buckets = n_output_buckets
        self.uncertainty_width = uncertainty_width
        self.n_alpha_planes = n_alpha_planes
        self.km_iterations = km_iterations
        self.norm_conorm = norm_conorm
        self.member_function = member_function
        self.trapz_method = trapz_method
        self.refine_gt2 = refine_gt2
        self.refine_gt2_n_sweeps = refine_gt2_n_sweeps
        self.refine_gt2_km_iterations = refine_gt2_km_iterations
        self.refine_gt2_n_folds = refine_gt2_n_folds
        self.conformal_calibration = conformal_calibration
        self.conformal_alpha = conformal_alpha
        self.conformal_calibration_frac = conformal_calibration_frac
        self.random_state = random_state
        self.max_samples = max_samples

        # Will be set during fit
        self.model_ = None
        self.y_bucket_mean_ = None
        self.corr_terms_ = None
        self.conformal_margin_ = None
        self.refine_gt2_info_ = None
        self.y_min_ = None
        self.y_max_ = None
        self.feature_names_in_ = None
        self.norms_ = None
        self._base_regressor = None

    def fit(self, X, y):
        """Fit the GT2 regressor by first fitting a Type-1 model, then
        converting to GT2.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input data.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : GT2TribbleRegressor
            Fitted estimator.
        """
        X, y = check_X_y(X, y, accept_sparse=False, dtype=None, multi_output=False)

        if isinstance(X, np.ndarray):
            feature_names = [f"x{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=feature_names)
        else:
            feature_names = list(X.columns)

        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        self.feature_names_in_ = feature_names
        self.y_min_ = float(y.min())
        self.y_max_ = float(y.max())
        self.norms_ = resolve_norm_pair(self.norm_conorm)

        if self.conformal_calibration:
            # Held out before anything else touches the data, so the
            # calibration split satisfies conformal prediction's
            # exchangeability requirement (see `conformal_calibration_margin`).
            X, X_calib, y, y_calib = train_test_split(
                X, y, test_size=self.conformal_calibration_frac,
                random_state=self.random_state,
            )

        base = TribbleRegressor(
            top_n=self.top_n,
            top_p=self.top_p,
            n_gaussians=self.n_gaussians,
            n_output_buckets=self.n_output_buckets,
            norm_conorm=self.norm_conorm,
            member_function=self.member_function,
            trapz_method=self.trapz_method,
            random_state=self.random_state,
            max_samples=self.max_samples,
        )
        base.fit(X, y)
        self._base_regressor = base

        self.model_ = self._convert_to_gt2(base.model_)
        self.y_bucket_mean_ = base.y_bucket_mean_
        self.corr_terms_ = base.corr_terms_

        if self.refine_gt2:
            km_iterations_for_refine = (
                self.refine_gt2_km_iterations
                or self.km_iterations
                or 15
            )
            self.model_, self.corr_terms_, self.y_bucket_mean_, self.refine_gt2_info_ = (
                refine_gt2_regressor_antecedents(
                    X, y, self.model_, self.norms_, base.top_features_,
                    order=base.tsk_order, l2_reg=base.l2_reg, basis=base.consequent_basis,
                    cross_pairs=base.cross_pairs_,
                    n_alpha_planes=self.n_alpha_planes, km_iterations=km_iterations_for_refine,
                    n_sweeps=self.refine_gt2_n_sweeps, n_folds=self.refine_gt2_n_folds,
                    seed=self.random_state,
                    verbose=False,
                )
            )

        if self.conformal_calibration:
            y_lower_calib, y_upper_calib = self._predict_interval_arrays(X_calib)
            self.conformal_margin_ = conformal_calibration_margin(
                y_calib, y_lower_calib, y_upper_calib, self.conformal_alpha
            )

        return self

    def _rule_bounds_and_values(self, X):
        """Shared setup for `predict`/`predict_intervals`: each alpha-plane's
        raw firing bounds and the (alpha-independent) per-rule consequent
        output, keyed on the same rule ordering."""
        firing_uppers, firing_lowers, alphas, labels = gt2_rule_firing(
            self.model_, X, self._base_regressor.top_features_, self.norms_,
            n_alpha_planes=self.n_alpha_planes,
        )
        base = self._base_regressor
        rule_values = rule_consequent_values(
            X, base.top_features_, labels, self.y_bucket_mean_, self.corr_terms_,
            order=base.tsk_order, basis=base.consequent_basis, cross_pairs=base.cross_pairs_,
        )
        return firing_uppers, firing_lowers, alphas, rule_values

    def predict(self, X):
        """Predict target values via alpha-combined Karnik-Mendel type
        reduction of the GT2 TSK output.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Predicted target values.
        """
        check_is_fitted(self, ["model_", "y_min_", "y_max_"])

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        else:
            X = pd.DataFrame(X.values, columns=self.feature_names_in_)

        y_lower, y_upper = self._predict_interval_arrays(X)
        return 0.5 * (y_lower + y_upper)

    def _predict_interval_arrays(self, X):
        """(y_l, y_r): the alpha-combined Karnik-Mendel output interval.

        `km_iterations=None`/`0` skips each plane's switch-point search for a
        faster, approximate per-plane interval (plain weighted average using
        that plane's own raw firing strengths), mirroring
        `IT2TribbleRegressor._predict_interval_arrays`'s fast path exactly --
        just run once per alpha-plane before the alpha-weighted combination.
        """
        firing_uppers, firing_lowers, alphas, rule_values = self._rule_bounds_and_values(X)

        if self.km_iterations is None or self.km_iterations == 0:
            y_uppers = [np.sum(_normalize_firing_strengths(fu) * rule_values, axis=1) for fu in firing_uppers]
            y_lowers = [np.sum(_normalize_firing_strengths(fl) * rule_values, axis=1) for fl in firing_lowers]
            y_upper = alpha_weighted_average(alphas, y_uppers)
            y_lower = alpha_weighted_average(alphas, y_lowers)
            return np.minimum(y_lower, y_upper), np.maximum(y_lower, y_upper)

        return gt2_karnik_mendel_tsk(
            rule_values, firing_uppers, firing_lowers, alphas, max_iterations=self.km_iterations
        )

    def predict_intervals(self, X):
        """Predict confidence intervals for target values.

        Returns the alpha-combined Karnik-Mendel type-reduced output
        interval -- guaranteed by construction to contain `predict`'s point
        estimate, which is this interval's midpoint, exactly as in
        `IT2TribbleRegressor` -- widened by `conformal_margin_` on both sides
        when `conformal_calibration=True` (the margin is symmetric, so this
        guarantee holds regardless).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_lower : ndarray of shape (n_samples,)
            Lower bound predictions.

        y_upper : ndarray of shape (n_samples,)
            Upper bound predictions.
        """
        check_is_fitted(self, ["model_", "y_min_", "y_max_"])

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        else:
            X = pd.DataFrame(X.values, columns=self.feature_names_in_)

        y_lower, y_upper = self._predict_interval_arrays(X)
        if self.conformal_margin_ is not None:
            y_lower = y_lower - self.conformal_margin_
            y_upper = y_upper + self.conformal_margin_
        return y_lower, y_upper

    def _convert_to_gt2(self, type1_model) -> GT2GaussianMixtureModel:
        """Convert a Type-1 model to GT2 -- identical to
        `GT2TribbleClassifier._convert_to_gt2` (see that method's docstring)."""
        feature_models = {}

        for feature_name, type1_feature_model in type1_model.feature_models.items():
            label_models = {}

            for label, type1_label_model in type1_feature_model.label_models.items():
                gt2_mfs = []

                for mf in type1_label_model.memberships:
                    upper_mf, lower_mf = widen_membership(mf, self.uncertainty_width)
                    gt2_mfs.append(to_gt2_membership(upper_mf, lower_mf, principal_mf=mf))

                label_models[label] = GT2LabelModel(gt2_mfs)

            feature_models[feature_name] = GT2FeatureModel(label_models)

        return GT2GaussianMixtureModel(feature_models)
