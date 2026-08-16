"""Interval Type-2 Fuzzy Regressor."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted

from .gauss_data import (
    IT2GaussianMembership,
    IT2FeatureModel,
    IT2LabelModel,
    IT2GaussianMixtureModel,
    GaussianMembership,
    DefaultNormCornorm,
    DefaultMemberFunction,
    resolve_norm_pair,
)
from .gaussian_regressor import TribbleRegressor
from .it2_kernel import it2_firing_strengths, karnik_mendel_tsk
from .regression import rule_consequent_values, _normalize_firing_strengths


class T2TribbleRegressor(BaseEstimator, RegressorMixin):
    """Interval Type-2 Fuzzy regressor with uncertainty quantification.

    Converts a Type-1 TSK regressor to IT2 by creating upper and lower bound
    membership functions from the learned Gaussian parameters. Outputs are
    automatically scaled back to the original target range.

    Parameters
    ----------
    top_n : int, default=-1
        Number of top features to select based on differentiation score.
        If > 0, top_p is ignored.

    top_p : float, default=0.95
        Per-feature score threshold for feature selection.

    n_gaussians : int, default=0
        Number of Gaussians per feature per label (0 for automatic).

    n_output_buckets : int, default=2
        Number of output buckets for partitioning y during training.

    uncertainty_width : float, default=0.5
        Controls the footprint of uncertainty. Each Gaussian (mu, sigma) creates:
            upper_mf: mu, sigma * (1 + uncertainty_width)
            lower_mf: mu, sigma * (1 - uncertainty_width/2)

    km_iterations : int | None, default=10
        Number of Karnik-Mendel iterations for type reduction.
        None or 0 uses simple averaging (faster).

    norm_conorm : str, default="probability"
        Fuzzy operator family: "min/max", "probability", "luk", "hamacher", "einstein".

    random_state : int, default=42
        Seed for reproducibility.

    max_samples : int | None, default=None
        Cap on rows used per (feature, label) when fitting Gaussians.

    Attributes
    ----------
    model_ : IT2GaussianMixtureModel
        The fitted IT2 model with upper and lower membership functions.

    y_min_ : float
        Minimum target value from training (for scaling).

    y_max_ : float
        Maximum target value from training (for scaling).

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
        km_iterations=10,
        norm_conorm=DefaultNormCornorm,
        random_state=42,
        max_samples=None,
    ):
        self.top_n = top_n
        self.top_p = top_p
        self.n_gaussians = n_gaussians
        self.n_output_buckets = n_output_buckets
        self.uncertainty_width = uncertainty_width
        self.km_iterations = km_iterations
        self.norm_conorm = norm_conorm
        self.random_state = random_state
        self.max_samples = max_samples

        # Will be set during fit
        self.model_ = None
        self.y_min_ = None
        self.y_max_ = None
        self.feature_names_in_ = None
        self.norms_ = None
        self._base_regressor = None

    def fit(self, X, y):
        """Fit the IT2 regressor by first fitting a Type-1 model, then converting to IT2.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input data.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : T2TribbleRegressor
            Fitted estimator.
        """
        X, y = check_X_y(X, y, accept_sparse=False, dtype=None, multi_output=False)

        # Convert to DataFrame if needed
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

        # Fit base Type-1 regressor
        base = TribbleRegressor(
            top_n=self.top_n,
            top_p=self.top_p,
            n_gaussians=self.n_gaussians,
            n_output_buckets=self.n_output_buckets,
            norm_conorm=self.norm_conorm,
            random_state=self.random_state,
            max_samples=self.max_samples,
        )
        base.fit(X, y)
        self._base_regressor = base

        # Convert Type-1 model to IT2
        self.model_ = self._convert_to_it2(base.model_)

        return self

    def _rule_bounds_and_values(self, X):
        """Shared setup for `predict`/`predict_intervals`: each rule's raw firing
        bounds (from the IT2 antecedents) and its own crisp consequent output
        (from the base Type-1 regressor's fitted consequents), both keyed on the
        same rule ordering."""
        firing_upper, firing_lower, _, labels = it2_firing_strengths(
            X, self.model_, self.norms_, km_iterations=None
        )
        base = self._base_regressor
        rule_values = rule_consequent_values(
            X, base.top_features_, labels, base.y_bucket_mean_, base.corr_terms_,
            order=base.tsk_order, basis=base.consequent_basis, cross_pairs=base.cross_pairs_,
        )
        return firing_upper, firing_lower, rule_values

    def predict(self, X):
        """Predict target values via Karnik-Mendel type reduction of the type-2
        TSK output.

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
            # Ensure DataFrame has correct column names
            X = pd.DataFrame(X.values, columns=self.feature_names_in_)

        y_lower, y_upper = self._predict_interval_arrays(X)
        return 0.5 * (y_lower + y_upper)

    def _predict_interval_arrays(self, X):
        """(y_l, y_r): the Karnik-Mendel output interval, computed by combining
        every rule's own consequent value under its interval firing-strength
        weight (`it2_kernel.karnik_mendel_tsk`) -- rather than, as previously,
        type-reducing each rule's firing strength to a crisp weight *first* and
        only then running it through the same weighted-consequent evaluation
        Type-1 uses. That earlier two-stage pipeline never gave the switch-point
        search the one thing it needs to do anything: each rule's own consequent
        value (see `it2_kernel`'s module docstring). It also could not guarantee
        `predict`'s output landed inside `predict_intervals`'s bounds -- each of
        the three stood on its own row-normalization, so at the boundary between
        `predict`'s crisp weights and `predict_intervals`'s raw upper/lower
        bounds, roughly 3% of rows on a `make_regression` target fell outside
        the interval, needing the min/max-of-three workaround this replaces.
        Karnik-Mendel structurally cannot fail that check: it directly searches
        `y_l` and `y_r` as the minimum and maximum of the same weighted average
        `predict`'s midpoint is drawn from.

        `km_iterations=None`/`0` skips the switch-point search for a faster,
        approximate interval: each bound is the plain weighted average using
        that bound's own raw firing strengths as weights (no cross-rule
        optimization), which is not guaranteed to bracket the crisp midpoint.
        """
        firing_upper, firing_lower, rule_values = self._rule_bounds_and_values(X)

        if self.km_iterations is None or self.km_iterations == 0:
            y_upper = np.sum(_normalize_firing_strengths(firing_upper) * rule_values, axis=1)
            y_lower = np.sum(_normalize_firing_strengths(firing_lower) * rule_values, axis=1)
            return np.minimum(y_lower, y_upper), np.maximum(y_lower, y_upper)

        return karnik_mendel_tsk(
            rule_values, firing_lower, firing_upper, max_iterations=self.km_iterations
        )

    def predict_intervals(self, X):
        """Predict confidence intervals for target values.

        Returns the Karnik-Mendel type-reduced output interval -- guaranteed by
        construction to contain `predict`'s point estimate, which is this
        interval's midpoint.

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
            # Ensure DataFrame has correct column names
            X = pd.DataFrame(X.values, columns=self.feature_names_in_)

        return self._predict_interval_arrays(X)

    def _convert_to_it2(self, type1_model) -> IT2GaussianMixtureModel:
        """Convert a Type-1 model to IT2.

        For each Gaussian membership (mu, sigma), creates:
            upper_mf: mu, sigma * (1 + uncertainty_width)  [wider, more permissive]
            lower_mf: mu, sigma * (1 - uncertainty_width/2)  [narrower, more restrictive]

        Ensures all sigmas are at least 1e-4 to avoid degenerate memberships.
        """
        feature_models = {}
        min_sigma = 1e-4

        for feature_name, type1_feature_model in type1_model.feature_models.items():
            label_models = {}

            for label, type1_label_model in type1_feature_model.label_models.items():
                it2_mfs = []

                for gauss_mf in type1_label_model.memberships:
                    # Ensure base sigma is not zero or negligible
                    base_sigma = max(gauss_mf.sigma, min_sigma)

                    # Create upper and lower bounds by expanding/shrinking sigma
                    # Upper: wider sigma (more permissive, fires more readily)
                    # Lower: narrower sigma (more restrictive, fires less readily)
                    upper_mf = GaussianMembership(
                        mu=gauss_mf.mu,
                        sigma=base_sigma * (1.0 + self.uncertainty_width),
                        id=gauss_mf.id,
                    )
                    lower_mf = GaussianMembership(
                        mu=gauss_mf.mu,
                        sigma=base_sigma * max(0.1, 1.0 - self.uncertainty_width),
                        id=gauss_mf.id,
                    )

                    it2_mf = IT2GaussianMembership(
                        upper_mf=upper_mf,
                        lower_mf=lower_mf,
                    )
                    it2_mfs.append(it2_mf)

                label_models[label] = IT2LabelModel(it2_mfs)

            feature_models[feature_name] = IT2FeatureModel(label_models)

        return IT2GaussianMixtureModel(feature_models)
