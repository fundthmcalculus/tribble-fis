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
    resolve_norm_pair,
)
from .gaussian_regressor import MixtureOfGaussiansFuzzyRegressor
from .it2_kernel import it2_firing_strengths


class IntervalType2FuzzyRegressor(BaseEstimator, RegressorMixin):
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
        Controls the footprint of uncertainty. Each Gaussian (mu, sigma) is
        expanded to:
            upper_mf: mu + uncertainty_width * sigma, sigma
            lower_mf: mu - uncertainty_width * sigma, sigma

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
        self : IntervalType2FuzzyRegressor
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
        base = MixtureOfGaussiansFuzzyRegressor(
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

    def predict(self, X):
        """Predict target values using IT2 firing strengths with type reduction.

        Returns crisp point estimates via Karnik-Mendel type reduction.

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

        _, _, firing_crisp, _ = it2_firing_strengths(
            X, self.model_, self.norms_, km_iterations=self.km_iterations
        )

        # Combine firing strengths to get single prediction per sample
        # Take weighted average (mean of firing strengths per sample)
        y_normalized = np.mean(firing_crisp, axis=1)

        # Scale back to original target range
        y_pred = self.y_min_ + y_normalized * (self.y_max_ - self.y_min_)

        return y_pred

    def predict_intervals(self, X):
        """Predict confidence intervals for target values.

        Returns the upper and lower bound predictions without type reduction.

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

        firing_upper, firing_lower, _, _ = it2_firing_strengths(
            X, self.model_, self.norms_, km_iterations=None
        )

        # Normalize and scale back to original range
        y_upper_normalized = np.mean(firing_upper, axis=1)
        y_lower_normalized = np.mean(firing_lower, axis=1)

        y_upper = self.y_min_ + y_upper_normalized * (self.y_max_ - self.y_min_)
        y_lower = self.y_min_ + y_lower_normalized * (self.y_max_ - self.y_min_)

        return y_lower, y_upper

    def _convert_to_it2(self, type1_model) -> IT2GaussianMixtureModel:
        """Convert a Type-1 GaussianMixtureModel to IT2.

        For each Gaussian (mu, sigma), creates:
            upper_mf: mu + uncertainty_width * sigma, sigma
            lower_mf: mu - uncertainty_width * sigma, sigma
        """
        feature_models = {}

        for feature_name, type1_feature_model in type1_model.feature_models.items():
            label_models = {}

            for label, type1_label_model in type1_feature_model.label_models.items():
                it2_mfs = []

                for gauss_mf in type1_label_model.memberships:
                    # Create upper and lower bounds from the Gaussian
                    upper_mf = GaussianMembership(
                        mu=gauss_mf.mu + self.uncertainty_width * gauss_mf.sigma,
                        sigma=gauss_mf.sigma,
                        id=gauss_mf.id,
                    )
                    lower_mf = GaussianMembership(
                        mu=gauss_mf.mu - self.uncertainty_width * gauss_mf.sigma,
                        sigma=gauss_mf.sigma,
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
