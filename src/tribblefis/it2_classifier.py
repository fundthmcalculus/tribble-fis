"""Interval Type-2 Fuzzy Classifier."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets

from .gauss_data import (
    IT2GaussianMembership,
    IT2FeatureModel,
    IT2LabelModel,
    IT2GaussianMixtureModel,
    GaussianMembership,
    DefaultNormCornorm,
    resolve_norm_pair,
)
from .gaussian_classifier import MixtureOfGaussiansFuzzyClassifier
from .it2_kernel import it2_firing_strengths


class IntervalType2FuzzyClassifier(BaseEstimator, ClassifierMixin):
    """Interval Type-2 Fuzzy classifier with automatic uncertainty quantification.

    Converts a Type-1 TSK classifier to IT2 by creating upper and lower bound
    membership functions from the learned Gaussian parameters. The uncertainty
    width is controlled by the `uncertainty_width` parameter.

    Parameters
    ----------
    top_n : int, default=-1
        Number of top features to select based on differentiation score.
        If > 0, top_p is ignored.

    top_p : float, default=0.95
        Per-feature score threshold for feature selection.

    n_gaussians : int, default=0
        Number of Gaussians per feature per label (0 for automatic).

    uncertainty_width : float, default=0.5
        Controls the footprint of uncertainty. Each Gaussian (mu, sigma) is
        expanded to:
            upper_mf: mu + uncertainty_width * sigma, sigma
            lower_mf: mu - uncertainty_width * sigma, sigma
        Larger values create wider uncertainty intervals.

    km_iterations : int | None, default=10
        Number of Karnik-Mendel iterations for type reduction.
        None or 0 uses simple averaging (faster, less accurate).
        Higher values give tighter type-reduced outputs.

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

    classes_ : ndarray
        Unique class labels from training.

    feature_names_in_ : list
        Feature names from X during fit.
    """

    def __init__(
        self,
        top_n=-1,
        top_p=0.95,
        n_gaussians=0,
        uncertainty_width=0.5,
        km_iterations=10,
        norm_conorm=DefaultNormCornorm,
        random_state=42,
        max_samples=None,
    ):
        self.top_n = top_n
        self.top_p = top_p
        self.n_gaussians = n_gaussians
        self.uncertainty_width = uncertainty_width
        self.km_iterations = km_iterations
        self.norm_conorm = norm_conorm
        self.random_state = random_state
        self.max_samples = max_samples

        # Will be set during fit
        self.model_ = None
        self.classes_ = None
        self.feature_names_in_ = None
        self.norms_ = None
        self._base_classifier = None

    def fit(self, X, y):
        """Fit the IT2 classifier by first fitting a Type-1 model, then converting to IT2.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input data.

        y : array-like of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : IntervalType2FuzzyClassifier
            Fitted estimator.
        """
        X, y = check_X_y(X, y, accept_sparse=False, dtype=None)
        check_classification_targets(y)

        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            feature_names = [f"x{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=feature_names)
        else:
            feature_names = list(X.columns)

        self.feature_names_in_ = feature_names
        self.classes_ = np.unique(y)
        self.norms_ = resolve_norm_pair(self.norm_conorm)

        # Fit base Type-1 classifier
        base = MixtureOfGaussiansFuzzyClassifier(
            top_n=self.top_n,
            top_p=self.top_p,
            n_gaussians=self.n_gaussians,
            norm_conorm=self.norm_conorm,
            random_state=self.random_state,
            max_samples=self.max_samples,
        )
        base.fit(X, y)
        self._base_classifier = base

        # Convert Type-1 model to IT2
        self.model_ = self._convert_to_it2(base.model_)

        return self

    def predict(self, X):
        """Predict class labels using IT2 firing strengths.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self, ["model_", "classes_"])

        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        _, _, firing_crisp, labels = it2_firing_strengths(
            X, self.model_, self.norms_, km_iterations=self.km_iterations
        )

        # Argmax over crisp firing strengths
        class_indices = np.argmax(firing_crisp, axis=1)
        return self.classes_[class_indices]

    def predict_intervals(self, X):
        """Predict confidence intervals for each class.

        Returns the upper and lower bound firing strengths before type reduction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        upper : ndarray of shape (n_samples, n_classes)
            Upper bound firing strengths.

        lower : ndarray of shape (n_samples, n_classes)
            Lower bound firing strengths.
        """
        check_is_fitted(self, ["model_", "classes_"])

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        firing_upper, firing_lower, _, _ = it2_firing_strengths(
            X, self.model_, self.norms_, km_iterations=None
        )

        return firing_upper, firing_lower

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
