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
from .gaussian_classifier import TribbleClassifier
from .it2_kernel import it2_firing_strengths
from .it2_refine import refine_it2_antecedents


class IT2TribbleClassifier(BaseEstimator, ClassifierMixin):
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
        Controls the footprint of uncertainty. For Gaussian MFs:
        upper_sigma = sigma * (1 + uncertainty_width), lower_sigma = sigma * (1 - uncertainty_width/2)

    km_iterations : int | None, default=10
        Number of Karnik-Mendel iterations for type reduction.
        None or 0 uses simple averaging (faster, less accurate).
        Higher values give tighter type-reduced outputs.

    norm_conorm : str, default="probability"
        Fuzzy operator family: "min/max", "probability", "luk", "hamacher", "einstein".

    refine : bool, default=False
        If True, refines the Type-1 model before converting to IT2. The refined
        parameters are then used as the center points for the uncertainty expansion.
        Refinement is applied to the base Type-1 model, improving the discriminative
        power of both upper and lower membership functions.

    refine_method : str, default="coordinate"
        Method for antecedent refinement: "coordinate" (block coordinate descent) or "none".

    refine_l2_shrink : float, default=0.05
        L2 regularization strength during refinement.

    refine_it2 : bool, default=False
        If True, additionally refines the IT2 upper/lower Gaussian antecedents
        *after* conversion (`it2_refine.refine_it2_antecedents`), directly
        against the type-reduced classification loss -- distinct from `refine`,
        which only ever touches the pre-conversion Type-1 model and never sees
        the footprint of uncertainty it becomes. Off by default because it is
        additional optimization work on top of `refine`, not a substitute.

    refine_it2_n_sweeps : int, default=3
        Number of coordinate-descent sweeps for `refine_it2` (see
        `it2_refine.refine_it2_antecedents`).

    refine_it2_l2_shrink : float, default=0.05
        L2 anchor strength for `refine_it2`'s coordinate descent.

    random_state : int, default=42
        Seed for reproducibility.

    max_samples : int | None, default=None
        Cap on rows used per (feature, label) when fitting memberships.

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
        refine=False,
        refine_method="coordinate",
        refine_l2_shrink=0.05,
        refine_it2=False,
        refine_it2_n_sweeps=3,
        refine_it2_l2_shrink=0.05,
        random_state=42,
        max_samples=None,
    ):
        self.top_n = top_n
        self.top_p = top_p
        self.n_gaussians = n_gaussians
        self.uncertainty_width = uncertainty_width
        self.km_iterations = km_iterations
        self.norm_conorm = norm_conorm
        self.refine = refine
        self.refine_method = refine_method
        self.refine_l2_shrink = refine_l2_shrink
        self.refine_it2 = refine_it2
        self.refine_it2_n_sweeps = refine_it2_n_sweeps
        self.refine_it2_l2_shrink = refine_it2_l2_shrink
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
        self : IT2TribbleClassifier
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
        base = TribbleClassifier(
            top_n=self.top_n,
            top_p=self.top_p,
            n_gaussians=self.n_gaussians,
            norm_conorm=self.norm_conorm,
            refine=self.refine,
            refine_method=self.refine_method,
            refine_l2_shrink=self.refine_l2_shrink,
            random_state=self.random_state,
            max_samples=self.max_samples,
        )
        base.fit(X, y)
        self._base_classifier = base

        # Convert Type-1 model to IT2
        self.model_ = self._convert_to_it2(base.model_)

        if self.refine_it2:
            # `it2_firing_strengths`'s columns are `sorted(model.all_output_labels)`,
            # which is exactly `self.classes_` (`predict` indexes `class_indices`
            # into `self.classes_` the same way) -- so mapping y through
            # `searchsorted` on the (already-sorted) `classes_` gives the column
            # index each row's true label corresponds to.
            y_idx = np.searchsorted(self.classes_, y)
            self.model_ = refine_it2_antecedents(
                X, y_idx, self.model_, self.norms_,
                km_iterations=self.km_iterations if self.km_iterations else 10,
                n_sweeps=self.refine_it2_n_sweeps,
                l2_shrink=self.refine_it2_l2_shrink,
                verbose=False,
            )

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

        # Convert to DataFrame if needed, with correct column names
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        else:
            # Ensure DataFrame has correct column names
            X = pd.DataFrame(X.values, columns=self.feature_names_in_)

        _, _, firing_crisp, labels = it2_firing_strengths(
            X, self.model_, self.norms_, km_iterations=self.km_iterations
        )

        # Argmax over crisp firing strengths
        class_indices = np.argmax(firing_crisp, axis=1)
        return self.classes_[class_indices]

    def predict_intervals(self, X):
        """Predict per-class firing-strength intervals.

        Returns the upper and lower bound firing strengths before type reduction.

        This is antecedent-boundary ambiguity, not a calibrated confidence
        score: because upper/lower bounds share the same `mu` (only `sigma`
        is scaled -- see `_convert_to_it2`), `upper - lower` is exactly 0 at
        `mu` for any `uncertainty_width`, rises moving away from `mu`, then
        falls back toward 0 again in the tails as both bounds decay -- a hump,
        not a ramp in firing strength. Empirically (issue #149,
        `docs/t1-it2-gt2-tradeoff.md`) this means width does not track
        correctness monotonically: whether correct or incorrect predictions
        land on the rising or falling side of that hump depends on the
        dataset, and is not something `uncertainty_width` can fix.

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
        else:
            # Ensure DataFrame has correct column names
            X = pd.DataFrame(X.values, columns=self.feature_names_in_)

        firing_upper, firing_lower, _, _ = it2_firing_strengths(
            X, self.model_, self.norms_, km_iterations=None
        )

        return firing_upper, firing_lower

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
