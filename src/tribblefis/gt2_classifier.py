"""General Type-2 Fuzzy Classifier (alpha-plane representation)."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets

from .gauss_data import (
    GT2GaussianMembership,
    GT2FeatureModel,
    GT2LabelModel,
    GT2GaussianMixtureModel,
    GaussianMembership,
    DefaultNormCornorm,
    resolve_norm_pair,
)
from .gaussian_classifier import TribbleClassifier
from .gt2_kernel import gt2_firing_strengths, extract_alpha_plane_model
from .it2_kernel import it2_firing_strengths
from .gt2_refine import refine_gt2_antecedents


class GT2TribbleClassifier(BaseEstimator, ClassifierMixin):
    """General Type-2 Fuzzy classifier via alpha-plane decomposition.

    Converts a Type-1 TSK classifier to GT2 the same way `IT2TribbleClassifier`
    converts to IT2 -- upper/lower Gaussian bounds from the learned (mu, sigma)
    -- plus one extra Gaussian, `principal_mf`, carrying the *original*
    (un-widened) Type-1 sigma: the single most-likely membership within the
    footprint of uncertainty. Inference decomposes each membership's implied
    triangular secondary grade into `n_alpha_planes` ordinary IT2 sets
    (`gt2_kernel.gt2_firing_strengths`), running today's IT2 forward pass
    unchanged once per plane and combining with an alpha-weighted average --
    see `docs/gt2-evaluation.md` for the survey this implements and
    `gt2_kernel.py`'s module docstring for the mechanism.

    Parameters
    ----------
    top_n, top_p, n_gaussians, norm_conorm, refine, refine_method,
    refine_l2_shrink, random_state, max_samples : as in `IT2TribbleClassifier`
        -- the Type-1 base model this converts is fit identically.

    uncertainty_width : float, default=0.5
        Controls the footprint of uncertainty, exactly as in
        `IT2TribbleClassifier`: upper_sigma = sigma * (1 + uncertainty_width),
        lower_sigma = sigma * max(0.1, 1 - uncertainty_width).

    n_alpha_planes : int, default=5
        Number of alpha-planes to combine per `gt2_kernel.gt2_firing_strengths`
        call. Cost is linear in this (measured in `docs/gt2-evaluation.md`);
        5-10 is the recommended range for this library's typical model sizes.

    km_iterations : int | None, default=10
        Passed through to each alpha-plane's own `it2_firing_strengths` call.
        Accepted for API parity with `IT2TribbleClassifier`; as with that
        estimator, the per-rule classification reduction is provably the
        midpoint regardless of this value (see
        `it2_kernel.karnik_mendel_type_reduction`'s docstring), so it has no
        effect on `predict`'s output here either -- only on
        `predict_intervals`'s cost, which does not use it at all (a plain
        upper/lower average, no search).

    refine_gt2 : bool, default=False
        If True, refines the GT2 upper/lower/principal Gaussian antecedents
        *after* conversion (`gt2_refine.refine_gt2_antecedents`), directly
        against the alpha-combined classification loss -- the GT2 analogue of
        `IT2TribbleClassifier`'s `refine_it2`.

    refine_gt2_n_sweeps : int, default=3
        Number of coordinate-descent sweeps for `refine_gt2`.

    refine_gt2_l2_shrink : float, default=0.05
        L2 anchor strength for `refine_gt2`'s coordinate descent.

    Attributes
    ----------
    model_ : GT2GaussianMixtureModel
        The fitted GT2 model.

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
        n_alpha_planes=5,
        km_iterations=10,
        norm_conorm=DefaultNormCornorm,
        refine=False,
        refine_method="coordinate",
        refine_l2_shrink=0.05,
        refine_gt2=False,
        refine_gt2_n_sweeps=3,
        refine_gt2_l2_shrink=0.05,
        random_state=42,
        max_samples=None,
    ):
        self.top_n = top_n
        self.top_p = top_p
        self.n_gaussians = n_gaussians
        self.uncertainty_width = uncertainty_width
        self.n_alpha_planes = n_alpha_planes
        self.km_iterations = km_iterations
        self.norm_conorm = norm_conorm
        self.refine = refine
        self.refine_method = refine_method
        self.refine_l2_shrink = refine_l2_shrink
        self.refine_gt2 = refine_gt2
        self.refine_gt2_n_sweeps = refine_gt2_n_sweeps
        self.refine_gt2_l2_shrink = refine_gt2_l2_shrink
        self.random_state = random_state
        self.max_samples = max_samples

        # Will be set during fit
        self.model_ = None
        self.classes_ = None
        self.feature_names_in_ = None
        self.norms_ = None
        self._base_classifier = None

    def fit(self, X, y):
        """Fit the GT2 classifier by first fitting a Type-1 model, then
        converting to GT2.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input data.

        y : array-like of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : GT2TribbleClassifier
            Fitted estimator.
        """
        X, y = check_X_y(X, y, accept_sparse=False, dtype=None)
        check_classification_targets(y)

        if isinstance(X, np.ndarray):
            feature_names = [f"x{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=feature_names)
        else:
            feature_names = list(X.columns)

        self.feature_names_in_ = feature_names
        self.classes_ = np.unique(y)
        self.norms_ = resolve_norm_pair(self.norm_conorm)

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

        self.model_ = self._convert_to_gt2(base.model_)

        if self.refine_gt2:
            # Same column convention `IT2TribbleClassifier.fit` relies on:
            # `gt2_firing_strengths`'s columns are `sorted(model.all_output_labels)`,
            # which is exactly `self.classes_`.
            y_idx = np.searchsorted(self.classes_, y)
            self.model_ = refine_gt2_antecedents(
                X, y_idx, self.model_, self.norms_,
                n_alpha_planes=self.n_alpha_planes,
                n_sweeps=self.refine_gt2_n_sweeps,
                l2_shrink=self.refine_gt2_l2_shrink,
                verbose=False,
            )

        return self

    def predict(self, X):
        """Predict class labels using alpha-combined GT2 firing strengths.

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

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        else:
            X = pd.DataFrame(X.values, columns=self.feature_names_in_)

        firing_crisp, labels = gt2_firing_strengths(
            X, self.model_, self.norms_,
            n_alpha_planes=self.n_alpha_planes, km_iterations=self.km_iterations,
        )

        class_indices = np.argmax(firing_crisp, axis=1)
        return self.classes_[class_indices]

    def predict_intervals(self, X):
        """Predict confidence intervals for each class -- the widest
        (alpha=0) footprint of uncertainty, exactly `IT2TribbleClassifier`'s
        own `predict_intervals` bounds, since alpha=0 is precisely today's
        IT2 conversion of the same Type-1 base.

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
            X = pd.DataFrame(X.values, columns=self.feature_names_in_)

        it2_model_alpha0 = extract_alpha_plane_model(self.model_, 0.0)
        firing_upper, firing_lower, _, _ = it2_firing_strengths(
            X, it2_model_alpha0, self.norms_, km_iterations=None
        )

        return firing_upper, firing_lower

    def _convert_to_gt2(self, type1_model) -> GT2GaussianMixtureModel:
        """Convert a Type-1 model to GT2.

        For each Gaussian membership (mu, sigma), creates the same
        upper_mf/lower_mf pair `IT2TribbleClassifier._convert_to_it2` does,
        plus `principal_mf`: the *original*, un-widened Type-1 (mu, sigma) --
        the Type-1 heuristic fit already is this feature/label's single most
        representative membership, so it is carried through unchanged rather
        than re-derived as the interval midpoint.
        """
        feature_models = {}
        min_sigma = 1e-4

        for feature_name, type1_feature_model in type1_model.feature_models.items():
            label_models = {}

            for label, type1_label_model in type1_feature_model.label_models.items():
                gt2_mfs = []

                for gauss_mf in type1_label_model.memberships:
                    base_sigma = max(gauss_mf.sigma, min_sigma)

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
                    principal_mf = GaussianMembership(
                        mu=gauss_mf.mu,
                        sigma=base_sigma,
                        id=gauss_mf.id,
                    )

                    gt2_mf = GT2GaussianMembership(
                        upper_mf=upper_mf,
                        lower_mf=lower_mf,
                        principal_mf=principal_mf,
                    )
                    gt2_mfs.append(gt2_mf)

                label_models[label] = GT2LabelModel(gt2_mfs)

            feature_models[feature_name] = GT2FeatureModel(label_models)

        return GT2GaussianMixtureModel(feature_models)
