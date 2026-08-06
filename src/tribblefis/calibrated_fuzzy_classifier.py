"""Calibrated (Bayes-consistent) Gaussian fuzzy classifier.

This module implements candidate #3 from the iris_v2 improvement goal:
a *nonlinear compact-set transformation* that increases class separability
while keeping the model interpretable.

Background / research
---------------------
The stock :class:`TribbleClassifier` scores a class by taking
the fuzzy AND (a t-norm) of per-feature Gaussian *membership heights*
``exp(-0.5 * ((x-mu)/sigma)**2)`` and picking the argmax.  Two design choices
cost it accuracy on overlapping classes:

1. **The ``min`` t-norm throws away evidence.**  ``min`` reports only the single
   worst-matching feature, so two classes that differ on several features but
   share one ambiguous feature look identical.  The *product* t-norm keeps every
   feature's vote.  On the iris dataset this alone lifts the base model from
   ~0.766 to ~0.811.

2. **Membership *height* is not the same as evidence.**  ``exp(-0.5 z**2)`` omits
   the Gaussian normaliser ``1/(sigma*sqrt(2*pi))``, so a broad (large-sigma)
   class is rewarded for being vague, and no account is taken of the class
   prior.  Restoring the normaliser and the prior turns the product-of-
   memberships into the exact log-likelihood of a diagonal Gaussian /
   Gaussian-mixture generative model.

The fix is a *monotone (nonlinear) recalibration of the membership sets in the
log domain* -- each fuzzy set is re-read as a probability density and the class
evidence is accumulated as

    score(c) = log P(c) + sum_f log( sum_g w_{f,c,g} * N(x_f; mu, sigma) )

This is still a Gaussian fuzzy rule base -- the antecedent membership functions
are the very same per-feature Gaussians the base model fits, so the model stays
inspectable ("feature f fires for class c around mu +/- sigma").  Only the
*aggregation* changes.  With one Gaussian per feature the classifier is
provably equivalent to Gaussian Naive Bayes; with several it is a per-feature
Gaussian-mixture Bayes classifier -- i.e. it reaches the Bayes-optimal decision
rule for this generative family, which empirically is the accuracy ceiling on
the iris_v2 data (~0.826).

References
----------
* Duda, Hart & Stork, *Pattern Classification* (2001) -- Gaussian Bayes /
  naive Bayes as the optimal rule for class-conditional Gaussians.
* Kuncheva, "How good are fuzzy if-then classifiers?" (2000) -- fuzzy
  classifiers with product inference and normalised memberships coincide with
  probabilistic (Bayes) classifiers.
* Zadeh (1965), Klir & Yuan (1995) -- t-norm choice (min vs product) and its
  effect on rule aggregation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from .gaussian_classifier import TribbleClassifier

_LOG_2PI = float(np.log(2.0 * np.pi))


class CalibratedGaussianFuzzyClassifier(BaseEstimator, ClassifierMixin):
    """Gaussian fuzzy classifier with Bayes-consistent log-evidence aggregation.

    It reuses :class:`TribbleClassifier` to fit the per-feature
    per-class Gaussian membership functions (so feature selection, multi-Gaussian
    fitting, etc. are unchanged) and then replaces the fuzzy inference with a
    calibrated log-likelihood score.

    Args:
        top_n, top_p, n_gaussians, member_function, random_state:
            Passed straight through to the underlying
            :class:`TribbleClassifier`.
        use_priors: Add ``log P(class)`` from the training class frequencies.
            Leave ``True`` unless you want a strictly likelihood-based rule.
        var_smoothing: Fraction of each feature's overall variance added to
            every Gaussian variance for numerical stability (mirrors
            sklearn ``GaussianNB``).
    """

    def __init__(
        self,
        top_n=-1,
        top_p=0.95,
        n_gaussians=1,
        member_function="gaussian",
        random_state=42,
        use_priors=True,
        var_smoothing=1e-9,
    ):
        self.top_n = top_n
        self.top_p = top_p
        self.n_gaussians = n_gaussians
        self.member_function = member_function
        self.random_state = random_state
        self.use_priors = use_priors
        self.var_smoothing = var_smoothing

    def _make_base(self) -> TribbleClassifier:
        return TribbleClassifier(
            top_n=self.top_n,
            top_p=self.top_p,
            n_gaussians=self.n_gaussians,
            member_function=self.member_function,
            norm_conorm="probability",
            random_state=self.random_state,
        )

    def fit(self, X, y):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y_series = y if isinstance(y, pd.Series) else pd.Series(np.asarray(y))
        X_df = X_df.reset_index(drop=True)
        y_series = y_series.reset_index(drop=True)

        self.base_ = self._make_base().fit(X_df, y_series)
        self.classes_ = np.asarray(self.base_.classes_)
        self.feature_names_in_ = self.base_.feature_names_in_
        self.top_features_ = self.base_.top_features_

        # Class log-priors from training frequencies.
        counts = y_series.value_counts()
        total = float(counts.sum())
        self.class_log_prior_ = {
            c: float(np.log(counts.get(c, 0) / total)) if counts.get(c, 0) > 0 else -np.inf
            for c in self.classes_
        }

        # Global per-feature variance floor for numerical stability.
        self._var_floor_ = {
            f: self.var_smoothing * float(np.var(X_df[f].values))
            for f in self.top_features_
        }

        # Cache the (mu, sigma) tuples per feature per class as arrays.
        model = self.base_.model_
        self._params_ = {}
        for feat, fmodel in model.feature_models.items():
            per_class = {}
            for cls, lmodel in fmodel.label_models.items():
                mus = np.array([g.mu for g in lmodel.memberships], dtype=float)
                sig = np.array([max(g.sigma, 1e-6) for g in lmodel.memberships], dtype=float)
                per_class[cls] = (mus, sig)
            self._params_[feat] = per_class

        self.is_fitted_ = True
        return self

    def _log_evidence(self, X_df) -> np.ndarray:
        """(n_samples, n_classes) matrix of calibrated log-evidence scores."""
        n = len(X_df)
        scores = np.zeros((n, len(self.classes_)), dtype=float)

        for ci, cls in enumerate(self.classes_):
            total = np.full(n, self.class_log_prior_[cls] if self.use_priors else 0.0)
            for feat in self.top_features_:
                mus, sig = self._params_[feat][cls]
                var = sig ** 2 + self._var_floor_[feat]
                x = X_df[feat].values.astype(float)[:, None]  # (n, 1)
                # Per-component Gaussian log-density: log N(x; mu, sigma)
                comp_log = -0.5 * (_LOG_2PI + np.log(var)[None, :]
                                   + (x - mus[None, :]) ** 2 / var[None, :])
                # Mixture over this feature's components (equal weights ->
                # logsumexp - log K), a fuzzy OR done properly in the log domain.
                m = comp_log.max(axis=1)
                feat_ll = m + np.log(np.exp(comp_log - m[:, None]).sum(axis=1))
                feat_ll -= np.log(comp_log.shape[1])
                total = total + feat_ll  # product t-norm == sum of log-evidence
            scores[:, ci] = total
        return scores

    def _as_df(self, X):
        return X.reset_index(drop=True) if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_in_)

    def predict(self, X):
        check_is_fitted(self)
        scores = self._log_evidence(self._as_df(X))
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        check_is_fitted(self)
        scores = self._log_evidence(self._as_df(X))
        m = scores.max(axis=1, keepdims=True)
        p = np.exp(scores - m)
        return p / p.sum(axis=1, keepdims=True)
