"""Bagged / random-subspace ensemble of Gaussian fuzzy classifiers.

This module implements candidate #1 from the iris_v2 improvement goal:
a random-forest-style ensemble of :class:`TribbleClassifier`
sub-models combined by popularity (hard) or averaged-probability (soft) voting.

Design
------
Each sub-model is trained on a bootstrap resample of the rows (bagging) and,
optionally, a random subset of the columns (the "random subspace" method that
gives random forests their decorrelation).  Because each sub-model re-runs the
fuzzy feature ranking and re-fits its Gaussians on its own view of the data, the
ensemble members disagree on the hard, overlapping samples -- exactly where a
vote can help -- while agreeing on the easy ones.

Interpretability is preserved: every member is itself a full Gaussian fuzzy rule
base, and :attr:`feature_usage_` reports how often each feature was drawn, so the
ensemble can still be read as "these features, fired around these centres, in
this many sub-rules."

References
----------
* Breiman, "Bagging Predictors" (1996) and "Random Forests" (2001).
* Ho, "The Random Subspace Method for Constructing Decision Forests" (1998).
* Kuncheva, *Combining Pattern Classifiers* (2004) -- majority vote and the
  conditions under which an ensemble beats its members (decorrelated,
  better-than-chance base learners).

Caveat
------
Ensembling can only recover variance error, not the irreducible Bayes error.
On iris_v2 the class overlap is the dominant error term, so the ensemble tracks
its members (~0.81-0.83) rather than exceeding the ~0.826 Bayes ceiling.  It is
built here for completeness and to make that trade-off measurable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from .gaussian_classifier import TribbleClassifier
from .calibrated_fuzzy_classifier import CalibratedGaussianFuzzyClassifier


class BaggedFuzzyClassifier(BaseEstimator, ClassifierMixin):
    """A random-forest-style ensemble of Gaussian fuzzy classifiers.

    Args:
        n_estimators: Number of sub-models to build.
        max_samples: Fraction of the training rows drawn (with replacement) for
            each sub-model's bootstrap.
        max_features: Number of features each sub-model may use. ``"sqrt"`` uses
            ``round(sqrt(n_features))`` (random-forest default), ``"all"`` uses
            every feature, an int uses that many, a float in (0, 1] uses that
            fraction.
        voting: ``"soft"`` averages :meth:`predict_proba` across members;
            ``"hard"`` takes the plurality of member label predictions.
        base: ``"calibrated"`` (Bayes-consistent members, recommended) or
            ``"fuzzy"`` (stock product-norm members).
        n_gaussians, top_p: Passed through to every sub-model.
        random_state: Seed controlling bootstraps and feature draws.
    """

    def __init__(
        self,
        n_estimators=25,
        max_samples=0.8,
        max_features="sqrt",
        voting="soft",
        base="calibrated",
        n_gaussians=1,
        top_p=1.0,
        random_state=42,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.voting = voting
        self.base = base
        self.n_gaussians = n_gaussians
        self.top_p = top_p
        self.random_state = random_state

    def _make_member(self, seed):
        # Each member gets a distinct seed so their k-means starts (and thus the
        # members) decorrelate; an identical seed makes members converge to the
        # same fit on overlapping bootstraps, defeating the ensemble.
        if self.base == "calibrated":
            return CalibratedGaussianFuzzyClassifier(
                n_gaussians=self.n_gaussians, top_p=self.top_p, random_state=seed
            )
        return TribbleClassifier(
            n_gaussians=self.n_gaussians, top_p=self.top_p,
            norm_conorm="probability", random_state=seed,
        )

    def _n_features_to_draw(self, n_features: int) -> int:
        mf = self.max_features
        if mf == "sqrt":
            return max(1, int(round(np.sqrt(n_features))))
        if mf == "all" or mf is None:
            return n_features
        if isinstance(mf, float):
            return max(1, int(round(mf * n_features)))
        return max(1, min(int(mf), n_features))

    def fit(self, X, y):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y_series = y if isinstance(y, pd.Series) else pd.Series(np.asarray(y))
        X_df = X_df.reset_index(drop=True)
        y_series = y_series.reset_index(drop=True)

        self.feature_names_in_ = X_df.columns.tolist()
        self.classes_ = np.unique(y_series.values)
        rng = np.random.RandomState(self.random_state)

        n = len(X_df)
        n_draw = self._n_features_to_draw(len(self.feature_names_in_))
        n_boot = max(1, int(round(self.max_samples * n)))

        self.estimators_ = []
        self.estimator_features_ = []
        usage = {f: 0 for f in self.feature_names_in_}

        for i in range(self.n_estimators):
            rows = rng.randint(0, n, size=n_boot)
            feats = list(rng.choice(self.feature_names_in_, size=n_draw, replace=False))
            y_boot = y_series.iloc[rows]
            # A bootstrap that misses a class cannot vote on it; resample once.
            tries = 0
            while y_boot.nunique() < len(self.classes_) and tries < 5:
                rows = rng.randint(0, n, size=n_boot)
                y_boot = y_series.iloc[rows]
                tries += 1
            member_seed = None if self.random_state is None else self.random_state + i
            member = self._make_member(member_seed)
            member.fit(X_df.iloc[rows][feats].reset_index(drop=True),
                       y_boot.reset_index(drop=True))
            self.estimators_.append(member)
            self.estimator_features_.append(feats)
            for f in feats:
                usage[f] += 1

        self.feature_usage_ = usage
        self.is_fitted_ = True
        return self

    def _as_df(self, X):
        return X.reset_index(drop=True) if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_in_)

    def predict_proba(self, X):
        check_is_fitted(self)
        X_df = self._as_df(X)
        agg = np.zeros((len(X_df), len(self.classes_)), dtype=float)
        cls_idx = {c: i for i, c in enumerate(self.classes_)}
        for member, feats in zip(self.estimators_, self.estimator_features_):
            proba = member.predict_proba(X_df[feats])
            for j, c in enumerate(member.classes_):
                agg[:, cls_idx[c]] += proba[:, j]
        return agg / agg.sum(axis=1, keepdims=True)

    def predict(self, X):
        check_is_fitted(self)
        X_df = self._as_df(X)
        if self.voting == "soft":
            return self.classes_[np.argmax(self.predict_proba(X_df), axis=1)]
        # Hard vote: plurality of member label predictions.
        cls_idx = {c: i for i, c in enumerate(self.classes_)}
        votes = np.zeros((len(X_df), len(self.classes_)), dtype=int)
        for member, feats in zip(self.estimators_, self.estimator_features_):
            for i, lab in enumerate(member.predict(X_df[feats])):
                votes[i, cls_idx[lab]] += 1
        return self.classes_[np.argmax(votes, axis=1)]
