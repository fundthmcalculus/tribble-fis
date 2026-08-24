"""Regression tests locking in fixes to the shared log-transform handling
(``LogTransformMixin`` in ``gauss_math.py``).

Before the fix, ``MixtureOfGaussiansFuzzyClassifier`` accepted a ``log_transform``
constructor argument but never stored it as ``self.log_transform`` and never
called ``_apply_log_transform`` from ``fit``/``predict``/``predict_proba``:
``log_transform=True`` was a silent no-op, and the missing attribute broke
``get_params()``/``clone()`` (which in turn made
``MixtureOfGaussiansFuzzySequenceClassifier``'s cross-validated confusion
estimate silently fall back to in-sample predictions -- see
``test_oof_predictions_warns_and_falls_back_on_cv_failure`` below).
"""

import warnings
import unittest

import numpy as np
import pandas as pd

from tribblefis.gaussian_classifier import (
    MixtureOfGaussiansFuzzyClassifier,
    MixtureOfGaussiansFuzzySequenceClassifier,
)
from tribblefis.gaussian_regressor import MixtureOfGaussiansFuzzyRegressor
from sklearn.base import clone


def _wide_dynamic_range_data(n=200, seed=0, n_classes=2):
    """A feature with a wide dynamic range (log-transform candidate) plus a
    plain feature, and a label correlated with the wide-range feature."""
    rng = np.random.default_rng(seed)
    # log-uniform so the dynamic range (log10(max/min)) reliably clears the
    # detection threshold (3.0) regardless of sample count; a plain
    # rng.uniform(1, 10000, n) can fail to trigger since the sample min drifts
    # up as n grows.
    wide = 10.0 ** rng.uniform(0.0, 6.0, n)
    plain = rng.normal(0.0, 1.0, n)
    X = pd.DataFrame({"wide": wide, "plain": plain})
    if n_classes:
        y = (wide > np.median(wide)).astype(int)
        return X, y
    y = np.log1p(wide) + plain * 0.1
    return X, y


class TestClassifierLogTransform(unittest.TestCase):

    def test_log_transform_param_survives_get_params_and_clone(self):
        """The missing `self.log_transform = log_transform` assignment made
        get_params()/clone() raise AttributeError for this estimator."""
        clf = MixtureOfGaussiansFuzzyClassifier(log_transform=True, top_p=1.0, n_gaussians=1)
        params = clf.get_params()
        self.assertIn("log_transform", params)
        self.assertTrue(params["log_transform"])

        cloned = clone(clf)
        self.assertTrue(cloned.log_transform)

    def test_log_transform_true_actually_applies_and_is_reused_at_predict(self):
        X, y = _wide_dynamic_range_data()
        clf = MixtureOfGaussiansFuzzyClassifier(log_transform=True, top_p=1.0, n_gaussians=1)
        clf.fit(X, y)

        # Fit-time detection must have found and recorded the wide-range feature.
        self.assertIn("wide", clf.log_transformed_features_)

        # predict() must reuse the fit-time offset (already_fitted path) rather
        # than re-detecting on new data; this only works if `_apply_log_transform`
        # is actually being called from predict().
        preds = clf.predict(X)
        self.assertEqual(len(preds), len(X))
        self.assertTrue(np.all(np.isfinite(preds.astype(float))))

    def test_log_transform_false_is_still_a_true_no_op(self):
        X, y = _wide_dynamic_range_data()
        clf = MixtureOfGaussiansFuzzyClassifier(log_transform=False, top_p=1.0, n_gaussians=1)
        clf.fit(X, y)
        self.assertEqual(clf.log_transformed_features_, {})


class TestRegressorLogTransform(unittest.TestCase):

    def test_log_transform_true_actually_applies_and_is_reused_at_predict(self):
        X, y = _wide_dynamic_range_data(n_classes=0)
        reg = MixtureOfGaussiansFuzzyRegressor(log_transform=True, n_gaussians=1, tsk_order="0th")
        reg.fit(X, y)

        self.assertIn("wide", reg.log_transformed_features_)

        preds = reg.predict(X)
        self.assertEqual(len(preds), len(X))
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_log_transform_param_survives_get_params_and_clone(self):
        reg = MixtureOfGaussiansFuzzyRegressor(log_transform=True, n_gaussians=1)
        self.assertTrue(reg.get_params()["log_transform"])
        self.assertTrue(clone(reg).log_transform)


class _FailingClassifier(MixtureOfGaussiansFuzzyClassifier):
    """Stand-in for a classifier whose `fit` fails inside cross_val_predict --
    exactly what happened before the fix, since `clone()` itself raised on the
    missing `log_transform` attribute."""

    def fit(self, X, y):
        raise RuntimeError("simulated cross-validation failure")


class TestSequenceClassifierOofFallbackWarns(unittest.TestCase):
    """Before the fix, `_oof_predictions` swallowed any cross-validation
    failure with a bare `except Exception: pass`, silently returning in-sample
    (overfit) predictions for confusion estimation with no signal anything had
    gone wrong."""

    def _blobs(self, n=90, seed=0):
        rng = np.random.default_rng(seed)
        n0 = n // 2
        x0 = rng.normal([0.0, 0.0], [1.0, 1.0], size=(n0, 2))
        x1 = rng.normal([3.0, 0.0], [1.0, 1.0], size=(n - n0, 2))
        X = pd.DataFrame(np.vstack([x0, x1]), columns=["a", "b"])
        y = pd.Series([0] * n0 + [1] * (n - n0))
        return X, y

    def test_oof_predictions_warns_and_falls_back_on_cv_failure(self):
        X, y = self._blobs()
        seq = MixtureOfGaussiansFuzzySequenceClassifier(top_p=1.0, n_gaussians=1, cv=3)
        base = MixtureOfGaussiansFuzzyClassifier(top_p=1.0, n_gaussians=1).fit(X, y)
        seq.layers_ = [base]
        seq._make_layer = lambda: _FailingClassifier(top_p=1.0, n_gaussians=1)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            oof = seq._oof_predictions(X, y)

        self.assertTrue(any(issubclass(w.category, RuntimeWarning) for w in caught))
        np.testing.assert_array_equal(oof, np.asarray(base.predict(X), dtype=object))


if __name__ == "__main__":
    unittest.main()
