"""max_samples is exposed on the public estimators, not just on gauss_math's
internal helpers.

Before this, create_gaussian_membership_dict/create_trapz_membership_dict
accepted a row cap, but MixtureOfGaussiansFuzzyClassifier and
MixtureOfGaussiansFuzzyRegressor had no way to pass one in -- the cap used to
be silently hardcoded at 20,000 rows and there was no way to restore or
change it through the public API.
"""

import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone

from tribblefis.gaussian_classifier import MixtureOfGaussiansFuzzyClassifier
from tribblefis.gaussian_regressor import (
    MixtureOfGaussiansFuzzyRegressor,
    MimoGaussianPredictor,
)


def _classification_frame(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "a": np.concatenate([rng.normal(0, 1, n // 2), rng.normal(5, 1, n // 2)]),
            "b": rng.normal(0, 1, n),
        }
    )
    y = pd.Series(["lo"] * (n // 2) + ["hi"] * (n // 2))
    return X, y


class TestClassifierMaxSamples(unittest.TestCase):
    def test_is_a_constructor_parameter(self):
        clf = MixtureOfGaussiansFuzzyClassifier(max_samples=123)
        self.assertEqual(clf.get_params()["max_samples"], 123)

    def test_default_is_none(self):
        clf = MixtureOfGaussiansFuzzyClassifier()
        self.assertIsNone(clf.max_samples)

    def test_clone_round_trips(self):
        clf = MixtureOfGaussiansFuzzyClassifier(max_samples=50)
        cloned = clone(clf)
        self.assertEqual(cloned.max_samples, 50)

    def test_fit_and_predict_with_a_cap(self):
        X, y = _classification_frame(300)
        clf = MixtureOfGaussiansFuzzyClassifier(n_gaussians=2, max_samples=50, random_state=0)
        clf.fit(X, y)
        preds = clf.predict(X)
        self.assertEqual(len(preds), len(y))

    def test_capped_and_uncapped_fits_both_succeed_and_can_differ(self):
        X, y = _classification_frame(300)
        capped = MixtureOfGaussiansFuzzyClassifier(n_gaussians=2, max_samples=20, random_state=0)
        uncapped = MixtureOfGaussiansFuzzyClassifier(n_gaussians=2, max_samples=None, random_state=0)
        capped.fit(X, y)
        uncapped.fit(X, y)
        # Both must produce a usable model; whether the memberships differ is
        # a property of the data, not asserted here.
        self.assertEqual(len(capped.predict(X)), len(y))
        self.assertEqual(len(uncapped.predict(X)), len(y))


class TestRegressorMaxSamples(unittest.TestCase):
    def _regression_frame(self, n=300, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.uniform(-3, 3, n)
        X = pd.DataFrame({"x": x})
        y = pd.Series(np.sin(x) + rng.normal(0, 0.05, n))
        return X, y

    def test_is_a_constructor_parameter(self):
        reg = MixtureOfGaussiansFuzzyRegressor(max_samples=77)
        self.assertEqual(reg.get_params()["max_samples"], 77)

    def test_clone_round_trips(self):
        reg = MixtureOfGaussiansFuzzyRegressor(max_samples=30)
        self.assertEqual(clone(reg).max_samples, 30)

    def test_fit_and_predict_with_a_cap(self):
        X, y = self._regression_frame(300)
        reg = MixtureOfGaussiansFuzzyRegressor(n_gaussians=2, max_samples=40, random_state=0)
        reg.fit(X, y)
        preds = reg.predict(X)
        self.assertEqual(len(preds), len(y))

    def test_mimo_predictor_forwards_max_samples(self):
        mimo = MimoGaussianPredictor(n_gaussians=2, max_samples=40, random_state=0)
        regressor = mimo._make_regressor()
        self.assertEqual(regressor.max_samples, 40)


if __name__ == "__main__":
    unittest.main()
