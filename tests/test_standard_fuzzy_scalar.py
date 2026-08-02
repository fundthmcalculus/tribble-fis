import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import make_pipeline

from tribblefis.gaussian_classifier import MixtureOfGaussiansFuzzyClassifier
from tribblefis.gaussian_regressor import MixtureOfGaussiansFuzzyRegressor
from tribblefis.scaling import StandardFuzzyScalar


def _wide_range_column(rng, n):
    """Multi-scale content: some rows near 1e-2, some near 1e5 -- a robust
    trigger for the dynamic-range log-transform heuristic regardless of which
    particular samples land near the extremes."""
    low = rng.uniform(0.01, 1.0, n // 2)
    high = rng.uniform(1e4, 1e5, n - n // 2)
    return rng.permutation(np.concatenate([low, high]))


class TestStandardFuzzyScalar(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        n = 200
        self.X = pd.DataFrame(
            {
                "wide": _wide_range_column(rng, n),
                "narrow": rng.uniform(10.0, 20.0, n),
            }
        )
        self.rng = rng

    def test_detects_wide_dynamic_range_feature(self):
        scaler = StandardFuzzyScalar().fit(self.X)
        self.assertEqual(scaler.log_features_, ["wide"])

    def test_no_log_detection_when_disabled(self):
        scaler = StandardFuzzyScalar(log_dynamic_range=None).fit(self.X)
        self.assertEqual(scaler.log_features_, [])

    def test_output_bounded_to_unit_interval(self):
        scaler = StandardFuzzyScalar()
        Xt = scaler.fit_transform(self.X)
        self.assertGreaterEqual(Xt.min(), 0.0)
        self.assertLessEqual(Xt.max(), 1.0)
        np.testing.assert_allclose(Xt.min(axis=0), [0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(Xt.max(axis=0), [1.0, 1.0], atol=1e-10)

    def test_custom_feature_range(self):
        scaler = StandardFuzzyScalar(feature_range=(-1.0, 1.0))
        Xt = scaler.fit_transform(self.X)
        self.assertGreaterEqual(Xt.min(), -1.0)
        self.assertLessEqual(Xt.max(), 1.0)

    def test_inverse_transform_round_trips(self):
        scaler = StandardFuzzyScalar()
        Xt = scaler.fit_transform(self.X)
        Xinv = scaler.inverse_transform(Xt)
        np.testing.assert_allclose(Xinv, self.X.to_numpy(), atol=1e-8)

    def test_clips_out_of_range_values_at_transform_time(self):
        scaler = StandardFuzzyScalar().fit(self.X)
        X_test = self.X.copy()
        X_test.iloc[0, X_test.columns.get_loc("narrow")] = 1e9
        Xt = scaler.transform(X_test)
        self.assertLessEqual(Xt.max(), 1.0)

    def test_ndarray_input(self):
        scaler = StandardFuzzyScalar()
        Xt = scaler.fit_transform(self.X.to_numpy())
        self.assertEqual(list(scaler.feature_names_in_), ["feature_0", "feature_1"])
        self.assertGreaterEqual(Xt.min(), 0.0)
        self.assertLessEqual(Xt.max(), 1.0)

    def test_get_feature_names_out(self):
        scaler = StandardFuzzyScalar().fit(self.X)
        np.testing.assert_array_equal(
            scaler.get_feature_names_out(), np.array(["wide", "narrow"], dtype=object)
        )

    def test_sklearn_clone_and_get_params(self):
        scaler = StandardFuzzyScalar(log_dynamic_range=2.5)
        cloned = clone(scaler)
        self.assertEqual(cloned.get_params(), scaler.get_params())
        self.assertEqual(cloned.get_params()["log_dynamic_range"], 2.5)

    def test_pipeline_with_classifier(self):
        y = (self.X["wide"] > np.median(self.X["wide"])).astype(int)
        pipe = make_pipeline(StandardFuzzyScalar(), MixtureOfGaussiansFuzzyClassifier())
        pipe.fit(self.X, y)
        preds = pipe.predict(self.X)
        self.assertEqual(len(preds), len(y))

    def test_pipeline_with_regressor(self):
        y = self.X["wide"] * 2 + self.X["narrow"]
        pipe = make_pipeline(StandardFuzzyScalar(), MixtureOfGaussiansFuzzyRegressor())
        pipe.fit(self.X, y)
        preds = pipe.predict(self.X)
        self.assertEqual(len(preds), len(y))

    def test_constant_feature_does_not_divide_by_zero(self):
        X = self.X.copy()
        X["constant"] = 5.0
        scaler = StandardFuzzyScalar().fit(X)
        Xt = scaler.transform(X)
        self.assertTrue(np.all(np.isfinite(Xt)))


if __name__ == "__main__":
    unittest.main()
