"""Issue #253: rank/bin-based feature scorers (hellinger, jensen_shannon, ks, mutual_info).

- Each scorer ranks a class-informative feature above pure noise.
- A binary flag with class-proportion imbalance but no class dependence scores 0, and an
  informative count ranks above it (the case Wasserstein gets wrong on PhiUSIIL).
- ks / hellinger / jensen_shannon are invariant to monotone transforms and to rescaling.
- The KS statistic matches scipy; disjoint supports give the maximal distance (1).
- mutual_info is deterministic under its seed.
- The new names reach TribbleClassifier and TribbleRegressor through scoring_method.
"""

import unittest
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from tribblefis.gauss_math import (
    _hellinger_distance,
    _jensen_shannon_distance_binned,
    _ks_statistic,
    calculate_gaussian_correlation,
    calculate_interaction_scores,
)
from tribblefis.gaussian_classifier import TribbleClassifier
from tribblefis.gaussian_regressor import TribbleRegressor

NEW_METHODS = ["hellinger", "jensen_shannon", "ks", "mutual_info"]
BOUNDED_METHODS = ["hellinger", "jensen_shannon", "ks"]


def _signal_and_noise(seed=0, n=600):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    X = pd.DataFrame({
        "signal": rng.normal(size=n) + 2.0 * y,
        "noise_a": rng.normal(size=n),
        "noise_b": rng.normal(size=n),
    })
    return X, pd.Series(y, name="y")


def _score_map(scores):
    return dict(scores)


class TestRanksInformativeFeatureFirst(unittest.TestCase):
    def test_each_new_method_puts_signal_first(self):
        X, y = _signal_and_noise()
        for method in NEW_METHODS:
            with self.subTest(method=method):
                ranked = calculate_gaussian_correlation(X, y, method=method, top_n=-1,
                                                        correlation_threshold=1.0)
                self.assertEqual(ranked[0][0], "signal")
                self.assertTrue(all(np.isfinite(s) for _, s in ranked))

    def test_multiclass_target_is_finite_and_signal_first(self):
        rng = np.random.default_rng(1)
        y = pd.Series(rng.integers(0, 3, 900))
        X = pd.DataFrame({
            "signal": rng.normal(size=900) + 1.5 * y.to_numpy(),
            "noise": rng.normal(size=900),
        })
        for method in BOUNDED_METHODS:
            with self.subTest(method=method):
                ranked = _score_map(calculate_gaussian_correlation(
                    X, y, method=method, correlation_threshold=1.0))
                self.assertGreater(ranked["signal"], ranked["noise"])


class TestBinaryFlagImbalance(unittest.TestCase):
    """A flag independent of the label must not beat an informative count."""

    def _data(self, seed=2, n=4000):
        rng = np.random.default_rng(seed)
        y = (rng.random(n) < 0.1).astype(int)  # 90/10 class imbalance
        flag = (rng.random(n) < 0.5).astype(float)  # same P(flag=1) in both classes
        count = rng.poisson(3.0 + 6.0 * y).astype(float)  # informative, many-valued
        return pd.DataFrame({"flag": flag, "count": count}), pd.Series(y)

    def test_independent_flag_scores_zero_and_count_wins(self):
        X, y = self._data()
        for method in BOUNDED_METHODS:
            with self.subTest(method=method):
                scores = _score_map(calculate_gaussian_correlation(
                    X, y, method=method, correlation_threshold=1.0))
                if method != "ks":  # KS has no bins: the flag only shows sampling noise
                    self.assertEqual(scores["flag"], 0.0)
                self.assertGreater(scores["count"], 5 * scores["flag"])
        ranked = calculate_gaussian_correlation(X, y, method="hellinger", correlation_threshold=1.0)
        self.assertEqual(ranked[0][0], "count")


class TestInvariance(unittest.TestCase):
    def setUp(self):
        X, y = _signal_and_noise(seed=3)
        self.X, self.y = X, y

    def test_monotone_transform_leaves_rank_scores_unchanged(self):
        base = self.X.copy()
        warped = base.copy()
        warped["signal"] = np.exp(base["signal"] / 2.0)
        warped["noise_a"] = base["noise_a"] ** 3
        for method in BOUNDED_METHODS:
            with self.subTest(method=method):
                a = _score_map(calculate_gaussian_correlation(base, self.y, method=method,
                                                              correlation_threshold=1.0))
                b = _score_map(calculate_gaussian_correlation(warped, self.y, method=method,
                                                              correlation_threshold=1.0))
                for col in base.columns:
                    self.assertAlmostEqual(a[col], b[col], places=12, msg=col)

    def test_rescaling_leaves_scores_unchanged(self):
        scaled = self.X.copy()
        scaled["signal"] = self.X["signal"] * 1e6
        scaled["noise_b"] = self.X["noise_b"] * 1e-4
        for method in BOUNDED_METHODS:
            with self.subTest(method=method):
                a = _score_map(calculate_gaussian_correlation(self.X, self.y, method=method,
                                                              correlation_threshold=1.0))
                b = _score_map(calculate_gaussian_correlation(scaled, self.y, method=method,
                                                              correlation_threshold=1.0))
                for col in self.X.columns:
                    self.assertAlmostEqual(a[col], b[col], places=12, msg=col)


class TestDistanceValues(unittest.TestCase):
    def test_ks_matches_scipy_statistic(self):
        rng = np.random.default_rng(4)
        a = rng.normal(size=300)
        b = rng.normal(0.4, 1.3, size=250)
        self.assertAlmostEqual(_ks_statistic(a, b), ks_2samp(a, b).statistic, places=12)

    def test_identical_samples_have_zero_distance(self):
        a = np.arange(40, dtype=float)
        self.assertEqual(_hellinger_distance(a, a.copy()), 0.0)
        self.assertEqual(_jensen_shannon_distance_binned(a, a.copy()), 0.0)
        self.assertEqual(_ks_statistic(a, a.copy()), 0.0)

    def test_disjoint_supports_give_maximal_distance(self):
        # Four distinct values, 20 bins collapse to 4 edges: bins {0,1} hold a, bin {2} holds b.
        a = np.r_[np.zeros(50), np.ones(50)]
        b = np.r_[np.full(50, 2.0), np.full(50, 3.0)]
        self.assertAlmostEqual(_hellinger_distance(a, b), 1.0, places=12)
        self.assertAlmostEqual(_jensen_shannon_distance_binned(a, b), 1.0, places=12)
        self.assertAlmostEqual(_ks_statistic(a, b), 1.0, places=12)

    def test_distances_are_symmetric_and_bounded(self):
        rng = np.random.default_rng(5)
        a = rng.exponential(size=200)
        b = rng.exponential(1.5, size=180)
        for fn in (_hellinger_distance, _jensen_shannon_distance_binned, _ks_statistic):
            with self.subTest(fn=fn.__name__):
                d_ab, d_ba = fn(a, b), fn(b, a)
                self.assertAlmostEqual(d_ab, d_ba, places=12)
                self.assertTrue(0.0 <= d_ab <= 1.0 + 1e-12)


class TestMutualInfoDeterminism(unittest.TestCase):
    def test_same_seed_same_scores(self):
        X, y = _signal_and_noise(seed=6)
        first = calculate_gaussian_correlation(X, y, method="mutual_info", correlation_threshold=1.0)
        second = calculate_gaussian_correlation(X, y, method="mutual_info", correlation_threshold=1.0)
        self.assertEqual(first, second)

    def test_constant_column_scores_zero(self):
        X, y = _signal_and_noise(seed=7)
        X["const"] = 3.0
        scores = _score_map(calculate_gaussian_correlation(X, y, method="mutual_info",
                                                           correlation_threshold=1.0))
        self.assertEqual(scores["const"], 0.0)


class TestValidationAndRouting(unittest.TestCase):
    def test_mutual_info_rejected_by_product_interaction_engine(self):
        X, y = _signal_and_noise(seed=8)
        ranked = calculate_gaussian_correlation(X, y, method="ks", correlation_threshold=1.0)
        with self.assertRaisesRegex(ValueError, "mutual_info"):
            calculate_interaction_scores(X, y, ranked, method="mutual_info", engine="product")

    def test_rank_methods_accepted_by_product_interaction_engine(self):
        X, y = _signal_and_noise(seed=9)
        ranked = calculate_gaussian_correlation(X, y, method="ks", correlation_threshold=1.0)
        out = calculate_interaction_scores(X, y, ranked, method="hellinger", engine="product",
                                           candidate_pool=["signal", "noise_a"])
        self.assertEqual(len(out), 1)

    def test_unknown_method_still_rejected(self):
        X, y = _signal_and_noise(seed=10)
        with self.assertRaisesRegex(ValueError, "method must be one of"):
            calculate_gaussian_correlation(X, y, method="kolmogorov")


class TestEstimatorIntegration(unittest.TestCase):
    def test_classifier_accepts_each_new_scoring_method(self):
        X, y = _signal_and_noise(seed=11)
        for method in NEW_METHODS:
            with self.subTest(method=method), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf = TribbleClassifier(top_n=1, scoring_method=method,
                                        correlation_threshold=1.0, random_state=0).fit(X, y)
                self.assertEqual(clf.top_features_[0], "signal")
                self.assertEqual(len(clf.predict(X)), len(X))

    def test_regressor_accepts_rank_scoring_method(self):
        rng = np.random.default_rng(12)
        X = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
        target = 2.0 * X["a"].to_numpy() + 0.1 * rng.normal(size=300)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reg = TribbleRegressor(top_n=1, scoring_method="ks", correlation_threshold=1.0,
                                   random_state=0)
            reg.fit(X, target)
        self.assertEqual(len(reg.predict(X)), len(X))


if __name__ == "__main__":
    unittest.main()
