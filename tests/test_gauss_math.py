"""Tests for the k-means-based Gaussian mixture selection in gauss_math.py.

Covers the code introduced to replace the old EM-fit-then-discard selector:
_hard_partition_gaussians, _mixture_bic, fit_gaussian_mixture_1d's BIC search,
its distinct-value cap (both the automatic and the explicit n_gaussians>0
paths), the variance floor, and fit_gaussians' max_samples subsampling.
"""

import unittest

import numpy as np
import pandas as pd

from tribblefis.gauss_math import (
    BIC_VARIANCE_FLOOR_FRAC,
    _hard_partition_gaussians,
    _mixture_bic,
    fit_gaussian_mixture_1d,
    fit_gaussians,
    find_optimal_gaussians,
)


class TestHardPartitionGaussians(unittest.TestCase):
    def test_single_cluster_matches_mean_and_std(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        labels = np.zeros(len(data), dtype=int)
        out = _hard_partition_gaussians(data, labels, 1)
        self.assertEqual(len(out), 1)
        mu, sd, weight = out[0]
        self.assertAlmostEqual(mu, data.mean())
        self.assertAlmostEqual(sd, data.std())
        self.assertAlmostEqual(weight, 1.0)

    def test_two_clusters_separate_correctly(self):
        data = np.array([0.0, 0.1, -0.1, 10.0, 10.1, 9.9])
        labels = np.array([0, 0, 0, 1, 1, 1])
        out = _hard_partition_gaussians(data, labels, 2)
        self.assertEqual(len(out), 2)
        means = sorted(c[0] for c in out)
        self.assertAlmostEqual(means[0], 0.0, places=1)
        self.assertAlmostEqual(means[1], 10.0, places=1)
        self.assertAlmostEqual(sum(c[2] for c in out), 1.0)

    def test_empty_cluster_is_dropped_not_padded(self):
        data = np.array([1.0, 2.0, 3.0])
        labels = np.array([0, 0, 0])
        # Cluster index 1 has no members; it must not appear as a degenerate
        # (nan, nan) entry.
        out = _hard_partition_gaussians(data, labels, 2)
        self.assertEqual(len(out), 1)


class TestMixtureBic(unittest.TestCase):
    def test_empty_components_is_infinite(self):
        self.assertEqual(_mixture_bic(np.array([1.0, 2.0]), [], 1e-6), np.inf)

    def test_better_fit_scores_lower_bic(self):
        rng = np.random.default_rng(0)
        data = np.concatenate([rng.normal(0, 0.5, 200), rng.normal(10, 0.5, 200)])
        one = [(float(data.mean()), float(data.std()), 1.0)]
        two = _hard_partition_gaussians(
            data, (data > 5).astype(int), 2
        )
        var_floor = BIC_VARIANCE_FLOOR_FRAC * data.var()
        bic_one = _mixture_bic(data, one, var_floor)
        bic_two = _mixture_bic(data, two, var_floor)
        self.assertLess(bic_two, bic_one)


class TestFitGaussianMixture1d(unittest.TestCase):
    def test_automatic_selection_recovers_two_well_separated_clusters(self):
        rng = np.random.default_rng(1)
        data = np.concatenate([rng.normal(0, 0.3, 100), rng.normal(20, 0.3, 100)])
        memberships, n_selected = fit_gaussian_mixture_1d(data, n_gaussians=0, max_gaussians=4)
        self.assertEqual(n_selected, 2)
        self.assertEqual(len(memberships), 2)

    def test_automatic_selection_capped_by_distinct_values(self):
        # Only two distinct values in the whole column: max_gaussians=4 must
        # not push k past what the data can support.
        data = np.array([1.0] * 20 + [2.0] * 20)
        memberships, n_selected = fit_gaussian_mixture_1d(data, n_gaussians=0, max_gaussians=4)
        self.assertLessEqual(n_selected, 2)

    def test_explicit_n_gaussians_capped_by_distinct_values(self):
        # Regression test: asking for more explicit components than the data
        # has distinct values must not be handed to KMeans uncapped -- that
        # used to raise a ConvergenceWarning and throw the fit away.
        data = np.array([1.0] * 20 + [2.0] * 20)
        with np.errstate(all="raise"):
            memberships, n_selected = fit_gaussian_mixture_1d(data, n_gaussians=5)
        self.assertLessEqual(n_selected, 2)
        self.assertEqual(len(memberships), n_selected)

    def test_explicit_n_gaussians_below_distinct_values_is_unaffected(self):
        rng = np.random.default_rng(2)
        data = np.concatenate([rng.normal(0, 0.3, 50), rng.normal(20, 0.3, 50)])
        memberships, n_selected = fit_gaussian_mixture_1d(data, n_gaussians=2)
        self.assertEqual(n_selected, 2)

    def test_single_point_no_crash(self):
        memberships, n_selected = fit_gaussian_mixture_1d(np.array([5.0]))
        self.assertEqual(n_selected, 1)
        self.assertAlmostEqual(memberships[0].mu, 5.0)

    def test_empty_data_returns_empty(self):
        memberships, n_selected = fit_gaussian_mixture_1d(np.array([]))
        self.assertEqual(memberships, [])
        self.assertEqual(n_selected, 0)

    def test_degenerate_singleton_cluster_does_not_diverge(self):
        # A cluster that lands on a single repeated value has zero raw
        # variance; the variance floor must keep its BIC contribution finite
        # so it cannot win the selection purely by infinite likelihood.
        data = np.array([3.0] * 5 + list(np.linspace(0, 1, 20)))
        memberships, n_selected = fit_gaussian_mixture_1d(data, n_gaussians=0, max_gaussians=4)
        self.assertTrue(all(np.isfinite(m.sigma) for m in memberships))
        self.assertGreater(n_selected, 0)


class TestFindOptimalGaussians(unittest.TestCase):
    def test_matches_fit_gaussian_mixture_1d_selection(self):
        rng = np.random.default_rng(3)
        data = np.concatenate([rng.normal(0, 0.3, 80), rng.normal(15, 0.3, 80)])
        n = find_optimal_gaussians(data, max_gaussians=4)
        _, n_selected = fit_gaussian_mixture_1d(data, n_gaussians=0, max_gaussians=4)
        self.assertEqual(n, n_selected)

    def test_single_row_returns_one(self):
        self.assertEqual(find_optimal_gaussians(np.array([1.0])), 1)


class TestFitGaussiansMaxSamples(unittest.TestCase):
    def _frame(self, n=500):
        rng = np.random.default_rng(4)
        X = pd.DataFrame({"f": rng.normal(0, 1, n)})
        y = pd.Series(["a"] * n)
        return X, y

    def test_max_samples_none_uses_all_rows(self):
        X, y = self._frame(50)
        # No cap: every call with the same seed must be identical since there
        # is nothing to subsample.
        g1 = fit_gaussians(X, y, "f", "a", n_gaussians=1, max_samples=None, random_state=0)
        g2 = fit_gaussians(X, y, "f", "a", n_gaussians=1, max_samples=None, random_state=0)
        self.assertEqual(g1[0].mu, g2[0].mu)

    def test_max_samples_caps_without_error(self):
        X, y = self._frame(500)
        gaussians = fit_gaussians(X, y, "f", "a", n_gaussians=1, max_samples=10, random_state=0)
        self.assertEqual(len(gaussians), 1)

    def test_max_samples_is_deterministic_given_random_state(self):
        X, y = self._frame(500)
        g1 = fit_gaussians(X, y, "f", "a", n_gaussians=1, max_samples=10, random_state=7)
        g2 = fit_gaussians(X, y, "f", "a", n_gaussians=1, max_samples=10, random_state=7)
        self.assertEqual(g1[0].mu, g2[0].mu)
        self.assertEqual(g1[0].sigma, g2[0].sigma)

    def test_max_samples_larger_than_data_is_a_no_op(self):
        X, y = self._frame(20)
        g1 = fit_gaussians(X, y, "f", "a", n_gaussians=1, max_samples=1000, random_state=0)
        g2 = fit_gaussians(X, y, "f", "a", n_gaussians=1, max_samples=None, random_state=0)
        self.assertEqual(g1[0].mu, g2[0].mu)


class TestBICSelectionTieStability(unittest.TestCase):
    """A larger k that ties or sub-ULP-beats the best BIC must not be selected.

    Degenerate higher-k fits routinely tie the best BIC (a redundant component
    lands on a duplicate value / empty cluster and collapses back). A bare
    `bic < best_bic` argmin is then decided by sub-ULP noise, which flips the
    selected integer k across CPUs/BLAS/numba builds. See the cross-platform
    reproducibility investigation. `_bic_improves` resolves such ties to the
    smaller, parsimonious k deterministically.
    """

    def test_exact_tie_rejected(self):
        from tribblefis.gauss_math import _bic_improves

        self.assertFalse(_bic_improves(5.0, 5.0))
        self.assertFalse(_bic_improves(-2000.44, -2000.44))

    def test_sub_margin_improvement_rejected(self):
        # This is the case a plain `bic < best_bic` would WRONGLY accept, flipping
        # k on nothing but float noise. best=5.0, candidate 5e-7 lower (< 1e-6 rel).
        from tribblefis.gauss_math import _bic_improves

        self.assertLess(5.0 - 4.9999995, 5.0 * 1e-6)   # confirm it is within margin
        self.assertFalse(_bic_improves(4.9999995, 5.0))

    def test_clear_improvement_accepted(self):
        from tribblefis.gauss_math import _bic_improves

        self.assertTrue(_bic_improves(2.0, 5.0))       # genuine gain
        self.assertTrue(_bic_improves(5.0, np.inf))    # first candidate always wins

    def test_negative_best_bic_margin_has_right_sign(self):
        # With a negative best BIC the margin must still make acceptance HARDER,
        # not easier (abs() guards the sign).
        from tribblefis.gauss_math import _bic_improves

        self.assertFalse(_bic_improves(-100.0, -100.0))
        self.assertFalse(_bic_improves(-100.00001, -100.0))  # within 1e-6 rel margin
        self.assertTrue(_bic_improves(-101.0, -100.0))       # clear improvement

    def test_degenerate_larger_k_not_selected_end_to_end(self):
        # 3 well-separated clusters + a near-duplicate point: n_distinct >= 4 so k=4
        # is tried, but the data supports only 3 clusters, so the k=4 candidate is
        # degenerate and ties k=3. Selection must return 3.
        data = np.array([0.0] * 30 + [100.0] * 30 + [200.0] * 30 + [200.0000001])
        _, n_selected = fit_gaussian_mixture_1d(data, n_gaussians=0, max_gaussians=4)
        self.assertEqual(n_selected, 3)


if __name__ == "__main__":
    unittest.main()
