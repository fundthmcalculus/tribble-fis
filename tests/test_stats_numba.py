"""Tests for the numpy/numba statistics helpers in stats_numba.py.

`wasserstein_distance` is covered against **analytic** values rather than against
scipy, deliberately: dropping the scipy dependency is why these helpers exist, so
a test that reimports it to check them would defeat the point and would go green
if both were wrong the same way.

The scale test is the regression. The original implementation returned

    sum(|F_u - F_v|) / len(all_quantiles)

-- the *mean* CDF gap over the support, with no ``dx`` weighting. That is
dimensionless, bounded in [0, 1], and completely invariant to the scale of the
data: multiplying both samples by 1000 left it unchanged. W1 has the units of x
and must scale with it.

Two call-site assumptions in ``gauss_math._pairwise_label_distance`` depend on
that, and both were being violated:

* it divides by the pooled standard deviation "for scale invariance", which is
  only meaningful if the raw distance is scale-*dependent*;
* it squashes "the unbounded pooled-std-normalized wasserstein distance" through
  ``w / (1 + w)``, which is only meaningful if the input is unbounded.
"""

import unittest

import numpy as np

from tribblefis.stats_numba import wasserstein_distance


class TestWassersteinDistance(unittest.TestCase):
    """W1(u, v) = integral |F_u(x) - F_v(x)| dx."""

    def test_two_point_samples_match_analytic(self):
        # Equal-size samples: W1 is the mean absolute difference of the sorted
        # values. (|0-0| + |1-2|) / 2 = 0.5
        self.assertAlmostEqual(wasserstein_distance([0.0, 1.0], [0.0, 2.0]), 0.5)

    def test_translation_gives_the_shift(self):
        # Shifting one sample by c moves every quantile by c, so W1 == c.
        self.assertAlmostEqual(wasserstein_distance([0.0, 1.0, 2.0], [3.0, 4.0, 5.0]), 3.0)

    def test_unequal_sample_sizes(self):
        # u = [0, 1] (n=2), v = [0, 2, 4] (n=3). Support 0 < 1 < 2 < 4, widths
        # [1, 1, 2], |F_u - F_v| = [|1/2 - 1/3|, |1 - 1/3|, |1 - 2/3|].
        # W1 = 1/6 + 2/3 + 2/3 = 1.5
        self.assertAlmostEqual(wasserstein_distance([0.0, 1.0], [0.0, 2.0, 4.0]), 1.5)

    def test_identical_samples_is_zero(self):
        self.assertAlmostEqual(wasserstein_distance([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 0.0)

    def test_symmetric(self):
        u = [0.0, 1.0, 5.0, 9.0]
        v = [2.0, 3.0, 3.5]
        self.assertAlmostEqual(wasserstein_distance(u, v), wasserstein_distance(v, u))

    def test_scales_with_the_data(self):
        """The regression: W1 has the units of x, so scaling the data scales it.

        The previous implementation returned the same number for every k here.
        """
        rng = np.random.default_rng(0)
        u = rng.normal(0.0, 1.0, 400)
        v = rng.normal(1.5, 2.0, 400)
        base = wasserstein_distance(u, v)
        self.assertGreater(base, 0.0)
        for k in (10.0, 100.0, 1000.0):
            scaled = wasserstein_distance(u * k, v * k)
            self.assertAlmostEqual(scaled / base, k, delta=k * 1e-9)

    def test_not_bounded_above_by_one(self):
        """A distance in the data's units has no upper bound.

        The previous implementation was a mean of CDF gaps and so could never
        exceed 1, which silently capped the term the composite score blends.
        """
        self.assertAlmostEqual(wasserstein_distance([0.0], [1000.0]), 1000.0)

    def test_empty_input_returns_zero(self):
        # Pre-existing contract, kept: an empty sample is not an error here.
        self.assertEqual(wasserstein_distance([], [1.0, 2.0]), 0.0)
        self.assertEqual(wasserstein_distance([1.0, 2.0], []), 0.0)

    def test_single_shared_value(self):
        # One distinct support point: no interval to integrate over.
        self.assertAlmostEqual(wasserstein_distance([2.0, 2.0], [2.0]), 0.0)

    def test_accepts_lists_and_nested_arrays(self):
        flat = wasserstein_distance([0.0, 1.0], [0.0, 2.0])
        nested = wasserstein_distance(np.array([[0.0], [1.0]]), np.array([[0.0], [2.0]]))
        self.assertAlmostEqual(flat, nested)


if __name__ == "__main__":
    unittest.main()
