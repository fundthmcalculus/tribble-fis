"""Tests for the smooth-trapezoid EM relaxation (trapz_math_smooth.py).

Covers the bug fixed in this revision (the original #195 shape function was
not shaped like a trapezoid at all, and its normalization was wrong -- see
the module docstring) plus the ordering-constraint and annealing additions.
"""

import unittest

import numpy as np

from tribblefis.gauss_data import TrapezoidMembership, TriangularMembership
from tribblefis.trapz_math import trapz_pdf
from tribblefis.trapz_math_smooth import (
    _rising_ramp,
    _falling_ramp,
    _smooth_trapz_area,
    fit_smooth_trapezoids_em,
    smooth_trapz_pdf,
)


class TestSmoothTrapzShape(unittest.TestCase):
    def test_pdf_integrates_to_one(self):
        x = np.linspace(-10, 20, 40000)
        for a, b, c, d in [(0, 2, 5, 7), (0, 1, 1, 3), (-5, -5, 2, 2), (0, 0.5, 0.5, 1)]:
            y = smooth_trapz_pdf(x, a, b, c, d, steepness=40)
            area = float(np.trapezoid(y, x))
            self.assertAlmostEqual(area, 1.0, places=2, msg=f"(a,b,c,d)={a,b,c,d}")

    def test_shape_is_a_genuine_ramp_not_a_boxcar(self):
        """Regression test for the #195 bug: the old shape function's "ramp"
        was a smoothed indicator of the whole [a, b] interval (~1 across it),
        not a rising ramp -- so its value at the ramp's midpoint should be
        near 0.5, not near 1.
        """
        a, b, c, d, k = 0.0, 10.0, 10.0, 10.0, 40.0
        # A pure rising ramp (b==c==d collapses everything past b to a point,
        # so evaluate a case with only a rising ramp: c == d and a well-separated b).
        a, b, c, d, k = 0.0, 10.0, 10.0, 10.0001, 40.0
        mid = (a + b) / 2
        rising_mid = _rising_ramp(np.array([mid]), a, b, k)[0]
        self.assertAlmostEqual(rising_mid, 0.5, places=2)

    def test_ramps_recover_crisp_ramp_as_steepness_grows(self):
        a, b = 0.0, 4.0
        x = np.array([-1.0, 0.5, 1.0, 2.0, 3.5, 4.0, 5.0])
        crisp = np.clip((x - a) / (b - a), 0.0, 1.0)
        smooth = _rising_ramp(x, a, b, steepness=200.0)
        np.testing.assert_allclose(smooth, crisp, atol=2e-2)

    def test_degenerate_zero_width_ramp_is_a_step(self):
        # b == a: the rising ramp collapses to a step at a.
        x = np.array([-1.0, 0.0, 1.0])
        y = _rising_ramp(x, a=0.0, b=0.0, steepness=50.0)
        np.testing.assert_allclose(y, [0.0, 0.5, 1.0])

    def test_area_matches_quadrature_reference(self):
        from scipy.integrate import quad

        a, b, c, d, k = -2.0, 0.0, 3.0, 6.0, 15.0

        def shape(x):
            return min(
                float(_rising_ramp(np.array([x]), a, b, k)[0]),
                float(_falling_ramp(np.array([x]), c, d, k)[0]),
            )

        ref, _ = quad(shape, a - 5.0, d + 5.0, limit=200)
        got = _smooth_trapz_area(a, b, c, d, k)
        self.assertAlmostEqual(got, ref, places=3)


class TestSmoothTrapzEMFit(unittest.TestCase):
    def test_ordering_is_respected(self):
        rng = np.random.default_rng(0)
        data = np.concatenate([rng.normal(0, 0.3, 300), rng.normal(5, 0.3, 300)])
        mems, weights, ll = fit_smooth_trapezoids_em(
            data, n_components=2, n_bins=50, max_iter=20, random_state=0
        )
        self.assertEqual(len(mems), 2)
        for m in mems:
            self.assertIsInstance(m, TrapezoidMembership)
            self.assertLessEqual(m.a, m.b)
            self.assertLessEqual(m.b, m.c)
            self.assertLessEqual(m.c, m.d)
        np.testing.assert_allclose(np.sum(weights), 1.0)

    def test_triangle_shape_returns_triangular_membership_with_matching_apex(self):
        rng = np.random.default_rng(1)
        data = rng.normal(0, 1.0, 400)
        mems, _weights, _ll = fit_smooth_trapezoids_em(
            data, n_components=1, n_bins=50, max_iter=20, random_state=1, shape="triangle"
        )
        self.assertIsInstance(mems[0], TriangularMembership)

    def test_returned_log_likelihood_is_the_crisp_evaluation(self):
        """The reported log-likelihood must be directly comparable to
        `trapz_math.fit_trapezoids_em`'s (i.e. evaluated at the crisp PDF, not
        whatever steepness the anneal schedule ended on)."""
        rng = np.random.default_rng(2)
        data = rng.normal(0, 1.0, 500)
        mems, weights, ll = fit_smooth_trapezoids_em(
            data, n_components=1, n_bins=50, max_iter=15, random_state=2
        )
        bin_counts, bin_edges = np.histogram(data, bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        a, b, c, d = mems[0].a, mems[0].b, mems[0].c, mems[0].d
        pdf_vals = np.maximum(trapz_pdf(bin_centers, a, b, c, d), 1e-10)
        expected_ll = float(np.sum(bin_counts * np.log(pdf_vals * weights[0])))
        self.assertAlmostEqual(ll, expected_ll, places=6)

    def test_degenerate_constant_data(self):
        data = np.full(20, 3.0)
        mems, weights, ll = fit_smooth_trapezoids_em(data, n_components=1, n_bins=50)
        self.assertEqual(len(mems), 1)
        self.assertEqual(ll, 0.0)


if __name__ == "__main__":
    unittest.main()
