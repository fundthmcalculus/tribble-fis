"""Tests for tribblefis.triangle_fit (Gaussian -> triangle fitting, issue #92).

The two half-width constants are pinned to values derived offline by
minimizing, respectively, the L1 (MAE) and L2 (MSE) error between a
symmetric unit-peak triangle and a standard Gaussian (both in units of
sigma) via golden-section search over Simpson-quadrature integrals -- see
the module docstring in triangle_fit.py for the full derivation. They are
hardcoded here (rather than re-run with a live optimizer) so this test stays
fast and deterministic; if these ever need to change, the derivation should
be redone and this test updated deliberately, not silently.
"""

import unittest

import numpy as np

from tribblefis.gauss_data import GaussianMembership, GaussianMixtureModel, FeatureModel, LabelModel, TriangularMembership
from tribblefis.triangle_fit import (
    GAUSSIAN_TRIANGLE_MAE_HALF_WIDTH,
    GAUSSIAN_TRIANGLE_MSE_HALF_WIDTH,
    fit_triangle_to_gaussian,
    fit_triangles_to_mixture,
)


def _l1_error(half_width: float, upper: float = 20.0, n: int = 20000) -> float:
    """Simpson-quadrature approximation of integral(|g(x) - t(x)| dx) over R,
    using symmetry to integrate only x >= 0."""
    xs = np.linspace(0.0, upper, n + 1)
    g = np.exp(-0.5 * xs**2)
    t = np.maximum(0.0, 1.0 - np.abs(xs) / half_width)
    y = np.abs(g - t)
    weights = np.ones(n + 1)
    weights[1:-1:2] = 4
    weights[2:-1:2] = 2
    h = upper / n
    return 2.0 * float(np.sum(weights * y) * h / 3.0)


class TestDerivedConstants(unittest.TestCase):
    def test_mae_half_width_value(self):
        self.assertAlmostEqual(GAUSSIAN_TRIANGLE_MAE_HALF_WIDTH, 2.33293, places=4)

    def test_mse_half_width_value(self):
        self.assertAlmostEqual(GAUSSIAN_TRIANGLE_MSE_HALF_WIDTH, 2.37547, places=4)

    def test_mae_half_width_is_a_local_minimum_of_l1_error(self):
        # The MAE-optimal width should beat its immediate neighbours on the L1
        # objective it was fit for.
        best = _l1_error(GAUSSIAN_TRIANGLE_MAE_HALF_WIDTH)
        for delta in (-0.05, 0.05, -0.2, 0.2):
            self.assertLess(best, _l1_error(GAUSSIAN_TRIANGLE_MAE_HALF_WIDTH + delta))

    def test_fitted_widths_beat_a_literal_three_sigma_triangle(self):
        # A literal +/-3 sigma triangle is a common rule of thumb, but is
        # strictly worse on the MAE objective than either fitted width.
        self.assertLess(_l1_error(GAUSSIAN_TRIANGLE_MAE_HALF_WIDTH), _l1_error(3.0))
        self.assertLess(_l1_error(GAUSSIAN_TRIANGLE_MSE_HALF_WIDTH), _l1_error(3.0))


class TestFitTriangleToGaussian(unittest.TestCase):
    def test_default_half_width(self):
        mf = GaussianMembership.create(mu=5.0, sigma=2.0)
        tri = fit_triangle_to_gaussian(mf)
        self.assertIsInstance(tri, TriangularMembership)
        self.assertAlmostEqual(tri.b, 5.0)
        self.assertAlmostEqual(tri.a, 5.0 - GAUSSIAN_TRIANGLE_MAE_HALF_WIDTH * 2.0)
        self.assertAlmostEqual(tri.c, 5.0 + GAUSSIAN_TRIANGLE_MAE_HALF_WIDTH * 2.0)

    def test_id_is_preserved(self):
        mf = GaussianMembership.create(mu=0.0, sigma=1.0)
        tri = fit_triangle_to_gaussian(mf)
        self.assertEqual(tri.id, mf.id)

    def test_custom_half_width(self):
        mf = GaussianMembership.create(mu=0.0, sigma=1.0)
        tri = fit_triangle_to_gaussian(mf, half_width_sigma=3.0)
        self.assertAlmostEqual(tri.a, -3.0)
        self.assertAlmostEqual(tri.c, 3.0)

    def test_peak_is_unity(self):
        mf = GaussianMembership.create(mu=2.0, sigma=0.5)
        tri = fit_triangle_to_gaussian(mf)
        self.assertAlmostEqual(float(tri.evaluate(np.array([2.0]))[0]), 1.0)
        self.assertAlmostEqual(float(mf.evaluate(np.array([2.0]))[0]), 1.0)


class TestFitTrianglesToMixture(unittest.TestCase):
    def test_structure_and_ids_preserved(self):
        gm = GaussianMixtureModel(
            feature_models={
                "a": FeatureModel(
                    label_models={
                        0: LabelModel(memberships=[GaussianMembership.create(mu=0.0, sigma=1.0)]),
                        1: LabelModel(memberships=[GaussianMembership.create(mu=5.0, sigma=1.5)]),
                    }
                )
            }
        )
        tri_gm = fit_triangles_to_mixture(gm)
        self.assertEqual(set(tri_gm.feature_models.keys()), {"a"})
        self.assertEqual(set(tri_gm.feature_models["a"].label_models.keys()), {0, 1})
        for label in (0, 1):
            orig = gm.feature_models["a"].label_models[label].memberships[0]
            fitted = tri_gm.feature_models["a"].label_models[label].memberships[0]
            self.assertIsInstance(fitted, TriangularMembership)
            self.assertEqual(fitted.id, orig.id)
            self.assertAlmostEqual(fitted.b, orig.mu)

    def test_non_gaussian_members_pass_through(self):
        tri = TriangularMembership.create(a=0.0, b=1.0, c=2.0)
        gm = GaussianMixtureModel(
            feature_models={"a": FeatureModel(label_models={0: LabelModel(memberships=[tri])})}
        )
        out = fit_triangles_to_mixture(gm)
        self.assertIs(out.feature_models["a"].label_models[0].memberships[0], tri)


if __name__ == "__main__":
    unittest.main()
