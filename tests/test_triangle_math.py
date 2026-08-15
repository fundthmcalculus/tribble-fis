"""
Tests for triangular membership function fitting via histogram-based EM.

Mirrors tests/test_trapz_math.py's structure: the triangle is fit as a
trapezoid whose plateau has collapsed to a single apex point, so most of the
same scenarios (unimodal, bimodal, degenerate, BIC selection) apply directly,
just with one fewer free parameter per component.
"""

import unittest

import numpy as np
import pandas as pd

from tribblefis.triangle_math import (
    triangle_pdf,
    TriangleMixtureModel,
    fit_triangles_em,
    find_optimal_triangles,
    fit_triangles,
    create_triangle_membership_dict,
)
from tribblefis.trapz_math import trapz_pdf
from tribblefis.gauss_data import TriangularMembership, GaussianMixtureModel
from tribblefis.gaussian_classifier import TribbleClassifier


class TestTrianglePDF(unittest.TestCase):
    """Test triangular PDF computation."""

    def test_triangle_pdf_shape(self):
        x = np.linspace(-2, 2, 100)
        y = triangle_pdf(x, a=-1, b=0, c=1)
        self.assertEqual(y.shape, x.shape)

    def test_triangle_pdf_matches_degenerate_trapz_pdf(self):
        """triangle_pdf(x, a, b, c) must equal trapz_pdf(x, a, b, b, c) exactly --
        that equivalence is the whole basis for reusing the trapezoid machinery."""
        x = np.linspace(-3, 3, 200)
        y_tri = triangle_pdf(x, a=-1.5, b=0.25, c=1.75)
        y_trapz = trapz_pdf(x, -1.5, 0.25, 0.25, 1.75)
        np.testing.assert_allclose(y_tri, y_trapz)

    def test_triangle_pdf_peak_at_apex(self):
        x = np.linspace(-2, 2, 1000)
        y = triangle_pdf(x, a=-1, b=-0.25, c=1)
        max_idx = np.argmax(y)
        self.assertAlmostEqual(x[max_idx], -0.25, delta=0.01)

    def test_triangle_pdf_outside_support(self):
        x_left = np.array([-5.0])
        x_right = np.array([5.0])
        a, b, c = -1, 0, 1
        self.assertTrue(np.allclose(triangle_pdf(x_left, a, b, c), 0))
        self.assertTrue(np.allclose(triangle_pdf(x_right, a, b, c), 0))

    def test_triangle_pdf_degenerate_point(self):
        x = np.array([0.0, 0.5, 1.0])
        y = triangle_pdf(x, a=0.5, b=0.5, c=0.5)
        self.assertTrue(np.allclose(y, 0))


class TestTriangleMixtureModel(unittest.TestCase):
    """Test TriangleMixtureModel class."""

    def test_fit_unimodal(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 500)
        model = TriangleMixtureModel(n_components=1, n_bins=50)
        model.fit(data)

        self.assertEqual(len(model.triangles_), 1)
        self.assertTrue(np.allclose(model.weights_, [1.0]))
        tri = model.triangles_[0]
        center = (tri.a + tri.c) / 2
        self.assertGreater(center, -1.0)
        self.assertLess(center, 1.0)

    def test_fit_bimodal(self):
        np.random.seed(42)
        data = np.concatenate([
            np.random.normal(-3, 0.5, 500),
            np.random.normal(3, 0.5, 500)
        ])
        model = TriangleMixtureModel(n_components=2, n_bins=50)
        model.fit(data)

        self.assertEqual(len(model.triangles_), 2)
        self.assertEqual(len(model.weights_), 2)
        self.assertTrue(np.allclose(model.weights_.sum(), 1.0))

        centers = sorted([(t.a + t.c) / 2 for t in model.triangles_])
        self.assertLess(centers[0], -1)
        self.assertGreater(centers[1], 1)

    def test_fit_auto_select_components(self):
        np.random.seed(42)
        data = np.concatenate([
            np.random.normal(-2, 0.5, 500),
            np.random.normal(2, 0.5, 500)
        ])
        model = TriangleMixtureModel(n_components=0, max_components=4, n_bins=50)
        model.fit(data)

        self.assertEqual(len(model.triangles_), 2)

    def test_weights_sum_to_one(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 500)
        model = TriangleMixtureModel(n_components=2)
        model.fit(data)

        self.assertTrue(np.allclose(model.weights_.sum(), 1.0))

    def test_bic_computed(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 500)
        model = TriangleMixtureModel(n_components=1)
        model.fit(data)

        self.assertIsNotNone(model.bic_)
        self.assertGreater(model.bic_, 0)

    def test_bic_uses_one_fewer_param_than_trapezoid(self):
        """A triangle has 3 free shape params vs the trapezoid's 4, so at
        matched K and log-likelihood the triangle BIC must be strictly lower."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 500)
        N = len(data)
        ll = -100.0
        K = 2
        triangle_bic = (4 * K - 1) * np.log(N) - 2 * ll
        trapz_bic = (5 * K - 1) * np.log(N) - 2 * ll
        self.assertLess(triangle_bic, trapz_bic)

    def test_fit_degenerate_data_single_value(self):
        data = np.ones(100) * 5.0
        model = TriangleMixtureModel(n_components=1)
        model.fit(data)

        self.assertEqual(len(model.triangles_), 1)
        tri = model.triangles_[0]
        self.assertEqual(tri.a, 5.0)
        self.assertEqual(tri.c, 5.0)
        self.assertEqual(tri.b, 5.0)


class TestOptimalTrianglesSelection(unittest.TestCase):
    """Test BIC-based model selection."""

    def test_bic_unimodal(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 500)
        optimal_k = find_optimal_triangles(data, max_components=4)
        self.assertEqual(optimal_k, 1)

    def test_bic_bimodal(self):
        np.random.seed(42)
        data = np.concatenate([
            np.random.normal(-3, 0.5, 500),
            np.random.normal(3, 0.5, 500)
        ])
        optimal_k = find_optimal_triangles(data, max_components=4)
        self.assertEqual(optimal_k, 2)


class TestFitTriangles(unittest.TestCase):
    """Test fit_triangles function."""

    def test_fit_triangles_basic(self):
        np.random.seed(42)
        X = pd.DataFrame({
            'feature': np.concatenate([
                np.random.normal(0, 1, 100),
                np.random.normal(3, 1, 100)
            ])
        })
        y = pd.Series([0] * 100 + [1] * 100)

        tri0 = fit_triangles(X, y, 'feature', label_value=0, n_triangles=1)
        tri1 = fit_triangles(X, y, 'feature', label_value=1, n_triangles=1)

        self.assertGreaterEqual(len(tri0), 1)
        self.assertGreaterEqual(len(tri1), 1)

        center0 = (tri0[0].a + tri0[0].c) / 2
        center1 = (tri1[0].a + tri1[0].c) / 2
        self.assertLess(center0, center1)


class TestCreateTriangleMembershipDict(unittest.TestCase):
    """Test the create_triangle_membership_dict function."""

    def test_create_model_basic(self):
        np.random.seed(42)
        X = pd.DataFrame({
            'f1': np.random.normal(0, 1, 100),
            'f2': np.random.normal(1, 1, 100),
        })
        y = pd.Series(np.concatenate([np.zeros(50, dtype=int), np.ones(50, dtype=int)]))

        model = create_triangle_membership_dict(X, y, top_n_var_names=['f1', 'f2'], n_triangles=1)

        self.assertIsInstance(model, GaussianMixtureModel)
        self.assertIn('f1', model.feature_models)
        self.assertIn('f2', model.feature_models)

    def test_model_has_triangles(self):
        np.random.seed(42)
        X = pd.DataFrame({'feature': np.random.normal(0, 1, 100)})
        y = pd.Series(np.concatenate([np.zeros(50, dtype=int), np.ones(50, dtype=int)]))

        model = create_triangle_membership_dict(X, y, top_n_var_names=['feature'], n_triangles=1)

        has_triangles = False
        for feature_model in model.feature_models.values():
            for label_model in feature_model.label_models.values():
                for mf in label_model.memberships:
                    if isinstance(mf, TriangularMembership):
                        has_triangles = True

        self.assertTrue(has_triangles)


class TestEMConvergence(unittest.TestCase):
    """Test EM algorithm convergence properties."""

    def test_em_produces_valid_triangles(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 500)
        triangles, weights, ll = fit_triangles_em(
            data, n_components=2, n_bins=50, max_iter=100
        )

        for tri in triangles:
            self.assertLessEqual(tri.a, tri.b)
            self.assertLessEqual(tri.b, tri.c)

    def test_em_weights_positive_and_sum_to_one(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 500)
        triangles, weights, ll = fit_triangles_em(
            data, n_components=2, n_bins=50
        )

        self.assertTrue(np.all(weights >= 0))
        self.assertTrue(np.allclose(weights.sum(), 1.0))


class TestNumericalStability(unittest.TestCase):
    """Test numerical stability and edge cases."""

    def test_handles_very_small_data(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 10)
        model = TriangleMixtureModel(n_components=1)
        model.fit(data)
        self.assertEqual(len(model.triangles_), 1)

    def test_handles_large_data(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 10000)
        model = TriangleMixtureModel(n_components=1, n_bins=100)
        model.fit(data)
        self.assertEqual(len(model.triangles_), 1)

    def test_handles_negative_data(self):
        np.random.seed(42)
        data = np.random.normal(-100, 10, 500)
        model = TriangleMixtureModel(n_components=1)
        model.fit(data)
        self.assertEqual(len(model.triangles_), 1)
        tri = model.triangles_[0]
        self.assertLess(tri.c, 0)


class TestTribbleClassifierTriangular(unittest.TestCase):
    """Test TribbleClassifier(member_function="triangular") end to end."""

    def test_fit_predict(self):
        np.random.seed(42)
        X = pd.DataFrame({
            'a': np.concatenate([np.random.normal(0, 1, 100), np.random.normal(5, 1, 100)]),
            'b': np.concatenate([np.random.normal(-2, 1, 100), np.random.normal(2, 1, 100)]),
        })
        y = pd.Series([0] * 100 + [1] * 100)

        clf = TribbleClassifier(member_function="triangular", top_p=1.0, n_gaussians=1, random_state=0)
        clf.fit(X, y)
        preds = clf.predict(X)

        self.assertEqual(len(preds), len(y))
        for feature_model in clf.model_.feature_models.values():
            for label_model in feature_model.label_models.values():
                for mf in label_model.memberships:
                    self.assertIsInstance(mf, TriangularMembership)

    def test_unknown_member_function_raises(self):
        X = pd.DataFrame({'a': np.random.normal(0, 1, 20)})
        y = pd.Series([0] * 10 + [1] * 10)
        clf = TribbleClassifier(member_function="not-a-real-shape")
        with self.assertRaises(ValueError):
            clf.fit(X, y)


if __name__ == '__main__':
    unittest.main()
