"""
Comprehensive tests for trapezoidal membership function fitting via EM.

Tests verify:
1. Trapezoid PDF computation
2. Initialization strategy (histogram-driven peaks)
3. E-step responsibility computation
4. M-step convergence (weights and parameters)
5. BIC-based model selection
6. Edge cases (unimodal, bimodal, multimodal data)
7. Numerical stability (log-likelihood tracking)
8. Degenerate component handling
9. Performance comparison with fast histogram method
"""

import unittest
import time
import numpy as np
import pandas as pd
from tribblefis.trapz_math import (
    trapz_pdf,
    TrapzMixtureModel,
    fit_trapezoids_em,
    find_optimal_trapezoids,
    fit_trapezoids,
    create_trapz_membership_dict,
    _solve_ordered_params,
)
from tribblefis.trapz_math_fast import (
    fit_trapezoids_fast,
    trapz_pdf_fast,
)
from tribblefis.gauss_data import TrapezoidMembership, TriangularMembership, GaussianMixtureModel


class TestTrapzPDF(unittest.TestCase):
    """Test trapezoidal PDF computation."""

    def test_trapz_pdf_shape(self):
        """Test that trapz_pdf returns correct shape."""
        x = np.linspace(-2, 2, 100)
        y = trapz_pdf(x, a=-1, b=-0.5, c=0.5, d=1)
        self.assertEqual(y.shape, x.shape)

    def test_trapz_pdf_flat_top(self):
        """Test trapz_pdf has max value in [b, c] plateau."""
        x = np.linspace(-2, 2, 1000)
        y = trapz_pdf(x, a=-1, b=-0.5, c=0.5, d=1)
        max_idx = np.argmax(y)
        x_at_max = x[max_idx]
        # Max should be in the plateau region [b, c]
        self.assertGreaterEqual(x_at_max, -0.5)
        self.assertLessEqual(x_at_max, 0.5)

    def test_trapz_pdf_outside_support(self):
        """Test trapz_pdf is zero outside [a, d]."""
        x_left = np.array([-5.0])
        x_right = np.array([5.0])
        a, b, c, d = -1, -0.5, 0.5, 1
        self.assertTrue(np.allclose(trapz_pdf(x_left, a, b, c, d), 0))
        self.assertTrue(np.allclose(trapz_pdf(x_right, a, b, c, d), 0))

    def test_trapz_pdf_symmetry(self):
        """Test trapz_pdf is symmetric for symmetric parameters."""
        x = np.linspace(-2, 2, 100)
        # Symmetric trapezoid around 0
        y = trapz_pdf(x, a=-1, b=-0.5, c=0.5, d=1)
        # Should be symmetric
        left = y[x < 0]
        right = y[x > 0]
        self.assertTrue(np.allclose(left, right[::-1][:len(left)]))

    def test_trapz_pdf_rectangular(self):
        """Test that rectangle (b == c) is valid."""
        x = np.linspace(-2, 2, 100)
        y = trapz_pdf(x, a=-1, b=0, c=0, d=1)
        # Should have max value >= 0
        self.assertGreaterEqual(np.max(y), 0)

    def test_trapz_pdf_degenerate_point(self):
        """Test degenerate case where a == b == c == d (point)."""
        x = np.array([0.0, 0.5, 1.0])
        y = trapz_pdf(x, a=0.5, b=0.5, c=0.5, d=0.5)
        # All values should be 0 (no area)
        self.assertTrue(np.allclose(y, 0))


class TestSolveOrderedParams(unittest.TestCase):
    """`_solve_ordered_params` replaced `scipy.optimize.minimize(method="SLSQP",
    constraints=[...])`'s ``a <= b <= c <= d`` ordering constraint with a
    gap reparametrization (see its docstring). These tests target that
    reparametrization directly, independent of the EM M-step that calls it.
    """

    def test_result_is_ordered_and_within_bounds(self):
        def objective(params):
            # Minimized at (0.1, 0.4, 0.6, 0.9), safely inside [0, 1].
            target = np.array([0.1, 0.4, 0.6, 0.9])
            return float(np.sum((np.asarray(params) - target) ** 2))

        x0 = np.array([0.2, 0.3, 0.5, 0.8])
        solved, _ = _solve_ordered_params(objective, x0, data_min=0.0, data_max=1.0)
        self.assertEqual(len(solved), 4)
        self.assertTrue(np.all(np.diff(solved) >= -1e-9))
        self.assertTrue(np.all(solved >= 0.0 - 1e-9))
        self.assertTrue(np.all(solved <= 1.0 + 1e-9))

    def test_recovers_near_optimum_inside_bounds(self):
        def objective(params):
            target = np.array([0.1, 0.4, 0.6, 0.9])
            return float(np.sum((np.asarray(params) - target) ** 2))

        x0 = np.array([0.0, 0.25, 0.5, 0.75])
        solved, solved_obj = _solve_ordered_params(objective, x0, data_min=0.0, data_max=1.0)
        np.testing.assert_allclose(solved, [0.1, 0.4, 0.6, 0.9], atol=0.05)
        self.assertLess(solved_obj, 1e-3)

    def test_three_parameter_triangle_case(self):
        def objective(params):
            target = np.array([0.2, 0.5, 0.8])
            return float(np.sum((np.asarray(params) - target) ** 2))

        x0 = np.array([0.1, 0.4, 0.9])
        solved, _ = _solve_ordered_params(objective, x0, data_min=0.0, data_max=1.0)
        self.assertEqual(len(solved), 3)
        self.assertTrue(np.all(np.diff(solved) >= -1e-9))


class TestTrapzMixtureModel(unittest.TestCase):
    """Test TrapzMixtureModel class."""

    def test_fit_unimodal(self):
        """Test fitting unimodal Gaussian data."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 500)
        model = TrapzMixtureModel(n_components=1, n_bins=50)
        model.fit(data)

        self.assertEqual(len(model.trapezoids_), 1)
        self.assertTrue(np.allclose(model.weights_, [1.0]))
        # Trapezoid should be centered around 0
        trapz = model.trapezoids_[0]
        center = (trapz.a + trapz.d) / 2
        self.assertGreater(center, -0.5)
        self.assertLess(center, 0.5)

    def test_fit_bimodal(self):
        """Test fitting bimodal data with two well-separated modes."""
        np.random.seed(42)
        data = np.concatenate([
            np.random.normal(-3, 0.5, 500),
            np.random.normal(3, 0.5, 500)
        ])
        model = TrapzMixtureModel(n_components=2, n_bins=50)
        model.fit(data)

        self.assertEqual(len(model.trapezoids_), 2)
        self.assertEqual(len(model.weights_), 2)
        self.assertTrue(np.allclose(model.weights_.sum(), 1.0))

        # Trapezoids should be near -3 and 3
        centers = sorted([(t.a + t.d) / 2 for t in model.trapezoids_])
        self.assertLess(centers[0], -1)  # First mode near -3
        self.assertGreater(centers[1], 1)   # Second mode near 3

    def test_fit_auto_select_components(self):
        """Test auto-selection of number of components via BIC."""
        np.random.seed(42)
        # Bimodal data
        data = np.concatenate([
            np.random.normal(-2, 0.5, 500),
            np.random.normal(2, 0.5, 500)
        ])
        model = TrapzMixtureModel(n_components=0, max_components=4, n_bins=50)
        model.fit(data)

        # Should select K=2 for bimodal data
        self.assertEqual(len(model.trapezoids_), 2)

    def test_weights_sum_to_one(self):
        """Test that mixing weights sum to 1."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 500)
        model = TrapzMixtureModel(n_components=2)
        model.fit(data)

        self.assertTrue(np.allclose(model.weights_.sum(), 1.0))

    def test_bic_computed(self):
        """Test that BIC is computed and reasonable."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 500)
        model = TrapzMixtureModel(n_components=1)
        model.fit(data)

        self.assertIsNotNone(model.bic_)
        self.assertGreater(model.bic_, 0)  # BIC should be positive

    def test_convergence(self):
        """Test that log-likelihood is reasonable."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 500)
        model = TrapzMixtureModel(n_components=1, max_iter=10, n_bins=50)
        model.fit(data)

        # Log-likelihood should be reasonable (not too negative)
        self.assertGreater(model.log_likelihood_, -5000)

    def test_fit_degenerate_data_single_value(self):
        """Test fitting when all data is identical."""
        data = np.ones(100) * 5.0
        model = TrapzMixtureModel(n_components=1)
        model.fit(data)

        self.assertEqual(len(model.trapezoids_), 1)
        # Trapezoid should be at the single value
        trapz = model.trapezoids_[0]
        self.assertEqual(trapz.a, 5.0)
        self.assertEqual(trapz.d, 5.0)


class TestOptimalTrapezoidsSelection(unittest.TestCase):
    """Test BIC-based model selection."""

    def test_bic_unimodal(self):
        """Test BIC selects K=1 for unimodal data."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 500)
        optimal_k = find_optimal_trapezoids(data, max_components=4)
        self.assertEqual(optimal_k, 1)

    def test_bic_bimodal(self):
        """Test BIC selects K=2 for bimodal data."""
        np.random.seed(42)
        data = np.concatenate([
            np.random.normal(-3, 0.5, 500),
            np.random.normal(3, 0.5, 500)
        ])
        optimal_k = find_optimal_trapezoids(data, max_components=4)
        self.assertEqual(optimal_k, 2)

    def test_bic_trimodal(self):
        """Test BIC selects K=3 for trimodal data."""
        np.random.seed(42)
        data = np.concatenate([
            np.random.normal(-4, 0.4, 300),
            np.random.normal(0, 0.4, 300),
            np.random.normal(4, 0.4, 300)
        ])
        optimal_k = find_optimal_trapezoids(data, max_components=4)
        self.assertEqual(optimal_k, 3)


class TestFitTrapezoids(unittest.TestCase):
    """Test fit_trapezoids function."""

    def test_fit_trapezoids_basic(self):
        """Test basic trapezoid fitting to labeled data."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature': np.concatenate([
                np.random.normal(0, 1, 100),
                np.random.normal(3, 1, 100)
            ])
        })
        y = pd.Series([0] * 100 + [1] * 100)

        trapz0 = fit_trapezoids(X, y, 'feature', label_value=0, n_trapezoids=1)
        trapz1 = fit_trapezoids(X, y, 'feature', label_value=1, n_trapezoids=1)

        self.assertGreaterEqual(len(trapz0), 1)
        self.assertGreaterEqual(len(trapz1), 1)

        # Centers should reflect the data distribution
        center0 = (trapz0[0].a + trapz0[0].d) / 2
        center1 = (trapz1[0].a + trapz1[0].d) / 2
        self.assertLess(center0, center1)


class TestCreateTrapzMembershipDict(unittest.TestCase):
    """Test the create_trapz_membership_dict function."""

    def test_create_model_basic(self):
        """Test creating a trapezoid membership model."""
        np.random.seed(42)
        X = pd.DataFrame({
            'f1': np.random.normal(0, 1, 100),
            'f2': np.random.normal(1, 1, 100),
            'f3': np.random.normal(2, 1, 100),
        })
        y = pd.Series(np.concatenate([np.zeros(50, dtype=int), np.ones(50, dtype=int)]))

        model = create_trapz_membership_dict(X, y, top_n_var_names=['f1', 'f2'], n_trapezoids=1)

        self.assertIsInstance(model, GaussianMixtureModel)
        self.assertIn('f1', model.feature_models)
        self.assertIn('f2', model.feature_models)

    def test_model_has_trapezoids(self):
        """Test that created model contains TrapezoidMembership objects."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature': np.random.normal(0, 1, 100),
        })
        y = pd.Series(np.concatenate([np.zeros(50, dtype=int), np.ones(50, dtype=int)]))

        model = create_trapz_membership_dict(X, y, top_n_var_names=['feature'], n_trapezoids=1)

        has_trapezoids = False
        for feature_model in model.feature_models.values():
            for label_model in feature_model.label_models.values():
                for mf in label_model.memberships:
                    if isinstance(mf, TrapezoidMembership):
                        has_trapezoids = True

        self.assertTrue(has_trapezoids)


class TestEMConvergence(unittest.TestCase):
    """Test EM algorithm convergence properties."""

    def test_em_produces_valid_trapezoids(self):
        """Test that EM produces valid trapezoids (a <= b <= c <= d)."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 500)
        trapezoids, weights, ll = fit_trapezoids_em(
            data, n_components=2, n_bins=50, max_iter=100
        )

        for trapz in trapezoids:
            self.assertLessEqual(trapz.a, trapz.b)
            self.assertLessEqual(trapz.b, trapz.c)
            self.assertLessEqual(trapz.c, trapz.d)

    def test_em_weights_positive_and_sum_to_one(self):
        """Test that EM weights are positive and sum to 1."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 500)
        trapezoids, weights, ll = fit_trapezoids_em(
            data, n_components=2, n_bins=50
        )

        self.assertTrue(np.all(weights >= 0))
        self.assertTrue(np.allclose(weights.sum(), 1.0))


class TestNumericalStability(unittest.TestCase):
    """Test numerical stability and edge cases."""

    def test_handles_very_small_data(self):
        """Test fitting on small sample size."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 10)
        model = TrapzMixtureModel(n_components=1)
        model.fit(data)
        self.assertEqual(len(model.trapezoids_), 1)

    def test_handles_large_data(self):
        """Test fitting on large sample size."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 10000)
        model = TrapzMixtureModel(n_components=1, n_bins=100)
        model.fit(data)
        self.assertEqual(len(model.trapezoids_), 1)

    def test_handles_moderate_scale_difference(self):
        """Test fitting data with moderate scale difference."""
        np.random.seed(42)
        data = np.concatenate([
            np.random.normal(0, 1, 500),
            np.random.normal(10, 1, 500)
        ])
        model = TrapzMixtureModel(n_components=2)
        model.fit(data)
        self.assertGreaterEqual(len(model.trapezoids_), 1)

    def test_handles_negative_data(self):
        """Test fitting negative values."""
        np.random.seed(42)
        data = np.random.normal(-100, 10, 500)
        model = TrapzMixtureModel(n_components=1)
        model.fit(data)
        self.assertEqual(len(model.trapezoids_), 1)
        trapz = model.trapezoids_[0]
        self.assertLess(trapz.d, 0)


class TestFastHistogramMethod(unittest.TestCase):
    """Test the fast histogram-based fitting method."""

    def test_fast_method_basic(self):
        """Test basic fast trapezoid fitting."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 500)
        trapezoids, weights = fit_trapezoids_fast(data, n_bins=50)

        self.assertGreater(len(trapezoids), 0)
        self.assertTrue(np.allclose(weights.sum(), 1.0))

    def test_fast_method_unimodal(self):
        """Test fast method on unimodal data."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        trapezoids, weights = fit_trapezoids_fast(data, n_bins=50)

        # Fast method should create multiple trapezoids for contiguous bins
        self.assertGreaterEqual(len(trapezoids), 1)
        # All trapezoids should be valid
        for trapz in trapezoids:
            self.assertLessEqual(trapz.a, trapz.b)
            self.assertLessEqual(trapz.b, trapz.c)
            self.assertLessEqual(trapz.c, trapz.d)

    def test_fast_method_bimodal(self):
        """Test fast method on bimodal data."""
        np.random.seed(42)
        data = np.concatenate([
            np.random.normal(-3, 0.5, 500),
            np.random.normal(3, 0.5, 500)
        ])
        trapezoids, weights = fit_trapezoids_fast(data, n_bins=50)

        self.assertGreater(len(trapezoids), 0)
        self.assertTrue(np.allclose(weights.sum(), 1.0))

    def test_fast_method_equal_weights(self):
        """Test that fast method uses equal weights."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 500)
        trapezoids, weights = fit_trapezoids_fast(data, n_bins=50)

        # All weights should be equal (uniform distribution)
        expected_weight = 1.0 / len(trapezoids)
        self.assertTrue(np.allclose(weights, expected_weight))

    def test_fast_method_edge_case_empty(self):
        """Test fast method with empty data."""
        data = np.array([])
        trapezoids, weights = fit_trapezoids_fast(data, n_bins=50)

        self.assertEqual(len(trapezoids), 0)
        self.assertEqual(len(weights), 0)

    def test_fast_method_single_value(self):
        """Test fast method when all data is identical."""
        data = np.ones(100) * 5.0
        trapezoids, weights = fit_trapezoids_fast(data, n_bins=50)

        self.assertGreaterEqual(len(trapezoids), 1)
        # Trapezoid should be centered at the single value
        trapz = trapezoids[0]
        self.assertAlmostEqual(trapz.a, 5.0, places=1)
        self.assertAlmostEqual(trapz.d, 5.0, places=1)


class TestPerformanceComparison(unittest.TestCase):
    """Performance benchmark comparing EM vs Fast methods."""

    def test_performance_speedup_unimodal(self):
        """Verify fast method is significantly faster on unimodal data."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)

        # Time EM method
        start_em = time.perf_counter()
        em_result = TrapzMixtureModel(n_components=1).fit(data)
        em_time = time.perf_counter() - start_em

        # Time fast method
        start_fast = time.perf_counter()
        fit_trapezoids_fast(data, n_bins=50)
        fast_time = time.perf_counter() - start_fast

        # Fast should be significantly faster (at least 100x)
        speedup = em_time / fast_time
        self.assertGreater(speedup, 100)
        # Print for informational purposes
        print(f"\nUnimodal Speedup: {speedup:.1f}x (EM: {em_time:.4f}s, Fast: {fast_time:.6f}s)")

    def test_performance_speedup_bimodal(self):
        """Verify fast method is significantly faster on bimodal data."""
        np.random.seed(42)
        data = np.concatenate([
            np.random.normal(-3, 0.5, 500),
            np.random.normal(3, 0.5, 500)
        ])

        # Time EM method
        start_em = time.perf_counter()
        em_result = TrapzMixtureModel(n_components=2).fit(data)
        em_time = time.perf_counter() - start_em

        # Time fast method
        start_fast = time.perf_counter()
        fit_trapezoids_fast(data, n_bins=50)
        fast_time = time.perf_counter() - start_fast

        # Fast should be significantly faster (at least 100x)
        speedup = em_time / fast_time
        self.assertGreater(speedup, 100)
        print(f"\nBimodal Speedup: {speedup:.1f}x (EM: {em_time:.4f}s, Fast: {fast_time:.6f}s)")

    def test_performance_consistency_fast(self):
        """Verify fast method produces consistent results."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 500)

        # Run twice
        trapz1, weights1 = fit_trapezoids_fast(data, n_bins=50)
        trapz2, weights2 = fit_trapezoids_fast(data, n_bins=50)

        # Results should be identical (deterministic)
        self.assertEqual(len(trapz1), len(trapz2))
        for t1, t2 in zip(trapz1, trapz2):
            self.assertEqual(t1.a, t2.a)
            self.assertEqual(t1.b, t2.b)
            self.assertEqual(t1.c, t2.c)
            self.assertEqual(t1.d, t2.d)
        self.assertTrue(np.allclose(weights1, weights2))

    def test_fast_method_pdf_evaluation(self):
        """Test that fast trapezoid PDFs evaluate correctly."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 500)
        trapezoids, _ = fit_trapezoids_fast(data, n_bins=50)

        # Evaluate PDF at multiple points
        x = np.linspace(data.min() - 1, data.max() + 1, 100)
        for trapz in trapezoids:
            pdf_vals = trapz_pdf_fast(x, trapz.a, trapz.b, trapz.c, trapz.d)
            # PDF values should be non-negative
            self.assertTrue(np.all(pdf_vals >= 0))
            # PDF should be zero outside [a, d]
            self.assertTrue(np.allclose(pdf_vals[x < trapz.a], 0))
            self.assertTrue(np.allclose(pdf_vals[x > trapz.d], 0))

    def test_component_count_em_vs_fast(self):
        """Compare component counts between EM and fast methods."""
        np.random.seed(42)
        data = np.concatenate([
            np.random.normal(-3, 0.5, 500),
            np.random.normal(3, 0.5, 500)
        ])

        # EM method (auto-selects components)
        model_em = TrapzMixtureModel(n_components=0, max_components=4).fit(data)
        n_em = len(model_em.trapezoids_)

        # Fast method
        trapz_fast, _ = fit_trapezoids_fast(data, n_bins=50)
        n_fast = len(trapz_fast)

        print(f"\nComponent Count - EM: {n_em}, Fast: {n_fast}")
        # Fast typically creates more (literal) trapezoids
        self.assertGreaterEqual(n_fast, 1)
        self.assertGreaterEqual(n_em, 1)


class TestShapeParameterUnification(unittest.TestCase):
    """fit_trapezoids_em (and everything built on it) is one engine for both
    shapes -- a triangle is a trapezoid whose plateau has collapsed to a
    single apex point. These pin the two things that unification promises:
    the default (shape="trapezoid") behavior is completely unchanged, and
    shape="triangle" runs through the identical code path, just returning
    TriangularMembership objects with one fewer optimized parameter."""

    def test_default_shape_is_trapezoid(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 300)
        trapezoids, weights, ll = fit_trapezoids_em(data, n_components=1, n_bins=50)
        self.assertIsInstance(trapezoids[0], TrapezoidMembership)

    def test_shape_triangle_returns_triangular_membership(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 300)
        triangles, weights, ll = fit_trapezoids_em(data, n_components=1, n_bins=50, shape="triangle")
        self.assertIsInstance(triangles[0], TriangularMembership)

    def test_trapezoid_shape_unaffected_by_triangle_support(self):
        """Adding shape="triangle" must not perturb a single bit of the
        default trapezoid path's output -- same seed, same result as before
        the shape parameter existed."""
        np.random.seed(42)
        data = np.concatenate([np.random.normal(-3, 0.5, 300), np.random.normal(3, 0.5, 300)])
        trapezoids_a, weights_a, ll_a = fit_trapezoids_em(data, n_components=2, n_bins=50)
        trapezoids_b, weights_b, ll_b = fit_trapezoids_em(data, n_components=2, n_bins=50, shape="trapezoid")
        self.assertEqual(len(trapezoids_a), len(trapezoids_b))
        for ta, tb in zip(trapezoids_a, trapezoids_b):
            self.assertEqual((ta.a, ta.b, ta.c, ta.d), (tb.a, tb.b, tb.c, tb.d))
        self.assertEqual(ll_a, ll_b)

    def test_triangle_shape_has_collapsed_plateau(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 300)
        triangles, _weights, _ll = fit_trapezoids_em(data, n_components=1, n_bins=50, shape="triangle")
        # TriangularMembership itself has no separate b/c plateau -- a is the
        # left foot, b the apex, c the right foot -- so there is nothing to
        # collapse at this layer; this just pins the field count/shape.
        tri = triangles[0]
        self.assertLessEqual(tri.a, tri.b)
        self.assertLessEqual(tri.b, tri.c)

    def test_trapz_mixture_model_shape_param_selects_membership_type(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 300)
        trapz_model = TrapzMixtureModel(n_components=1, shape="trapezoid").fit(data)
        tri_model = TrapzMixtureModel(n_components=1, shape="triangle").fit(data)
        self.assertIsInstance(trapz_model.trapezoids_[0], TrapezoidMembership)
        self.assertIsInstance(tri_model.trapezoids_[0], TriangularMembership)

    def test_bic_formula_differs_by_one_parameter_per_component(self):
        """Uses genuinely separated bimodal data so the requested component
        count (2) is actually what gets fitted -- unimodal data collapses to
        1 component regardless of request (a pre-existing histogram-peak-init
        limitation, see docs/triangle-em-resolution-evaluation.md), which
        would make a hardcoded K=2 expectation wrong for the wrong reason."""
        np.random.seed(42)
        data = np.concatenate([np.random.normal(-3, 0.5, 300), np.random.normal(3, 0.5, 300)])
        trapz_model = TrapzMixtureModel(n_components=2, shape="trapezoid").fit(data)
        tri_model = TrapzMixtureModel(n_components=2, shape="triangle").fit(data)
        N = len(data)
        K_trapz = len(trapz_model.trapezoids_)
        K_tri = len(tri_model.trapezoids_)
        self.assertEqual(K_trapz, 2)
        self.assertEqual(K_tri, 2)
        expected_trapz_params = 5 * K_trapz - 1
        expected_tri_params = 4 * K_tri - 1
        self.assertAlmostEqual(
            trapz_model.bic_, expected_trapz_params * np.log(N) - 2 * trapz_model.log_likelihood_
        )
        self.assertAlmostEqual(
            tri_model.bic_, expected_tri_params * np.log(N) - 2 * tri_model.log_likelihood_
        )


if __name__ == '__main__':
    unittest.main()


def test_width_reg_widens_em_support():
    """width_reg > 0 counters the density-MLE support collapse in the EM M-step,
    producing wider, partition-friendly trapezoids. Backward compatible:
    width_reg=0 (default) is unchanged. See issue #163."""
    rng = np.random.default_rng(0)
    data = np.concatenate([rng.normal(-2.0, 0.4, 400), rng.normal(2.0, 0.4, 400)])

    def mean_support(width_reg):
        mfs, _w, _ll = fit_trapezoids_em(
            data, n_components=2, n_bins=50, random_state=42, width_reg=width_reg
        )
        return float(np.mean([mf.d - mf.a for mf in mfs]))

    base = mean_support(0.0)
    wide = mean_support(1.0)
    assert wide > base * 1.1, f"expected wider support, got base={base:.2f} wide={wide:.2f}"
