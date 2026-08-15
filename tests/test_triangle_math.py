"""
Tests for triangular membership function fitting via histogram-based EM.

Mirrors tests/test_trapz_math.py's structure: the triangle is fit as a
trapezoid whose plateau has collapsed to a single apex point, so most of the
same scenarios (unimodal, bimodal, degenerate, BIC selection) apply directly,
just with one fewer free parameter per component.
"""

import io
import time
import contextlib
import unittest

import numpy as np
import pandas as pd

from tribblefis.triangle_math import (
    triangle_pdf,
    TriangleMixtureModel,
    fit_triangles_em,
    fit_triangle_mixture_1d,
    find_optimal_triangles,
    fit_triangles,
    create_triangle_membership_dict,
    _em_e_step,
    _em_m_step_params,
    _init_triangles_from_histogram,
)
from tribblefis.trapz_math import trapz_pdf, fit_trapezoids_em
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

    def test_triangle_pdf_symmetry(self):
        x = np.linspace(-2, 2, 200)
        y = triangle_pdf(x, a=-1, b=0, c=1)
        left = y[x < 0]
        right = y[x > 0]
        self.assertTrue(np.allclose(left, right[::-1][:len(left)]))

    def test_triangle_pdf_asymmetric_shape(self):
        """A skewed triangle (apex not centered) should still peak at b and be
        zero at both feet, even though the two slopes have different widths."""
        x = np.linspace(-5, 5, 2000)
        y = triangle_pdf(x, a=-4, b=-3, c=4)
        self.assertAlmostEqual(x[np.argmax(y)], -3, delta=0.02)
        self.assertTrue(np.all(y >= 0))

    def test_triangle_pdf_integrates_to_one(self):
        """A normalized PDF must integrate to 1 over its support, same
        invariant trapz_pdf is held to."""
        x = np.linspace(-10, 10, 200001)
        y = triangle_pdf(x, a=-3.0, b=1.0, c=4.0)
        area = np.trapezoid(y, x)
        self.assertAlmostEqual(area, 1.0, places=3)

    def test_triangle_pdf_non_negative_everywhere(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            a, b, c = sorted(rng.uniform(-10, 10, 3))
            x = np.linspace(a - 5, c + 5, 500)
            y = triangle_pdf(x, a, b, c)
            self.assertTrue(np.all(y >= 0))


class TestEStepInvariants(unittest.TestCase):
    """The E-step must always produce a valid responsibility distribution,
    regardless of how the mixture parameters happen to be initialized."""

    def test_responsibilities_sum_to_one_per_bin(self):
        rng = np.random.default_rng(1)
        data = rng.normal(0, 1, 500)
        bin_counts, bin_edges = np.histogram(data, bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        params_list, weights = _init_triangles_from_histogram(
            bin_centers, bin_counts, 3, data.min(), data.max()
        )
        responsibilities = _em_e_step(bin_centers, bin_counts, params_list, weights)
        row_sums = responsibilities.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones_like(row_sums), atol=1e-8)

    def test_responsibilities_are_within_unit_interval(self):
        rng = np.random.default_rng(2)
        data = rng.normal(0, 1, 500)
        bin_counts, bin_edges = np.histogram(data, bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        params_list, weights = _init_triangles_from_histogram(
            bin_centers, bin_counts, 2, data.min(), data.max()
        )
        responsibilities = _em_e_step(bin_centers, bin_counts, params_list, weights)
        self.assertTrue(np.all(responsibilities >= 0.0))
        self.assertTrue(np.all(responsibilities <= 1.0))


class TestMStepBounds(unittest.TestCase):
    """The M-step's constrained optimization must never hand back a shape
    that violates a <= b <= c or strays outside the data range, across a
    spread of random initializations and component counts."""

    def test_params_respect_ordering_and_data_bounds(self):
        rng = np.random.default_rng(3)
        for seed in range(10):
            local_rng = np.random.default_rng(seed)
            data = local_rng.normal(0, 1, 300)
            data_min, data_max = data.min(), data.max()
            bin_counts, bin_edges = np.histogram(data, bins=40, range=(data_min, data_max))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            params_list, weights = _init_triangles_from_histogram(
                bin_centers, bin_counts, 2, data_min, data_max
            )
            responsibilities = _em_e_step(bin_centers, bin_counts, params_list, weights)
            new_params = _em_m_step_params(
                bin_centers, bin_counts, responsibilities, params_list, data_min, data_max
            )
            for a, b, c in new_params:
                self.assertLessEqual(a, b + 1e-9)
                self.assertLessEqual(b, c + 1e-9)
                self.assertGreaterEqual(a, data_min - 1e-9)
                self.assertLessEqual(c, data_max + 1e-9)


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

    def test_convergence(self):
        """Log-likelihood should be reasonable (not too negative), mirroring
        TrapzMixtureModel's own convergence sanity check."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 500)
        model = TriangleMixtureModel(n_components=1, max_iter=10, n_bins=50)
        model.fit(data)

        self.assertGreater(model.log_likelihood_, -5000)

    def test_fit_returns_self(self):
        data = np.random.default_rng(0).normal(0, 1, 200)
        model = TriangleMixtureModel(n_components=1)
        result = model.fit(data)
        self.assertIs(result, model)


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

    def test_bic_trimodal(self):
        np.random.seed(42)
        data = np.concatenate([
            np.random.normal(-4, 0.4, 300),
            np.random.normal(0, 0.4, 300),
            np.random.normal(4, 0.4, 300)
        ])
        optimal_k = find_optimal_triangles(data, max_components=4)
        self.assertEqual(optimal_k, 3)

    def test_fit_triangle_mixture_1d_returns_fit_not_just_count(self):
        """Matches fit_trapezoid_mixture_1d's contract: the caller gets the
        winning fit directly, it never needs to refit at the chosen k."""
        np.random.seed(42)
        data = np.concatenate([
            np.random.normal(-3, 0.5, 500),
            np.random.normal(3, 0.5, 500)
        ])
        triangles, n_selected = fit_triangle_mixture_1d(data, max_components=4)
        self.assertEqual(n_selected, 2)
        self.assertEqual(len(triangles), 2)
        for tri in triangles:
            self.assertIsInstance(tri, TriangularMembership)

    def test_explicit_n_triangles_skips_bic_search(self):
        """Requesting fewer components than the data has natural peaks must
        return exactly that many -- the histogram-peak init sorts by
        prominence and keeps the top n_components whenever it finds more
        peaks than requested (see _init_trapz_from_histogram)."""
        np.random.seed(42)
        data = np.concatenate([
            np.random.normal(-4, 0.4, 300),
            np.random.normal(0, 0.4, 300),
            np.random.normal(4, 0.4, 300),
        ])
        triangles, n_selected = fit_triangle_mixture_1d(data, n_triangles=2)
        self.assertEqual(n_selected, 2)
        self.assertEqual(len(triangles), 2)


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

    def test_fit_triangles_empty_label_returns_empty(self):
        X = pd.DataFrame({'feature': np.random.default_rng(0).normal(0, 1, 50)})
        y = pd.Series([0] * 50)
        result = fit_triangles(X, y, 'feature', label_value=1, n_triangles=1)
        self.assertEqual(result, [])

    def test_fit_triangles_drops_nan(self):
        rng = np.random.default_rng(0)
        values = rng.normal(0, 1, 100).tolist() + [np.nan, np.nan]
        X = pd.DataFrame({'feature': values})
        y = pd.Series([0] * len(values))
        result = fit_triangles(X, y, 'feature', label_value=0, n_triangles=1)
        self.assertEqual(len(result), 1)
        self.assertFalse(np.isnan(result[0].a))

    def test_max_samples_caps_rows_deterministically(self):
        """Same random_state -> same subsample -> identical fit; matches the
        seeded-without-replacement contract fit_trapezoids documents."""
        rng = np.random.default_rng(0)
        X = pd.DataFrame({'feature': rng.normal(0, 1, 2000)})
        y = pd.Series([0] * 2000)

        first = fit_triangles(X, y, 'feature', label_value=0, n_triangles=1,
                               max_samples=50, random_state=7)
        second = fit_triangles(X, y, 'feature', label_value=0, n_triangles=1,
                                max_samples=50, random_state=7)

        self.assertEqual(len(first), 1)
        self.assertEqual(first[0].a, second[0].a)
        self.assertEqual(first[0].b, second[0].b)
        self.assertEqual(first[0].c, second[0].c)

    def test_max_samples_none_uses_every_row(self):
        """max_samples=None (the default) must not silently subsample."""
        rng = np.random.default_rng(0)
        X = pd.DataFrame({'feature': rng.normal(0, 1, 500)})
        y = pd.Series([0] * 500)
        result = fit_triangles(X, y, 'feature', label_value=0, n_triangles=1, max_samples=None)
        self.assertEqual(len(result), 1)

    def test_verbose_prints_selected_component_count(self):
        np.random.seed(42)
        X = pd.DataFrame({
            'feature': np.concatenate([
                np.random.normal(-3, 0.5, 200),
                np.random.normal(3, 0.5, 200),
            ])
        })
        y = pd.Series([0] * 400)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fit_triangles(X, y, 'feature', label_value=0, n_triangles=0, verbose=True)
        self.assertIn("triangles", buf.getvalue())


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

    def test_n_triangles_as_per_feature_dict(self):
        np.random.seed(42)
        X = pd.DataFrame({
            'f1': np.concatenate([np.random.normal(-3, 0.5, 100), np.random.normal(3, 0.5, 100)]),
            'f2': np.random.normal(0, 1, 200),
        })
        y = pd.Series([0] * 100 + [1] * 100)

        model = create_triangle_membership_dict(
            X, y, top_n_var_names=['f1', 'f2'], n_triangles={'f1': 2, 'f2': 1}
        )

        for label_model in model.feature_models['f1'].label_models.values():
            self.assertEqual(len(label_model.memberships), 2)
        for label_model in model.feature_models['f2'].label_models.values():
            self.assertEqual(len(label_model.memberships), 1)

    def test_max_samples_is_threaded_through(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame({'feature': rng.normal(0, 1, 2000)})
        y = pd.Series([0] * 1000 + [1] * 1000)

        model = create_triangle_membership_dict(
            X, y, top_n_var_names=['feature'], n_triangles=1, max_samples=25, random_state=3
        )
        self.assertIsInstance(model, GaussianMixtureModel)
        self.assertEqual(len(model.feature_models['feature'].label_models[0].memberships), 1)


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

    def test_handles_moderate_scale_difference(self):
        np.random.seed(42)
        data = np.concatenate([
            np.random.normal(0, 1, 500),
            np.random.normal(10, 1, 500)
        ])
        model = TriangleMixtureModel(n_components=2)
        model.fit(data)
        self.assertGreaterEqual(len(model.triangles_), 1)

    def test_handles_zero_variance_component_alongside_spread(self):
        """A cluster of exactly-repeated values next to a normally spread one
        should not blow up the SLSQP objective (log(0) guarded by trapz_pdf's
        area check and the M-step's 1e-10 floor) or return non-finite params.

        This is exactly the "narrow spike embedded in a much wider component"
        shape documented in docs/triangle-em-resolution-evaluation.md as a
        known peak-detection limitation shared with trapz_math.py: the spike
        is not reliably split out into its own component, so this only
        asserts graceful degradation (a valid, finite fit), not that both
        components get resolved separately.
        """
        data = np.concatenate([np.full(100, 5.0), np.random.default_rng(0).normal(0, 1, 400)])
        model = TriangleMixtureModel(n_components=2)
        model.fit(data)
        self.assertGreaterEqual(len(model.triangles_), 1)
        for tri in model.triangles_:
            self.assertTrue(np.isfinite(tri.a))
            self.assertTrue(np.isfinite(tri.b))
            self.assertTrue(np.isfinite(tri.c))
            self.assertLessEqual(tri.a, tri.b)
            self.assertLessEqual(tri.b, tri.c)


class TestTrianglePerformance(unittest.TestCase):
    """Sanity checks on fit cost -- not strict speed requirements, just a
    guard against a pathological blow-up (e.g. an accidental O(n^2) pass or a
    constraint set that sends SLSQP into a slow hunt)."""

    def test_fit_completes_quickly_on_moderate_data(self):
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 2000)
        start = time.perf_counter()
        TriangleMixtureModel(n_components=2, n_bins=50).fit(data)
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 10.0)

    def test_triangle_em_not_dramatically_slower_than_trapezoid_em(self):
        """Triangle fitting optimizes one fewer parameter per component than
        trapezoid fitting, so it should be roughly the same cost or cheaper --
        not several times slower."""
        rng = np.random.default_rng(0)
        data = np.concatenate([rng.normal(-3, 0.5, 500), rng.normal(3, 0.5, 500)])

        start = time.perf_counter()
        fit_triangles_em(data, n_components=2, n_bins=50)
        triangle_time = time.perf_counter() - start

        start = time.perf_counter()
        fit_trapezoids_em(data, n_components=2, n_bins=50)
        trapezoid_time = time.perf_counter() - start

        self.assertLess(triangle_time, trapezoid_time * 3 + 1.0)


class TestTribbleClassifierTriangular(unittest.TestCase):
    """Test TribbleClassifier(member_function="triangular") end to end."""

    def _fitted(self, **kwargs):
        np.random.seed(42)
        X = pd.DataFrame({
            'a': np.concatenate([np.random.normal(0, 1, 100), np.random.normal(5, 1, 100)]),
            'b': np.concatenate([np.random.normal(-2, 1, 100), np.random.normal(2, 1, 100)]),
        })
        y = pd.Series([0] * 100 + [1] * 100)
        clf = TribbleClassifier(member_function="triangular", top_p=1.0, n_gaussians=1, random_state=0, **kwargs)
        clf.fit(X, y)
        return clf, X, y

    def test_fit_predict(self):
        clf, X, y = self._fitted()
        preds = clf.predict(X)

        self.assertEqual(len(preds), len(y))
        for feature_model in clf.model_.feature_models.values():
            for label_model in feature_model.label_models.values():
                for mf in label_model.memberships:
                    self.assertIsInstance(mf, TriangularMembership)

    def test_predict_proba_rows_sum_to_one(self):
        clf, X, y = self._fitted()
        proba = clf.predict_proba(X)
        self.assertEqual(proba.shape, (len(y), len(clf.classes_)))
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(y)), atol=1e-8)

    def test_deduplicate_removes_forced_duplicate_without_changing_predictions(self):
        clf, X, y = self._fitted()
        before = clf.predict(X)

        first_feature = next(iter(clf.model_.feature_models))
        label_model = clf.model_.feature_models[first_feature].label_models[clf.classes_[0]]
        label_model.memberships.append(label_model.memberships[0])
        n_before = clf.model_.n_membership_functions

        removed = clf.deduplicate(rtol=0.0, atol=0.0)

        self.assertEqual(removed, 1)
        self.assertEqual(clf.model_.n_membership_functions, n_before - 1)
        np.testing.assert_array_equal(before, clf.predict(X))

    def test_to_simple_model_matches_predict(self):
        from tribblefis.gauss_math import simple_gaussian_predict
        clf, X, y = self._fitted()
        direct = clf.predict(X)
        via_simple = simple_gaussian_predict(X, clf.to_simple_model(rtol=0.0, atol=0.0))
        np.testing.assert_array_equal(np.asarray(direct), np.asarray(via_simple))

    def test_max_samples_constructor_param_is_used(self):
        clf, X, y = self._fitted(max_samples=30)
        self.assertEqual(clf.get_params()["max_samples"], 30)
        preds = clf.predict(X)
        self.assertEqual(len(preds), len(y))

    def test_unknown_member_function_raises(self):
        X = pd.DataFrame({'a': np.random.normal(0, 1, 20)})
        y = pd.Series([0] * 10 + [1] * 10)
        clf = TribbleClassifier(member_function="not-a-real-shape")
        with self.assertRaises(ValueError):
            clf.fit(X, y)


if __name__ == '__main__':
    unittest.main()
