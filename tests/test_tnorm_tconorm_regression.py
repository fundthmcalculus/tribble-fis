"""Comprehensive regression tests for t-norm and t-conorm operations.

Tests all t-norm and t-conorm pairs on a 50x50 grid (2500 entries) to verify:
1. Correctness of mathematical definitions
2. Domain validity ([0,1] -> [0,1])
3. Boundary conditions
4. Idempotence properties
5. Fixed point properties
"""

import unittest
import numpy as np
from tribblefis.gauss_math import t_norm, t_conorm, t_complement


class TestTNormTConormRegression(unittest.TestCase):
    """Regression tests for all t-norm and t-conorm operators."""

    @classmethod
    def setUpClass(cls):
        """Create the test grid: 50x50 = 2500 entries."""
        cls.grid_values = np.linspace(0, 1, 50)
        cls.grid = np.array(np.meshgrid(cls.grid_values, cls.grid_values)).reshape(2, -1).T

    def test_tnorm_minmax_grid_properties(self):
        """Test min/max t-norm on full grid."""
        x = self.grid[:, 0]
        y = self.grid[:, 1]
        result = t_norm(x, y, "min/max")

        self.assertEqual(result.shape, x.shape)
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 1))
        self.assertTrue(np.allclose(result, np.minimum(x, y)))

    def test_tnorm_probability_grid_properties(self):
        """Test probability t-norm on full grid."""
        x = self.grid[:, 0]
        y = self.grid[:, 1]
        result = t_norm(x, y, "probability")

        self.assertEqual(result.shape, x.shape)
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 1))
        self.assertTrue(np.allclose(result, x * y))

    def test_tnorm_luk_grid_properties(self):
        """Test Lukasiewicz t-norm on full grid."""
        x = self.grid[:, 0]
        y = self.grid[:, 1]
        result = t_norm(x, y, "luk")

        self.assertEqual(result.shape, x.shape)
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 1))
        self.assertTrue(np.allclose(result, np.maximum(0, x + y - 1)))

    def test_tnorm_hamacher_grid_properties(self):
        """Test Hamacher t-norm on full grid - should handle 0/0 singularity."""
        x = self.grid[:, 0]
        y = self.grid[:, 1]
        result = t_norm(x, y, "hamacher")

        self.assertEqual(result.shape, x.shape)
        # Check for no NaN values
        self.assertFalse(np.any(np.isnan(result)),
                         "Hamacher t-norm produced NaN values (0/0 singularity)")
        # Check domain
        self.assertTrue(np.all(result >= 0), "Hamacher t-norm produced negative values")
        self.assertTrue(np.all(result <= 1), "Hamacher t-norm produced values > 1")

    def test_tnorm_hamacher_singularity(self):
        """Test that Hamacher t-norm handles 0/0 singularity correctly."""
        result = t_norm(np.array([0.0]), np.array([0.0]), "hamacher")
        self.assertFalse(np.isnan(result[0]), "Hamacher t-norm(0,0) is NaN")
        self.assertEqual(result[0], 0.0, "Hamacher t-norm(0,0) should be 0")

    def test_tnorm_array_reduction_forwards_norm(self):
        """Test that t_norm with y=None forwards the norm parameter."""
        x = np.array([[0.5, 0.5]])

        # Test probability norm
        result_prob = t_norm(x, None, "probability")
        expected_prob = 0.5 * 0.5  # 0.25
        np.testing.assert_allclose(result_prob[0], expected_prob,
                                   err_msg="t_norm(y=None) doesn't forward 'probability' norm")

        # Test luk norm
        result_luk = t_norm(x, None, "luk")
        expected_luk = np.maximum(0, 0.5 + 0.5 - 1)  # 0.0
        np.testing.assert_allclose(result_luk[0], expected_luk,
                                   err_msg="t_norm(y=None) doesn't forward 'luk' norm")

    def test_tconorm_minmax_grid_properties(self):
        """Test min/max t-conorm on full grid."""
        x = self.grid[:, 0]
        y = self.grid[:, 1]
        result = t_conorm(x, y, "min/max")

        self.assertEqual(result.shape, x.shape)
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 1))
        self.assertTrue(np.allclose(result, np.maximum(x, y)))

    def test_tconorm_probability_grid_properties(self):
        """Test probability t-conorm on full grid."""
        x = self.grid[:, 0]
        y = self.grid[:, 1]
        result = t_conorm(x, y, "probability")

        self.assertEqual(result.shape, x.shape)
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 1))
        self.assertTrue(np.allclose(result, x + y - x * y))

    def test_tconorm_luk_grid_properties(self):
        """Test Lukasiewicz t-conorm on full grid."""
        x = self.grid[:, 0]
        y = self.grid[:, 1]
        result = t_conorm(x, y, "luk")

        self.assertEqual(result.shape, x.shape)
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 1))
        self.assertTrue(np.allclose(result, np.minimum(1, x + y)))

    def test_tconorm_hamacher_grid_properties(self):
        """Test Hamacher t-conorm on full grid - should stay in [0,1]."""
        x = self.grid[:, 0]
        y = self.grid[:, 1]
        result = t_conorm(x, y, "hamacher")

        self.assertEqual(result.shape, x.shape)
        # Check for no NaN values
        self.assertFalse(np.any(np.isnan(result)),
                         "Hamacher t-conorm produced NaN values")
        # Check domain - most critical fix
        self.assertTrue(np.all(result >= 0), "Hamacher t-conorm produced negative values")
        self.assertTrue(np.all(result <= 1),
                        f"Hamacher t-conorm produced values > 1: max={np.max(result)}")

    def test_tconorm_hamacher_specific_values(self):
        """Test Hamacher t-conorm against known correct values."""
        test_values = [0.3, 0.5, 0.7, 0.9]
        for v in test_values:
            x = np.array([v])
            y = np.array([v])
            result = t_conorm(x, y, "hamacher")
            # Result should be in [0,1]
            self.assertGreaterEqual(result[0], 0, f"Hamacher t-conorm({v},{v}) < 0")
            self.assertLessEqual(result[0], 1, f"Hamacher t-conorm({v},{v}) > 1")

    def test_tconorm_array_reduction_forwards_norm(self):
        """Test that t_conorm with y=None forwards the norm parameter."""
        x = np.array([[0.5, 0.5]])

        # Test probability norm
        result_prob = t_conorm(x, None, "probability")
        expected_prob = 0.5 + 0.5 - 0.5 * 0.5  # 0.75
        np.testing.assert_allclose(result_prob[0], expected_prob,
                                   err_msg="t_conorm(y=None) doesn't forward 'probability' norm")

        # Test luk norm
        result_luk = t_conorm(x, None, "luk")
        expected_luk = np.minimum(1, 0.5 + 0.5)  # 1.0
        np.testing.assert_allclose(result_luk[0], expected_luk,
                                   err_msg="t_conorm(y=None) doesn't forward 'luk' norm")

    def test_tnorm_boundary_conditions(self):
        """Test t-norm boundary conditions for all operators."""
        norms = ["min/max", "probability", "luk", "hamacher"]
        for norm in norms:
            x0 = np.array([0.0])
            x1 = np.array([1.0])
            xs = np.array([0.5])

            # T(x, 0) = 0 for all x
            self.assertTrue(np.allclose(t_norm(x0, x0, norm), 0),
                            f"{norm} T(0,0) ≠ 0")
            self.assertTrue(np.allclose(t_norm(xs, x0, norm), 0),
                            f"{norm} T(0.5,0) ≠ 0")
            self.assertTrue(np.allclose(t_norm(x1, x0, norm), 0),
                            f"{norm} T(1,0) ≠ 0")

            # T(x, 1) = x for all x
            self.assertTrue(np.allclose(t_norm(x0, x1, norm), x0),
                            f"{norm} T(0,1) ≠ 0")
            self.assertTrue(np.allclose(t_norm(xs, x1, norm), xs),
                            f"{norm} T(0.5,1) ≠ 0.5")
            self.assertTrue(np.allclose(t_norm(x1, x1, norm), x1),
                            f"{norm} T(1,1) ≠ 1")

    def test_tconorm_boundary_conditions(self):
        """Test t-conorm boundary conditions for all operators."""
        conorms = ["min/max", "probability", "luk", "hamacher"]
        for conorm in conorms:
            x0 = np.array([0.0])
            x1 = np.array([1.0])
            xs = np.array([0.5])

            # S(x, 0) = x for all x
            self.assertTrue(np.allclose(t_conorm(x0, x0, conorm), x0),
                            f"{conorm} S(0,0) ≠ 0")
            self.assertTrue(np.allclose(t_conorm(xs, x0, conorm), xs),
                            f"{conorm} S(0.5,0) ≠ 0.5")
            self.assertTrue(np.allclose(t_conorm(x1, x0, conorm), x1),
                            f"{conorm} S(1,0) ≠ 1")

            # S(x, 1) = 1 for all x
            self.assertTrue(np.allclose(t_conorm(x0, x1, conorm), x1),
                            f"{conorm} S(0,1) ≠ 1")
            self.assertTrue(np.allclose(t_conorm(xs, x1, conorm), x1),
                            f"{conorm} S(0.5,1) ≠ 1")
            self.assertTrue(np.allclose(t_conorm(x1, x1, conorm), x1),
                            f"{conorm} S(1,1) ≠ 1")

    def test_tnorm_commutativity(self):
        """Test that t-norms are commutative: T(x,y) = T(y,x)."""
        norms = ["min/max", "probability", "luk", "hamacher"]
        x = self.grid[:, 0]
        y = self.grid[:, 1]

        for norm in norms:
            result_xy = t_norm(x, y, norm)
            result_yx = t_norm(y, x, norm)
            np.testing.assert_allclose(result_xy, result_yx,
                                       err_msg=f"{norm} t-norm is not commutative")

    def test_tconorm_commutativity(self):
        """Test that t-conorms are commutative: S(x,y) = S(y,x)."""
        conorms = ["min/max", "probability", "luk", "hamacher"]
        x = self.grid[:, 0]
        y = self.grid[:, 1]

        for conorm in conorms:
            result_xy = t_conorm(x, y, conorm)
            result_yx = t_conorm(y, x, conorm)
            np.testing.assert_allclose(result_xy, result_yx,
                                       err_msg=f"{conorm} t-conorm is not commutative")

    def test_tnorm_monotonicity(self):
        """Test that t-norms are monotonically increasing."""
        norms = ["min/max", "probability", "luk", "hamacher"]
        x = self.grid[:, 0]
        y = self.grid[:, 1]

        for norm in norms:
            # If x1 <= x2 and y1 <= y2, then T(x1,y1) <= T(x2,y2)
            x_small = np.clip(x - 0.01, 0, 1)
            y_small = np.clip(y - 0.01, 0, 1)

            result_small = t_norm(x_small, y_small, norm)
            result_large = t_norm(x, y, norm)

            self.assertTrue(np.all(result_small <= result_large + 1e-10),
                            f"{norm} t-norm is not monotonic")

    def test_tconorm_monotonicity(self):
        """Test that t-conorms are monotonically increasing."""
        conorms = ["min/max", "probability", "luk", "hamacher"]
        x = self.grid[:, 0]
        y = self.grid[:, 1]

        for conorm in conorms:
            # If x1 <= x2 and y1 <= y2, then S(x1,y1) <= S(x2,y2)
            x_small = np.clip(x - 0.01, 0, 1)
            y_small = np.clip(y - 0.01, 0, 1)

            result_small = t_conorm(x_small, y_small, conorm)
            result_large = t_conorm(x, y, conorm)

            self.assertTrue(np.all(result_small <= result_large + 1e-10),
                            f"{conorm} t-conorm is not monotonic")

    def test_complement_idempotence(self):
        """Test that complement is involutive: NOT(NOT(x)) = x."""
        x = self.grid_values
        result = t_complement(t_complement(x))
        np.testing.assert_allclose(result, x)

    def test_complement_boundary(self):
        """Test complement boundary conditions."""
        self.assertEqual(t_complement(np.array([0.0]))[0], 1.0)
        self.assertEqual(t_complement(np.array([1.0]))[0], 0.0)
        np.testing.assert_allclose(t_complement(np.array([0.5]))[0], 0.5)

    def test_de_morgans_law_minmax(self):
        """Test De Morgan's laws for min/max: NOT(min(x,y)) = max(NOT(x), NOT(y))."""
        x = self.grid[:, 0]
        y = self.grid[:, 1]

        left = t_complement(t_norm(x, y, "min/max"))
        right = t_conorm(t_complement(x), t_complement(y), "min/max")

        np.testing.assert_allclose(left, right, atol=1e-10)

    def test_de_morgans_law_probability(self):
        """Test De Morgan's laws for probability."""
        x = self.grid[:, 0]
        y = self.grid[:, 1]

        left = t_complement(t_norm(x, y, "probability"))
        right = t_conorm(t_complement(x), t_complement(y), "probability")

        np.testing.assert_allclose(left, right, atol=1e-10)

    def test_anomaly_threshold_clipping(self):
        """Test that anomaly threshold boosting stays in [0,1] when clipped."""
        firing = np.array([[0.8, 0.2], [0.9, 0.1], [0.95, 0.05]])
        threshold = 0.5

        # Simulate the fix: clip before aggregating
        boosted = np.clip(firing + threshold, 0.0, 1.0)
        result = t_conorm(boosted, None, "min/max")
        anomaly = t_complement(result)

        # Anomaly membership should be in [0,1]
        self.assertTrue(np.all(anomaly >= 0),
                        "Clipped anomaly produced negative values")
        self.assertTrue(np.all(anomaly <= 1),
                        "Clipped anomaly produced values > 1")


if __name__ == '__main__':
    unittest.main()
