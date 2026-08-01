"""Comprehensive regression tests for t-norm and t-conorm operations.

Tests all t-norm and t-conorm pairs on a 50x50 grid (2500 entries) to verify:
1. Correctness of mathematical definitions for all operators
2. Domain validity ([0,1] -> [0,1]) for all cases
3. Boundary conditions for all operators
4. Singularity/edge case handling for all operators
5. Parameter forwarding in array-reduction branches
"""

import unittest
import numpy as np
from tribblefis.gauss_math import t_norm, t_conorm, t_complement


class TestTNormTConormRegression(unittest.TestCase):
    """Regression tests for all t-norm and t-conorm operators."""

    # All available operators
    NORMS = ["min/max", "probability", "luk", "hamacher"]
    CONORMS = ["min/max", "probability", "luk", "hamacher"]

    @classmethod
    def setUpClass(cls):
        """Create the test grid: 50x50 = 2500 entries."""
        cls.grid_values = np.linspace(0, 1, 50)
        cls.grid = np.array(np.meshgrid(cls.grid_values, cls.grid_values)).reshape(2, -1).T

        # Edge case values for boundary testing
        cls.edge_cases = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.5, 0.5],
            [0.0, 0.5],
            [0.5, 0.0],
            [1.0, 0.5],
            [0.5, 1.0],
        ])

    def test_tnorm_grid_properties_all_operators(self):
        """Test all t-norms on full 50x50 grid (2500 entries) for domain validity."""
        x = self.grid[:, 0]
        y = self.grid[:, 1]

        for norm in self.NORMS:
            with self.subTest(norm=norm):
                result = t_norm(x, y, norm)

                # Check shape and no NaN
                self.assertEqual(result.shape, x.shape)
                self.assertFalse(np.any(np.isnan(result)),
                                f"{norm} t-norm produced NaN values")

                # Check domain [0,1]
                self.assertTrue(np.all(result >= -1e-10),
                               f"{norm} t-norm produced negative values")
                self.assertTrue(np.all(result <= 1 + 1e-10),
                               f"{norm} t-norm produced values > 1")

    def test_tnorm_edge_cases_all_operators(self):
        """Test all t-norms on edge cases for singularity and boundary handling."""
        for norm in self.NORMS:
            with self.subTest(norm=norm):
                for x_val, y_val in self.edge_cases:
                    result = t_norm(np.array([x_val]), np.array([y_val]), norm)

                    # No NaN or Inf
                    self.assertFalse(np.isnan(result[0]),
                                    f"{norm} T({x_val},{y_val}) is NaN")
                    self.assertFalse(np.isinf(result[0]),
                                    f"{norm} T({x_val},{y_val}) is Inf")

                    # Domain check
                    self.assertGreaterEqual(result[0], -1e-10,
                                          f"{norm} T({x_val},{y_val}) < 0")
                    self.assertLessEqual(result[0], 1 + 1e-10,
                                        f"{norm} T({x_val},{y_val}) > 1")

    def test_tconorm_grid_properties_all_operators(self):
        """Test all t-conorms on full 50x50 grid (2500 entries) for domain validity."""
        x = self.grid[:, 0]
        y = self.grid[:, 1]

        for conorm in self.CONORMS:
            with self.subTest(conorm=conorm):
                result = t_conorm(x, y, conorm)

                # Check shape and no NaN
                self.assertEqual(result.shape, x.shape)
                self.assertFalse(np.any(np.isnan(result)),
                                f"{conorm} t-conorm produced NaN values")

                # Check domain [0,1] with floating point tolerance
                self.assertTrue(np.all(result >= -1e-10),
                               f"{conorm} t-conorm produced negative values")
                self.assertTrue(np.all(result <= 1 + 1e-10),
                               f"{conorm} t-conorm produced values > 1: max={np.max(result)}")

    def test_tconorm_edge_cases_all_operators(self):
        """Test all t-conorms on edge cases for singularity and boundary handling."""
        for conorm in self.CONORMS:
            with self.subTest(conorm=conorm):
                for x_val, y_val in self.edge_cases:
                    result = t_conorm(np.array([x_val]), np.array([y_val]), conorm)

                    # No NaN or Inf
                    self.assertFalse(np.isnan(result[0]),
                                    f"{conorm} S({x_val},{y_val}) is NaN")
                    self.assertFalse(np.isinf(result[0]),
                                    f"{conorm} S({x_val},{y_val}) is Inf")

                    # Domain check
                    self.assertGreaterEqual(result[0], -1e-10,
                                          f"{conorm} S({x_val},{y_val}) < 0")
                    self.assertLessEqual(result[0], 1 + 1e-10,
                                        f"{conorm} S({x_val},{y_val}) > 1")

    def test_tnorm_array_reduction_forwards_norm_all_operators(self):
        """Test that t_norm with y=None forwards norm parameter for all operators."""
        test_cases = [
            ("min/max", np.array([[0.3, 0.7]]), 0.3),  # min
            ("probability", np.array([[0.5, 0.5]]), 0.25),
            ("luk", np.array([[0.5, 0.5]]), 0.0),  # max(0, 0.5+0.5-1)
            ("hamacher", np.array([[0.5, 0.5]]), 0.33333333),
        ]

        for norm, x, expected in test_cases:
            with self.subTest(norm=norm):
                result = t_norm(x, None, norm)
                np.testing.assert_allclose(result[0], expected, rtol=1e-5,
                                          err_msg=f"t_norm(y=None, {norm}) doesn't forward norm")

    def test_tconorm_array_reduction_forwards_norm_all_operators(self):
        """Test that t_conorm with y=None forwards norm parameter for all operators."""
        test_cases = [
            ("min/max", np.array([[0.3, 0.7]]), 0.7),  # max
            ("probability", np.array([[0.5, 0.5]]), 0.75),  # 0.5+0.5-0.25
            ("luk", np.array([[0.5, 0.5]]), 1.0),  # min(1, 0.5+0.5)
            ("hamacher", np.array([[0.5, 0.5]]), 0.66666667),
        ]

        for conorm, x, expected in test_cases:
            with self.subTest(conorm=conorm):
                result = t_conorm(x, None, conorm)
                np.testing.assert_allclose(result[0], expected, rtol=1e-5,
                                          err_msg=f"t_conorm(y=None, {conorm}) doesn't forward norm")

    def test_tnorm_boundary_conditions_all_operators(self):
        """Test t-norm boundary conditions for all operators.

        Must satisfy:
        - T(x, 0) = 0 for all x
        - T(x, 1) = x for all x
        """
        for norm in self.NORMS:
            with self.subTest(norm=norm):
                for x_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
                    x = np.array([x_val])

                    # T(x, 0) = 0
                    result_x0 = t_norm(x, np.array([0.0]), norm)
                    np.testing.assert_allclose(result_x0[0], 0.0, atol=1e-10,
                                              err_msg=f"{norm} T({x_val},0) ≠ 0")

                    # T(x, 1) = x
                    result_x1 = t_norm(x, np.array([1.0]), norm)
                    np.testing.assert_allclose(result_x1[0], x_val, atol=1e-10,
                                              err_msg=f"{norm} T({x_val},1) ≠ {x_val}")

    def test_tconorm_boundary_conditions_all_operators(self):
        """Test t-conorm boundary conditions for all operators.

        Must satisfy:
        - S(x, 0) = x for all x
        - S(x, 1) = 1 for all x
        """
        for conorm in self.CONORMS:
            with self.subTest(conorm=conorm):
                for x_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
                    x = np.array([x_val])

                    # S(x, 0) = x
                    result_x0 = t_conorm(x, np.array([0.0]), conorm)
                    np.testing.assert_allclose(result_x0[0], x_val, atol=1e-10,
                                              err_msg=f"{conorm} S({x_val},0) ≠ {x_val}")

                    # S(x, 1) = 1
                    result_x1 = t_conorm(x, np.array([1.0]), conorm)
                    np.testing.assert_allclose(result_x1[0], 1.0, atol=1e-10,
                                              err_msg=f"{conorm} S({x_val},1) ≠ 1")

    def test_tnorm_commutativity_all_operators(self):
        """Test that t-norms are commutative: T(x,y) = T(y,x)."""
        x = self.grid[:, 0]
        y = self.grid[:, 1]

        for norm in self.NORMS:
            with self.subTest(norm=norm):
                result_xy = t_norm(x, y, norm)
                result_yx = t_norm(y, x, norm)
                np.testing.assert_allclose(result_xy, result_yx, rtol=1e-10,
                                           err_msg=f"{norm} t-norm is not commutative")

    def test_tconorm_commutativity_all_operators(self):
        """Test that t-conorms are commutative: S(x,y) = S(y,x)."""
        x = self.grid[:, 0]
        y = self.grid[:, 1]

        for conorm in self.CONORMS:
            with self.subTest(conorm=conorm):
                result_xy = t_conorm(x, y, conorm)
                result_yx = t_conorm(y, x, conorm)
                np.testing.assert_allclose(result_xy, result_yx, rtol=1e-10,
                                           err_msg=f"{conorm} t-conorm is not commutative")

    def test_tnorm_monotonicity_all_operators(self):
        """Test that t-norms are monotonically increasing for all operators."""
        x = self.grid[:, 0]
        y = self.grid[:, 1]

        for norm in self.NORMS:
            with self.subTest(norm=norm):
                # If x1 <= x2 and y1 <= y2, then T(x1,y1) <= T(x2,y2)
                x_small = np.clip(x - 0.01, 0, 1)
                y_small = np.clip(y - 0.01, 0, 1)

                result_small = t_norm(x_small, y_small, norm)
                result_large = t_norm(x, y, norm)

                self.assertTrue(np.all(result_small <= result_large + 1e-10),
                                f"{norm} t-norm is not monotonic")

    def test_tconorm_monotonicity_all_operators(self):
        """Test that t-conorms are monotonically increasing for all operators."""
        x = self.grid[:, 0]
        y = self.grid[:, 1]

        for conorm in self.CONORMS:
            with self.subTest(conorm=conorm):
                # If x1 <= x2 and y1 <= y2, then S(x1,y1) <= S(x2,y2)
                x_small = np.clip(x - 0.01, 0, 1)
                y_small = np.clip(y - 0.01, 0, 1)

                result_small = t_conorm(x_small, y_small, conorm)
                result_large = t_conorm(x, y, conorm)

                self.assertTrue(np.all(result_small <= result_large + 1e-10),
                                f"{conorm} t-conorm is not monotonic")

    def test_complement_idempotence(self):
        """Test that complement is involutive: NOT(NOT(x)) = x for all x."""
        x = self.grid_values
        result = t_complement(t_complement(x))
        np.testing.assert_allclose(result, x)

    def test_complement_boundary(self):
        """Test complement boundary conditions."""
        self.assertEqual(t_complement(np.array([0.0]))[0], 1.0)
        self.assertEqual(t_complement(np.array([1.0]))[0], 0.0)
        np.testing.assert_allclose(t_complement(np.array([0.5]))[0], 0.5)

    def test_de_morgans_law_all_operators(self):
        """Test De Morgan's laws for all operator pairs.

        For dual pairs (same operator name), NOT(T(x,y)) should equal S(NOT(x), NOT(y))
        """
        x = self.grid[:, 0]
        y = self.grid[:, 1]

        for norm in self.NORMS:
            with self.subTest(pair=norm):
                # De Morgan's law: NOT(T(x,y)) = S(NOT(x), NOT(y))
                left = t_complement(t_norm(x, y, norm))
                right = t_conorm(t_complement(x), t_complement(y), norm)

                np.testing.assert_allclose(left, right, atol=1e-10,
                                          err_msg=f"De Morgan's law fails for {norm}")

    def test_anomaly_threshold_clipping_all_norms(self):
        """Test that anomaly threshold boosting stays in [0,1] when clipped for all norms."""
        firing = np.array([[0.8, 0.2], [0.9, 0.1], [0.95, 0.05]])
        threshold = 0.5

        for norm in self.NORMS:
            with self.subTest(norm=norm):
                # Simulate the fix: clip before aggregating
                boosted = np.clip(firing + threshold, 0.0, 1.0)
                result = t_conorm(boosted, None, norm)
                anomaly = t_complement(result)

                # Anomaly membership should be in [0,1]
                self.assertTrue(np.all(anomaly >= -1e-10),
                                f"{norm}: Clipped anomaly produced negative values")
                self.assertTrue(np.all(anomaly <= 1 + 1e-10),
                                f"{norm}: Clipped anomaly produced values > 1")

    def test_hamacher_specific_values(self):
        """Test Hamacher operators against known correct mathematical values."""
        # From issue #24: Hamacher t-norm(0,0) should be 0, not NaN
        result = t_norm(np.array([0.0]), np.array([0.0]), "hamacher")
        self.assertEqual(result[0], 0.0, "Hamacher t-norm(0,0) should be 0, not NaN")

        # From issue #23: Hamacher t-conorm values should be in [0,1]
        # S(0.5, 0.5) = (0.5+0.5-2*0.25)/(1-0.25) = 0.5/0.75 = 0.6667
        result = t_conorm(np.array([0.5]), np.array([0.5]), "hamacher")
        np.testing.assert_allclose(result[0], 0.66666667, rtol=1e-5)

        # Test the values from issue #23
        test_values = [0.3, 0.5, 0.7, 0.9]
        for v in test_values:
            result = t_conorm(np.array([v]), np.array([v]), "hamacher")
            self.assertGreaterEqual(result[0], 0, f"Hamacher S({v},{v}) < 0")
            self.assertLessEqual(result[0], 1, f"Hamacher S({v},{v}) > 1")


if __name__ == '__main__':
    unittest.main()
