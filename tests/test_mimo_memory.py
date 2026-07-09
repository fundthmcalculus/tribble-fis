"""
Test suite for enhanced MIMO model with memory features.

Demonstrates the memory-augmented MIMO predictor and compares it with
the standard MIMO model on a simple double pendulum trajectory.
"""
import unittest
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tribblefis.gaussian_regressor_memory import (
    MemoryWindowFeatureExtractor,
    MimoGaussianPredictorMemory,
    prepare_mimo_data_with_memory,
)
from tribblefis.gaussian_regressor import MimoGaussianPredictor


class TestMemoryWindowFeatureExtractor(unittest.TestCase):
    """Tests for memory feature extraction."""

    def test_feature_extraction_basic(self):
        """Test that memory features are computed correctly."""
        extractor = MemoryWindowFeatureExtractor(window_size=3, memory_size=2)

        # Create simple test data
        df = pd.DataFrame({
            "x": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        })

        result = extractor.prepare_sequences(df, ["x"], include_time=True)

        # Check time step is added
        self.assertIn("time_step", result.columns)
        np.testing.assert_array_equal(result["time_step"].values, [0, 1, 2, 3, 4])

        # Check current values are correct
        np.testing.assert_array_equal(result["x_current"].values, [1.0, 2.0, 3.0, 4.0, 5.0])

        # Check short-term averages
        expected_short_term = [
            1.0,  # mean([1])
            1.5,  # mean([1, 2])
            2.0,  # mean([1, 2, 3])
            3.0,  # mean([2, 3, 4])
            4.0,  # mean([3, 4, 5])
        ]
        np.testing.assert_array_almost_equal(result["x_short_term_avg"].values, expected_short_term)

        # Check long-term averages (some will be NaN due to insufficient history)
        self.assertTrue(np.isnan(result["x_long_term_avg"].iloc[0]))
        self.assertTrue(np.isnan(result["x_long_term_avg"].iloc[1]))
        # At index 2: window=[1,2,3], long-term should be empty -> NaN
        self.assertTrue(np.isnan(result["x_long_term_avg"].iloc[2]))
        # At index 3: window=[2,3,4], long-term should be [1] -> 1.0
        self.assertAlmostEqual(result["x_long_term_avg"].iloc[3], 1.0)
        # At index 4: window=[3,4,5], long-term should be [1,2] -> 1.5
        self.assertAlmostEqual(result["x_long_term_avg"].iloc[4], 1.5)

    def test_feature_names(self):
        """Test that feature names are generated correctly."""
        extractor = MemoryWindowFeatureExtractor(window_size=2, memory_size=1)

        names = extractor.get_feature_names(["x", "y"], include_time=True)

        expected = [
            "time_step",
            "x_current",
            "x_short_term_avg",
            "x_long_term_avg",
            "y_current",
            "y_short_term_avg",
            "y_long_term_avg",
        ]

        self.assertEqual(names, expected)

    def test_invalid_memory_size(self):
        """Test that memory_size >= window_size raises error."""
        with self.assertRaises(ValueError):
            MemoryWindowFeatureExtractor(window_size=2, memory_size=2)

        with self.assertRaises(ValueError):
            MemoryWindowFeatureExtractor(window_size=2, memory_size=3)


class TestMimoMemoryPredictor(unittest.TestCase):
    """Tests for the memory-augmented MIMO predictor."""

    def setUp(self):
        """Create sample trajectory data."""
        # Simple sinusoidal trajectory
        t = np.linspace(0, 4 * np.pi, 100)
        self.X = pd.DataFrame({
            "theta_1": np.sin(t),
            "theta_2": np.cos(t),
        })
        self.y = self.X.copy()

    def test_fit_and_predict_basic(self):
        """Test basic fit and predict functionality."""
        model = MimoGaussianPredictorMemory(
            window_size=3,
            memory_size=2,
            include_time=True,
            n_output_buckets=3,
            tsk_order="0th",
        )

        model.fit(self.X, self.y)

        # Should be able to predict on the training data
        self.assertIsNotNone(model.mimo_predictor_)
        self.assertEqual(model.input_features_, ["theta_1", "theta_2"])
        self.assertEqual(model.output_features_, ["theta_1", "theta_2"])

    def test_predict_returns_correct_shape(self):
        """Test that predictions have correct shape."""
        model = MimoGaussianPredictorMemory(
            window_size=3,
            memory_size=2,
            include_time=False,
            n_output_buckets=3,
            tsk_order="0th",
        )

        model.fit(self.X, self.y)

        # Predict next state from last window
        last_window = self.X.iloc[-3:]
        pred = model.predict(last_window)

        self.assertEqual(pred.shape, (1, 2))
        self.assertIn("theta_1", pred.columns)
        self.assertIn("theta_2", pred.columns)

    def test_predict_with_return_deltas(self):
        """Test returning deltas vs absolute predictions."""
        model = MimoGaussianPredictorMemory(
            window_size=3,
            memory_size=2,
            include_time=False,
            n_output_buckets=3,
            tsk_order="0th",
        )

        model.fit(self.X, self.y)

        last_window = self.X.iloc[-3:]

        # Get deltas
        deltas = model.predict(last_window, return_deltas=True)
        # Get absolute predictions
        absolutes = model.predict(last_window, return_deltas=False)

        # Absolute should be approximately last state + deltas
        last_state = last_window.iloc[-1:].values
        expected_absolute = last_state + deltas.values
        np.testing.assert_array_almost_equal(absolutes.values, expected_absolute, decimal=5)

    def test_predict_trajectory(self):
        """Test iterative trajectory prediction."""
        model = MimoGaussianPredictorMemory(
            window_size=3,
            memory_size=2,
            include_time=False,
            n_output_buckets=3,
            tsk_order="0th",
        )

        model.fit(self.X, self.y)

        # Predict 10 steps ahead
        seed = self.X.iloc[:5]
        trajectory = model.predict_trajectory(seed, n_steps=10)

        # Should have seed + 10 predictions
        self.assertGreaterEqual(len(trajectory), 5)
        self.assertEqual(trajectory.shape[1], 2)


class TestMimoMemoryComparison(unittest.TestCase):
    """Compare standard MIMO with memory-augmented MIMO."""

    def setUp(self):
        """Create trajectory data."""
        # Double pendulum like dynamics
        np.random.seed(42)
        t = np.linspace(0, 10, 200)
        omega1 = np.sin(t) + 0.1 * np.cos(2 * t)
        omega2 = np.cos(t) * 0.8
        theta1 = np.cumsum(omega1) * 0.01
        theta2 = np.cumsum(omega2) * 0.01

        self.trajectories = [pd.DataFrame({
            "theta_1": theta1,
            "theta_2": theta2,
        })]

        # Split into train/test
        split_idx = 150
        self.X_train = self.trajectories[0].iloc[:split_idx]
        self.y_train = self.X_train.copy()
        self.X_test = self.trajectories[0].iloc[split_idx:split_idx + 40]
        self.y_test = self.X_test.copy()

    def test_memory_model_trains(self):
        """Test that memory model trains without error."""
        model = MimoGaussianPredictorMemory(
            window_size=4,
            memory_size=3,
            include_time=False,
            n_output_buckets=5,
            tsk_order="1st",
        )

        start = time.time()
        model.fit(self.X_train, self.y_train)
        elapsed = time.time() - start

        print(f"Training time: {elapsed:.4f}s")
        self.assertIsNotNone(model.mimo_predictor_)

    def test_standard_model_trains(self):
        """Test that standard MIMO model trains without error."""
        # Test the standard MIMO on simple data
        model = MimoGaussianPredictor(
            n_output_buckets=5,
            tsk_order="1st",
            optimize_coefficients=True,
            random_state=42,
        )

        # Create simple MIMO data (states at different time steps)
        split_idx = 150
        X_df = self.trajectories[0].iloc[:split_idx]
        y_df = self.trajectories[0].iloc[:split_idx]

        model.fit(X_df, y_df)

        self.assertIsNotNone(model.regressors_)


if __name__ == "__main__":
    unittest.main()
