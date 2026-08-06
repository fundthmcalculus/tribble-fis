"""Synthetic regression test for Interval Type-2 Fuzzy Regressor."""

import numpy as np
import pytest
import pandas as pd
from sklearn.metrics import mean_squared_error

from tribblefis.it2_regressor import IntervalType2FuzzyRegressor


@pytest.fixture
def synthetic_regression_data():
    """Generate synthetic nonlinear regression data with controlled noise."""
    np.random.seed(42)

    # Generate input
    x = np.linspace(-2, 2, 150)
    y_true = np.sin(3 * x)

    # Add heteroscedastic noise (scales with output magnitude)
    noise = 0.08 * np.abs(y_true) * np.random.randn(len(x))
    y = y_true + noise

    X = pd.DataFrame({"x": x})
    y = pd.Series(y)

    # Split into train and test
    split_idx = int(0.7 * len(x))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    y_true_test = y_true[split_idx:]

    return X_train, X_test, y_train, y_test, y_true_test


def test_it2_regressor_fit_predict(synthetic_regression_data):
    """Test IT2 regressor fit and predict on synthetic data."""
    X_train, X_test, y_train, y_test, y_true_test = synthetic_regression_data

    regressor = IntervalType2FuzzyRegressor(
        top_n=2,
        n_gaussians=2,
        n_output_buckets=3,
        uncertainty_width=0.5,
        km_iterations=10,
        random_state=42,
    )

    # Should fit without errors
    regressor.fit(X_train, y_train)

    # Should have learned model
    assert regressor.model_ is not None
    assert regressor.y_min_ is not None
    assert regressor.y_max_ is not None

    # Should predict on test set
    y_pred = regressor.predict(X_test)
    assert y_pred.shape == (len(X_test),)

    # Check that predictions are in reasonable range
    assert np.all(y_pred >= regressor.y_min_ - 0.1)
    assert np.all(y_pred <= regressor.y_max_ + 0.1)

    # Check RMSE (should be reasonable but not perfect)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    assert rmse < 2.0, f"RMSE {rmse} seems too high"


def test_it2_regressor_intervals_validity(synthetic_regression_data):
    """Test that prediction intervals are valid (lower <= upper)."""
    X_train, X_test, y_train, y_test, y_true_test = synthetic_regression_data

    regressor = IntervalType2FuzzyRegressor(
        top_n=2,
        n_gaussians=2,
        n_output_buckets=3,
        uncertainty_width=0.5,
        km_iterations=None,  # Use averaging
        random_state=42,
    )
    regressor.fit(X_train, y_train)

    y_lower, y_upper = regressor.predict_intervals(X_test)

    # Lower should always be <= upper
    assert np.all(y_lower <= y_upper), "Lower bounds should be <= upper bounds"

    # Both should be in expected range
    assert np.all(y_lower >= regressor.y_min_ - 0.1)
    assert np.all(y_upper <= regressor.y_max_ + 0.1)

    # Get crisp predictions
    y_crisp = regressor.predict(X_test)

    # Crisp should be between lower and upper (approximately)
    # Note: Type reduction might shift the crisp value slightly, so we use loose bounds
    assert np.all(y_crisp >= y_lower - 0.01)
    assert np.all(y_crisp <= y_upper + 0.01)


def test_it2_regressor_uncertainty_width_effect(synthetic_regression_data):
    """Test that larger uncertainty_width produces wider intervals."""
    X_train, X_test, y_train, y_test, y_true_test = synthetic_regression_data

    regressor_narrow = IntervalType2FuzzyRegressor(
        top_n=2,
        n_gaussians=2,
        n_output_buckets=3,
        uncertainty_width=0.2,  # Narrow
        km_iterations=None,
        random_state=42,
    )
    regressor_narrow.fit(X_train, y_train)

    regressor_wide = IntervalType2FuzzyRegressor(
        top_n=2,
        n_gaussians=2,
        n_output_buckets=3,
        uncertainty_width=1.0,  # Wide
        km_iterations=None,
        random_state=42,
    )
    regressor_wide.fit(X_train, y_train)

    # Get intervals
    y_lower_narrow, y_upper_narrow = regressor_narrow.predict_intervals(X_test)
    y_lower_wide, y_upper_wide = regressor_wide.predict_intervals(X_test)

    # Compute interval widths
    width_narrow = (y_upper_narrow - y_lower_narrow).mean()
    width_wide = (y_upper_wide - y_lower_wide).mean()

    assert width_wide > width_narrow, "Larger uncertainty_width should produce wider intervals"


def test_it2_regressor_km_vs_averaging(synthetic_regression_data):
    """Test that KM type reduction produces different results than averaging."""
    X_train, X_test, y_train, y_test, y_true_test = synthetic_regression_data

    regressor_km = IntervalType2FuzzyRegressor(
        top_n=2,
        n_gaussians=2,
        n_output_buckets=3,
        uncertainty_width=0.5,
        km_iterations=10,
        random_state=42,
    )
    regressor_km.fit(X_train, y_train)

    regressor_avg = IntervalType2FuzzyRegressor(
        top_n=2,
        n_gaussians=2,
        n_output_buckets=3,
        uncertainty_width=0.5,
        km_iterations=None,  # Averaging
        random_state=42,
    )
    regressor_avg.fit(X_train, y_train)

    # Get predictions
    y_pred_km = regressor_km.predict(X_test)
    y_pred_avg = regressor_avg.predict(X_test)

    # Predictions should be different
    assert not np.allclose(y_pred_km, y_pred_avg), "KM and averaging should produce different results"


def test_it2_regressor_intervals_contain_true_values(synthetic_regression_data):
    """Test that prediction intervals often contain true target values."""
    X_train, X_test, y_train, y_test, y_true_test = synthetic_regression_data

    regressor = IntervalType2FuzzyRegressor(
        top_n=2,
        n_gaussians=3,
        n_output_buckets=4,
        uncertainty_width=0.8,  # Wider uncertainty to ensure coverage
        km_iterations=None,
        random_state=42,
    )
    regressor.fit(X_train, y_train)

    y_lower, y_upper = regressor.predict_intervals(X_test)

    # Check what fraction of true values fall within intervals
    contained = (y_test >= y_lower) & (y_test <= y_upper)
    coverage = np.mean(contained)

    # Should capture at least some of the true values
    # With wider uncertainty, we expect decent coverage
    assert coverage > 0.5, f"Coverage {coverage} is too low; intervals should contain true values"
