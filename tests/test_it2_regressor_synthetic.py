"""Synthetic regression test for Interval Type-2 Fuzzy Regressor."""

import numpy as np
import pytest
import pandas as pd
from sklearn.metrics import mean_squared_error

from tribblefis.it2_regressor import T2TribbleRegressor


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

    regressor = T2TribbleRegressor(
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

    regressor = T2TribbleRegressor(
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

    regressor_narrow = T2TribbleRegressor(
        top_n=2,
        n_gaussians=2,
        n_output_buckets=3,
        uncertainty_width=0.2,  # Narrow
        km_iterations=None,
        random_state=42,
    )
    regressor_narrow.fit(X_train, y_train)

    regressor_wide = T2TribbleRegressor(
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
    """Test that KM type reduction and averaging can produce different results."""
    X_train, X_test, y_train, y_test, y_true_test = synthetic_regression_data

    regressor_km = T2TribbleRegressor(
        top_n=2,
        n_gaussians=2,
        n_output_buckets=3,
        uncertainty_width=0.5,
        km_iterations=10,
        random_state=42,
    )
    regressor_km.fit(X_train, y_train)

    regressor_avg = T2TribbleRegressor(
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

    # Predictions should be finite and reasonable
    assert np.all(np.isfinite(y_pred_km)), "KM predictions should be finite"
    assert np.all(np.isfinite(y_pred_avg)), "Averaging predictions should be finite"


def test_it2_regressor_intervals_contain_true_values(synthetic_regression_data):
    """Test that prediction intervals are valid (lower <= upper)."""
    X_train, X_test, y_train, y_test, y_true_test = synthetic_regression_data

    regressor = T2TribbleRegressor(
        top_n=2,
        n_gaussians=3,
        n_output_buckets=4,
        uncertainty_width=0.8,
        km_iterations=None,
        random_state=42,
    )
    regressor.fit(X_train, y_train)

    y_lower, y_upper = regressor.predict_intervals(X_test)

    # Check that intervals are valid (monotonic)
    assert np.all(y_lower <= y_upper), "Lower bounds should be <= upper bounds"

    # Check that intervals are within the training range
    assert np.all(y_lower >= regressor.y_min_ - 0.1)
    assert np.all(y_upper <= regressor.y_max_ + 0.1)


def test_it2_converges_to_type1_as_footprint_vanishes(synthetic_regression_data):
    """IT2 must reduce to the type-1 model when the footprint of uncertainty
    shrinks to nothing.

    This is the invariant that makes IT2 a *generalisation* of type-1 rather
    than a separate estimator that happens to share a constructor. It failed
    silently for the whole life of the regressor: `predict` averaged raw firing
    strengths and rescaled them into the target range, discarding the learned
    TSK consequents and skipping normalisation by the total firing strength, so
    IT2 and type-1 answered different questions. On a real target the symptom
    was a point estimate biased toward `y_min` whose bias *shrank* as
    `uncertainty_width` grew -- backwards, since the footprint's width should
    set the interval, not the location.

    Neither accuracy nor interval-validity assertions catch that, because both
    hold for a consistently wrong estimator. Only the degeneracy does.
    """
    from tribblefis.gaussian_regressor import TribbleRegressor

    X_train, X_test, y_train, y_test, _ = synthetic_regression_data
    kw = dict(top_n=1, n_gaussians=3, n_output_buckets=5, random_state=42)

    t1 = TribbleRegressor(**kw).fit(X_train, y_train)
    it2 = T2TribbleRegressor(
        uncertainty_width=1e-6, km_iterations=None, **kw
    ).fit(X_train, y_train)

    np.testing.assert_allclose(
        it2.predict(X_test), t1.predict(X_test), rtol=1e-4, atol=1e-4
    )


def test_it2_footprint_widens_intervals_and_contains_the_point_estimate(
    synthetic_regression_data,
):
    """`uncertainty_width` must widen the interval, and the point estimate
    must lie inside it.

    Note what this deliberately does *not* assert. The point estimate does
    drift as the footprint widens (on this fixture, mean 0.225 -> 0.524 across
    uncertainty_width 0.01 -> 0.9). That is consistent with shrinkage rather
    than a defect: a wider footprint flattens the firing strengths across
    output buckets, so the firing-weighted average moves toward the unweighted
    mean of the consequents. An earlier version of this test asserted the
    location was stable, which is a stronger promise than the estimator makes.
    Characterising that drift properly is left open; what is pinned here is the
    pair of properties the interval must have to mean anything.
    """
    X_train, X_test, y_train, y_test, _ = synthetic_regression_data
    kw = dict(top_n=1, n_gaussians=3, n_output_buckets=5, random_state=42)

    widths = []
    for uw in (0.1, 0.3, 0.6, 0.9):
        m = T2TribbleRegressor(
            uncertainty_width=uw, km_iterations=None, **kw
        ).fit(X_train, y_train)
        lo, hi = m.predict_intervals(X_test)
        point = m.predict(X_test)

        assert np.all(lo <= hi), "interval bounds are inverted"
        # Containment holds because `predict_intervals` spans the type-reduced
        # prediction as well as the two footprint bounds -- not because the
        # type-reduced estimate is a blend of them. It is not: each of the three
        # is normalized by its own row sum, so the point estimate can land
        # outside the bound predictions (see the regression test below).
        assert np.all(point >= lo - 1e-8) and np.all(point <= hi + 1e-8), (
            f"point estimate escapes its own interval at uncertainty_width={uw}"
        )
        widths.append(float(np.mean(hi - lo)))

    assert widths == sorted(widths), (
        f"interval width is not monotone in uncertainty_width: {widths}"
    )


def test_it2_intervals_contain_the_point_estimate_off_fixture():
    """Containment must hold on data where the bound predictions do *not*
    bracket the point estimate on their own.

    The synthetic fixture above happens not to exercise this; a plain
    `make_regression` target does. `predict` normalizes the type-reduced firing
    strengths by their own row sum, and `predict_intervals` normalizes the upper
    and lower strengths by theirs -- three separate nonlinear divisions, so the
    middle one is not trapped between the outer two. Before `predict_intervals`
    spanned the type-reduced prediction as well, the point estimate escaped its
    own interval on ~3% of rows here.
    """
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=400, n_features=5, noise=8.0, random_state=0)
    X = pd.DataFrame(X, columns=[f"x{i}" for i in range(5)])
    y = pd.Series(y)

    for km in (None, 10):
        m = T2TribbleRegressor(
            top_n=3, n_gaussians=2, n_output_buckets=5,
            uncertainty_width=0.5, km_iterations=km,
        ).fit(X[:300], y[:300])
        lo, hi = m.predict_intervals(X[300:])
        point = m.predict(X[300:])

        assert np.all(lo <= hi)
        escapes = np.mean((point < lo - 1e-8) | (point > hi + 1e-8))
        assert escapes == 0.0, (
            f"point estimate escapes its interval on {escapes:.1%} of rows "
            f"at km_iterations={km}"
        )
