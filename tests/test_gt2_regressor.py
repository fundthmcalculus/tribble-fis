"""Synthetic regression test for General Type-2 Fuzzy Regressor."""

import numpy as np
import pytest
import pandas as pd
from sklearn.metrics import mean_squared_error

from tribblefis.gt2_regressor import GT2TribbleRegressor
from tribblefis.it2_regressor import IT2TribbleRegressor


@pytest.fixture
def synthetic_regression_data():
    """Generate synthetic nonlinear regression data with controlled noise."""
    np.random.seed(42)

    x = np.linspace(-2, 2, 150)
    y_true = np.sin(3 * x)

    noise = 0.08 * np.abs(y_true) * np.random.randn(len(x))
    y = y_true + noise

    X = pd.DataFrame({"x": x})
    y = pd.Series(y)

    split_idx = int(0.7 * len(x))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    y_true_test = y_true[split_idx:]

    return X_train, X_test, y_train, y_test, y_true_test


def test_gt2_regressor_fit_predict(synthetic_regression_data):
    """Test GT2 regressor fit and predict on synthetic data."""
    X_train, X_test, y_train, y_test, y_true_test = synthetic_regression_data

    regressor = GT2TribbleRegressor(
        top_n=2, n_gaussians=2, n_output_buckets=3,
        uncertainty_width=0.5, n_alpha_planes=5, km_iterations=10, random_state=42,
    )

    regressor.fit(X_train, y_train)

    assert regressor.model_ is not None
    assert regressor.y_min_ is not None
    assert regressor.y_max_ is not None

    y_pred = regressor.predict(X_test)
    assert y_pred.shape == (len(X_test),)

    assert np.all(y_pred >= regressor.y_min_ - 0.1)
    assert np.all(y_pred <= regressor.y_max_ + 0.1)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    assert rmse < 2.0, f"RMSE {rmse} seems too high"


def test_gt2_regressor_intervals_validity(synthetic_regression_data):
    """Test that prediction intervals are valid (lower <= upper) and contain
    the point estimate."""
    X_train, X_test, y_train, y_test, y_true_test = synthetic_regression_data

    regressor = GT2TribbleRegressor(
        top_n=2, n_gaussians=2, n_output_buckets=3,
        uncertainty_width=0.5, km_iterations=None, random_state=42,
    )
    regressor.fit(X_train, y_train)

    y_lower, y_upper = regressor.predict_intervals(X_test)

    assert np.all(y_lower <= y_upper), "Lower bounds should be <= upper bounds"
    assert np.all(y_lower >= regressor.y_min_ - 0.1)
    assert np.all(y_upper <= regressor.y_max_ + 0.1)

    y_crisp = regressor.predict(X_test)
    assert np.all(y_crisp >= y_lower - 0.01)
    assert np.all(y_crisp <= y_upper + 0.01)


def test_gt2_regressor_uncertainty_width_effect(synthetic_regression_data):
    """Larger uncertainty_width should widen the alpha=0 boundary and
    therefore, in practice, the combined interval too."""
    X_train, X_test, y_train, y_test, y_true_test = synthetic_regression_data

    regressor_narrow = GT2TribbleRegressor(
        top_n=2, n_gaussians=2, n_output_buckets=3,
        uncertainty_width=0.2, km_iterations=None, random_state=42,
    )
    regressor_narrow.fit(X_train, y_train)

    regressor_wide = GT2TribbleRegressor(
        top_n=2, n_gaussians=2, n_output_buckets=3,
        uncertainty_width=1.0, km_iterations=None, random_state=42,
    )
    regressor_wide.fit(X_train, y_train)

    y_lower_narrow, y_upper_narrow = regressor_narrow.predict_intervals(X_test)
    y_lower_wide, y_upper_wide = regressor_wide.predict_intervals(X_test)

    width_narrow = (y_upper_narrow - y_lower_narrow).mean()
    width_wide = (y_upper_wide - y_lower_wide).mean()

    assert width_wide > width_narrow, "Larger uncertainty_width should produce wider intervals"


def test_gt2_regressor_fast_path_and_km_search_both_finite(synthetic_regression_data):
    X_train, X_test, y_train, y_test, y_true_test = synthetic_regression_data

    regressor_km = GT2TribbleRegressor(
        top_n=2, n_gaussians=2, n_output_buckets=3,
        uncertainty_width=0.5, km_iterations=10, random_state=42,
    )
    regressor_km.fit(X_train, y_train)

    regressor_avg = GT2TribbleRegressor(
        top_n=2, n_gaussians=2, n_output_buckets=3,
        uncertainty_width=0.5, km_iterations=None, random_state=42,
    )
    regressor_avg.fit(X_train, y_train)

    y_pred_km = regressor_km.predict(X_test)
    y_pred_avg = regressor_avg.predict(X_test)

    assert np.all(np.isfinite(y_pred_km)), "KM predictions should be finite"
    assert np.all(np.isfinite(y_pred_avg)), "Averaging predictions should be finite"


def test_gt2_converges_to_type1_as_footprint_vanishes(synthetic_regression_data):
    """GT2 must reduce to the Type-1 model when the footprint of uncertainty
    shrinks to nothing -- the GT2 analogue of
    `test_it2_regressor_synthetic.test_it2_converges_to_type1_as_footprint_vanishes`.

    As `uncertainty_width -> 0`, `upper_mf.sigma`, `lower_mf.sigma`, and
    `principal_mf.sigma` all converge to the same base sigma, so every
    alpha-plane's footprint collapses to the same crisp Type-1 model and the
    alpha-weighted combination of identical values is that same value.
    """
    from tribblefis.gaussian_regressor import TribbleRegressor

    X_train, X_test, y_train, y_test, _ = synthetic_regression_data
    kw = dict(top_n=1, n_gaussians=3, n_output_buckets=5, random_state=42)

    t1 = TribbleRegressor(**kw).fit(X_train, y_train)
    gt2 = GT2TribbleRegressor(
        uncertainty_width=1e-6, km_iterations=None, **kw
    ).fit(X_train, y_train)

    np.testing.assert_allclose(
        gt2.predict(X_test), t1.predict(X_test), rtol=1e-4, atol=1e-4
    )


def test_gt2_footprint_widens_intervals_and_contains_the_point_estimate(
    synthetic_regression_data,
):
    """`uncertainty_width` must widen the interval, and the point estimate
    must lie inside it -- the GT2 analogue of the equivalent IT2 test."""
    X_train, X_test, y_train, y_test, _ = synthetic_regression_data
    kw = dict(top_n=1, n_gaussians=3, n_output_buckets=5, random_state=42)

    widths = []
    for uw in (0.1, 0.3, 0.6, 0.9):
        m = GT2TribbleRegressor(
            uncertainty_width=uw, km_iterations=None, **kw
        ).fit(X_train, y_train)
        lo, hi = m.predict_intervals(X_test)
        point = m.predict(X_test)

        assert np.all(lo <= hi), "interval bounds are inverted"
        assert np.all(point >= lo - 1e-8) and np.all(point <= hi + 1e-8), (
            f"point estimate escapes its own interval at uncertainty_width={uw}"
        )
        widths.append(float(np.mean(hi - lo)))

    assert widths == sorted(widths), (
        f"interval width is not monotone in uncertainty_width: {widths}"
    )


def test_gt2_intervals_contain_the_point_estimate_off_fixture():
    """Containment on data that does not happen to make it trivial -- the
    GT2 analogue of
    `test_it2_regressor_synthetic.test_it2_intervals_contain_the_point_estimate_off_fixture`."""
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=400, n_features=5, noise=8.0, random_state=0)
    X = pd.DataFrame(X, columns=[f"x{i}" for i in range(5)])
    y = pd.Series(y)

    for km in (None, 10):
        m = GT2TribbleRegressor(
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


def test_gt2_regressor_rmse_is_comparable_to_it2(synthetic_regression_data):
    """Guard against a gross regression: GT2's alpha-weighted combination
    should not blow up RMSE relative to plain IT2 on the same base fit.

    This originally needed a wider membership spread to dodge a
    `karnik_mendel_tsk` zero-firing-threshold mismatch (`gauss_data.
    ZERO_FIRING_THRESHOLD`'s docstring) that made a handful of this
    fixture's extrapolated test rows disagree sharply with Type-1/IT2's own
    zero-firing fallback; with the two gates unified, the original
    `(top_n=2, n_gaussians=2, n_output_buckets=3)` shape is back."""
    X_train, X_test, y_train, y_test, _ = synthetic_regression_data
    kw = dict(top_n=2, n_gaussians=2, n_output_buckets=3, uncertainty_width=0.5, random_state=42)

    gt2 = GT2TribbleRegressor(**kw).fit(X_train, y_train)
    it2 = IT2TribbleRegressor(**kw).fit(X_train, y_train)

    gt2_rmse = np.sqrt(mean_squared_error(y_test, gt2.predict(X_test)))
    it2_rmse = np.sqrt(mean_squared_error(y_test, it2.predict(X_test)))
    assert gt2_rmse <= it2_rmse * 1.5 + 0.1
