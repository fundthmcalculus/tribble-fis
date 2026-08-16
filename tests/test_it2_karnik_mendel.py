"""Correctness tests for the real (cross-rule) Karnik-Mendel type reduction.

`it2_kernel.karnik_mendel_tsk` replaces the previous placeholder, which never
actually searched anything (see `it2_kernel.karnik_mendel_type_reduction`'s
docstring for the proof that it always just returned the midpoint). These
tests validate the new switch-point search against a brute-force reference and
check the guarantees the IT2 regressor now relies on.
"""

from itertools import product

import numpy as np
import pandas as pd
import pytest

from tribblefis.it2_kernel import karnik_mendel_tsk
from tribblefis.it2_regressor import IntervalType2FuzzyRegressor


def _brute_force_km(y, f_lower, f_upper, n_grid=20):
    """Reference min/max of `sum(w*y)/sum(w)` over a fine grid of admissible
    weights `w_i in [f_lower_i, f_upper_i]` -- exhaustive, not the switch-point
    search, so it is safe as an independent oracle for small rule counts."""
    grids = [np.linspace(f_lower[i], f_upper[i], n_grid) for i in range(len(y))]
    lo, hi = np.inf, -np.inf
    for combo in product(*grids):
        w = np.array(combo)
        s = w.sum()
        if s <= 0:
            continue
        val = float((w * y).sum() / s)
        lo, hi = min(lo, val), max(hi, val)
    return lo, hi


@pytest.mark.parametrize("seed", range(15))
def test_karnik_mendel_tsk_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    n_rules = int(rng.integers(2, 5))
    y = rng.uniform(-5, 5, size=n_rules)
    f_lower = rng.uniform(0.0, 0.5, size=n_rules)
    f_upper = f_lower + rng.uniform(0.01, 0.5, size=n_rules)

    y_l, y_r = karnik_mendel_tsk(y[None, :], f_lower[None, :], f_upper[None, :], max_iterations=50)
    expect_l, expect_r = _brute_force_km(y, f_lower, f_upper)

    assert y_l[0] == pytest.approx(expect_l, abs=2e-2)
    assert y_r[0] == pytest.approx(expect_r, abs=2e-2)
    # KM's own guarantee: the searched interval must be no wider than any single
    # admissible combination's spread and must bracket the equal-weight average.
    equal_weight_avg = float(np.mean(y_lower_upper_avg(y, f_lower, f_upper)))
    assert y_l[0] <= equal_weight_avg + 1e-6
    assert y_r[0] >= equal_weight_avg - 1e-6


def y_lower_upper_avg(y, f_lower, f_upper):
    f = 0.5 * (f_lower + f_upper)
    return [(f * y).sum() / f.sum()]


def test_karnik_mendel_tsk_single_rule_is_that_rules_value():
    y = np.array([[3.5]])
    f_lower = np.array([[0.2]])
    f_upper = np.array([[0.9]])
    y_l, y_r = karnik_mendel_tsk(y, f_lower, f_upper)
    assert y_l[0] == pytest.approx(3.5)
    assert y_r[0] == pytest.approx(3.5)


def test_karnik_mendel_tsk_zero_firing_row_is_zero():
    y = np.array([[1.0, -1.0]])
    f_lower = np.array([[0.0, 0.0]])
    f_upper = np.array([[0.0, 0.0]])
    y_l, y_r = karnik_mendel_tsk(y, f_lower, f_upper)
    assert y_l[0] == 0.0
    assert y_r[0] == 0.0


def test_karnik_mendel_tsk_batches_multiple_rows_independently():
    """The parallel per-row search must not cross-contaminate rows."""
    rng = np.random.default_rng(0)
    n_samples, n_rules = 200, 4
    y = rng.uniform(-3, 3, size=(n_samples, n_rules))
    f_lower = rng.uniform(0, 0.4, size=(n_samples, n_rules))
    f_upper = f_lower + rng.uniform(0.01, 0.4, size=(n_samples, n_rules))

    y_l_batch, y_r_batch = karnik_mendel_tsk(y, f_lower, f_upper)
    for i in range(n_samples):
        y_l_row, y_r_row = karnik_mendel_tsk(y[i:i + 1], f_lower[i:i + 1], f_upper[i:i + 1])
        assert y_l_batch[i] == pytest.approx(y_l_row[0], abs=1e-9)
        assert y_r_batch[i] == pytest.approx(y_r_row[0], abs=1e-9)


@pytest.fixture
def synthetic_regression_data():
    rng = np.random.default_rng(42)
    x = np.linspace(-2, 2, 150)
    y = np.sin(3 * x) + 0.05 * rng.standard_normal(len(x))
    return pd.DataFrame({"x": x}), pd.Series(y)


def test_regressor_predict_is_exact_midpoint_of_predict_intervals(synthetic_regression_data):
    """The point estimate is defined as the KM interval's midpoint, so it must
    equal it exactly -- not merely fall within some tolerance -- for both the
    KM search path and the fast averaging path."""
    X, y = synthetic_regression_data
    for km_iterations in (None, 10):
        reg = IntervalType2FuzzyRegressor(
            top_n=1, n_gaussians=2, n_output_buckets=3,
            uncertainty_width=0.5, km_iterations=km_iterations, random_state=42,
        )
        reg.fit(X, y)
        y_pred = reg.predict(X)
        y_lower, y_upper = reg.predict_intervals(X)
        np.testing.assert_allclose(y_pred, 0.5 * (y_lower + y_upper), atol=1e-10)


def test_regressor_km_path_guarantees_containment(synthetic_regression_data):
    """With the real KM search, `predict_intervals` structurally cannot exclude
    `predict`'s point estimate -- unlike the old two-stage pipeline, which
    measurably violated this on ~3% of rows (see `it2_regressor.py`)."""
    X, y = synthetic_regression_data
    reg = IntervalType2FuzzyRegressor(
        top_n=1, n_gaussians=3, n_output_buckets=4,
        uncertainty_width=0.6, km_iterations=15, random_state=1,
    )
    reg.fit(X, y)
    y_pred = reg.predict(X)
    y_lower, y_upper = reg.predict_intervals(X)
    assert np.all(y_lower <= y_pred + 1e-9)
    assert np.all(y_pred <= y_upper + 1e-9)
