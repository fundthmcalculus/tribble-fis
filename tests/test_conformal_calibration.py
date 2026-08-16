"""Tests for #149's split-conformal fix: `conformal_calibration` on IT2/GT2 regressors.

`predict_intervals()`'s raw interval only encodes antecedent firing
disagreement, not residual/aleatoric noise, so its coverage plateaus well
under any target regardless of `uncertainty_width` (see
`docs/t1-it2-gt2-tradeoff.md`). These tests check that the additive
split-conformal margin actually closes that gap without disturbing `predict`.
"""

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from tribblefis.it2_regressor import IT2TribbleRegressor
from tribblefis.gt2_regressor import GT2TribbleRegressor
from tribblefis.regression import conformal_calibration_margin


@pytest.fixture
def noisy_regression_data():
    X, y = make_regression(
        n_samples=600, n_features=5, n_informative=3, noise=15.0, random_state=42
    )
    return train_test_split(X, y, test_size=0.3, random_state=42)


def _coverage(y_lower, y_upper, y_true):
    return float(np.mean((y_true >= y_lower) & (y_true <= y_upper)))


@pytest.mark.parametrize("regressor_cls", [IT2TribbleRegressor, GT2TribbleRegressor])
def test_conformal_calibration_off_by_default(regressor_cls, noisy_regression_data):
    X_train, X_test, y_train, y_test = noisy_regression_data
    model = regressor_cls(n_gaussians=2, top_n=3, random_state=42)
    model.fit(X_train, y_train)

    assert model.conformal_margin_ is None

    y_lower, y_upper = model.predict_intervals(X_test)
    y_pred = model.predict(X_test)
    assert np.allclose(y_pred, 0.5 * (y_lower + y_upper))


@pytest.mark.parametrize("regressor_cls", [IT2TribbleRegressor, GT2TribbleRegressor])
def test_conformal_calibration_improves_coverage(regressor_cls, noisy_regression_data):
    X_train, X_test, y_train, y_test = noisy_regression_data

    raw = regressor_cls(n_gaussians=2, top_n=3, random_state=42)
    raw.fit(X_train, y_train)
    raw_lower, raw_upper = raw.predict_intervals(X_test)
    raw_coverage = _coverage(raw_lower, raw_upper, y_test)

    calibrated = regressor_cls(
        n_gaussians=2, top_n=3, random_state=42,
        conformal_calibration=True, conformal_alpha=0.1,
    )
    calibrated.fit(X_train, y_train)
    assert calibrated.conformal_margin_ is not None
    assert calibrated.conformal_margin_ >= 0.0

    cal_lower, cal_upper = calibrated.predict_intervals(X_test)
    cal_coverage = _coverage(cal_lower, cal_upper, y_test)

    # The raw interval structurally caps out well under any real target (see
    # docs/t1-it2-gt2-tradeoff.md); the conformal margin should close most of
    # that gap towards the 90% target, without needing to hit it exactly on
    # one held-out split.
    assert cal_coverage > raw_coverage
    assert cal_coverage >= 0.75


@pytest.mark.parametrize("regressor_cls", [IT2TribbleRegressor, GT2TribbleRegressor])
def test_conformal_calibration_keeps_predict_midpoint_invariant(regressor_cls, noisy_regression_data):
    X_train, X_test, y_train, _ = noisy_regression_data
    model = regressor_cls(
        n_gaussians=2, top_n=3, random_state=42, conformal_calibration=True,
    )
    model.fit(X_train, y_train)

    y_lower, y_upper = model.predict_intervals(X_test)
    y_pred = model.predict(X_test)
    # The margin is symmetric, so it cancels out of the midpoint regardless
    # of whether calibration is on.
    assert np.allclose(y_pred, 0.5 * (y_lower + y_upper))


def test_conformal_calibration_margin_shrinks_with_looser_alpha():
    rng = np.random.default_rng(0)
    y_calib = rng.normal(size=200)
    y_lower = np.full(200, -0.2)
    y_upper = np.full(200, 0.2)

    tight = conformal_calibration_margin(y_calib, y_lower, y_upper, alpha=0.05)
    loose = conformal_calibration_margin(y_calib, y_lower, y_upper, alpha=0.3)
    assert tight >= loose >= 0.0


def test_conformal_calibration_margin_zero_when_already_covered():
    y_calib = np.zeros(50)
    y_lower = np.full(50, -1.0)
    y_upper = np.full(50, 1.0)
    assert conformal_calibration_margin(y_calib, y_lower, y_upper, alpha=0.1) == 0.0
