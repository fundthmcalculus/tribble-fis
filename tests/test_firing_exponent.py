"""Tests for the ``firing_exponent`` blend-concentration knob.

``firing_exponent`` raises rule firing strengths to a power before the shared
row-normalization, in the solve and at predict time alike. 1.0 is the shipped
TSK weighting and must be an exact no-op; other values re-weight the blend.
"""

import contextlib
import io

import numpy as np
import pandas as pd
import pytest

from tribblefis.gaussian_regressor import TribbleRegressor
from tribblefis.regression import apply_firing_exponent


def _data(seed=0, n=400):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.uniform(0, 1, (n, 4)), columns=[f"x{i}" for i in range(4)])
    y = np.sin(3 * X.x0) + X.x1**2 + 0.5 * X.x2 + 0.05 * rng.normal(size=n)
    return X.iloc[:300], y[:300], X.iloc[300:]


def _fit_predict(gamma, Xtr, ytr, Xte):
    m = TribbleRegressor(
        tsk_order="full-2nd",
        n_output_buckets=3,
        l2_reg=1e-2,
        random_state=0,
        firing_exponent=gamma,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        m.fit(Xtr, ytr)
        return m.predict(Xte)


# --- the pure helper --------------------------------------------------------
def test_apply_firing_exponent_is_noop_at_one():
    f = np.array([[0.1, 0.2, 0.7], [0.5, 0.5, 0.5]])
    out = apply_firing_exponent(f, 1.0)
    assert out is f or np.array_equal(out, f)


def test_apply_firing_exponent_preserves_dead_rows():
    # a row no rule covers (all zero) must stay all-zero for any exponent, so
    # the no-rule-fires convention survives.
    f = np.array([[0.1, 0.2, 0.7], [0.0, 0.0, 0.0]])
    for gamma in (0.5, 2.0, 8.0):
        out = apply_firing_exponent(f, gamma)
        assert np.all(out[1] == 0.0)


def test_apply_firing_exponent_scale_invariant_direction():
    # gamma>1 concentrates mass on the max; gamma<1 spreads it out. Compare the
    # post-normalization weight of the strongest rule.
    f = np.array([[0.2, 0.3, 0.5]])

    def top_weight(gamma):
        s = apply_firing_exponent(f, gamma)
        return (s / s.sum(axis=1, keepdims=True))[0].max()

    assert top_weight(4.0) > top_weight(1.0) > top_weight(0.25)


# --- through the estimator ---------------------------------------------------
def test_firing_exponent_one_matches_default_exactly():
    Xtr, ytr, Xte = _data()
    default = TribbleRegressor(
        tsk_order="full-2nd", n_output_buckets=3, l2_reg=1e-2, random_state=0
    )
    with contextlib.redirect_stdout(io.StringIO()):
        default.fit(Xtr, ytr)
        p_default = default.predict(Xte)
    p_one = _fit_predict(1.0, Xtr, ytr, Xte)
    assert np.array_equal(p_default, p_one)


def test_firing_exponent_changes_predictions():
    Xtr, ytr, Xte = _data()
    p1 = _fit_predict(1.0, Xtr, ytr, Xte)
    assert not np.allclose(p1, _fit_predict(0.5, Xtr, ytr, Xte))
    assert not np.allclose(p1, _fit_predict(2.0, Xtr, ytr, Xte))


def test_firing_exponent_is_a_clonable_param():
    m = TribbleRegressor(firing_exponent=0.5)
    assert m.get_params()["firing_exponent"] == 0.5
    assert TribbleRegressor(**m.get_params()).firing_exponent == 0.5
