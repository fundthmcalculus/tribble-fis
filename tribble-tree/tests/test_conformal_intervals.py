"""Split-conformal `predict_intervals` on the tree regressors (#268).

Mirrors `tests/test_conformal_calibration.py` (the IT2/GT2 version, #149) for
`DeconstructedHierarchicalRegressor` and `HierarchicalFuzzyExpertsRegressor`,
plus the property that matters most for a tree: calibration rows must never
reach topology selection, gate building or EM.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import fuzzytree.deconstruct as deconstruct_mod
import fuzzytree.hme as hme_mod
from fuzzytree import DeconstructedHierarchicalRegressor, HierarchicalFuzzyExpertsRegressor

TOPOLOGY = {"TOTAL": ["G1", "G2", "G3"], "G1": ["a", "b"], "G2": ["c", "d"], "G3": ["e", "f"]}
FLAT = {"n_output_buckets": 5, "tsk_order": "1st", "top_n": -1, "random_state": 42}


def make_data(n, seed):
    """Stage A's generator (`demo_deconstruct_synthetic.make_data`)."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({c: rng.uniform(0, 10, n) for c in "abcdef"})
    g1 = np.sin(X["a"] / 2) * 5 + 0.5 * X["b"]
    g2 = np.cos(X["c"] / 3) * 4 - 0.3 * X["d"]
    g3 = np.tanh((X["e"] - 5) / 2) * 6 + 0.2 * X["f"]
    y = 1.0 * g1 + 0.7 * g2 + 1.4 * g3 + rng.normal(0, 0.3, n)
    return X, y.to_numpy()


def _coverage(lo, hi, y):
    return float(np.mean((y >= lo) & (y <= hi)))


def _deconstructed(**kw):
    return DeconstructedHierarchicalRegressor(flat_regressor_kwargs=FLAT, **kw)


def _hme(**kw):
    return HierarchicalFuzzyExpertsRegressor(max_depth=2, gate_style="gaussian", **kw)


@pytest.fixture(scope="module")
def data():
    X, y = make_data(2400, seed=0)
    return X.iloc[:1200], y[:1200], X.iloc[1200:].reset_index(drop=True), y[1200:]


# -- off by default ----------------------------------------------------------

def test_deconstructed_off_by_default_is_unchanged(data):
    X, y, X_te, _ = data
    default = _deconstructed().fit(X, y, TOPOLOGY)
    explicit = _deconstructed(conformal_calibration=False).fit(X, y, TOPOLOGY)
    assert default.conformal_margin_ is None
    np.testing.assert_array_equal(default.predict(X_te), explicit.predict(X_te))
    with pytest.raises(RuntimeError, match="conformal_calibration=True"):
        default.predict_intervals(X_te)


def test_hme_off_by_default_is_unchanged(data):
    X, y, X_te, _ = data
    default = _hme().fit(X, y)
    explicit = _hme(conformal_calibration=False).fit(X, y)
    assert default.conformal_margin_ is None
    np.testing.assert_array_equal(default.predict(X_te), explicit.predict(X_te))
    with pytest.raises(RuntimeError, match="conformal_calibration=True"):
        default.predict_intervals(X_te)


# -- coverage ----------------------------------------------------------------

@pytest.mark.parametrize("make", [
    lambda seed: (_deconstructed(conformal_calibration=True, conformal_random_state=seed), TOPOLOGY),
    lambda seed: (_hme(conformal_calibration=True, random_state=seed), None),
], ids=["deconstructed", "hme"])
def test_coverage_hits_target(make):
    covs = []
    for seed in (0, 1, 2):
        X, y = make_data(3000, seed=seed)
        X_tr, y_tr = X.iloc[:1500], y[:1500]
        X_te, y_te = X.iloc[1500:].reset_index(drop=True), y[1500:]
        model, topo = make(seed)
        model.fit(X_tr, y_tr, topo) if topo else model.fit(X_tr, y_tr)
        assert model.conformal_margin_ > 0
        lo, hi = model.predict_intervals(X_te)
        np.testing.assert_allclose(0.5 * (lo + hi), model.predict(X_te))
        covs.append(_coverage(lo, hi, y_te))
    # Marginal coverage >= 0.9 in expectation; three seeds of 300 calibration
    # rows put the mean within a few points of it.
    assert abs(np.mean(covs) - 0.9) <= 0.04, covs


# -- calibration rows never reach the model ----------------------------------

def test_deconstructed_auto_topology_never_sees_calibration_rows(monkeypatch, data):
    X, y, _, _ = data
    seen = []
    real = deconstruct_mod.select_topology

    def spy(X_sel, y_sel, *a, **kw):
        seen.append(len(X_sel))
        return real(X_sel, y_sel, *a, **kw)

    monkeypatch.setattr(deconstruct_mod, "select_topology", spy)
    model = _deconstructed(conformal_calibration=True, conformal_calibration_frac=0.25)
    model.fit(X, y, "auto")
    assert seen == [900]
    assert model.topology_source_ == "auto"


def test_hme_gate_tree_never_sees_calibration_rows(monkeypatch, data):
    X, y, _, _ = data
    seen = []
    real = hme_mod.build_tree

    def spy(X_gate, *a, **kw):
        seen.append(len(X_gate))
        return real(X_gate, *a, **kw)

    monkeypatch.setattr(hme_mod, "build_tree", spy)
    _hme(conformal_calibration=True, conformal_calibration_frac=0.25).fit(X, y)
    assert seen == [900]


def test_deconstructed_slices_leaf_targets(data):
    X, y, X_te, _ = data
    g1 = np.sin(X["a"].to_numpy() / 2) * 5 + 0.5 * X["b"].to_numpy()
    model = _deconstructed(conformal_calibration=True)
    model.fit(X, y, TOPOLOGY, leaf_targets={"G1": g1})
    lo, hi = model.predict_intervals(X_te)
    assert np.all(hi > lo)


def test_hme_refine_em_uses_fit_split_and_resizes_margin(data):
    X, y, X_te, _ = data
    model = _hme(conformal_calibration=True).fit(X, y)
    before = model.conformal_margin_
    pred_before = model.predict(X_te)
    with pytest.raises(ValueError, match="without X/y"):
        model.refine_em(X, y, max_iter=2)
    model.refine_em(max_iter=3)
    assert model.conformal_margin_ is not None
    if not np.allclose(model.predict(X_te), pred_before):
        assert model.conformal_margin_ != before


def test_hme_refine_em_still_requires_data_without_conformal(data):
    X, y, _, _ = data
    model = _hme().fit(X, y)
    with pytest.raises(TypeError):
        model.refine_em(max_iter=1)
    assert model.refine_em(X, y, max_iter=1) is model


# -- validation --------------------------------------------------------------

@pytest.mark.parametrize("kw", [{"conformal_alpha": 0.0}, {"conformal_alpha": 1.0},
                                {"conformal_calibration_frac": 0.0},
                                {"conformal_calibration_frac": 1.0}])
def test_rejects_bad_parameters(kw, data):
    X, y, _, _ = data
    with pytest.raises(ValueError):
        _deconstructed(conformal_calibration=True, **kw).fit(X, y, TOPOLOGY)
    with pytest.raises(ValueError):
        _hme(conformal_calibration=True, **kw).fit(X, y)


def test_warns_when_calibration_split_too_small(data):
    X, y, _, _ = data
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _deconstructed(conformal_calibration=True, conformal_alpha=0.01,
                       conformal_calibration_frac=0.05).fit(X.iloc[:400], y[:400], TOPOLOGY)
    assert any("too few" in str(w.message) for w in caught)
