"""Parity and dispatch tests for the optional compiled kernel.

Skipped wholesale when the extension has not been built, which is the default
state of a fresh checkout -- the point of the accelerator being optional is that
`pytest` passes without a C compiler anywhere in sight. What is *not* skipped is
:func:`test_kernel_absence_is_survivable`, because the fallback has to work on
the machines that skip everything else.
"""

import uuid

import numpy as np
import pandas as pd
import pytest

from tribblefis import kernel as K
from tribblefis.gauss_data import (
    AnomalyParameters,
    FeatureModel,
    GaussianMembership,
    GaussianMixtureModel,
    LabelModel,
    NormPair,
    TrapezoidMembership,
)
from tribblefis.gauss_math import tsk_firing_strengths

requires_kernel = pytest.mark.skipif(
    not K.HAVE_CYTHON_KERNEL,
    reason="compiled kernel not built (python setup_cython.py build_ext --inplace)",
)

NORMS = ["min/max", "probability", "luk", "hamacher", "einstein"]
SHAPES = [(500, 3, 2, 1), (1000, 8, 3, 3), (997, 5, 4, 2), (2000, 12, 6, 4)]

# The compiled kernel calls libm's `exp`; NumPy calls its own vectorized one.
# They are not contractually the same function, so parity is asserted to a few
# ULP rather than exactly. On the reference machine the two agree bit for bit,
# and `test_backends_agree_bitwise_where_they_can` records that as an observation
# without making the suite depend on it.
ULP_TOL = 1e-13


def _build(n_samples, n_features, n_labels, n_mf, seed=0, ragged=False):
    rng = np.random.default_rng(seed)
    cols = [f"f{i}" for i in range(n_features)]
    X = pd.DataFrame(rng.normal(0, 2, size=(n_samples, n_features)), columns=cols)
    feature_models = {}
    for fi, name in enumerate(cols):
        label_models = {}
        for label in range(n_labels):
            k = n_mf - 1 if (ragged and fi == 0 and label == 0 and n_mf > 1) else n_mf
            label_models[label] = LabelModel(memberships=[
                GaussianMembership(mu=float(rng.normal(0, 2)),
                                   sigma=float(rng.uniform(0.3, 2.0)),
                                   id=uuid.UUID(bytes=rng.bytes(16)))
                for _ in range(k)
            ])
        feature_models[name] = FeatureModel(label_models=label_models)
    return X, GaussianMixtureModel(feature_models=feature_models)


def _prepare(X, model):
    arrays = {c: X[c].to_numpy() for c in X.columns}
    compiled = K.compile_model(model, list(arrays))
    return compiled, compiled.feature_matrix(arrays)


@requires_kernel
@pytest.mark.parametrize("norm", NORMS)
@pytest.mark.parametrize("shape", SHAPES)
def test_compiled_matches_numpy(norm, shape):
    X, model = _build(*shape, seed=abs(hash((norm, shape))) % 1000)
    compiled, matrix = _prepare(X, model)
    norms = NormPair(norm, norm)
    expected = K.firing_strengths_numpy(compiled, matrix, norms)
    got = K.firing_strengths(compiled, matrix, norms, backend="cython")
    assert np.allclose(expected, got, rtol=ULP_TOL, atol=0.0)


@requires_kernel
@pytest.mark.parametrize("norm", NORMS)
def test_compiled_handles_padded_cells(norm):
    X, model = _build(800, 5, 3, 3, seed=21, ragged=True)
    compiled, matrix = _prepare(X, model)
    assert not compiled.is_dense
    norms = NormPair(norm, norm)
    expected = K.firing_strengths_numpy(compiled, matrix, norms)
    got = K.firing_strengths(compiled, matrix, norms, backend="cython")
    assert np.allclose(expected, got, rtol=ULP_TOL, atol=0.0)


@requires_kernel
def test_compiled_honours_a_mixed_norm_pair():
    X, model = _build(600, 6, 4, 2, seed=22)
    compiled, matrix = _prepare(X, model)
    norms = NormPair(t_norm="probability", t_conorm="min/max")
    expected = K.firing_strengths_numpy(compiled, matrix, norms)
    got = K.firing_strengths(compiled, matrix, norms, backend="cython")
    assert np.allclose(expected, got, rtol=ULP_TOL, atol=0.0)


@requires_kernel
@pytest.mark.parametrize("threads", ["1", "2", "8"])
def test_thread_count_does_not_change_the_result(monkeypatch, threads):
    """Samples are independent, so the row loop is embarrassingly parallel --
    but only if nothing is shared between iterations. A result that moves with
    the thread count would mean it is."""
    X, model = _build(3000, 6, 4, 3, seed=23)
    compiled, matrix = _prepare(X, model)
    norms = NormPair("probability", "probability")

    monkeypatch.setenv("TRIBBLEFIS_NUM_THREADS", "1")
    serial = K.firing_strengths(compiled, matrix, norms, backend="cython")
    monkeypatch.setenv("TRIBBLEFIS_NUM_THREADS", threads)
    parallel = K.firing_strengths(compiled, matrix, norms, backend="cython")
    assert np.array_equal(serial, parallel)


@requires_kernel
def test_backends_agree_bitwise_where_they_can():
    """Documents the stronger property actually observed: on a build without
    fast-math, libm's exp and NumPy's agree exactly, so the whole forward pass
    does. Recorded as its own test so that if a platform ever breaks it, the
    failure names the reason rather than looking like a correctness bug."""
    X, model = _build(1200, 7, 3, 2, seed=24)
    compiled, matrix = _prepare(X, model)
    norms = NormPair("min/max", "min/max")
    expected = K.firing_strengths_numpy(compiled, matrix, norms)
    got = K.firing_strengths(compiled, matrix, norms, backend="cython")
    assert np.array_equal(expected, got), (
        "compiled and NumPy exp diverged; if this is a fast-math build that is "
        "expected -- the allclose parity tests are the binding contract"
    )


@requires_kernel
def test_tsk_firing_strengths_fast_path_matches_the_reference_loop(monkeypatch):
    """`tsk_firing_strengths` substitutes the kernel silently, so the substitution
    is checked against the loop it replaced -- with the kernel disabled."""
    X, model = _build(1500, 6, 4, 2, seed=25)
    norms = NormPair("min/max", "min/max")
    fast, labels_fast = tsk_firing_strengths(X, model, norms=norms)

    monkeypatch.setattr(K, "HAVE_CYTHON_KERNEL", False)
    slow, labels_slow = tsk_firing_strengths(X, model, norms=norms)

    assert list(labels_fast) == list(labels_slow)
    assert np.allclose(fast, slow, rtol=ULP_TOL, atol=0.0)


@requires_kernel
def test_fast_path_reproduces_the_anomaly_column(monkeypatch):
    """The anomaly column is derived from the class columns, not from the
    memberships, so the fast path computes it separately -- and must get the
    same answer."""
    X, model = _build(900, 5, 3, 2, seed=26)
    details = AnomalyParameters(include_anomaly=True, threshold=0.4,
                                label="anomaly", norm_conorm="probability")
    fast, labels_fast = tsk_firing_strengths(X, model, anomaly_details=details)

    monkeypatch.setattr(K, "HAVE_CYTHON_KERNEL", False)
    slow, labels_slow = tsk_firing_strengths(X, model, anomaly_details=details)

    assert labels_fast == labels_slow
    assert labels_fast[-1] == "anomaly"
    assert np.allclose(fast, slow, rtol=ULP_TOL, atol=0.0)


@requires_kernel
def test_uncompilable_model_falls_through_to_the_reference_loop():
    """A trapezoid model cannot use the fast path; it must still work."""
    X, model = _build(300, 3, 2, 1, seed=27)
    fname = next(iter(model.feature_models))
    label = next(iter(model.feature_models[fname].label_models))
    model.feature_models[fname].label_models[label] = LabelModel(
        memberships=[TrapezoidMembership.create(-2.0, -1.0, 1.0, 2.0)]
    )
    assert not K.can_compile(model, list(X.columns))
    strengths, labels = tsk_firing_strengths(X, model)
    assert strengths.shape == (300, len(labels))
    assert np.all(np.isfinite(strengths))


def test_backend_argument_is_validated():
    X, model = _build(50, 2, 2, 1, seed=28)
    compiled, matrix = _prepare(X, model)
    norms = NormPair("min/max", "min/max")
    with pytest.raises(ValueError, match="backend must be"):
        K.firing_strengths(compiled, matrix, norms, backend="gpu")


def test_kernel_absence_is_survivable(monkeypatch):
    """With the extension unavailable, `backend='auto'` must still work and
    `backend='cython'` must say plainly what to do about it."""
    X, model = _build(200, 3, 2, 2, seed=29)
    compiled, matrix = _prepare(X, model)
    norms = NormPair("min/max", "min/max")
    expected = K.firing_strengths_numpy(compiled, matrix, norms)

    monkeypatch.setattr(K, "HAVE_CYTHON_KERNEL", False)
    assert np.array_equal(K.firing_strengths(compiled, matrix, norms), expected)
    with pytest.raises(RuntimeError, match="setup_cython.py"):
        K.firing_strengths(compiled, matrix, norms, backend="cython")


def test_norm_codes_cover_every_family():
    """The kernel dispatches on an integer code; a family added to `gauss_data`
    without a code here would silently fall into the kernel's `else` branch."""
    from tribblefis.gauss_data import NORM_FAMILIES

    assert set(K._NORM_CODES) == set(NORM_FAMILIES)
    assert sorted(K._NORM_CODES.values()) == list(range(len(NORM_FAMILIES)))
