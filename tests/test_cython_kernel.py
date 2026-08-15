"""Parity and dispatch tests for the optional compiled kernel.

Skipped wholesale when the extension has not been built, which is the default
state of a fresh checkout -- the point of the accelerator being optional is that
`pytest` passes without a C compiler anywhere in sight. What is *not* skipped is
:func:`test_kernel_absence_is_survivable`, because the fallback has to work on
the machines that skip everything else.
"""

import math
import uuid
import zlib

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

ULP_TOL = 1e-13

# The one and only way the two backends differ: the compiled kernel calls libm's
# scalar `exp`, NumPy calls its own SIMD one. They are not contractually the same
# function -- on NumPy 2.4 / AVX-512 they disagree by exactly 1 ULP on ~5% of
# inputs. Everything downstream of `exp` is bit-identical, which
# `test_folds_are_bit_identical_given_the_same_exp` proves directly, so these
# tolerances bound *only* how far the folds amplify that 1 ULP.
#
# Firing strengths live in [0, 1], so absolute error is their natural scale.
# `atol=0.0` with a pure `rtol` was the wrong contract: Łukasiewicz's
# `max(0, a + b - 1)` lands on values near zero by construction, where any
# absolute perturbation is an unbounded *relative* one.
#
# Worst absolute error measured over SHAPES x 6 seeds on x86-64/glibc:
#   min/max 1.1e-16   probability 6.7e-16   luk 6.7e-16   einstein 1.3e-15
# Hamacher is the outlier at 8.8e-13, ~4000 ULP, and it is inherent rather than a
# defect: both its folds divide by a difference that cancels -- `1 - ab` in the
# conorm as `ab -> 1`, `a + b - ab` in the t-norm as both go to zero -- so a
# 1-ULP input perturbation is amplified once per fold step, and the t-norm chains
# once per feature (the worst case above is the 12-feature shape).
ATOL = {norm: 1e-14 for norm in NORMS}
ATOL["hamacher"] = 1e-11


def _seed_for(*parts) -> int:
    """Stable across processes, unlike `hash()`.

    `abs(hash((norm, shape))) % 1000` seeded these tests for a while, but Python
    randomizes `hash()` of strings per process (PYTHONHASHSEED), so every run
    drew *different* models and a different subset of parametrizations tripped
    the tolerance. A numerical parity test that samples fresh data each run
    cannot be a contract.
    """
    return zlib.crc32("".join(map(str, parts)).encode()) % 1000


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
    X, model = _build(*shape, seed=_seed_for(norm, shape))
    compiled, matrix = _prepare(X, model)
    norms = NormPair(norm, norm)
    expected = K.firing_strengths_numpy(compiled, matrix, norms)
    got = K.firing_strengths(compiled, matrix, norms, backend="cython")
    assert np.allclose(expected, got, rtol=ULP_TOL, atol=ATOL[norm])


@requires_kernel
@pytest.mark.parametrize("norm", NORMS)
def test_compiled_handles_padded_cells(norm):
    X, model = _build(800, 5, 3, 3, seed=21, ragged=True)
    compiled, matrix = _prepare(X, model)
    assert not compiled.is_dense
    norms = NormPair(norm, norm)
    expected = K.firing_strengths_numpy(compiled, matrix, norms)
    got = K.firing_strengths(compiled, matrix, norms, backend="cython")
    assert np.allclose(expected, got, rtol=ULP_TOL, atol=ATOL[norm])


@requires_kernel
def test_compiled_honours_a_mixed_norm_pair():
    X, model = _build(600, 6, 4, 2, seed=22)
    compiled, matrix = _prepare(X, model)
    norms = NormPair(t_norm="probability", t_conorm="min/max")
    expected = K.firing_strengths_numpy(compiled, matrix, norms)
    got = K.firing_strengths(compiled, matrix, norms, backend="cython")
    assert np.allclose(expected, got, rtol=ULP_TOL, atol=ATOL["probability"])


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
@pytest.mark.parametrize("norm", NORMS)
def test_folds_are_bit_identical_given_the_same_exp(monkeypatch, norm):
    """The tight contract: every arithmetic step *after* `exp` matches exactly.

    This used to assert that the whole forward pass was bit-identical, which
    held only where libm's `exp` and NumPy's happened to agree -- true on the
    machine it was written on, false on NumPy 2.4 with AVX-512, where they
    differ by 1 ULP on ~5% of inputs. So it failed as if the kernel were wrong
    when the kernel was not wrong.

    Feeding both backends the *same* exp values separates the two claims. What
    is left is exact and platform-independent: the compiled folds are the NumPy
    folds, bit for bit, for every norm family -- including the ill-conditioned
    Hamacher one, where `test_compiled_matches_numpy` can only afford a loose
    absolute tolerance. That looseness is bounded amplification of the exp
    difference, not slack in the arithmetic, and this is what says so.
    """
    X, model = _build(1200, 7, 3, 2, seed=24)
    compiled, matrix = _prepare(X, model)
    norms = NormPair(norm, norm)
    got = K.firing_strengths(compiled, matrix, norms, backend="cython")

    # `kernel.firing_strengths_numpy` calls `np.exp`; route it through the same
    # libm `exp` the compiled kernel is linked against. Slow, hence one shape.
    libm = np.frompyfunc(math.exp, 1, 1)

    def libm_exp(x, out=None, **kwargs):
        result = libm(x).astype(np.float64)
        if out is None:
            return result
        out[...] = result
        return out

    monkeypatch.setattr(np, "exp", libm_exp)
    expected = K.firing_strengths_numpy(compiled, matrix, norms)

    assert np.array_equal(expected, got), (
        f"compiled and NumPy folds differ for {norm} even on identical exp "
        "values -- that is an arithmetic divergence, not a libm/SIMD one"
    )


@requires_kernel
def test_the_exp_difference_is_at_most_one_ulp():
    """Bounds the *only* thing the two backends do differently.

    The parity tolerances above are derived from this number, so if a platform's
    `exp` ever drifts further -- a fast-math build, a new SIMD path -- this test
    names the cause directly instead of leaving the parity tests to fail with no
    explanation of which half moved.
    """
    rng = np.random.default_rng(0)
    d = (rng.normal(0, 2, 200_000) - rng.normal(0, 2)) / rng.uniform(0.3, 2.0)
    arg = -0.5 * d * d

    numpy_exp = np.exp(arg)
    libm_exp = np.array([math.exp(v) for v in arg])

    differing = numpy_exp != libm_exp
    if not differing.any():
        return  # platform where they agree exactly; nothing to bound
    ulps = np.abs(numpy_exp - libm_exp)[differing] / np.spacing(libm_exp[differing])
    assert ulps.max() <= 1.0, (
        f"libm and NumPy exp differ by up to {ulps.max():.1f} ULP; the parity "
        "tolerances in this module assume 1"
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
    assert np.allclose(fast, slow, rtol=ULP_TOL, atol=ATOL["min/max"])


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
    assert np.allclose(fast, slow, rtol=ULP_TOL, atol=ATOL["probability"])


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
