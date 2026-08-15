"""Parity tests for the Torch backend.

This backend is explicitly *not* bit-exact -- CUDA's ``exp`` differs from libm's
by about an ULP -- so parity here is a tight tolerance rather than equality. What
must be exact is everything around the arithmetic: label ordering, the padding
mask, chunk boundaries, and the mapping from a batched parameter row to the
result row it produced.

Most tests run on whatever device is available, including CPU-only PyTorch, so
they still have something to say on a machine with no GPU. Tests that exist to
measure or exercise CUDA specifically are marked.
"""

import uuid

import numpy as np
import pandas as pd
import pytest

from tribblefis import gpu, kernel as K
from tribblefis.gauss_data import (
    FeatureModel,
    GaussianMembership,
    GaussianMixtureModel,
    LabelModel,
    NormPair,
)

requires_torch = pytest.mark.skipif(
    not gpu.is_available(), reason="PyTorch not installed"
)
requires_cuda = pytest.mark.skipif(
    not gpu.is_available(require_cuda=True), reason="no CUDA device"
)

NORMS = ["min/max", "probability", "luk", "hamacher", "einstein"]

# One ULP of float64 is ~2.2e-16; allow a small multiple to cover the accumulated
# difference through a fold. Anything looser would stop catching real bugs.
F64_TOL = 1e-12
F32_TOL = 2e-5


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


def _setup(n_samples=400, n_features=5, n_labels=3, n_mf=2, seed=0, ragged=False):
    X, model = _build(n_samples, n_features, n_labels, n_mf, seed=seed, ragged=ragged)
    arrays = {c: X[c].to_numpy() for c in X.columns}
    compiled = K.compile_model(model, list(arrays))
    return compiled, compiled.feature_matrix(arrays)


@requires_torch
@pytest.mark.parametrize("norm", NORMS)
def test_torch_float64_matches_the_cpu_kernel(norm):
    compiled, matrix = _setup(seed=1)
    norms = NormPair(norm, norm)
    expected = K.firing_strengths(compiled, matrix, norms)
    got = gpu.TorchFIS(compiled, matrix, norms).firing_strengths()
    assert got.shape == expected.shape
    assert np.allclose(expected, got, rtol=F64_TOL, atol=1e-14)


@requires_torch
def test_torch_float32_is_close_but_not_exact():
    """float32 is a real numerical change, not a rounding detail. Asserting the
    looser bound *and* that it is genuinely looser keeps anyone from assuming
    float32 is free."""
    compiled, matrix = _setup(n_samples=2000, seed=2)
    norms = NormPair("min/max", "min/max")
    expected = K.firing_strengths(compiled, matrix, norms)
    got = gpu.TorchFIS(compiled, matrix, norms, dtype="float32").firing_strengths()
    assert np.allclose(expected, got, rtol=F32_TOL, atol=1e-6)


@requires_torch
def test_torch_handles_padded_cells():
    compiled, matrix = _setup(n_samples=300, n_features=4, n_labels=3, n_mf=3,
                              seed=3, ragged=True)
    assert not compiled.is_dense
    norms = NormPair("probability", "probability")
    expected = K.firing_strengths(compiled, matrix, norms)
    got = gpu.TorchFIS(compiled, matrix, norms).firing_strengths()
    assert np.allclose(expected, got, rtol=F64_TOL, atol=1e-14)


@requires_torch
def test_torch_honours_a_mixed_norm_pair():
    compiled, matrix = _setup(n_samples=500, n_features=6, n_labels=4, seed=4)
    norms = NormPair(t_norm="probability", t_conorm="min/max")
    expected = K.firing_strengths(compiled, matrix, norms)
    got = gpu.TorchFIS(compiled, matrix, norms).firing_strengths()
    assert np.allclose(expected, got, rtol=F64_TOL, atol=1e-14)


@requires_torch
def test_chunking_is_invisible(monkeypatch):
    """Large inputs are chunked along the sample axis to bound device memory.
    A chunk boundary must not be observable in the output."""
    compiled, matrix = _setup(n_samples=1500, n_features=4, n_labels=3, seed=5)
    norms = NormPair("min/max", "min/max")
    whole = gpu.TorchFIS(compiled, matrix, norms).firing_strengths()
    monkeypatch.setattr(gpu, "_CHUNK_BYTES", 1)   # forces one chunk per row
    chunked = gpu.TorchFIS(compiled, matrix, norms).firing_strengths()
    assert np.array_equal(whole, chunked)


@requires_torch
def test_set_params_reaches_the_device():
    compiled, matrix = _setup(seed=6)
    norms = NormPair("min/max", "min/max")
    handle = gpu.TorchFIS(compiled, matrix, norms)
    before = handle.firing_strengths()

    vec = compiled.extract_params() + 0.5
    handle.set_params(vec)
    after = handle.firing_strengths()
    assert not np.allclose(before, after)

    # `set_params` writes through to the shared CompiledFIS, so the CPU kernel
    # now reads the same parameters the device does.
    assert np.allclose(
        after, K.firing_strengths(compiled, matrix, norms), rtol=F64_TOL, atol=1e-14
    )


@requires_torch
def test_batch_rows_match_evaluating_each_candidate_alone():
    """The batched path's whole risk is mixing candidates up. Each row must equal
    what that candidate produces on its own."""
    compiled, matrix = _setup(n_samples=250, n_features=4, n_labels=3, n_mf=2, seed=7)
    norms = NormPair("min/max", "min/max")
    handle = gpu.TorchFIS(compiled.copy(), matrix, norms)

    rng = np.random.default_rng(8)
    base = compiled.extract_params()
    params = base + rng.normal(0, 0.3, size=(5, base.size))
    params[:, 1::2] = np.abs(params[:, 1::2]) + 0.1

    batched = handle.firing_strengths_batch(params)
    assert batched.shape == (5, matrix.shape[0], compiled.n_labels)
    for p in range(5):
        single = compiled.copy()
        single.set_params(params[p])
        expected = K.firing_strengths(single, matrix, norms)
        assert np.allclose(batched[p], expected, rtol=F64_TOL, atol=1e-14), (
            f"candidate {p}"
        )


@requires_torch
def test_batch_rejects_a_mis_shaped_parameter_matrix():
    compiled, matrix = _setup(seed=9)
    handle = gpu.TorchFIS(compiled, matrix, NormPair("min/max", "min/max"))
    with pytest.raises(ValueError, match="param_matrix"):
        handle.firing_strengths_batch(np.zeros((3, 5)))


@requires_torch
def test_rejects_an_unsupported_dtype():
    compiled, matrix = _setup(seed=10)
    with pytest.raises(ValueError, match="dtype"):
        gpu.TorchFIS(compiled, matrix, NormPair("min/max", "min/max"), dtype="float16")


@requires_torch
def test_rejects_a_mis_shaped_feature_matrix():
    compiled, _ = _setup(seed=11)
    with pytest.raises(ValueError, match="feature_matrix"):
        gpu.TorchFIS(compiled, np.zeros((10, 99)), NormPair("min/max", "min/max"))


@requires_torch
def test_kernel_backend_torch_routes_here():
    compiled, matrix = _setup(seed=12)
    norms = NormPair("min/max", "min/max")
    expected = K.firing_strengths(compiled, matrix, norms)
    got = K.firing_strengths(compiled, matrix, norms, backend="torch")
    assert np.allclose(expected, got, rtol=F64_TOL, atol=1e-14)


def test_torch_is_never_chosen_automatically():
    """`auto` must resolve to a CPU backend, never the GPU one. If this ever
    flips, every benchmark checksum in the repo silently becomes a different
    number.

    Asserted as "auto's output is bit-identical to one of the two CPU backends",
    which is what the claim actually is. It used to be asserted against the NumPy
    backend alone, so it also silently required libm's `exp` and NumPy's to
    agree bit for bit -- they differ by 1 ULP on NumPy 2.4 with AVX-512, and
    this failed there for a reason that has nothing to do with Torch. Both CPU
    backends are exact candidates because `auto` picks between them on a size
    heuristic (`_cython_is_faster`), so either answer proves Torch was not used.
    """
    compiled, matrix = _setup(n_samples=100, seed=13)
    norms = NormPair("min/max", "min/max")
    auto = K.firing_strengths(compiled, matrix, norms, backend="auto")

    candidates = [K.firing_strengths_numpy(compiled, matrix, norms)]
    if K.HAVE_CYTHON_KERNEL:
        candidates.append(
            K.firing_strengths(compiled, matrix, norms, backend="cython")
        )
    assert any(np.array_equal(auto, cpu) for cpu in candidates)


def test_backend_validation_lists_torch():
    compiled, matrix = _setup(n_samples=50, seed=14)
    with pytest.raises(ValueError, match="torch"):
        K.firing_strengths(compiled, matrix, NormPair("min/max", "min/max"),
                           backend="opencl")


@pytest.mark.skipif(gpu.is_available(), reason="PyTorch is installed here")
def test_absent_torch_reports_clearly():  # pragma: no cover - env dependent
    compiled, matrix = _setup(n_samples=20, seed=15)
    with pytest.raises(RuntimeError, match="PyTorch"):
        gpu.TorchFIS(compiled, matrix, NormPair("min/max", "min/max"))


@requires_cuda
def test_cuda_is_the_default_device_when_present():
    assert gpu.default_device() == "cuda"
    compiled, matrix = _setup(seed=16)
    handle = gpu.TorchFIS(compiled, matrix, NormPair("min/max", "min/max"))
    assert handle.device.type == "cuda"
