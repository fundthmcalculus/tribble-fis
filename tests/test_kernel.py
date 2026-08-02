"""The compiled kernel must be *bit*-identical to the reference forward pass.

Not "close": identical. The whole optimization stack is measured by benchmark
checksums, and a kernel that is merely close would make every one of those
checksums a judgement call. Every assertion here is therefore
``np.array_equal``, not ``np.allclose``.
"""

import uuid

import numpy as np
import pandas as pd
import pytest

from tribblefis import kernel as K
from tribblefis.gauss_data import (
    FeatureModel,
    GaussianMembership,
    GaussianMixtureModel,
    LabelModel,
    NormPair,
    TrapezoidMembership,
)
from tribblefis.gauss_math import tsk_firing_strengths
from tribblefis.refine import apply_gaussian_params, extract_gaussian_params

NORMS = ["min/max", "probability", "luk", "hamacher", "einstein"]
SHAPES = [
    (500, 3, 2, 1),     # single membership per cell
    (1000, 8, 3, 3),
    (997, 5, 4, 2),     # sample count not a multiple of the block size
    (2000, 12, 6, 4),
]


def _build(n_samples, n_features, n_labels, n_mf, seed=0, ragged=False):
    """A random model plus matching data. `ragged` gives one cell fewer
    memberships than the others, which is what exercises the padding mask."""
    rng = np.random.default_rng(seed)
    cols = [f"f{i}" for i in range(n_features)]
    X = pd.DataFrame(rng.normal(0, 2, size=(n_samples, n_features)), columns=cols)

    feature_models = {}
    for fi, name in enumerate(cols):
        label_models = {}
        for label in range(n_labels):
            k = n_mf
            if ragged and fi == 0 and label == 0 and n_mf > 1:
                k = n_mf - 1
            label_models[label] = LabelModel(memberships=[
                GaussianMembership(mu=float(rng.normal(0, 2)),
                                   sigma=float(rng.uniform(0.3, 2.0)),
                                   id=uuid.UUID(bytes=rng.bytes(16)))
                for _ in range(k)
            ])
        feature_models[name] = FeatureModel(label_models=label_models)
    return X, GaussianMixtureModel(feature_models=feature_models)


@pytest.mark.parametrize("norm", NORMS)
@pytest.mark.parametrize("shape", SHAPES)
def test_kernel_is_bit_identical_to_reference(norm, shape):
    X, model = _build(*shape, seed=hash((norm, shape)) % 1000)
    feature_arrays = {c: X[c].to_numpy() for c in X.columns}
    norms = NormPair(norm, norm)

    expected, labels = tsk_firing_strengths(
        X, model, norms=norms, feature_arrays=feature_arrays
    )
    compiled = K.compile_model(model, list(feature_arrays))
    got = K.firing_strengths(compiled, compiled.feature_matrix(feature_arrays), norms)

    assert list(compiled.labels) == list(labels)
    assert np.array_equal(expected, got)


@pytest.mark.parametrize("norm", NORMS)
def test_padded_cells_do_not_perturb_the_fold(norm):
    """A ragged model pads to a rectangular block; the padding must be exactly
    the t-conorm identity, or short cells would silently read different."""
    X, model = _build(800, 5, 3, 3, seed=11, ragged=True)
    feature_arrays = {c: X[c].to_numpy() for c in X.columns}
    norms = NormPair(norm, norm)

    expected, _ = tsk_firing_strengths(X, model, norms=norms, feature_arrays=feature_arrays)
    compiled = K.compile_model(model, list(feature_arrays))
    assert not compiled.is_dense
    got = K.firing_strengths(compiled, compiled.feature_matrix(feature_arrays), norms)
    assert np.array_equal(expected, got)


def test_mixed_norm_pair_is_honoured():
    """The t-norm and t-conorm are folded independently, so a mixed pair must
    still match -- this catches a kernel that reads one setting for both."""
    X, model = _build(600, 6, 4, 2, seed=5)
    feature_arrays = {c: X[c].to_numpy() for c in X.columns}
    norms = NormPair(t_norm="probability", t_conorm="min/max")
    expected, _ = tsk_firing_strengths(X, model, norms=norms, feature_arrays=feature_arrays)
    compiled = K.compile_model(model, list(feature_arrays))
    got = K.firing_strengths(compiled, compiled.feature_matrix(feature_arrays), norms)
    assert np.array_equal(expected, got)


def test_blocking_does_not_change_the_result(monkeypatch):
    """Row blocking is a cache optimization and must be invisible; force a tiny
    block so many blocks run, and a full-size one so none does."""
    X, model = _build(1500, 4, 3, 2, seed=6)
    feature_arrays = {c: X[c].to_numpy() for c in X.columns}
    norms = NormPair("min/max", "min/max")
    compiled = K.compile_model(model, list(feature_arrays))
    matrix = compiled.feature_matrix(feature_arrays)

    monkeypatch.setattr(K, "_BLOCK_BYTES", 1 << 30)
    one_block = K.firing_strengths(compiled, matrix, norms)
    monkeypatch.setattr(K, "_BLOCK_BYTES", 1)
    many_blocks = K.firing_strengths(compiled, matrix, norms)
    assert np.array_equal(one_block, many_blocks)


# ---------------------------------------------------------------------------
# What compile_model refuses, and why.
# ---------------------------------------------------------------------------

def test_rejects_non_gaussian_memberships():
    _, model = _build(10, 2, 2, 1, seed=1)
    fname = next(iter(model.feature_models))
    fmodel = model.feature_models[fname]
    label = next(iter(fmodel.label_models))
    fmodel.label_models[label] = LabelModel(
        memberships=[TrapezoidMembership.create(0.0, 1.0, 2.0, 3.0)]
    )
    assert not K.can_compile(model)
    with pytest.raises(K.NotCompilable, match="non-Gaussian"):
        K.compile_model(model)


def test_rejects_ragged_label_sets():
    """A feature missing a label is skipped by the reference loop. That is the
    t-norm's identity mathematically, but under Lukasiewicz ``max(0, z + 1 - 1)``
    is not ``z`` for tiny ``z``, so the flat layout declines rather than risk it."""
    _, model = _build(10, 2, 3, 1, seed=2)
    fname = list(model.feature_models)[1]
    del model.feature_models[fname].label_models[2]
    assert not K.can_compile(model)
    with pytest.raises(K.NotCompilable, match="labels"):
        K.compile_model(model)


def test_rejects_a_feature_with_no_data_column():
    _, model = _build(10, 3, 2, 1, seed=3)
    assert K.can_compile(model, ["f0", "f1", "f2"])
    assert not K.can_compile(model, ["f0", "f1"])
    with pytest.raises(K.NotCompilable, match="no data column"):
        K.compile_model(model, ["f0", "f1"])


def test_rejects_an_empty_model():
    with pytest.raises(K.NotCompilable):
        K.compile_model(GaussianMixtureModel(feature_models={}))


# ---------------------------------------------------------------------------
# Parameter round-trip against the reference implementation in `refine`.
# ---------------------------------------------------------------------------

def test_extract_params_matches_refine_order():
    _, model = _build(10, 4, 3, 2, seed=4)
    compiled = K.compile_model(model)
    assert np.array_equal(extract_gaussian_params(model), compiled.extract_params())


def test_set_params_matches_apply_gaussian_params():
    _, model = _build(10, 4, 3, 2, seed=7)
    compiled = K.compile_model(model)
    rng = np.random.default_rng(0)
    vec = extract_gaussian_params(model) + rng.normal(0, 0.3, size=compiled.n_slots * 2)

    compiled.set_params(vec)
    expected = K.compile_model(apply_gaussian_params(model, vec))
    assert np.array_equal(compiled.mu, expected.mu)
    assert np.array_equal(compiled.sigma, expected.sigma)


def test_set_params_floors_sigma_like_the_reference():
    _, model = _build(10, 2, 2, 1, seed=8)
    compiled = K.compile_model(model)
    vec = compiled.extract_params()
    vec[1::2] = -5.0                      # every sigma degenerate
    compiled.set_params(vec)
    assert np.all(compiled.sigma[compiled.active > 0] == 1e-6)


def test_to_model_round_trips_and_keeps_ids():
    _, model = _build(10, 3, 2, 2, seed=9)
    compiled = K.compile_model(model)
    rng = np.random.default_rng(1)
    vec = extract_gaussian_params(model) + rng.normal(0, 0.2, size=compiled.n_slots * 2)
    compiled.set_params(vec)
    assert K.to_model(compiled, model) == apply_gaussian_params(model, vec)


def test_set_params_rejects_a_wrong_length_vector():
    _, model = _build(10, 2, 2, 1, seed=10)
    compiled = K.compile_model(model)
    with pytest.raises(ValueError, match="expected"):
        compiled.set_params(np.zeros(3))


def test_labels_are_sorted_even_when_insertion_order_is_not():
    """Firing-strength columns follow `ordered_keys` (sorted) while the refine
    parameter vector follows insertion order. When they disagree, `slot_index`
    is the only thing keeping the two aligned."""
    rng = np.random.default_rng(12)

    def mf():
        return GaussianMembership(mu=float(rng.normal()), sigma=1.0,
                                  id=uuid.UUID(bytes=rng.bytes(16)))

    # Insert labels 2, 0, 1 -- deliberately unsorted.
    model = GaussianMixtureModel(feature_models={
        "f0": FeatureModel(label_models={
            lab: LabelModel(memberships=[mf()]) for lab in (2, 0, 1)
        })
    })
    compiled = K.compile_model(model)
    assert list(compiled.labels) == [0, 1, 2]
    assert np.array_equal(extract_gaussian_params(model), compiled.extract_params())

    X = pd.DataFrame({"f0": rng.normal(0, 2, size=200)})
    norms = NormPair("min/max", "min/max")
    expected, labels = tsk_firing_strengths(X, model, norms=norms)
    got = K.firing_strengths(compiled, compiled.feature_matrix({"f0": X["f0"].to_numpy()}), norms)
    assert list(labels) == list(compiled.labels)
    assert np.array_equal(expected, got)
