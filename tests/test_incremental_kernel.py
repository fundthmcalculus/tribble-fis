"""The incremental evaluator must be indistinguishable from a full forward pass.

It is a cache, and a cache is only sound if every read is what a recompute would
have produced. So these tests never check "close" and never check a single call
in isolation: they check bit-equality after long sequences of proposals,
rejections and commits, which is the only way a stale entry actually shows up.
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
)

NORMS = ["min/max", "probability", "luk", "hamacher", "einstein"]


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


def _setup(n_samples=400, n_features=5, n_labels=3, n_mf=2, seed=0, norm="min/max",
           ragged=False):
    X, model = _build(n_samples, n_features, n_labels, n_mf, seed=seed, ragged=ragged)
    arrays = {c: X[c].to_numpy() for c in X.columns}
    compiled = K.compile_model(model, list(arrays))
    matrix = compiled.feature_matrix(arrays)
    return compiled, matrix, NormPair(norm, norm)


def _full(compiled, matrix, norms):
    """A fresh full pass from a *copy*, so it cannot be affected by the cache."""
    return K.firing_strengths(compiled.copy(), matrix, norms)


@pytest.mark.parametrize("norm", NORMS)
def test_base_matches_a_full_pass(norm):
    compiled, matrix, norms = _setup(norm=norm, seed=1)
    inc = K.IncrementalFIS(compiled, matrix, norms)
    assert np.array_equal(inc.base, _full(compiled, matrix, norms))


@pytest.mark.parametrize("norm", NORMS)
def test_a_proposal_matches_recomputing_that_model(norm):
    compiled, matrix, norms = _setup(norm=norm, seed=2)
    inc = K.IncrementalFIS(compiled, matrix, norms)
    rng = np.random.default_rng(5)

    for slot in range(compiled.n_slots):
        mu, sigma = float(rng.normal(0, 2)), float(rng.uniform(0.3, 2.0))
        got = inc.evaluate_slot(slot, mu, sigma).copy()

        reference = compiled.copy()
        vec = reference.extract_params()
        vec[2 * slot], vec[2 * slot + 1] = mu, sigma
        reference.set_params(vec)
        assert np.array_equal(got, K.firing_strengths(reference, matrix, norms)), (
            f"slot {slot} under {norm}"
        )


@pytest.mark.parametrize("norm", NORMS)
def test_a_rejected_proposal_leaves_no_trace(norm):
    """The cache must only move on commit; otherwise a sub-problem that tries 25
    candidates and accepts none would corrupt every later block."""
    compiled, matrix, norms = _setup(norm=norm, seed=3)
    inc = K.IncrementalFIS(compiled, matrix, norms)
    before_base = inc.base.copy()
    before_cells = inc.cells.copy()
    before_mu = compiled.mu.copy()

    rng = np.random.default_rng(6)
    for slot in range(compiled.n_slots):
        inc.evaluate_slot(slot, float(rng.normal(0, 5)), float(rng.uniform(0.1, 3.0)))

    assert np.array_equal(inc.base, before_base)
    assert np.array_equal(inc.cells, before_cells)
    assert np.array_equal(compiled.mu, before_mu)


@pytest.mark.parametrize("norm", NORMS)
def test_a_long_accept_reject_sequence_stays_exact(norm):
    """The real usage pattern: many proposals, some committed, over many slots.
    Drift shows up here or nowhere."""
    compiled, matrix, norms = _setup(n_samples=600, n_features=6, n_labels=4, n_mf=3,
                                     seed=4, norm=norm)
    inc = K.IncrementalFIS(compiled, matrix, norms)
    rng = np.random.default_rng(7)

    for step in range(120):
        slot = int(rng.integers(0, compiled.n_slots))
        mu, sigma = float(rng.normal(0, 2)), float(rng.uniform(0.3, 2.0))
        inc.evaluate_slot(slot, mu, sigma)
        if step % 3 == 0:
            inc.commit()
            # After a commit the cached state must equal a full recompute of the
            # model the commit produced.
            assert np.array_equal(inc.base, _full(compiled, matrix, norms)), (
                f"drift after commit at step {step} under {norm}"
            )


def test_commit_updates_the_compiled_parameters():
    compiled, matrix, norms = _setup(seed=8)
    inc = K.IncrementalFIS(compiled, matrix, norms)
    inc.evaluate_slot(3, 1.25, 0.75)
    inc.commit()
    params = compiled.extract_params()
    assert params[6] == 1.25
    assert params[7] == 0.75


def test_commit_floors_sigma_like_everything_else():
    compiled, matrix, norms = _setup(seed=9)
    inc = K.IncrementalFIS(compiled, matrix, norms)
    inc.evaluate_slot(0, 0.0, -3.0)
    inc.commit()
    assert compiled.extract_params()[1] == 1e-6


def test_commit_without_a_proposal_is_an_error():
    compiled, matrix, norms = _setup(seed=10)
    inc = K.IncrementalFIS(compiled, matrix, norms)
    with pytest.raises(RuntimeError, match="commit"):
        inc.commit()


def test_refresh_picks_up_out_of_band_parameter_changes():
    """A full-objective evaluation writes parameters straight into the compiled
    model. The cache has to be told, or it silently answers for the old ones."""
    compiled, matrix, norms = _setup(seed=11)
    inc = K.IncrementalFIS(compiled, matrix, norms)
    vec = compiled.extract_params()
    vec += 0.4
    compiled.set_params(vec)
    assert not np.array_equal(inc.base, _full(compiled, matrix, norms))
    inc.refresh()
    assert np.array_equal(inc.base, _full(compiled, matrix, norms))


def test_handles_ragged_cells():
    compiled, matrix, norms = _setup(n_samples=300, n_features=4, n_labels=3, n_mf=3,
                                     seed=12, ragged=True)
    assert not compiled.is_dense
    inc = K.IncrementalFIS(compiled, matrix, norms)
    assert np.array_equal(inc.base, _full(compiled, matrix, norms))
    inc.evaluate_slot(1, 0.5, 1.5)
    inc.commit()
    assert np.array_equal(inc.base, _full(compiled, matrix, norms))


def test_slot_decoding_survives_unsorted_label_insertion():
    """`slot_index` is the only thing tying refine's parameter order to the
    array layout when insertion order and sorted order disagree; a decode bug
    would move the wrong membership function."""
    rng = np.random.default_rng(13)

    def mf():
        return GaussianMembership(mu=float(rng.normal()), sigma=1.0,
                                  id=uuid.UUID(bytes=rng.bytes(16)))

    model = GaussianMixtureModel(feature_models={
        f"f{i}": FeatureModel(label_models={
            lab: LabelModel(memberships=[mf(), mf()]) for lab in (2, 0, 1)
        })
        for i in range(3)
    })
    X = pd.DataFrame({f"f{i}": rng.normal(0, 2, size=250) for i in range(3)})
    arrays = {c: X[c].to_numpy() for c in X.columns}
    compiled = K.compile_model(model, list(arrays))
    matrix = compiled.feature_matrix(arrays)
    norms = NormPair("min/max", "min/max")
    inc = K.IncrementalFIS(compiled, matrix, norms)

    for slot in range(compiled.n_slots):
        got = inc.evaluate_slot(slot, 0.3 * slot, 0.9).copy()
        reference = compiled.copy()
        vec = reference.extract_params()
        vec[2 * slot], vec[2 * slot + 1] = 0.3 * slot, 0.9
        reference.set_params(vec)
        assert np.array_equal(got, K.firing_strengths(reference, matrix, norms))


@pytest.mark.skipif(not K.HAVE_CYTHON_KERNEL, reason="compiled kernel not built")
def test_numpy_and_compiled_incremental_agree(monkeypatch):
    """Both backends implement the cache; they must not disagree about it."""
    compiled, matrix, norms = _setup(n_samples=500, n_features=5, n_labels=3, n_mf=2,
                                     seed=14, norm="probability")
    fast = K.IncrementalFIS(compiled.copy(), matrix, norms)
    monkeypatch.setattr(K, "HAVE_CYTHON_KERNEL", False)
    slow = K.IncrementalFIS(compiled.copy(), matrix, norms)

    assert np.allclose(fast.base, slow.base, rtol=1e-13, atol=0.0)
    for slot in (0, 3, 7):
        a = fast.evaluate_slot(slot, 0.5, 1.1).copy()
        b = slow.evaluate_slot(slot, 0.5, 1.1).copy()
        assert np.allclose(a, b, rtol=1e-13, atol=0.0)
