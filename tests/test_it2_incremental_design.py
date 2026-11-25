"""Premises of docs/it2-gt2-acceleration-design.md (issue #146).

The design ports IT2 refinement onto two `kernel.IncrementalFIS` objects
(upper and lower). That is only sound if (1) the upper and lower Type-1
models compile to the *same* layout, so one IT2 slot maps to one slot index
in both, and (2) the incremental path reproduces the full forward pass the
refiner uses today.
"""

import numpy as np
import pandas as pd

from benchmarks.it2_incremental_probe import _build
from benchmarks.it2_incremental_probe import main as probe_main
from tribblefis import kernel
from tribblefis.gauss_data import DefaultNormCornorm, resolve_norm_pair
from tribblefis.gauss_math import tsk_firing_strengths
from tribblefis.it2_kernel import _extract_lower_model, _extract_upper_model


def _setup():
    rng = np.random.default_rng(1)
    model = _build(n_features=4, n_labels=3, n_mf=2, rng=rng)
    X = pd.DataFrame(rng.normal(size=(200, 4)), columns=[f"x{i}" for i in range(4)])
    arrays = {c: X[c].to_numpy() for c in X.columns}
    norms = resolve_norm_pair(DefaultNormCornorm)
    return model, X, arrays, norms


def test_upper_and_lower_compile_to_one_layout():
    model, X, _, _ = _setup()
    up = kernel.compile_model(_extract_upper_model(model), list(X.columns))
    lo = kernel.compile_model(_extract_lower_model(model), list(X.columns))
    assert up.feature_names == lo.feature_names
    assert up.labels == lo.labels
    assert up.mu.shape == lo.mu.shape
    np.testing.assert_array_equal(up.slot_index, lo.slot_index)
    np.testing.assert_array_equal(up.mu, lo.mu)  # shared mu, per the IT2 FOU
    assert np.all(up.sigma >= lo.sigma)


def test_incremental_matches_full_rebuild_for_both_bounds():
    model, X, arrays, norms = _setup()
    for extract in (_extract_upper_model, _extract_lower_model):
        t1 = extract(model)
        compiled = kernel.compile_model(t1, list(X.columns))
        fm = compiled.feature_matrix(arrays)
        inc = kernel.IncrementalFIS(compiled, fm, norms)
        # Unperturbed: incremental cache == today's rebuild-and-evaluate path.
        expected, _ = tsk_firing_strengths(X, t1, norms=norms, feature_arrays=arrays)
        np.testing.assert_allclose(inc.base, expected, rtol=0, atol=1e-12)
        # Perturbed: moving one slot == a full pass with that slot moved.
        base = compiled.extract_params()
        for slot in range(compiled.n_slots):
            mu, sigma = base[2 * slot] + 0.3, base[2 * slot + 1] * 1.4
            got = inc.evaluate_slot(slot, mu, sigma)
            moved = compiled.copy()
            vec = base.copy()
            vec[2 * slot], vec[2 * slot + 1] = mu, sigma
            moved.set_params(vec)
            want = kernel.firing_strengths(moved, fm, norms, backend="numpy")
            np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def test_probe_runs():
    assert probe_main(["--n", "200", "--features", "3", "--labels", "2",
                       "--mf", "2", "--candidates", "2", "--repeats", "1"]) == 0
