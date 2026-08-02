"""The benchmark harness is part of the repo's contract, so it gets a test.

These are correctness checks on the harness, not timings: a benchmark that has
silently stopped exercising the code it claims to exercise is worse than no
benchmark, because its numbers still look like receipts.
"""

import numpy as np
import pytest

from benchmarks.bench import _checksums_match, render_table, run_workload
from benchmarks.workloads import all_workloads, make_dataset, make_model


def test_make_dataset_and_model_are_deterministic():
    X1, y1 = make_dataset(50, 4, 3, seed=7)
    X2, y2 = make_dataset(50, 4, 3, seed=7)
    assert np.array_equal(X1.to_numpy(), X2.to_numpy())
    assert np.array_equal(y1, y2)

    m1 = make_model(4, 3, 2, seed=7)
    m2 = make_model(4, 3, 2, seed=7)
    # Membership ids are seeded too, so the models compare equal outright --
    # without that, refinement checksums would drift run to run.
    assert m1 == m2


def test_model_shape_matches_request():
    model = make_model(n_features=5, n_labels=3, n_mf=2, seed=0)
    assert len(model.feature_models) == 5
    for fmodel in model.feature_models.values():
        assert sorted(fmodel.label_models) == [0, 1, 2]
        for lmodel in fmodel.label_models.values():
            assert len(lmodel.memberships) == 2


@pytest.mark.parametrize("name", ["forward-small", "predict-large", "refine-classifier"])
def test_workload_checksum_is_reproducible(name):
    """Two runs of the same workload must agree exactly.

    This is the property the whole suite rests on: `--compare` reads a changed
    checksum as "the optimization changed the answer", which is only a useful
    signal if an unchanged implementation reproduces its checksum.
    """
    workload = next(w for w in all_workloads() if w.name == name)
    # Shrink the run so the test suite stays fast; repeats do not affect the
    # checksum, only how many times it is recomputed.
    workload.repeats = 1
    workload.warmups = 0
    first = run_workload(workload)
    second = run_workload(workload)
    assert first["checksum"] == second["checksum"]
    assert first["min_s"] > 0


def test_checksum_comparison_tolerates_last_bit_drift_only():
    assert _checksums_match(1.0, 1.0 + 1e-15)
    assert not _checksums_match(1.0, 1.0 + 1e-6)
    assert _checksums_match(0.0, 0.0)


def test_render_table_flags_a_changed_checksum():
    results = [{"name": "w", "min_s": 1.0, "median_s": 1.0, "checksum": 2.0}]
    baseline = {"results": [{"name": "w", "min_s": 2.0, "checksum": 1.0}]}
    table = render_table(results, baseline)
    assert "CHANGED" in table
    assert "2.00x" in table


def test_all_workload_names_are_unique():
    names = [w.name for w in all_workloads()]
    assert len(names) == len(set(names))
