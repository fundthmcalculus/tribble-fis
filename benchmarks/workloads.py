"""Deterministic benchmark workloads for the fuzzy-inference hot paths.

Every workload is fully seeded: the same commit and the same machine must
produce the same ``checksum`` on every run. That is what makes the numbers a
receipt rather than an anecdote -- an optimization is only allowed to change the
*time* column, never the checksum.

Three things are measured, because they are three different kinds of cost:

``forward-*``
    One :func:`tribblefis.gauss_math.tsk_firing_strengths` call. This is the
    inference kernel and it is also the innermost operation of every refinement
    fitness evaluation, so it dominates both prediction and training.

``refine-*``
    An end-to-end antecedent refinement. This is the actual *training* cost the
    project cares about, and it is the forward pass multiplied by tens of
    thousands of fitness evaluations plus the per-evaluation model rebuild.

``predict-*``
    The deployed scikit-learn entry point, so that a kernel speedup can be shown
    to survive the pandas/validation overhead around it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from tribblefis.gauss_data import (
    FeatureModel,
    GaussianMembership,
    GaussianMixtureModel,
    LabelModel,
    NormPair,
)


# ---------------------------------------------------------------------------
# Synthetic data + model construction.
# ---------------------------------------------------------------------------

def make_dataset(
    n_samples: int, n_features: int, n_labels: int, seed: int = 0
) -> tuple[pd.DataFrame, np.ndarray]:
    """A separable-ish Gaussian-blob classification set.

    Built directly rather than fit from a real corpus so the benchmark has no
    dependency on network access or on the clustering heuristics whose runtime
    we are *not* measuring here.
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, 3.0, size=(n_labels, n_features))
    y = rng.integers(0, n_labels, size=n_samples)
    X = centers[y] + rng.normal(0.0, 1.0, size=(n_samples, n_features))
    cols = [f"f{i}" for i in range(n_features)]
    return pd.DataFrame(X, columns=cols), y


def make_model(
    n_features: int, n_labels: int, n_mf: int, seed: int = 0
) -> GaussianMixtureModel:
    """A ``GaussianMixtureModel`` with ``n_mf`` Gaussians per (feature, label).

    Membership ids are drawn from a seeded UUID stream rather than
    :meth:`GaussianMembership.create`'s ``uuid4`` so that two runs of the same
    workload build byte-identical models.
    """
    rng = np.random.default_rng(seed + 1)
    feature_models: dict[str, FeatureModel] = {}
    for fi in range(n_features):
        label_models: dict[int, LabelModel] = {}
        for label in range(n_labels):
            mfs = [
                GaussianMembership(
                    mu=float(rng.normal(0.0, 3.0)),
                    sigma=float(rng.uniform(0.5, 2.0)),
                    id=uuid.UUID(bytes=rng.bytes(16)),
                )
                for _ in range(n_mf)
            ]
            label_models[label] = LabelModel(memberships=mfs)
        feature_models[f"f{fi}"] = FeatureModel(label_models=label_models)
    return GaussianMixtureModel(feature_models=feature_models)


# ---------------------------------------------------------------------------
# Workload definition.
# ---------------------------------------------------------------------------

@dataclass
class Workload:
    """One benchmarked operation.

    ``setup`` is untimed and runs once; ``run`` is what the timer wraps and is
    called repeatedly; ``checksum`` reduces ``run``'s output to a single float
    that must not move when the implementation underneath changes.
    """

    name: str
    description: str
    setup: Callable[[], Any]
    run: Callable[[Any], Any]
    checksum: Callable[[Any], float]
    # Refinement workloads take seconds each; the forward pass takes
    # milliseconds. Repeat counts are per-workload so both are measured
    # tightly without the suite taking an hour.
    repeats: int = 5
    warmups: int = 1
    tags: tuple[str, ...] = ()
    # Some workloads need hardware or an optional dependency the machine may not
    # have. They are skipped with a reason rather than dropped from the list, so
    # a results file always says whether a GPU row is missing because the GPU is
    # slow or because there was no GPU.
    available: Callable[[], tuple[bool, str]] = lambda: (True, "")


def _array_checksum(a: np.ndarray) -> float:
    """Order-sensitive reduction of an array to one float.

    A plain ``sum`` would be blind to a permutation of the columns, which is
    exactly the kind of mistake a rewritten kernel makes, so weight each element
    by its flat index.
    """
    flat = np.asarray(a, dtype=float).ravel()
    w = np.arange(1, flat.size + 1, dtype=float)
    return float(np.dot(np.nan_to_num(flat), w) / flat.size)


# ---------------------------------------------------------------------------
# Forward-pass workloads.
# ---------------------------------------------------------------------------

def _forward_workload(
    name: str, n_samples: int, n_features: int, n_labels: int, n_mf: int,
    norm: str, repeats: int,
) -> Workload:
    def setup():
        X, _ = make_dataset(n_samples, n_features, n_labels, seed=0)
        model = make_model(n_features, n_labels, n_mf, seed=0)
        # Pre-extracted columns: the deployed refinement path already does this
        # (see gauss_math.tsk_firing_strengths' `feature_arrays`), so including
        # the pandas lookup here would measure a cost the hot loop does not pay.
        feature_arrays = {c: X[c].to_numpy() for c in X.columns}
        return X, model, feature_arrays, NormPair(norm, norm)

    def run(state):
        from tribblefis.gauss_math import tsk_firing_strengths

        X, model, feature_arrays, norms = state
        fs, _labels = tsk_firing_strengths(
            X, model, norms=norms, feature_arrays=feature_arrays
        )
        return fs

    return Workload(
        name=name,
        description=(
            f"tsk_firing_strengths: {n_samples} samples x {n_features} features "
            f"x {n_labels} labels x {n_mf} MF, {norm} norms"
        ),
        setup=setup,
        run=run,
        checksum=_array_checksum,
        repeats=repeats,
        tags=("forward",),
    )


# ---------------------------------------------------------------------------
# GPU workloads.
# ---------------------------------------------------------------------------

def _cuda_available() -> tuple[bool, str]:
    try:
        from tribblefis import gpu
    except ImportError as exc:  # pragma: no cover
        return False, f"tribblefis.gpu unimportable: {exc}"
    if not gpu.is_available():
        return False, "PyTorch not installed"
    if not gpu.is_available(require_cuda=True):
        return False, "no CUDA device"
    return True, ""


def _resident_forward_workload(
    name: str, n_samples: int, n_features: int, n_labels: int, n_mf: int, repeats: int
) -> Workload:
    """CPU forward pass with the model compiled and the matrix built in `setup`.

    The GPU workloads hold their data on the device across calls, so this is what
    they have to be compared against: the same timing boundary, measuring the
    kernel rather than the marshalling. `forward-large` and friends deliberately
    include the pandas-facing path; this one deliberately does not.
    """

    def setup():
        from tribblefis import kernel

        X, _ = make_dataset(n_samples, n_features, n_labels, seed=0)
        model = make_model(n_features, n_labels, n_mf, seed=0)
        arrays = {c: X[c].to_numpy() for c in X.columns}
        compiled = kernel.compile_model(model, list(arrays))
        return compiled, compiled.feature_matrix(arrays), NormPair("min/max", "min/max")

    def run(state):
        from tribblefis import kernel

        compiled, matrix, norms = state
        return kernel.firing_strengths(compiled, matrix, norms)

    return Workload(
        name=name,
        description=(
            f"kernel.firing_strengths (resident data): {n_samples} samples x "
            f"{n_features} features x {n_labels} labels x {n_mf} MF"
        ),
        setup=setup,
        run=run,
        checksum=_array_checksum,
        repeats=repeats,
        tags=("forward", "resident"),
    )


def _cpu_batch_workload(
    name: str, n_samples: int, n_features: int, n_labels: int, n_mf: int,
    n_candidates: int, repeats: int,
) -> Workload:
    """The CPU counterpart of the batched GPU workload: the same `P` candidates,
    evaluated one at a time, which is all a CPU can do."""

    def setup():
        from tribblefis import kernel

        X, _ = make_dataset(n_samples, n_features, n_labels, seed=0)
        model = make_model(n_features, n_labels, n_mf, seed=0)
        arrays = {c: X[c].to_numpy() for c in X.columns}
        compiled = kernel.compile_model(model, list(arrays))
        return (compiled, compiled.feature_matrix(arrays),
                NormPair("min/max", "min/max"),
                _candidate_params(compiled, n_candidates))

    def run(state):
        from tribblefis import kernel

        compiled, matrix, norms, params = state
        return np.stack([
            kernel.firing_strengths(_with_params(compiled, p), matrix, norms)
            for p in params
        ])

    return Workload(
        name=name,
        description=(
            f"kernel.firing_strengths x {n_candidates} candidates (float64): "
            f"{n_samples} samples x {n_features} features x {n_labels} labels "
            f"x {n_mf} MF"
        ),
        setup=setup,
        run=run,
        checksum=_array_checksum,
        repeats=repeats,
        tags=("batch", "resident"),
    )


def _candidate_params(compiled, n_candidates: int) -> np.ndarray:
    """A seeded population of parameter vectors around the model's own."""
    rng = np.random.default_rng(0)
    base = compiled.extract_params()
    params = base + rng.normal(0.0, 0.25, size=(n_candidates, base.size))
    params[:, 1::2] = np.abs(params[:, 1::2]) + 0.1      # keep sigma positive
    return params


def _with_params(compiled, vec):
    compiled.set_params(vec)
    return compiled


def _gpu_forward_workload(
    name: str, n_samples: int, n_features: int, n_labels: int, n_mf: int,
    dtype: str, repeats: int,
) -> Workload:
    """A resident-data GPU forward pass.

    The sample matrix is uploaded in `setup`, so this times the kernel and the
    result download but not the upload -- matching how the backend is meant to be
    used (hold a `TorchFIS`, evaluate many times). `forward-huge` is the CPU
    row to compare against; it is the same shape and seed.
    """

    def setup():
        import torch
        from tribblefis import gpu, kernel

        X, _ = make_dataset(n_samples, n_features, n_labels, seed=0)
        model = make_model(n_features, n_labels, n_mf, seed=0)
        arrays = {c: X[c].to_numpy() for c in X.columns}
        compiled = kernel.compile_model(model, list(arrays))
        handle = gpu.TorchFIS(
            compiled, compiled.feature_matrix(arrays),
            NormPair("min/max", "min/max"), dtype=dtype,
        )
        return handle, torch

    def run(state):
        handle, torch = state
        out = handle.firing_strengths()
        torch.cuda.synchronize()
        return out

    return Workload(
        name=name,
        description=(
            f"TorchFIS.firing_strengths ({dtype}, resident data): {n_samples} "
            f"samples x {n_features} features x {n_labels} labels x {n_mf} MF"
        ),
        setup=setup,
        run=run,
        checksum=_array_checksum,
        repeats=repeats,
        tags=("forward", "gpu"),
        available=_cuda_available,
    )


def _gpu_batch_workload(
    name: str, n_samples: int, n_features: int, n_labels: int, n_mf: int,
    n_candidates: int, repeats: int, batched: bool = True,
) -> Workload:
    """`n_candidates` parameter vectors evaluated on the GPU.

    Run both ways on purpose. `batched=True` submits them as one tensor,
    `batched=False` loops. They measure the same to within noise -- the device is
    saturated by a single candidate at this size -- and keeping both rows is what
    stops "we added batching" from being read as "batching made it faster".

    The checksum covers all `P` results, so a batching bug that broadcast one
    candidate over the rest would be caught rather than rewarded with a speedup.
    """

    def setup():
        import torch
        from tribblefis import gpu, kernel

        X, _ = make_dataset(n_samples, n_features, n_labels, seed=0)
        model = make_model(n_features, n_labels, n_mf, seed=0)
        arrays = {c: X[c].to_numpy() for c in X.columns}
        compiled = kernel.compile_model(model, list(arrays))
        handle = gpu.TorchFIS(
            compiled, compiled.feature_matrix(arrays),
            NormPair("min/max", "min/max"), dtype="float32",
        )
        return handle, _candidate_params(compiled, n_candidates), torch

    def run(state):
        handle, params, torch = state
        if batched:
            out = handle.firing_strengths_batch(params)
        else:
            rows = []
            for p in params:
                handle.set_params(p)
                rows.append(handle.firing_strengths())
            out = np.stack(rows)
        torch.cuda.synchronize()
        return out

    how = "firing_strengths_batch" if batched else "firing_strengths in a loop"
    return Workload(
        name=name,
        description=(
            f"TorchFIS.{how} (float32): {n_candidates} candidates "
            f"x {n_samples} samples x {n_features} features x {n_labels} labels "
            f"x {n_mf} MF"
        ),
        setup=setup,
        run=run,
        checksum=_array_checksum,
        repeats=repeats,
        tags=("gpu", "batch"),
        available=_cuda_available,
    )


# ---------------------------------------------------------------------------
# Training (refinement) workloads.
# ---------------------------------------------------------------------------

def _refine_classifier_workload(
    name: str, n_samples: int, n_features: int, n_labels: int, n_mf: int,
    n_sweeps: int, repeats: int,
) -> Workload:
    def setup():
        X, y = make_dataset(n_samples, n_features, n_labels, seed=1)
        model = make_model(n_features, n_labels, n_mf, seed=1)
        return X, y, model

    def run(state):
        from tribblefis.refine import refine_classifier_antecedents

        X, y, model = state
        refined, info = refine_classifier_antecedents(
            model, X, y, method="coordinate", n_sweeps=n_sweeps,
            seed=42, verbose=False,
        )
        return refined, info

    def checksum(result):
        from tribblefis.refine import extract_gaussian_params

        refined, info = result
        # Both the parameters found and the objective reached must be stable:
        # a faster search that lands somewhere else is a different algorithm,
        # not a speedup, and this checksum is what says so.
        return _array_checksum(extract_gaussian_params(refined)) + float(
            info.get("train_obj", 0.0)
        )

    return Workload(
        name=name,
        description=(
            f"refine_classifier_antecedents(coordinate): {n_samples}x{n_features}, "
            f"{n_labels} labels, {n_mf} MF, {n_sweeps} sweeps "
            f"({2 * n_features * n_labels * n_mf} free params)"
        ),
        setup=setup,
        run=run,
        checksum=checksum,
        repeats=repeats,
        warmups=0,
        tags=("train",),
    )


# ---------------------------------------------------------------------------
# Interaction-score workload.
#
# `calculate_interaction_scores` is the new O(n_features^2) cost center this
# candidate-cross-term-detection feature introduces (every other cost in this
# file is O(n_features) or O(n_rules)) -- this workload is what keeps that
# growth honest as feature counts rise.
# ---------------------------------------------------------------------------

def _interaction_score_workload(
    name: str, n_samples: int, n_features: int, n_labels: int, repeats: int,
) -> Workload:
    def setup():
        X, y = make_dataset(n_samples, n_features, n_labels, seed=3)
        return X, pd.Series(y)

    def run(state):
        from tribblefis.gauss_math import calculate_gaussian_correlation, calculate_interaction_scores

        X, y = state
        diffs = calculate_gaussian_correlation(X, y)
        return calculate_interaction_scores(X, y, diffs)

    def checksum(result):
        # Order-sensitive: a re-ranking that changed which pair comes out on
        # top, not just how fast, must move this number.
        flat = np.array([lift for _fi, _fj, lift in result], dtype=float)
        return _array_checksum(flat)

    return Workload(
        name=name,
        description=(
            f"calculate_interaction_scores: {n_samples} samples x {n_features} "
            f"features x {n_labels} labels ({n_features * (n_features - 1) // 2} pairs)"
        ),
        setup=setup,
        run=run,
        checksum=checksum,
        repeats=repeats,
        tags=("interaction",),
    )


# ---------------------------------------------------------------------------
# Deployed-estimator workload.
# ---------------------------------------------------------------------------

def _predict_workload(
    name: str, n_samples: int, n_features: int, n_labels: int, repeats: int
) -> Workload:
    def setup():
        X, y = make_dataset(n_samples, n_features, n_labels, seed=2)
        model = make_model(n_features, n_labels, 3, seed=2)
        from tribblefis.gaussian_classifier import TribbleClassifier

        clf = TribbleClassifier()
        # Inject the synthetic model rather than fitting: `fit` runs KMeans/BIC
        # selection, which is a separate cost with its own optimizations and
        # would swamp the inference time this workload exists to measure.
        clf.model_ = model
        clf.classes_ = np.arange(n_labels)
        clf.feature_names_in_ = list(X.columns)
        clf.is_fitted_ = True
        return clf, X

    def run(state):
        clf, X = state
        return clf.predict_proba(X)

    return Workload(
        name=name,
        description=(
            f"TribbleClassifier.predict_proba: {n_samples} samples "
            f"x {n_features} features x {n_labels} labels x 3 MF"
        ),
        setup=setup,
        run=run,
        checksum=_array_checksum,
        repeats=repeats,
        tags=("predict",),
    )


# ---------------------------------------------------------------------------
# The suite.
# ---------------------------------------------------------------------------

def all_workloads() -> list[Workload]:
    return [
        # Small/wide/large sweep the two axes a kernel rewrite trades against:
        # per-call Python overhead (small) and per-element arithmetic (large).
        # Many repeats: once the compiled kernel took this workload under 200 us,
        # 20 repeats gave a run-to-run spread of +/-25% on the min, enough to
        # report a 0.79x "regression" on a code path nothing had touched. A
        # benchmark that noisy is not evidence.
        _forward_workload("forward-small", 1_000, 8, 3, 3, "min/max", repeats=300),
        _forward_workload("forward-wide", 2_000, 40, 6, 4, "min/max", repeats=10),
        _forward_workload("forward-large", 50_000, 20, 8, 4, "min/max", repeats=5),
        # The probability family is the one the analytic-gradient path uses, and
        # it is arithmetically heavier than min/max, so it is timed separately.
        _forward_workload("forward-prob", 20_000, 20, 8, 4, "probability", repeats=5),
        _predict_workload("predict-large", 50_000, 20, 8, repeats=5),
        # The headline training numbers. `refine-classifier` is small on purpose
        # -- it runs in under a second and catches regressions quickly -- but at
        # that size the forward pass is only about a fifth of the work, the rest
        # being SciPy's L-BFGS-B machinery over 48 tiny sub-problems. A model of
        # a size anyone would actually deploy inverts that ratio, so
        # `refine-classifier-wide` exists to keep optimizations honest about
        # which regime they help.
        _refine_classifier_workload(
            "refine-classifier", 1_000, 8, 3, 2, n_sweeps=2, repeats=3
        ),
        _refine_classifier_workload(
            "refine-classifier-wide", 4_000, 20, 6, 3, n_sweeps=1, repeats=2
        ),
        # Pairs grow as n_choose_2, not linearly like everything above --
        # "wide" here means more features, the axis this cost actually scales on.
        _interaction_score_workload("interaction-score", 800, 10, 4, repeats=5),
        _interaction_score_workload("interaction-score-wide", 800, 30, 4, repeats=3),
        # CPU/GPU pairs. Each `-cpu` row is the same shape, the same seed and the
        # same timing boundary as the `-gpu*` row beneath it, so the two can be
        # read against each other directly.
        _resident_forward_workload("forward-huge-cpu", 1_000_000, 20, 8, 4, repeats=3),
        _gpu_forward_workload("forward-huge-gpu64", 1_000_000, 20, 8, 4,
                              "float64", repeats=3),
        _gpu_forward_workload("forward-huge-gpu32", 1_000_000, 20, 8, 4,
                              "float32", repeats=3),
        _cpu_batch_workload("batch-candidates-cpu", 4_000, 20, 6, 3,
                            n_candidates=64, repeats=3),
        _gpu_batch_workload("batch-candidates-gpu", 4_000, 20, 6, 3,
                            n_candidates=64, repeats=3),
        _gpu_batch_workload("batch-candidates-gpu-seq", 4_000, 20, 6, 3,
                            n_candidates=64, repeats=3, batched=False),
    ]


def workloads_by_name(names: list[str] | None) -> list[Workload]:
    every = all_workloads()
    if not names:
        return every
    index = {w.name: w for w in every}
    missing = [n for n in names if n not in index]
    if missing:
        raise SystemExit(
            f"unknown workload(s) {missing}; available: {sorted(index)}"
        )
    return [index[n] for n in names]
