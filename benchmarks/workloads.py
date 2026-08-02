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
from dataclasses import dataclass
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
# Deployed-estimator workload.
# ---------------------------------------------------------------------------

def _predict_workload(
    name: str, n_samples: int, n_features: int, n_labels: int, repeats: int
) -> Workload:
    def setup():
        X, y = make_dataset(n_samples, n_features, n_labels, seed=2)
        model = make_model(n_features, n_labels, 3, seed=2)
        from tribblefis.gaussian_classifier import MixtureOfGaussiansFuzzyClassifier

        clf = MixtureOfGaussiansFuzzyClassifier()
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
            f"MixtureOfGaussiansFuzzyClassifier.predict_proba: {n_samples} samples "
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
        _forward_workload("forward-small", 1_000, 8, 3, 3, "min/max", repeats=20),
        _forward_workload("forward-wide", 2_000, 40, 6, 4, "min/max", repeats=10),
        _forward_workload("forward-large", 50_000, 20, 8, 4, "min/max", repeats=5),
        # The probability family is the one the analytic-gradient path uses, and
        # it is arithmetically heavier than min/max, so it is timed separately.
        _forward_workload("forward-prob", 20_000, 20, 8, 4, "probability", repeats=5),
        _predict_workload("predict-large", 50_000, 20, 8, repeats=5),
        # The headline training number. Small on purpose: at the default
        # settings this already runs tens of thousands of forward passes.
        _refine_classifier_workload(
            "refine-classifier", 1_000, 8, 3, 2, n_sweeps=2, repeats=3
        ),
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
