"""Refinement must take the *same* search trajectory with the compiled kernel
as without it.

The compiled path changes how a candidate is evaluated, not what it scores. If
that were even slightly untrue, L-BFGS-B would branch differently somewhere in
its thousand-odd evaluations and the two runs would land on different
antecedents -- so comparing the final parameters end to end is a far stronger
check than comparing a single fitness value.
"""

import uuid

import numpy as np
import pandas as pd
import pytest

from tribblefis import refine as R
from tribblefis.gauss_data import (
    FeatureModel,
    GaussianMembership,
    GaussianMixtureModel,
    LabelModel,
)
from tribblefis.kernel import NotCompilable


def _problem(n_samples=400, n_features=5, n_labels=3, n_mf=2, seed=0):
    rng = np.random.default_rng(seed)
    cols = [f"f{i}" for i in range(n_features)]
    centers = rng.normal(0, 3, size=(n_labels, n_features))
    y = rng.integers(0, n_labels, size=n_samples)
    X = pd.DataFrame(centers[y] + rng.normal(0, 1, size=(n_samples, n_features)),
                     columns=cols)

    feature_models = {}
    for name in cols:
        feature_models[name] = FeatureModel(label_models={
            label: LabelModel(memberships=[
                GaussianMembership(mu=float(rng.normal(0, 3)),
                                   sigma=float(rng.uniform(0.5, 2.0)),
                                   id=uuid.UUID(bytes=rng.bytes(16)))
                for _ in range(n_mf)
            ])
            for label in range(n_labels)
        })
    return X, y, GaussianMixtureModel(feature_models=feature_models)


def _run(X, y):
    return R.refine_classifier_antecedents(
        _MODEL, X, y, method="coordinate", n_sweeps=2, seed=42, verbose=False
    )


_X, _Y, _MODEL = _problem()


def test_compiled_and_reference_fitness_agree_exactly():
    x0 = R.extract_gaussian_params(_MODEL)
    bounds = R.build_param_bounds(_MODEL, _X)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    compiled_fitness = R._make_classifier_fitness(_MODEL, _X, _Y, 0.05, x0, lo, hi)

    rng = np.random.default_rng(3)
    for _ in range(20):
        vec = np.clip(x0 + rng.normal(0, 0.5, size=x0.size), lo, hi)
        # Rebuild the reference value the long way, through the model tree.
        candidate = R.apply_gaussian_params(_MODEL, vec)
        proba, labels = R._classifier_proba(_X, candidate)
        col = {lab: i for i, lab in enumerate(labels)}
        y_idx = np.array([col[v] for v in _Y])
        width = np.where((hi - lo) > 0, hi - lo, 1.0)
        expected = R._cross_entropy(proba, y_idx) + 0.05 * float(
            np.mean(((vec - x0) / width) ** 2)
        )
        assert compiled_fitness(vec) == expected


def test_refinement_result_is_unchanged_by_the_compiled_path(monkeypatch):
    fast_model, fast_info = _run(_X, _Y)

    # Force the fallback by making compilation fail, then rerun.
    def refuse(*_args, **_kwargs):
        raise NotCompilable("forced for test")

    monkeypatch.setattr(R, "compile_model", refuse)
    slow_model, slow_info = _run(_X, _Y)

    assert np.array_equal(
        R.extract_gaussian_params(fast_model), R.extract_gaussian_params(slow_model)
    )
    assert fast_info["train_obj"] == slow_info["train_obj"]
    assert fast_info["n_eval"] == slow_info["n_eval"]
    assert fast_info["refined"] == slow_info["refined"]


@pytest.mark.parametrize("n_mf", [1, 3])
def test_refinement_parity_across_membership_counts(monkeypatch, n_mf):
    X, y, model = _problem(n_samples=300, n_features=4, n_labels=2, n_mf=n_mf, seed=n_mf)
    kwargs = dict(method="coordinate", n_sweeps=1, seed=7, verbose=False)
    fast, _ = R.refine_classifier_antecedents(model, X, y, **kwargs)

    monkeypatch.setattr(
        R, "compile_model", lambda *a, **k: (_ for _ in ()).throw(NotCompilable("x"))
    )
    slow, _ = R.refine_classifier_antecedents(model, X, y, **kwargs)
    assert np.array_equal(
        R.extract_gaussian_params(fast), R.extract_gaussian_params(slow)
    )


def test_non_gaussian_model_still_refines_via_the_fallback():
    """A trapezoid model cannot be compiled; refinement must still run (and
    report that it had no Gaussian parameters to move)."""
    from tribblefis.gauss_data import TrapezoidMembership

    X, y, _ = _problem(n_samples=120, n_features=2, n_labels=2, n_mf=1, seed=99)
    model = GaussianMixtureModel(feature_models={
        name: FeatureModel(label_models={
            label: LabelModel(memberships=[TrapezoidMembership.create(-3, -1, 1, 3)])
            for label in range(2)
        })
        for name in X.columns
    })
    _out, info = R.refine_classifier_antecedents(model, X, y, verbose=False)
    assert info["refined"] is False
    assert info["reason"] == "no_gaussian_memberships"
