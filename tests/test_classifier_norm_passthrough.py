"""Refinement must optimise the model that will actually be deployed.

The firing strengths -- and therefore the entire cross-entropy surface the search
descends -- are a function of the (t-norm, t-conorm) pair. Refining under one
pair and predicting under another tunes a model nobody runs. That was the
behaviour before this change: `refine_classifier_antecedents` hard-coded the
library default while `MixtureOfGaussiansFuzzyClassifier` predicted with
whatever `norm_conorm` the caller asked for.
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
    NormPair,
)
from tribblefis.gaussian_classifier import MixtureOfGaussiansFuzzyClassifier

FAMILIES = ["min/max", "probability", "luk", "hamacher", "einstein"]


def _problem(n=400, n_features=5, n_labels=3, n_mf=2, seed=0):
    rng = np.random.default_rng(seed)
    cols = [f"f{i}" for i in range(n_features)]
    centers = rng.normal(0, 3, size=(n_labels, n_features))
    y = rng.integers(0, n_labels, size=n)
    X = pd.DataFrame(centers[y] + rng.normal(0, 1, size=(n, n_features)), columns=cols)
    model = GaussianMixtureModel(feature_models={
        name: FeatureModel(label_models={
            lab: LabelModel(memberships=[
                GaussianMembership(mu=float(rng.normal(0, 2)),
                                   sigma=float(rng.uniform(0.4, 2.0)),
                                   id=uuid.UUID(bytes=rng.bytes(16)))
                for _ in range(n_mf)
            ])
            for lab in range(n_labels)
        })
        for name in cols
    })
    return X, y, model


@pytest.mark.parametrize("family", FAMILIES)
def test_norms_reach_the_objective(family):
    """A different operator pair must produce a different loss surface, and so
    a different fitness value at the same point."""
    X, y, model = _problem(seed=1)
    x0 = R.extract_gaussian_params(model)
    bounds = R.build_param_bounds(model, X)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    default = R._make_classifier_fitness(model, X, y, 0.0, x0, lo, hi)
    chosen = R._make_classifier_fitness(model, X, y, 0.0, x0, lo, hi,
                                        NormPair(family, family))
    if family == "min/max":
        assert chosen(x0) == default(x0)
    else:
        assert chosen(x0) != default(x0), (
            f"{family} produced the same objective as min/max -- the pair is "
            f"being ignored somewhere"
        )


@pytest.mark.parametrize("family", ["probability", "einstein"])
def test_refinement_under_a_pair_is_evaluated_under_that_pair(family):
    """End to end: refining under `family` must improve `family`'s own held-out
    loss, which is only meaningful if the accept/reject guard also used it."""
    X, y, model = _problem(n=600, seed=2)
    norms = NormPair(family, family)
    out, info = R.refine_classifier_antecedents(
        model, X, y, norms=norms, n_sweeps=2, seed=42, verbose=False)

    before = R._classifier_val_ce(X, y, model, norms)
    after = R._classifier_val_ce(X, y, out, norms)
    assert after <= before + 1e-9
    assert info["init_val_ce"] != R._classifier_val_ce(
        X, y, model, NormPair("min/max", "min/max")
    ) or family == "min/max", "the guard is still scoring under min/max"


def test_classifier_passes_its_norm_choice_into_refinement(monkeypatch):
    """The estimator holds the operator choice; it is the only thing that can
    hand it to the refiner."""
    seen = {}
    original = R.refine_classifier_antecedents

    def spy(model, X, y, **kwargs):
        seen["norms"] = kwargs.get("norms")
        return original(model, X, y, **kwargs)

    # `fit` imports the function from `refine` at call time, so patching the
    # source module is what the estimator will actually pick up.
    monkeypatch.setattr(R, "refine_classifier_antecedents", spy)

    X, y, _model = _problem(n=300, seed=3)
    clf = MixtureOfGaussiansFuzzyClassifier(
        norm_conorm="probability", refine=True, n_gaussians=2)
    clf.fit(X, y)

    assert seen["norms"] == NormPair("probability", "probability"), (
        f"refinement received {seen['norms']!r}; it must match the estimator's "
        f"norm_conorm or it tunes a model the estimator will not run"
    )


def test_default_still_matches_the_library_default():
    """Callers that pass nothing must be unaffected by the new parameter."""
    X, y, model = _problem(n=300, seed=4)
    a, ia = R.refine_classifier_antecedents(model, X, y, n_sweeps=1, seed=7,
                                            verbose=False)
    from tribblefis.gauss_data import resolve_norm_pair

    b, ib = R.refine_classifier_antecedents(model, X, y, n_sweeps=1, seed=7,
                                            norms=resolve_norm_pair(), verbose=False)
    assert np.array_equal(R.extract_gaussian_params(a), R.extract_gaussian_params(b))
    assert ia["train_obj"] == ib["train_obj"]


# ---------------------------------------------------------------------------
# Sub-problem solver selection.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("solver", sorted(R._SUB_SOLVERS))
def test_every_advertised_sub_solver_runs(solver):
    """Each entry in the registry spells its own budget option; a wrong spelling
    shows up as an ignored budget or a TypeError, not as a wrong answer."""
    X, y, model = _problem(n=300, n_features=4, n_labels=3, seed=5)
    out, info = R.refine_classifier_antecedents(
        model, X, y, n_sweeps=1, seed=7, sub_method=solver, verbose=False)
    params = R.extract_gaussian_params(out)
    assert np.all(np.isfinite(params))
    assert info["n_eval"] > 1


def test_unknown_sub_solver_is_rejected_by_name():
    X, y, model = _problem(n=200, seed=6)
    with pytest.raises(ValueError, match="sub_method"):
        R.refine_classifier_antecedents(model, X, y, n_sweeps=1,
                                        sub_method="Newton-CG", verbose=False)


def test_gradient_free_solver_ignores_the_analytic_gradient():
    """Powell takes no jac. Asking for both must not hand it a tuple-returning
    objective, which it would try to compare as a scalar."""
    X, y, model = _problem(n=300, seed=8)
    out, _info = R.refine_classifier_antecedents(
        model, X, y, n_sweeps=1, seed=7, sub_method="Powell",
        analytic_gradient=True, verbose=False)
    assert np.all(np.isfinite(R.extract_gaussian_params(out)))
