"""Acceptance guards: each strategy must implement what it claims.

The choice *between* them is a measured question, settled in
`docs/refinement-guard-evaluation.md`. These tests only pin the mechanics --
that a strict guard is actually strict, that the permissive default really does
train on all the data, and that the statistics are the statistics.
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


# ---------------------------------------------------------------------------
# The decision rules in isolation.
# ---------------------------------------------------------------------------

def _evidence(n, init_right, refined_right, ce_init=1.0, ce_refined=1.0):
    """Synthetic per-row evidence with exactly the given correct counts."""
    ok_i = np.zeros(n, dtype=bool)
    ok_r = np.zeros(n, dtype=bool)
    ok_i[:init_right] = True
    ok_r[-refined_right:] = refined_right > 0
    return ok_i, ok_r, np.full(n, ce_init), np.full(n, ce_refined)


@pytest.mark.parametrize("strategy", R.GUARDS)
def test_none_accepts_and_others_can_refuse(strategy):
    """`none` must be unconditional; everything else must be able to say no."""
    ok_i, ok_r, ll_i, ll_r = _evidence(100, init_right=90, refined_right=50,
                                       ce_init=0.2, ce_refined=2.0)
    accept, info = R._apply_guard(strategy, ok_i, ok_r, ll_i, ll_r)
    assert info["guard"] == strategy
    if strategy == "none":
        assert accept, "guard='none' must keep the refinement unconditionally"
    else:
        assert not accept, f"{strategy} accepted a clearly worse refinement"


def test_every_guard_accepts_an_overwhelming_improvement():
    ok_i, ok_r, ll_i, ll_r = _evidence(200, init_right=60, refined_right=190,
                                       ce_init=2.0, ce_refined=0.1)
    for strategy in R.GUARDS:
        accept, _ = R._apply_guard(strategy, ok_i, ok_r, ll_i, ll_r)
        assert accept, f"{strategy} rejected an unambiguous improvement"


def test_strictness_is_ordered_on_a_marginal_case():
    """One extra correct row out of 40 is noise. `legacy` takes it; the two
    rules built to resist exactly that must not."""
    n = 40
    ok_i = np.zeros(n, dtype=bool)
    ok_i[:30] = True
    ok_r = ok_i.copy()
    ok_r[30] = True                      # one more correct: +2.5 points
    ll = np.full(n, 1.0)

    assert R._apply_guard("legacy", ok_i, ok_r, ll, ll)[0]
    assert not R._apply_guard("mcnemar", ok_i, ok_r, ll, ll)[0]
    assert not R._apply_guard("effect-size", ok_i, ok_r, ll, ll)[0]


def test_ce_guard_reads_cross_entropy_when_accuracy_ties():
    n = 50
    ok = np.ones(n, dtype=bool)
    better = np.full(n, 0.5)
    worse = np.full(n, 0.9)
    assert R._apply_guard("ce", ok, ok, worse, better)[0]
    assert not R._apply_guard("ce", ok, ok, better, worse)[0]


def test_mcnemar_p_value_matches_the_exact_binomial():
    """Hand-checked values: the discordant counts here are small enough that a
    normal approximation would be visibly wrong, which is why it is exact."""
    assert R._mcnemar_p(0, 0) == 1.0
    assert R._mcnemar_p(5, 5) == 1.0
    # 6 vs 0 discordant -> 2 * (1/2)^6
    assert R._mcnemar_p(6, 0) == pytest.approx(2 * 0.5 ** 6)
    assert R._mcnemar_p(0, 6) == pytest.approx(2 * 0.5 ** 6)
    # More evidence must never raise the p-value.
    assert R._mcnemar_p(10, 0) < R._mcnemar_p(6, 0) < R._mcnemar_p(3, 0)


def test_unknown_guard_is_rejected_by_name():
    X, y, model = _problem(n=200, seed=1)
    with pytest.raises(ValueError, match="guard"):
        R.refine_classifier_antecedents(model, X, y, guard="vibes", verbose=False)


# ---------------------------------------------------------------------------
# End to end.
# ---------------------------------------------------------------------------

def test_default_guard_is_none():
    """A measured default, so it gets a test that will notice if it drifts."""
    import inspect

    sig = inspect.signature(R.refine_classifier_antecedents)
    assert sig.parameters["guard"].default == "none"


@pytest.mark.slow
def test_none_trains_on_all_the_data():
    """Half the value of dropping the guard is reclaiming `val_fraction` of the
    training set. If the split were still being taken, shrinking val_fraction
    would change the result -- with `none` it must not."""
    X, y, model = _problem(n=500, seed=2)
    kw = dict(n_sweeps=1, seed=42, verbose=False)
    a, _ = R.refine_classifier_antecedents(model, X, y, guard="none",
                                           val_fraction=0.25, **kw)
    b, _ = R.refine_classifier_antecedents(model, X, y, guard="none",
                                           val_fraction=0.5, **kw)
    assert np.array_equal(R.extract_gaussian_params(a), R.extract_gaussian_params(b))

    # A guard that does referee must still be sensitive to it.
    c, _ = R.refine_classifier_antecedents(model, X, y, guard="legacy",
                                           val_fraction=0.25, **kw)
    d, _ = R.refine_classifier_antecedents(model, X, y, guard="legacy",
                                           val_fraction=0.5, **kw)
    assert not np.array_equal(
        R.extract_gaussian_params(c), R.extract_gaussian_params(d)
    ), "val_fraction should still matter when a guard is actually used"


@pytest.mark.parametrize("guard", R.GUARDS)
def test_every_guard_runs_end_to_end_and_reports_itself(guard):
    X, y, model = _problem(n=400, seed=3)
    out, info = R.refine_classifier_antecedents(
        model, X, y, guard=guard, n_sweeps=1, seed=42, verbose=False)
    assert np.all(np.isfinite(R.extract_gaussian_params(out)))
    assert info["guard"] == guard
    assert isinstance(info["refined"], bool)
    if guard == "none":
        assert info["refined"] is True


def test_rejecting_returns_the_starting_model_untouched():
    """When a guard says no, the caller must get the original object's
    parameters back -- not a nearly-identical refinement."""
    X, y, model = _problem(n=300, seed=4)
    ok_i, ok_r, ll_i, ll_r = _evidence(50, 45, 10, ce_init=0.1, ce_refined=3.0)
    assert not R._apply_guard("legacy", ok_i, ok_r, ll_i, ll_r)[0]

    out, info = R.refine_classifier_antecedents(
        model, X, y, guard="mcnemar", n_sweeps=1, seed=42, verbose=False)
    if not info["refined"]:
        assert np.array_equal(
            R.extract_gaussian_params(out), R.extract_gaussian_params(model)
        )


# ---------------------------------------------------------------------------
# The Ruspini refiner keeps its guard. That is a measured difference from the
# classifier, not an oversight, so it is pinned here -- otherwise the next
# person to notice the inconsistency will "fix" it.
# ---------------------------------------------------------------------------

def _ruspini_problem(n=300, seed=0):
    from tribblefis.gauss_math import create_gaussian_membership_dict
    from tribblefis.ruspini import ruspinize_model

    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 3, size=(3, 3))
    y = rng.integers(0, 3, size=n)
    cols = ["a", "b", "c"]
    X = pd.DataFrame(centers[y] + rng.normal(0, 1, size=(n, 3)), columns=cols)
    gm = create_gaussian_membership_dict(X, pd.Series(y), top_n_var_names=cols,
                                         n_gaussians=1)
    return X, y, ruspinize_model(gm, X, y)


def test_ruspini_guard_default_is_legacy_and_is_deliberate():
    """The classifier dropped its guard; this one measured as a wash
    (+0.0049 +/- 0.0058), so it keeps it. See docs/refinement-guard-evaluation.md."""
    import inspect

    assert inspect.signature(
        R.refine_ruspini_partition).parameters["guard"].default == "legacy"
    assert inspect.signature(
        R.refine_classifier_antecedents).parameters["guard"].default == "none"


@pytest.mark.parametrize("guard", R.GUARDS)
def test_ruspini_accepts_every_guard(guard):
    X, y, rm = _ruspini_problem(seed=1)
    out, info = R.refine_ruspini_partition(
        rm, X, y, guard=guard, n_sweeps=1, seed=0, verbose=False)
    assert info["guard"] == guard
    assert np.all(np.isfinite(out.extract_knots()))
    if guard == "none":
        assert info["refined"] is True


def test_ruspini_rejects_an_unknown_guard():
    X, y, rm = _ruspini_problem(seed=2)
    with pytest.raises(ValueError, match="guard"):
        R.refine_ruspini_partition(rm, X, y, guard="vibes", verbose=False)
