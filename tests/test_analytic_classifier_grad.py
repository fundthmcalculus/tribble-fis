"""The analytic classifier gradient must be the real derivative.

Checked against a central finite difference of the *same* objective, which is
the only comparison that can catch a chain-rule slip. Under ``min/max`` the
quantity is a subgradient, so it matches wherever the active branch is unique --
which is everywhere except a measure-zero set, and random test points are not in
it.

Whether using this gradient is a good idea is a separate question, answered by
measurement in ``docs/analytic-gradient-evaluation.md``. These tests only
establish that it is correct.
"""

import uuid

import numpy as np
import pandas as pd
import pytest

from tribblefis import kernel as K, refine as R
from tribblefis.gauss_data import (
    FeatureModel,
    GaussianMembership,
    GaussianMixtureModel,
    LabelModel,
    NormPair,
)

requires_kernel = pytest.mark.skipif(
    not K.HAVE_CYTHON_KERNEL, reason="analytic gradients need the compiled kernel"
)

SMOOTH_AND_DEFAULT = ["probability", "min/max"]


def _problem(n=500, n_features=5, n_labels=3, n_mf=2, seed=0, norm="min/max"):
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
    arrays = {c: X[c].to_numpy() for c in X.columns}
    compiled = K.compile_model(model, list(arrays))
    inc = K.IncrementalFIS(compiled, compiled.feature_matrix(arrays), NormPair(norm, norm))
    ce = R._CrossEntropy(y, n_labels, (n, n_labels))
    return inc, ce, y


@requires_kernel
@pytest.mark.parametrize("norm", SMOOTH_AND_DEFAULT)
def test_gradient_matches_central_finite_difference(norm):
    inc, ce, _y = _problem(seed=1, norm=norm)
    rng = np.random.default_rng(2)

    def value(slot, v):
        return ce(inc.evaluate_slot(slot, v[0], v[1]))

    worst = 0.0
    for slot in range(inc.compiled.n_slots):
        v = np.array([float(rng.normal(0, 1.5)), float(rng.uniform(0.5, 1.8))])
        col = inc.target_label_index(slot)
        fs, d_mu, d_sigma = inc.evaluate_slot_with_grad(slot, v[0], v[1])
        val, grad = ce.with_column_grad(fs, col, (d_mu, d_sigma))

        # The value returned alongside the gradient must be the same value the
        # plain path returns -- otherwise L-BFGS-B is optimizing one function
        # while following another's slope.
        assert val == value(slot, v)

        h = 1e-6
        fd = np.array([
            (value(slot, v + [h, 0]) - value(slot, v - [h, 0])) / (2 * h),
            (value(slot, v + [0, h]) - value(slot, v - [0, h])) / (2 * h),
        ])
        worst = max(worst, float(np.max(np.abs(grad - fd)) / max(np.max(np.abs(fd)), 1e-8)))
    assert worst < 1e-5, f"worst relative gradient error {worst:.3e}"


@requires_kernel
def test_gradient_is_zero_for_an_inactive_min_max_branch():
    """Documents the structural fact behind the whole evaluation: under min/max,
    a membership function that is not the arg-min/arg-max anywhere has a gradient
    of exactly zero, because moving it changes nothing until a branch switches."""
    inc, ce, _y = _problem(seed=3, norm="min/max")
    rng = np.random.default_rng(4)
    zero = 0
    for slot in range(inc.compiled.n_slots):
        # Push the membership far from the data, so it never wins the max.
        fs, d_mu, d_sigma = inc.evaluate_slot_with_grad(slot, 500.0, 0.05)
        col = inc.target_label_index(slot)
        _val, grad = ce.with_column_grad(fs, col, (d_mu, d_sigma))
        if not grad.any():
            zero += 1
    assert zero > 0, "expected at least one exactly-flat subgradient"


@requires_kernel
def test_reported_support_matches_reality():
    """`supports_gradient` gates the dispatch, so it must not claim a family the
    kernel has no partials for."""
    for norm in ("min/max", "probability"):
        inc, _ce, _y = _problem(n=50, seed=5, norm=norm)
        assert inc.supports_gradient()
    for norm in ("luk", "hamacher", "einstein"):
        inc, _ce, _y = _problem(n=50, seed=6, norm=norm)
        assert not inc.supports_gradient()
        with pytest.raises(RuntimeError, match="min/max or probability"):
            inc.evaluate_slot_with_grad(0, 0.0, 1.0)


@requires_kernel
def test_regularisation_gradient_matches_its_own_finite_difference():
    """The ridge term is added to both the value and the gradient by hand; a sign
    or scale slip there would be invisible in the fuzzy part of the test above."""
    rng = np.random.default_rng(7)
    n_params = 8
    x0 = rng.normal(0, 1, n_params)
    width = np.abs(rng.normal(2, 0.5, n_params))
    l2 = 0.05

    class Stub:
        pass

    obj = Stub()
    obj.l2_shrink, obj.x0, obj.width = l2, x0, width
    reg = R._CompiledClassifierObjective._reg.__get__(obj)
    reg_grad = R._CompiledClassifierObjective._reg_grad.__get__(obj)

    vec = x0 + rng.normal(0, 0.4, n_params)
    idx = (2, 3)
    got = reg_grad(vec, idx)
    h = 1e-7
    for j, i in enumerate(idx):
        up, dn = vec.copy(), vec.copy()
        up[i] += h
        dn[i] -= h
        expected = (reg(up) - reg(dn)) / (2 * h)
        assert abs(got[j] - expected) < 1e-6 * max(abs(expected), 1.0)


@requires_kernel
def test_refinement_runs_and_stays_finite_with_the_gradient_on():
    """End to end. It is allowed to land somewhere different from the
    finite-difference path -- that is the documented behaviour -- but it must
    produce a usable model and use markedly fewer evaluations."""
    rng = np.random.default_rng(8)
    n_labels, n_features = 3, 5
    centers = rng.normal(0, 3, size=(n_labels, n_features))
    y = rng.integers(0, n_labels, size=400)
    cols = [f"f{i}" for i in range(n_features)]
    X = pd.DataFrame(centers[y] + rng.normal(0, 1, size=(400, n_features)), columns=cols)
    model = GaussianMixtureModel(feature_models={
        name: FeatureModel(label_models={
            lab: LabelModel(memberships=[
                GaussianMembership(mu=float(rng.normal(0, 3)), sigma=1.0,
                                   id=uuid.UUID(bytes=rng.bytes(16)))
                for _ in range(2)
            ])
            for lab in range(n_labels)
        })
        for name in cols
    })

    kwargs = dict(method="coordinate", n_sweeps=2, seed=42, verbose=False)
    _fd_model, fd = R.refine_classifier_antecedents(
        model, X, y, analytic_gradient=False, **kwargs)
    an_model, an = R.refine_classifier_antecedents(
        model, X, y, analytic_gradient=True, **kwargs)

    assert np.all(np.isfinite(R.extract_gaussian_params(an_model)))
    assert an["n_eval"] < fd["n_eval"], (
        "the whole point is fewer evaluations; if it is not doing that, the "
        "gradient is not reaching L-BFGS-B"
    )


def test_analytic_gradient_defaults_to_the_smoothness_rule():
    """The default is ``None`` -- "use it where it is the real derivative".

    That is the whole safety argument: under a smooth pair the closed form *is*
    the gradient and is accuracy-neutral; under a piecewise-smooth one it is a
    subgradient and measured as an accuracy lottery. A plain ``True`` default
    would silently take the second deal too.
    """
    import inspect

    sig = inspect.signature(R.refine_classifier_antecedents)
    assert sig.parameters["analytic_gradient"].default is None

    assert R._smooth_objective(NormPair("probability", "probability"))
    for family in ("min/max", "luk", "hamacher", "einstein"):
        assert not R._smooth_objective(NormPair(family, family)), family
    # A mixed pair is only smooth if both halves are.
    assert not R._smooth_objective(NormPair("probability", "min/max"))


def test_auto_rule_engages_under_the_default_family_and_not_under_min_max():
    """End to end: the rule has to reach the solver, not just exist. Fewer
    evaluations is the observable signature of a gradient being supplied."""
    X, y, model = _problem_for_auto()
    prob = NormPair("probability", "probability")
    minmax = NormPair("min/max", "min/max")
    kw = dict(n_sweeps=1, seed=11, verbose=False)

    auto_prob = R.refine_classifier_antecedents(model, X, y, norms=prob, **kw)[1]
    off_prob = R.refine_classifier_antecedents(
        model, X, y, norms=prob, analytic_gradient=False, **kw)[1]
    auto_mm = R.refine_classifier_antecedents(model, X, y, norms=minmax, **kw)[1]
    off_mm = R.refine_classifier_antecedents(
        model, X, y, norms=minmax, analytic_gradient=False, **kw)[1]

    if K.HAVE_CYTHON_KERNEL:
        assert auto_prob["n_eval"] < off_prob["n_eval"], "auto should be on here"
    assert auto_mm["n_eval"] == off_mm["n_eval"], "auto must stay off under min/max"


def _problem_for_auto():
    rng = np.random.default_rng(21)
    n_labels, n_features = 3, 5
    centers = rng.normal(0, 3, size=(n_labels, n_features))
    y = rng.integers(0, n_labels, size=400)
    cols = [f"f{i}" for i in range(n_features)]
    X = pd.DataFrame(centers[y] + rng.normal(0, 1, size=(400, n_features)), columns=cols)
    model = GaussianMixtureModel(feature_models={
        name: FeatureModel(label_models={
            lab: LabelModel(memberships=[
                GaussianMembership(mu=float(rng.normal(0, 3)), sigma=1.0,
                                   id=uuid.UUID(bytes=rng.bytes(16)))
                for _ in range(2)
            ])
            for lab in range(n_labels)
        })
        for name in cols
    })
    return X, y, model
