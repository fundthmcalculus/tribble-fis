"""Tests for the GT2 alpha-plane kernel (`gt2_kernel.py`).

Mirrors `test_it2_benchmark.py` (hand-crafted model) and
`test_it2_karnik_mendel.py` (independent validation of the type-reduction
math) for the GT2 case, per the validation-benchmark design in
`docs/gt2-evaluation.md`.
"""

import uuid

import numpy as np
import pandas as pd
import pytest

from tribblefis.gauss_data import (
    GT2GaussianMembership,
    GT2FeatureModel,
    GT2LabelModel,
    GT2GaussianMixtureModel,
    GaussianMembership,
    resolve_norm_pair,
)
from tribblefis.gt2_kernel import (
    default_alpha_levels,
    extract_alpha_plane_model,
    gt2_firing_strengths,
    gt2_rule_firing,
    gt2_karnik_mendel_tsk,
)
from tribblefis.it2_kernel import it2_firing_strengths, karnik_mendel_tsk


@pytest.fixture
def hand_crafted_gt2_classifier_model():
    """Same "low"/"high" scalar-boundary problem as
    `test_it2_benchmark.hand_crafted_it2_model`, extended with a `principal_mf`
    per membership (the original, un-widened sigma -- the same choice
    `gt2_classifier._convert_to_gt2` makes at fit time)."""
    def mk(mu, base_sigma, width=0.5):
        upper = GaussianMembership(mu=mu, sigma=base_sigma * (1 + width), id=uuid.uuid4())
        lower = GaussianMembership(mu=mu, sigma=base_sigma * max(0.1, 1 - width), id=uuid.uuid4())
        principal = GaussianMembership(mu=mu, sigma=base_sigma, id=uuid.uuid4())
        return GT2GaussianMembership(upper_mf=upper, lower_mf=lower, principal_mf=principal)

    it2_low = mk(-1.0, 1.0)
    it2_high = mk(1.0, 1.0)

    feature_models = {
        "x": GT2FeatureModel(
            label_models={
                0: GT2LabelModel(memberships=[it2_low]),
                1: GT2LabelModel(memberships=[it2_high]),
            }
        )
    }
    return GT2GaussianMixtureModel(feature_models=feature_models)


def test_gt2_firing_strengths_shape_and_bounds(hand_crafted_gt2_classifier_model):
    model = hand_crafted_gt2_classifier_model
    norms = resolve_norm_pair("probability")
    X = pd.DataFrame({"x": [-2, -1, 0, 1, 2]})

    firing_crisp, labels = gt2_firing_strengths(X, model, norms, n_alpha_planes=5)

    assert firing_crisp.shape == (5, 2)
    assert labels == [0, 1]
    assert np.all(firing_crisp >= 0)
    # Class 0 ("low") should score higher for negative x, class 1 for positive.
    assert firing_crisp[0, 0] > firing_crisp[0, 1]
    assert firing_crisp[4, 1] > firing_crisp[4, 0]


def test_default_alpha_levels_excludes_zero_weight_boundary():
    alphas = default_alpha_levels(5)
    assert len(alphas) == 5
    assert np.all(alphas > 0)
    assert alphas[-1] == pytest.approx(1.0)


def test_extract_alpha_plane_model_matches_membership_level_alpha_cut(hand_crafted_gt2_classifier_model):
    model = hand_crafted_gt2_classifier_model
    it2_model = extract_alpha_plane_model(model, 0.3)
    gt2_mf = model.feature_models["x"].label_models[0].memberships[0]
    expected = gt2_mf.alpha_cut(0.3)
    got = it2_model.feature_models["x"].label_models[0].memberships[0]
    assert got.upper_mf.sigma == pytest.approx(expected.upper_mf.sigma)
    assert got.lower_mf.sigma == pytest.approx(expected.lower_mf.sigma)


# ---------------------------------------------------------------------------
# Regression-shaped: gt2_rule_firing + gt2_karnik_mendel_tsk.
# ---------------------------------------------------------------------------

@pytest.fixture
def hand_crafted_gt2_regression_model():
    """Three rules on a single feature, each with its own GT2 footprint and a
    distinct principal sigma -- enough asymmetry that the alpha-weighted
    combination actually depends on more than one plane (a single-rule model
    can't exercise this: with one rule, Karnik-Mendel's output is always just
    that rule's own consequent value, independent of firing strength or
    alpha -- see `it2_kernel`'s own single-rule degenerate case)."""
    def mk(mu, sigma_lower, sigma_principal, sigma_upper):
        return GT2GaussianMembership(
            upper_mf=GaussianMembership(mu=mu, sigma=sigma_upper, id=uuid.uuid4()),
            lower_mf=GaussianMembership(mu=mu, sigma=sigma_lower, id=uuid.uuid4()),
            principal_mf=GaussianMembership(mu=mu, sigma=sigma_principal, id=uuid.uuid4()),
        )

    feature_models = {
        "x": GT2FeatureModel(
            label_models={
                0: GT2LabelModel(memberships=[mk(-2.0, 0.5, 0.8, 1.5)]),
                1: GT2LabelModel(memberships=[mk(0.0, 0.4, 0.6, 1.2)]),
                2: GT2LabelModel(memberships=[mk(2.0, 0.6, 1.0, 1.8)]),
            }
        )
    }
    return GT2GaussianMixtureModel(feature_models=feature_models)


def _combined_interval(model, X, rule_values, norms, n_alpha_planes):
    firing_uppers, firing_lowers, alphas, _ = gt2_rule_firing(
        model, X, ["x"], norms, n_alpha_planes=n_alpha_planes
    )
    return gt2_karnik_mendel_tsk(rule_values, firing_uppers, firing_lowers, alphas)


def test_gt2_combined_interval_is_contained_in_the_alpha_zero_boundary(hand_crafted_gt2_regression_model):
    """Every alpha-plane's firing bounds narrow *inside* the alpha=0 (widest,
    today's-IT2-equivalent) footprint, and Karnik-Mendel is monotonic in its
    firing-bound arguments, so each plane's own KM interval -- and therefore
    the alpha-weighted combination of them -- must lie within the alpha=0
    plane's own KM interval."""
    model = hand_crafted_gt2_regression_model
    norms = resolve_norm_pair("probability")
    X = pd.DataFrame({"x": np.linspace(-4, 4, 9)})
    rule_values = np.tile(np.array([-1.0, 0.0, 1.0]), (len(X), 1))

    y_l, y_r = _combined_interval(model, X, rule_values, norms, n_alpha_planes=5)

    it2_model_alpha0 = extract_alpha_plane_model(model, 0.0)
    from tribblefis.it2_kernel import _extract_upper_model, _extract_lower_model
    from tribblefis.gauss_math import tsk_firing_strengths

    upper_model = _extract_upper_model(it2_model_alpha0)
    lower_model = _extract_lower_model(it2_model_alpha0)
    fu0, _ = tsk_firing_strengths(X[["x"]], upper_model, norms=norms)
    fl0, _ = tsk_firing_strengths(X[["x"]], lower_model, norms=norms)
    y_l0, y_r0 = karnik_mendel_tsk(rule_values, fl0, fu0)

    assert np.all(y_l >= y_l0 - 1e-9)
    assert np.all(y_r <= y_r0 + 1e-9)
    assert np.all(y_l <= y_r + 1e-9)


def test_gt2_combination_converges_as_alpha_planes_increase(hand_crafted_gt2_regression_model):
    """More alpha-planes should approximate the same underlying (continuous
    alpha) integral more closely -- a bug in the weighting or plane
    construction would typically break this monotonic convergence rather than
    merely shift the answer slightly."""
    model = hand_crafted_gt2_regression_model
    norms = resolve_norm_pair("probability")
    X = pd.DataFrame({"x": np.linspace(-4, 4, 9)})
    rule_values = np.tile(np.array([-1.0, 0.0, 1.0]), (len(X), 1))

    reference_l, reference_r = _combined_interval(model, X, rule_values, norms, n_alpha_planes=200)
    err_coarse_l, err_coarse_r = _combined_interval(model, X, rule_values, norms, n_alpha_planes=3)
    err_fine_l, err_fine_r = _combined_interval(model, X, rule_values, norms, n_alpha_planes=20)

    coarse_err = np.mean(np.abs(err_coarse_l - reference_l) + np.abs(err_coarse_r - reference_r))
    fine_err = np.mean(np.abs(err_fine_l - reference_l) + np.abs(err_fine_r - reference_r))
    assert fine_err < coarse_err


def test_gt2_karnik_mendel_tsk_matches_it2_kernel_per_plane(hand_crafted_gt2_regression_model):
    """`gt2_karnik_mendel_tsk` must reduce to a plain alpha-weighted average of
    exactly what `it2_kernel.karnik_mendel_tsk` returns per plane -- no extra
    transformation snuck in."""
    model = hand_crafted_gt2_regression_model
    norms = resolve_norm_pair("probability")
    X = pd.DataFrame({"x": np.linspace(-4, 4, 5)})
    rule_values = np.tile(np.array([-1.0, 0.0, 1.0]), (len(X), 1))

    firing_uppers, firing_lowers, alphas, _ = gt2_rule_firing(model, X, ["x"], norms, n_alpha_planes=4)
    got_l, got_r = gt2_karnik_mendel_tsk(rule_values, firing_uppers, firing_lowers, alphas)

    manual_l = np.zeros(len(X))
    manual_r = np.zeros(len(X))
    for alpha, fu, fl in zip(alphas, firing_uppers, firing_lowers):
        y_l, y_r = karnik_mendel_tsk(rule_values, fl, fu)
        manual_l += alpha * y_l
        manual_r += alpha * y_r
    manual_l /= np.sum(alphas)
    manual_r /= np.sum(alphas)

    np.testing.assert_allclose(got_l, manual_l)
    np.testing.assert_allclose(got_r, manual_r)
