"""GT2 refinement coverage for member_function="trap"/"triangular" (#144).

Direct GT2 analogue of `test_it2_refine_member_function.py`, one dimension
wider per side per slot (the extra "principal" component) -- see
`gt2_refine.py`'s module docstring.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from tribblefis.gt2_classifier import GT2TribbleClassifier
from tribblefis.gt2_regressor import GT2TribbleRegressor
from tribblefis.gt2_kernel import extract_alpha_plane_model
from tribblefis.it2_kernel import it2_firing_strengths
from tribblefis.gt2_refine import (
    _cross_entropy_loss,
    _iter_gt2_slots,
    refine_gt2_antecedents,
    refine_gt2_regressor_antecedents,
)
from tribblefis.gauss_data import GT2TrapezoidMembership, GT2TriangularMembership

MEMBER_FUNCTIONS = ["trap", "triangular"]


@pytest.fixture
def synthetic_classification_data():
    X, y = make_classification(
        n_samples=150, n_features=4, n_informative=3, n_redundant=0,
        n_classes=3, n_clusters_per_class=1, random_state=7,
    )
    feature_names = [f"x{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=feature_names), y


def _flat_params(model):
    params = []
    for *_, gt2_mf in _iter_gt2_slots(model):
        params.extend([gt2_mf.upper_mf.a, gt2_mf.upper_mf.b, gt2_mf.lower_mf.a, gt2_mf.principal_mf.a])
    return np.array(params)


@pytest.mark.parametrize("member_function", MEMBER_FUNCTIONS)
def test_refine_gt2_slots_are_the_expected_type(member_function, synthetic_classification_data):
    X, y = synthetic_classification_data
    clf = GT2TribbleClassifier(top_n=3, member_function=member_function, uncertainty_width=0.5, n_alpha_planes=3, random_state=0)
    clf.fit(X, y)

    expected_type = GT2TrapezoidMembership if member_function == "trap" else GT2TriangularMembership
    slots = list(_iter_gt2_slots(clf.model_))
    assert slots, "expected at least one slot to refine"
    assert all(isinstance(gt2_mf, expected_type) for *_, gt2_mf in slots)


@pytest.mark.slow
@pytest.mark.parametrize("member_function", MEMBER_FUNCTIONS)
def test_refine_gt2_antecedents_never_increases_training_loss(member_function, synthetic_classification_data):
    X, y = synthetic_classification_data
    clf = GT2TribbleClassifier(top_n=3, member_function=member_function, uncertainty_width=0.5, n_alpha_planes=3, random_state=0)
    clf.fit(X, y)

    norms = clf.norms_
    y_idx = np.searchsorted(clf.classes_, y)
    init_loss = _cross_entropy_loss(clf.model_, X, y_idx, norms, n_alpha_planes=3, km_iterations=None)

    refined = refine_gt2_antecedents(
        X, y_idx, clf.model_, norms, n_sweeps=3, sub_maxfun=15, n_alpha_planes=3, verbose=False,
    )
    refined_loss = _cross_entropy_loss(refined, X, y_idx, norms, n_alpha_planes=3, km_iterations=None)

    assert refined_loss <= init_loss + 1e-9


@pytest.mark.slow
@pytest.mark.parametrize("member_function", MEMBER_FUNCTIONS)
def test_refine_gt2_antecedents_actually_changes_parameters(member_function, synthetic_classification_data):
    X, y = synthetic_classification_data
    clf = GT2TribbleClassifier(top_n=3, member_function=member_function, uncertainty_width=0.5, n_alpha_planes=3, random_state=0)
    clf.fit(X, y)
    norms = clf.norms_
    y_idx = np.searchsorted(clf.classes_, y)

    refined = refine_gt2_antecedents(
        X, y_idx, clf.model_, norms, n_sweeps=3, sub_maxfun=15, n_alpha_planes=3, verbose=False,
    )

    before = _flat_params(clf.model_)
    after = _flat_params(refined)
    assert not np.allclose(before, after), "refinement left every antecedent parameter unchanged"


@pytest.mark.slow
@pytest.mark.parametrize("member_function", MEMBER_FUNCTIONS)
def test_refine_gt2_antecedents_preserves_ordering_invariant(member_function, synthetic_classification_data):
    """`a_upper <= a_principal <= a_lower` (and the right-side mirror) must
    survive refinement, or alpha_cut's narrowing property breaks -- the
    trapezoid/triangular analogue of `sigma_lower <= sigma_principal <=
    sigma_upper`."""
    X, y = synthetic_classification_data
    clf = GT2TribbleClassifier(top_n=3, member_function=member_function, uncertainty_width=0.5, n_alpha_planes=3, random_state=0)
    clf.fit(X, y)
    norms = clf.norms_
    y_idx = np.searchsorted(clf.classes_, y)

    refined = refine_gt2_antecedents(
        X, y_idx, clf.model_, norms, n_sweeps=3, sub_maxfun=15, n_alpha_planes=3, verbose=False,
    )

    for *_, gt2_mf in _iter_gt2_slots(refined):
        assert gt2_mf.upper_mf.a <= gt2_mf.principal_mf.a + 1e-9
        assert gt2_mf.principal_mf.a <= gt2_mf.lower_mf.a + 1e-9
        right_upper = gt2_mf.upper_mf.d if member_function == "trap" else gt2_mf.upper_mf.c
        right_principal = gt2_mf.principal_mf.d if member_function == "trap" else gt2_mf.principal_mf.c
        right_lower = gt2_mf.lower_mf.d if member_function == "trap" else gt2_mf.lower_mf.c
        assert right_lower <= right_principal + 1e-9
        assert right_principal <= right_upper + 1e-9

    it2_model_alpha0 = extract_alpha_plane_model(refined, 0.0)
    firing_upper, firing_lower, _, _ = it2_firing_strengths(X, it2_model_alpha0, norms, km_iterations=None)
    assert np.all(firing_lower <= firing_upper + 1e-9)


@pytest.mark.slow
@pytest.mark.parametrize("member_function", MEMBER_FUNCTIONS)
def test_refine_gt2_regressor_antecedents_never_increases_cv_loss(member_function):
    X, y = make_regression(n_samples=200, n_features=4, n_informative=3, noise=5.0, random_state=3)
    feature_names = [f"x{i}" for i in range(X.shape[1])]
    X = pd.DataFrame(X, columns=feature_names)

    reg = GT2TribbleRegressor(top_n=3, n_gaussians=2, member_function=member_function, uncertainty_width=0.5, n_alpha_planes=3, random_state=0)
    reg.fit(X, y)

    norms = reg.norms_
    base = reg._base_regressor
    refined_model, _, _, info = refine_gt2_regressor_antecedents(
        X, y, reg.model_, norms, base.top_features_, order=base.tsk_order, l2_reg=base.l2_reg,
        basis=base.consequent_basis, cross_pairs=base.cross_pairs_, n_alpha_planes=3,
        n_sweeps=2, sub_maxfun=15, verbose=False,
    )

    it2_model_alpha0 = extract_alpha_plane_model(refined_model, 0.0)
    firing_upper, firing_lower, _, _ = it2_firing_strengths(X, it2_model_alpha0, norms, km_iterations=None)
    assert np.all(firing_lower <= firing_upper + 1e-9)
    assert info["val_mse"] <= info["init_val_mse"] + 1e-9
