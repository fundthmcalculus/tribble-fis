"""IT2 refinement coverage for member_function="trap"/"triangular" (#144).

`it2_refine.py` previously refined only Gaussian antecedents by design
(non-Gaussian slots were silently skipped). These mirror `test_it2_refine.py`'s
Gaussian invariant checks (never-worse, actually-changes-parameters,
firing_lower<=firing_upper preserved) for the two new slot types.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from tribblefis.it2_classifier import IT2TribbleClassifier
from tribblefis.it2_regressor import IT2TribbleRegressor
from tribblefis.it2_kernel import it2_firing_strengths
from tribblefis.it2_refine import (
    _cross_entropy_loss,
    _iter_it2_slots,
    refine_it2_antecedents,
    refine_it2_regressor_antecedents,
)
from tribblefis.gauss_data import IT2TrapezoidMembership, IT2TriangularMembership

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
    for *_, it2_mf in _iter_it2_slots(model):
        params.extend([it2_mf.upper_mf.a, it2_mf.upper_mf.b, it2_mf.lower_mf.a, it2_mf.lower_mf.b])
    return np.array(params)


@pytest.mark.parametrize("member_function", MEMBER_FUNCTIONS)
def test_refine_it2_slots_are_the_expected_type(member_function, synthetic_classification_data):
    X, y = synthetic_classification_data
    clf = IT2TribbleClassifier(top_n=3, member_function=member_function, uncertainty_width=0.5, random_state=0)
    clf.fit(X, y)

    expected_type = IT2TrapezoidMembership if member_function == "trap" else IT2TriangularMembership
    slots = list(_iter_it2_slots(clf.model_))
    assert slots, "expected at least one slot to refine"
    assert all(isinstance(it2_mf, expected_type) for *_, it2_mf in slots)


@pytest.mark.parametrize("member_function", MEMBER_FUNCTIONS)
def test_refine_it2_antecedents_never_increases_training_loss(member_function, synthetic_classification_data):
    X, y = synthetic_classification_data
    clf = IT2TribbleClassifier(top_n=3, member_function=member_function, uncertainty_width=0.5, km_iterations=10, random_state=0)
    clf.fit(X, y)

    norms = clf.norms_
    y_idx = np.searchsorted(clf.classes_, y)
    init_loss = _cross_entropy_loss(clf.model_, X, y_idx, norms, km_iterations=10)

    refined = refine_it2_antecedents(
        X, y_idx, clf.model_, norms, n_sweeps=3, sub_maxfun=15, km_iterations=10, verbose=False,
    )
    refined_loss = _cross_entropy_loss(refined, X, y_idx, norms, km_iterations=10)

    assert refined_loss <= init_loss + 1e-9


@pytest.mark.parametrize("member_function", MEMBER_FUNCTIONS)
def test_refine_it2_antecedents_actually_changes_parameters(member_function, synthetic_classification_data):
    X, y = synthetic_classification_data
    clf = IT2TribbleClassifier(top_n=3, member_function=member_function, uncertainty_width=0.5, km_iterations=10, random_state=0)
    clf.fit(X, y)
    norms = clf.norms_
    y_idx = np.searchsorted(clf.classes_, y)

    refined = refine_it2_antecedents(
        X, y_idx, clf.model_, norms, n_sweeps=3, sub_maxfun=15, km_iterations=10, verbose=False,
    )

    before = _flat_params(clf.model_)
    after = _flat_params(refined)
    assert not np.allclose(before, after), "refinement left every antecedent parameter unchanged"


@pytest.mark.parametrize("member_function", MEMBER_FUNCTIONS)
def test_refine_it2_antecedents_preserves_lower_le_upper_invariant(member_function, synthetic_classification_data):
    X, y = synthetic_classification_data
    clf = IT2TribbleClassifier(top_n=3, member_function=member_function, uncertainty_width=0.5, km_iterations=10, random_state=0)
    clf.fit(X, y)
    norms = clf.norms_
    y_idx = np.searchsorted(clf.classes_, y)

    refined = refine_it2_antecedents(
        X, y_idx, clf.model_, norms, n_sweeps=3, sub_maxfun=15, km_iterations=10, verbose=False,
    )

    firing_upper, firing_lower, _, _ = it2_firing_strengths(X, refined, norms, km_iterations=None)
    assert np.all(firing_lower <= firing_upper + 1e-9)


@pytest.mark.parametrize("member_function", MEMBER_FUNCTIONS)
def test_refine_it2_regressor_antecedents_never_increases_cv_loss(member_function):
    X, y = make_regression(n_samples=200, n_features=4, n_informative=3, noise=5.0, random_state=3)
    feature_names = [f"x{i}" for i in range(X.shape[1])]
    X = pd.DataFrame(X, columns=feature_names)

    reg = IT2TribbleRegressor(top_n=3, n_gaussians=2, member_function=member_function, uncertainty_width=0.5, random_state=0)
    reg.fit(X, y)

    norms = reg.norms_
    base = reg._base_regressor
    _, _, _, init_info = refine_it2_regressor_antecedents(
        X, y, reg.model_, norms, base.top_features_, order=base.tsk_order, l2_reg=base.l2_reg,
        basis=base.consequent_basis, cross_pairs=base.cross_pairs_,
        method="none", n_sweeps=0, verbose=False,
    )
    refined_model, _, _, info = refine_it2_regressor_antecedents(
        X, y, reg.model_, norms, base.top_features_, order=base.tsk_order, l2_reg=base.l2_reg,
        basis=base.consequent_basis, cross_pairs=base.cross_pairs_,
        n_sweeps=2, sub_maxfun=15, verbose=False,
    )

    firing_upper, firing_lower, _, _ = it2_firing_strengths(X, refined_model, norms, km_iterations=None)
    assert np.all(firing_lower <= firing_upper + 1e-9)
    assert info["val_mse"] <= info["init_val_mse"] + 1e-9
