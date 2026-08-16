"""Tests for post-fit IT2 antecedent refinement (`it2_refine.py`).

Previously `it2_refine.refine_it2_antecedents` was unreachable dead code: its
gradient computation was a stub that always returned `0.001` regardless of
input (see the module's git history), so no parameter it touched could ever
move. These tests exercise the real coordinate-descent replacement and the
`T2TribbleClassifier.refine_it2` option that now wires it in.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from tribblefis.it2_classifier import T2TribbleClassifier
from tribblefis.it2_refine import (
    _cross_entropy_loss,
    _iter_it2_gaussian_slots,
    refine_it2_antecedents,
)
from tribblefis.gauss_data import resolve_norm_pair


@pytest.fixture
def synthetic_classification_data():
    X, y = make_classification(
        n_samples=150, n_features=4, n_informative=3, n_redundant=0,
        n_classes=3, n_clusters_per_class=1, random_state=7,
    )
    feature_names = [f"x{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=feature_names), y


def test_refine_it2_antecedents_never_increases_training_loss(synthetic_classification_data):
    X, y = synthetic_classification_data
    clf = T2TribbleClassifier(top_n=3, uncertainty_width=0.5, km_iterations=10, random_state=0)
    clf.fit(X, y)

    norms = clf.norms_
    y_idx = np.searchsorted(clf.classes_, y)
    init_loss = _cross_entropy_loss(clf.model_, X, y_idx, norms, km_iterations=10)

    refined = refine_it2_antecedents(
        X, y_idx, clf.model_, norms, n_sweeps=3, sub_maxfun=15, km_iterations=10, verbose=False,
    )
    refined_loss = _cross_entropy_loss(refined, X, y_idx, norms, km_iterations=10)

    assert refined_loss <= init_loss + 1e-9


def test_refine_it2_antecedents_actually_changes_parameters(synthetic_classification_data):
    """Guards against a silent no-op regression: the whole point of fixing the
    stubbed gradient is that refinement can move parameters at all."""
    X, y = synthetic_classification_data
    clf = T2TribbleClassifier(top_n=3, uncertainty_width=0.5, km_iterations=10, random_state=0)
    clf.fit(X, y)
    norms = clf.norms_
    y_idx = np.searchsorted(clf.classes_, y)

    refined = refine_it2_antecedents(
        X, y_idx, clf.model_, norms, n_sweeps=3, sub_maxfun=15, km_iterations=10, verbose=False,
    )

    before = np.array([mf.mu for *_, mf in _iter_it2_gaussian_slots(clf.model_)]
                       + [mf.sigma for *_, mf in _iter_it2_gaussian_slots(clf.model_)])
    after = np.array([mf.mu for *_, mf in _iter_it2_gaussian_slots(refined)]
                      + [mf.sigma for *_, mf in _iter_it2_gaussian_slots(refined)])
    assert not np.allclose(before, after), "refinement left every antecedent parameter unchanged"


def test_refine_it2_antecedents_method_none_is_identity(synthetic_classification_data):
    X, y = synthetic_classification_data
    clf = T2TribbleClassifier(top_n=3, random_state=0)
    clf.fit(X, y)
    y_idx = np.searchsorted(clf.classes_, y)

    refined = refine_it2_antecedents(X, y_idx, clf.model_, clf.norms_, method="none")
    assert refined is clf.model_


def test_refine_it2_antecedents_rejects_unknown_method(synthetic_classification_data):
    X, y = synthetic_classification_data
    clf = T2TribbleClassifier(top_n=3, random_state=0)
    clf.fit(X, y)
    y_idx = np.searchsorted(clf.classes_, y)

    with pytest.raises(ValueError):
        refine_it2_antecedents(X, y_idx, clf.model_, clf.norms_, method="bogus")


def test_classifier_refine_it2_option_fits_and_predicts(synthetic_classification_data):
    X, y = synthetic_classification_data
    clf = T2TribbleClassifier(
        top_n=3, uncertainty_width=0.5, km_iterations=10, random_state=0,
        refine_it2=True, refine_it2_n_sweeps=2,
    )
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape
    assert set(np.unique(y_pred)) <= set(clf.classes_)


def test_classifier_refine_it2_does_not_worsen_training_accuracy_much(synthetic_classification_data):
    """Refinement directly optimizes cross-entropy, not accuracy, so an exact
    accuracy improvement isn't guaranteed on a single run -- but it should not
    make the model dramatically worse either."""
    X, y = synthetic_classification_data

    baseline = T2TribbleClassifier(
        top_n=3, uncertainty_width=0.5, km_iterations=10, random_state=0, refine_it2=False,
    )
    baseline.fit(X, y)
    baseline_acc = np.mean(baseline.predict(X) == y)

    refined_clf = T2TribbleClassifier(
        top_n=3, uncertainty_width=0.5, km_iterations=10, random_state=0,
        refine_it2=True, refine_it2_n_sweeps=3,
    )
    refined_clf.fit(X, y)
    refined_acc = np.mean(refined_clf.predict(X) == y)

    assert refined_acc >= baseline_acc - 0.05
