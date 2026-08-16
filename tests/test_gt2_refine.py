"""Tests for post-fit GT2 antecedent refinement (`gt2_refine.py`).

Direct GT2 analogue of `test_it2_refine.py` -- see that file's own docstring
for the IT2 history this mirrors (a previously-dead-code refiner, and the
independent-halves invariant bug). The GT2-specific invariant this module
adds is `sigma_lower <= sigma_principal <= sigma_upper`, one dimension wider
than IT2's own `sigma_lower <= sigma_upper`.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from tribblefis.gt2_classifier import GT2TribbleClassifier
from tribblefis.gt2_kernel import gt2_firing_strengths, extract_alpha_plane_model
from tribblefis.it2_kernel import it2_firing_strengths
from tribblefis.gt2_refine import (
    _cross_entropy_loss,
    _iter_gt2_slots,
    refine_gt2_antecedents,
)


@pytest.fixture
def synthetic_classification_data():
    X, y = make_classification(
        n_samples=150, n_features=4, n_informative=3, n_redundant=0,
        n_classes=3, n_clusters_per_class=1, random_state=7,
    )
    feature_names = [f"x{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=feature_names), y


def test_refine_gt2_antecedents_never_increases_training_loss(synthetic_classification_data):
    X, y = synthetic_classification_data
    clf = GT2TribbleClassifier(top_n=3, uncertainty_width=0.5, n_alpha_planes=3, random_state=0)
    clf.fit(X, y)

    norms = clf.norms_
    y_idx = np.searchsorted(clf.classes_, y)
    init_loss = _cross_entropy_loss(clf.model_, X, y_idx, norms, n_alpha_planes=3, km_iterations=None)

    refined = refine_gt2_antecedents(
        X, y_idx, clf.model_, norms, n_sweeps=3, sub_maxfun=15, n_alpha_planes=3, verbose=False,
    )
    refined_loss = _cross_entropy_loss(refined, X, y_idx, norms, n_alpha_planes=3, km_iterations=None)

    assert refined_loss <= init_loss + 1e-9


def test_refine_gt2_antecedents_actually_changes_parameters(synthetic_classification_data):
    """Guards against a silent no-op: the whole point of the coordinate
    descent is that it can move parameters at all."""
    X, y = synthetic_classification_data
    clf = GT2TribbleClassifier(top_n=3, uncertainty_width=0.5, n_alpha_planes=3, random_state=0)
    clf.fit(X, y)
    norms = clf.norms_
    y_idx = np.searchsorted(clf.classes_, y)

    refined = refine_gt2_antecedents(
        X, y_idx, clf.model_, norms, n_sweeps=3, sub_maxfun=15, n_alpha_planes=3, verbose=False,
    )

    def _flat_params(model):
        params = []
        for *_, gt2_mf in _iter_gt2_slots(model):
            params.extend([
                gt2_mf.upper_mf.mu, gt2_mf.upper_mf.sigma,
                gt2_mf.lower_mf.sigma, gt2_mf.principal_mf.sigma,
            ])
        return np.array(params)

    before = _flat_params(clf.model_)
    after = _flat_params(refined)
    assert not np.allclose(before, after), "refinement left every antecedent parameter unchanged"


def test_refine_gt2_antecedents_preserves_sigma_ordering_invariant(synthetic_classification_data):
    """The GT2 analogue of IT2's `sigma_lower <= sigma_upper` invariant test:
    `sigma_lower <= sigma_principal <= sigma_upper` must survive refinement,
    or `alpha_cut`'s narrowing property (and everything downstream that
    depends on it -- containment, monotonic convergence) breaks."""
    X, y = synthetic_classification_data
    clf = GT2TribbleClassifier(top_n=3, uncertainty_width=0.5, n_alpha_planes=3, random_state=0)
    clf.fit(X, y)
    norms = clf.norms_
    y_idx = np.searchsorted(clf.classes_, y)

    refined = refine_gt2_antecedents(
        X, y_idx, clf.model_, norms, n_sweeps=3, sub_maxfun=15, n_alpha_planes=3, verbose=False,
    )

    for *_, gt2_mf in _iter_gt2_slots(refined):
        assert gt2_mf.lower_mf.sigma <= gt2_mf.principal_mf.sigma + 1e-9
        assert gt2_mf.principal_mf.sigma <= gt2_mf.upper_mf.sigma + 1e-9

    it2_model_alpha0 = extract_alpha_plane_model(refined, 0.0)
    firing_upper, firing_lower, _, _ = it2_firing_strengths(X, it2_model_alpha0, norms, km_iterations=None)
    assert np.all(firing_lower <= firing_upper + 1e-9)


def test_refine_gt2_antecedents_method_none_is_identity(synthetic_classification_data):
    X, y = synthetic_classification_data
    clf = GT2TribbleClassifier(top_n=3, random_state=0)
    clf.fit(X, y)
    y_idx = np.searchsorted(clf.classes_, y)

    refined = refine_gt2_antecedents(X, y_idx, clf.model_, clf.norms_, method="none")
    assert refined is clf.model_


def test_refine_gt2_antecedents_rejects_unknown_method(synthetic_classification_data):
    X, y = synthetic_classification_data
    clf = GT2TribbleClassifier(top_n=3, random_state=0)
    clf.fit(X, y)
    y_idx = np.searchsorted(clf.classes_, y)

    with pytest.raises(ValueError):
        refine_gt2_antecedents(X, y_idx, clf.model_, clf.norms_, method="bogus")


def test_classifier_refine_gt2_option_fits_and_predicts(synthetic_classification_data):
    X, y = synthetic_classification_data
    clf = GT2TribbleClassifier(
        top_n=3, uncertainty_width=0.5, n_alpha_planes=3, random_state=0,
        refine_gt2=True, refine_gt2_n_sweeps=2,
    )
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape
    assert set(np.unique(y_pred)) <= set(clf.classes_)


def test_classifier_refine_gt2_does_not_worsen_training_accuracy_much(synthetic_classification_data):
    """Refinement directly optimizes cross-entropy, not accuracy, so an exact
    accuracy improvement isn't guaranteed on a single run -- but it should not
    make the model dramatically worse either."""
    X, y = synthetic_classification_data

    baseline = GT2TribbleClassifier(
        top_n=3, uncertainty_width=0.5, n_alpha_planes=3, random_state=0, refine_gt2=False,
    )
    baseline.fit(X, y)
    baseline_acc = np.mean(baseline.predict(X) == y)

    refined_clf = GT2TribbleClassifier(
        top_n=3, uncertainty_width=0.5, n_alpha_planes=3, random_state=0,
        refine_gt2=True, refine_gt2_n_sweeps=3,
    )
    refined_clf.fit(X, y)
    refined_acc = np.mean(refined_clf.predict(X) == y)

    assert refined_acc >= baseline_acc - 0.05
