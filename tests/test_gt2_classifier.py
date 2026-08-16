"""Iris classification test for General Type-2 Fuzzy Classifier."""

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from tribblefis.gt2_classifier import GT2TribbleClassifier
from tribblefis.it2_classifier import IT2TribbleClassifier


@pytest.fixture
def iris_data():
    """Load and split iris dataset."""
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target
    )
    return X_train, X_test, y_train, y_test


def test_gt2_iris_fit_predict(iris_data):
    """Test GT2 classifier fit and predict on iris dataset."""
    X_train, X_test, y_train, y_test = iris_data

    clf = GT2TribbleClassifier(
        top_n=3,
        uncertainty_width=0.5,
        n_alpha_planes=5,
        random_state=42,
    )

    clf.fit(X_train, y_train)

    assert hasattr(clf, "classes_")
    assert len(clf.classes_) == 3
    np.testing.assert_array_equal(clf.classes_, np.array([0, 1, 2]))

    assert clf.model_ is not None
    assert clf.model_.n_classes == 3

    y_pred = clf.predict(X_test)
    assert y_pred.shape == (len(X_test),)
    assert all(c in clf.classes_ for c in y_pred)

    accuracy = np.mean(y_pred == y_test)
    assert accuracy > 0.75, f"Accuracy {accuracy} is too low"


def test_gt2_iris_uncertainty_width_effect(iris_data):
    """Wider footprint of uncertainty should widen the alpha=0 boundary
    reported by `predict_intervals`, exactly as for IT2."""
    X_train, X_test, y_train, y_test = iris_data

    clf_narrow = GT2TribbleClassifier(
        top_n=3, uncertainty_width=0.2, random_state=42,
    )
    clf_narrow.fit(X_train, y_train)

    clf_wide = GT2TribbleClassifier(
        top_n=3, uncertainty_width=1.0, random_state=42,
    )
    clf_wide.fit(X_train, y_train)

    upper_narrow, lower_narrow = clf_narrow.predict_intervals(X_test)
    upper_wide, lower_wide = clf_wide.predict_intervals(X_test)

    width_narrow = (upper_narrow - lower_narrow).mean()
    width_wide = (upper_wide - lower_wide).mean()

    assert width_wide > width_narrow, "Wider uncertainty_width should produce larger intervals"


def test_gt2_iris_intervals_validity(iris_data):
    """Test that firing strength intervals are valid (lower <= upper)."""
    X_train, X_test, y_train, y_test = iris_data

    clf = GT2TribbleClassifier(top_n=3, uncertainty_width=0.5, random_state=42)
    clf.fit(X_train, y_train)

    upper, lower = clf.predict_intervals(X_test)

    assert np.all(lower <= upper), "Lower bounds should be <= upper bounds"
    assert np.all(lower >= 0), "Lower bounds should be non-negative"
    assert np.all(upper >= 0), "Upper bounds should be non-negative"


def test_gt2_km_iterations_has_no_effect_on_classifier_prediction(iris_data):
    """`km_iterations` only ever reaches each alpha-plane's own per-rule
    reduction, which is provably the midpoint regardless of its value (see
    `it2_kernel.karnik_mendel_type_reduction`'s docstring) -- so, unlike the
    regressor, the classifier's predictions must be identical across values,
    not merely "may differ"."""
    X_train, X_test, y_train, _ = iris_data

    clf_km = GT2TribbleClassifier(top_n=3, uncertainty_width=0.5, km_iterations=10, random_state=42)
    clf_km.fit(X_train, y_train)

    clf_avg = GT2TribbleClassifier(top_n=3, uncertainty_width=0.5, km_iterations=None, random_state=42)
    clf_avg.fit(X_train, y_train)

    np.testing.assert_array_equal(clf_km.predict(X_test), clf_avg.predict(X_test))


def test_gt2_alpha_planes_converge_as_count_increases(iris_data):
    """More alpha-planes should change predictions less between successive
    counts -- a coarse sanity check that the combination is well-behaved
    (mirrors the convergence property `test_gt2_kernel.py` checks directly on
    the kernel, one level up through the full estimator)."""
    X_train, X_test, y_train, _ = iris_data

    def fit_predict_proba_like(n_alpha_planes):
        clf = GT2TribbleClassifier(
            top_n=3, uncertainty_width=0.5, n_alpha_planes=n_alpha_planes, random_state=42,
        )
        clf.fit(X_train, y_train)
        from tribblefis.gt2_kernel import gt2_firing_strengths
        import pandas as pd

        Xdf = pd.DataFrame(X_test, columns=clf.feature_names_in_)
        firing_crisp, _ = gt2_firing_strengths(
            Xdf, clf.model_, clf.norms_, n_alpha_planes=n_alpha_planes
        )
        return firing_crisp

    coarse = fit_predict_proba_like(3)
    fine = fit_predict_proba_like(20)
    finest = fit_predict_proba_like(60)

    coarse_vs_finest = np.mean(np.abs(coarse - finest))
    fine_vs_finest = np.mean(np.abs(fine - finest))
    assert fine_vs_finest < coarse_vs_finest


def test_gt2_classifier_accuracy_is_comparable_to_it2(iris_data):
    """Not a strict superiority claim (this is a single dataset/split), just
    a guard against a gross regression: GT2's alpha-weighted combination
    should not tank accuracy relative to plain IT2 on the same base fit."""
    X_train, X_test, y_train, y_test = iris_data

    gt2 = GT2TribbleClassifier(top_n=3, uncertainty_width=0.5, random_state=42)
    gt2.fit(X_train, y_train)
    it2 = IT2TribbleClassifier(top_n=3, uncertainty_width=0.5, random_state=42)
    it2.fit(X_train, y_train)

    gt2_acc = np.mean(gt2.predict(X_test) == y_test)
    it2_acc = np.mean(it2.predict(X_test) == y_test)
    assert gt2_acc >= it2_acc - 0.1
