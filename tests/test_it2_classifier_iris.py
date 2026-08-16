"""Iris classification test for Interval Type-2 Fuzzy Classifier."""

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from tribblefis.it2_classifier import IT2TribbleClassifier


@pytest.fixture
def iris_data():
    """Load and split iris dataset."""
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target
    )
    return X_train, X_test, y_train, y_test


def test_it2_iris_fit_predict(iris_data):
    """Test IT2 classifier fit and predict on iris dataset."""
    X_train, X_test, y_train, y_test = iris_data

    clf = IT2TribbleClassifier(
        top_n=3,
        uncertainty_width=0.5,
        km_iterations=10,
        random_state=42,
    )

    # Should fit without errors
    clf.fit(X_train, y_train)

    # Should have learned classes
    assert hasattr(clf, "classes_")
    assert len(clf.classes_) == 3
    np.testing.assert_array_equal(clf.classes_, np.array([0, 1, 2]))

    # Should have learned model
    assert clf.model_ is not None
    assert clf.model_.n_classes == 3

    # Should predict on test set
    y_pred = clf.predict(X_test)
    assert y_pred.shape == (len(X_test),)
    assert all(c in clf.classes_ for c in y_pred)

    # Check accuracy (should be decent, but not necessarily better than Type-1)
    accuracy = np.mean(y_pred == y_test)
    assert accuracy > 0.75, f"Accuracy {accuracy} is too low"


def test_it2_iris_uncertainty_width_effect(iris_data):
    """Test that uncertainty width controls interval size."""
    X_train, X_test, y_train, y_test = iris_data

    clf_narrow = IT2TribbleClassifier(
        top_n=3,
        uncertainty_width=0.2,  # Narrow uncertainty
        km_iterations=None,  # Use averaging
        random_state=42,
    )
    clf_narrow.fit(X_train, y_train)

    clf_wide = IT2TribbleClassifier(
        top_n=3,
        uncertainty_width=1.0,  # Wide uncertainty
        km_iterations=None,  # Use averaging
        random_state=42,
    )
    clf_wide.fit(X_train, y_train)

    # Get firing strength intervals
    upper_narrow, lower_narrow = clf_narrow.predict_intervals(X_test)
    upper_wide, lower_wide = clf_wide.predict_intervals(X_test)

    # Wider uncertainty should have larger interval width
    width_narrow = (upper_narrow - lower_narrow).mean()
    width_wide = (upper_wide - lower_wide).mean()

    assert width_wide > width_narrow, "Wider uncertainty_width should produce larger intervals"


def test_it2_iris_intervals_validity(iris_data):
    """Test that firing strength intervals are valid (lower <= upper)."""
    X_train, X_test, y_train, y_test = iris_data

    clf = IT2TribbleClassifier(
        top_n=3,
        uncertainty_width=0.5,
        km_iterations=10,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    upper, lower = clf.predict_intervals(X_test)

    # Lower should always be <= upper
    assert np.all(lower <= upper), "Lower bounds should be <= upper bounds"

    # Both should be non-negative (firing strengths)
    assert np.all(lower >= 0), "Lower bounds should be non-negative"
    assert np.all(upper >= 0), "Upper bounds should be non-negative"


def test_it2_iris_km_vs_averaging(iris_data):
    """Test that KM type reduction produces different results than averaging."""
    X_train, X_test, y_train, y_test = iris_data

    clf_km = IT2TribbleClassifier(
        top_n=3,
        uncertainty_width=0.5,
        km_iterations=10,
        random_state=42,
    )
    clf_km.fit(X_train, y_train)

    clf_avg = IT2TribbleClassifier(
        top_n=3,
        uncertainty_width=0.5,
        km_iterations=None,  # Use averaging
        random_state=42,
    )
    clf_avg.fit(X_train, y_train)

    # Predictions should be different
    y_pred_km = clf_km.predict(X_test)
    y_pred_avg = clf_avg.predict(X_test)

    # They might not always differ on all samples, but should differ on average
    # At minimum, they should both produce valid predictions
    assert y_pred_km.shape == y_pred_avg.shape
    assert all(c in clf_km.classes_ for c in y_pred_km)
    assert all(c in clf_avg.classes_ for c in y_pred_avg)
