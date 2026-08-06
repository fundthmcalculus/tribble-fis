"""Tests for IT2 with different membership function types."""

import numpy as np
import pytest
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from tribblefis.it2_classifier import IntervalType2FuzzyClassifier
from tribblefis.it2_regressor import IntervalType2FuzzyRegressor


class TestIT2TrapezoidMembership:
    """Test IT2 with trapezoidal membership functions."""

    @pytest.fixture
    def iris_data(self):
        """Load iris dataset."""
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target
        )
        return X_train, X_test, y_train, y_test

    def test_trapz_classifier_fit_predict(self, iris_data):
        """Test IT2 classifier with trapezoidal memberships."""
        X_train, X_test, y_train, y_test = iris_data

        clf = IntervalType2FuzzyClassifier(
            top_n=3,
            member_function="trap",
            uncertainty_width=0.3,
            km_iterations=10,
            random_state=42,
        )

        clf.fit(X_train, y_train)
        assert clf.model_ is not None

        y_pred = clf.predict(X_test)
        assert y_pred.shape == (len(X_test),)

        accuracy = np.mean(y_pred == y_test)
        assert accuracy > 0.7, f"Trapz accuracy {accuracy} seems too low"

    def test_trapz_intervals(self, iris_data):
        """Test that trapezoid IT2 produces valid intervals."""
        X_train, X_test, y_train, y_test = iris_data

        clf = IntervalType2FuzzyClassifier(
            top_n=3,
            member_function="trap",
            uncertainty_width=0.3,
            km_iterations=None,
            random_state=42,
        )

        clf.fit(X_train, y_train)
        upper, lower = clf.predict_intervals(X_test)

        # Validate bounds
        assert np.all(lower <= upper), "Bounds should be ordered"
        assert np.all(lower >= 0) and np.all(upper >= 0), "Should be non-negative"


class TestIT2TriangularMembership:
    """Test IT2 with triangular membership functions."""

    @pytest.fixture
    def iris_data(self):
        """Load iris dataset."""
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target
        )
        return X_train, X_test, y_train, y_test

    def test_triangular_classifier_fit_predict(self, iris_data):
        """Test IT2 classifier with triangular memberships."""
        X_train, X_test, y_train, y_test = iris_data

        clf = IntervalType2FuzzyClassifier(
            top_n=3,
            member_function="triangular",
            uncertainty_width=0.3,
            km_iterations=10,
            random_state=42,
        )

        clf.fit(X_train, y_train)
        assert clf.model_ is not None

        y_pred = clf.predict(X_test)
        assert y_pred.shape == (len(X_test),)

        accuracy = np.mean(y_pred == y_test)
        assert accuracy > 0.7, f"Triangular accuracy {accuracy} seems too low"

    def test_triangular_intervals(self, iris_data):
        """Test that triangular IT2 produces valid intervals."""
        X_train, X_test, y_train, y_test = iris_data

        clf = IntervalType2FuzzyClassifier(
            top_n=3,
            member_function="triangular",
            uncertainty_width=0.3,
            km_iterations=None,
            random_state=42,
        )

        clf.fit(X_train, y_train)
        upper, lower = clf.predict_intervals(X_test)

        # Validate bounds
        assert np.all(lower <= upper), "Bounds should be ordered"
        assert np.all(lower >= 0) and np.all(upper >= 0), "Should be non-negative"


class TestIT2MembershipComparison:
    """Compare different membership types."""

    @pytest.fixture
    def iris_data(self):
        """Load iris dataset."""
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target
        )
        return X_train, X_test, y_train, y_test

    def test_gaussian_trapz_triangular_predictions(self, iris_data):
        """Compare predictions across membership types."""
        X_train, X_test, y_train, y_test = iris_data

        clf_gauss = IntervalType2FuzzyClassifier(
            top_n=3,
            member_function="gaussian",
            uncertainty_width=0.5,
            random_state=42,
        )
        clf_gauss.fit(X_train, y_train)

        clf_trapz = IntervalType2FuzzyClassifier(
            top_n=3,
            member_function="trap",
            uncertainty_width=0.3,
            random_state=42,
        )
        clf_trapz.fit(X_train, y_train)

        clf_tri = IntervalType2FuzzyClassifier(
            top_n=3,
            member_function="triangular",
            uncertainty_width=0.3,
            random_state=42,
        )
        clf_tri.fit(X_train, y_train)

        # All should make predictions
        y_pred_gauss = clf_gauss.predict(X_test)
        y_pred_trapz = clf_trapz.predict(X_test)
        y_pred_tri = clf_tri.predict(X_test)

        assert y_pred_gauss.shape == y_pred_trapz.shape == y_pred_tri.shape
        assert all(c in clf_gauss.classes_ for c in y_pred_gauss)
        assert all(c in clf_trapz.classes_ for c in y_pred_trapz)
        assert all(c in clf_tri.classes_ for c in y_pred_tri)


class TestIT2RegressionMembershipTypes:
    """Test IT2 regressor with different membership types."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic regression data."""
        np.random.seed(42)
        x = np.linspace(-2, 2, 100)
        y = np.sin(3 * x) + 0.05 * np.random.randn(len(x))
        X = pd.DataFrame({"x": x})
        y = pd.Series(y)

        split_idx = int(0.7 * len(x))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        return X_train, X_test, y_train, y_test

    def test_trapz_regressor(self, synthetic_data):
        """Test IT2 regressor with trapezoid memberships."""
        X_train, X_test, y_train, y_test = synthetic_data

        reg = IntervalType2FuzzyRegressor(
            top_n=2,
            n_gaussians=2,
            member_function="trap",
            uncertainty_width=0.3,
            random_state=42,
        )

        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        assert y_pred.shape == (len(X_test),)
        assert np.all(np.isfinite(y_pred))

    def test_triangular_regressor(self, synthetic_data):
        """Test IT2 regressor with triangular memberships."""
        X_train, X_test, y_train, y_test = synthetic_data

        reg = IntervalType2FuzzyRegressor(
            top_n=2,
            n_gaussians=2,
            member_function="triangular",
            uncertainty_width=0.3,
            random_state=42,
        )

        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        y_lower, y_upper = reg.predict_intervals(X_test)

        assert y_pred.shape == (len(X_test),)
        assert np.all(y_lower <= y_upper)
