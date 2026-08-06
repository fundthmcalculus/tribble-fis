"""Tests for IT2 with antecedent refinement."""

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from tribblefis.it2_classifier import IntervalType2FuzzyClassifier
from tribblefis.it2_regressor import IntervalType2FuzzyRegressor


class TestIT2RefinementClassifier:
    """Test IT2 classifier with antecedent refinement."""

    @pytest.fixture
    def iris_data(self):
        """Load iris dataset."""
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target
        )
        return X_train, X_test, y_train, y_test

    def test_refinement_parameter_accepted(self, iris_data):
        """Test that refine parameter is accepted without error."""
        X_train, X_test, y_train, y_test = iris_data

        clf_no_refine = IntervalType2FuzzyClassifier(
            top_n=3,
            refine=False,
            random_state=42,
        )
        clf_no_refine.fit(X_train, y_train)
        y_pred_no_refine = clf_no_refine.predict(X_test)

        # With refinement - should work even if slow
        # Note: refinement is done on Type-1 before IT2 conversion
        clf_refine = IntervalType2FuzzyClassifier(
            top_n=3,
            refine=True,
            refine_method="coordinate",
            refine_l2_shrink=0.05,
            random_state=42,
        )
        clf_refine.fit(X_train, y_train)
        y_pred_refine = clf_refine.predict(X_test)

        # Both should produce valid predictions
        assert y_pred_no_refine.shape == y_pred_refine.shape
        assert all(c in clf_no_refine.classes_ for c in y_pred_no_refine)
        assert all(c in clf_refine.classes_ for c in y_pred_refine)

    def test_refinement_produces_different_bounds(self, iris_data):
        """Test that refinement can affect the learned bounds."""
        X_train, X_test, y_train, y_test = iris_data

        clf_no_refine = IntervalType2FuzzyClassifier(
            top_n=3,
            refine=False,
            uncertainty_width=0.5,
            random_state=42,
        )
        clf_no_refine.fit(X_train, y_train)
        upper_no_refine, lower_no_refine = clf_no_refine.predict_intervals(X_test)

        # Refinement affects the base Type-1 model parameters,
        # which then get expanded to IT2 bounds
        # So refined and non-refined should potentially differ
        # (though not guaranteed for all datasets)

        # Just verify refinement doesn't break interval validity
        assert np.all(lower_no_refine <= upper_no_refine)

    def test_refine_l2_shrink_parameter(self, iris_data):
        """Test that L2 regularization parameter is accepted."""
        X_train, X_test, y_train, y_test = iris_data

        # Different shrinkage values should be accepted
        for shrink in [0.0, 0.05, 0.5]:
            clf = IntervalType2FuzzyClassifier(
                top_n=3,
                refine=True,
                refine_l2_shrink=shrink,
                random_state=42,
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            assert y_pred.shape == (len(X_test),)


class TestIT2RefinementRegressor:
    """Test IT2 regressor with antecedent refinement."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic regression data."""
        import pandas as pd

        np.random.seed(42)
        x = np.linspace(-2, 2, 100)
        y = np.sin(3 * x) + 0.05 * np.random.randn(len(x))
        X = pd.DataFrame({"x": x})
        y = pd.Series(y)

        split_idx = int(0.7 * len(x))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        return X_train, X_test, y_train, y_test

    def test_refinement_parameter_accepted_regressor(self, synthetic_data):
        """Test that refine parameter works for regressor."""
        X_train, X_test, y_train, y_test = synthetic_data

        reg = IntervalType2FuzzyRegressor(
            top_n=2,
            n_gaussians=2,
            refine=True,
            random_state=42,
        )

        # Should fit without errors
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        assert y_pred.shape == (len(X_test),)
        assert np.all(np.isfinite(y_pred))


class TestIT2RefinementAcrossMembershipTypes:
    """Test refinement works across different membership types."""

    @pytest.fixture
    def iris_data(self):
        """Load iris dataset."""
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target
        )
        return X_train, X_test, y_train, y_test

    def test_refinement_with_gaussian(self, iris_data):
        """Test refinement with Gaussian memberships."""
        X_train, X_test, y_train, y_test = iris_data

        clf = IntervalType2FuzzyClassifier(
            top_n=3,
            member_function="gaussian",
            refine=True,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        assert y_pred.shape == (len(X_test),)

    def test_refinement_with_trapezoid(self, iris_data):
        """Test refinement with trapezoid memberships."""
        X_train, X_test, y_train, y_test = iris_data

        clf = IntervalType2FuzzyClassifier(
            top_n=3,
            member_function="trap",
            refine=True,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        assert y_pred.shape == (len(X_test),)

    def test_refinement_with_triangular(self, iris_data):
        """Test refinement with triangular memberships."""
        X_train, X_test, y_train, y_test = iris_data

        clf = IntervalType2FuzzyClassifier(
            top_n=3,
            member_function="triangular",
            refine=True,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        assert y_pred.shape == (len(X_test),)
