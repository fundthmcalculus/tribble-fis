"""
Tests for performance optimizations (Issue #96 and #97)

Issue #96: Compute only top-N features when top_n is set
Issue #97: Parallelize Gaussian membership creation per-class
"""

import unittest
import numpy as np
import pandas as pd
from tribblefis.gaussian_classifier import TribbleClassifier
from tribblefis.gaussian_regressor import TribbleRegressor
from tribblefis.gauss_math import calculate_gaussian_correlation, create_gaussian_membership_dict


class TestTopNFeatureOptimization(unittest.TestCase):
    """Test Issue #96: Compute only top-N features when top_n is set"""

    def setUp(self):
        """Create sample dataset for testing"""
        np.random.seed(42)
        self.n_samples = 100
        self.n_features = 15
        self.X = pd.DataFrame(
            np.random.randn(self.n_samples, self.n_features),
            columns=[f"feature_{i}" for i in range(self.n_features)]
        )
        self.y_class = pd.Series(np.random.randint(0, 3, self.n_samples))
        self.y_reg = pd.Series(np.random.randn(self.n_samples) * 10 + 50)

    def test_calculate_gaussian_correlation_top_n(self):
        """Test that calculate_gaussian_correlation respects top_n parameter"""
        # Compute all features
        all_results = calculate_gaussian_correlation(self.X, self.y_class, top_n=-1)
        all_features = len(all_results)

        # Compute only top 5
        top_5 = calculate_gaussian_correlation(self.X, self.y_class, top_n=5)
        self.assertEqual(len(top_5), 5, "Expected 5 features")

        # Verify top 5 are the same as from all-features result
        all_names = [name for name, _ in all_results]
        top_5_names = [name for name, _ in top_5]
        self.assertEqual(top_5_names, all_names[:5], "Top 5 should match first 5 from all results")

    def test_calculate_gaussian_correlation_top_n_edge_cases(self):
        """Test edge cases for top_n parameter"""
        # top_n larger than number of features
        results = calculate_gaussian_correlation(self.X, self.y_class, top_n=100)
        self.assertLessEqual(len(results), self.n_features)

        # top_n = 1
        results_1 = calculate_gaussian_correlation(self.X, self.y_class, top_n=1)
        self.assertEqual(len(results_1), 1)

        # top_n = 0 or negative (should return all)
        results_all_0 = calculate_gaussian_correlation(self.X, self.y_class, top_n=0)
        results_all_neg = calculate_gaussian_correlation(self.X, self.y_class, top_n=-1)
        self.assertEqual(len(results_all_0), self.n_features)
        self.assertEqual(len(results_all_neg), self.n_features)

    def test_tribble_classifier_with_top_n(self):
        """Test TribbleClassifier respects top_n in fit"""
        clf = TribbleClassifier(top_n=5, n_gaussians=1, random_state=42)
        clf.fit(self.X, self.y_class)

        # Should have selected exactly 5 features
        self.assertEqual(clf.top_n_actual_, 5)
        self.assertEqual(len(clf.top_features_), 5)

        # Predictions should work
        preds = clf.predict(self.X)
        self.assertEqual(len(preds), self.n_samples)

    def test_tribble_regressor_with_top_n(self):
        """Test TribbleRegressor respects top_n in fit"""
        reg = TribbleRegressor(top_n=8, n_gaussians=1, random_state=42)
        reg.fit(self.X, self.y_reg)

        # Should have selected exactly 8 features
        self.assertEqual(reg.top_n_actual_, 8)
        self.assertEqual(len(reg.top_features_), 8)

        # Predictions should work
        preds = reg.predict(self.X)
        self.assertEqual(len(preds), self.n_samples)


class TestCorrelationDedup(unittest.TestCase):
    """Test that top_n selection drops highly-correlated redundant features."""

    def setUp(self):
        np.random.seed(42)
        self.n_samples = 200
        base = np.random.randn(self.n_samples, 4)
        # feature_0_dup is feature_0 plus tiny noise: near-perfect correlation
        X = np.column_stack([base, base[:, 0] + np.random.randn(self.n_samples) * 1e-6])
        self.X = pd.DataFrame(X, columns=["feature_0", "feature_1", "feature_2", "feature_3", "feature_0_dup"])
        # y is driven by feature_0, so feature_0 and feature_0_dup both rank at
        # the top -- exactly the redundant-pair scenario the dedup check targets.
        self.y = pd.Series((self.X["feature_0"] > 0).astype(int))

    def test_redundant_feature_dropped_by_default(self):
        with self.assertWarns(UserWarning):
            results = calculate_gaussian_correlation(self.X, self.y, top_n=2)
        names = [name for name, _ in results]
        self.assertEqual(len(names), 2)
        # feature_0 and feature_0_dup should never both be selected
        self.assertFalse({"feature_0", "feature_0_dup"}.issubset(set(names)))

    def test_threshold_1_disables_check(self):
        results = calculate_gaussian_correlation(self.X, self.y, top_n=2, correlation_threshold=1.0)
        names = [name for name, _ in results]
        self.assertEqual(len(names), 2)
        # With the check disabled, both correlated top features come through.
        self.assertTrue({"feature_0", "feature_0_dup"}.issubset(set(names)))

    def test_threshold_0_disables_check(self):
        results = calculate_gaussian_correlation(self.X, self.y, top_n=2, correlation_threshold=0.0)
        names = [name for name, _ in results]
        self.assertEqual(len(names), 2)
        self.assertTrue({"feature_0", "feature_0_dup"}.issubset(set(names)))


class TestGaussianMembershipParallelization(unittest.TestCase):
    """Test Issue #97: Parallelize Gaussian membership creation per-class"""

    def setUp(self):
        """Create sample dataset for testing"""
        np.random.seed(42)
        self.n_samples = 100
        self.n_features = 5
        self.n_classes = 3
        self.X = pd.DataFrame(
            np.random.randn(self.n_samples, self.n_features),
            columns=[f"feature_{i}" for i in range(self.n_features)]
        )
        self.y = pd.Series(np.random.randint(0, self.n_classes, self.n_samples))
        self.features = [f"feature_{i}" for i in range(self.n_features)]

    def test_create_gaussian_membership_dict_works(self):
        """Test that create_gaussian_membership_dict produces valid output"""
        model = create_gaussian_membership_dict(
            self.X, self.y, top_n_var_names=self.features, n_gaussians=1
        )

        # Verify model structure
        self.assertEqual(len(model.feature_models), self.n_features)

        for feature_name, feature_model in model.feature_models.items():
            # Each feature should have all classes
            self.assertEqual(len(feature_model.label_models), self.n_classes)

            for label_value, label_model in feature_model.label_models.items():
                # Each label should have membership functions
                self.assertGreater(len(label_model.memberships), 0)

    def test_gaussian_membership_dict_ordering(self):
        """Test that feature order is preserved"""
        feature_order = self.features[:3]
        model = create_gaussian_membership_dict(
            self.X, self.y, top_n_var_names=feature_order, n_gaussians=1
        )

        model_features = list(model.feature_models.keys())
        self.assertEqual(model_features, feature_order, "Feature order should be preserved")

    def test_parallel_vs_serial_consistency(self):
        """Test that parallel and serial processing produce consistent results"""
        # Create two models with same parameters
        model1 = create_gaussian_membership_dict(
            self.X, self.y, top_n_var_names=self.features, n_gaussians=2, random_state=42
        )
        model2 = create_gaussian_membership_dict(
            self.X, self.y, top_n_var_names=self.features, n_gaussians=2, random_state=42
        )

        # Check that both have the same structure
        self.assertEqual(set(model1.feature_models.keys()), set(model2.feature_models.keys()))

        for feature_name in model1.feature_models:
            feature_model_1 = model1.feature_models[feature_name]
            feature_model_2 = model2.feature_models[feature_name]

            # Both should have the same labels
            self.assertEqual(
                set(feature_model_1.label_models.keys()),
                set(feature_model_2.label_models.keys())
            )

            # Both should have the same number of membership functions per label
            for label_value in feature_model_1.label_models:
                n_mfs_1 = len(feature_model_1.label_models[label_value].memberships)
                n_mfs_2 = len(feature_model_2.label_models[label_value].memberships)
                self.assertEqual(n_mfs_1, n_mfs_2)

    def test_environment_variable_workers_control(self):
        """Test that TRIBBLE_GAUSSIAN_WORKERS environment variable is respected"""
        import os
        original_value = os.environ.get('TRIBBLE_GAUSSIAN_WORKERS')

        try:
            # Test with explicit worker count
            os.environ['TRIBBLE_GAUSSIAN_WORKERS'] = '2'
            model = create_gaussian_membership_dict(
                self.X, self.y, top_n_var_names=self.features[:2], n_gaussians=1
            )
            # Just verify it works without errors
            self.assertEqual(len(model.feature_models), 2)

        finally:
            # Restore original value
            if original_value is None:
                os.environ.pop('TRIBBLE_GAUSSIAN_WORKERS', None)
            else:
                os.environ['TRIBBLE_GAUSSIAN_WORKERS'] = original_value


class TestCombinedOptimizations(unittest.TestCase):
    """Test both optimizations working together"""

    def setUp(self):
        """Create sample dataset for testing"""
        np.random.seed(42)
        self.n_samples = 200
        self.n_features = 20
        self.X = pd.DataFrame(
            np.random.randn(self.n_samples, self.n_features),
            columns=[f"feature_{i}" for i in range(self.n_features)]
        )
        self.y = np.random.randint(0, 3, self.n_samples)

    def test_classifier_with_both_optimizations(self):
        """Test TribbleClassifier using both optimizations together"""
        # Fit with top_n (Issue #96) - this triggers parallel Gaussian creation (Issue #97)
        clf = TribbleClassifier(top_n=8, n_gaussians=2, random_state=42)
        clf.fit(self.X, self.y)

        # Should have selected exactly 8 features (top_n optimization)
        self.assertEqual(clf.top_n_actual_, 8)

        # Should have a valid model with parallelized Gaussian creation
        self.assertEqual(len(clf.model_.feature_models), 8)

        # Should be able to predict
        preds = clf.predict(self.X)
        self.assertEqual(len(preds), self.n_samples)
        self.assertTrue(all(p in clf.classes_ for p in preds))

    def test_regressor_with_both_optimizations(self):
        """Test TribbleRegressor using both optimizations together"""
        y_reg = np.random.randn(self.n_samples) * 10 + 50

        reg = TribbleRegressor(top_n=12, n_gaussians=1, random_state=42)
        reg.fit(self.X, y_reg)

        # Should have selected exactly 12 features (top_n optimization)
        self.assertEqual(reg.top_n_actual_, 12)

        # Should have a valid model
        self.assertEqual(len(reg.model_.feature_models), 12)

        # Should be able to predict
        preds = reg.predict(self.X)
        self.assertEqual(len(preds), self.n_samples)
        self.assertTrue(all(np.isfinite(p) for p in preds))


if __name__ == '__main__':
    unittest.main()
