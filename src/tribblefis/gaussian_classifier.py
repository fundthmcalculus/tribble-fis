import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets

from .gauss_math import (
    calculate_gaussian_correlation,
    take_top_features,
    create_gaussian_membership_dict,
    tsk_predict,
    tsk_firing_strengths,
    detect_and_apply_log_transform,
)

class MixtureOfGaussiansFuzzyClassifier(BaseEstimator, ClassifierMixin):
    """
    Gaussian Mixture Classifier that wraps the TSK-based Gaussian Mixture model.
    It follows scikit-learn's ClassifierMixin interface.
    """

    def __init__(self, top_n=-1, top_p=0.95, n_gaussians=0, log_transform=False, random_state=42):
        """
        Initialize the MixtureOfGaussiansFuzzyClassifier.

        Args:
            top_n: Number of top features to select based on differentiation score.
                   If > 0, top_p is ignored.
            top_p: Percentage of cumulative differentiation score to cover.
            n_gaussians: Number of Gaussians per feature per label (0 for automatic).
                         Can also be a dictionary mapping feature names or labels to number of Gaussians.
            log_transform: Whether to automatically suggest and apply log-transformation to features
                           that have a broad range of scales.
            random_state: Seed for random number generator for reproducibility.
        """
        self.is_fitted_: bool = False
        self.model_ = None
        self.top_features_ = None
        self.top_n_actual_ = None
        self.feature_differentiators_ = None
        self.classes_ = None
        self.feature_names_in_: list[str] = []
        self.log_transformed_features_: list[str] = []
        self.top_n = top_n
        self.top_p = top_p
        self.n_gaussians = n_gaussians
        self.log_transform = log_transform
        self.random_state = random_state

    def _apply_log_transform(self, X):
        """Check if features need log-transformation and apply it."""
        if not self.log_transform:
            return X

        X_transformed, features = detect_and_apply_log_transform(
            X, already_fitted=self.is_fitted_, fitted_features=self.log_transformed_features_
        )

        if not self.is_fitted_:
            self.log_transformed_features_ = features

        return X_transformed

    def fit(self, X, y):
        """
        Fit the Gaussian Mixture model.

        Args:
            X: Training data (n_samples, n_features)
            y: Target values (n_samples,)
        """
        # If X is a DataFrame, keep track of feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
        else:
            self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        # Standard sklearn validation
        X_array, y_array = check_X_y(X, y)
        check_classification_targets(y_array)

        # Store classes
        self.classes_ = np.unique(y_array)
        
        # We need X as DataFrame for the internal functions
        X_df = pd.DataFrame(X_array, columns=self.feature_names_in_)
        y_series = pd.Series(y_array)

        # 0. Apply log-transformation if requested
        X_df = self._apply_log_transform(X_df)

        # 1. Calculate feature differentiators
        self.feature_differentiators_ = calculate_gaussian_correlation(X_df, y_series)

        # 2. Select top features
        self.top_n_actual_, self.top_features_ = take_top_features(
            self.feature_differentiators_, top_p=self.top_p, top_n=self.top_n
        )

        # 3. Create Gaussian membership model
        self.model_ = create_gaussian_membership_dict(
            X_df, y_series, top_n_var_names=self.top_features_, n_gaussians=self.n_gaussians
        )

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Predict class labels for X.

        Args:
            X: Input data (n_samples, n_features)
        """
        check_is_fitted(self)
        
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)

        X_df = self._apply_log_transform(X_df)

        return tsk_predict(X_df, self.model_)

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Args:
            X: Input data (n_samples, n_features)
        """
        check_is_fitted(self)

        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)

        X_df = self._apply_log_transform(X_df)

        firing_strengths, labels = tsk_firing_strengths(X_df, self.model_)
        
        # Normalize firing strengths to get probabilities
        # Adding a small epsilon to avoid division by zero
        row_sums = firing_strengths.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1e-10
        
        probabilities = firing_strengths / row_sums
        
        # Ensure the columns match self.classes_ order
        label_to_idx = {label: i for i, label in enumerate(labels)}

        reordered_probs = np.zeros((len(X), len(self.classes_)))
        for i, cls in enumerate(self.classes_):
            if cls in label_to_idx:
                reordered_probs[:, i] = probabilities[:, label_to_idx[cls]]
                
        return reordered_probs

    def augment(self, X, y):
        """
        Augment the existing model with new data (similar to the 2-pass approach).
        """
        check_is_fitted(self)
        
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)
        y_series = pd.Series(y)

        X_df = self._apply_log_transform(X_df)
        
        new_model = create_gaussian_membership_dict(
            X_df, y_series, top_n_var_names=self.top_features_, n_gaussians=self.n_gaussians
        )
        
        self.model_ = self.model_.augment(new_model)
        return self
