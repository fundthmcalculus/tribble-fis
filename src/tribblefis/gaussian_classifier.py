import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets

from .gauss_data import AnomalyParameters
from .gauss_math import (
    calculate_gaussian_correlation,
    take_top_features,
    create_gaussian_membership_dict,
    tsk_predict,
    tsk_firing_strengths
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
        """
        Check if features need log-transformation and apply it.
        """
        if not self.log_transform:
            return X

        X_transformed = X.copy()
        
        # If already fitted, use the stored list of features to transform
        if self.is_fitted_:
            for col in self.log_transformed_features_:
                X_transformed[col] = np.log1p(X_transformed[col].clip(lower=0))
            return X_transformed

        # During fit, identify features that need transformation
        self.log_transformed_features_ = []
        for col in X.columns:
            # We check if the feature has a broad range of scales
            # A common heuristic is the ratio of max/min, but that's sensitive to outliers.
            # Another is looking at the number of orders of magnitude.
            vals = X[col].dropna()
            if len(vals) == 0:
                continue
                
            # Only consider positive values for log transform suggestion
            # If there are many zeros or negative values, we might need an offset, 
            # but log1p(clip(0)) is a safe start.
            
            # Simple heuristic: if max / (min + epsilon) > 1000 and max > 1.0
            # Or if the distribution is highly skewed.
            v_min = vals.min()
            v_max = vals.max()
            
            if v_max > v_min and v_max > 0:
                # Use a small epsilon to avoid division by zero
                # If values span more than 3 orders of magnitude
                if v_min > 0:
                    ratio = v_max / v_min
                else:
                    # If min is 0 or negative, we look at the range relative to a small value
                    # or just check if max is large.
                    ratio = v_max / (vals[vals > 0].min() if any(vals > 0) else 1e-6)
                
                if ratio > 1000:
                    self.log_transformed_features_.append(col)
                    print(f"  Suggesting log-transform for feature '{col}' (ratio max/min: {ratio:.2f})")
        
        for col in self.log_transformed_features_:
            X_transformed[col] = np.log1p(X_transformed[col].clip(lower=0))
            
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

    def firing_strengths(self, X, anomaly_details: AnomalyParameters | None = None):
        """
        Compute the raw TSK firing strengths for X, optionally including an
        anomaly column.

        Args:
            X: Input data (n_samples, n_features)
            anomaly_details: If provided, an extra "anomaly" column is appended
                whose strength rises as every class membership falls.

        Returns:
            (firing_strengths, labels) where ``firing_strengths`` is a
            (n_samples, n_labels) array and ``labels`` lists the column labels
            (the anomaly label is last when ``anomaly_details`` is supplied).
        """
        check_is_fitted(self)

        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)

        X_df = self._apply_log_transform(X_df)

        return tsk_firing_strengths(X_df, self.model_, anomaly_details=anomaly_details)

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


class MixtureOfGaussiansFuzzySequenceClassifier(BaseEstimator, ClassifierMixin):
    """
    A sequence (cascade) of :class:`MixtureOfGaussiansFuzzyClassifier` models.

    The first ("primary") model is fit on all of the training data. Each
    subsequent model is a *binary specialist* attached to one
    **(confused class, true class) pair** ``(P, T)`` — the largest off-diagonal
    cell of the running confusion matrix, i.e. the predicted class ``P`` and the
    true class ``T`` it is most often mistaken for. A specialist is fit on every
    training row the upstream stage *predicted* as ``P``, with the true labels
    collapsed to ``{P, T}``, so it learns only to peel the ``T`` rows out of the
    ``P`` region. Specialists run with anomaly detection enabled so they know the
    boundary of their region.

    At prediction time the models are applied one after another:

    * The primary model produces an initial prediction for every sample.
    * Each specialist is consulted only for the samples whose *current*
      prediction equals that specialist's confused class ``P``. For such a
      sample:

      - if the specialist flags it as an ``anomaly`` (the sample lies outside
        the confused region the specialist was trained on), the sample is frozen
        and no further specialists are applied to it — the running prediction
        (``P``) is kept;
      - otherwise the specialist's binary verdict *refines* the running
        prediction (relabelling to ``T`` or keeping ``P``), and the sample may go
        on to a later specialist keyed to that new label.

    This follows scikit-learn's ``ClassifierMixin`` interface.
    """

    def __init__(
        self,
        top_n=-1,
        top_p=0.95,
        n_gaussians=0,
        log_transform=False,
        random_state=42,
        max_layers=4,
        anomaly_threshold=0.99,
        anomaly_label="anomaly",
        norm_conorm="min/max",
        member_function="gaussian",
        min_confused=20,
        min_class_samples=5,
    ):
        """
        Args:
            top_n, top_p, n_gaussians, log_transform, random_state:
                Passed through to every underlying
                :class:`MixtureOfGaussiansFuzzyClassifier` layer.
            max_layers: Maximum number of models in the cascade (including the
                primary model). The cascade may be shorter if no further
                confused class is worth specializing on.
            anomaly_threshold: Anomaly threshold used by the specialist layers
                (see :class:`AnomalyParameters`). Higher values make a specialist
                more willing to declare a sample an anomaly (and stop).
            anomaly_label: Label used to mark anomalies. It must not collide with
                a real class label.
            norm_conorm, member_function: Fuzzy operators used when evaluating
                the specialist layers' anomaly-aware firing strengths.
            min_confused: Minimum number of rows predicted as a given confused
                class required before any pair rooted at it is specialized.
            min_class_samples: Minimum number of confused rows in a
                ``(predicted, true)`` pair before a specialist is trained for it;
                rarer confusions are left as the confused-class prediction.
        """
        self.is_fitted_: bool = False
        # layers_[0] is the primary model; layers_[1:] mirror specialists_.
        self.layers_: list[MixtureOfGaussiansFuzzyClassifier] = []
        # Each entry is (confused_class, true_class, specialist_model).
        self.specialists_: list[tuple] = []
        self.classes_ = None
        self.feature_names_in_: list[str] = []
        self.top_n = top_n
        self.top_p = top_p
        self.n_gaussians = n_gaussians
        self.log_transform = log_transform
        self.random_state = random_state
        self.max_layers = max_layers
        self.anomaly_threshold = anomaly_threshold
        self.anomaly_label = anomaly_label
        self.norm_conorm = norm_conorm
        self.member_function = member_function
        self.min_confused = min_confused
        self.min_class_samples = min_class_samples

    def _make_layer(self) -> MixtureOfGaussiansFuzzyClassifier:
        return MixtureOfGaussiansFuzzyClassifier(
            top_n=self.top_n,
            top_p=self.top_p,
            n_gaussians=self.n_gaussians,
            log_transform=self.log_transform,
            random_state=self.random_state,
        )

    def _anomaly_params(self) -> AnomalyParameters:
        return AnomalyParameters(
            include_anomaly=True,
            threshold=self.anomaly_threshold,
            label=self.anomaly_label,
            norm_conorm=self.norm_conorm,
            member_function=self.member_function,
        )

    @staticmethod
    def _as_frame_series(X, y):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y_series = y if isinstance(y, pd.Series) else pd.Series(np.asarray(y))
        # Align indices so boolean masking lines up cleanly.
        X_df = X_df.reset_index(drop=True)
        y_series = y_series.reset_index(drop=True)
        return X_df, y_series

    def _most_confused_pair(self, running, y_true, handled) -> tuple | None:
        """The unhandled ``(predicted_class, true_class)`` confusion with the
        most misclassified rows.

        This is the largest off-diagonal cell of the running confusion matrix:
        the predicted class ``P`` and the true class ``T != P`` that ``P`` is most
        often mistaken for. Returns ``None`` when no remaining pair has enough
        rows in its predicted region and enough confused samples to be worth a
        specialist.
        """
        best_pair, best_errors = None, 0
        for p in np.unique(running):
            rows = running == p
            n_rows = int(rows.sum())
            if n_rows < self.min_confused:
                continue
            true_here = y_true[rows]
            for t in np.unique(true_here):
                if t == p or (p, t) in handled:
                    continue
                errors = int(np.sum(true_here == t))
                if errors > best_errors and errors >= self.min_class_samples:
                    best_pair, best_errors = (p, t), errors
        return best_pair

    def fit(self, X, y):
        """
        Fit the cascade.

        The primary model is fit on all data. Then, repeatedly, the single
        ``(predicted_class P, true_class T)`` pair with the most remaining
        confusion (the largest off-diagonal confusion-matrix cell) is selected
        and a *binary* specialist is trained on every row predicted as ``P`` to
        arbitrate between ``T`` (peel off) and ``P`` (keep). The running
        predictions are updated with each specialist before the next pair is
        chosen, up to ``max_layers`` models.
        """
        X_df, y_series = self._as_frame_series(X, y)
        self.feature_names_in_ = X_df.columns.tolist()
        self.classes_ = np.unique(y_series.values)

        if self.anomaly_label in set(self.classes_.tolist()):
            raise ValueError(
                f"anomaly_label={self.anomaly_label!r} collides with a real class label."
            )

        # Layer 0: the primary model, trained on everything.
        primary = self._make_layer()
        primary.fit(X_df, y_series)
        self.layers_ = [primary]
        self.specialists_ = []

        y_true = y_series.values.astype(object)
        running = np.asarray(primary.predict(X_df), dtype=object)
        handled: set = set()

        for _ in range(1, self.max_layers):
            pair = self._most_confused_pair(running, y_true, handled)
            if pair is None:
                break
            confused_class, true_class = pair
            handled.add(pair)

            # Train a binary specialist on every row the cascade currently
            # predicts as ``confused_class``. Its job is to peel off the rows
            # that are really ``true_class`` while leaving the rest as the
            # confused class, so the true labels are collapsed to {P, T}.
            rows = running == confused_class
            y_region = y_series[rows].reset_index(drop=True)
            y_sub = y_region.where(y_region == true_class, confused_class)
            X_sub = X_df[rows].reset_index(drop=True)

            if y_sub.nunique() < 2:
                # Nothing for the specialist to disambiguate here; try another pair.
                continue

            specialist = self._make_layer()
            specialist.fit(X_sub, y_sub)
            self.specialists_.append((confused_class, true_class, specialist))
            self.layers_.append(specialist)

            # Update the running predictions for the affected rows so the next
            # confused pair is chosen against the refined state.
            running[rows] = np.asarray(specialist.predict(X_df[rows]), dtype=object)

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Predict class labels for X by running the cascade.

        Returns an array of class labels. The anomaly label is never returned;
        an anomaly flag from a specialist only freezes the sample (stopping any
        further specialists) and leaves the running prediction in place.
        """
        check_is_fitted(self)
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_in_)

        preds = np.asarray(self.layers_[0].predict(X_df), dtype=object)

        # ``frozen`` marks samples a specialist declared anomalous; no further
        # specialist may touch them.
        frozen = np.zeros(len(X_df), dtype=bool)
        anomaly_params = self._anomaly_params()

        for confused_class, _true_class, specialist in self.specialists_:
            # Only route in the samples whose current prediction *is* this
            # specialist's confused class.
            target = (preds == confused_class) & ~frozen
            if not target.any():
                continue

            firing_strengths, labels = specialist.firing_strengths(X_df, anomaly_details=anomaly_params)
            best_idx = np.argmax(firing_strengths, axis=1)
            layer_pred = np.array([labels[i] for i in best_idx], dtype=object)
            is_anomaly = layer_pred == self.anomaly_label

            # Refine non-anomalous targets; freeze anomalous ones (keep the
            # confused class and stop applying further specialists to them).
            refine = target & ~is_anomaly
            preds[refine] = layer_pred[refine]
            frozen = frozen | (target & is_anomaly)

        return preds

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Probabilities come from the primary model, which is the only layer
        guaranteed to span every class. The cascade refines the hard label via
        :meth:`predict`; for calibrated probabilities prefer that method's
        output combined with the primary scores.
        """
        check_is_fitted(self)
        return self.layers_[0].predict_proba(X)

    @property
    def confused_classes_(self) -> list:
        """The confused (predicted) class each specialist is keyed to, in order."""
        return [cls for cls, _, _ in self.specialists_]

    @property
    def confused_pairs_(self) -> list:
        """The ``(predicted_class, true_class)`` pair each specialist arbitrates."""
        return [(cls, true) for cls, true, _ in self.specialists_]

    @property
    def n_layers(self) -> int:
        return len(self.layers_)
