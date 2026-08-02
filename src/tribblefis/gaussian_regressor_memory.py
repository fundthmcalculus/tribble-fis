"""
Enhanced MIMO Gaussian Regressor with Memory Features.

Extends the standard MIMO model to include temporal memory features:
- Current time step (or explicit time)
- Average of last-N time steps (short-term memory)
- Previous average-of-last-N time steps (long-term memory)

This enables a modified LSTM-like operation without recurrent layers.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .gaussian_regressor import MimoGaussianPredictor


class MemoryWindowFeatureExtractor:
    """Extracts memory-augmented features from sequential data."""

    def __init__(self, window_size: int = 3, memory_size: int = 5):
        """
        Initialize memory feature extractor.

        Args:
            window_size: Size of current window for averaging (short-term memory).
                        If 1, uses current value only.
            memory_size: Size of previous window for long-term memory.
                        Must be < window_size for meaningful separation.
        """
        if memory_size >= window_size:
            raise ValueError(f"memory_size ({memory_size}) must be < window_size ({window_size})")

        self.window_size = window_size
        self.memory_size = memory_size

    def prepare_sequences(
        self, df: pd.DataFrame, feature_columns: list[str], include_time: bool = True
    ) -> pd.DataFrame:
        """
        Prepare sequential data with memory features.

        For each row t, creates:
        - time_step: current time index (if include_time=True)
        - {feat}_current: feature value at time t
        - {feat}_short_term_avg: average of features[t-window_size+1:t+1]
        - {feat}_long_term_avg: average of features[t-window_size-memory_size:t-window_size+1]

        Args:
            df: Input DataFrame with feature columns
            feature_columns: List of column names to include
            include_time: Whether to add explicit time_step feature

        Returns:
            DataFrame with augmented memory features
        """
        n_samples = len(df)
        result = pd.DataFrame()

        # Add time step as a feature if requested
        if include_time:
            result["time_step"] = np.arange(n_samples)

        # For each original feature, create memory variants
        for feat in feature_columns:
            values = df[feat].values

            # Current value
            result[f"{feat}_current"] = values

            # Short-term memory (average of last window_size steps)
            short_term = np.full(n_samples, np.nan)
            for i in range(n_samples):
                start_idx = max(0, i - self.window_size + 1)
                short_term[i] = np.mean(values[start_idx : i + 1])
            result[f"{feat}_short_term_avg"] = short_term

            # Long-term memory (average of steps before the short-term window)
            long_term = np.full(n_samples, np.nan)
            for i in range(n_samples):
                long_start = max(0, i - self.window_size - self.memory_size + 1)
                long_end = max(0, i - self.window_size + 1)
                if long_start < long_end:
                    long_term[i] = np.mean(values[long_start:long_end])
                else:
                    long_term[i] = np.nan
            result[f"{feat}_long_term_avg"] = long_term

        return result

    def get_feature_names(self, feature_columns: list[str], include_time: bool = True) -> list[str]:
        """Get the full list of memory-augmented feature names."""
        names = []
        if include_time:
            names.append("time_step")
        for feat in feature_columns:
            names.extend(
                [
                    f"{feat}_current",
                    f"{feat}_short_term_avg",
                    f"{feat}_long_term_avg",
                ]
            )
        return names


def prepare_mimo_data_with_memory(
    trajectories: list[pd.DataFrame],
    input_features: list[str],
    output_features: list[str],
    window_size: int = 3,
    memory_size: int = 5,
    include_time: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare MIMO training data with memory features.

    Args:
        trajectories: List of trajectory DataFrames
        input_features: Features to include in input
        output_features: Features to predict (as deltas)
        window_size: Size of short-term memory window
        memory_size: Size of long-term memory window
        include_time: Whether to include explicit time feature

    Returns:
        Tuple of (X_combined, y_combined) arrays
    """
    extractor = MemoryWindowFeatureExtractor(window_size=window_size, memory_size=memory_size)

    all_X = []
    all_y = []

    for df in trajectories:
        # Create memory features for inputs
        X_augmented = extractor.prepare_sequences(df, input_features, include_time=include_time)

        # Remove rows with NaN (due to insufficient history)
        valid_rows = ~X_augmented.isna().any(axis=1)
        X_augmented = X_augmented[valid_rows].reset_index(drop=True)

        # Align outputs: predict delta at next time step
        # Since we removed NaNs, we need to index from the original df
        valid_indices = np.where(valid_rows)[0]
        valid_indices_for_y = valid_indices[valid_indices < len(df) - 1]

        # Get deltas for valid indices (y[t] = state[t+1] - state[t])
        y_deltas = []
        for idx in valid_indices_for_y:
            delta = df[output_features].iloc[idx + 1].values - df[output_features].iloc[idx].values
            y_deltas.append(delta)

        if len(y_deltas) == 0:
            continue

        X_aligned = X_augmented.iloc[: len(y_deltas)].values
        y_aligned = np.array(y_deltas)

        all_X.append(X_aligned)
        all_y.append(y_aligned)

    if not all_X:
        raise ValueError("No valid data could be extracted from trajectories")

    X_combined = np.vstack(all_X)
    y_combined = np.vstack(all_y)

    return X_combined, y_combined


class MimoGaussianPredictorMemory(BaseEstimator, RegressorMixin):
    """
    Enhanced MIMO predictor with memory features.

    Combines current state, short-term memory (recent average), and long-term
    memory (older average) to enable temporal reasoning without explicit RNNs.
    """

    def __init__(
        self,
        window_size: int = 3,
        memory_size: int = 5,
        include_time: bool = True,
        top_n: int = -1,
        top_p: float = 0.95,
        n_gaussians: int = 0,
        log_transform: bool = False,
        n_output_buckets: int = 15,
        tsk_order: str = "1st",
        optimize_coefficients: bool = True,
        random_state: int = 42,
    ):
        """
        Initialize MIMO predictor with memory.

        Args:
            window_size: Size of short-term memory window
            memory_size: Size of long-term memory window (must be < window_size)
            include_time: Whether to include explicit time_step feature
            top_n: Number of top features for base regressor
            top_p: Per-feature score threshold for base regressor, not cumulative
                   coverage: a feature is kept when its own normalized score is
                   >= (1 - top_p). Ignored if top_n > 0.
            n_gaussians: Number of Gaussians per feature (0=automatic)
            log_transform: Whether to apply log-transform to features
            n_output_buckets: Number of output buckets for TSK
            tsk_order: TSK polynomial order ('0th', '1st', '2nd', '3rd', 'full-2nd')
            optimize_coefficients: Whether to optimize TSK coefficients
            random_state: Random seed for reproducibility
        """
        self.window_size = window_size
        self.memory_size = memory_size
        self.include_time = include_time
        self.top_n = top_n
        self.top_p = top_p
        self.n_gaussians = n_gaussians
        self.log_transform = log_transform
        self.n_output_buckets = n_output_buckets
        self.tsk_order = tsk_order
        self.optimize_coefficients = optimize_coefficients
        self.random_state = random_state

        self.feature_extractor_ = MemoryWindowFeatureExtractor(
            window_size=window_size, memory_size=memory_size
        )
        self.mimo_predictor_ = None
        self.input_features_ = None
        self.output_features_ = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, input_features=None, output_features=None):
        """
        Fit the memory-augmented MIMO model.

        Args:
            X: Input DataFrame with feature columns
            y: Output DataFrame with target columns (NOT deltas, absolute values)
            input_features: List of input feature names (if None, uses all X columns)
            output_features: List of output feature names (if None, uses all y columns)

        Returns:
            self
        """
        # Store feature names
        if input_features is None:
            input_features = X.columns.tolist()
        if output_features is None:
            output_features = y.columns.tolist() if isinstance(y, pd.DataFrame) else [f"output_{i}" for i in range(y.shape[1])]

        self.input_features_ = input_features
        self.output_features_ = output_features

        # Prepare trajectories as list (single trajectory in batch mode)
        # or handle as individual trajectories
        trajectories = [X.reset_index(drop=True)]

        # Prepare data with memory features
        X_augmented, y_deltas = prepare_mimo_data_with_memory(
            trajectories,
            input_features=self.input_features_,
            output_features=self.output_features_,
            window_size=self.window_size,
            memory_size=self.memory_size,
            include_time=self.include_time,
        )

        # Get augmented feature names for the base regressor
        aug_feature_names = self.feature_extractor_.get_feature_names(
            self.input_features_, include_time=self.include_time
        )

        # Create and fit base MIMO regressor
        self.mimo_predictor_ = MimoGaussianPredictor(
            top_n=self.top_n,
            top_p=self.top_p,
            n_gaussians=self.n_gaussians,
            log_transform=self.log_transform,
            n_output_buckets=self.n_output_buckets,
            tsk_order=self.tsk_order,
            optimize_coefficients=self.optimize_coefficients,
            random_state=self.random_state,
        )

        X_aug_df = pd.DataFrame(X_augmented, columns=aug_feature_names)
        y_delta_df = pd.DataFrame(y_deltas, columns=self.output_features_)

        self.mimo_predictor_.fit(X_aug_df, y_delta_df)

        return self

    def predict(self, X: pd.DataFrame, return_deltas: bool = False) -> pd.DataFrame:
        """
        Predict next state(s) using memory-augmented features.

        Args:
            X: Input DataFrame (should contain history for memory computation)
            return_deltas: If True, return deltas; if False, return absolute states

        Returns:
            DataFrame with predictions (deltas or absolute values depending on return_deltas)
        """
        check_is_fitted(self)

        # Prepare memory features
        X_augmented = self.feature_extractor_.prepare_sequences(
            X, self.input_features_, include_time=self.include_time
        )

        # Use last row for prediction (or handle all rows if needed)
        X_pred = X_augmented.iloc[-1:].copy()

        # Predict deltas
        y_deltas = self.mimo_predictor_.predict(X_pred)

        if return_deltas:
            return y_deltas

        # Convert deltas to absolute states by adding to last state
        last_state = X.iloc[-1:][self.output_features_].copy()
        predictions = last_state.values + y_deltas.values
        return pd.DataFrame(predictions, columns=self.output_features_)

    def predict_trajectory(self, initial_window: pd.DataFrame, n_steps: int) -> pd.DataFrame:
        """
        Predict a full trajectory using iterative rollout with memory.

        Args:
            initial_window: Initial state(s) DataFrame
            n_steps: Number of steps to predict

        Returns:
            DataFrame of shape (n_steps+1, n_features) with predicted trajectory
        """
        check_is_fitted(self)

        # Start with initial window
        trajectory = initial_window[self.output_features_].copy()

        for step in range(n_steps):
            # Get current window for memory computation
            # Use all available history, but at least window_size rows
            current_window = trajectory.iloc[-(self.window_size):].copy()

            # Get memory-augmented features
            X_mem = self.feature_extractor_.prepare_sequences(
                current_window, self.output_features_, include_time=self.include_time
            )

            # Use the last row for prediction
            if X_mem.isna().any(axis=1).iloc[-1]:
                # Skip if memory features contain NaN
                break

            X_pred = X_mem.iloc[-1:].copy()
            y_delta = self.mimo_predictor_.predict(X_pred)

            # Compute next state
            next_state = trajectory.iloc[-1:].values + y_delta.values
            trajectory = pd.concat(
                [trajectory, pd.DataFrame(next_state, columns=self.output_features_)],
                ignore_index=True,
            )

        return trajectory
