import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tribblefis.gaussian_regressor import MixtureOfGaussiansFuzzyRegressor

def set_axes_style(ax: Axes):
    ax.set_facecolor('#16213e')
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')


def angles_to_xy(theta1, theta2, l1, l2):
    """Convert angles to Cartesian coordinates."""
    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)
    x2 = x1 + l2 * np.sin(theta2)
    y2 = y1 - l2 * np.cos(theta2)
    return x1, y1, x2, y2


def load_and_prepare_data(trajectories: list[pd.DataFrame], input_features: list[str], output_features:list[str], window_size=1,file_glob: str = 'simulation_0*.csv'):
    """
    Load all simulation files and prepare data for prediction.

    For window_size=1: features are current state, target is next state.
    For window_size>1: features are last n timesteps, target is next state.

    Args:
        trajectories: list of simulation trajectories
        input_features: list of input features to use
        output_features: list of output features to use
        window_size: number of past timesteps to use as features
        file_glob: For picking the existing simulation data.

    Returns:
        tuple: (X, y) where X is features and y is target
    """
    print(f"Loading {len(trajectories)} simulation files...")

    all_X = []
    all_y = []

    for df in trajectories:
        if window_size == 1:
            # Single timestep: current state -> next state
            X = df[input_features].iloc[:-1].values
            y = df[output_features].iloc[1:].values
        else:
            # Multi-step window
            X = []
            y = []
            for j in range(len(df) - window_size):
                # Take last window_size timesteps as features
                window = df[input_features].iloc[j:j+window_size].values.flatten()
                X.append(window)
                # Next timestep as target
                y.append(df[output_features].iloc[j+window_size].values)

            if X:
                X = np.array(X)
                y = np.array(y)

        all_X.append(X)
        all_y.append(y)

    X_combined = np.vstack(all_X)
    y_combined = np.vstack(all_y)

    print(f"Combined data shape: X={X_combined.shape}, y={y_combined.shape}")

    return X_combined, y_combined


def train_and_evaluate_single_step(n_bins:int, output_features: list[str], X_train, y_train, X_test, y_test):
    """
    Train regressor for single-step prediction (current state -> next state).

    Args:
        n_bins: number of bins for output features
        output_features: Names of features to output
        X: features (current state)
        y: targets (next state, continuous)
        test_size: fraction of data for testing

    Returns:
        dict with evaluation metrics
    """
    print("\n" + "="*60)
    print("SINGLE-STEP PREDICTION MODEL")
    print("="*60)

    # Train regressor on continuous target (first output feature)
    y_train_scalar = y_train[:, 0] if y_train.ndim > 1 else y_train
    y_test_scalar = y_test[:, 0] if y_test.ndim > 1 else y_test

    regressor = MixtureOfGaussiansFuzzyRegressor(
        n_output_buckets=n_bins, tsk_order="2nd", optimize_coefficients=True,
        random_state=42
    )
    regressor.fit(X_train, y_train_scalar)

    # Predict continuous values
    y_pred = regressor.predict(X_test)

    # Calculate regression metrics
    mse = mean_squared_error(y_test_scalar, y_pred)
    mae = mean_absolute_error(y_test_scalar, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_scalar, y_pred)

    print(f"\n{output_features[0]}:")
    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  R²:   {r2:.6f}")

    return {
        'model_type': 'single_step',
        'regressor': regressor,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'n_test_samples': len(X_test),
        'y_test': y_test_scalar,
        'y_pred': y_pred,
    }


def train_and_evaluate_window(n_bins:int, output_features: list[str], X_train, y_train, X_test, y_test, window_size=3, test_size=0.2):
    """
    Train regressor for multi-step prediction using sliding window.

    Args:
        n_bins: number of bins for output features
        output_features: list of output features
        X: features (windowed state history)
        y: targets (next state, continuous)
        window_size: number of past timesteps used
        test_size: fraction of data for testing

    Returns:
        dict with evaluation metrics
    """
    print("\n" + "="*60)
    print(f"MULTI-STEP WINDOW PREDICTION MODEL (window_size={window_size})")
    print("="*60)

    y_train_scalar = y_train[:, 0] if y_train.ndim > 1 else y_train
    y_test_scalar = y_test[:, 0] if y_test.ndim > 1 else y_test

    # Train regressor
    regressor = MixtureOfGaussiansFuzzyRegressor(
        n_output_buckets=n_bins, tsk_order="1st", optimize_coefficients=True,
        random_state=42
    )
    regressor.fit(X_train, y_train_scalar)

    # Predict continuous values
    y_pred = regressor.predict(X_test)

    # Calculate regression metrics
    mse = mean_squared_error(y_test_scalar, y_pred)
    mae = mean_absolute_error(y_test_scalar, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_scalar, y_pred)

    print(f"\n{output_features[0]}:")
    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  R²:   {r2:.6f}")

    return {
        'model_type': 'multi_window',
        'window_size': window_size,
        'regressor': regressor,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'n_test_samples': len(X_test),
        'y_test': y_test_scalar,
        'y_pred': y_pred,
    }


def plot_prediction_comparison(output_features: list[str], results_single, results_window):
    """
    Plot comparison of predicted vs actual values for both models.
    Returns the figure object.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Double Pendulum Prediction Comparison', fontsize=16, fontweight='bold')

    # Single-step actual vs predicted scatter
    ax = axes[0, 0]
    y_test = results_single['y_test']
    y_pred = results_single['y_pred']
    ax.scatter(y_test, y_pred, alpha=0.5, s=20, edgecolors='k', linewidth=0.3)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel(f'Actual {output_features[0]} (rad)', fontsize=11)
    ax.set_ylabel(f'Predicted {output_features[0]} (rad)', fontsize=11)
    ax.set_title(f'Single-Step: Actual vs Predicted\nR²={results_single["r2"]:.4f}, RMSE={results_single["rmse"]:.4f}', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Single-step residuals
    ax = axes[0, 1]
    residuals_single = y_test - y_pred
    ax.scatter(y_pred, residuals_single, alpha=0.5, s=20, edgecolors='k', linewidth=0.3)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel(f'Predicted {output_features[0]} (rad)', fontsize=11)
    ax.set_ylabel('Residual (Actual - Predicted)', fontsize=11)
    ax.set_title('Single-Step: Residual Plot', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Multi-step actual vs predicted scatter
    ax = axes[1, 0]
    y_test = results_window['y_test']
    y_pred = results_window['y_pred']
    ax.scatter(y_test, y_pred, alpha=0.5, s=20, edgecolors='k', linewidth=0.3, color='green')
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel(f'Actual {output_features[0]} (rad)', fontsize=11)
    ax.set_ylabel(f'Predicted {output_features[0]} (rad)', fontsize=11)
    ax.set_title(f'Multi-Step (window=3): Actual vs Predicted\nR²={results_window["r2"]:.4f}, RMSE={results_window["rmse"]:.4f}', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Multi-step residuals
    ax = axes[1, 1]
    residuals_window = y_test - y_pred
    ax.scatter(y_pred, residuals_window, alpha=0.5, s=20, edgecolors='k', linewidth=0.3, color='green')
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel(f'Predicted {output_features[0]} (rad)', fontsize=11)
    ax.set_ylabel(f'Residual (Actual - Predicted)', fontsize=11)
    ax.set_title('Multi-Step: Residual Plot', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_second_pendulum_position(output_features: list[str], results_single, results_window, dt=0.01):
    """
    Plot the actual and predicted position of the second pendulum ({OUTPUT_FEATURES[0]}) as a function of time.
    Shows detailed comparison between actual and predicted trajectories.
    Returns the figure object.
    """
    fig, axes = plt.subplots(4, 1, figsize=(10, 15))
    fig.suptitle(f'{output_features[0]} Over Time', fontsize=16, fontweight='bold')

    # Convert sample indices to time (assuming dt = 0.01 seconds between samples)
    def sample_to_time(indices, dt):
        return indices * dt

    # Single-step model - full trace
    ax = axes[0]
    y_test = results_single['y_test']
    y_pred = results_single['y_pred']
    time_indices = sample_to_time(np.arange(len(y_test)), dt)

    ax.plot(time_indices, y_test, 'b-', linewidth=2, label='Actual', alpha=0.8)
    ax.plot(time_indices, y_pred, 'r--', linewidth=1.5, label='Predicted', alpha=0.8)
    ax.fill_between(time_indices, y_test, y_pred, alpha=0.1, color='gray', label='Error')
    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_ylabel(f'{output_features[0]} (radians)', fontsize=11)
    ax.set_title(f'Single-Step Model: {output_features[0]} Position Over Time (R²={results_single["r2"]:.4f}, MAE={results_single["mae"]:.4f})', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Multi-step model - full trace
    ax = axes[1]
    y_test = results_window['y_test']
    y_pred = results_window['y_pred']
    time_indices = sample_to_time(np.arange(len(y_test)), dt)

    ax.plot(time_indices, y_test, 'b-', linewidth=2, label='Actual', alpha=0.8)
    ax.plot(time_indices, y_pred, 'g--', linewidth=1.5, label='Predicted', alpha=0.8)
    ax.fill_between(time_indices, y_test, y_pred, alpha=0.1, color='gray', label='Error')
    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_ylabel(f'{output_features[0]} (radians)', fontsize=11)
    ax.set_title(f'Multi-Step Window Model: {output_features[0]} Position Over Time (R²={results_window["r2"]:.4f}, MAE={results_window["mae"]:.4f})', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Error over time
    ax = axes[2]
    y_test_single = results_single['y_test']
    y_pred_single = results_single['y_pred']
    y_test_window = results_window['y_test']
    y_pred_window = results_window['y_pred']

    error_single = np.abs(y_test_single - y_pred_single)
    error_window = np.abs(y_test_window - y_pred_window)

    time_single = sample_to_time(np.arange(len(error_single)), dt)
    time_window = sample_to_time(np.arange(len(error_window)), dt)

    ax.plot(time_single, error_single, 'r-', linewidth=1, label='Single-Step Error', alpha=0.7)
    ax.plot(time_window, error_window, 'g-', linewidth=1, label='Multi-Step Error', alpha=0.7)
    ax.axhline(y=np.mean(error_single), color='r', linestyle=':', linewidth=2, label=f'Single-Step Mean Error: {np.mean(error_single):.4f}')
    ax.axhline(y=np.mean(error_window), color='g', linestyle=':', linewidth=2, label=f'Multi-Step Mean Error: {np.mean(error_window):.4f}')
    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_ylabel('Absolute Error (radians)', fontsize=11)
    ax.set_title('Prediction Error Over Time', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Actual compared positions.
    ax = axes[3]
    y_test_single = results_single['y_test']
    y_pred_single = results_single['y_pred']
    y_test_window = results_window['y_test']
    y_pred_window = results_window['y_pred']

    ax.plot(y_test_single, y_pred_single, 'r-', linewidth=1, label='Single-Step predictions', alpha=0.7)
    ax.plot(y_test_window, y_pred_window, 'g-', linewidth=1, label='Multi-Step predictions', alpha=0.7)
    ax.set_xlabel('Angle', fontsize=11)
    ax.set_ylabel('Angle', fontsize=11)
    ax.set_title('Phasing plot', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def find_nearest_trajectories(test_trajectory: pd.DataFrame, train_trajectories: list[pd.DataFrame],
                               k: int = 2, features: list[str] | None = None) -> list[tuple[int, float, pd.DataFrame]]:
    """
    Find the k nearest training trajectories to the test trajectory based on Euclidean distance.

    Uses mean Euclidean distance across the specified features. Useful for analyzing whether
    the FIS diverges chaotically or remains stable near the training data manifold.

    Args:
        test_trajectory: DataFrame with test trajectory
        train_trajectories: list of DataFrames with training trajectories
        k: number of nearest neighbors to return
        features: list of feature columns to use for distance calculation; if None, uses all columns

    Returns:
        List of tuples (train_idx, distance, train_trajectory) sorted by distance (nearest first)
    """
    if features is None:
        features = test_trajectory.columns.tolist()

    test_data = test_trajectory[features].values

    distances = []
    for idx, train_traj in enumerate(train_trajectories):
        train_data = train_traj[features].values
        # Align lengths by truncating to minimum
        min_len = min(len(test_data), len(train_data))
        test_trunc = test_data[:min_len]
        train_trunc = train_data[:min_len]
        # Mean Euclidean distance per timestep
        dist = np.mean(np.linalg.norm(test_trunc - train_trunc, axis=1))
        distances.append((idx, dist, train_traj))

    # Sort by distance and return top k
    distances.sort(key=lambda x: x[1])
    return distances[:k]


def plot_test_vs_nearest_training(test_trajectory: pd.DataFrame, train_trajectories: list[pd.DataFrame],
                                   dt: float = 0.01, features: list[str] | None = None, k: int = 2) -> plt.Figure:
    """
    Plot the test trajectory alongside the k nearest training trajectories to visualize stability.

    Helps determine if the FIS prediction diverges chaotically or remains near the training
    data manifold. Shows individual feature timeseries with test (blue) and nearest training
    (orange and green) overlaid.

    Args:
        test_trajectory: DataFrame with test trajectory (actual)
        train_trajectories: list of DataFrames with training trajectories
        dt: timestep in seconds
        features: list of features to plot; if None, uses all columns in test_trajectory
        k: number of nearest training trajectories to show

    Returns:
        matplotlib Figure object
    """
    if features is None:
        features = test_trajectory.columns.tolist()

    # Find nearest trajectories
    nearest = find_nearest_trajectories(test_trajectory, train_trajectories, k=k, features=features)

    # Create subplots
    ncols = min(len(features), 2)
    nrows = (len(features) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    fig.suptitle(f'Test vs {k} Nearest Training Trajectories', fontsize=14, fontweight='bold')

    t_test = np.arange(len(test_trajectory)) * dt

    colors = ['orange', 'green', 'purple', 'red', 'brown']  # colors for nearest training trajectories

    for feat_idx, feat in enumerate(features):
        ax = axes_flat[feat_idx]

        # Plot test trajectory
        test_values = test_trajectory[feat].values
        ax.plot(t_test, test_values, 'b-', linewidth=2, label='Test (reference)', alpha=0.9, zorder=10)

        # Plot nearest training trajectories
        for rank, (train_idx, distance, train_traj) in enumerate(nearest):
            train_values = train_traj[feat].values
            t_train = np.arange(len(train_traj)) * dt

            color = colors[rank % len(colors)]
            label = f'Train {train_idx} (d={distance:.4f})'
            ax.plot(t_train, train_values, color=color, linestyle='--', linewidth=1.5,
                   label=label, alpha=0.7)

        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel(feat, fontsize=10)
        ax.set_title(f'{feat} Over Time', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)

    # Hide unused subplots
    for idx in range(len(features), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    return fig