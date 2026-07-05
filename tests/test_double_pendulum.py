"""
Test double pendulum simulation with Gaussian mixture regressor prediction.

Simulates double pendulum using Lagrangian mechanics, generates datasets
with random initial conditions, trains fuzzy regressors to predict state
transitions, and evaluates prediction accuracy on continuous outputs.
Includes MIMO full-state prediction and iterative rollout with GIF animation.
"""
import unittest
import sys
import time
from argparse import ArgumentError
from collections import namedtuple
from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tests.ode_helpers import (load_and_prepare_data, train_and_evaluate_single_step, set_axes_style,
                              angles_to_xy, plot_test_vs_nearest_training)
from tests.test_fuzzy_ode import initialize_model

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tribblefis.gaussian_regressor import MimoGaussianPredictor

FeatureStep = namedtuple('FeatureStep', ['step_name', 'step_offset', 'col_name'])


class SimulationParams:
    """Simulation parameters for double pendulum."""
    dt = 0.01  # timestep in seconds


@contextmanager
def time_this(label="Operation"):
    """Simple timer context manager using perf_counter."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"{label} took {elapsed:.4f} seconds")


# Comparison set: https://arxiv.org/pdf/2504.13453
N_BINS = 3
MIMO_WINDOW_SIZE = 1
# INPUT_FEATURES = ['theta_1','theta_2', 'omega_1', 'alpha_1', 'omega_2', 'alpha_2']
INPUT_FEATURES = ['theta_1','theta_2']
OUTPUT_FEATURES = INPUT_FEATURES.copy()



def mimo_input_feature_names(window_size: int, feature_names: list[str] | None = None) -> list[FeatureStep]:
    """Return input column names for MIMO data with given window size.

    window_size=1 -> INPUT_FEATURES
    window_size=N -> ['theta_1_step0', ..., 'theta_2_step0', ..., 'theta_2_step{N-1}']
    where step0 is oldest, step(window_size-1) is most recent.

    Returns:
        List of FeatureStep namedtuples containing step_name, step_offset, and col_name.
    """
    if feature_names is None:
        feature_names = INPUT_FEATURES.copy()

    if window_size == 1:
        return [FeatureStep(step_name=feat, step_offset=0, col_name=feat) for feat in feature_names]

    return [
        FeatureStep(step_name=f"{feat}_step{i}", step_offset=i, col_name=feat)
        for i in range(window_size)
        for feat in feature_names
    ]


def load_and_prepare_mimo_data(trajectories: list[pd.DataFrame], window_size: int = 1):
    """
    Load all simulation files and prepare data for MIMO full-state prediction.

    For window_size=1: input is full state at time t, output is full state at t+1.
    For window_size>1: input is the flattened states at [t-window+1, ..., t],
                       output is the full state at t+1.
    """
    print(f"Loading {len(trajectories)} simulation files for MIMO (window={window_size})...")

    all_X = []
    all_y = []

    for df in trajectories:
        y = np.diff(df[OUTPUT_FEATURES].iloc[(window_size-1):].values, axis=0)
        if window_size == 1:
            X = df[INPUT_FEATURES].iloc[:-1].values
        else:
            X = get_mimo_df(df, window_size)
        all_X.append(X)
        all_y.append(y)

    X_combined = np.vstack(all_X)
    y_combined = np.vstack(all_y)

    print(f"MIMO combined data shape: X={X_combined.shape}, y={y_combined.shape}")

    return X_combined, y_combined


def get_mimo_df(df: pd.DataFrame, window_size: int) -> pd.DataFrame:
    feature_steps = mimo_input_feature_names(window_size)
    X = pd.DataFrame()
    for f_step in feature_steps:
        X[f_step.step_name] = df[f_step.col_name].iloc[f_step.step_offset:-(window_size - f_step.step_offset)].values
    return X


def train_and_evaluate_mimo(X_train, y_train, X_test, y_test, window_size: int = 1):
    """Train MIMO regressor for full-state prediction."""
    print("\n" + "="*60)
    print(f"MIMO FULL-STATE PREDICTION MODEL (window={window_size})")
    print("="*60)

    feature_steps = mimo_input_feature_names(window_size)
    input_cols = [fs.step_name for fs in feature_steps]
    regressor = MimoGaussianPredictor(
        n_output_buckets=N_BINS, tsk_order="1st", optimize_coefficients=True,
        random_state=42
    )

    X_train_df = pd.DataFrame(X_train, columns=input_cols)
    y_train_df = pd.DataFrame(y_train, columns=OUTPUT_FEATURES)
    X_test_df = pd.DataFrame(X_test, columns=input_cols)
    y_test_df = pd.DataFrame(y_test, columns=OUTPUT_FEATURES)

    regressor.fit(X_train_df, y_train_df)

    y_pred_df = regressor.predict(X_test_df)

    metrics = {}
    print("\nPer-output metrics:")
    for col in OUTPUT_FEATURES:
        mse = mean_squared_error(y_test_df[col], y_pred_df[col])
        mae = mean_absolute_error(y_test_df[col], y_pred_df[col])
        r2 = r2_score(y_test_df[col], y_pred_df[col])
        metrics[col] = {'mse': mse, 'mae': mae, 'rmse': np.sqrt(mse), 'r2': r2}
        print(f"  {col:10s}: R2={r2:.4f}  RMSE={np.sqrt(mse):.4f}  MAE={mae:.4f}")

    return {
        'model_type': 'mimo',
        'window_size': window_size,
        'regressor': regressor,
        'metrics': metrics,
        'y_test': y_test_df,
        'y_pred': y_pred_df,
    }


def run_iterative_prediction(
    regressor, initial_window_df: pd.DataFrame, n_steps,
    window_size: int = 1,
    verbose: bool = True,
):
    """
    Iteratively apply MIMO regressor from an initial window of states.

    For window_size=1, initial_window_df should have 1 row (the seed state).
    For window_size=N, initial_window_df should have N rows (rows oldest->newest).
    The first row of the returned trajectory corresponds to the *last* row of
    initial_window_df so that the full output aligns with the actual trajectory
    starting at index window_size-1.

    Stops early if predictions become NaN (chaotic divergence). The returned
    DataFrame is padded with NaN for any steps after divergence.

    Returns:
        DataFrame of shape (n_steps+1, n_features)
    """
    if len(initial_window_df) < window_size:
        raise ArgumentError("Initial data must be greater than or equal to window size.")

    running_state = initial_window_df.copy()
    if window_size > 1:
        running_state = get_mimo_df(initial_window_df, window_size)
    diverged_at = None

    for step in range(n_steps):
        if diverged_at:
            running_state = pd.concat([running_state, pd.DataFrame([np.full(running_state.shape[1],np.nan)], columns=running_state.columns)],
                                      ignore_index=True)
        else:
            # Predict DELTAS so we can scale to preserve energy!
            next_state_delta_df = regressor.predict(running_state[-window_size:])
            if window_size == 1:
                new_state = running_state.iloc[-1,:] + next_state_delta_df
            else:
                # Roll backwards: shift step1->step0, step2->step1, etc.
                # The new prediction becomes the most recent state (highest step index)
                feature_steps = mimo_input_feature_names(window_size)
                new_row = {}
                for fs in feature_steps:
                    if fs.step_offset < window_size - 1:
                        # Shift from next step: step1 data goes to step0, etc.
                        next_step_name = f"{fs.col_name}_step{fs.step_offset + 1}"
                        new_row[fs.step_name] = running_state.iloc[-1][next_step_name]
                    else:
                        # Most recent step gets the new prediction
                        most_recent_col = f"{fs.col_name}_step{window_size - 1}"
                        new_row[fs.step_name] = running_state.iloc[-1][most_recent_col] + next_state_delta_df.iloc[0][
                            fs.col_name]
                new_state = pd.DataFrame([new_row])

            # Set better bounds
            if np.any(np.isnan(new_state)) or np.any(np.abs(new_state) > 1e4):
                if diverged_at is None:
                    diverged_at = step + 1
                    if verbose:
                        print(f"  Warning: prediction diverged at step {diverged_at} -- padding remainder with NaN.")
            else:
                running_state = pd.concat([running_state, new_state],
                                          ignore_index=True)
    if diverged_at is not None and verbose:
        print(f"  Valid prediction steps: {diverged_at} / {n_steps + 1}")
    return running_state


def create_pendulum_animation(actual_df, predicted_df, output_path, dt=0.01, max_frames=500, fps=30):
    """
    Create a GIF animation comparing actual vs predicted double pendulum motion.

    Args:
        actual_df: DataFrame with actual trajectory (ALL_STATE_FEATURES columns)
        predicted_df: DataFrame with predicted trajectory (same columns)
        output_path: path to save the .gif file
        dt: timestep between rows
        max_frames: cap on frames to keep the GIF manageable
        fps: frames per second in the output GIF
    """
    n = min(len(actual_df), len(predicted_df), max_frames)
    step = max(1, len(actual_df) // max_frames)

    actual_frames = actual_df.iloc[::step].head(max_frames)
    pred_frames = predicted_df.iloc[::step].head(max_frames)
    n_frames = min(len(actual_frames), len(pred_frames))

    # Fill NaN in predicted frames with last valid value so the animation doesn't break
    pred_frames = pred_frames.copy().ffill().fillna(0.0)

    x1_act, y1_act, x2_act, y2_act = angles_to_xy(
        actual_frames['theta_1'].values, actual_frames['theta_2'].values, 1.0, 1.0
    )
    x1_pred, y1_pred, x2_pred, y2_pred = angles_to_xy(
        pred_frames['theta_1'].values, pred_frames['theta_2'].values, 1.0, 1.0
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.patch.set_facecolor('#1a1a2e')
    for ax in axes:
        set_axes_style(ax)

    axes[0].set_title('Actual', color='white', fontsize=13, fontweight='bold')
    axes[1].set_title('Predicted (iterative rollout)', color='white', fontsize=13, fontweight='bold')
    fig.suptitle('Double Pendulum: Actual vs Predicted', color='white', fontsize=14, fontweight='bold')

    trail_len = 40

    # Actual panel objects
    act_trail2, = axes[0].plot([], [], '-', color='#00d4ff', linewidth=1.2, alpha=0.5)
    act_rod1, = axes[0].plot([], [], 'o-', color='#e0e0e0', lw=2.5, ms=6, markerfacecolor='white')
    act_rod2, = axes[0].plot([], [], 'o-', color='#e0e0e0', lw=2.5, ms=6, markerfacecolor='#00d4ff')

    # Predicted panel objects
    pred_trail2, = axes[1].plot([], [], '-', color='#ff6b6b', linewidth=1.2, alpha=0.5)
    pred_rod1, = axes[1].plot([], [], 'o-', color='#e0e0e0', lw=2.5, ms=6, markerfacecolor='white')
    pred_rod2, = axes[1].plot([], [], 'o-', color='#e0e0e0', lw=2.5, ms=6, markerfacecolor='#ff6b6b')

    time_text = fig.text(0.5, 0.01, '', ha='center', color='#aaaaaa', fontsize=10)

    def init():
        for artist in [act_trail2, act_rod1, act_rod2, pred_trail2, pred_rod1, pred_rod2]:
            artist.set_data([], [])
        time_text.set_text('')
        return act_trail2, act_rod1, act_rod2, pred_trail2, pred_rod1, pred_rod2, time_text

    def update(frame):
        i = frame
        t_start = max(0, i - trail_len)

        act_trail2.set_data(x2_act[t_start:i+1], y2_act[t_start:i+1])
        act_rod1.set_data([0, x1_act[i]], [0, y1_act[i]])
        act_rod2.set_data([x1_act[i], x2_act[i]], [y1_act[i], y2_act[i]])

        pred_trail2.set_data(x2_pred[t_start:i+1], y2_pred[t_start:i+1])
        pred_rod1.set_data([0, x1_pred[i]], [0, y1_pred[i]])
        pred_rod2.set_data([x1_pred[i], x2_pred[i]], [y1_pred[i], y2_pred[i]])

        time_text.set_text(f't = {i * step * dt:.2f} s')
        return act_trail2, act_rod1, act_rod2, pred_trail2, pred_rod1, pred_rod2, time_text

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, init_func=init,
        interval=1000 // fps, blit=True
    )

    writer = animation.PillowWriter(fps=fps)
    ani.save(str(output_path), writer=writer)
    plt.close(fig)
    print(f"  Animation saved to: {output_path}")
    return output_path


def create_pendulum_animation_with_training(actual_df, predicted_df, train_trajectories, output_path, dt=0.01, max_frames=500, fps=30):
    """
    Create a GIF animation showing test case, prediction, and 2 nearest training trajectories.

    Args:
        actual_df: DataFrame with actual test trajectory
        predicted_df: DataFrame with predicted trajectory
        train_trajectories: list of training trajectory DataFrames
        output_path: path to save the .gif file
        dt: timestep between rows
        max_frames: cap on frames to keep the GIF manageable
        fps: frames per second in the output GIF
    """
    from tests.ode_helpers import find_nearest_trajectories

    # Find nearest training trajectories
    nearest_list = find_nearest_trajectories(actual_df, train_trajectories, k=2, features=OUTPUT_FEATURES)

    step = max(1, len(actual_df) // max_frames)
    actual_frames = actual_df.iloc[::step].head(max_frames)
    pred_frames = predicted_df.iloc[::step].head(max_frames)
    n_frames = min(len(actual_frames), len(pred_frames))

    pred_frames = pred_frames.copy().ffill().fillna(0.0)

    x1_act, y1_act, x2_act, y2_act = angles_to_xy(
        actual_frames['theta_1'].values, actual_frames['theta_2'].values, 1.0, 1.0
    )
    x1_pred, y1_pred, x2_pred, y2_pred = angles_to_xy(
        pred_frames['theta_1'].values, pred_frames['theta_2'].values, 1.0, 1.0
    )

    # Prepare nearest training trajectories
    nearest_data = []
    colors_train = ['#ffa500', '#00ff00']
    for rank, (train_idx, dist, train_traj) in enumerate(nearest_list):
        train_frames = train_traj[OUTPUT_FEATURES].iloc[::step].head(max_frames)
        x1_t, y1_t, x2_t, y2_t = angles_to_xy(
            train_frames['theta_1'].values, train_frames['theta_2'].values, 1.0, 1.0
        )
        nearest_data.append({
            'x1': x1_t, 'y1': y1_t, 'x2': x2_t, 'y2': y2_t,
            'idx': train_idx, 'dist': dist, 'color': colors_train[rank]
        })

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('#1a1a2e')
    set_axes_style(ax)

    ax.set_title('Double Pendulum: All Trajectories Overlaid', color='white', fontsize=14, fontweight='bold')
    fig.suptitle(f'Training (bg) → Test Reference → Test FIS (front)', color='white', fontsize=12)

    # Different tail lengths for each trajectory to make them visually distinct
    trail_lens = {
        'train1': 15,      # shortest (orange)
        'train2': 25,      # medium (green)
        'actual': 35,      # longer (cyan)
        'predicted': 50,   # longest (red)
    }

    artists = {
        'train1': {
            'trail2': ax.plot([], [], '-', color=colors_train[0], linewidth=1.0, alpha=0.3)[0],
            'rod1': ax.plot([], [], 'o-', color='#e0e0e0', lw=2, ms=5, markerfacecolor='white', alpha=0.3)[0],
            'rod2': ax.plot([], [], 'o-', color='#e0e0e0', lw=2, ms=5, markerfacecolor=colors_train[0], alpha=0.3)[0],
        },
        'train2': {
            'trail2': ax.plot([], [], '-', color=colors_train[1], linewidth=1.0, alpha=0.3)[0],
            'rod1': ax.plot([], [], 'o-', color='#e0e0e0', lw=2, ms=5, markerfacecolor='white', alpha=0.3)[0],
            'rod2': ax.plot([], [], 'o-', color='#e0e0e0', lw=2, ms=5, markerfacecolor=colors_train[1], alpha=0.3)[0],
        },
        'actual': {
            'trail2': ax.plot([], [], '-', color='#00d4ff', linewidth=1.5, alpha=0.6)[0],
            'rod1': ax.plot([], [], 'o-', color='#e0e0e0', lw=2.5, ms=6, markerfacecolor='white', alpha=0.8)[0],
            'rod2': ax.plot([], [], 'o-', color='#e0e0e0', lw=2.5, ms=6, markerfacecolor='#00d4ff')[0],
        },
        'predicted': {
            'trail2': ax.plot([], [], '-', color='#ff6b6b', linewidth=1.5, alpha=0.8)[0],
            'rod1': ax.plot([], [], 'o-', color='#e0e0e0', lw=2.5, ms=6, markerfacecolor='white')[0],
            'rod2': ax.plot([], [], 'o-', color='#e0e0e0', lw=2.5, ms=6, markerfacecolor='#ff6b6b')[0],
        },
    }

    time_text = fig.text(0.5, 0.01, '', ha='center', color='#aaaaaa', fontsize=11)
    legend_text = fig.text(0.5, 0.05, 'Train1 (orange, tail=15) · Train2 (green, tail=25) · Test Ref (cyan, tail=35) · Test FIS (red, tail=50)',
                           ha='center', color='#aaaaaa', fontsize=9)

    def init():
        for key in artists:
            for artist in artists[key].values():
                artist.set_data([], [])
        time_text.set_text('')
        result = []
        for key in artists:
            for a in artists[key].values():
                result.append(a)
        result.extend([time_text, legend_text])
        return result

    def update(frame):
        i = frame

        # Training 1 (background) - shortest tail
        t_start = max(0, i - trail_lens['train1'])
        if i < len(nearest_data[0]['x2']):
            artists['train1']['trail2'].set_data(nearest_data[0]['x2'][t_start:i+1], nearest_data[0]['y2'][t_start:i+1])
            artists['train1']['rod1'].set_data([0, nearest_data[0]['x1'][i]], [0, nearest_data[0]['y1'][i]])
            artists['train1']['rod2'].set_data([nearest_data[0]['x1'][i], nearest_data[0]['x2'][i]], [nearest_data[0]['y1'][i], nearest_data[0]['y2'][i]])

        # Training 2 (background) - medium tail
        t_start = max(0, i - trail_lens['train2'])
        if i < len(nearest_data[1]['x2']):
            artists['train2']['trail2'].set_data(nearest_data[1]['x2'][t_start:i+1], nearest_data[1]['y2'][t_start:i+1])
            artists['train2']['rod1'].set_data([0, nearest_data[1]['x1'][i]], [0, nearest_data[1]['y1'][i]])
            artists['train2']['rod2'].set_data([nearest_data[1]['x1'][i], nearest_data[1]['x2'][i]], [nearest_data[1]['y1'][i], nearest_data[1]['y2'][i]])

        # Actual (test reference) - longer tail
        t_start = max(0, i - trail_lens['actual'])
        artists['actual']['trail2'].set_data(x2_act[t_start:i+1], y2_act[t_start:i+1])
        artists['actual']['rod1'].set_data([0, x1_act[i]], [0, y1_act[i]])
        artists['actual']['rod2'].set_data([x1_act[i], x2_act[i]], [y1_act[i], y2_act[i]])

        # Predicted (front) - longest tail
        t_start = max(0, i - trail_lens['predicted'])
        artists['predicted']['trail2'].set_data(x2_pred[t_start:i+1], y2_pred[t_start:i+1])
        artists['predicted']['rod1'].set_data([0, x1_pred[i]], [0, y1_pred[i]])
        artists['predicted']['rod2'].set_data([x1_pred[i], x2_pred[i]], [y1_pred[i], y2_pred[i]])

        time_text.set_text(f't = {i * step * dt:.2f} s')

        result = []
        for key in artists:
            for a in artists[key].values():
                result.append(a)
        result.extend([time_text, legend_text])
        return result

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, init_func=init,
        interval=1000 // fps, blit=True
    )

    writer = animation.PillowWriter(fps=fps)
    ani.save(str(output_path), writer=writer)
    plt.close(fig)
    print(f"  Animation saved to: {output_path}")
    return output_path


def plot_prediction_comparison(results_single, results_window):
    """Plot comparison of predicted vs actual values for both models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Double Pendulum Prediction Comparison', fontsize=16, fontweight='bold')

    ax = axes[0, 0]
    y_test = results_single['y_test']
    y_pred = results_single['y_pred']
    ax.scatter(y_test, y_pred, alpha=0.5, s=20, edgecolors='k', linewidth=0.3)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel(f'Actual {OUTPUT_FEATURES[0]} (rad)', fontsize=11)
    ax.set_ylabel(f'Predicted {OUTPUT_FEATURES[0]} (rad)', fontsize=11)
    ax.set_title(f'Single-Step: Actual vs Predicted\nR²={results_single["r2"]:.4f}, RMSE={results_single["rmse"]:.4f}', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    ax = axes[0, 1]
    residuals_single = y_test - y_pred
    ax.scatter(y_pred, residuals_single, alpha=0.5, s=20, edgecolors='k', linewidth=0.3)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel(f'Predicted {OUTPUT_FEATURES[0]} (rad)', fontsize=11)
    ax.set_ylabel('Residual (Actual - Predicted)', fontsize=11)
    ax.set_title('Single-Step: Residual Plot', fontsize=11)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    y_test = results_window['y_test']
    y_pred = results_window['y_pred']
    ax.scatter(y_test, y_pred, alpha=0.5, s=20, edgecolors='k', linewidth=0.3, color='green')
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel(f'Actual {OUTPUT_FEATURES[0]} (rad)', fontsize=11)
    ax.set_ylabel(f'Predicted {OUTPUT_FEATURES[0]} (rad)', fontsize=11)
    ax.set_title(f'Multi-Step (window=3): Actual vs Predicted\nR²={results_window["r2"]:.4f}, RMSE={results_window["rmse"]:.4f}', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    ax = axes[1, 1]
    residuals_window = y_test - y_pred
    ax.scatter(y_pred, residuals_window, alpha=0.5, s=20, edgecolors='k', linewidth=0.3, color='green')
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel(f'Predicted {OUTPUT_FEATURES[0]} (rad)', fontsize=11)
    ax.set_ylabel(f'Residual (Actual - Predicted)', fontsize=11)
    ax.set_title('Multi-Step: Residual Plot', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_mimo_state_trajectories(actual_df, predicted_df, dt=0.01):
    """Plot OUTPUT_FEATURES: actual vs predicted rollout."""
    n_out = len(OUTPUT_FEATURES)
    ncols = min(n_out, 2)
    nrows = (n_out + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flat
    fig.suptitle('MIMO Iterative Rollout: Actual vs Predicted', fontsize=14, fontweight='bold')

    t_actual = np.arange(len(actual_df)) * dt
    pred_time = np.arange(len(predicted_df)) * dt
    n = min(len(actual_df), len(predicted_df))

    for idx, col in enumerate(OUTPUT_FEATURES):
        ax = axes_flat[idx]
        act = actual_df[col].values[:n]
        pred = predicted_df[col].values[:n]
        valid_pred = ~np.isnan(pred)

        ax.plot(t_actual[:n], act, 'b-', linewidth=1.5, label='Actual', alpha=0.9)
        ax.plot(pred_time[:n][valid_pred], pred[valid_pred], 'r--', linewidth=1.2,
                label='Predicted', alpha=0.7)

        if valid_pred.any():
            mae = np.mean(np.abs(act[valid_pred] - pred[valid_pred]))
            title_str = f'{col}  (MAE={mae:.4f}, valid={valid_pred.sum()}/{n})'
        else:
            title_str = f'{col}  (no valid predictions)'

        ax.set_title(title_str, fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    return fig


class TestDoublePendulumFuzzyPrediction(unittest.TestCase):
    """Integration test for double pendulum fuzzy regression with MIMO."""

    def test_double_pendulum_fuzzy_prediction(self):
        """
        Main test: simulate double pendulum and train fuzzy regression models.
        Includes MIMO full-state prediction and iterative rollout with GIF animation.
        """

        # Step 1: Generate simulation data
        with time_this("gen-sim-data"):
            # 1) Create the simulation data for various initial conditions.
            train_results, test_results = initialize_model()

        # Step 2: Single-step prediction (theta_2 only)
        print("\n" + "#"*60)
        print("# STEP 2: Single-Step Prediction Model")
        print("#"*60)
        with time_this('train-single'):
            X_single_train, y_single_train = load_and_prepare_data(train_results.trajectories,INPUT_FEATURES, OUTPUT_FEATURES, window_size=1)
            X_single_test, y_single_test = load_and_prepare_data(test_results.trajectories,INPUT_FEATURES, OUTPUT_FEATURES, window_size=1)
            results_single = train_and_evaluate_single_step(N_BINS,OUTPUT_FEATURES, X_single_train, y_single_train, X_single_test, y_single_test)

        # Step 4: MIMO full-state prediction (window=1)
        print("\n" + "#"*60)
        print(f"# STEP 4: MIMO Full-State Prediction Model (window={MIMO_WINDOW_SIZE})")
        print("#"*60)
        with time_this('train-mimo-window1'):
            X_mimo_train, y_mimo_train = load_and_prepare_mimo_data(train_results.trajectories, window_size=MIMO_WINDOW_SIZE)
            X_mimo_test, y_mimo_test = load_and_prepare_mimo_data(test_results.trajectories, window_size=MIMO_WINDOW_SIZE)
            results_mimo = train_and_evaluate_mimo(
                X_mimo_train, y_mimo_train, X_mimo_test, y_mimo_test, window_size=MIMO_WINDOW_SIZE
            )

        # Step 4b: MIMO with window_size=3 for temporal context
        print("\n" + "#"*60)
        print("# STEP 4b: MIMO Full-State Prediction Model (window=3)")
        print("#"*60)
        with time_this('train-mimo-window3'):
            X_mimo_train_w3, y_mimo_train_w3 = load_and_prepare_mimo_data(train_results.trajectories, window_size=3)
            X_mimo_test_w3, y_mimo_test_w3 = load_and_prepare_mimo_data(test_results.trajectories, window_size=3)
            results_mimo_w3 = train_and_evaluate_mimo(
                X_mimo_train_w3, y_mimo_train_w3, X_mimo_test_w3, y_mimo_test_w3, window_size=3
            )

        # Step 4c: Multi-window stability comparison
        print("\n" + "#"*60)
        print("# STEP 4c: Multi-Window Stability Comparison")
        print("#"*60)
        print("Training MIMO models with different memory windows to assess stability...")

        window_sizes = [1, 3, 5, 7, 10]
        mimo_results_by_window = {}

        for ws in window_sizes:
            print(f"\n  Training window_size={ws}...")
            with time_this(f'train-mimo-window{ws}'):
                X_train_ws, y_train_ws = load_and_prepare_mimo_data(train_results.trajectories, window_size=ws)
                X_test_ws, y_test_ws = load_and_prepare_mimo_data(test_results.trajectories, window_size=ws)
                results_ws = train_and_evaluate_mimo(
                    X_train_ws, y_train_ws, X_test_ws, y_test_ws, window_size=ws
                )
                mimo_results_by_window[ws] = results_ws

        # Step 5: Iterative rollout from initial conditions with all window sizes
        print("\n" + "#"*60)
        print("# STEP 5: Iterative Rollout from Initial Conditions (Multiple Window Sizes)")
        print("#"*60)
        # Run iterative predictions for all window sizes
        tst_df = test_results.trajectories[0]
        actual_trajectory_base = tst_df[OUTPUT_FEATURES].reset_index(drop=True)

        predicted_trajectories_by_window = {}
        stability_summary = []

        for ws in window_sizes:
            n_steps = len(tst_df) - ws
            print(f"\nWindow size {ws}: running {n_steps} iterative prediction steps...")

            with time_this(f'iterative-rollout-window{ws}'):
                predicted_traj = run_iterative_prediction(
                    mimo_results_by_window[ws]['regressor'], tst_df, n_steps, window_size=ws, verbose=False
                )
                predicted_trajectories_by_window[ws] = predicted_traj

            # Find divergence point (where predictions become NaN)
            # For window_size > 1, extract the most recent state columns
            if ws == 1:
                check_col = 'theta_1'
            else:
                check_col = f'theta_1_step{ws - 1}'

            if check_col in predicted_traj.columns:
                valid_mask = ~np.isnan(predicted_traj[check_col].values)
                if valid_mask.any():
                    divergence_idx = np.where(~valid_mask)[0]
                    if len(divergence_idx) > 0:
                        divergence_idx = divergence_idx[0]
                    else:
                        divergence_idx = len(predicted_traj)
                else:
                    divergence_idx = 0
            else:
                divergence_idx = len(predicted_traj)

            divergence_time = divergence_idx * SimulationParams.dt
            stability_summary.append({
                'window_size': ws,
                'divergence_step': divergence_idx,
                'divergence_time_s': divergence_time,
                'trajectory_length': len(predicted_traj)
            })

            print(f"  Window {ws}: Diverged at step {divergence_idx} ({divergence_time:.2f}s) / {len(tst_df)} total steps")

        # Use default window size for detailed analysis
        predicted_trajectory = predicted_trajectories_by_window[MIMO_WINDOW_SIZE]
        actual_trajectory = tst_df[OUTPUT_FEATURES].iloc[MIMO_WINDOW_SIZE - 1:].reset_index(drop=True)
        n = min(len(actual_trajectory), len(predicted_trajectory))

        def rollout_metrics(label, predicted, actual, n):
            print(f"\n{label}:")
            for col in OUTPUT_FEATURES:
                act = actual[col].values[:n]
                pred = predicted[col].values[:n]
                valid = ~np.isnan(pred)
                if valid.sum() < 2:
                    print(f"  {col:10s}: insufficient valid predictions")
                    continue
                mae = np.mean(np.abs(act[valid] - pred[valid]))
                r2 = r2_score(act[valid], pred[valid])
                print(f"  {col:10s}: MAE={mae:.4f}  R2={r2:.4f}  (valid={valid.sum()}/{n})")

        rollout_metrics("Iterative Rollout Metrics (window=1)", predicted_trajectory, actual_trajectory, n)

        # Step 5b: Stability comparison summary
        print("\n" + "="*70)
        print("STABILITY COMPARISON: DIVERGENCE ANALYSIS BY MEMORY WINDOW")
        print("="*70)
        print(f"\n{'Window Size':<15} {'Divergence Step':<20} {'Divergence Time (s)':<25} {'Total Duration (s)':<20}")
        print("-" * 80)
        for summary in stability_summary:
            print(f"{summary['window_size']:<15} {summary['divergence_step']:<20} {summary['divergence_time_s']:<25.2f} {summary['trajectory_length']*SimulationParams.dt:<20.2f}")

        best_window = max(stability_summary, key=lambda x: x['divergence_time_s'])
        print("\n" + "="*70)
        print(f"Best Stability: Window size {best_window['window_size']} stays stable for {best_window['divergence_time_s']:.2f}s")
        print("="*70)

        # Step 6: Summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print("\nSingle-Step Model:")
        print(f"  R2:   {results_single['r2']:.6f}")
        print(f"  RMSE: {results_single['rmse']:.6f}")
        print(f"  MAE:  {results_single['mae']:.6f}")

        print("\nMIMO Model (window=1):")
        for col in OUTPUT_FEATURES:
            m = results_mimo['metrics'][col]
            print(f"  {col:10s}: R2={m['r2']:.4f}  RMSE={m['rmse']:.4f}")

        print("\nMIMO Model (window=3):")
        for col in OUTPUT_FEATURES:
            m = results_mimo_w3['metrics'][col]
            print(f"  {col:10s}: R2={m['r2']:.4f}  RMSE={m['rmse']:.4f}")

        # Step 7: Plots
        print("\n" + "="*60)
        print("GENERATING VISUALIZATION PLOTS")
        print("="*60)

        print("\nPlot 3: MIMO Iterative Rollout State Trajectories")
        fig3 = plot_mimo_state_trajectories(actual_trajectory, predicted_trajectory, dt=SimulationParams.dt)
        fig3.savefig("mimo_iterative_trajectories.png", dpi=200, bbox_inches='tight')
        plt.close(fig3)

        # Step 8: GIF animations
        print("\n" + "="*60)
        print("GENERATING GIF ANIMATIONS")
        print("="*60)
        gif_path = "double_pendulum_comparison.gif"
        create_pendulum_animation(
            actual_trajectory, predicted_trajectory,
        gif_path, dt=SimulationParams.dt, max_frames=300, fps=25
        )

        gif_path_with_training = "double_pendulum_with_training.gif"
        with time_this("create-animation-with-training"):
            create_pendulum_animation_with_training(
                actual_trajectory, predicted_trajectory, train_results.trajectories,
                gif_path_with_training, dt=SimulationParams.dt, max_frames=300, fps=25
            )

        # Step 9: Plot test vs nearest training trajectories to visualize stability
        print("\n" + "="*60)
        print("TEST VS NEAREST TRAINING TRAJECTORIES")
        print("="*60)
        fig_nearest = plot_test_vs_nearest_training(
            actual_trajectory, train_results.trajectories, dt=SimulationParams.dt,
            features=OUTPUT_FEATURES, k=2
        )
        fig_nearest.savefig("test_vs_nearest_training.png", dpi=200, bbox_inches='tight')
        plt.close(fig_nearest)
        print("  Saved to: test_vs_nearest_training.png")

        print("\nTest completed successfully!")

        # Basic assertions to verify models trained successfully
        self.assertIsNotNone(results_single['regressor'])
        self.assertIsNotNone(results_mimo['regressor'])
        self.assertIsNotNone(results_mimo_w3['regressor'])
        for ws in window_sizes:
            self.assertIn(ws, mimo_results_by_window)
            self.assertIsNotNone(mimo_results_by_window[ws]['regressor'])
        self.assertGreater(len(actual_trajectory), 0)
        self.assertGreater(len(predicted_trajectory), 0)
        self.assertGreater(len(stability_summary), 0)


if __name__ == '__main__':
    unittest.main()
