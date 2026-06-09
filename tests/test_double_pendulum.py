"""
Test double pendulum simulation with Gaussian mixture regressor prediction.

Simulates double pendulum using Lagrangian mechanics, generates datasets
with random initial conditions, trains fuzzy regressors to predict state
transitions, and evaluates prediction accuracy on continuous outputs.
Includes MIMO full-state prediction and iterative rollout with GIF animation.
"""
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import odeint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import sys

import tribblefis.gauss_data

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tribblefis.gaussian_regressor import MixtureOfGaussiansFuzzyRegressor, MimoGaussianPredictor


@contextmanager
def time_this(label="Operation"):
    """Simple timer context manager using perf_counter."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"{label} took {elapsed:.4f} seconds")


# Comparison set: https://arxiv.org/pdf/2504.13453
N_BINS = 2
MIMO_WINDOW_SIZE = 2
INPUT_FEATURES = ['theta_1','theta_2', 'omega_1', 'alpha_1', 'omega_2', 'alpha_2']
# OUTPUT_FEATURES = INPUT_FEATURES.copy()
OUTPUT_FEATURES = ['theta_1', 'theta_2']

# tribblefis.gauss_data.DefaultNormCornorm = 'probability'

class DoublePendulum:
    """Double pendulum simulator using Lagrangian mechanics."""

    def __init__(self, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81):
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.g = g

    def equations_of_motion(self, state, t):
        """
        Compute double pendulum equations of motion using Lagrangian approach.

        State: [theta_1, omega_1, theta_2, omega_2]
        Returns: [omega_1, alpha_1, omega_2, alpha_2]
        """
        theta1, omega1, theta2, omega2 = state
        # Found here: https://web.mit.edu/jorloff/www/chaosTalk/double-pendulum/double-pendulum-en.html
        delta_theta = theta1 - theta2

        # Common terms
        denom1 = self.l1 *(2*self.m1 + self.m2 - self.m2 *np.cos(2*delta_theta))
        num11 = -self.g*(2*self.m1 + self.m2)*np.sin(theta1)
        num12 = -self.m2*self.g*np.sin(delta_theta - theta2) # theta1-2theta2
        num13 = -2*np.sin(delta_theta)*self.m2*(omega2**2 * self.l2 + omega1**2 *self.l1 * np.cos(delta_theta))
        alpha1 = (num11 + num12 + num13) / denom1

        num21 = omega1**2 *self.l1 *(self.m1+self.m2)
        num22 = self.g*(self.m1+self.m2)*np.cos(theta1)
        num23 = omega2**2 + self.l2*self.m2 * np.cos(delta_theta)
        denom2 = self.l2 *(2*self.m1 + self.m2 - self.m2 * np.cos(2*delta_theta))
        alpha2 = 2*np.sin(delta_theta)*(num21 + num22 + num23) / denom2

        return [omega1, alpha1, omega2, alpha2]

    def simulate(self, theta1_0, omega1_0, theta2_0, omega2_0, duration=10.0, dt=0.001):
        """
        Simulate double pendulum from initial conditions.

        Returns:
            DataFrame with columns: theta_1, omega_1, alpha_1, theta_2, omega_2, alpha_2
        """
        t = np.arange(0, duration, dt)
        state0 = [theta1_0, omega1_0, theta2_0, omega2_0]

        solution = odeint(self.equations_of_motion, state0, t)

        theta1 = solution[:, 0]
        omega1 = solution[:, 1]
        theta2 = solution[:, 2]
        omega2 = solution[:, 3]

        alpha1_vals = []
        alpha2_vals = []
        for i, state in enumerate(solution):
            _, a1, _, a2 = self.equations_of_motion(state, t[i])
            alpha1_vals.append(a1)
            alpha2_vals.append(a2)

        return pd.DataFrame({
            'theta_1': theta1,
            'omega_1': omega1,
            'alpha_1': alpha1_vals,
            'theta_2': theta2,
            'omega_2': omega2,
            'alpha_2': alpha2_vals
        })


def generate_simulation_data(output_dir, num_simulations=50, duration=10.0, dt=0.01):
    """Generate and save simulation data for multiple random initial conditions."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pendulum = DoublePendulum()

    print(f"Generating {num_simulations} simulations...")
    theta1 = 120 * np.pi / 180
    omega1 = 0.0
    omega2 = 0.0
    # Sourced from: https://arxiv.org/pdf/2504.13453
    theta2s = np.arange(0, 3.00001, 0.1)
    for ij in range(len(theta2s)):
        theta2 = theta2s[ij]
        theta2 *= np.pi / 180
        df = pendulum.simulate(theta1, omega1, theta2, omega2, duration, dt)
        filepath = output_path / f"simulation_{ij:04d}.csv"
        df.to_csv(filepath, index=False)

    df_tst1 = pendulum.simulate(theta1, omega1, 2.05 * np.pi / 180.0, omega2, duration, dt)
    df_tst1.to_csv(output_path / "simulation_tst1.csv", index=False)
    df_tst2 = pendulum.simulate(theta1, omega1, 2.05 * np.pi / 180.0, omega2, duration, dt)
    df_tst2.to_csv(output_path / "simulation_tst2.csv", index=False)

    print(f"Data saved to {output_path}")
    return output_path


def load_and_prepare_data(data_dir, window_size=1, file_glob: str = 'simulation_0*.csv'):
    """
    Load all simulation files and prepare data for single-output prediction.

    For window_size=1: features are current state, target is next state.
    """
    data_path = Path(data_dir)
    files = sorted(data_path.glob(file_glob))

    print(f"Loading {len(files)} simulation files...")

    all_X = []
    all_y = []

    for filepath in files:
        df = pd.read_csv(filepath)

        if window_size == 1:
            X = df[INPUT_FEATURES].iloc[:-1].values
            y = df[OUTPUT_FEATURES].iloc[1:].values
        else:
            X = []
            y = []
            for j in range(len(df) - window_size):
                window = df[INPUT_FEATURES].iloc[j:j+window_size].values.flatten()
                X.append(window)
                y.append(df[OUTPUT_FEATURES].iloc[j+window_size].values)

            if X:
                X = np.array(X)
                y = np.array(y)

        all_X.append(X)
        all_y.append(y)

    X_combined = np.vstack(all_X)
    y_combined = np.vstack(all_y)

    print(f"Combined data shape: X={X_combined.shape}, y={y_combined.shape}")

    return X_combined, y_combined


def mimo_input_feature_names(window_size: int) -> list[str]:
    """Return input column names for MIMO data with given window size.

    window_size=1 → OUTPUT_FEATURES
    window_size=N → ['theta_1_step0', ..., 'theta_2_step0', ..., 'theta_2_step{N-1}']
    where step0 is oldest, step(window_size-1) is most recent.
    """
    if window_size == 1:
        return OUTPUT_FEATURES[:]
    return [f"{feat}_step{i}" for i in range(window_size) for feat in OUTPUT_FEATURES]


def load_and_prepare_mimo_data(
    data_dir, file_glob: str = 'simulation_0*.csv', window_size: int = 1
):
    """
    Load all simulation files and prepare data for MIMO full-state prediction.

    For window_size=1: input is full state at time t, output is full state at t+1.
    For window_size>1: input is the flattened states at [t-window+1, ..., t],
                       output is the full state at t+1.
    """
    data_path = Path(data_dir)
    files = sorted(data_path.glob(file_glob))

    print(f"Loading {len(files)} simulation files for MIMO (window={window_size})...")

    all_X = []
    all_y = []

    for filepath in files:
        df = pd.read_csv(filepath)
        if window_size == 1:
            X = df[OUTPUT_FEATURES].iloc[:-1].values
            y = df[OUTPUT_FEATURES].iloc[1:].values
        else:
            X_rows, y_rows = [], []
            for j in range(len(df) - window_size):
                window = df[OUTPUT_FEATURES].iloc[j:j + window_size].values.flatten()
                X_rows.append(window)
                y_rows.append(df[OUTPUT_FEATURES].iloc[j + window_size].values)
            if X_rows:
                X = np.array(X_rows)
                y = np.array(y_rows)
        all_X.append(X)
        all_y.append(y)

    X_combined = np.vstack(all_X)
    y_combined = np.vstack(all_y)

    print(f"MIMO combined data shape: X={X_combined.shape}, y={y_combined.shape}")

    return X_combined, y_combined


def train_and_evaluate_single_step(X_train, y_train, X_test, y_test):
    """Train regressor for single-step prediction (current state -> next state)."""
    print("\n" + "="*60)
    print("SINGLE-STEP PREDICTION MODEL")
    print("="*60)

    y_train_scalar = y_train[:, 0] if y_train.ndim > 1 else y_train
    y_test_scalar = y_test[:, 0] if y_test.ndim > 1 else y_test

    regressor = MixtureOfGaussiansFuzzyRegressor(
        n_output_buckets=N_BINS, tsk_order="2nd", optimize_coefficients=True,
        random_state=42
    )
    regressor.fit(X_train, y_train_scalar)

    y_pred = regressor.predict(X_test)

    mse = mean_squared_error(y_test_scalar, y_pred)
    mae = mean_absolute_error(y_test_scalar, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_scalar, y_pred)

    print(f"\n{OUTPUT_FEATURES[0]}:")
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


def train_and_evaluate_window(X_train, y_train, X_test, y_test, window_size=3, test_size=0.2):
    """Train regressor for multi-step prediction using sliding window."""
    print("\n" + "="*60)
    print(f"MULTI-STEP WINDOW PREDICTION MODEL (window_size={window_size})")
    print("="*60)

    y_train_scalar = y_train[:, 0] if y_train.ndim > 1 else y_train
    y_test_scalar = y_test[:, 0] if y_test.ndim > 1 else y_test

    regressor = MixtureOfGaussiansFuzzyRegressor(
        n_output_buckets=N_BINS, tsk_order="1st", optimize_coefficients=True,
        random_state=42
    )
    regressor.fit(X_train, y_train_scalar)

    y_pred = regressor.predict(X_test)

    mse = mean_squared_error(y_test_scalar, y_pred)
    mae = mean_absolute_error(y_test_scalar, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_scalar, y_pred)

    print(f"\n{OUTPUT_FEATURES[0]}:")
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


def train_and_evaluate_mimo(X_train, y_train, X_test, y_test, window_size: int = 1):
    """Train MIMO regressor for full-state prediction."""
    print("\n" + "="*60)
    print(f"MIMO FULL-STATE PREDICTION MODEL (window={window_size})")
    print("="*60)

    input_cols = mimo_input_feature_names(window_size)
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
        print(f"  {col:10s}: R²={r2:.4f}  RMSE={np.sqrt(mse):.4f}  MAE={mae:.4f}")

    return {
        'model_type': 'mimo',
        'window_size': window_size,
        'regressor': regressor,
        'metrics': metrics,
        'y_test': y_test_df,
        'y_pred': y_pred_df,
    }


def run_iterative_prediction(regressor, initial_window_df, n_steps, window_size: int = 1):
    """
    Iteratively apply MIMO regressor from an initial window of states.

    For window_size=1, initial_window_df should have 1 row (the seed state).
    For window_size=N, initial_window_df should have N rows (rows oldest→newest).
    The first row of the returned trajectory corresponds to the *last* row of
    initial_window_df so that the full output aligns with the actual trajectory
    starting at index window_size-1.

    Stops early if predictions become NaN (chaotic divergence). The returned
    DataFrame is padded with NaN for any steps after divergence.

    Returns:
        DataFrame of shape (n_steps+1, n_features)
    """
    from collections import deque

    input_cols = mimo_input_feature_names(window_size)

    # Seed the rolling buffer from initial_window_df.
    # If fewer rows supplied than window_size, pad at the front with the first row.
    seed_rows = [initial_window_df.iloc[i].values.copy() for i in range(len(initial_window_df))]
    while len(seed_rows) < window_size:
        seed_rows.insert(0, seed_rows[0].copy())

    buffer = deque(seed_rows[-window_size:], maxlen=window_size)

    # The "current" state for trajectory comparison is the last row in the window.
    states = [buffer[-1].copy()]
    diverged_at = None

    for step in range(n_steps):
        input_flat = np.concatenate(list(buffer))
        input_df = pd.DataFrame([input_flat], columns=input_cols)

        next_state_df = regressor.predict(input_df)
        row = next_state_df.values[0]

        if np.any(np.isnan(row)) or np.any(np.abs(row) > 1e6):
            if diverged_at is None:
                diverged_at = step + 1
                print(f"  Warning: prediction diverged at step {diverged_at} — padding remainder with NaN. Row={row}")
            row = np.full(len(OUTPUT_FEATURES), np.nan)

        states.append(row)
        buffer.append(row)

    result = pd.DataFrame(np.array(states), columns=OUTPUT_FEATURES)
    if diverged_at is not None:
        print(f"  Valid prediction steps: {diverged_at} / {n_steps + 1}")
    return result


def angles_to_xy(theta1, theta2, l1=1.0, l2=1.0):
    """Convert pendulum angles to Cartesian coordinates of both masses."""
    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)
    x2 = x1 + l2 * np.sin(theta2)
    y2 = y1 - l2 * np.cos(theta2)
    return x1, y1, x2, y2


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
        actual_frames['theta_1'].values, actual_frames['theta_2'].values
    )
    x1_pred, y1_pred, x2_pred, y2_pred = angles_to_xy(
        pred_frames['theta_1'].values, pred_frames['theta_2'].values
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.patch.set_facecolor('#1a1a2e')
    for ax in axes:
        ax.set_facecolor('#16213e')
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    axes[0].set_title('Actual', color='white', fontsize=13, fontweight='bold')
    axes[1].set_title('Predicted (iterative rollout)', color='white', fontsize=13, fontweight='bold')
    fig.suptitle('Double Pendulum: Actual vs Predicted', color='white', fontsize=14, fontweight='bold')

    trail_len = 40
    trail_alpha = np.linspace(0.05, 0.5, trail_len)

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


def plot_second_pendulum_position(results_single, results_window, dt=0.01):
    """Plot the actual and predicted position of the second pendulum as a function of time."""
    fig, axes = plt.subplots(4, 1, figsize=(10, 15))
    fig.suptitle(f'{OUTPUT_FEATURES[0]} Over Time', fontsize=16, fontweight='bold')

    def sample_to_time(indices, dt):
        return indices * dt

    ax = axes[0]
    y_test = results_single['y_test']
    y_pred = results_single['y_pred']
    time_indices = sample_to_time(np.arange(len(y_test)), dt)
    ax.plot(time_indices, y_test, 'b-', linewidth=2, label='Actual', alpha=0.8)
    ax.plot(time_indices, y_pred, 'r--', linewidth=1.5, label='Predicted', alpha=0.8)
    ax.fill_between(time_indices, y_test, y_pred, alpha=0.1, color='gray', label='Error')
    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_ylabel(f'{OUTPUT_FEATURES[0]} (radians)', fontsize=11)
    ax.set_title(f'Single-Step Model: {OUTPUT_FEATURES[0]} Position Over Time (R²={results_single["r2"]:.4f}, MAE={results_single["mae"]:.4f})', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    y_test = results_window['y_test']
    y_pred = results_window['y_pred']
    time_indices = sample_to_time(np.arange(len(y_test)), dt)
    ax.plot(time_indices, y_test, 'b-', linewidth=2, label='Actual', alpha=0.8)
    ax.plot(time_indices, y_pred, 'g--', linewidth=1.5, label='Predicted', alpha=0.8)
    ax.fill_between(time_indices, y_test, y_pred, alpha=0.1, color='gray', label='Error')
    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_ylabel(f'{OUTPUT_FEATURES[0]} (radians)', fontsize=11)
    ax.set_title(f'Multi-Step Window Model: {OUTPUT_FEATURES[0]} Position Over Time (R²={results_window["r2"]:.4f}, MAE={results_window["mae"]:.4f})', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

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

    ax = axes[3]
    ax.plot(y_test_single, y_pred_single, 'r-', linewidth=1, label='Single-Step predictions', alpha=0.7)
    ax.plot(y_test_window, y_pred_window, 'g-', linewidth=1, label='Multi-Step predictions', alpha=0.7)
    ax.set_xlabel('Angle', fontsize=11)
    ax.set_ylabel('Angle', fontsize=11)
    ax.set_title('Phasing plot', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_mimo_state_trajectories(actual_df, predicted_df, dt=0.01):
    """Plot OUTPUT_FEATURES state variables: actual vs iterative MIMO prediction."""
    n_out = len(OUTPUT_FEATURES)
    ncols = min(n_out, 2)
    nrows = (n_out + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flat
    fig.suptitle('MIMO Iterative Rollout: Actual vs Predicted State Trajectories', fontsize=14, fontweight='bold')

    t_actual = np.arange(len(actual_df)) * dt
    pred_time = np.arange(len(predicted_df)) * dt
    n = min(len(actual_df), len(predicted_df))

    for idx, col in enumerate(OUTPUT_FEATURES):
        ax = axes_flat[idx]
        act = actual_df[col].values[:n]
        pred = predicted_df[col].values[:n]
        valid = ~np.isnan(pred)
        ax.plot(t_actual[:n], act, 'b-', linewidth=1.5, label='Actual', alpha=0.8)
        ax.plot(pred_time[:n][valid], pred[valid], 'r--', linewidth=1.5, label='Predicted', alpha=0.8)
        if valid.any():
            ax.fill_between(t_actual[:n][valid], act[valid], pred[valid], alpha=0.1, color='gray')
            mae = np.mean(np.abs(act[valid] - pred[valid]))
            title = f'{col}  (MAE={mae:.4f}, valid={valid.sum()}/{n})'
        else:
            title = f'{col}  (no valid predictions)'
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    return fig


def test_double_pendulum_fuzzy_prediction():
    """
    Main test: simulate double pendulum and train fuzzy regression models.
    Includes MIMO full-state prediction and iterative rollout with GIF animation.
    """
    test_dir = Path(__file__).parent
    data_dir = test_dir / "double_pendulum_data"
    dt = 0.01

    # Step 1: Generate simulation data
    with time_this("gen-sim-data"):
        generate_simulation_data(
            data_dir, num_simulations=15, duration=3.0, dt=dt
        )

    # Step 2: Single-step prediction (theta_2 only)
    print("\n" + "#"*60)
    print("# STEP 2: Single-Step Prediction Model")
    print("#"*60)
    with time_this('train-single'):
        X_single_train, y_single_train = load_and_prepare_data(data_dir, window_size=1)
        X_single_test, y_single_test = load_and_prepare_data(data_dir, file_glob='simulation_tst*.csv', window_size=1)
        results_single = train_and_evaluate_single_step(X_single_train, y_single_train, X_single_test, y_single_test)

    # Step 3: Multi-step window prediction
    # print("\n" + "#"*60)
    # print("# STEP 3: Multi-Step Window Prediction Model")
    # print("#"*60)
    # window_size = 3
    # with time_this('train-multi-step'):
    #     X_window_train, y_window_train = load_and_prepare_data(data_dir, window_size=window_size)
    #     X_window_test, y_window_test = load_and_prepare_data(data_dir, file_glob='simulation_tst*.csv', window_size=window_size)
    #     results_window = train_and_evaluate_window(X_window_train, y_window_train, X_window_test, y_window_test, window_size=window_size)

    # Step 4: MIMO full-state prediction
    print("\n" + "#"*60)
    print(f"# STEP 4: MIMO Full-State Prediction Model (window={MIMO_WINDOW_SIZE})")
    print("#"*60)
    with time_this('train-mimo'):
        X_mimo_train, y_mimo_train = load_and_prepare_mimo_data(data_dir, window_size=MIMO_WINDOW_SIZE)
        X_mimo_test, y_mimo_test = load_and_prepare_mimo_data(
            data_dir, file_glob='simulation_tst*.csv', window_size=MIMO_WINDOW_SIZE
        )
        results_mimo = train_and_evaluate_mimo(
            X_mimo_train, y_mimo_train, X_mimo_test, y_mimo_test, window_size=MIMO_WINDOW_SIZE
        )

    # Step 5: Iterative rollout from initial conditions
    print("\n" + "#"*60)
    print("# STEP 5: Iterative Rollout from Initial Conditions")
    print("#"*60)
    tst_df = pd.read_csv(data_dir / "simulation_tst1.csv")
    # Seed window: first MIMO_WINDOW_SIZE rows; predict everything after the seed.
    initial_window = tst_df[OUTPUT_FEATURES].iloc[:MIMO_WINDOW_SIZE]
    n_steps = len(tst_df) - MIMO_WINDOW_SIZE

    print(f"Seed window ({MIMO_WINDOW_SIZE} rows):\n{initial_window.to_string()}")
    print(f"Running {n_steps} iterative prediction steps...")

    with time_this('iterative-rollout'):
        predicted_trajectory = run_iterative_prediction(
            results_mimo['regressor'], initial_window, n_steps, window_size=MIMO_WINDOW_SIZE
        )

    # Align actual to start at the last row of the seed window.
    actual_trajectory = tst_df[OUTPUT_FEATURES].iloc[MIMO_WINDOW_SIZE - 1:].reset_index(drop=True)
    n = min(len(actual_trajectory), len(predicted_trajectory))

    print("\nIterative Rollout Metrics (valid steps only):")
    for col in OUTPUT_FEATURES:
        act = actual_trajectory[col].values[:n]
        pred = predicted_trajectory[col].values[:n]
        valid = ~np.isnan(pred)
        if valid.sum() < 2:
            print(f"  {col:10s}: insufficient valid predictions")
            continue
        mae = np.mean(np.abs(act[valid] - pred[valid]))
        r2 = r2_score(act[valid], pred[valid])
        print(f"  {col:10s}: MAE={mae:.4f}  R²={r2:.4f}  (valid={valid.sum()}/{n})")

    # Step 6: Summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print("\nSingle-Step Model:")
    print(f"  R²:   {results_single['r2']:.6f}")
    print(f"  RMSE: {results_single['rmse']:.6f}")
    print(f"  MAE:  {results_single['mae']:.6f}")

    # print("\nMulti-Step Window Model:")
    # print(f"  R²:   {results_window['r2']:.6f}")
    # print(f"  RMSE: {results_window['rmse']:.6f}")
    # print(f"  MAE:  {results_window['mae']:.6f}")

    print("\nMIMO Model (1-step):")
    for col in OUTPUT_FEATURES:
        m = results_mimo['metrics'][col]
        print(f"  {col:10s}: R²={m['r2']:.4f}  RMSE={m['rmse']:.4f}")

    # if results_single['r2'] > results_window['r2']:
    #     print("\n  Single-step model shows better R² score for theta_2")
    # else:
    #     print("\n  Multi-step window model shows better R² score for theta_2")

    # Step 7: Plots
    print("\n" + "="*60)
    print("GENERATING VISUALIZATION PLOTS")
    print("="*60)

    # print("\nPlot 1: Scatter and Residual Comparison")
    # fig1 = plot_prediction_comparison(results_single, results_window)
    # fig1.savefig(test_dir / "prediction_comparison.png", dpi=200, bbox_inches='tight')
    # plt.close(fig1)
    #
    # print("\nPlot 2: Second Pendulum Position Over Time")
    # fig2 = plot_second_pendulum_position(results_single, results_window, dt=dt)
    # fig2.savefig(test_dir / "second_pendulum_position.png", dpi=200, bbox_inches='tight')
    # plt.close(fig2)

    print("\nPlot 3: MIMO Iterative Rollout State Trajectories")
    fig3 = plot_mimo_state_trajectories(actual_trajectory, predicted_trajectory, dt=dt)
    fig3.savefig(test_dir / "mimo_iterative_trajectories.png", dpi=200, bbox_inches='tight')
    plt.close(fig3)

    # Step 8: GIF animation
    print("\n" + "="*60)
    print("GENERATING GIF ANIMATION")
    print("="*60)
    gif_path = test_dir / "double_pendulum_comparison.gif"
    create_pendulum_animation(
        actual_trajectory, predicted_trajectory,
        gif_path, dt=dt, max_frames=300, fps=25
    )

    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_double_pendulum_fuzzy_prediction()
