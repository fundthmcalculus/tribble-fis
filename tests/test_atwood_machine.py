"""
Test double pendulum simulation with Gaussian mixture regressor prediction.

Simulates double pendulum using Lagrangian mechanics, generates datasets
with random initial conditions, trains fuzzy regressors to predict state
transitions, and evaluates prediction accuracy on continuous outputs.
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tribblefis.gaussian_regressor import MixtureOfGaussiansFuzzyRegressor


# Comparison set: https://arxiv.org/pdf/2504.13453
N_BINS = 15
OUTPUT_FEATURES = ['theta']
INPUT_FEATURES = ['r_dot', 'r_ddot', 'omega', 'alpha']

class AtwoodMachine:
    """Double pendulum simulator using Lagrangian mechanics."""

    def __init__(self, m1=2.0, m2=1.0, r0 = 2.0, k=20.0, g=9.81):
        """
        Initialize double pendulum.

        Args:
            m1, m2: masses of block 1 pendulum 2
            r0: length of rope initially
            g: gravitational acceleration
        """
        self.m1 = m1
        self.m2 = m2
        self.r0 = r0
        self.k = k
        self.g = g

    def equations_of_motion(self, state, t):
        """
        Compute double pendulum equations of motion using Lagrangian approach.

        State: [r, theta, r_dot, omega]
        Returns: [r_dot, omega, r_ddot, alpha]
        """
        r, theta, r_dot, omega = state
        r_ddot = (r*omega**2+self.g*np.cos(theta) - self.k / self.m2 * (r-self.r0))/(self.m1 / self.m2 + 1)
        alpha = (-2*r_dot*omega-self.g*np.sin(theta))/r
        return [r_dot, omega, r_ddot, alpha]

    def simulate(self, r_0, theta_0, r_dot0, omega_0, duration=10.0, dt=0.001):
        """
        Simulate double pendulum from initial conditions.

        Args:
            r_0, theta_0, r_dot0, omega_0: initial conditions
            duration: simulation time in seconds
            dt: timestep in seconds
        """
        t = np.arange(0, duration, dt)
        state0 = [r_0, theta_0, r_dot0, omega_0]

        solution = odeint(self.equations_of_motion, state0, t)

        # Compute alpha values from derivatives
        r = solution[:, 0]
        theta = solution[:, 1]
        r_dot = solution[:, 2]
        omega = solution[:, 3]

        # Compute alpha (angular acceleration) at each point
        r_ddot = []
        alpha = []
        for i, state in enumerate(solution):
            _, _, a1, a2 = self.equations_of_motion(state, t[i])
            r_ddot.append(a1)
            alpha.append(a2)

        return pd.DataFrame({
            'r': r,
            'r_dot': r_dot,
            'r_ddot': r_ddot,
            'theta': theta,
            'omega': omega,
            'alpha': alpha
        })


def generate_simulation_data(output_dir, num_simulations=50, duration=10.0, dt=0.01):
    """
    Generate and save simulation data for multiple random initial conditions.

    Args:
        output_dir: directory to save simulation files
        num_simulations: number of random initial conditions to simulate
        duration: simulation duration in seconds
        dt: timestep in seconds
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pendulum = AtwoodMachine()

    print(f"Generating {num_simulations} simulations...")
    # Sourced from: https://arxiv.org/pdf/2504.13453
    rs = np.arange(1, 3.00001, 0.1)
    for ij, r_c in enumerate(rs):
        # Simulate
        df = pendulum.simulate(r_c, np.pi/2.0, 0.0, 0.0, duration, dt)
        # Save
        filepath = output_path / f"simulation_{ij:04d}.csv"
        df.to_csv(filepath, index=False)

    df_tst1 = pendulum.simulate(2.05, np.pi/2.0, 0.0, 0.0, duration, dt)
    df_tst1.to_csv(output_path / "simulation_tst1.csv", index=False)
    df_tst2 = pendulum.simulate(2.0, np.pi/2.0, 0.1, 0.0, duration, dt)
    df_tst2.to_csv(output_path / "simulation_tst2.csv", index=False)


    print(f"Data saved to {output_path}")
    return output_path


def load_and_prepare_data(data_dir, window_size=1, file_glob: str = 'simulation_0*.csv'):
    """
    Load all simulation files and prepare data for prediction.

    For window_size=1: features are current state, target is next state.
    For window_size>1: features are last n timesteps, target is next state.

    Args:
        data_dir: directory containing simulation CSV files
        window_size: number of past timesteps to use as features
        file_glob: For picking the existing simulation data.

    Returns:
        tuple: (X, y) where X is features and y is target
    """
    data_path = Path(data_dir)
    files = sorted(data_path.glob(file_glob))

    print(f"Loading {len(files)} simulation files...")

    all_X = []
    all_y = []

    for filepath in files:
        df = pd.read_csv(filepath)

        if window_size == 1:
            # Single timestep: current state -> next state
            X = df[INPUT_FEATURES].iloc[:-1].values
            y = df[OUTPUT_FEATURES].iloc[1:].values
        else:
            # Multi-step window
            X = []
            y = []
            for j in range(len(df) - window_size):
                # Take last window_size timesteps as features
                window = df[INPUT_FEATURES].iloc[j:j+window_size].values.flatten()
                X.append(window)
                # Next timestep as target
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


def train_and_evaluate_single_step(X_train, y_train, X_test, y_test):
    """
    Train regressor for single-step prediction (current state -> next state).

    Args:
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
        n_output_buckets=N_BINS, tsk_order="2nd", optimize_coefficients=True,
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
    """
    Train regressor for multi-step prediction using sliding window.

    Args:
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

    # Train regressor on continuous target (first output feature)
    y_train_scalar = y_train[:, 0] if y_train.ndim > 1 else y_train
    y_test_scalar = y_test[:, 0] if y_test.ndim > 1 else y_test

    # Train regressor
    regressor = MixtureOfGaussiansFuzzyRegressor(
        n_output_buckets=N_BINS, tsk_order="1st", optimize_coefficients=True,
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


def plot_prediction_comparison(results_single, results_window):
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
    ax.set_xlabel(f'Actual {OUTPUT_FEATURES[0]} (rad)', fontsize=11)
    ax.set_ylabel(f'Predicted {OUTPUT_FEATURES[0]} (rad)', fontsize=11)
    ax.set_title(f'Single-Step: Actual vs Predicted\nR²={results_single["r2"]:.4f}, RMSE={results_single["rmse"]:.4f}', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Single-step residuals
    ax = axes[0, 1]
    residuals_single = y_test - y_pred
    ax.scatter(y_pred, residuals_single, alpha=0.5, s=20, edgecolors='k', linewidth=0.3)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel(f'Predicted {OUTPUT_FEATURES[0]} (rad)', fontsize=11)
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
    ax.set_xlabel(f'Actual {OUTPUT_FEATURES[0]} (rad)', fontsize=11)
    ax.set_ylabel(f'Predicted {OUTPUT_FEATURES[0]} (rad)', fontsize=11)
    ax.set_title(f'Multi-Step (window=3): Actual vs Predicted\nR²={results_window["r2"]:.4f}, RMSE={results_window["rmse"]:.4f}', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Multi-step residuals
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
    """
    Plot the actual and predicted position of the second pendulum ({OUTPUT_FEATURES[0]}) as a function of time.
    Shows detailed comparison between actual and predicted trajectories.
    Returns the figure object.
    """
    fig, axes = plt.subplots(4, 1, figsize=(10, 15))
    fig.suptitle(f'{OUTPUT_FEATURES[0]} Over Time', fontsize=16, fontweight='bold')

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
    ax.set_ylabel(f'{OUTPUT_FEATURES[0]} (radians)', fontsize=11)
    ax.set_title(f'Single-Step Model: {OUTPUT_FEATURES[0]} Position Over Time (R²={results_single["r2"]:.4f}, MAE={results_single["mae"]:.4f})', fontsize=12)
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
    ax.set_ylabel(f'{OUTPUT_FEATURES[0]} (radians)', fontsize=11)
    ax.set_title(f'Multi-Step Window Model: {OUTPUT_FEATURES[0]} Position Over Time (R²={results_window["r2"]:.4f}, MAE={results_window["mae"]:.4f})', fontsize=12)
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


def plot_trace_comparison(results_single, results_window):
    """
    Plot time-series traces of predicted vs actual values.
    Returns the figure object.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle('Prediction Traces: Actual vs Predicted', fontsize=16, fontweight='bold')

    # Single-step trace
    ax = axes[0]
    y_test = results_single['y_test']
    y_pred = results_single['y_pred']
    indices = np.arange(len(y_test))
    ax.plot(indices, y_test, 'b-', linewidth=1.5, label='Actual', alpha=0.7)
    ax.plot(indices, y_pred, 'r--', linewidth=1.5, label='Predicted', alpha=0.7)
    ax.set_xlabel('Test Sample Index', fontsize=11)
    ax.set_ylabel(f'{OUTPUT_FEATURES[0]} (rad)', fontsize=11)
    ax.set_title(f'Single-Step Predictions (R²={results_single["r2"]:.4f})', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Multi-step trace
    ax = axes[1]
    y_test = results_window['y_test']
    y_pred = results_window['y_pred']
    indices = np.arange(len(y_test))
    ax.plot(indices, y_test, 'b-', linewidth=1.5, label='Actual', alpha=0.7)
    ax.plot(indices, y_pred, 'g--', linewidth=1.5, label='Predicted', alpha=0.7)
    ax.set_xlabel('Test Sample Index', fontsize=11)
    ax.set_ylabel(f'{OUTPUT_FEATURES[0]} (rad)', fontsize=11)
    ax.set_title(f'Multi-Step Window Predictions (window=3, R²={results_window["r2"]:.4f})', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


class TestAtwoodMachineFuzzyPrediction(unittest.TestCase):
    """Integration test for double pendulum fuzzy regression."""

    def test_double_pendulum_fuzzy_prediction(self):
        """
        Main test: simulate double pendulum and train fuzzy regression models.
        """
        # Setup
        test_dir = Path(__file__).parent
        data_dir = test_dir / "atwood_data"

        # Step 1-3: Generate simulation data
        generate_simulation_data(
            data_dir, num_simulations=15, duration=3.0, dt=0.01
        )

        # Step 4: Single-step prediction
        print("\n" + "#"*60)
        print("# STEP 4: Single-Step Prediction Model")
        print("#"*60)
        X_single_train, y_single_train = load_and_prepare_data(data_dir, window_size=1)
        X_single_test, y_single_test = load_and_prepare_data(data_dir, file_glob='simulation_tst*.csv', window_size=1)
        results_single = train_and_evaluate_single_step(X_single_train, y_single_train, X_single_test, y_single_test)

        # Step 5: Multi-step window prediction
        print("\n" + "#"*60)
        print("# STEP 5: Multi-Step Window Prediction Model")
        print("#"*60)
        window_size = 3
        X_window_train, y_window_train = load_and_prepare_data(data_dir, window_size=window_size)
        X_window_test, y_window_test = load_and_prepare_data(data_dir, file_glob='simulation_tst*.csv', window_size=window_size)
        results_window = train_and_evaluate_window(X_window_train, y_window_train, X_window_test, y_window_test, window_size=window_size)

        # Step 6: Summary evaluation and assertions
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print("\nSingle-Step Model:")
        print(f"  R²:   {results_single['r2']:.6f}")
        print(f"  RMSE: {results_single['rmse']:.6f}")
        print(f"  MAE:  {results_single['mae']:.6f}")

        print("\nMulti-Step Window Model:")
        print(f"  R²:   {results_window['r2']:.6f}")
        print(f"  RMSE: {results_window['rmse']:.6f}")
        print(f"  MAE:  {results_window['mae']:.6f}")

        print("\nComparison:")
        if results_single['r2'] > results_window['r2']:
            print("  Single-step model shows better R² score")
        else:
            print("  Multi-step window model shows better R² score")

        # Plot results
        print("\n" + "="*60)
        print("GENERATING VISUALIZATION PLOTS")
        print("="*60)

        print("\nPlot 1: Scatter and Residual Comparison")
        fig1 = plot_prediction_comparison(results_single, results_window)
        plot_file_1 = test_dir / "prediction_comparison.png"
        fig1.savefig(plot_file_1, dpi=200, bbox_inches='tight')
        print(f"  Saved to: {plot_file_1}")
        plt.close(fig1)

        print("\nPlot 3: Second Pendulum Position Over Time")
        fig3 = plot_second_pendulum_position(results_single, results_window)
        plot_file_3 = test_dir / "second_pendulum_position.png"
        fig3.savefig(plot_file_3, dpi=200, bbox_inches='tight')
        print(f"  Saved to: {plot_file_3}")
        plt.close(fig3)

        print("\nTest completed successfully!")

        # Basic assertions to verify models trained successfully
        self.assertIsNotNone(results_single['regressor'])
        self.assertIsNotNone(results_window['regressor'])
        self.assertGreater(len(results_single['y_pred']), 0)
        self.assertGreater(len(results_window['y_pred']), 0)


if __name__ == '__main__':
    unittest.main()
