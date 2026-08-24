"""
Test double pendulum simulation with Gaussian mixture regressor prediction.

Simulates double pendulum using Lagrangian mechanics, generates datasets
with random initial conditions, trains fuzzy regressors to predict state
transitions, and evaluates prediction accuracy on continuous outputs.
"""

import unittest
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.integrate import odeint

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tribblefis.gaussian_regressor import MixtureOfGaussiansFuzzyRegressor
from tests.ode_helpers import plot_second_pendulum_position, plot_prediction_comparison, load_and_prepare_data, \
    train_and_evaluate_window, train_and_evaluate_single_step

# Comparison set: https://arxiv.org/pdf/2504.13453
N_BINS = 2
OUTPUT_FEATURES = ['r','theta','r_dot', 'omega', 'r_ddot', 'alpha']
INPUT_FEATURES = ['r','theta','r_dot', 'omega', 'r_ddot', 'alpha']

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


def rollout_predict(models, pendulum, initial_state, n_steps, dt=0.01):
    """
    Perform iterative roll-out prediction using model predictions.
    Feeds previous predictions back as input for next step.

    Args:
        models: tuple of (regressor_r, regressor_theta)
        pendulum: AtwoodMachine instance for equations of motion
        initial_state: [r, theta, r_dot, omega]
        n_steps: number of steps to predict
        dt: timestep

    Returns:
        trajectory: array of [r, theta, r_dot, omega] states
    """
    regressor_r, regressor_theta = models
    trajectory = [initial_state.copy()]

    for step in range(n_steps):
        current = trajectory[-1]
        r, theta, r_dot, omega = current

        # Compute accelerations using true equations of motion
        _, _, r_ddot, alpha = pendulum.equations_of_motion(current, 0)

        # Model input: [r_dot, r_ddot, omega, alpha]
        input_vec = np.array([[r_dot, r_ddot, omega, alpha]])

        # Predict next position using separate models
        r_next = regressor_r.predict(input_vec)[0]
        theta_next = regressor_theta.predict(input_vec)[0]

        # Check for NaN or out-of-bounds values
        if np.isnan(r_next) or np.isnan(theta_next):
            # Stop rollout if model diverges
            return np.array(trajectory)

        # Clamp r to reasonable bounds (must be positive and within simulation range)
        r_next = np.clip(r_next, 0.5, 5.0)

        # Estimate velocities using finite differences with damping
        r_dot_next = (r_next - r) / dt
        omega_next = (theta_next - theta) / dt

        # Clamp velocities to prevent instability
        r_dot_next = np.clip(r_dot_next, -10, 10)
        omega_next = np.clip(omega_next, -10, 10)

        next_state = np.array([r_next, theta_next, r_dot_next, omega_next])
        trajectory.append(next_state)

    return np.array(trajectory)


def train_and_evaluate_rollout(X_train, y_train, initial_states_test, pendulum, n_steps=100, dt=0.01):
    """
    Train regressor for roll-out prediction and evaluate on test trajectories.

    Predicts position r from [r_dot, r_ddot, omega, alpha], then iteratively
    feeds predictions back for multi-step horizons.

    Args:
        X_train: training input features
        y_train: training output features (r and theta, but uses r)
        initial_states_test: list of initial states to test roll-out on
        pendulum: AtwoodMachine instance
        n_steps: steps to roll out
        dt: timestep

    Returns:
        dict with evaluation metrics and trajectories
    """
    print("\n" + "="*60)
    print("ROLL-OUT PREDICTION MODEL")
    print("="*60)

    # Train regressors for r and theta outputs
    y_train_r = y_train[:, 0] if y_train.ndim > 1 else y_train
    y_train_theta = y_train[:, 1] if y_train.ndim > 1 else y_train

    # rollout_predict() only ever has [r_dot, r_ddot, omega, alpha] available at
    # each step (r/theta themselves are what it's trying to predict), so the
    # model must be trained on that same 4-feature subset of X_train -- fitting
    # on the full INPUT_FEATURES set here mismatches rollout_predict's 4-column
    # input at inference time.
    rollout_feature_idx = [INPUT_FEATURES.index(f) for f in ("r_dot", "r_ddot", "omega", "alpha")]
    X_train_rollout = X_train[:, rollout_feature_idx]

    regressor_r = MixtureOfGaussiansFuzzyRegressor(
        n_output_buckets=N_BINS, tsk_order="1st", optimize_coefficients=True,
        random_state=42
    )
    regressor_r.fit(X_train_rollout, y_train_r)

    regressor_theta = MixtureOfGaussiansFuzzyRegressor(
        n_output_buckets=N_BINS, tsk_order="1st", optimize_coefficients=True,
        random_state=42
    )
    regressor_theta.fit(X_train_rollout, y_train_theta)

    # Perform roll-out predictions on test initial conditions
    rollout_trajectories = []
    for init_state in initial_states_test:
        traj = rollout_predict((regressor_r, regressor_theta), pendulum, init_state, n_steps, dt)
        rollout_trajectories.append(traj)

    print(f"\nRoll-out prediction completed for {len(rollout_trajectories)} trajectories")
    print(f"Each trajectory: {n_steps} steps")

    return {
        'model_type': 'rollout',
        'regressor_r': regressor_r,
        'regressor_theta': regressor_theta,
        'rollout_trajectories': rollout_trajectories,
        'n_steps': n_steps,
        'dt': dt,
    }


def plot_rollout_comparison(results_rollout, actual_trajectories, dt=0.01):
    """
    Plot roll-out predictions vs actual trajectories.
    Returns the figure object.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Roll-Out Prediction Comparison', fontsize=16, fontweight='bold')

    # Extract predicted and actual r (position)
    ax = axes[0, 0]
    for i, (pred_traj, actual_traj) in enumerate(zip(
        results_rollout['rollout_trajectories'],
        actual_trajectories
    )):
        # Truncate actual to match prediction length
        n_pred = len(pred_traj)
        actual_trunc = actual_traj[:n_pred]
        time = np.arange(len(actual_trunc)) * dt
        ax.plot(time, actual_trunc[:, 0], 'b-', linewidth=1.5, alpha=0.6, label='Actual' if i == 0 else '')
        ax.plot(time, pred_traj[:, 0], 'r--', linewidth=1, alpha=0.6, label='Predicted' if i == 0 else '')

    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_ylabel('r (meters)', fontsize=11)
    ax.set_title('Roll-out: Position r Over Time', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Extract predicted and actual theta (angle)
    ax = axes[0, 1]
    for i, (pred_traj, actual_traj) in enumerate(zip(
        results_rollout['rollout_trajectories'],
        actual_trajectories
    )):
        # Truncate actual to match prediction length
        n_pred = len(pred_traj)
        actual_trunc = actual_traj[:n_pred]
        time = np.arange(len(actual_trunc)) * dt
        ax.plot(time, actual_trunc[:, 1], 'b-', linewidth=1.5, alpha=0.6, label='Actual' if i == 0 else '')
        ax.plot(time, pred_traj[:, 1], 'g--', linewidth=1, alpha=0.6, label='Predicted' if i == 0 else '')

    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_ylabel('θ (radians)', fontsize=11)
    ax.set_title('Roll-out: Angle θ Over Time', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Position error over time
    ax = axes[1, 0]
    for i, (pred_traj, actual_traj) in enumerate(zip(
        results_rollout['rollout_trajectories'],
        actual_trajectories
    )):
        # Truncate actual to match prediction length
        n_pred = len(pred_traj)
        actual_trunc = actual_traj[:n_pred]
        error_r = np.abs(actual_trunc[:, 0] - pred_traj[:, 0])
        time = np.arange(len(error_r)) * dt
        ax.plot(time, error_r, linewidth=1.5, alpha=0.7)

    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_ylabel('|Error| (meters)', fontsize=11)
    ax.set_title('Roll-out: Position r Prediction Error', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Angle error over time
    ax = axes[1, 1]
    for i, (pred_traj, actual_traj) in enumerate(zip(
        results_rollout['rollout_trajectories'],
        actual_trajectories
    )):
        # Truncate actual to match prediction length
        n_pred = len(pred_traj)
        actual_trunc = actual_traj[:n_pred]
        error_theta = np.abs(actual_trunc[:, 1] - pred_traj[:, 1])
        time = np.arange(len(error_theta)) * dt
        ax.plot(time, error_theta, linewidth=1.5, alpha=0.7)

    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_ylabel('|Error| (radians)', fontsize=11)
    ax.set_title('Roll-out: Angle θ Prediction Error', fontsize=12)
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


@pytest.mark.slow
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
        X_single_train, y_single_train = load_and_prepare_data(data_dir,input_features=INPUT_FEATURES, output_features=OUTPUT_FEATURES, window_size=1)
        X_single_test, y_single_test = load_and_prepare_data(data_dir, file_glob='simulation_tst*.csv', input_features=INPUT_FEATURES, output_features=OUTPUT_FEATURES, window_size=1)
        results_single = train_and_evaluate_single_step(N_BINS, OUTPUT_FEATURES, X_single_train, y_single_train, X_single_test, y_single_test)

        # Step 5: Multi-step window prediction
        print("\n" + "#"*60)
        print("# STEP 5: Multi-Step Window Prediction Model")
        print("#"*60)
        window_size = 3
        X_window_train, y_window_train = load_and_prepare_data(data_dir, input_features=INPUT_FEATURES, output_features=OUTPUT_FEATURES, window_size=window_size)
        X_window_test, y_window_test = load_and_prepare_data(data_dir, file_glob='simulation_tst*.csv', input_features=INPUT_FEATURES, output_features=OUTPUT_FEATURES, window_size=window_size)
        results_window = train_and_evaluate_window(N_BINS, OUTPUT_FEATURES,X_window_train, y_window_train, X_window_test, y_window_test, window_size=window_size)

        # Step 6: Roll-out prediction
        print("\n" + "#"*60)
        print("# STEP 6: Roll-Out Prediction Model")
        print("#"*60)
        X_rollout_train, y_rollout_train = load_and_prepare_data(data_dir, input_features=INPUT_FEATURES, output_features=OUTPUT_FEATURES, window_size=1)

        # Load actual test trajectories for comparison
        test_files = sorted((data_dir).glob('simulation_tst*.csv'))
        actual_test_trajectories = []
        initial_states_test = []
        for test_file in test_files:
            df = pd.read_csv(test_file)
            actual_test_trajectories.append(df[['r', 'theta', 'r_dot', 'omega']].values)
            initial_states_test.append(df[['r', 'theta', 'r_dot', 'omega']].iloc[0].values)

        # Train roll-out model
        pendulum = AtwoodMachine()
        results_rollout = train_and_evaluate_rollout(
            X_rollout_train, y_rollout_train,
            initial_states_test, pendulum,
            n_steps=100, dt=0.01
        )

        # Step 7: Summary evaluation
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

        # Evaluate roll-out trajectories (limit actual to rollout length)
        print("\nRoll-Out Model (Multi-step Horizon):")
        rollout_errors_r = []
        rollout_errors_theta = []
        for i, (pred_traj, actual_traj) in enumerate(zip(results_rollout['rollout_trajectories'], actual_test_trajectories)):
            # Truncate actual trajectory to match prediction length
            n_pred = len(pred_traj)
            actual_truncated = actual_traj[:n_pred]

            # Check for NaN in predictions
            if np.any(np.isnan(pred_traj)):
                print(f"  Warning: Trajectory {i} contains NaN predictions (diverged)")
                continue

            error_r = np.abs(actual_truncated[:, 0] - pred_traj[:, 0])
            error_theta = np.abs(actual_truncated[:, 1] - pred_traj[:, 1])
            rollout_errors_r.extend(error_r)
            rollout_errors_theta.extend(error_theta)

        if rollout_errors_r:
            rollout_mae_r = np.mean(rollout_errors_r)
            rollout_mae_theta = np.mean(rollout_errors_theta)
            print(f"  MAE (r):     {rollout_mae_r:.6f}")
            print(f"  MAE (θ):     {rollout_mae_theta:.6f}")
        else:
            print(f"  All trajectories diverged (NaN)")
            rollout_mae_r, rollout_mae_theta = np.nan, np.nan

        print(f"  Mean horizon: {results_rollout['n_steps']} steps")

        # Plot results
        print("\n" + "="*60)
        print("GENERATING VISUALIZATION PLOTS")
        print("="*60)

        print("\nPlot 1: Scatter and Residual Comparison")
        fig1 = plot_prediction_comparison(OUTPUT_FEATURES, results_single, results_window)
        plot_file_1 = test_dir / "prediction_comparison.png"
        fig1.savefig(plot_file_1, dpi=200, bbox_inches='tight')
        print(f"  Saved to: {plot_file_1}")
        plt.close(fig1)

        print("\nPlot 2: Roll-Out Prediction Comparison")
        fig2 = plot_rollout_comparison(results_rollout, actual_test_trajectories)
        plot_file_2 = test_dir / "rollout_comparison.png"
        fig2.savefig(plot_file_2, dpi=200, bbox_inches='tight')
        print(f"  Saved to: {plot_file_2}")
        plt.close(fig2)

        print("\nPlot 3: Second Pendulum Position Over Time")
        fig3 = plot_second_pendulum_position(OUTPUT_FEATURES, results_single, results_window)
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
