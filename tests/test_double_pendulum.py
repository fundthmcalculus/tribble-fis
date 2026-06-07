"""
Test double pendulum simulation with Gaussian mixture classifier prediction.

Simulates double pendulum using Lagrangian mechanics, generates datasets
with random initial conditions, trains fuzzy classifiers to predict state
transitions, and evaluates prediction accuracy.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tribblefis.gaussian_classifier import MixtureOfGaussiansFuzzyClassifier

N_BINS = 10
OUTPUT_FEATURES = ['theta_2']

class DoublePendulum:
    """Double pendulum simulator using Lagrangian mechanics."""

    def __init__(self, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81):
        """
        Initialize double pendulum.

        Args:
            m1, m2: masses of pendulum 1 and 2
            l1, l2: lengths of pendulum 1 and 2
            g: gravitational acceleration
        """
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

    def simulate(self, theta1_0, omega1_0, theta2_0, omega2_0, duration=10.0, dt=0.01):
        """
        Simulate double pendulum from initial conditions.

        Args:
            theta1_0, omega1_0, theta2_0, omega2_0: initial conditions
            duration: simulation time in seconds
            dt: timestep in seconds

        Returns:
            DataFrame with columns: theta_1, omega_1, alpha_1, theta_2, omega_2, alpha_2
        """
        t = np.arange(0, duration, dt)
        state0 = [theta1_0, omega1_0, theta2_0, omega2_0]

        solution = odeint(self.equations_of_motion, state0, t)

        # Compute alpha values from derivatives
        theta1 = solution[:, 0]
        omega1 = solution[:, 1]
        theta2 = solution[:, 2]
        omega2 = solution[:, 3]

        # Compute alpha (angular acceleration) at each point
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

    pendulum = DoublePendulum()

    print(f"Generating {num_simulations} simulations...")
    for i in range(num_simulations):
        # Random initial conditions
        theta1_0 = np.random.uniform(-np.pi, np.pi)
        omega1_0 = np.random.uniform(-2, 2)
        theta2_0 = np.random.uniform(-np.pi, np.pi)
        omega2_0 = np.random.uniform(-2, 2)

        # Simulate
        df = pendulum.simulate(theta1_0, omega1_0, theta2_0, omega2_0, duration, dt)

        # Save
        filepath = output_path / f"simulation_{i:04d}.csv"
        df.to_csv(filepath, index=False)

        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{num_simulations}")

    print(f"Data saved to {output_path}")
    return output_path


def load_and_prepare_data(data_dir, window_size=1):
    """
    Load all simulation files and prepare data for prediction.

    For window_size=1: features are current state, target is next state.
    For window_size>1: features are last n timesteps, target is next state.

    Args:
        data_dir: directory containing simulation CSV files
        window_size: number of past timesteps to use as features

    Returns:
        tuple: (X, y) where X is features and y is target
    """
    data_path = Path(data_dir)
    files = sorted(data_path.glob("simulation_*.csv"))

    print(f"Loading {len(files)} simulation files...")

    all_X = []
    all_y = []

    feature_cols = ['omega_1', 'alpha_1', 'omega_2', 'alpha_2']
    output_cols = ['theta_2']

    for filepath in files:
        df = pd.read_csv(filepath)

        if window_size == 1:
            # Single timestep: current state -> next state
            X = df[feature_cols].iloc[:-1].values
            y = df[output_cols].iloc[1:].values
        else:
            # Multi-step window
            X = []
            y = []
            for j in range(len(df) - window_size):
                # Take last window_size timesteps as features
                window = df[feature_cols].iloc[j:j+window_size].values.flatten()
                X.append(window)
                # Next timestep as target
                y.append(df[output_cols].iloc[j+window_size].values)

            if X:
                X = np.array(X)
                y = np.array(y)

        all_X.append(X)
        all_y.append(y)

    X_combined = np.vstack(all_X)
    y_combined = np.vstack(all_y)

    print(f"Combined data shape: X={X_combined.shape}, y={y_combined.shape}")

    return X_combined, y_combined


def train_and_evaluate_single_step(X, y, test_size=0.2):
    """
    Train classifier for single-step prediction (current state -> next state).

    Args:
        X: features (current state)
        y: targets (next state)
        test_size: fraction of data for testing

    Returns:
        dict with evaluation metrics
    """
    print("\n" + "="*60)
    print("SINGLE-STEP PREDICTION MODEL")
    print("="*60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # For classification, we need to discretize the continuous targets
    # Create bins for each output feature
    n_bins = N_BINS

    # Discretize targets into classes based on bins
    y_train_binned = np.zeros((len(y_train), len(feature_names)), dtype=int)
    y_test_binned = np.zeros((len(y_test), len(feature_names)), dtype=int)

    for feat_idx, feat_name in enumerate(feature_names):
        bins = np.percentile(y_train[:, feat_idx], np.linspace(0, 100, n_bins+1))
        y_train_binned[:, feat_idx] = np.digitize(y_train[:, feat_idx], bins) - 1
        y_test_binned[:, feat_idx] = np.digitize(y_test[:, feat_idx], bins) - 1

    # Create a composite class from feature bins
    y_train_class = y_train_binned[:, 0] * (n_bins**3) + y_train_binned[:, 1] * (n_bins**2) + \
                    y_train_binned[:, 2] * n_bins + y_train_binned[:, 3]
    y_test_class = y_test_binned[:, 0] * (n_bins**3) + y_test_binned[:, 1] * (n_bins**2) + \
                   y_test_binned[:, 2] * n_bins + y_test_binned[:, 3]

    # Train classifier
    clf = MixtureOfGaussiansFuzzyClassifier(
        top_n=3, n_gaussians=2, log_transform=False, random_state=42
    )
    clf.fit(X_train, y_train_class)

    # Predict
    y_pred_class = clf.predict(X_test)

    # Decode predictions back to feature values
    y_pred_decoded = np.zeros((len(y_pred_class), len(OUTPUT_FEATURES)))
    for i, cls in enumerate(y_pred_class):
        y_pred_decoded[i, 0] = cls // (n_bins**3)
        y_pred_decoded[i, 1] = (cls % (n_bins**3)) // (n_bins**2)
        y_pred_decoded[i, 2] = (cls % (n_bins**2)) // n_bins
        y_pred_decoded[i, 3] = cls % n_bins

    # Calculate MSE and MAE for each feature
    mse_per_feature = []
    mae_per_feature = []

    for feat_idx in range(len(OUTPUT_FEATURES)):
        mse = mean_squared_error(y_test_binned[:, feat_idx], y_pred_decoded[:, feat_idx])
        mae = mean_absolute_error(y_test_binned[:, feat_idx], y_pred_decoded[:, feat_idx])
        mse_per_feature.append(mse)
        mae_per_feature.append(mae)
        print(f"\n{OUTPUT_FEATURES[feat_idx]}:")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")

    # Overall accuracy (bin prediction)
    accuracy = np.mean(y_pred_class == y_test_class)
    print(f"\nComposite class accuracy: {accuracy:.4f}")

    return {
        'model_type': 'single_step',
        'classifier': clf,
        'mse_per_feature': mse_per_feature,
        'mae_per_feature': mae_per_feature,
        'accuracy': accuracy,
        'n_test_samples': len(X_test),
    }


def train_and_evaluate_window(X, y, window_size=3, test_size=0.2):
    """
    Train classifier for multi-step prediction using sliding window.

    Args:
        X: features (windowed state history)
        y: targets (next state)
        window_size: number of past timesteps used
        test_size: fraction of data for testing

    Returns:
        dict with evaluation metrics
    """
    print("\n" + "="*60)
    print(f"MULTI-STEP WINDOW PREDICTION MODEL (window_size={window_size})")
    print("="*60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Discretize targets
    n_bins = N_BINS

    y_train_binned = np.zeros((len(y_train), len(OUTPUT_FEATURES)), dtype=int)
    y_test_binned = np.zeros((len(y_test), len(OUTPUT_FEATURES)), dtype=int)

    for feat_idx, feat_name in enumerate(OUTPUT_FEATURES):
        bins = np.percentile(y_train[:, feat_idx], np.linspace(0, 100, n_bins+1))
        y_train_binned[:, feat_idx] = np.digitize(y_train[:, feat_idx], bins) - 1
        y_test_binned[:, feat_idx] = np.digitize(y_test[:, feat_idx], bins) - 1

    # Composite class
    y_train_class = y_train_binned[:, 0] * (n_bins**3) + y_train_binned[:, 1] * (n_bins**2) + \
                    y_train_binned[:, 2] * n_bins + y_train_binned[:, 3]
    y_test_class = y_test_binned[:, 0] * (n_bins**3) + y_test_binned[:, 1] * (n_bins**2) + \
                   y_test_binned[:, 2] * n_bins + y_test_binned[:, 3]

    # Train classifier
    clf = MixtureOfGaussiansFuzzyClassifier(
        top_n=5, n_gaussians=3, log_transform=False, random_state=42
    )
    clf.fit(X_train, y_train_class)

    # Predict
    y_pred_class = clf.predict(X_test)

    # Decode predictions
    y_pred_decoded = np.zeros((len(y_pred_class), len(OUTPUT_FEATURES)))
    for i, cls in enumerate(y_pred_class):
        y_pred_decoded[i, 0] = cls // (n_bins**3)
        y_pred_decoded[i, 1] = (cls % (n_bins**3)) // (n_bins**2)
        y_pred_decoded[i, 2] = (cls % (n_bins**2)) // n_bins
        y_pred_decoded[i, 3] = cls % n_bins

    # Calculate metrics
    mse_per_feature = []
    mae_per_feature = []

    for feat_idx in range(len(OUTPUT_FEATURES)):
        mse = mean_squared_error(y_test_binned[:, feat_idx], y_pred_decoded[:, feat_idx])
        mae = mean_absolute_error(y_test_binned[:, feat_idx], y_pred_decoded[:, feat_idx])
        mse_per_feature.append(mse)
        mae_per_feature.append(mae)
        print(f"\n{OUTPUT_FEATURES[feat_idx]}:")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")

    accuracy = np.mean(y_pred_class == y_test_class)
    print(f"\nComposite class accuracy: {accuracy:.4f}")

    return {
        'model_type': 'multi_window',
        'window_size': window_size,
        'classifier': clf,
        'mse_per_feature': mse_per_feature,
        'mae_per_feature': mae_per_feature,
        'accuracy': accuracy,
        'n_test_samples': len(X_test),
    }


def test_double_pendulum_fuzzy_prediction():
    """
    Main test: simulate double pendulum and train fuzzy predictive models.
    """
    # Setup
    test_dir = Path(__file__).parent
    data_dir = test_dir / "double_pendulum_data"

    # Step 1-3: Generate simulation data
    # Check if data already exists
    if (data_dir / "simulation_0000.csv").exists():
        print(f"Simulation data already exists in {data_dir}, skipping generation")
    else:
        generate_simulation_data(
            data_dir, num_simulations=15, duration=3.0, dt=0.01
        )

    # Step 4: Single-step prediction
    print("\n" + "#"*60)
    print("# STEP 4: Single-Step Prediction Model")
    print("#"*60)
    X_single, y_single = load_and_prepare_data(data_dir, window_size=1)
    results_single = train_and_evaluate_single_step(X_single, y_single)

    # Step 5: Multi-step window prediction
    print("\n" + "#"*60)
    print("# STEP 5: Multi-Step Window Prediction Model")
    print("#"*60)
    X_window, y_window = load_and_prepare_data(data_dir, window_size=3)
    results_window = train_and_evaluate_window(X_window, y_window, window_size=3)

    # Step 6: Summary evaluation
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print("\nSingle-Step Model:")
    print(f"  Accuracy: {results_single['accuracy']:.4f}")
    print(f"  Mean MSE: {np.mean(results_single['mse_per_feature']):.4f}")
    print(f"  Mean MAE: {np.mean(results_single['mae_per_feature']):.4f}")

    print("\nMulti-Step Window Model:")
    print(f"  Accuracy: {results_window['accuracy']:.4f}")
    print(f"  Mean MSE: {np.mean(results_window['mse_per_feature']):.4f}")
    print(f"  Mean MAE: {np.mean(results_window['mae_per_feature']):.4f}")

    print("\nComparison:")
    if results_single['accuracy'] > results_window['accuracy']:
        print("  Single-step model shows better accuracy")
    else:
        print("  Multi-step window model shows better accuracy")

    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_double_pendulum_fuzzy_prediction()
