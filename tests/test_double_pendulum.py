"""
Test double pendulum simulation with exponential integrator + FIS prediction.

Uses exponential integration to separate linear from nonlinear dynamics.
The linear component is integrated exactly, while the nonlinear residual
is learned via fuzzy inference system (FIS). Combines exact linear dynamics
with learned nonlinear corrections for improved long-term accuracy.
"""
import sys
import time
from argparse import ArgumentError
from collections import namedtuple
from contextlib import contextmanager
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import expm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tests.ode_helpers import load_and_prepare_data, train_and_evaluate_single_step, set_axes_style, angles_to_xy
from tests.test_fuzzy_ode import initialize_model, DoublePendulum

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tribblefis.gaussian_regressor import MimoGaussianPredictor

FeatureStep = namedtuple('FeatureStep', ['step_name', 'step_offset', 'col_name'])


def linearized_jacobian(pendulum: DoublePendulum, state: np.ndarray) -> np.ndarray:
    """
    Compute the Jacobian (first-order linearization) of the double pendulum.

    Uses small-angle approximation around equilibrium: sin(x) ≈ x, cos(x) ≈ 1
    This gives the dominant linear restoring forces and coupling.

    Returns the 4x4 Jacobian matrix for the system [θ₁, ω₁, θ₂, ω₂].
    """
    g = pendulum.g
    l1 = pendulum.l1
    l2 = pendulum.l2
    m1 = pendulum.m1
    m2 = pendulum.m2

    J = np.zeros((4, 4))

    # Kinematic terms: dθ/dt = ω
    J[0, 1] = 1.0
    J[2, 3] = 1.0

    # Dynamic terms under small-angle approximation
    # α₁ ≈ -g(2m₁+m₂)/(l₁(2m₁+m₂)) * θ₁ - gm₂/(l₁(2m₁+m₂)) * θ₂
    # α₂ ≈ -g(m₁+m₂)/(l₂(2m₁+m₂)) * θ₂ + g(m₁+m₂)/(l₂(2m₁+m₂)) * θ₁

    denom1 = l1 * (2*m1 + m2)
    denom2 = l2 * (2*m1 + m2)

    J[1, 0] = -g * (2*m1 + m2) / denom1      # ∂α₁/∂θ₁
    J[1, 2] = -g * m2 / denom1              # ∂α₁/∂θ₂

    J[3, 0] = g * (m1 + m2) / denom2        # ∂α₂/∂θ₁
    J[3, 2] = -g * (m1 + m2) / denom2       # ∂α₂/∂θ₂

    return J


def exponential_step(pendulum: DoublePendulum, state: np.ndarray, dt: float) -> np.ndarray:
    """
    Integrate one step using exponential integrator on linearized dynamics.

    Computes: state_new ≈ exp(J*dt) * state
    where J is the Jacobian of the full nonlinear system at the current state.
    """
    J = linearized_jacobian(pendulum, state)
    exp_J_dt = expm(J * dt)
    return exp_J_dt @ state


def compute_nonlinear_residual(
    pendulum: DoublePendulum, state: np.ndarray, full_derivative: np.ndarray, dt: float
) -> np.ndarray:
    """
    Extract the nonlinear residual from full dynamics.

    nonlinear_residual = (full_derivative) - (linear_part)
    where linear_part is approximated by (exp_J*state - state) / dt

    Args:
        pendulum: The ODE system
        state: Current state vector
        full_derivative: Full time derivative from equations_of_motion
        dt: Timestep

    Returns:
        The nonlinear residual vector
    """
    # Linearized step
    J = linearized_jacobian(pendulum, state)
    linear_derivative = J @ state

    # Nonlinear residual is the difference
    return full_derivative - linear_derivative


def compute_total_energy(pendulum: DoublePendulum, state: np.ndarray) -> float:
    """
    Compute total mechanical energy of the double pendulum.

    E_total = KE1 + KE2 + PE1 + PE2

    Args:
        pendulum: The ODE system with mass and length parameters
        state: [θ₁, ω₁, θ₂, ω₂]

    Returns:
        Total mechanical energy (in reference frame where hanging down is PE=0)
    """
    theta1, omega1, theta2, omega2 = state
    g = pendulum.g
    m1 = pendulum.m1
    m2 = pendulum.m2
    l1 = pendulum.l1
    l2 = pendulum.l2

    # Kinetic energies
    ke1 = 0.5 * m1 * (l1 * omega1) ** 2
    ke2 = 0.5 * m2 * (l2 * omega2) ** 2

    # Potential energies (taking hanging down as zero reference)
    # Height of mass 1: h1 = -l1*cos(θ1)
    # Height of mass 2: h2 = -l1*cos(θ1) - l2*cos(θ2)
    pe1 = m1 * g * l1 * (1 - np.cos(theta1))
    pe2 = m2 * g * (l1 * (1 - np.cos(theta1)) + l2 * (1 - np.cos(theta2)))

    return ke1 + ke2 + pe1 + pe2


def enforce_energy_conservation(
    pendulum: DoublePendulum,
    state_current: np.ndarray,
    state_predicted: np.ndarray,
    allow_dissipation: bool = True,
) -> tuple[np.ndarray, float]:
    """
    Enforce energy conservation by scaling velocities if necessary.

    If predicted energy exceeds current energy (physically impossible),
    scale angular velocities to match the initial energy.
    If allow_dissipation=False, scale to exactly conserve energy.

    Args:
        pendulum: The ODE system
        state_current: Current state [θ₁, ω₁, θ₂, ω₂]
        state_predicted: Predicted state [θ₁, ω₁, θ₂, ω₂]
        allow_dissipation: If True, only scale down if energy increased;
                          if False, always conserve energy exactly

    Returns:
        (corrected_state, energy_ratio): Corrected state and E_pred/E_current ratio
    """
    energy_current = compute_total_energy(pendulum, state_current)
    energy_predicted = compute_total_energy(pendulum, state_predicted)

    # If energy is conserved or dissipated, return as-is
    if energy_predicted <= energy_current and allow_dissipation:
        return state_predicted, energy_predicted / max(energy_current, 1e-10)

    # Energy increased (unphysical) - scale velocities
    # Assume potential energy components are correct (angles are right)
    theta1_pred, omega1_pred, theta2_pred, omega2_pred = state_predicted
    g = pendulum.g
    m1 = pendulum.m1
    m2 = pendulum.m2
    l1 = pendulum.l1
    l2 = pendulum.l2

    # Compute PE from predicted angles
    pe_predicted = (
        m1 * g * l1 * (1 - np.cos(theta1_pred)) +
        m2 * g * (l1 * (1 - np.cos(theta1_pred)) + l2 * (1 - np.cos(theta2_pred)))
    )

    # Available KE = E_current - PE_predicted
    available_ke = energy_current - pe_predicted

    if available_ke < 0:
        # Even with zero velocities, PE exceeds initial energy
        # This means the pendulum climbed higher than it should
        # Reset to predicted angles but zero velocities
        return np.array([theta1_pred, 0.0, theta2_pred, 0.0]), 0.0

    # Scale velocities to match available kinetic energy
    # Total KE_current = 0.5 * m1 * (l1*ω1)² + 0.5 * m2 * (l2*ω2)²
    ke_predicted = 0.5 * m1 * (l1 * omega1_pred) ** 2 + 0.5 * m2 * (l2 * omega2_pred) ** 2

    if ke_predicted > 1e-10:
        scale_factor = np.sqrt(available_ke / ke_predicted)
        omega1_corrected = omega1_pred * scale_factor
        omega2_corrected = omega2_pred * scale_factor
    else:
        omega1_corrected = 0.0
        omega2_corrected = 0.0

    state_corrected = np.array([theta1_pred, omega1_corrected, theta2_pred, omega2_corrected])
    energy_corrected = compute_total_energy(pendulum, state_corrected)

    return state_corrected, energy_corrected / max(energy_current, 1e-10)


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


def load_and_prepare_nonlinear_residual_data(
    pendulum: DoublePendulum, trajectories: list[pd.DataFrame], dt: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract nonlinear residuals from trajectories for exponential integrator approach.

    For each state transition, compute:
    - Input: current state [θ₁, ω₁, θ₂, ω₂]
    - Output: nonlinear residual in ACCELERATIONS only [α₁, α₂]
              (kinematic equations are already linear, so we skip them)

    This trains the FIS only on the nonlinear component of accelerations,
    while linear dynamics are handled exactly by exponential integration.

    Args:
        pendulum: DoublePendulum instance
        trajectories: List of state DataFrames with columns matching state_labels
        dt: Time step between samples

    Returns:
        (X_residual, y_residual): State and acceleration nonlinear residuals
    """
    print(f"Extracting nonlinear residuals from {len(trajectories)} trajectories...")

    all_X = []
    all_y = []

    for df in trajectories:
        # Get state values: [theta_1, omega_1, theta_2, omega_2]
        states = df[['theta_1', 'omega_1', 'theta_2', 'omega_2']].values

        # Compute full derivatives and extract nonlinear residuals
        for i in range(len(states) - 1):
            state = states[i]
            full_derivative = np.array(pendulum.equations_of_motion(state, i * dt))

            # Compute nonlinear residual
            residual = compute_nonlinear_residual(pendulum, state, full_derivative, dt)

            # Only keep acceleration components (indices 1 and 3)
            # Kinematic equations are already linear, so residual is zero
            accel_residual = residual[[1, 3]]  # [d_alpha_1, d_alpha_2]

            all_X.append(state)
            all_y.append(accel_residual)

    X_residual = np.vstack(all_X)
    y_residual = np.vstack(all_y)

    print(f"Nonlinear residual data (accelerations only): X={X_residual.shape}, y={y_residual.shape}")
    return X_residual, y_residual


def load_and_prepare_mimo_data(trajectories: list[pd.DataFrame], window_size: int = 1):
    """
    Load all simulation files and prepare data for MIMO full-state prediction.

    For window_size=1: input is full state at time t, output is state delta (t+1) - (t).
    For window_size>1: input is the flattened states at [t-window+1, ..., t],
                       output is the state delta (t+1) - (t).
    """
    print(f"Loading {len(trajectories)} simulation files for MIMO (window={window_size})...")

    all_X = []
    all_y = []

    for df in trajectories:
        # Output is always the state difference (next - current)
        y = np.diff(df[OUTPUT_FEATURES].values, axis=0)

        if window_size == 1:
            # Input is current state, output is next state
            X = df[INPUT_FEATURES].iloc[:-1].values
        else:
            # Input is windowed history
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


def train_and_evaluate_exp_integrator(
    pendulum: DoublePendulum, X_train, y_train, X_test, y_test
):
    """
    Train a regressor on nonlinear residuals (exponential integrator approach).

    The FIS is trained to predict only the nonlinear acceleration components,
    while linear dynamics are handled exactly via exponential integration.
    """
    print("\n" + "="*60)
    print("EXPONENTIAL INTEGRATOR + FIS MODEL")
    print("="*60)

    state_cols = ['theta_1', 'omega_1', 'theta_2', 'omega_2']
    residual_cols = ['d_alpha_1', 'd_alpha_2']  # Only accelerations (kinematic is already linear)

    regressor = MimoGaussianPredictor(
        n_output_buckets=N_BINS, tsk_order="1st", optimize_coefficients=True,
        random_state=42
    )

    X_train_df = pd.DataFrame(X_train, columns=state_cols)
    y_train_df = pd.DataFrame(y_train, columns=residual_cols)
    X_test_df = pd.DataFrame(X_test, columns=state_cols)
    y_test_df = pd.DataFrame(y_test, columns=residual_cols)

    regressor.fit(X_train_df, y_train_df)
    y_pred_df = regressor.predict(X_test_df)

    metrics = {}
    print("\nNonlinear acceleration residual prediction metrics:")
    for col in residual_cols:
        mse = mean_squared_error(y_test_df[col], y_pred_df[col])
        mae = mean_absolute_error(y_test_df[col], y_pred_df[col])
        r2 = r2_score(y_test_df[col], y_pred_df[col])
        metrics[col] = {'mse': mse, 'mae': mae, 'rmse': np.sqrt(mse), 'r2': r2}
        print(f"  {col:12s}: R2={r2:.4f}  RMSE={np.sqrt(mse):.6f}  MAE={mae:.6f}")

    return {
        'model_type': 'exp_integrator',
        'pendulum': pendulum,
        'regressor': regressor,
        'metrics': metrics,
        'y_test': y_test_df,
        'y_pred': y_pred_df,
    }


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


def run_iterative_exp_integrator_prediction(
    pendulum: DoublePendulum,
    regressor,
    initial_state: np.ndarray,
    n_steps: int,
    dt: float = 0.01,
    verbose: bool = True,
    enforce_energy: bool = True,
) -> pd.DataFrame:
    """
    Iteratively predict using exponential integrator + FIS + energy conservation.

    Combines:
    1. Exact linear integration via exponential integrator
    2. Learned nonlinear corrections to accelerations from FIS
    3. Energy conservation enforcement (prevents unphysical energy increases)

    Args:
        pendulum: The ODE system
        regressor: Trained regressor predicting acceleration nonlinear residuals
        initial_state: Initial state [θ₁, ω₁, θ₂, ω₂]
        n_steps: Number of integration steps
        dt: Time step
        verbose: Print warnings about divergence
        enforce_energy: If True, enforce energy conservation by scaling velocities

    Returns:
        DataFrame with shape (n_steps, 4) containing full state trajectory
    """
    state_labels = ['theta_1', 'omega_1', 'theta_2', 'omega_2']
    states = [initial_state.copy()]
    diverged_at = None
    energy_corrections = []

    for step in range(n_steps):
        if diverged_at:
            # Pad with NaN after divergence
            states.append(np.full(4, np.nan))
            energy_corrections.append(np.nan)
        else:
            current_state = states[-1]

            # Exponential integrator step (linear part)
            J = linearized_jacobian(pendulum, current_state)
            exp_J_dt = expm(J * dt)
            linear_prediction = exp_J_dt @ current_state

            # FIS prediction of nonlinear acceleration residuals
            state_df = pd.DataFrame([current_state], columns=state_labels)
            accel_residual_pred = regressor.predict(state_df).values.flatten()  # [d_alpha_1, d_alpha_2]

            # Apply corrections only to acceleration components (indices 1, 3)
            new_state = linear_prediction.copy()
            new_state[1] += accel_residual_pred[0] * dt  # Correct α₁
            new_state[3] += accel_residual_pred[1] * dt  # Correct α₂

            # Enforce energy conservation if enabled
            if enforce_energy:
                new_state, energy_ratio = enforce_energy_conservation(
                    pendulum, current_state, new_state, allow_dissipation=True
                )
                energy_corrections.append(energy_ratio)
            else:
                energy_corrections.append(1.0)

            # Check for divergence
            if np.any(np.isnan(new_state)) or np.any(np.abs(new_state) > 1e6):
                if diverged_at is None:
                    diverged_at = step + 1
                    if verbose:
                        print(f"  Warning: prediction diverged at step {diverged_at}")
            else:
                states.append(new_state)

    if diverged_at is not None and verbose:
        print(f"  Valid prediction steps: {diverged_at} / {n_steps}")

    # Pad energy_corrections to match states length
    # First state has no correction applied, so prepend 1.0
    energy_corrections_padded = [1.0] + energy_corrections

    # Add energy conservation statistics
    valid_ratios = np.array(energy_corrections_padded[1:diverged_at if diverged_at else len(energy_corrections_padded)])
    valid_ratios = valid_ratios[~np.isnan(valid_ratios)]
    if len(valid_ratios) > 0 and verbose:
        energy_increased = np.sum(valid_ratios > 1.001)  # Allow 0.1% tolerance
        if energy_increased > 0:
            print(f"  Energy conservation: {energy_increased} steps had unphysical energy increase (corrected)")

    df = pd.DataFrame(states, columns=state_labels)
    df['energy_ratio'] = energy_corrections_padded[:len(states)]  # Add energy ratio for reference
    return df


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
                        new_row[fs.step_name] = running_state.iloc[-1][fs.col_name] + next_state_delta_df.iloc[0][
                            fs.col_name]
                new_state = pd.DataFrame([new_row])

            # TODO - Set better bounds
            if np.any(np.isnan(new_state)) or np.any(np.abs(new_state) > 1e6):
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

    l1, l2 = 1.0, 1.0  # default lengths
    x1_act, y1_act, x2_act, y2_act = angles_to_xy(
        actual_frames['theta_1'].values, actual_frames['theta_2'].values, l1, l2
    )
    x1_pred, y1_pred, x2_pred, y2_pred = angles_to_xy(
        pred_frames['theta_1'].values, pred_frames['theta_2'].values, l1, l2
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


def test_double_pendulum_fuzzy_prediction():
    """
    Main test: simulate double pendulum and train fuzzy regression models.
    Includes exponential integrator approach + MIMO full-state prediction
    and iterative rollout with GIF animation.
    """

    # Step 1: Generate simulation data
    with time_this("gen-sim-data"):
        train_results, test_results = initialize_model()
        pendulum = train_results.model

    # Step 2: Exponential integrator + FIS (nonlinear residual)
    print("\n" + "#"*60)
    print("# STEP 2: Exponential Integrator + FIS Model")
    print("#"*60)
    with time_this('train-exp-integrator'):
        X_exp_train, y_exp_train = load_and_prepare_nonlinear_residual_data(
            pendulum, train_results.trajectories, dt=train_results.params.dt
        )
        X_exp_test, y_exp_test = load_and_prepare_nonlinear_residual_data(
            pendulum, test_results.trajectories, dt=test_results.params.dt
        )
        results_exp = train_and_evaluate_exp_integrator(
            pendulum, X_exp_train, y_exp_train, X_exp_test, y_exp_test
        )

    # Step 3: MIMO full-state prediction
    print("\n" + "#"*60)
    print(f"# STEP 3: MIMO Full-State Prediction Model (window={MIMO_WINDOW_SIZE})")
    print("#"*60)
    with time_this('train-mimo'):
        X_mimo_train, y_mimo_train = load_and_prepare_mimo_data(train_results.trajectories, window_size=MIMO_WINDOW_SIZE)
        X_mimo_test, y_mimo_test = load_and_prepare_mimo_data(test_results.trajectories, window_size=MIMO_WINDOW_SIZE)
        results_mimo = train_and_evaluate_mimo(
            X_mimo_train, y_mimo_train, X_mimo_test, y_mimo_test, window_size=MIMO_WINDOW_SIZE
        )

    # Step 4: Iterative rollout using exponential integrator with energy conservation
    print("\n" + "#"*60)
    print("# STEP 4: Iterative Rollout - Exponential Integrator + Energy Conservation")
    print("#"*60)
    tst_df = test_results.trajectories[0]
    initial_state = tst_df[['theta_1', 'omega_1', 'theta_2', 'omega_2']].iloc[0].values
    n_steps = len(tst_df) - 1

    print(f"Running {n_steps} iterative prediction steps with energy conservation...")

    with time_this('iterative-exp-integrator'):
        predicted_traj_with_energy = run_iterative_exp_integrator_prediction(
            pendulum, results_exp['regressor'], initial_state, n_steps,
            dt=test_results.params.dt, verbose=True, enforce_energy=True
        )

    # Extract energy ratio for analysis before removing it
    energy_ratios = predicted_traj_with_energy['energy_ratio'].values
    predicted_trajectory_exp = predicted_traj_with_energy[['theta_1', 'omega_1', 'theta_2', 'omega_2']]

    actual_trajectory = tst_df[['theta_1', 'omega_1', 'theta_2', 'omega_2']].reset_index(drop=True)
    n = min(len(actual_trajectory), len(predicted_trajectory_exp))

    def rollout_metrics(label, predicted, actual, n):
        print(f"\n{label}:")
        state_cols = ['theta_1', 'omega_1', 'theta_2', 'omega_2']
        for col in state_cols:
            act = actual[col].values[:n]
            pred = predicted[col].values[:n]
            valid = ~np.isnan(pred)
            if valid.sum() < 2:
                print(f"  {col:10s}: insufficient valid predictions")
                continue
            mae = np.mean(np.abs(act[valid] - pred[valid]))
            r2 = r2_score(act[valid], pred[valid])
            print(f"  {col:10s}: MAE={mae:.4f}  R2={r2:.4f}  (valid={valid.sum()}/{n})")

    rollout_metrics("Exponential Integrator + Energy Conservation Rollout", predicted_trajectory_exp, actual_trajectory, n)

    # Energy conservation analysis
    valid_energy = energy_ratios[~np.isnan(energy_ratios)]
    if len(valid_energy) > 0:
        print(f"\nEnergy Conservation Statistics:")
        print(f"  Initial energy: {compute_total_energy(pendulum, initial_state):.4f}")
        print(f"  Min energy ratio (E_pred/E_init): {np.min(valid_energy):.4f}")
        print(f"  Max energy ratio: {np.max(valid_energy):.4f}")
        print(f"  Mean energy ratio: {np.mean(valid_energy):.4f}")
        energy_increases = np.sum(energy_ratios[~np.isnan(energy_ratios)] > 1.001)
        print(f"  Steps with unphysical energy increase (>0.1%): {energy_increases}")

    # Step 5: Summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)

    print("\nExponential Integrator + FIS Model (nonlinear acceleration residual):")
    residual_cols = ['d_alpha_1', 'd_alpha_2']
    for col in residual_cols:
        m = results_exp['metrics'][col]
        print(f"  {col:12s}: R2={m['r2']:.4f}  RMSE={m['rmse']:.6f}")

    print("\nMIMO Model (full-state, 1-step):")
    output_cols = ['theta_1', 'theta_2']
    for col in output_cols:
        m = results_mimo['metrics'][col]
        print(f"  {col:10s}: R2={m['r2']:.4f}  RMSE={m['rmse']:.4f}")

    # Step 6: Plots
    print("\n" + "="*60)
    print("GENERATING VISUALIZATION PLOTS")
    print("="*60)

    print("\nPlot: Exponential Integrator Rollout State Trajectories")
    fig_exp = plot_mimo_state_trajectories(
        actual_trajectory, predicted_trajectory_exp, dt=train_results.params.dt
    )
    fig_exp.savefig("exp_integrator_trajectories.png", dpi=200, bbox_inches='tight')
    plt.close(fig_exp)

    # Step 7: GIF animation (actual vs predicted exponential integrator rollout)
    print("\n" + "="*60)
    print("GENERATING GIF ANIMATION")
    print("="*60)
    gif_path = "double_pendulum_exp_integrator.gif"
    create_pendulum_animation(
        actual_trajectory, predicted_trajectory_exp,
        gif_path, dt=train_results.params.dt, max_frames=300, fps=25
    )

    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_double_pendulum_fuzzy_prediction()
