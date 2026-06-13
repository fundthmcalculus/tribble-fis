from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pandas import DataFrame

from tests.ode_helpers import angles_to_xy, set_axes_style
from tests.odemodel import OdeSystem
from tribblefis.gaussian_regressor import MimoGaussianPredictor


@dataclass
class PendulumParameters:
    theta1: float
    omega1: float
    theta2: float
    omega2: float
    dt: float
    duration: float



class DoublePendulum(OdeSystem):
    """Double pendulum simulator using Lagrangian mechanics."""

    def __init__(self, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81):
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.g = g

    @property
    def state_labels(self) -> list[str]:
        return ['theta_1', 'omega_1', 'theta_2', 'omega_2']

    @property
    def derivative_labels(self) -> list[str]:
        return ["omega_1", "alpha_1", "omega_2", "alpha_2"]

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

    def animate(self, axes, state, t):
        """Animate double pendulum on the given axes."""
        theta1, omega1, theta2, omega2 = state
        l1, l2 = self.l1, self.l2

        x1 = l1 * np.sin(theta1)
        y1 = -l1 * np.cos(theta1)
        x2 = x1 + l2 * np.sin(theta2)
        y2 = y1 - l2 * np.cos(theta2)

        axes.clear()
        axes.plot([0, x1, x2], [0, y1, y2], 'o-', color='#00d4ff', lw=2.5, ms=8)
        axes.set_xlim(-2.5, 2.5)
        axes.set_ylim(-2.5, 0.5)
        axes.set_aspect('equal')
        axes.grid(True, alpha=0.2)


def test_tribble_ode():
    """Test ODE system with fuzzy regression model."""
    # 1) Create the simulation data for various initial conditions.
    X_combined, y_combined, pendulum, params, trajectories = initialize_model()

    ode_m = MimoGaussianPredictor()
    ode_m.fit(X_combined, y_combined)

    def gauss_fcn(s, t):
        s_df = pd.DataFrame([s], columns=pendulum.state_labels)
        return ode_m.predict(s_df).values.flatten()

    # 3) Roll out using odeint; fall back to Euler if odeint diverges.
    test_ic = np.array([params.theta1, params.omega1, 2.05 * np.pi / 180.0, params.omega2])
    actual_trajectory = pendulum.simulate(test_ic, duration=params.duration, dt=params.dt)
    t_span = np.arange(0, params.duration, params.dt)

    # Euler integration is more stable than odeint for learned models
    state = test_ic.copy().astype(float)
    states = [state.copy()]
    for _ in range(len(t_span) - 1):
        ds = gauss_fcn(state, 0)
        # TODO - Use 2nd order trapz method?
        state = state + ds * params.dt
        if not np.isfinite(state).all():
            print(f"Warning: Euler diverged at step {len(states)}, stopping early.")
            break
        states.append(state.copy())
    predicted = pd.DataFrame(states, columns=pendulum.state_labels)
    print(f"Euler rollout completed ({len(states)} steps).")

    print(f"Predicted trajectory shape: {predicted.shape}")

    # 4) Animate the trajectories with dark theme and error/phase plots
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor('#1a1a2e')

    ax_actual = plt.subplot(2, 2, 1)
    ax_pred = plt.subplot(2, 2, 2)
    ax_error = plt.subplot(2, 2, 3)
    ax_phase = plt.subplot(2, 2, 4)

    # Style animation axes
    for ax in [ax_actual, ax_pred]:
        set_axes_style(ax)

    # Style error and phase axes
    for ax in [ax_error, ax_phase]:
        ax.set_facecolor('#16213e')
        ax.grid(True, alpha=0.2, color='#444')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    ax_actual.set_title('Actual', color='white', fontsize=13, fontweight='bold')
    ax_pred.set_title('Predicted', color='white', fontsize=13, fontweight='bold')
    ax_error.set_title('Running Error', color='white', fontsize=11, fontweight='bold')
    ax_phase.set_title('Phase Space (θ₂ vs ω₂)', color='white', fontsize=11, fontweight='bold')

    ax_error.set_xlabel('Time (s)', color='white', fontsize=10)
    ax_error.set_ylabel('Position Error', color='white', fontsize=10)
    ax_phase.set_xlabel('θ₂ (rad)', color='white', fontsize=10)
    ax_phase.set_ylabel('ω₂ (rad/s)', color='white', fontsize=10)

    fig.suptitle('Double Pendulum: Actual vs Predicted', color='white', fontsize=14, fontweight='bold')

    n_frames = min(len(actual_trajectory), len(predicted), 300)
    step = max(1, len(actual_trajectory) // n_frames)
    trail_len = 40

    # Pre-create line objects for actual trajectory
    act_trail2, = ax_actual.plot([], [], '-', color='#00d4ff', linewidth=1.2, alpha=0.5)
    act_rod1, = ax_actual.plot([], [], 'o-', color='#e0e0e0', lw=2.5, ms=6, markerfacecolor='white')
    act_rod2, = ax_actual.plot([], [], 'o-', color='#e0e0e0', lw=2.5, ms=6, markerfacecolor='#00d4ff')

    # Pre-create line objects for predicted trajectory
    pred_trail2, = ax_pred.plot([], [], '-', color='#ff6b6b', linewidth=1.2, alpha=0.5)
    pred_rod1, = ax_pred.plot([], [], 'o-', color='#e0e0e0', lw=2.5, ms=6, markerfacecolor='white')
    pred_rod2, = ax_pred.plot([], [], 'o-', color='#e0e0e0', lw=2.5, ms=6, markerfacecolor='#ff6b6b')

    # Pre-create line objects for error and phase plots
    err_line, = ax_error.plot([], [], color='#ff6b6b', linewidth=2)
    phase_act, = ax_phase.plot([], [], color='#00d4ff', linewidth=1.5, label='Actual', marker='o', markersize=3)
    phase_pred, = ax_phase.plot([], [], color='#ff6b6b', linewidth=1.5, label='Predicted', marker='s', markersize=3)
    ax_phase.legend(loc='upper right', fontsize=9, framealpha=0.9)

    def update(frame):
        idx = frame * step
        t_start = max(0, idx - trail_len)

        # Actual trajectory
        theta1_act = actual_trajectory.iloc[t_start:idx+1, 0].values
        theta2_act = actual_trajectory.iloc[t_start:idx+1, 2].values
        x1_act, y1_act, x2_act, y2_act = angles_to_xy(theta1_act, theta2_act, pendulum.l1, pendulum.l2)

        act_trail2.set_data(x2_act, y2_act)
        act_rod1.set_data([0, x1_act[-1]], [0, y1_act[-1]])
        act_rod2.set_data([x1_act[-1], x2_act[-1]], [y1_act[-1], y2_act[-1]])

        # Predicted trajectory
        idx_pred = min(idx, len(predicted) - 1)
        t_start_pred = max(0, idx_pred - trail_len)
        theta1_pred = predicted.iloc[t_start_pred:idx_pred+1, 0].values
        theta2_pred = predicted.iloc[t_start_pred:idx_pred+1, 2].values
        x1_pred, y1_pred, x2_pred, y2_pred = angles_to_xy(theta1_pred, theta2_pred, pendulum.l1, pendulum.l2)

        pred_trail2.set_data(x2_pred, y2_pred)
        pred_rod1.set_data([0, x1_pred[-1]], [0, y1_pred[-1]])
        pred_rod2.set_data([x1_pred[-1], x2_pred[-1]], [y1_pred[-1], y2_pred[-1]])

        # Running error: position error between actual and predicted
        time_vec = np.arange(idx + 1) * params.dt
        act_pos = np.sqrt(
            actual_trajectory.iloc[:idx+1, 0].values**2 +
            actual_trajectory.iloc[:idx+1, 2].values**2
        )
        pred_pos = np.sqrt(
            predicted.iloc[:idx+1, 0].values**2 +
            predicted.iloc[:idx+1, 2].values**2
        )
        error = np.abs(act_pos - pred_pos)
        err_line.set_data(time_vec, error)
        ax_error.set_xlim(0, max(params.duration, max(time_vec) * 1.05))
        err_max = max(error) if np.any(error) else 1
        ax_error.set_ylim(0, max(err_max * 1.1, 0.1))

        # Phase space: theta2 vs omega2
        phase_act.set_data(
            actual_trajectory.iloc[:idx+1, 2].values,
            actual_trajectory.iloc[:idx+1, 3].values
        )
        phase_pred.set_data(
            predicted.iloc[:idx+1, 2].values,
            predicted.iloc[:idx+1, 3].values
        )
        ax_phase.set_xlim(-4, 4)
        ax_phase.set_ylim(-6, 6)

        return act_trail2, act_rod1, act_rod2, pred_trail2, pred_rod1, pred_rod2, err_line, phase_act, phase_pred

    n_display_frames = n_frames // step
    ani = animation.FuncAnimation(
        fig, update, frames=n_display_frames, interval=50, blit=True, repeat=True
    )

    plt.tight_layout()

    # Save animation to file
    try:
        from pathlib import Path
        output_dir = Path(__file__).parent
        output_file = output_dir / "tribble_ode_animation.gif"
        writer = animation.PillowWriter(fps=20)
        ani.save(str(output_file), writer=writer)
        print(f"Animation saved to: {output_file}")
    except Exception as e:
        print(f"Warning: Could not save animation: {e}")

    plt.close(fig)
    print("Test completed successfully!")


def initialize_model() -> tuple[DataFrame, DataFrame, DoublePendulum, PendulumParameters, list[DataFrame]]:
    pendulum = DoublePendulum()
    trajectories = []
    theta2s = np.arange(1.5, 3.00001, 0.25)  # TODO - 0.1
    params = PendulumParameters(theta1=120 * np.pi / 180,
                                omega1=0.0,
                                omega2=0.0,
                                dt=0.01,
                                duration=3.001,
                                theta2=0.0)
    for ij in range(len(theta2s)):
        theta2 = theta2s[ij]
        theta2 *= np.pi / 180
        ic = tuple([params.theta1, params.omega1, theta2, params.omega2])
        df = pendulum.simulate(ic, duration=params.duration, dt=params.dt)
        trajectories.append(df)

    print(f"Generated {len(trajectories)} trajectories")
    print(f"First trajectory shape: {trajectories[0].shape}")

    # 2) Use MimoRegressor to fit the model (placeholder for integration).
    # This would use MimoGaussianPredictor after combining trajectory data
    X_combined = pd.concat([t.iloc[:-1] for t in trajectories], ignore_index=True)
    # 1st order integration
    y_combined = pd.concat([t.diff().iloc[1:] / params.dt for t in trajectories], ignore_index=True)

    print(f"Training data shape: X={X_combined.shape}, y={y_combined.shape}")
    return X_combined, y_combined, pendulum, params, trajectories


if __name__ == "__main__":
    test_tribble_ode()