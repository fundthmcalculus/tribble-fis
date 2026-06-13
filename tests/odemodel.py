from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.integrate import odeint


class OdeSystem(ABC):
    """Base class for ODE systems with common simulation functionality."""

    @property
    def state_dim(self) -> int:
        """Dimension of the state vector."""
        return len(self.state_labels)

    @property
    @abstractmethod
    def state_labels(self) -> list[str]:
        """Labels for each component of the state vector."""
        pass

    @property
    @abstractmethod
    def derivative_labels(self) -> list[str]:
        """Labels for each component of the state vector."""
        pass

    @abstractmethod
    def equations_of_motion(self, state, t):
        """
        Compute derivatives of the state.

        Args:
            state: Current state vector
            t: Current time

        Returns:
            List of state derivatives [d/dt state[0], d/dt state[1], ...]
        """
        pass

    @abstractmethod
    def animate(self, axes, state, t):
        """
        Create animation of the system on the given axes.
        Args:
            axes: Axes to create animation on
            state: Current state vector
            t: Current time
        """
        pass

    def simulate(self, state0, duration=10.0, dt=0.001, include_derivatives: bool = False) -> pd.DataFrame:
        """
        Simulate the ODE system from initial conditions.

        Args:
            state0: Initial state vector
            duration: Total simulation time
            dt: Time step for output

        Returns:
            DataFrame with columns matching state_labels
        """
        if len(state0) != self.state_dim:
            raise ValueError(
                f"Initial state dimension {len(state0)} does not match "
                f"system dimension {self.state_dim}"
            )

        t = np.arange(0, duration, dt)
        solution = odeint(self.equations_of_motion, state0, t)

        df = pd.DataFrame(solution, columns=self.state_labels)
        if not include_derivatives:
            return df
        # Insert the other columns:
        for i_lbl, d_lbl in enumerate(self.derivative_labels):
            if d_lbl not in self.state_labels:
                df[d_lbl] = 0.0
        # TODO - This is pretty slow, let's find a faster way later.
        # Now add the relevant derivative which isn't included.
        for i_t, state in enumerate(solution):
            d_state = self.equations_of_motion(state, t[i_t])
            for i_lbl, d_lbl in enumerate(self.derivative_labels):
                df.iloc[i_t, i_lbl] = d_state[i_lbl]
        return df