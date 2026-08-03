"""State-space model simulation: AR(1) process and linear Gaussian SSM."""
import numpy as np
from numpy.typing import NDArray


class AR1Process:
    """AR(1) process: X_t = phi * X_{t-1} + sigma * epsilon_t."""

    def __init__(self, phi: float, sigma: float, stationary: bool = True):
        self.phi = phi
        self.sigma = sigma
        if stationary and abs(phi) >= 1:
            raise ValueError(
                f"|phi| >= 1 ({phi}) is not stationary. Set stationary=False to allow."
            )

    def simulate(
        self, n_steps: int, x0: float = 0.0, rng: np.random.Generator | None = None
    ) -> NDArray[np.float64]:
        """Simulate AR(1) trajectory.

        Args:
            n_steps: Number of time steps to simulate.
            x0: Initial state.
            rng: Random number generator for reproducibility.

        Returns:
            Array of shape (n_steps,) with the simulated states.
        """
        if rng is None:
            rng = np.random.default_rng()
        states = np.empty(n_steps)
        states[0] = x0
        noise = rng.normal(0, self.sigma, size=n_steps)
        for t in range(1, n_steps):
            states[t] = self.phi * states[t - 1] + noise[t]
        return states

    def transition_density(self, x_prev: float, x_curr: float) -> float:
        """Compute transition density p(x_curr | x_prev)."""
        mean = self.phi * x_prev
        return (1.0 / (self.sigma * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x_curr - mean) / self.sigma) ** 2
        )


class LinearGaussianSSM:
    """Linear Gaussian state-space model.

    X_t = phi * X_{t-1} + sigma_v * v_t    (state transition)
    Y_t = X_t + sigma_w * w_t               (observation)
    """

    def __init__(
        self,
        transition_phi: float,
        transition_sigma: float,
        observation_sigma: float,
    ):
        self.transition = AR1Process(phi=transition_phi, sigma=transition_sigma)
        self.observation_sigma = observation_sigma

    def simulate(
        self,
        n_steps: int,
        x0: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Simulate states and observations from the SSM.

        Returns:
            Tuple of (states, observations), each of length n_steps.
        """
        if rng is None:
            rng = np.random.default_rng()
        states = self.transition.simulate(n_steps, x0, rng)
        obs_noise = rng.normal(0, self.observation_sigma, size=n_steps)
        observations = states + obs_noise
        return states, observations