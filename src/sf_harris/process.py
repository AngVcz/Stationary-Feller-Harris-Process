"""SF-Harris process: simulation and transition dynamics.

The SF-Harris process is a discrete-time Markov chain with:
  x_t | x_{t-1} ~ (1 - e^{-alpha}) Q + e^{-alpha} delta_{x_{t-1}}

At each step, with probability (1 - e^{-alpha}), the process regenerates
from invariant distribution Q. Otherwise it stays at its current value.

For continuous Q, observing a change (x_t != x_{t-1}) implies a jump.
For discrete Q, jumps can be hidden (process jumps but draws same value).
"""
import numpy as np
from numpy.typing import NDArray
from typing import Callable


class SFHarrisProcess:
    """SF-Harris process simulator and likelihood computation.

    Args:
        alpha: Dependence/regeneration parameter. Higher alpha means more
            frequent jumps. alpha=0 means the process never moves.
        Q_sample: Function(rng) -> value that draws from invariant dist Q.
        Q_density: Function(value) -> float computing Q density (or PMF).
            Required for likelihood-based estimation methods.
    """

    def __init__(
        self,
        alpha: float,
        Q_sample: Callable,
        Q_density: Callable | None = None,
        Q_pmf_same: float | None = None,
    ):
        self.alpha = alpha
        self.Q_sample = Q_sample
        self.Q_density = Q_density
        self.Q_pmf_same = Q_pmf_same  # P_Q(x_t = x_{t-1}), for discrete Q

    @property
    def stay_prob(self) -> float:
        """P(no jump in one time step) = e^{-alpha}."""
        return np.exp(-self.alpha)

    @property
    def jump_prob(self) -> float:
        """P(at least one jump in one time step) = 1 - e^{-alpha}."""
        return 1.0 - np.exp(-self.alpha)

    def simulate(self, k: int, rng: np.random.Generator | None = None) -> NDArray[np.float64]:
        """Simulate k observations from the SF-Harris process.

        Args:
            k: Number of observations.
            rng: Random number generator for reproducibility.

        Returns:
            Array of shape (k,) with the simulated observations.
        """
        if rng is None:
            rng = np.random.default_rng()

        obs = np.empty(k)
        obs[0] = self.Q_sample(rng)

        p_stay = self.stay_prob
        for t in range(1, k):
            if rng.uniform() < p_stay:
                obs[t] = obs[t - 1]
            else:
                obs[t] = self.Q_sample(rng)

        return obs

    def count_transitions(self, obs: NDArray[np.float64]) -> tuple[int, int]:
        """Count stays and changes in an observation sequence.

        Args:
            obs: Observation sequence of length k.

        Returns:
            (n_stay, n_change): number of consecutive pairs where
                x_t == x_{t-1} and x_t != x_{t-1}.
        """
        changes = obs[1:] != obs[:-1]
        n_change = int(np.sum(changes))
        n_stay = len(obs) - 1 - n_change
        return n_stay, n_change

    def log_likelihood_continuous(
        self,
        obs: NDArray[np.float64],
        alpha: float | None = None,
    ) -> float:
        """Log-likelihood for continuous Q (e.g., GIG).

        For continuous Q, x_t = x_{t-1} only occurs when there's no jump.
        The likelihood decomposes as:
            L = q(x_1) * prod_{t: same} e^{-alpha} * prod_{t: diff} (1-e^{-alpha}) * q(x_t)

        Args:
            obs: Observation sequence.
            alpha: Override alpha for likelihood evaluation. Uses self.alpha if None.

        Returns:
            Log-likelihood value.
        """
        if self.Q_density is None:
            raise ValueError("Q_density required for likelihood computation")

        a = alpha if alpha is not None else self.alpha
        n_stay, n_change = self.count_transitions(obs)

        # Contribution from stays: n_stay * log(e^{-alpha}) = -n_stay * alpha
        # Contribution from changes: n_change * log(1 - e^{-alpha})
        # Plus Q density at each change point and initial value
        ll = -n_stay * a + n_change * np.log(1.0 - np.exp(-a))

        # Initial observation
        ll += np.log(max(self.Q_density(obs[0]), 1e-300))

        # Change observations
        changes = obs[1:] != obs[:-1]
        change_idx = np.where(changes)[0] + 1  # indices of x_t where x_t != x_{t-1}
        for i in change_idx:
            ll += np.log(max(self.Q_density(obs[i]), 1e-300))

        return ll

    def log_likelihood_discrete(
        self,
        obs: NDArray[np.float64],
        alpha: float | None = None,
    ) -> float:
        """Log-likelihood for discrete Q (e.g., Uniform{1,...,5}).

        For discrete Q, hidden jumps are possible (jump occurs but lands on
        same value). P(observe same) = e^{-alpha} + (1-e^{-alpha}) * P_Q(x_t = x_{t-1}).

        Args:
            obs: Observation sequence.
            alpha: Override alpha for likelihood evaluation.

        Returns:
            Log-likelihood value.
        """
        if self.Q_density is None:
            raise ValueError("Q_density (PMF) required for likelihood computation")
        if self.Q_pmf_same is None:
            raise ValueError("Q_pmf_same required for discrete likelihood")

        a = alpha if alpha is not None else self.alpha
        p_jump = 1.0 - np.exp(-a)
        p_stay = np.exp(-a)

        ll = np.log(max(self.Q_density(obs[0]), 1e-300))

        for t in range(1, len(obs)):
            q_t = self.Q_density(obs[t])
            if np.isclose(obs[t], obs[t - 1]):
                # P(x_t = x_{t-1}) = e^{-alpha} + (1-e^{-alpha}) * P_Q(same)
                p_same = p_stay + p_jump * self.Q_pmf_same
                ll += np.log(max(p_same, 1e-300))
            else:
                # P(x_t != x_{t-1}, x_t) = (1-e^{-alpha}) * P_Q(x_t)
                ll += np.log(max(p_jump * q_t, 1e-300))

        return ll