"""Proceso SF-Harris: x_t | x_{t-1} ~ (1 - e^{-alpha}) Q + e^{-alpha} delta_{x_{t-1}}.

Con prob. 1 - e^{-alpha} se regenera de Q; si no, se queda igual.
Q continua: x_t != x_{t-1} <=> salto. Q discreta: puede haber saltos ocultos.
"""
import numpy as np
from numpy.typing import NDArray
from typing import Callable


class SFHarrisProcess:
    """Simulador del proceso SF-Harris.

    alpha: parámetro de regeneración (alpha=0 => nunca se mueve).
    Q_sample: rng -> valor de Q. Q_density: densidad (o PMF) de Q.
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
        # P(no salto en un paso) = e^{-alpha}
        return np.exp(-self.alpha)

    @property
    def jump_prob(self) -> float:
        # P(al menos un salto) = 1 - e^{-alpha}
        return 1.0 - np.exp(-self.alpha)

    def simulate(self, k: int, rng: np.random.Generator | None = None) -> NDArray[np.float64]:
        """Simula k observaciones del proceso."""
        if rng is None:
            rng = np.random.default_rng()

        obs = np.empty(k)
        obs[0] = self.Q_sample(rng)  # x_0 ~ Q

        # paso a paso: con prob. e^{-alpha} se queda, si no salta
        p_stay = self.stay_prob
        for t in range(1, k):
            if rng.uniform() < p_stay:
                obs[t] = obs[t - 1]
            else:
                obs[t] = self.Q_sample(rng)

        return obs

    def count_transitions(self, obs: NDArray[np.float64]) -> tuple[int, int]:
        """Cuenta pares (x_t == x_{t-1}) y (x_t != x_{t-1}); regresa (n_stay, n_change)."""
        changes = obs[1:] != obs[:-1]
        n_change = int(np.sum(changes))
        n_stay = len(obs) - 1 - n_change
        return n_stay, n_change

    def log_likelihood_continuous(
        self,
        obs: NDArray[np.float64],
        alpha: float | None = None,
    ) -> float:
        """Log-verosimilitud con Q continua (x_t = x_{t-1} <=> no hubo salto).

        L = q(x_1) * prod_{t: igual} e^{-alpha} * prod_{t: dif} (1-e^{-alpha}) q(x_t)
        """
        if self.Q_density is None:
            raise ValueError("Q_density required for likelihood computation")

        a = alpha if alpha is not None else self.alpha
        n_stay, n_change = self.count_transitions(obs)

        # estancias: n_stay * log(e^{-alpha}); saltos: n_change * log(1 - e^{-alpha})
        # más densidad de Q en x_0 y en cada punto de cambio
        ll = -n_stay * a + n_change * np.log(1.0 - np.exp(-a))

        ll += np.log(max(self.Q_density(obs[0]), 1e-300))  # x_0

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
        """Log-verosimilitud con Q discreta (puede haber saltos ocultos).

        P(observar igual) = e^{-alpha} + (1-e^{-alpha}) * P_Q(x_t = x_{t-1})
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
                # P(igual) = e^{-alpha} + (1-e^{-alpha}) * P_Q(mismo valor)
                p_same = p_stay + p_jump * self.Q_pmf_same
                ll += np.log(max(p_same, 1e-300))
            else:
                # P(diferente, x_t) = (1-e^{-alpha}) * P_Q(x_t)
                ll += np.log(max(p_jump * q_t, 1e-300))

        return ll