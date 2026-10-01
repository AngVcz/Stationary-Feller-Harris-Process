"""Distribuciones invariantes del proceso SF-Harris: Uniforme Discreta y GIG."""
import numpy as np
from numpy.typing import NDArray
from scipy.special import kv as bessel_kv


class DiscreteUniformQ:
    """Uniforme discreta en {1, ..., m}: P(X = i) = 1/m."""

    def __init__(self, m: int = 5):
        self.m = m
        self.values = np.arange(1, m + 1, dtype=np.float64)

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.choice(self.values))

    def pmf(self, x: float) -> float:
        xi = int(round(x))
        if 1 <= xi <= self.m:
            return 1.0 / self.m
        return 0.0

    def pmf_array(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        result = np.zeros_like(x)
        mask = (x >= 1) & (x <= self.m) & (np.abs(x - np.round(x)) < 0.5)
        result[mask] = 1.0 / self.m
        return result

    @property
    def pmf_same(self) -> float:
        # P_Q(X_t = X_{t-1}) = 1/m (uniforme discreta)
        return 1.0 / self.m


class GIGQ:
    """GIG: f(x) = eta^lam / (2 K_lam(kappa)) * x^(lam-1) * exp{-(kappa/2)(eta x + 1/(eta x))}.

    Parámetros estándar: chi = kappa/eta, psi = kappa*eta.
    Muestreo: X = Z/eta con Z ~ GIG(lam, 1, kappa^2) (scipy).
    """

    def __init__(self, lam: float, kappa: float, eta: float):
        if kappa <= 0:
            raise ValueError(f"kappa must be > 0, got {kappa}")
        if eta <= 0:
            raise ValueError(f"eta must be > 0, got {eta}")
        self.lam = lam
        self.kappa = kappa
        self.eta = eta

    def sample(self, rng: np.random.Generator) -> float:
        from scipy.stats import geninvgauss
        # cambio de variable: Z ~ geninvgauss(lam, kappa), X = Z/eta
        # => densidad GIG(lam, kappa, eta) con el mismo K_lam(kappa)
        z = geninvgauss.rvs(self.lam, self.kappa, random_state=rng)
        return float(z / self.eta)

    def sample_n(self, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
        from scipy.stats import geninvgauss
        z = geninvgauss.rvs(self.lam, self.kappa, size=n, random_state=rng)
        return z / self.eta

    def density(self, x: float) -> float:
        if x <= 0:
            return 0.0
        log_f = (
            self.lam * np.log(self.eta)
            + (self.lam - 1) * np.log(x)
            - (self.kappa / 2) * (self.eta * x + 1.0 / (self.eta * x))
            - np.log(2) - np.log(bessel_kv(self.lam, self.kappa))
        )
        return np.exp(log_f)

    def density_array(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        result = np.zeros_like(x)
        mask = x > 0
        log_f = (
            self.lam * np.log(self.eta)
            + (self.lam - 1) * np.log(x[mask])
            - (self.kappa / 2) * (self.eta * x[mask] + 1.0 / (self.eta * x[mask]))
            - np.log(2) - np.log(bessel_kv(self.lam, self.kappa))
        )
        result[mask] = np.exp(log_f)
        return result

    def log_density(self, x: float) -> float:
        if x <= 0:
            return -np.inf
        return (
            self.lam * np.log(self.eta)
            + (self.lam - 1) * np.log(x)
            - (self.kappa / 2) * (self.eta * x + 1.0 / (self.eta * x))
            - np.log(2) - np.log(bessel_kv(self.lam, self.kappa))
        )

    def kl_divergence(self, other: "GIGQ") -> float:
        """KL(GIG(self) || GIG(other)), fórmula analítica en (chi, psi)."""
        from scipy.special import kv as bessel_kv

        # chi = kappa*eta, psi = kappa/eta (parámetros estándar de GIG)
        chi_p = self.kappa * self.eta
        psi_p = self.kappa / self.eta
        chi_q = other.kappa * other.eta
        psi_q = other.kappa / other.eta
        omega_p = self.kappa
        omega_q = other.kappa

        try:
            K_p = bessel_kv(self.lam, omega_p)
            K_q = bessel_kv(other.lam, omega_q)
            if K_p <= 0 or K_q <= 0 or not np.isfinite(K_p) or not np.isfinite(K_q):
                return self._kl_divergence_mc(other)

            E_X = np.sqrt(psi_p / chi_p) * bessel_kv(self.lam + 1, omega_p) / K_p
            E_invX = np.sqrt(chi_p / psi_p) * bessel_kv(self.lam - 1, omega_p) / K_p

            # E[log X] via numerical derivative of log K_lam(omega)
            eps = 1e-5
            Kp_plus = bessel_kv(self.lam + eps, omega_p)
            Kp_minus = bessel_kv(self.lam - eps, omega_p)
            if Kp_plus <= 0 or Kp_minus <= 0:
                return self._kl_divergence_mc(other)
            E_logX = (np.log(Kp_plus) - np.log(Kp_minus)) / (2 * eps) + 0.5 * np.log(psi_p / chi_p)

            kl = ((self.lam / 2) * np.log(chi_p / psi_p)
                  - (other.lam / 2) * np.log(chi_q / psi_q)
                  - np.log(K_p) + np.log(K_q)
                  + (self.lam - other.lam) * E_logX
                  - (chi_p - chi_q) / 2 * E_X
                  - (psi_p - psi_q) / 2 * E_invX)

            return max(kl, 0.0) if np.isfinite(kl) else self._kl_divergence_mc(other)
        except (ValueError, OverflowError):
            return self._kl_divergence_mc(other)

    def _kl_divergence_mc(self, other: "GIGQ", n_samples: int = 50000) -> float:
        # fallback Monte Carlo para la KL (cuando la fórmula analítica falla)
        rng = np.random.default_rng(42)
        samples = self.sample_n(n_samples, rng)
        log_ratios = np.array([
            self.log_density(x) - other.log_density(x)
            for x in samples if x > 0
        ])
        log_ratios = log_ratios[np.isfinite(log_ratios)]
        if len(log_ratios) == 0:
            return 10.0
        return max(float(np.mean(log_ratios)), 0.0)

    def _mode(self) -> float:
        # moda: raíz de f'(x) = 0 => x* = ((lam-1) + sqrt((lam-1)^2 + kappa^2)) / (kappa*eta)
        discriminant = (self.lam - 1) ** 2 + self.kappa ** 2
        if discriminant < 0:
            return 1.0 / self.eta
        return ((self.lam - 1) + np.sqrt(discriminant)) / (self.kappa * self.eta)