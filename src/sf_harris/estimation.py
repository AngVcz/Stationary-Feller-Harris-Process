"""Estimation methods for the SF-Harris process.

Implements:
- NDNJ: Nonparametric density-based non-jump estimator
- MLE: Maximum likelihood estimation (alpha-only and full)
- EM: Expectation-Maximization
- Gibbs: Gibbs sampler with ARMS (basic implementation)
"""
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar, minimize

from .process import SFHarrisProcess
from .distributions import DiscreteUniformQ, GIGQ


def ndnj_estimate(obs: NDArray[np.float64], return_params: bool = False, rng: np.random.Generator | None = None):
    """NDNJ (Nonparametric Density-based Non-Jump) estimator for alpha.

    Algorithm:
    1. Find J = {j : x_j != x_{j-1}} (observed change positions)
    2. If J is empty, estimate alpha = 0
    3. Compute the raw jump probability: p_hat = n_changes / n_total
    4. Convert to alpha: alpha_hat = -log(1 - p_hat)

    The raw estimator 1/mean(inter-arrival times) estimates 1-e^{-alpha}
    (the jump probability per step). Converting via -log(1-p) gives alpha.

    For discrete Q, some jumps may be hidden (process jumps to same value),
    making this estimator biased downward.

    Args:
        obs: Observation sequence of length k.
        return_params: If True, also estimate GIG parameters using Gibbs posterior mean.
        rng: Random number generator (required when return_params=True).

    Returns:
        Estimated alpha value (float), or dict with alpha, lam, kappa, eta
        if return_params=True.
    """
    n_total = len(obs) - 1
    if n_total == 0:
        if return_params:
            return {"alpha": 0.0, "lam": 0.0, "kappa": 1.0, "eta": 1.0}
        return 0.0

    n_changes = int(np.sum(obs[1:] != obs[:-1]))

    if n_changes == 0:
        if return_params:
            return {"alpha": 0.0, "lam": 0.0, "kappa": 1.0, "eta": 1.0}
        return 0.0

    p_hat = n_changes / n_total

    # Cap p_hat to avoid log(0)
    if p_hat >= 1.0:
        alpha_hat = 50.0
    else:
        alpha_hat = -np.log(1.0 - p_hat)

    if not return_params:
        return alpha_hat

    # Estimate GIG parameters via Gibbs posterior mean (avoids MLE instability)
    gig_params = gibbs_q_posterior_mean(obs, rng=rng)
    return {"alpha": alpha_hat, **gig_params}


def mle_alpha_continuous(obs: NDArray[np.float64]) -> float:
    """MLE for alpha with continuous Q (e.g., GIG).

    For continuous Q, x_t = x_{t-1} implies no jump, and x_t != x_{t-1}
    implies a jump. The MLE is:
        alpha_hat = -log(n_stay / n_total)

    where n_stay = number of consecutive equal pairs, n_total = k-1.

    Args:
        obs: Observation sequence.

    Returns:
        MLE estimate of alpha. Returns 0.0 if no transitions.
    """
    n_total = len(obs) - 1
    if n_total == 0:
        return 0.0

    n_stay = int(np.sum(obs[1:] == obs[:-1]))
    n_change = n_total - n_stay

    if n_stay == 0:
        # All transitions are changes: alpha -> infinity
        # Cap at 10 (reasonable upper bound for most simulation ranges)
        return 50.0

    if n_change == 0:
        # No changes at all: alpha = 0
        return 0.0

    ratio = n_stay / n_total
    if ratio <= 0:
        return 50.0
    if ratio >= 1:
        return 0.0

    return -np.log(ratio)


def mle_alpha_discrete(
    obs: NDArray[np.float64],
    Q: DiscreteUniformQ,
) -> float:
    """MLE for alpha with discrete Q (e.g., Uniform{1,...,m}).

    For discrete Q, P(observe same) = e^{-alpha} + (1-e^{-alpha}) * (1/m).
    The MLE is found by numerical optimization of the full likelihood.

    Args:
        obs: Observation sequence.
        Q: Discrete uniform distribution.

    Returns:
        MLE estimate of alpha.
    """
    n_total = len(obs) - 1
    if n_total == 0:
        return 0.0

    n_stay = int(np.sum(obs[1:] == obs[:-1]))
    n_change = n_total - n_stay

    if n_change == 0:
        return 0.0

    pmf_same = Q.pmf_same  # 1/m for uniform

    def neg_log_lik(alpha):
        if alpha <= 0:
            return 1e10
        p_stay = np.exp(-alpha)
        p_jump = 1.0 - p_stay
        p_obs_same = p_stay + p_jump * pmf_same
        p_obs_diff = p_jump * (1.0 - pmf_same)

        ll = n_stay * np.log(p_obs_same) + n_change * np.log(p_obs_diff)

        # Add Q PMF contributions
        for t in range(1, len(obs)):
            if not np.isclose(obs[t], obs[t - 1]):
                ll += np.log(max(Q.pmf(obs[t]), 1e-300))

        # Initial observation
        ll += np.log(max(Q.pmf(obs[0]), 1e-300))

        return -ll

    result = minimize_scalar(neg_log_lik, bounds=(0.01, 50.0), method="bounded")
    return result.x


def mle_alpha(obs: NDArray[np.float64], Q=None) -> float:
    """MLE for alpha, dispatching to continuous or discrete version.

    Args:
        obs: Observation sequence.
        Q: Distribution object (DiscreteUniformQ or GIGQ). If None,
            uses continuous MLE.

    Returns:
        MLE estimate of alpha.
    """
    if isinstance(Q, DiscreteUniformQ):
        return mle_alpha_discrete(obs, Q)
    else:
        return mle_alpha_continuous(obs)


def mle_full_gig(obs: NDArray[np.float64], rng: np.random.Generator | None = None) -> dict:
    """Joint MLE for (alpha, lambda, kappa, eta) with GIG invariant distribution.

    For continuous Q, alpha MLE is analytical. GIG parameters are estimated
    via Gibbs posterior mean to avoid MLE instability from near-unidentifiability.

    Args:
        obs: Observation sequence.
        rng: Random number generator (for Gibbs Q estimation).

    Returns:
        Dict with keys: alpha, lam, kappa, eta.
    """
    n_total = len(obs) - 1
    if n_total < 1:
        return {"alpha": 0.0, "lam": 0.0, "kappa": 1.0, "eta": 1.0}

    n_change = n_total - int(np.sum(obs[1:] == obs[:-1]))

    # Alpha: analytical MLE for continuous Q
    alpha_hat = mle_alpha_continuous(obs) if n_change > 0 else 0.0

    # GIG parameters: Gibbs posterior mean (avoids MLE instability)
    gig_params = gibbs_q_posterior_mean(obs, rng=rng)
    return {"alpha": alpha_hat, **gig_params}


def _get_distinct_values(obs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Extract distinct values from observation sequence (including x₀).

    For continuous Q, each time x_t != x_{t-1}, the value x_t is a fresh
    draw from Q. x₀ is also from Q. Used by MLE which includes x₀ in likelihood.
    """
    changes = obs[1:] != obs[:-1]
    jump_vals = [obs[0]]
    for t in range(1, len(obs)):
        if changes[t - 1]:
            jump_vals.append(obs[t])
    return np.array(jump_vals)


def _get_jump_values(obs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Extract jump values only (excluding x₀).

    Per Anzarut's definition, J = {j : x_j != x_{j-1}}.
    Only the values at jump times, not the initial observation.
    Used by NDNJ for Q parameter estimation.
    """
    changes = obs[1:] != obs[:-1]
    return obs[1:][changes]


def _mle_gig(values: NDArray[np.float64], method: str = "de") -> dict:
    """Estimate GIG parameters by MLE on i.i.d. samples.

    Uses the standard (chi, psi) parameterization internally:
        chi = kappa*eta, psi = kappa/eta
        kappa = sqrt(chi*psi), eta = sqrt(chi/psi)

    The density is:
        f(x) = (chi/psi)^(lam/2) / (2*K_lam(sqrt(chi*psi)))
               * x^(lam-1) * exp(-(chi*x + psi/x)/2)

    Args:
        values: i.i.d. samples from GIG.
        method: Optimization method. "de" uses differential evolution (default,
            most robust for GIG). "nm" uses Nelder-Mead with multiple restarts.

    Returns:
        Dict with keys: lam, kappa, eta.
    """
    from scipy.optimize import minimize, differential_evolution
    from scipy.special import kv as bessel_kv

    n = len(values)
    if n < 3:
        return {"lam": 0.0, "kappa": 1.0, "eta": 1.0}

    values = np.asarray(values, dtype=np.float64)
    if np.any(values <= 0) or not np.all(np.isfinite(values)):
        pos = values[(values > 0) & np.isfinite(values)]
        if len(pos) < 3:
            return {"lam": 0.0, "kappa": 1.0, "eta": 1.0}
        values = pos

    sum_log_x = np.sum(np.log(values))
    sum_x = np.sum(values)
    sum_inv_x = np.sum(1.0 / values)

    def neg_log_lik(params):
        lam_p, log_chi, log_psi = params
        chi = np.exp(log_chi)
        psi = np.exp(log_psi)
        omega = np.sqrt(chi * psi)
        if omega < 0.01 or chi < 1e-6 or psi < 1e-6:
            return 1e10
        try:
            bessel_val = bessel_kv(lam_p, omega)
            if bessel_val <= 0 or not np.isfinite(bessel_val):
                return 1e10
        except (ValueError, OverflowError):
            return 1e10
        log_norm = (lam_p / 2) * np.log(chi / psi) - np.log(2) - np.log(bessel_val)
        ll = n * log_norm + (lam_p - 1) * sum_log_x - (chi * sum_x + psi * sum_inv_x) / 2
        if not np.isfinite(ll):
            return 1e10
        return -ll

    # Method of moments for starting point using GIG moments:
    # E[X] = sqrt(psi/chi) * K_{lam+1}(omega)/K_{lam}(omega) where omega = sqrt(chi*psi)
    # Var[X] = (chi/psi) * K_{lam+2}/K_{lam} + (E[X])^2 * (1 - K_{lam+1}^2/(K_lam * K_{lam+2}))
    # Simplified: use sample mean and variance to get initial (chi, psi)
    sample_mean = np.mean(values)
    sample_var = np.var(values)

    # Start from multiple initial guesses:
    # 1. Inverse-Gamma-motivated: chi ~ 1/var, psi ~ 1/mean
    chi0 = max(1.0 / max(sample_var, 1e-6), 0.01)
    psi0 = max(1.0 / max(sample_mean, 1e-6), 0.01)
    # 2. Gamma-motivated: chi ~ mean, psi ~ mean (for moderate kappa)
    chi1 = max(sample_mean, 0.01)
    psi1 = max(sample_mean, 0.01)
    # 3. Large-kappa-motivated: chi ~ kappa_hat / mean, psi ~ kappa_hat * mean
    # For GIG: chi = kappa*eta, psi = kappa/eta, eta ≈ 1/mean for large kappa
    # So: chi ≈ kappa_hat / mean, psi ≈ kappa_hat * mean
    kappa_hat = max(sample_mean**2 / max(sample_var, 1e-6), 0.1)
    chi2 = max(kappa_hat / max(sample_mean, 1e-6), 0.01)
    psi2 = max(kappa_hat * sample_mean, 0.01)

    best_nll = np.inf
    best_params = None

    # Profile likelihood approach: optimize (log_chi, log_psi) for a grid of lambda values
    # This avoids the local optimum problem where DE converges to boundary solutions.
    lam_grid = np.arange(-5, 5.5, 0.5).tolist() + np.arange(-20, 20.5, 4).tolist()
    lam_grid = sorted(set(lam_grid))

    for lam_try in lam_grid:
        # For each lambda, optimize (log_chi, log_psi) using Nelder-Mead
        def neg_ll_fixed_lam(params):
            return neg_log_lik([lam_try, params[0], params[1]])

        for x0_log_chi, x0_log_psi in [
            (np.log(chi0), np.log(psi0)),
            (np.log(chi1), np.log(psi1)),
            (np.log(chi2), np.log(psi2)),
        ]:
            try:
                result = minimize(neg_ll_fixed_lam, x0=[x0_log_chi, x0_log_psi],
                                  method="Nelder-Mead",
                                  options={"maxiter": 3000, "xatol": 1e-10, "fatol": 1e-10})
                if result.fun < best_nll:
                    best_nll = result.fun
                    best_params = np.array([lam_try, result.x[0], result.x[1]])
            except Exception:
                pass

    # Differential evolution as secondary method (may find better solutions in some cases)
    if method in ("de", "scipy"):
        try:
            de_result = differential_evolution(
                neg_log_lik,
                bounds=[(-20, 20), (-8, 14), (-8, 14)],
                seed=42, maxiter=3000, tol=1e-12, polish=True,
                popsize=30,
            )
            if de_result.fun < best_nll:
                best_nll = de_result.fun
                best_params = de_result.x
        except Exception:
            pass

    # Nelder-Mead with multiple restarts as secondary / fallback
    starts = [
        [0.0, np.log(chi0), np.log(psi0)],
        [1.0, np.log(chi0), np.log(psi0)],
        [-1.0, np.log(chi0), np.log(psi0)],
        [0.0, np.log(chi0 * 0.5), np.log(psi0 * 2)],
        [0.0, np.log(chi0 * 2), np.log(psi0 * 0.5)],
        [2.0, np.log(chi0), np.log(psi0)],
        [-2.0, np.log(chi0), np.log(psi0)],
        [0.0, np.log(chi0 * 0.1), np.log(psi0 * 10)],
        [0.0, np.log(chi0 * 10), np.log(psi0 * 0.1)],
        # Additional starts from method-of-moments guesses
        [0.0, np.log(chi1), np.log(psi1)],
        [0.0, np.log(chi2), np.log(psi2)],
        [1.0, np.log(chi2), np.log(psi2)],
        [-1.0, np.log(chi2), np.log(psi2)],
    ]

    for x0 in starts:
        try:
            result = minimize(neg_log_lik, x0=x0, method="Nelder-Mead",
                              options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8})
            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x
        except Exception:
            continue

    if best_params is None:
        return {"lam": 0.0, "kappa": max(np.sqrt(chi0 * psi0), 0.1),
                "eta": max(np.sqrt(chi0 / psi0), 0.01)}

    lam_hat = best_params[0]
    chi_hat = np.exp(best_params[1])
    psi_hat = np.exp(best_params[2])

    # Boundary recovery: if the optimizer hit a boundary, try harder starting
    # from data-informed initial points with more aggressive Nelder-Mead.
    kappa_est = np.sqrt(chi_hat * psi_hat)
    if abs(lam_hat) >= 19.5 or kappa_est < 0.5:
        # The optimizer got stuck at a boundary. Try a focused search around
        # the method-of-moments solution and a lambda grid near 0.
        recovery_starts = [
            [0.0, np.log(chi2), np.log(psi2)],
            [1.0, np.log(chi2), np.log(psi2)],
            [-1.0, np.log(chi2), np.log(psi2)],
            [0.0, np.log(chi1), np.log(psi1)],
            [0.5, np.log(chi2 * 2), np.log(psi2 * 0.5)],
            [-0.5, np.log(chi2 * 0.5), np.log(psi2 * 2)],
        ]
        for x0 in recovery_starts:
            try:
                result = minimize(neg_log_lik, x0=x0, method="Nelder-Mead",
                                  options={"maxiter": 10000, "xatol": 1e-12, "fatol": 1e-12})
                if result.fun < best_nll:
                    best_nll = result.fun
                    best_params = result.x
                    lam_hat = best_params[0]
                    chi_hat = np.exp(best_params[1])
                    psi_hat = np.exp(best_params[2])
            except Exception:
                pass

    # Convert (lam, chi, psi) -> (lam, kappa, eta)
    kappa_hat = np.sqrt(chi_hat * psi_hat)
    eta_hat = np.sqrt(chi_hat / psi_hat) if psi_hat > 1e-10 else 1.0

    return {
        "lam": np.clip(lam_hat, -10, 10),
        "kappa": np.clip(kappa_hat, 0.01, 200),
        "eta": np.clip(eta_hat, 0.001, 50),
    }


def em_estimate_discrete(
    obs: NDArray[np.float64],
    Q: DiscreteUniformQ,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> float:
    """EM algorithm for alpha with discrete Q (Uniform{1,...,m}).

    Handles hidden jumps where the process jumps but draws the same value.

    E-step: For each t where x_t = x_{t-1}, compute P(jump | same_obs):
        P(jump | same) = (1-e^{-alpha}) * (1/m) / (e^{-alpha} + (1-e^{-alpha})*(1/m))

    M-step: Update alpha using expected counts:
        alpha = -log(E[n_stay_no_jump] / n_total)

    Args:
        obs: Observation sequence.
        Q: Discrete uniform distribution.
        max_iter: Maximum EM iterations.
        tol: Convergence tolerance.

    Returns:
        EM estimate of alpha.
    """
    n_total = len(obs) - 1
    if n_total == 0:
        return 0.0

    n_stay = int(np.sum(obs[1:] == obs[:-1]))
    n_change = n_total - n_stay

    if n_change == 0:
        return 0.0

    pmf_same = Q.pmf_same  # 1/m for uniform
    alpha = mle_alpha_discrete(obs, Q)  # Start from MLE

    for _ in range(max_iter):
        alpha_old = alpha

        # E-step: expected number of "stays" (no jump) among same-value pairs
        p_jump = 1.0 - np.exp(-alpha)
        p_stay_true = np.exp(-alpha)
        p_obs_same = p_stay_true + p_jump * pmf_same
        expected_no_jump = n_stay * (p_stay_true / p_obs_same)

        # M-step: update alpha
        if expected_no_jump <= 0 or expected_no_jump >= n_total:
            break
        alpha = -np.log(expected_no_jump / n_total)
        alpha = max(alpha, 0.001)
        alpha = min(alpha, 50.0)  # cap at reasonable maximum

        if abs(alpha - alpha_old) < tol:
            break

    return alpha


def gibbs_estimate_discrete(
    obs: NDArray[np.float64],
    Q: DiscreteUniformQ,
    n_iter: int = 5000,
    burn_in: int = 1000,
    rng: np.random.Generator | None = None,
    prior_c: float = 0.01,
    method: str = "a",
) -> float:
    """Gibbs sampler for alpha with discrete Q (e.g., Uniform{1,...,m}).

    For discrete Q, "stays" (x_t = x_{t-1}) can be either true stays
    or hidden jumps (process jumped but drew same value from Q). The
    Gibbs sampler alternates between:
    - Z-step: sample latent jump indicators z_i for each stay
    - Alpha-step: sample alpha given the z indicators

    Args:
        obs: Observation sequence.
        Q: Discrete uniform distribution.
        n_iter: Number of Gibbs iterations.
        burn_in: Number of burn-in iterations to discard.
        rng: Random number generator.
        prior_c: Rate parameter for Exp(c) prior on alpha.
        method: "a" for MH random walk on alpha posterior,
                "b" for conjugate Gamma update with latent jump times.

    Returns:
        Posterior mean of alpha.
    """
    if rng is None:
        rng = np.random.default_rng()

    if method not in ("a", "b"):
        raise ValueError(f"method must be 'a' or 'b', got '{method}'")

    n_total = len(obs) - 1
    if n_total == 0:
        return 0.0

    n_stay = int(np.sum(obs[1:] == obs[:-1]))
    n_change = n_total - n_stay
    pmf_same = Q.pmf_same  # 1/m

    if n_change == 0 and n_stay == 0:
        return 0.0

    # Identify stay positions (where x_t == x_{t-1})
    stay_mask = obs[1:] == obs[:-1]
    stay_indices = np.where(stay_mask)[0]  # 0-based indices of stays
    # Identify change positions (where x_t != x_{t-1})
    change_indices = np.where(~stay_mask)[0] + 1  # 1-based positions of changes

    # Initialize alpha from MLE
    alpha = mle_alpha_discrete(obs, Q)
    alpha = np.clip(alpha, 0.01, 50.0)

    # Initialize z: latent jump indicators for stays
    # z_i = 1 means "jump happened" at stay position i
    z = np.zeros(n_stay, dtype=np.float64)
    # Initialize z from current alpha
    p_jump = 1.0 - np.exp(-alpha)
    p_stay_true = np.exp(-alpha)
    p_jump_given_same = p_jump * pmf_same / (p_stay_true + p_jump * pmf_same)
    z[:] = (rng.uniform(size=n_stay) < p_jump_given_same).astype(np.float64)

    alpha_samples = np.empty(n_iter)

    # Adaptive step size for method "a"
    alpha_step = max(0.5, alpha * 0.3)
    adapt_interval = 50
    alpha_accepts = 0

    for i in range(n_iter):
        # --- Z-step: sample z_i | alpha for each stay ---
        p_jump = 1.0 - np.exp(-alpha)
        p_stay_true = np.exp(-alpha)
        p_jump_given_same = p_jump * pmf_same / (p_stay_true + p_jump * pmf_same)
        z = (rng.uniform(size=n_stay) < p_jump_given_same).astype(np.float64)

        n_jump_from_stays = int(np.sum(z))
        n_total_jumps = n_change + n_jump_from_stays
        n_total_stays_true = n_stay - n_jump_from_stays  # true non-jump stays

        # --- Alpha-step ---
        if method == "a":
            # Gibbs-a: MH random walk on pi(alpha | z, data)
            # pi(alpha | z) ∝ (1 - e^{-alpha})^{n_total_jumps} * e^{-alpha * (n_total_stays_true + c)}
            def log_post_a(a):
                if a <= 0:
                    return -1e10
                return n_total_jumps * np.log(1.0 - np.exp(-a)) - a * (n_total_stays_true + prior_c)

            alpha_prop = alpha + rng.normal(0, alpha_step)
            if alpha_prop > 0:
                log_ratio = log_post_a(alpha_prop) - log_post_a(alpha)
                if np.log(rng.uniform()) < log_ratio:
                    alpha = alpha_prop
                    alpha_accepts += 1
            alpha = np.clip(alpha, 0.001, 50.0)
        else:
            # Gibbs-b: conjugate Gamma with latent jump times
            # Sample j_k ~ Uniform(i_k - 1, i_k) for each observed jump (change or hidden)
            # Then alpha ~ Gamma(n_total_jumps + 1, 1 / (j_m + c))
            all_jump_indices = []
            # Observed changes at positions change_indices (1-based)
            all_jump_indices.extend(change_indices.tolist())
            # Hidden jumps at stay positions where z=1
            for si in range(n_stay):
                if z[si] == 1:
                    all_jump_indices.append(stay_indices[si] + 1)  # 1-based

            if len(all_jump_indices) > 0:
                jump_times = np.array([rng.uniform(idx - 1, idx) for idx in all_jump_indices])
                j_m = max(jump_times[-1], 1.0)
            else:
                j_m = float(n_total)

            alpha = rng.gamma(n_total_jumps + 1, 1.0 / (j_m + prior_c))
            alpha = np.clip(alpha, 0.001, 50.0)

        # Adapt step size for method "a"
        if method == "a" and (i + 1) % adapt_interval == 0 and i > 0:
            rate = alpha_accepts / adapt_interval
            if rate > 0.44:
                alpha_step = min(alpha_step * 1.2, 50.0)
            elif rate < 0.23:
                alpha_step = max(alpha_step * 0.8, 0.05)
            alpha_accepts = 0

        alpha_samples[i] = alpha

    return float(np.mean(alpha_samples[burn_in:]))


def em_estimate_gig(
    obs: NDArray[np.float64],
    max_iter: int = 100,
    tol: float = 1e-6,
    rng: np.random.Generator | None = None,
) -> dict:
    """EM algorithm for (alpha, lambda, kappa, eta) with GIG Q.

    For continuous Q, the E-step is trivial (jumps are observed), so EM
    converges to the same alpha as MLE. GIG parameters are estimated via
    Gibbs posterior mean to avoid MLE instability.

    Args:
        obs: Observation sequence.
        max_iter: Maximum EM iterations (unused for continuous Q).
        tol: Convergence tolerance (unused for continuous Q).
        rng: Random number generator (for Gibbs Q estimation).

    Returns:
        Dict with alpha, lam, kappa, eta.
    """
    n_total = len(obs) - 1
    if n_total < 1:
        return {"alpha": 0.0, "lam": 0.0, "kappa": 1.0, "eta": 1.0}

    n_change = n_total - int(np.sum(obs[1:] == obs[:-1]))

    # For continuous Q: alpha converges immediately to MLE
    alpha_hat = mle_alpha_continuous(obs) if n_change > 0 else 0.0

    # GIG parameters: Gibbs posterior mean (avoids MLE instability)
    gig_params = gibbs_q_posterior_mean(obs, rng=rng)
    return {"alpha": alpha_hat, **gig_params}


def gibbs_estimate_gig(
    obs: NDArray[np.float64],
    n_iter: int = 5000,
    burn_in: int = 1000,
    rng: np.random.Generator | None = None,
    prior_c: float = 0.01,
    method: str = "a",
) -> dict:
    """Gibbs sampler for (alpha, lambda, kappa, eta) with GIG Q.

    Uses MH for alpha (correct posterior) and adaptive MH for GIG parameters
    in (chi, psi) parameterization with Robbins-Monro step size adaptation.

    The correct posterior for alpha (with prior Exp(c)) is:
        pi(alpha | data) proportional to (1 - exp(-alpha))^n_change * exp(-alpha * (n_stay + c))

    Args:
        obs: Observation sequence.
        n_iter: Number of Gibbs iterations.
        burn_in: Number of burn-in iterations to discard.
        rng: Random number generator.
        prior_c: Rate parameter for Exp(c) prior on alpha.
        method: "a" for MH random walk on alpha posterior (default),
                "b" for conjugate Gamma update with latent jump times.

    Returns:
        Dict with posterior means: alpha, lam, kappa, eta.
    """
    if rng is None:
        rng = np.random.default_rng()

    if method not in ("a", "b"):
        raise ValueError(f"method must be 'a' or 'b', got '{method}'")

    n_total = len(obs) - 1
    n_stay = int(np.sum(obs[1:] == obs[:-1]))
    n_change = n_total - n_stay

    # For Gibbs-b: precompute jump indices for latent jump time sampling
    jump_indices = np.where(obs[1:] != obs[:-1])[0] + 1  # 1-based positions

    distinct_vals = _get_distinct_values(obs)

    alpha_samples = np.empty(n_iter)
    lam_samples = np.empty(n_iter)
    kappa_samples = np.empty(n_iter)
    eta_samples = np.empty(n_iter)

    # Initial values from MLE
    alpha = mle_alpha_continuous(obs) if n_change > 0 else 0.01
    gig_params = _mle_gig(distinct_vals) if len(distinct_vals) >= 3 else {"lam": 0.0, "kappa": 1.0, "eta": 1.0}
    lam = gig_params["lam"]
    kappa = gig_params["kappa"]
    eta = gig_params["eta"]

    # Work in (lam, chi, psi) parameterization for MH within Gibbs
    chi = kappa * eta
    psi = kappa / eta

    def alpha_log_posterior(a):
        if a <= 0:
            return -1e10
        return n_change * np.log(1.0 - np.exp(-a)) - a * (n_stay + prior_c)

    sum_log_dv = np.sum(np.log(distinct_vals))
    sum_dv = np.sum(distinct_vals)
    sum_inv_dv = np.sum(1.0 / distinct_vals)
    nd = len(distinct_vals)

    # Adaptive MH step sizes with window-based acceptance tracking
    # Component-wise MH: update each parameter individually for better mixing
    alpha_step = max(0.5, alpha * 0.3)
    lam_step = 0.3
    log_chi_step = 0.2
    log_psi_step = 0.2
    adapt_interval = 50
    alpha_accepts = 0
    lam_accepts = 0
    chi_accepts = 0
    psi_accepts = 0

    def _gig_log_lik_params(lam_v, chi_v, psi_v):
        """Log-likelihood of distinct values under GIG(lam, chi, psi)."""
        from scipy.special import kv as bkv
        omega_v = np.sqrt(chi_v * psi_v)
        Kv = bkv(lam_v, omega_v)
        if Kv <= 0 or not np.isfinite(Kv):
            return -1e10
        ln = (lam_v / 2) * np.log(chi_v / psi_v) - np.log(2) - np.log(Kv)
        return nd * ln + (lam_v - 1) * sum_log_dv - (chi_v * sum_dv + psi_v * sum_inv_dv) / 2

    for i in range(n_iter):
        # --- 1. Sample alpha ---
        if method == "a":
            # Gibbs-a: MH random walk on pi(alpha | data)
            alpha_prop = alpha + rng.normal(0, alpha_step)
            if alpha_prop > 0:
                log_ratio_a = alpha_log_posterior(alpha_prop) - alpha_log_posterior(alpha)
                if np.log(rng.uniform()) < log_ratio_a:
                    alpha = alpha_prop
                    alpha_accepts += 1
            alpha = np.clip(alpha, 0.001, 50.0)
        else:
            # Gibbs-b: conjugate update with latent jump times
            # Sample j_k ~ Uniform(i_k - 1, i_k), then alpha ~ Gamma(m+1, 1/(j_m + c))
            if len(jump_indices) > 0:
                jump_times = np.array([rng.uniform(idx - 1, idx) for idx in jump_indices])
                j_m = jump_times[-1]
            else:
                j_m = float(n_total)
            alpha = rng.gamma(n_change + 1, 1.0 / (j_m + prior_c))
            alpha = np.clip(alpha, 0.001, 50.0)

        # --- 2. Sample lambda using component-wise MH ---
        lam_prop = lam + rng.normal(0, lam_step)
        ll_curr = _gig_log_lik_params(lam, chi, psi)
        ll_prop = _gig_log_lik_params(lam_prop, chi, psi)
        if np.isfinite(ll_prop) and np.isfinite(ll_curr):
            if np.log(rng.uniform()) < (ll_prop - ll_curr):
                lam = lam_prop
                lam_accepts += 1

        # --- 3. Sample chi (log-scale MH) ---
        log_chi_prop = np.log(chi) + rng.normal(0, log_chi_step)
        chi_prop = np.exp(log_chi_prop)
        if chi_prop > 0.01:
            ll_curr_chi = _gig_log_lik_params(lam, chi, psi)
            ll_prop_chi = _gig_log_lik_params(lam, chi_prop, psi)
            # Jacobian: proposing in log(chi), so add log(chi_prop) - log(chi)
            if np.isfinite(ll_prop_chi) and np.isfinite(ll_curr_chi):
                log_ratio_chi = (ll_prop_chi - ll_curr_chi) + np.log(chi_prop) - np.log(chi)
                if np.log(rng.uniform()) < log_ratio_chi:
                    chi = chi_prop
                    chi_accepts += 1

        # --- 4. Sample psi (log-scale MH) ---
        log_psi_prop = np.log(psi) + rng.normal(0, log_psi_step)
        psi_prop = np.exp(log_psi_prop)
        if psi_prop > 0.01:
            ll_curr_psi = _gig_log_lik_params(lam, chi, psi)
            ll_prop_psi = _gig_log_lik_params(lam, chi, psi_prop)
            if np.isfinite(ll_prop_psi) and np.isfinite(ll_curr_psi):
                log_ratio_psi = (ll_prop_psi - ll_curr_psi) + np.log(psi_prop) - np.log(psi)
                if np.log(rng.uniform()) < log_ratio_psi:
                    psi = psi_prop
                    psi_accepts += 1

        # Adapt step sizes every adapt_interval iterations
        if (i + 1) % adapt_interval == 0 and i > 0:
            if method == "a":
                alpha_rate = alpha_accepts / adapt_interval
                if alpha_rate > 0.44:
                    alpha_step = min(alpha_step * 1.2, 50.0)
                elif alpha_rate < 0.23:
                    alpha_step = max(alpha_step * 0.8, 0.05)
            lam_rate = lam_accepts / adapt_interval
            chi_rate = chi_accepts / adapt_interval
            psi_rate = psi_accepts / adapt_interval
            if lam_rate > 0.44:
                lam_step = min(lam_step * 1.2, 2.0)
            elif lam_rate < 0.23:
                lam_step = max(lam_step * 0.8, 0.05)
            if chi_rate > 0.44:
                log_chi_step = min(log_chi_step * 1.2, 1.0)
            elif chi_rate < 0.23:
                log_chi_step = max(log_chi_step * 0.8, 0.02)
            if psi_rate > 0.44:
                log_psi_step = min(log_psi_step * 1.2, 1.0)
            elif psi_rate < 0.23:
                log_psi_step = max(log_psi_step * 0.8, 0.02)
            alpha_accepts = 0
            lam_accepts = 0
            chi_accepts = 0
            psi_accepts = 0

        # Convert (lam, chi, psi) -> (lam, kappa, eta)
        kappa = np.sqrt(chi * psi)
        eta = np.sqrt(chi / psi) if psi > 1e-10 else eta

        alpha_samples[i] = alpha
        lam_samples[i] = lam
        kappa_samples[i] = kappa
        eta_samples[i] = eta

    return {
        "alpha": np.mean(alpha_samples[burn_in:]),
        "lam": np.mean(lam_samples[burn_in:]),
        "kappa": np.mean(kappa_samples[burn_in:]),
        "eta": np.mean(eta_samples[burn_in:]),
    }


def gibbs_q_posterior_mean(
    obs: NDArray[np.float64],
    n_iter: int = 2000,
    burn_in: int = 500,
    rng: np.random.Generator | None = None,
    prior_c: float = 0.01,
) -> dict:
    """Run Gibbs sampler and return posterior mean for (lam, kappa, eta) only.

    Used by NDNJ/MLE/EM to estimate Q parameters via posterior averaging
    instead of MLE, avoiding GIG near-unidentifiability issues.
    Alpha is estimated separately by each method.

    Args:
        obs: Observation sequence.
        n_iter: Number of Gibbs iterations.
        burn_in: Number of burn-in iterations to discard.
        rng: Random number generator.
        prior_c: Rate parameter for Exp(c) prior on alpha.

    Returns:
        Dict with keys: lam, kappa, eta (posterior means).
    """
    result = gibbs_estimate_gig(obs, n_iter=n_iter, burn_in=burn_in, rng=rng, prior_c=prior_c)
    return {k: v for k, v in result.items() if k != "alpha"}


def _gig_log_lik(values, lam, kappa, eta, bessel_kv_func):
    """Log-likelihood of i.i.d. samples under GIG(lam, kappa, eta)."""
    bessel_val = bessel_kv_func(lam, kappa)
    if bessel_val <= 0 or not np.isfinite(bessel_val):
        return -1e10

    log_norm = lam * np.log(eta) - np.log(2) - np.log(bessel_val)
    log_lik = 0.0
    for x in values:
        if x <= 0:
            continue
        log_lik += log_norm + (lam - 1) * np.log(x) - (kappa / 2) * (eta * x + 1.0 / (eta * x))

    return log_lik