"""Estimación para el proceso SF-Harris: NDNJ, MLE, EM y Gibbs."""
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar, minimize

from .process import SFHarrisProcess
from .distributions import DiscreteUniformQ, GIGQ


def ndnj_estimate(obs: NDArray[np.float64], return_params: bool = False, rng: np.random.Generator | None = None):
    """NDNJ: estima alpha a partir de la frecuencia de cambios observados.

    J = {j : x_j != x_{j-1}}; p_hat = |J|/(k-1) estima 1 - e^{-alpha}
    => alpha_hat = -log(1 - p_hat).
    Con Q discreta hay saltos ocultos (sesgo a la baja).

    return_params=True: agrega (lam, kappa, eta) vía media posterior Gibbs.
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

    # tope para evitar log(0)
    if p_hat >= 1.0:
        alpha_hat = 50.0
    else:
        alpha_hat = -np.log(1.0 - p_hat)

    if not return_params:
        return alpha_hat

    # Q vía media posterior Gibbs (evita la inestabilidad del MLE)
    gig_params = gibbs_q_posterior_mean(obs, rng=rng)
    return {"alpha": alpha_hat, **gig_params}


def mle_alpha_continuous(obs: NDArray[np.float64]) -> float:
    """MLE de alpha con Q continua: x_t = x_{t-1} <=> no hubo salto.

    alpha_hat = -log(n_stay / n_total).
    """
    n_total = len(obs) - 1
    if n_total == 0:
        return 0.0

    n_stay = int(np.sum(obs[1:] == obs[:-1]))
    n_change = n_total - n_stay

    if n_stay == 0:
        # todos cambian: alpha -> infinito (tope en 50)
        return 50.0

    if n_change == 0:
        return 0.0  # sin cambios: alpha = 0

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
    """MLE de alpha con Q discreta (Uniform{1,...,m}).

    P(observar igual) = e^{-alpha} + (1-e^{-alpha})*(1/m): optimización numérica.
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

        # contribución de la PMF de Q en cada valor nuevo y en x_0
        for t in range(1, len(obs)):
            if not np.isclose(obs[t], obs[t - 1]):
                ll += np.log(max(Q.pmf(obs[t]), 1e-300))

        ll += np.log(max(Q.pmf(obs[0]), 1e-300))

        return -ll

    result = minimize_scalar(neg_log_lik, bounds=(0.01, 50.0), method="bounded")
    return result.x


def mle_alpha(obs: NDArray[np.float64], Q=None) -> float:
    """MLE de alpha: despacha a la versión discreta o continua según Q."""
    if isinstance(Q, DiscreteUniformQ):
        return mle_alpha_discrete(obs, Q)
    else:
        return mle_alpha_continuous(obs)


def mle_full_gig(obs: NDArray[np.float64], rng: np.random.Generator | None = None) -> dict:
    """MLE conjunto (alpha, lam, kappa, eta) con Q ~ GIG.

    alpha analítico (Q continua); GIG vía media posterior Gibbs
    (el MLE de GIG es casi no identificable).
    """
    n_total = len(obs) - 1
    if n_total < 1:
        return {"alpha": 0.0, "lam": 0.0, "kappa": 1.0, "eta": 1.0}

    n_change = n_total - int(np.sum(obs[1:] == obs[:-1]))

    # alpha: MLE analítico (Q continua)
    alpha_hat = mle_alpha_continuous(obs) if n_change > 0 else 0.0

    # GIG: media posterior Gibbs (evita inestabilidad del MLE)
    gig_params = gibbs_q_posterior_mean(obs, rng=rng)
    return {"alpha": alpha_hat, **gig_params}


def _get_distinct_values(obs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Valores distintos de la secuencia (incluye x_0; cada salto es un draw fresco de Q)."""
    changes = obs[1:] != obs[:-1]
    jump_vals = [obs[0]]
    for t in range(1, len(obs)):
        if changes[t - 1]:
            jump_vals.append(obs[t])
    return np.array(jump_vals)


def _get_jump_values(obs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Solo valores en tiempos de salto J = {j : x_j != x_{j-1}} (sin x_0); lo usa NDNJ."""
    changes = obs[1:] != obs[:-1]
    return obs[1:][changes]


def _mle_gig(values: NDArray[np.float64], method: str = "de") -> dict:
    """MLE de GIG sobre muestras i.i.d., en parametrización (chi, psi).

    chi = kappa*eta, psi = kappa/eta => kappa = sqrt(chi*psi), eta = sqrt(chi/psi).
    method="de": evolución diferencial; "nm": Nelder-Mead con reinicios.
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

    # punto inicial por momentos:
    # E[X] = sqrt(psi/chi) K_{lam+1}(omega)/K_lam(omega), omega = sqrt(chi*psi)
    sample_mean = np.mean(values)
    sample_var = np.var(values)

    # 1) motivado por Gamma inversa: chi ~ 1/var, psi ~ 1/media
    chi0 = max(1.0 / max(sample_var, 1e-6), 0.01)
    psi0 = max(1.0 / max(sample_mean, 1e-6), 0.01)
    # 2) motivado por Gamma: chi ~ media, psi ~ media
    chi1 = max(sample_mean, 0.01)
    psi1 = max(sample_mean, 0.01)
    # 3) kappa grande: chi = kappa*eta ~ kappa/media, psi = kappa/eta ~ kappa*media
    kappa_hat = max(sample_mean**2 / max(sample_var, 1e-6), 0.1)
    chi2 = max(kappa_hat / max(sample_mean, 1e-6), 0.01)
    psi2 = max(kappa_hat * sample_mean, 0.01)

    best_nll = np.inf
    best_params = None

    # verosimilitud perfilada: para cada lambda, optimizar (log_chi, log_psi)
    # evita que DE caiga en soluciones de frontera
    lam_grid = np.arange(-5, 5.5, 0.5).tolist() + np.arange(-20, 20.5, 4).tolist()
    lam_grid = sorted(set(lam_grid))

    for lam_try in lam_grid:
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

    # DE como método secundario (a veces encuentra mejores soluciones)
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

    # Nelder-Mead con muchos reinicios (fallback)
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
        # reinicios extra desde los puntos de momentos
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

    # recuperación de frontera: si el optimizador pegó en la frontera,
    # reintentar cerca de la solución de momentos con NM más agresivo
    kappa_est = np.sqrt(chi_hat * psi_hat)
    if abs(lam_hat) >= 19.5 or kappa_est < 0.5:
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

    # (lam, chi, psi) -> (lam, kappa, eta)
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
    """EM para alpha con Q discreta (maneja saltos ocultos).

    E-step: P(salto | igual) = (1-e^{-alpha})(1/m) / (e^{-alpha} + (1-e^{-alpha})(1/m)).
    M-step: alpha = -log(E[n_stay verdadero] / n_total).
    """
    n_total = len(obs) - 1
    if n_total == 0:
        return 0.0

    n_stay = int(np.sum(obs[1:] == obs[:-1]))
    n_change = n_total - n_stay

    if n_change == 0:
        return 0.0

    pmf_same = Q.pmf_same  # 1/m
    alpha = mle_alpha_discrete(obs, Q)  # arranque en el MLE

    for _ in range(max_iter):
        alpha_old = alpha

        # E-step: estancias verdaderas esperadas entre los pares iguales
        p_jump = 1.0 - np.exp(-alpha)
        p_stay_true = np.exp(-alpha)
        p_obs_same = p_stay_true + p_jump * pmf_same
        expected_no_jump = n_stay * (p_stay_true / p_obs_same)

        # M-step: actualizar alpha
        if expected_no_jump <= 0 or expected_no_jump >= n_total:
            break
        alpha = -np.log(expected_no_jump / n_total)
        alpha = max(alpha, 0.001)
        alpha = min(alpha, 50.0)  # tope

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
    """Gibbs para alpha con Q discreta: alterna Z-step (indicadores latentes) y alpha-step.

    method="a": MH random walk sobre pi(alpha | z, datos);
    method="b": actualización Gamma conjugada con tiempos de salto latentes.
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

    # posiciones: estancias (x_t == x_{t-1}) y cambios (x_t != x_{t-1})
    stay_mask = obs[1:] == obs[:-1]
    stay_indices = np.where(stay_mask)[0]
    change_indices = np.where(~stay_mask)[0] + 1

    # alpha inicial desde el MLE
    alpha = mle_alpha_discrete(obs, Q)
    alpha = np.clip(alpha, 0.01, 50.0)

    # z: indicador latente de salto en cada estancia (z_i = 1 => saltó)
    z = np.zeros(n_stay, dtype=np.float64)
    # inicializar z con el alpha actual
    p_jump = 1.0 - np.exp(-alpha)
    p_stay_true = np.exp(-alpha)
    p_jump_given_same = p_jump * pmf_same / (p_stay_true + p_jump * pmf_same)
    z[:] = (rng.uniform(size=n_stay) < p_jump_given_same).astype(np.float64)

    alpha_samples = np.empty(n_iter)

    # tamaño de paso adaptativo para method "a"
    alpha_step = max(0.5, alpha * 0.3)
    adapt_interval = 50
    alpha_accepts = 0

    for i in range(n_iter):
        # --- Z-step: z_i | alpha en cada estancia ---
        p_jump = 1.0 - np.exp(-alpha)
        p_stay_true = np.exp(-alpha)
        p_jump_given_same = p_jump * pmf_same / (p_stay_true + p_jump * pmf_same)
        z = (rng.uniform(size=n_stay) < p_jump_given_same).astype(np.float64)

        n_jump_from_stays = int(np.sum(z))
        n_total_jumps = n_change + n_jump_from_stays
        n_total_stays_true = n_stay - n_jump_from_stays

        # --- Alpha-step ---
        if method == "a":
            # Gibbs-a: MH random walk; pi(alpha | z) ∝ (1-e^{-alpha})^{n_jumps} e^{-alpha(n_stays + c)}
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
            # Gibbs-b: Gamma conjugada con tiempos de salto latentes
            # j_k ~ Uniform(i_k - 1, i_k); alpha ~ Gamma(n_jumps + 1, 1/(j_m + c))
            all_jump_indices = []
            all_jump_indices.extend(change_indices.tolist())  # cambios observados
            for si in range(n_stay):  # saltos ocultos (z = 1)
                if z[si] == 1:
                    all_jump_indices.append(stay_indices[si] + 1)

            if len(all_jump_indices) > 0:
                jump_times = np.array([rng.uniform(idx - 1, idx) for idx in all_jump_indices])
                j_m = max(jump_times[-1], 1.0)
            else:
                j_m = float(n_total)

            alpha = rng.gamma(n_total_jumps + 1, 1.0 / (j_m + prior_c))
            alpha = np.clip(alpha, 0.001, 50.0)

        # adaptar el paso (method "a")
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
    """EM para (alpha, lam, kappa, eta) con GIG.

    Con Q continua los saltos se observan: alpha converge al MLE de inmediato.
    GIG vía media posterior Gibbs.
    """
    n_total = len(obs) - 1
    if n_total < 1:
        return {"alpha": 0.0, "lam": 0.0, "kappa": 1.0, "eta": 1.0}

    n_change = n_total - int(np.sum(obs[1:] == obs[:-1]))

    # Q continua: alpha converge de inmediato al MLE
    alpha_hat = mle_alpha_continuous(obs) if n_change > 0 else 0.0

    # GIG: media posterior Gibbs
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
    """Gibbs para (alpha, lam, kappa, eta) con GIG.

    alpha por MH (posterior correcta: pi ∝ (1-e^{-alpha})^{n_change} e^{-alpha(n_stay+c)});
    GIG por MH adaptativo por componentes en (chi, psi).
    """
    if rng is None:
        rng = np.random.default_rng()

    if method not in ("a", "b"):
        raise ValueError(f"method must be 'a' or 'b', got '{method}'")

    n_total = len(obs) - 1
    n_stay = int(np.sum(obs[1:] == obs[:-1]))
    n_change = n_total - n_stay

    # para Gibbs-b: índices de saltos (tiempos latentes)
    jump_indices = np.where(obs[1:] != obs[:-1])[0] + 1

    distinct_vals = _get_distinct_values(obs)

    alpha_samples = np.empty(n_iter)
    lam_samples = np.empty(n_iter)
    kappa_samples = np.empty(n_iter)
    eta_samples = np.empty(n_iter)

    # valores iniciales desde el MLE
    alpha = mle_alpha_continuous(obs) if n_change > 0 else 0.01
    gig_params = _mle_gig(distinct_vals) if len(distinct_vals) >= 3 else {"lam": 0.0, "kappa": 1.0, "eta": 1.0}
    lam = gig_params["lam"]
    kappa = gig_params["kappa"]
    eta = gig_params["eta"]

    # MH dentro de Gibbs en (lam, chi, psi)
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

    # pasos MH adaptativos, uno por componente (mejor mezcla)
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
        # log-verosimilitud de los valores distintos bajo GIG(lam, chi, psi)
        from scipy.special import kv as bkv
        omega_v = np.sqrt(chi_v * psi_v)
        Kv = bkv(lam_v, omega_v)
        if Kv <= 0 or not np.isfinite(Kv):
            return -1e10
        ln = (lam_v / 2) * np.log(chi_v / psi_v) - np.log(2) - np.log(Kv)
        return nd * ln + (lam_v - 1) * sum_log_dv - (chi_v * sum_dv + psi_v * sum_inv_dv) / 2

    for i in range(n_iter):
        # --- 1. alpha ---
        if method == "a":
            # Gibbs-a: MH random walk
            alpha_prop = alpha + rng.normal(0, alpha_step)
            if alpha_prop > 0:
                log_ratio_a = alpha_log_posterior(alpha_prop) - alpha_log_posterior(alpha)
                if np.log(rng.uniform()) < log_ratio_a:
                    alpha = alpha_prop
                    alpha_accepts += 1
            alpha = np.clip(alpha, 0.001, 50.0)
        else:
            # Gibbs-b: j_k ~ Uniform(i_k - 1, i_k); alpha ~ Gamma(m+1, 1/(j_m + c))
            if len(jump_indices) > 0:
                jump_times = np.array([rng.uniform(idx - 1, idx) for idx in jump_indices])
                j_m = jump_times[-1]
            else:
                j_m = float(n_total)
            alpha = rng.gamma(n_change + 1, 1.0 / (j_m + prior_c))
            alpha = np.clip(alpha, 0.001, 50.0)

        # --- 2. lambda (MH por componente) ---
        lam_prop = lam + rng.normal(0, lam_step)
        ll_curr = _gig_log_lik_params(lam, chi, psi)
        ll_prop = _gig_log_lik_params(lam_prop, chi, psi)
        if np.isfinite(ll_prop) and np.isfinite(ll_curr):
            if np.log(rng.uniform()) < (ll_prop - ll_curr):
                lam = lam_prop
                lam_accepts += 1

        # --- 3. chi (MH en escala log) ---
        log_chi_prop = np.log(chi) + rng.normal(0, log_chi_step)
        chi_prop = np.exp(log_chi_prop)
        if chi_prop > 0.01:
            ll_curr_chi = _gig_log_lik_params(lam, chi, psi)
            ll_prop_chi = _gig_log_lik_params(lam, chi_prop, psi)
            # jacobiano de proponer en log(chi)
            if np.isfinite(ll_prop_chi) and np.isfinite(ll_curr_chi):
                log_ratio_chi = (ll_prop_chi - ll_curr_chi) + np.log(chi_prop) - np.log(chi)
                if np.log(rng.uniform()) < log_ratio_chi:
                    chi = chi_prop
                    chi_accepts += 1

        # --- 4. psi (MH en escala log) ---
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

        # adaptar pasos cada adapt_interval iteraciones
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

        # (lam, chi, psi) -> (lam, kappa, eta) en cada iteración
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
    """Gibbs que regresa solo la media posterior de (lam, kappa, eta).

    NDNJ/MLE/EM la usan para estimar Q promediando la posterior
    (evita la casi-no-identificabilidad del MLE de GIG).
    """
    result = gibbs_estimate_gig(obs, n_iter=n_iter, burn_in=burn_in, rng=rng, prior_c=prior_c)
    return {k: v for k, v in result.items() if k != "alpha"}


def _gig_log_lik(values, lam, kappa, eta, bessel_kv_func):
    # log-verosimilitud de muestras i.i.d. bajo GIG(lam, kappa, eta)
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