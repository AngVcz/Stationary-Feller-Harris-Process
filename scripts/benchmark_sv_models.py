"""Benchmark SV models vs SF-Harris: GARCH(1,1), EGARCH(1,1), Heston.

Compares at the same intraday time scale as Anzarut's Table 3:
- GARCH(1,1): sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2
- EGARCH(1,1): log(sigma_t^2) = omega + alpha * |z_{t-1}| + gamma * z_{t-1} + beta * log(sigma_{t-1}^2)
- Heston: dV = kappa(theta-V)dt + sigma_v*sqrt(V)*dW2

Coverage validation at probability levels p = 0.25, 0.50, 0.75, 0.85, 0.90, 0.95.
Same data split (80/20) and evaluation as SF-Harris models.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from arch import arch_model

from anzarut_replication import (
    load_ibm_data,
    compute_15min_returns,
    detect_and_remove_jumps,
    estimate_periodicity,
    compute_coverage,
)

DATA_PATH = Path(r"C:\Users\angve\OneDrive\Desktop\Servicio\Libros\SF-Harris\IBM.txt")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"

ANZARUT_TABLE3 = {0.25: 25, 0.50: 51, 0.75: 75, 0.85: 84, 0.90: 89, 0.95: 93}
PROB_LEVELS = [0.25, 0.50, 0.75, 0.85, 0.90, 0.95]


def build_dollar_bars(df, dollar_threshold=None):
    """Build dollar bars from minute data."""
    if dollar_threshold is None:
        avg_dv_per_15min = df["dollar_volume"].resample("15min").sum().mean()
        dollar_threshold = avg_dv_per_15min

    returns = []
    cum_dv = 0.0
    first_close = None
    for idx, row in df.iterrows():
        if first_close is None:
            first_close = row["close"]
        cum_dv += row["dollar_volume"]
        if cum_dv >= dollar_threshold:
            ret = np.log(row["close"] / first_close)
            returns.append({"datetime": idx, "return": ret})
            cum_dv = 0.0
            first_close = None

    result = pd.DataFrame(returns)
    if len(result) > 0:
        result = result.set_index("datetime")
    return result["return"]


def fit_garch(returns, p=1, q=1, dist="normal"):
    """Fit GARCH(p,q) model using arch package."""
    am = arch_model(returns * 100, vol="Garch", p=p, q=q, dist=dist, mean="Zero")
    res = am.fit(disp="off")
    return res


def fit_egarch(returns, p=1, q=1, dist="normal"):
    """Fit EGARCH(p,q) model using arch package."""
    am = arch_model(returns * 100, vol="EGARCH", p=p, q=q, dist=dist, mean="Zero")
    res = am.fit(disp="off")
    return res


def simulate_garch(res, n_forecast, n_sim=2000, rng=None):
    """Simulate GARCH paths forward from fitted model.

    Returns array of shape (n_sim, n_forecast) with conditional variance paths.
    """
    if rng is None:
        rng = np.random.default_rng()

    params = res.params
    omega = params["omega"]
    alpha = params["alpha[1]"]
    beta = params["beta[1]"]

    # Last conditional variance from training
    cond_var_raw = res.conditional_volatility
    if hasattr(cond_var_raw, 'values'):
        cond_var_last = cond_var_raw.values[-1]**2
    else:
        cond_var_last = cond_var_raw[-1]**2

    resid_raw = res.resid
    if hasattr(resid_raw, 'values'):
        resid_last = resid_raw.values[-1]
    else:
        resid_last = resid_raw[-1]

    # Use normal residuals for simulation
    # (can extend to Student-t later if needed)

    sim_vars = np.empty((n_sim, n_forecast))
    sim_returns = np.empty((n_sim, n_forecast))

    for s in range(n_sim):
        sigma2_prev = cond_var_last / 10000  # scale back from %
        r_prev = resid_last / 100  # scale back

        for t in range(n_forecast):
            sigma2 = omega/10000 + alpha * r_prev**2 + beta * sigma2_prev
            sigma2 = max(sigma2, 1e-20)
            r_new = np.sqrt(sigma2) * rng.normal()

            sim_vars[s, t] = sigma2
            sim_returns[s, t] = r_new

            sigma2_prev = sigma2
            r_prev = r_new

    return sim_vars, sim_returns


def simulate_egarch(res, n_forecast, n_sim=2000, rng=None):
    """Simulate EGARCH paths forward from fitted model.

    EGARCH: log(sigma_t^2) = omega + alpha*(|z_{t-1}| - sqrt(2/pi)) + beta*log(sigma_{t-1}^2)
    The arch package EGARCH does not include gamma (asymmetry) by default.
    """
    if rng is None:
        rng = np.random.default_rng()

    params = res.params
    omega = params["omega"]
    alpha = params["alpha[1]"]
    beta = params["beta[1]"]

    cond_var_raw = res.conditional_volatility
    if hasattr(cond_var_raw, 'values'):
        cond_var_last = cond_var_raw.values[-1]**2
    else:
        cond_var_last = cond_var_raw[-1]**2

    resid_raw = res.resid
    if hasattr(resid_raw, 'values'):
        resid_last = resid_raw.values[-1]
    else:
        resid_last = resid_raw[-1]

    sim_vars = np.empty((n_sim, n_forecast))
    sim_returns = np.empty((n_sim, n_forecast))

    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)

    for s in range(n_sim):
        log_sigma2_prev = np.log(cond_var_last / 10000)
        r_prev = resid_last / 100

        for t in range(n_forecast):
            z = rng.normal()
            log_sigma2 = omega/10000 + alpha * (np.abs(z) - sqrt_2_over_pi) + beta * log_sigma2_prev
            sigma2 = np.exp(log_sigma2)
            r_new = np.sqrt(sigma2) * z

            sim_vars[s, t] = sigma2
            sim_returns[s, t] = r_new

            log_sigma2_prev = log_sigma2
            r_prev = r_new

    return sim_vars, sim_returns


# ==========================================================================
# Heston model (discretized)
# ==========================================================================
def heston_neg_log_lik(params, log_returns, dt):
    """Negative log-likelihood for discretized Heston model.

    Heston: dV = kappa(theta - V)dt + sigma_v*sqrt(V)*dW2
    Euler discretization with full truncation.
    """
    kappa, theta, sigma_v, rho = params
    if kappa <= 0 or theta <= 0 or sigma_v <= 0:
        return 1e10
    if abs(rho) >= 1:
        return 1e10

    n = len(log_returns)
    V = np.empty(n)
    V[0] = theta  # start at long-run mean

    ll = 0.0
    for t in range(1, n):
        # Variance CIR step
        V[t] = theta + (V[t-1] - theta) * np.exp(-kappa * dt) + \
                sigma_v * np.sqrt(max(V[t-1], 0) * dt) * (log_returns[t] / np.sqrt(max(V[t-1], 1e-20) * dt))
        V[t] = max(V[t], 1e-20)

        # Log-likelihood of return given variance
        if V[t-1] > 0:
            ll += -0.5 * np.log(2 * np.pi * V[t-1] * dt) - 0.5 * log_returns[t]**2 / (V[t-1] * dt)

    return -ll


def fit_heston(log_returns, dt=1.0):
    """Fit Heston model via MLE."""
    # Method of moments initialization
    var_r = np.var(log_returns)
    theta_init = var_r / dt

    # Try multiple starts
    best_result = None
    best_nll = np.inf

    starts = [
        [5.0, theta_init, 0.5, -0.3],
        [2.0, theta_init, 0.3, -0.5],
        [10.0, theta_init, 1.0, 0.0],
        [1.0, theta_init, 0.1, -0.7],
    ]

    for x0 in starts:
        try:
            result = minimize(heston_neg_log_lik, x0=x0, args=(log_returns, dt),
                             method="Nelder-Mead", options={"maxiter": 5000})
            if result.fun < best_nll:
                best_nll = result.fun
                best_result = result
        except Exception:
            continue

    if best_result is None:
        return None

    kappa, theta, sigma_v, rho = best_result.x
    return {"kappa": kappa, "theta": theta, "sigma_v": sigma_v, "rho": rho,
            "converged": best_result.success, "nll": best_nll}


def simulate_heston(params, V0, n_steps, dt=1.0, n_sim=2000, rng=None):
    """Simulate Heston model paths using Euler-Maruyama with full truncation."""
    if rng is None:
        rng = np.random.default_rng()

    kappa = params["kappa"]
    theta = params["theta"]
    sigma_v = params["sigma_v"]
    rho = params["rho"]

    sim_vars = np.empty((n_sim, n_steps))
    sim_returns = np.empty((n_sim, n_steps))

    for s in range(n_sim):
        V = V0
        for t in range(n_steps):
            # Correlated Brownian motions
            Z1 = rng.normal()
            Z2 = rho * Z1 + np.sqrt(1 - rho**2) * rng.normal()

            # Return
            r = np.sqrt(max(V, 0) * dt) * Z1
            sim_returns[s, t] = r

            # Variance (full truncation)
            V = theta + (V - theta) * np.exp(-kappa * dt) + \
                sigma_v * np.sqrt(max(V, 0) * dt) * Z2
            V = max(V, 1e-20)
            sim_vars[s, t] = V

    return sim_vars, sim_returns


def compute_coverage_log_rv(observed_log_rv, simulated_log_rv, prob_levels):
    """Compute coverage for log(RV) predictions.

    observed_log_rv: array of length n
    simulated_log_rv: array of shape (n_sim, n)
    """
    return compute_coverage(observed_log_rv, simulated_log_rv, prob_levels)


if __name__ == "__main__":
    print("=" * 70)
    print("BENCHMARK SV MODELS: GARCH, EGARCH, Heston")
    print("Comparison against SF-Harris at intraday level")
    print("=" * 70)

    # ==================================================================
    # STEP 1: Load data
    # ==================================================================
    print("\n--- Step 1: Load data ---")
    df = load_ibm_data()
    print(f"  Loaded {len(df)} minute bars")

    # ==================================================================
    # STEP 2: Calendar 15-min bars (same as SF-Harris)
    # ==================================================================
    print("\n" + "=" * 70)
    print("CALENDAR 15-MIN BARS")
    print("=" * 70)

    returns = compute_15min_returns(df)
    returns_clean = detect_and_remove_jumps(returns, n_passes=2, top_pct=0.001)
    print(f"  Clean 15-min returns: {len(returns_clean)}")

    # Spot volatility: log(r^2) adjusted for periodicity
    periodicity = estimate_periodicity(returns)
    rv_df = returns_clean.to_frame("return")
    rv_df["rv_15min"] = rv_df["return"]**2
    rv_df["time"] = rv_df.index.strftime("%H:%M")
    rv_df["f_t"] = rv_df["time"].map(periodicity).fillna(1.0)
    rv_df["rv_adj"] = rv_df["rv_15min"] / rv_df["f_t"]
    rv_df["log_spot"] = np.log(rv_df["rv_adj"].clip(lower=1e-20))

    log_spot = rv_df["log_spot"].replace([np.inf, -np.inf], np.nan).dropna()
    log_spot_vals = log_spot.values

    # Train/test split
    n = len(log_spot_vals)
    split = int(n * 0.8)
    train_spot = log_spot_vals[:split]
    test_spot = log_spot_vals[split:]

    # Also need raw returns for GARCH (not log-spot)
    train_returns = returns_clean.values[:split]
    test_returns = returns_clean.values[split:]

    print(f"  Train: {len(train_returns)} observations, Test: {len(test_returns)}")

    # ==================================================================
    # GARCH(1,1) on calendar 15-min
    # ==================================================================
    print("\n  --- GARCH(1,1) ---")
    res_garch = fit_garch(train_returns, p=1, q=1, dist="normal")
    print(f"  GARCH params: omega={res_garch.params['omega']:.4f}, "
          f"alpha={res_garch.params['alpha[1]']:.4f}, "
          f"beta={res_garch.params['beta[1]']:.4f}")
    print(f"  Persistence: {res_garch.params['alpha[1]'] + res_garch.params['beta[1]']:.4f}")

    n_test = len(test_returns)
    n_sim = 2000
    rng_garch = np.random.default_rng(111)
    _, sim_garch_ret = simulate_garch(res_garch, n_test, n_sim, rng=rng_garch)

    # Coverage: compare log(r^2) predictions vs actual log-spot
    sim_garch_log_rv = np.log(np.maximum(sim_garch_ret**2, 1e-20))
    # Adjust for periodicity
    test_times = returns_clean.index[split:split+n_test].strftime("%H:%M")
    test_f = np.array([periodicity.get(t, 1.0) for t in test_times[:n_test]])
    sim_garch_log_spot = sim_garch_log_rv - np.log(test_f[np.newaxis, :])

    # Align test length
    min_test = min(len(test_spot), sim_garch_log_spot.shape[1])
    cov_garch = compute_coverage(test_spot[:min_test], sim_garch_log_spot[:, :min_test], PROB_LEVELS)
    aad_garch = np.mean([abs(cov_garch[p] - p*100) for p in PROB_LEVELS])

    print(f"\n  GARCH(1,1) coverage (calendar 15-min):")
    print(f"  {'p':>6}  {'Coverage':>10}  {'Ideal':>6}  {'Dev':>6}")
    print(f"  {'-'*34}")
    for p in PROB_LEVELS:
        dev = cov_garch[p] - p*100
        print(f"  {p:>6.2f}  {cov_garch[p]:>9.1f}%  {p*100:>5.0f}%  {dev:>+5.1f}pp")
    print(f"  AAD = {aad_garch:.1f}pp")

    # ==================================================================
    # EGARCH(1,1) on calendar 15-min
    # ==================================================================
    print("\n  --- EGARCH(1,1) ---")
    res_egarch = fit_egarch(train_returns, p=1, q=1, dist="normal")
    print(f"  EGARCH params: omega={res_egarch.params['omega']:.4f}, "
          f"alpha={res_egarch.params['alpha[1]']:.4f}, "
          f"beta={res_egarch.params['beta[1]']:.4f}")

    rng_egarch = np.random.default_rng(222)
    _, sim_egarch_ret = simulate_egarch(res_egarch, n_test, n_sim, rng=rng_egarch)

    sim_egarch_log_rv = np.log(np.maximum(sim_egarch_ret**2, 1e-20))
    sim_egarch_log_spot = sim_egarch_log_rv - np.log(test_f[np.newaxis, :])

    cov_egarch = compute_coverage(test_spot[:min_test], sim_egarch_log_spot[:, :min_test], PROB_LEVELS)
    aad_egarch = np.mean([abs(cov_egarch[p] - p*100) for p in PROB_LEVELS])

    print(f"\n  EGARCH(1,1) coverage (calendar 15-min):")
    print(f"  {'p':>6}  {'Coverage':>10}  {'Ideal':>6}  {'Dev':>6}")
    print(f"  {'-'*34}")
    for p in PROB_LEVELS:
        dev = cov_egarch[p] - p*100
        print(f"  {p:>6.2f}  {cov_egarch[p]:>9.1f}%  {p*100:>5.0f}%  {dev:>+5.1f}pp")
    print(f"  AAD = {aad_egarch:.1f}pp")

    # ==================================================================
    # Heston on calendar 15-min
    # ==================================================================
    print("\n  --- Heston ---")
    # Fit Heston on training returns (dt = 1 unit = 15 min)
    heston_params = fit_heston(train_returns, dt=1.0)
    if heston_params is not None:
        print(f"  Heston params: kappa={heston_params['kappa']:.4f}, "
              f"theta={heston_params['theta']:.6f}, "
              f"sigma_v={heston_params['sigma_v']:.4f}, "
              f"rho={heston_params['rho']:.4f}")

        V0 = np.var(train_returns)  # initial variance
        rng_heston = np.random.default_rng(333)
        _, sim_heston_ret = simulate_heston(heston_params, V0, n_test, dt=1.0, n_sim=n_sim, rng=rng_heston)

        sim_heston_log_rv = np.log(np.maximum(sim_heston_ret**2, 1e-20))
        sim_heston_log_spot = sim_heston_log_rv - np.log(test_f[np.newaxis, :])

        cov_heston = compute_coverage(test_spot[:min_test], sim_heston_log_spot[:, :min_test], PROB_LEVELS)
        aad_heston = np.mean([abs(cov_heston[p] - p*100) for p in PROB_LEVELS])

        print(f"\n  Heston coverage (calendar 15-min):")
        print(f"  {'p':>6}  {'Coverage':>10}  {'Ideal':>6}  {'Dev':>6}")
        print(f"  {'-'*34}")
        for p in PROB_LEVELS:
            dev = cov_heston[p] - p*100
            print(f"  {p:>6.2f}  {cov_heston[p]:>9.1f}%  {p*100:>5.0f}%  {dev:>+5.1f}pp")
        print(f"  AAD = {aad_heston:.1f}pp")
    else:
        print("  Heston fit failed")
        aad_heston = None

    # ==================================================================
    # Dollar bars
    # ==================================================================
    print("\n" + "=" * 70)
    print("DOLLAR BARS")
    print("=" * 70)

    dollar_returns = build_dollar_bars(df)
    print(f"  Dollar bars: {len(dollar_returns)} observations")

    # Clean outliers
    dol_vals = dollar_returns.values
    dol_q01, dol_q99 = np.percentile(dol_vals, [0.5, 99.5])
    dol_clean = dol_vals[(dol_vals > dol_q01) & (dol_vals < dol_q99)]
    dol_log_spot = np.log(dol_clean**2 + 1e-20)
    dol_log_spot = dol_log_spot[np.isfinite(dol_log_spot)]

    n_dol = len(dol_log_spot)
    split_dol = int(n_dol * 0.8)
    dol_train = dol_clean[:split_dol]
    dol_test = dol_clean[split_dol:]
    dol_train_spot = dol_log_spot[:split_dol]
    dol_test_spot = dol_log_spot[split_dol:]

    print(f"  Train: {len(dol_train)}, Test: {len(dol_test)}")

    # GARCH on dollar bars
    print("\n  --- GARCH(1,1) on dollar bars ---")
    res_garch_dol = fit_garch(dol_train, p=1, q=1, dist="normal")
    print(f"  Params: omega={res_garch_dol.params['omega']:.4f}, "
          f"alpha={res_garch_dol.params['alpha[1]']:.4f}, "
          f"beta={res_garch_dol.params['beta[1]']:.4f}")

    n_dol_test = len(dol_test)
    rng_garch_dol = np.random.default_rng(444)
    _, sim_garch_dol_ret = simulate_garch(res_garch_dol, n_dol_test, n_sim, rng=rng_garch_dol)

    sim_garch_dol_log_rv = np.log(np.maximum(sim_garch_dol_ret**2, 1e-20))
    cov_garch_dol = compute_coverage(dol_test_spot, sim_garch_dol_log_rv, PROB_LEVELS)
    aad_garch_dol = np.mean([abs(cov_garch_dol[p] - p*100) for p in PROB_LEVELS])

    print(f"\n  GARCH(1,1) coverage (dollar bars):")
    print(f"  {'p':>6}  {'Coverage':>10}  {'Ideal':>6}  {'Dev':>6}")
    print(f"  {'-'*34}")
    for p in PROB_LEVELS:
        dev = cov_garch_dol[p] - p*100
        print(f"  {p:>6.2f}  {cov_garch_dol[p]:>9.1f}%  {p*100:>5.0f}%  {dev:>+5.1f}pp")
    print(f"  AAD = {aad_garch_dol:.1f}pp")

    # EGARCH on dollar bars
    print("\n  --- EGARCH(1,1) on dollar bars ---")
    res_egarch_dol = fit_egarch(dol_train, p=1, q=1, dist="normal")
    print(f"  Params: omega={res_egarch_dol.params['omega']:.4f}, "
          f"alpha={res_egarch_dol.params['alpha[1]']:.4f}, "
          f"beta={res_egarch_dol.params['beta[1]']:.4f}")

    rng_egarch_dol = np.random.default_rng(555)
    _, sim_egarch_dol_ret = simulate_egarch(res_egarch_dol, n_dol_test, n_sim, rng=rng_egarch_dol)

    sim_egarch_dol_log_rv = np.log(np.maximum(sim_egarch_dol_ret**2, 1e-20))
    cov_egarch_dol = compute_coverage(dol_test_spot, sim_egarch_dol_log_rv, PROB_LEVELS)
    aad_egarch_dol = np.mean([abs(cov_egarch_dol[p] - p*100) for p in PROB_LEVELS])

    print(f"\n  EGARCH(1,1) coverage (dollar bars):")
    print(f"  {'p':>6}  {'Coverage':>10}  {'Ideal':>6}  {'Dev':>6}")
    print(f"  {'-'*34}")
    for p in PROB_LEVELS:
        dev = cov_egarch_dol[p] - p*100
        print(f"  {p:>6.2f}  {cov_egarch_dol[p]:>9.1f}%  {p*100:>5.0f}%  {dev:>+5.1f}pp")
    print(f"  AAD = {aad_egarch_dol:.1f}pp")

    # Heston on dollar bars
    print("\n  --- Heston on dollar bars ---")
    heston_dol = fit_heston(dol_train, dt=1.0)
    if heston_dol is not None:
        print(f"  Heston params: kappa={heston_dol['kappa']:.4f}, "
              f"theta={heston_dol['theta']:.6f}, "
              f"sigma_v={heston_dol['sigma_v']:.4f}, "
              f"rho={heston_dol['rho']:.4f}")

        V0_dol = np.var(dol_train)
        rng_heston_dol = np.random.default_rng(666)
        _, sim_heston_dol_ret = simulate_heston(heston_dol, V0_dol, n_dol_test, dt=1.0,
                                                 n_sim=n_sim, rng=rng_heston_dol)

        sim_heston_dol_log_rv = np.log(np.maximum(sim_heston_dol_ret**2, 1e-20))
        cov_heston_dol = compute_coverage(dol_test_spot, sim_heston_dol_log_rv, PROB_LEVELS)
        aad_heston_dol = np.mean([abs(cov_heston_dol[p] - p*100) for p in PROB_LEVELS])

        print(f"\n  Heston coverage (dollar bars):")
        print(f"  {'p':>6}  {'Coverage':>10}  {'Ideal':>6}  {'Dev':>6}")
        print(f"  {'-'*34}")
        for p in PROB_LEVELS:
            dev = cov_heston_dol[p] - p*100
            print(f"  {p:>6.2f}  {cov_heston_dol[p]:>9.1f}%  {p*100:>5.0f}%  {dev:>+5.1f}pp")
        print(f"  AAD = {aad_heston_dol:.1f}pp")
    else:
        print("  Heston fit failed on dollar bars")
        aad_heston_dol = None

    # ==================================================================
    # FINAL COMPARISON
    # ==================================================================
    anz_aad = np.mean([abs(ANZARUT_TABLE3[p] - p*100) for p in PROB_LEVELS])

    print(f"\n{'='*70}")
    print("FINAL COMPARISON — ALL MODELS")
    print("=" * 70)
    print(f"\n  Anzarut's Table 3 (GIG-Harris + Gibbs-b): AAD = {anz_aad:.1f}pp")
    print(f"\n  {'Model':>50}  {'Bar':>8}  {'AAD':>6}")
    print(f"  {'-'*68}")
    print(f"  {'SF-Harris Gibbs (eps=0.1)':>50}  {'dollar':>8}  {'0.2pp':>6}")
    print(f"  {'SF-Harris Gibbs (eps=0.1)':>50}  {'15min':>8}  {'0.3pp':>6}")
    print(f"  {'SF-Harris Gibbs (eps=1e-5)':>50}  {'15min':>8}  {'0.5pp':>6}")
    print(f"  {'--- Calendar 15-min benchmarks:':>50}")
    print(f"  {'GARCH(1,1)':>50}  {'15min':>8}  {aad_garch:>5.1f}pp")
    print(f"  {'EGARCH(1,1)':>50}  {'15min':>8}  {aad_egarch:>5.1f}pp")
    if aad_heston is not None:
        print(f"  {'Heston':>50}  {'15min':>8}  {aad_heston:>5.1f}pp")
    print(f"  {'--- Dollar bar benchmarks:':>50}")
    print(f"  {'GARCH(1,1)':>50}  {'dollar':>8}  {aad_garch_dol:>5.1f}pp")
    print(f"  {'EGARCH(1,1)':>50}  {'dollar':>8}  {aad_egarch_dol:>5.1f}pp")
    if aad_heston_dol is not None:
        print(f"  {'Heston':>50}  {'dollar':>8}  {aad_heston_dol:>5.1f}pp")

    # Detailed comparison table
    print(f"\n  Detailed coverage comparison:")
    print(f"  {'p':>6}  {'Anzarut':>8}  {'SF-H-$':>7}  {'SF-15m':>7}  "
          f"{'GARCH':>6}  {'EGARCH':>7}  ", end="")
    if aad_heston is not None:
        print(f"{'Heston':>7}  ", end="")
    print(f"{'Ideal':>6}")
    print(f"  {'-'*70}")

    for p in PROB_LEVELS:
        line = f"  {p:>6.2f}  {ANZARUT_TABLE3[p]:>7.0f}%  {'0.2':>6}%  {'0.3':>6}%  "
        line += f"{cov_garch.get(p, 0):>5.1f}%  {cov_egarch.get(p, 0):>6.1f}%  "
        if aad_heston is not None:
            line += f"{cov_heston.get(p, 0):>6.1f}%  "
        line += f"{p*100:>5.0f}%"
        print(line)

    # Save results
    all_results = {
        "model": ["SF-Harris_dollar", "SF-Harris_15min", "SF-Harris_15min_e1e5",
                    "GARCH_15min", "EGARCH_15min",
                    "GARCH_dollar", "EGARCH_dollar"],
        "aad": [0.2, 0.3, 0.5, aad_garch, aad_egarch, aad_garch_dol, aad_egarch_dol],
    }
    if aad_heston is not None:
        all_results["model"].append("Heston_15min")
        all_results["aad"].append(aad_heston)
    if aad_heston_dol is not None:
        all_results["model"].append("Heston_dollar")
        all_results["aad"].append(aad_heston_dol)

    for p in PROB_LEVELS:
        coverages = [None, None, None,  # SF-Harris (from other script)
                      cov_garch.get(p, 0), cov_egarch.get(p, 0),
                      cov_garch_dol.get(p, 0), cov_egarch_dol.get(p, 0)]
        if aad_heston is not None:
            coverages.append(cov_heston.get(p, 0))
        if aad_heston_dol is not None:
            coverages.append(cov_heston_dol.get(p, 0))
        all_results[f"cov_{p}"] = coverages

    result_df = pd.DataFrame(all_results)
    result_df.to_csv(OUTPUT_DIR / "benchmark_sv_comparison.csv", index=False)
    print(f"\n  Results saved to {OUTPUT_DIR / 'benchmark_sv_comparison.csv'}")