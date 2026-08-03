"""Fit SF-Harris process with GIG invariant distribution to IBM realized variance.

This follows Anzarut Section 4:
1. Load processed IBM daily realized variance
2. Fit GIG distribution to realized variance (rescaled)
3. Also fit log-normal as comparison
4. Estimate alpha using ACF and threshold methods
5. Compare model-implied statistics with empirical data
6. Diagnostic plots (saved to data/)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import kv as bessel_kv
from scipy import stats

from src.sf_harris.distributions import GIGQ
from src.sf_harris.process import SFHarrisProcess

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"


def load_processed_rv():
    """Load the processed IBM daily realized variance."""
    filepath = OUTPUT_DIR / "ibm_daily_rv.csv"
    print(f"Loading processed data from {filepath}...")
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    return df


def acf_analysis(x: np.ndarray, max_lag: int = 20) -> np.ndarray:
    """Compute autocorrelation function."""
    n = len(x)
    acf_values = np.zeros(max_lag + 1)
    acf_values[0] = 1.0
    mean = np.mean(x)
    var = np.var(x)
    if var == 0:
        return acf_values
    for h in range(1, max_lag + 1):
        acf_values[h] = np.mean((x[:-h] - mean) * (x[h:] - mean)) / var
    return acf_values


def fit_gig_mle(data: np.ndarray, method: str = "de") -> dict:
    """Fit GIG distribution by MLE to positive data.

    Args:
        data: Positive data values.
        method: "de" for differential evolution (global), "lbfgs" for L-BFGS-B.
    """
    from scipy.optimize import differential_evolution

    n = len(data)
    data = np.asarray(data, dtype=np.float64)

    if np.any(data <= 0):
        raise ValueError("All data values must be positive for GIG fit")

    sum_x = np.sum(data)
    sum_inv_x = np.sum(1.0 / data)
    sum_log_x = np.sum(np.log(data))

    def optimal_eta(lam, kappa):
        a = kappa * sum_x
        b = -2 * n * lam
        c = -kappa * sum_inv_x
        discriminant = b**2 - 4 * a * c
        if discriminant < 0 or a <= 0:
            return 1.0 / np.mean(data)
        eta_plus = (-b + np.sqrt(discriminant)) / (2 * a)
        if eta_plus > 0:
            return eta_plus
        eta_minus = (-b - np.sqrt(discriminant)) / (2 * a)
        if eta_minus > 0:
            return eta_minus
        return 1.0 / np.mean(data)

    def neg_profile_log_lik(params):
        lam, log_kappa = params
        kappa = np.exp(log_kappa)
        if kappa < 0.01:
            return 1e10
        eta = optimal_eta(lam, kappa)
        if eta <= 0 or not np.isfinite(eta):
            return 1e10
        try:
            bessel_val = bessel_kv(lam, kappa)
            if bessel_val <= 0 or not np.isfinite(bessel_val):
                return 1e10
        except (ValueError, OverflowError):
            return 1e10

        log_norm = lam * np.log(eta) - np.log(2) - np.log(bessel_val)
        term1 = (lam - 1) * sum_log_x
        term2 = -(kappa * eta / 2) * sum_x - (kappa / (2 * eta)) * sum_inv_x
        log_lik = n * log_norm + term1 + term2
        if not np.isfinite(log_lik):
            return 1e10
        return -log_lik

    sample_mean = np.mean(data)
    sample_var = np.var(data)
    eta0 = 1.0 / max(sample_mean, 0.01)
    kappa_approx = max(1.0 / max(sample_var * eta0**2, 1e-6), 0.1)

    if method == "de":
        # Differential evolution for global optimization
        def neg_full_log_lik(params):
            """Full 3-parameter likelihood for DE optimization."""
            lam, log_kappa, log_eta = params
            kappa = np.exp(log_kappa)
            eta = np.exp(log_eta)

            if kappa < 0.001 or eta < 0.001:
                return 1e10

            try:
                bessel_val = bessel_kv(lam, kappa)
                if bessel_val <= 0 or not np.isfinite(bessel_val):
                    return 1e10
            except (ValueError, OverflowError):
                return 1e10

            log_norm = lam * np.log(eta) - np.log(2) - np.log(bessel_val)
            term1 = (lam - 1) * sum_log_x
            term2 = -(kappa * eta / 2) * sum_x - (kappa / (2 * eta)) * sum_inv_x
            log_lik = n * log_norm + term1 + term2

            if not np.isfinite(log_lik):
                return 1e10
            return -log_lik

        try:
            de_result = differential_evolution(
                neg_full_log_lik,
                bounds=[(-10, 10), (-5, 5), (-5, 5)],
                seed=42,
                maxiter=1000,
                tol=1e-8,
                polish=True,
            )
            lam_hat = de_result.x[0]
            kappa_hat = np.exp(de_result.x[1])
            eta_hat = np.exp(de_result.x[2])

            # Also try profile likelihood with DE's solution as starting point
            eta_opt = optimal_eta(lam_hat, kappa_hat)
            if eta_opt > 0 and np.isfinite(eta_opt):
                eta_hat = eta_opt

            return {
                "lam": np.clip(lam_hat, -10, 10),
                "kappa": np.clip(kappa_hat, 0.01, 200),
                "eta": np.clip(eta_hat, 0.001, 50),
                "nll": de_result.fun,
            }
        except Exception:
            pass  # Fall through to L-BFGS-B

    # L-BFGS-B with multiple restarts (fallback or method="lbfgs")
    starts = [
        [0.0, np.log(max(kappa_approx, 0.1))],
        [1.0, np.log(max(kappa_approx, 0.1))],
        [-1.0, np.log(max(kappa_approx, 0.1))],
        [0.0, np.log(max(kappa_approx * 0.5, 0.1))],
        [0.0, np.log(max(kappa_approx * 2, 0.1))],
        [-2.0, np.log(max(kappa_approx, 0.1))],
        [2.0, np.log(max(kappa_approx, 0.1))],
    ]

    best_result = None
    best_nll = np.inf

    for x0 in starts:
        try:
            result = minimize(
                neg_profile_log_lik,
                x0=x0,
                method="L-BFGS-B",
                bounds=[(-10, 10), (None, None)],
                options={"maxiter": 500},
            )
            if result.fun < best_nll:
                best_nll = result.fun
                best_result = result
        except Exception:
            continue

    if best_result is None:
        return {"lam": 0.0, "kappa": max(kappa_approx, 0.1), "eta": max(eta0, 0.01)}

    lam_hat = np.clip(best_result.x[0], -10, 10)
    kappa_hat = np.clip(np.exp(best_result.x[1]), 0.01, 200)
    eta_hat = np.clip(optimal_eta(lam_hat, kappa_hat), 0.001, 50)

    return {"lam": lam_hat, "kappa": kappa_hat, "eta": eta_hat, "nll": best_nll}


def fit_lognormal(data: np.ndarray) -> dict:
    """Fit log-normal distribution to positive data."""
    log_data = np.log(data)
    mu = np.mean(log_data)
    sigma = np.std(log_data)
    return {"mu": mu, "sigma": sigma}


def ks_goodness_of_fit(data: np.ndarray, dist_name: str, params: dict) -> dict:
    """Kolmogorov-Smirnov test for distributional fit."""
    if dist_name == "lognormal":
        mu, sigma = params["mu"], params["sigma"]
        statistic, p_value = stats.kstest(data, lambda x: stats.lognorm.cdf(x, sigma, scale=np.exp(mu)))
        return {"ks_stat": statistic, "ks_pvalue": p_value}
    return {}


def fit_sf_harris_full(rv_data: np.ndarray, log_rv_data: np.ndarray):
    """Full SF-Harris model fit to IBM realized variance."""
    print("=" * 70)
    print("SF-Harris Model Fit to IBM Realized Variance")
    print("=" * 70)

    # --- Step 1: Fit GIG to rescaled variance ---
    print("\n--- Step 1: Marginal Distribution Fit ---")
    print(f"  Variance range: [{rv_data.min():.6f}, {rv_data.max():.6f}]")
    print(f"  Variance mean: {rv_data.mean():.6f}, std: {rv_data.std():.6f}")

    # Rescale for numerical stability
    scale_factor = 1.0 / np.mean(rv_data)
    rv_scaled = rv_data * scale_factor
    print(f"  Scale factor: {scale_factor:.2f} (rescaling so mean ~ 1)")
    print(f"  Scaled range: [{rv_scaled.min():.4f}, {rv_scaled.max():.4f}]")

    # Fit GIG on rescaled data
    gig_params_scaled = fit_gig_mle(rv_scaled, method="de")
    gig_params = {
        "lam": gig_params_scaled["lam"],
        "kappa": gig_params_scaled["kappa"] / scale_factor,
        "eta": gig_params_scaled["eta"] * scale_factor,
    }
    print(f"\n  GIG parameters (rescaled):")
    print(f"    lambda = {gig_params_scaled['lam']:.4f}")
    print(f"    kappa  = {gig_params_scaled['kappa']:.4f}")
    print(f"    eta    = {gig_params_scaled['eta']:.4f}")
    if "nll" in gig_params_scaled:
        print(f"    Neg log-lik = {gig_params_scaled['nll']:.2f}")

    # Fit log-normal as comparison
    lognorm_params = fit_lognormal(rv_data)
    print(f"\n  Log-normal parameters (original scale):")
    print(f"    mu    = {lognorm_params['mu']:.4f}")
    print(f"    sigma = {lognorm_params['sigma']:.4f}")

    # Goodness of fit
    ks_lognorm = ks_goodness_of_fit(rv_data, "lognormal", lognorm_params)
    print(f"\n  KS test (log-normal): statistic={ks_lognorm['ks_stat']:.4f}, p-value={ks_lognorm['ks_pvalue']:.4f}")

    # Validate GIG by sampling
    print(f"\n  --- GIG Sampling Validation ---")
    try:
        Q_scaled = GIGQ(lam=gig_params_scaled["lam"], kappa=gig_params_scaled["kappa"], eta=gig_params_scaled["eta"])
        rng = np.random.default_rng(42)
        n = len(rv_data)
        samples_gig_scaled = Q_scaled.sample_n(n=n, rng=rng)
        samples_gig = samples_gig_scaled / scale_factor

        print(f"  Fitted GIG vs Empirical (variance scale):")
        print(f"    Mean:  {np.mean(samples_gig):.6f} vs {np.mean(rv_data):.6f}")
        print(f"    Std:   {np.std(samples_gig):.6f} vs {np.std(rv_data):.6f}")
        print(f"    Skew:  {pd.Series(samples_gig).skew():.4f} vs {pd.Series(rv_data).skew():.4f}")
        print(f"    Kurt:  {pd.Series(samples_gig).kurtosis():.4f} vs {pd.Series(rv_data).kurtosis():.4f}")
        gig_ok = True
    except Exception as e:
        print(f"  GIG sampling failed: {e}")
        gig_ok = False

    # Validate log-normal by sampling
    rng_ln = np.random.default_rng(42)
    samples_ln = rng_ln.lognormal(lognorm_params["mu"], lognorm_params["sigma"], size=len(rv_data))
    print(f"\n  Fitted Log-normal vs Empirical (variance scale):")
    print(f"    Mean:  {np.mean(samples_ln):.6f} vs {np.mean(rv_data):.6f}")
    print(f"    Std:   {np.std(samples_ln):.6f} vs {np.std(rv_data):.6f}")
    print(f"    Skew:  {pd.Series(samples_ln).skew():.4f} vs {pd.Series(rv_data).skew():.4f}")
    print(f"    Kurt:  {pd.Series(samples_ln).kurtosis():.4f} vs {pd.Series(rv_data).kurtosis():.4f}")

    # --- Step 2: ACF-based alpha estimation ---
    print("\n--- Step 2: ACF-based Alpha Estimation ---")
    acf = acf_analysis(log_rv_data, max_lag=20)

    print(f"  Autocorrelation of log-RV:")
    for lag in [1, 2, 5, 10, 20]:
        print(f"    rho({lag:2d}) = {acf[lag]:.4f}")

    # Alpha from ACF(1): SF-Harris predicts rho(h) ~ exp(-alpha*h)
    if acf[1] > 0:
        alpha_acf = -np.log(acf[1])
    else:
        alpha_acf = 50.0
    print(f"\n  Alpha from ACF(1): {alpha_acf:.4f}")
    print(f"  (Implied half-life: {np.log(2)/alpha_acf:.1f} days)")

    # Also fit exponential decay to ACF lags 1-10
    lags = np.arange(1, 11)
    acf_pos = acf[1:11]
    valid = acf_pos > 0
    if np.sum(valid) >= 2:
        log_acf = np.log(acf_pos[valid])
        slope = np.polyfit(lags[valid], log_acf, 1)[0]
        alpha_acf_fit = -slope
        print(f"  Alpha from ACF fit (lags 1-10): {alpha_acf_fit:.4f}")
        print(f"  (Implied half-life: {np.log(2)/alpha_acf_fit:.1f} days)")
    else:
        alpha_acf_fit = alpha_acf

    # --- Step 3: Threshold-based alpha ---
    print("\n--- Step 3: Threshold-based Alpha Estimation ---")
    diff = np.diff(log_rv_data)
    n_total = len(diff)
    diff_std = np.std(diff)

    threshold_alphas = {}
    for mult in [0.25, 0.5, 1.0, 1.5, 2.0]:
        threshold = diff_std * mult
        n_stays = int(np.sum(np.abs(diff) <= threshold))
        if n_stays > 0:
            alpha = -np.log(n_stays / n_total)
        else:
            alpha = 50.0
        threshold_alphas[mult] = {"alpha": alpha, "n_stays": n_stays}
        print(f"  Threshold={mult}*std ({threshold:.4f}): stays={n_stays} ({n_stays/n_total*100:.1f}%), alpha={alpha:.4f}")

    # --- Step 4: Model simulation comparison ---
    print("\n--- Step 4: Model Simulation Comparison ---")

    alpha_test = alpha_acf_fit  # use ACF-fit alpha

    if gig_ok:
        n_sims = 10
        print(f"\n  Simulating from SF-Harris(alpha={alpha_test:.2f}, GIG)...")
        print(f"  ({n_sims} replications, n={len(rv_data)} each)")

        sim_stats = {"mean": [], "std": [], "skew": [], "kurt": []}
        sim_acfs = []

        rng_sim = np.random.default_rng(42)
        for i in range(n_sims):
            process = SFHarrisProcess(
                alpha=alpha_test, Q_sample=Q_scaled.sample, Q_density=Q_scaled.density
            )
            sim = process.simulate(len(rv_data), rng=rng_sim)
            sim = sim / scale_factor  # transform back
            sim_stats["mean"].append(np.mean(sim))
            sim_stats["std"].append(np.std(sim))
            sim_stats["skew"].append(pd.Series(sim).skew())
            sim_stats["kurt"].append(pd.Series(sim).kurtosis())

            sim_log = np.log(sim)
            sim_acf = acf_analysis(sim_log, max_lag=10)
            sim_acfs.append(sim_acf)

        print(f"\n  SF-Harris vs Empirical (variance scale):")
        print(f"    Mean:  {np.mean(sim_stats['mean']):.6f} vs {np.mean(rv_data):.6f}")
        print(f"    Std:   {np.mean(sim_stats['std']):.6f} vs {np.std(rv_data):.6f}")
        print(f"    Skew:  {np.mean(sim_stats['skew']):.4f} vs {pd.Series(rv_data).skew():.4f}")
        print(f"    Kurt:  {np.mean(sim_stats['kurt']):.4f} vs {pd.Series(rv_data).kurtosis():.4f}")

        print(f"\n  ACF comparison (lag 1-5):")
        print(f"    Empirical: {', '.join(f'{acf[h]:.4f}' for h in range(1, 6))}")
        print(f"    Simulated: {', '.join(f'{np.mean([s[h] for s in sim_acfs]):.4f}' for h in range(1, 6))}")

        # Also simulate log-normal SF-Harris for comparison
        print(f"\n  Simulating from SF-Harris(alpha={alpha_test:.2f}, Log-normal)...")
        sim_ln_stats = {"mean": [], "std": [], "skew": [], "kurt": []}
        sim_ln_acfs = []

        rng_ln_sim = np.random.default_rng(123)
        mu_ln, sigma_ln = lognorm_params["mu"], lognorm_params["sigma"]

        for i in range(n_sims):
            # Log-normal Q: simulate SF-Harris where Q is log-normal
            rng_ln_sim_copy = np.random.default_rng(rng_ln_sim.integers(0, 2**31))
            # Custom sample function for log-normal
            def ln_sample(rng): return float(rng.lognormal(mu_ln, sigma_ln))
            def ln_density(x): return float(stats.lognorm.pdf(x, sigma_ln, scale=np.exp(mu_ln)))

            process_ln = SFHarrisProcess(
                alpha=alpha_test, Q_sample=ln_sample, Q_density=ln_density
            )
            sim_ln = process_ln.simulate(len(rv_data), rng=rng_ln_sim_copy)
            sim_ln_stats["mean"].append(np.mean(sim_ln))
            sim_ln_stats["std"].append(np.std(sim_ln))
            sim_ln_stats["skew"].append(pd.Series(sim_ln).skew())
            sim_ln_stats["kurt"].append(pd.Series(sim_ln).kurtosis())

            sim_ln_log = np.log(sim_ln)
            sim_ln_acf = acf_analysis(sim_ln_log, max_lag=10)
            sim_ln_acfs.append(sim_ln_acf)

        print(f"\n  SF-Harris (log-normal Q) vs Empirical:")
        print(f"    Mean:  {np.mean(sim_ln_stats['mean']):.6f} vs {np.mean(rv_data):.6f}")
        print(f"    Std:   {np.mean(sim_ln_stats['std']):.6f} vs {np.std(rv_data):.6f}")
        print(f"    Skew:  {np.mean(sim_ln_stats['skew']):.4f} vs {pd.Series(rv_data).skew():.4f}")
        print(f"    Kurt:  {np.mean(sim_ln_stats['kurt']):.4f} vs {pd.Series(rv_data).kurtosis():.4f}")

        print(f"\n  ACF comparison (lag 1-5):")
        print(f"    Empirical:              {', '.join(f'{acf[h]:.4f}' for h in range(1, 6))}")
        print(f"    SF-Harris (GIG):        {', '.join(f'{np.mean([s[h] for s in sim_acfs]):.4f}' for h in range(1, 6))}")
        print(f"    SF-Harris (log-normal): {', '.join(f'{np.mean([s[h] for s in sim_ln_acfs]):.4f}' for h in range(1, 6))}")
    else:
        print("\n  Skipping GIG simulation (GIG fitting failed)")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("Summary of SF-Harris Fit to IBM Realized Variance")
    print("=" * 70)
    print(f"\n  GIG parameters (rescaled, mean=1):")
    print(f"    lambda = {gig_params_scaled['lam']:.4f}")
    print(f"    kappa  = {gig_params_scaled['kappa']:.4f}")
    print(f"    eta    = {gig_params_scaled['eta']:.4f}")
    print(f"\n  Log-normal parameters:")
    print(f"    mu    = {lognorm_params['mu']:.4f}")
    print(f"    sigma = {lognorm_params['sigma']:.4f}")
    print(f"\n  Alpha estimates:")
    print(f"    ACF(1):       {alpha_acf:.4f}")
    print(f"    ACF fit(1-10): {alpha_acf_fit:.4f}")
    print(f"    Threshold 0.5std: {threshold_alphas[0.5]['alpha']:.4f}")
    print(f"\n  Key finding: Log-RV shows strong autocorrelation")
    print(f"    rho(1)  = {acf[1]:.4f}")
    print(f"    rho(5)  = {acf[5]:.4f}")
    print(f"    rho(10) = {acf[10]:.4f}")
    print(f"    rho(20) = {acf[20]:.4f}")
    print(f"  This slow decay suggests volatility clustering that the")
    print(f"  basic SF-Harris model (exponential ACF) may not fully capture.")

    return {
        "gig_params": gig_params,
        "gig_params_scaled": gig_params_scaled,
        "lognorm_params": lognorm_params,
        "alpha_acf": alpha_acf,
        "alpha_acf_fit": alpha_acf_fit,
        "acf": acf,
        "scale_factor": scale_factor,
    }


if __name__ == "__main__":
    df = load_processed_rv()
    rv_data = df["rv"].values
    log_rv_data = df["log_rv"].values

    print(f"\nData: {len(rv_data)} daily observations")
    print(f"  Variance range: [{rv_data.min():.6f}, {rv_data.max():.6f}]")
    print(f"  Log-RV range:   [{log_rv_data.min():.4f}, {log_rv_data.max():.4f}]")

    results = fit_sf_harris_full(rv_data, log_rv_data)