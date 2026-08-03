"""Mixture SF-Harris Table 3: Double exponential ACF + GIG-Q.

Key improvements over basic SF-Harris:
1. Double exponential ACF: r(h) = w1*exp(-a1*h) + w2*exp(-a2*h) captures long memory
2. Method-of-moments estimation (avoids poorly-converging Gibbs for ACF params)
3. Conditioning on last observed value for proper prediction
4. Multiple Q distributions: Gaussian, Student-t, GIG, empirical bootstrap
5. Bootstrap parameter uncertainty propagation

The mixture ACF captures both fast (intraday) and slow (multi-day) volatility decay,
which the single exponential ACF cannot match (R²=-13 vs R²=0.91).

The GIG distribution is the natural choice for Q in the SF-Harris process because:
- It's the invariant distribution of the Harris chain
- It has support on (0, inf) matching volatility (always positive)
- It captures both skewness and heavy tails
- Anzarut uses GIG-Q to achieve AAD=0.8pp in Table 3
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats
from scipy.special import kv as bessel_kv

DATA_PATH = Path(r"C:\Users\angve\OneDrive\Desktop\Servicio\Libros\SF-Harris\IBM.txt")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------------------
# Data loading and bar construction
# ---------------------------------------------------------------------------
def load_ibm_data(start_date="2012-01-01", end_date="2014-12-31"):
    df = pd.read_csv(
        DATA_PATH, header=None,
        names=["date", "time", "open", "high", "low", "close", "volume"],
    )
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="%m/%d/%Y %H:%M")
    df = df.drop(columns=["date", "time"]).set_index("datetime")
    df = df.loc[start_date:end_date]
    df["dollar_volume"] = df["close"] * df["volume"]
    return df


def compute_15min_returns(df):
    close_15min = df["close"].resample("15min").last().dropna()
    return np.log(close_15min).diff().dropna()


def compute_daily_rv(returns_series):
    if isinstance(returns_series, pd.DataFrame):
        ret = returns_series["return"]
    else:
        ret = returns_series
    rv = ret.groupby(ret.index.date).apply(lambda x: np.sum(x**2))
    rv.index = pd.to_datetime(rv.index)
    return rv


def build_dollar_bars(df, dollar_threshold):
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
    return result


# ---------------------------------------------------------------------------
# ACF analysis and fitting
# ---------------------------------------------------------------------------
def acf_analysis(x, max_lag=40):
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


def fit_double_exponential(lags, acf):
    """Fit r(h) = w1*exp(-a1*h) + w2*exp(-a2*h) to empirical ACF.

    Uses weighted least squares with weight = 1/(1+h) to emphasize short lags.
    Sigmoid for w in [0,1], log-transform for alphas > 0.
    """
    def neg_log_lik(params):
        log_w1, log_alpha1, log_alpha2 = params
        w1 = 1.0 / (1.0 + np.exp(-log_w1))
        w2 = 1.0 - w1
        alpha1 = np.exp(log_alpha1)
        alpha2 = np.exp(log_alpha2)
        fitted = w1 * np.exp(-alpha1 * lags) + w2 * np.exp(-alpha2 * lags)
        weights = 1.0 / (1 + lags)
        return np.sum(weights * (acf - fitted)**2)

    best_result = None
    best_nll = np.inf
    starts = [
        [0.0, np.log(1.0), np.log(0.1)],
        [1.0, np.log(2.0), np.log(0.05)],
        [-1.0, np.log(0.5), np.log(0.01)],
        [0.5, np.log(3.0), np.log(0.02)],
        [-0.5, np.log(1.5), np.log(0.08)],
    ]
    for x0 in starts:
        try:
            result = minimize(neg_log_lik, x0=x0, method="L-BFGS-B",
                              bounds=[(-5, 5), (-5, 5), (-5, 5)],
                              options={"maxiter": 1000})
            if result.fun < best_nll:
                best_nll = result.fun
                best_result = result
        except Exception:
            continue

    if best_result is None:
        return {"w1": 0.5, "alpha1": 1.0, "w2": 0.5, "alpha2": 0.1, "r_squared": 0}

    w1 = 1.0 / (1.0 + np.exp(-best_result.x[0]))
    w2 = 1.0 - w1
    alpha1 = np.exp(best_result.x[1])
    alpha2 = np.exp(best_result.x[2])

    fitted = w1 * np.exp(-alpha1 * lags) + w2 * np.exp(-alpha2 * lags)
    valid = acf > 0
    ss_res = np.sum((acf[valid] - fitted[valid])**2)
    ss_tot = np.sum((acf[valid] - np.mean(acf[valid]))**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return {"w1": w1, "alpha1": alpha1, "w2": w2, "alpha2": alpha2, "r_squared": r_squared}


# ---------------------------------------------------------------------------
# Method-of-moments estimation for mixture SF-Harris
# ---------------------------------------------------------------------------
def mixture_mom_estimate(log_rv, max_lag=20):
    """Estimate mixture SF-Harris parameters from ACF and variance.

    Key relationships:
      ACF(h) = R * r(h) where r(h) = w1*exp(-a1*h) + w2*exp(-a2*h)
      R = signal ratio = sigma_Q^2 / (sigma_Q^2 + sigma_obs^2)
      sigma_Q^2 = Var * R, sigma_obs^2 = Var * (1-R)

    Returns dict with all parameters.
    """
    mu = np.mean(log_rv)
    var = np.var(log_rv)
    emp_acf = acf_analysis(log_rv, max_lag=max_lag)
    lags = np.arange(1, max_lag + 1)
    acf_vec = emp_acf[1:max_lag + 1]

    # Fit double exponential to ACF
    acf_params = fit_double_exponential(lags, acf_vec)

    w1 = acf_params["w1"]
    alpha1 = acf_params["alpha1"]
    w2 = acf_params["w2"]
    alpha2 = acf_params["alpha2"]

    # Compute r(1) = theoretical ACF at lag 1
    r1 = w1 * np.exp(-alpha1) + w2 * np.exp(-alpha2)

    # R = ACF(1) / r(1) — signal ratio (fraction of variance from process, not noise)
    # When R > 1, the fitted ACF underestimates at lag 1; model log(RV) directly
    # without observation noise decomposition (sigma_obs ≈ 0, sigma_Q ≈ std).
    R_raw = emp_acf[1] / r1 if r1 > 0 else 0.5
    if R_raw > 1.0:
        # ACF(1) > r(1): mixture ACF decays too fast at lag 1.
        # Model log(RV) directly (no observation noise decomposition).
        R = 0.999
        sigma_Q = np.sqrt(var)
        sigma_obs = np.sqrt(var * 0.001)  # tiny for numerical stability
    else:
        R = np.clip(R_raw, 0.01, 0.99)
        sigma_Q_sq = var * R
        sigma_obs_sq = var * (1 - R)
        if sigma_Q_sq < 0.001:
            sigma_Q_sq = 0.001
        if sigma_obs_sq < 0.001:
            sigma_obs_sq = 0.001
        sigma_Q = np.sqrt(sigma_Q_sq)
        sigma_obs = np.sqrt(sigma_obs_sq)

    # Single exponential for comparison
    alpha_single = -np.log(max(emp_acf[1], 0.01))

    return {
        "mu": mu, "var": var, "sigma_Q": sigma_Q, "sigma_obs": sigma_obs,
        "R": R, "w1": w1, "alpha1": alpha1, "w2": w2, "alpha2": alpha2,
        "r_squared": acf_params["r_squared"],
        "r1": r1, "alpha_single": alpha_single,
        "acf": emp_acf,
    }


def bootstrap_mixture(log_rv, n_boot=500, rng=None, max_lag=20):
    """Bootstrap mixture SF-Harris parameter estimates for uncertainty propagation."""
    if rng is None:
        rng = np.random.default_rng()
    n = len(log_rv)
    boot_params = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_sample = log_rv[idx]
        try:
            params = mixture_mom_estimate(boot_sample, max_lag=max_lag)
            boot_params.append(params)
        except Exception:
            continue
    return boot_params


# ---------------------------------------------------------------------------
# Simulation: mixture SF-Harris with observation noise
# ---------------------------------------------------------------------------
def simulate_mixture(log_rv_train, log_rv_test, params, n_sim=2000,
                     Q_type="normal", Q_df=5, rng=None, last_val=None,
                     bootstrap_params=None):
    """Simulate mixture SF-Harris predictive trajectories.

    Uses the continuous-time renewal construction from Anzarut Section 7.
    Conditions on last observed value for proper prediction.
    Q_type: "normal", "t", "empirical", "gig"
    """
    if rng is None:
        rng = np.random.default_rng()

    n_test = len(log_rv_test)
    mu = params["mu"]
    sigma_Q = params["sigma_Q"]
    sigma_obs = params["sigma_obs"]
    w1, alpha1, w2, alpha2 = params["w1"], params["alpha1"], params["w2"], params["alpha2"]
    w_mix = np.array([w1, w2])
    w_norm = w_mix / np.sum(w_mix)
    alphas = np.array([alpha1, alpha2])

    # Pre-generate random numbers for speed
    max_jumps_per_day = 50  # safety limit for inner loop
    n_total = n_sim * n_test

    start_val = last_val if last_val is not None else mu
    simulated = np.empty((n_sim, n_test))

    # Pre-generate Q draws based on type
    if Q_type == "gig":
        rv_train = np.exp(log_rv_train)
        gig_params = fit_gig_moments(rv_train)
        from scipy.stats import geninvgauss
        lam, kappa, eta = gig_params
        # Pre-generate many GIG samples (much faster than one-by-one)
        n_q_draws = n_sim * n_test * max_jumps_per_day // 5  # ~10% used
        gig_z = geninvgauss.rvs(lam, kappa, size=n_q_draws, random_state=rng)
        gig_draws = np.log((kappa / eta) * gig_z)
        gig_draws = gig_draws[gig_draws > -50]  # filter extreme values
        q_idx = 0

    emp_data = log_rv_train - mu if Q_type == "empirical" else None

    for s in range(n_sim):
        # Optionally draw from bootstrap parameter uncertainty
        if bootstrap_params is not None:
            bp = bootstrap_params[rng.integers(0, len(bootstrap_params))]
            sim_mu = bp["mu"]
            sim_sigma_Q = bp["sigma_Q"]
            sim_sigma_obs = bp["sigma_obs"]
            sim_w1, sim_alpha1 = bp["w1"], bp["alpha1"]
            sim_w2, sim_alpha2 = bp["w2"], bp["alpha2"]
            sim_w_norm = np.array([sim_w1, sim_w2])
            sim_w_norm = sim_w_norm / np.sum(sim_w_norm)
            sim_alphas = np.array([sim_alpha1, sim_alpha2])
            # Variance scaling
            sim_model_var = sim_sigma_Q**2 + sim_sigma_obs**2
            sim_data_var = np.var(log_rv_train)
            if sim_model_var > 0:
                vscale = np.sqrt(sim_data_var / sim_model_var)
                sim_sigma_Q *= vscale
                sim_sigma_obs *= vscale
        else:
            sim_mu, sim_sigma_Q, sim_sigma_obs = mu, sigma_Q, sigma_obs
            sim_w_norm, sim_alphas = w_norm, alphas

        current_val = start_val

        for i in range(n_test):
            t = 0.0
            while t < 1.0:
                component = rng.choice(len(sim_w_norm), p=sim_w_norm)
                rho = sim_alphas[component]
                wait = rng.exponential(1.0 / rho)
                t += wait
                if t < 1.0:
                    # Jump: draw from Q
                    if Q_type == "normal":
                        current_val = rng.normal(sim_mu, sim_sigma_Q)
                    elif Q_type == "t":
                        scale = sim_sigma_Q * np.sqrt((Q_df - 2) / Q_df) if Q_df > 2 else sim_sigma_Q
                        current_val = sim_mu + rng.standard_t(Q_df) * scale
                    elif Q_type == "empirical":
                        current_val = sim_mu + rng.choice(emp_data)
                    elif Q_type == "gig":
                        if q_idx < len(gig_draws):
                            current_val = gig_draws[q_idx]
                            q_idx += 1
                        else:
                            current_val = rng.normal(sim_mu, sim_sigma_Q)

            simulated[s, i] = current_val + rng.normal(0, sim_sigma_obs)

    return simulated


def fit_gig_moments(rv_data):
    """Fit GIG parameters using method-of-moments from realized variance data.

    GIG density: f(x; lam, kappa, eta) proportional to
      x^{lam-1} * exp{-(kappa/2)(eta*x + 1/(eta*x))}

    Method-of-moments: match mean and variance of RV data.
    For the hyperbolic case (lam=-0.5), the GIG reduces to a
    hyperbolic distribution which is better behaved.
    """
    mean_rv = np.mean(rv_data)
    var_rv = np.var(rv_data)

    # Use lam = -0.5 (hyperbolic case, well-behaved)
    lam = -0.5

    # For lam = -0.5:
    # E[X] = eta/kappa * K_{0.5}(kappa) / K_{-0.5}(kappa) ≈ eta/kappa * 1
    # Since K_{0.5}(z) = K_{-0.5}(z) = sqrt(pi/(2z)) * exp(-z)
    # E[X] = eta/kappa
    # Var[X] = (eta/kappa)^2 * kappa / K_{-0.5}(kappa) * ... (complex)
    #
    # Simplified: set eta so that E[X] = mean_rv → eta = kappa * mean_rv
    # Then adjust kappa to match variance
    # For large kappa, GIG → Normal(eta/kappa, (eta/kappa)^2/kappa)
    # So sigma^2 ≈ (eta/kappa)^2 / kappa = mean_rv^2 / kappa
    # kappa ≈ mean_rv^2 / var_rv

    kappa = mean_rv**2 / var_rv if var_rv > 0 else 1.0
    kappa = max(kappa, 0.01)  # avoid degenerate
    eta = kappa / mean_rv  # so that E[X] ≈ mean_rv

    return (lam, kappa, eta)


# ---------------------------------------------------------------------------
# Coverage validation
# ---------------------------------------------------------------------------
def compute_coverage(actual, simulated, prob_levels):
    n_obs = len(actual)
    min_len = min(n_obs, simulated.shape[1])
    act = actual[:min_len]
    sims = simulated[:, :min_len]

    coverage = {}
    for p in prob_levels:
        lower = np.quantile(sims, (1 - p) / 2, axis=0)
        upper = np.quantile(sims, (1 + p) / 2, axis=0)
        inside = np.sum((act >= lower) & (act <= upper))
        coverage[p] = inside / min_len * 100
    return coverage


# ---------------------------------------------------------------------------
# Hybrid Gibbs sampler: ACF fixed at MoM, mu/sigma Gibbs-sampled
# ---------------------------------------------------------------------------
def gibbs_mixture(log_rv, acf_params, n_iter=3000, burn_in=1000, rng=None):
    """Gibbs sampler for mu and sigma with ACF parameters fixed at MoM estimates.

    The ACF parameters (w1, alpha1, w2, alpha2) are fixed at their MoM values
    because the MH step for them has poor convergence. Instead, we propagate
    uncertainty in mu and sigma through conjugate updates, which mixes well.

    This produces a proper joint posterior for (mu, sigma) that captures the
    correlation between them, leading to wider prediction intervals.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(log_rv)
    mu = np.mean(log_rv)
    sigma = np.std(log_rv)

    # Prior hyperparameters (weakly informative)
    kappa0 = 0.01
    mu0 = mu
    a0 = 3.0
    b0 = sigma**2

    samples = {
        "mu": np.empty(n_iter),
        "sigma": np.empty(n_iter),
    }

    mu_cur = mu
    sigma_cur = sigma

    for it in range(n_iter):
        # Sample mu | data, sigma (conjugate Normal-Normal)
        kappa_n = kappa0 + n
        mu_n = (kappa0 * mu0 + n * np.mean(log_rv)) / kappa_n
        mu_cur = rng.normal(mu_n, sigma_cur / np.sqrt(kappa_n))

        # Sample sigma^2 | data, mu (conjugate Normal-Inverse-Gamma)
        ss = np.sum((log_rv - mu_cur)**2)
        a_n = a0 + n / 2.0
        b_n = b0 + ss / 2.0 + 0.5 * kappa0 * n * (np.mean(log_rv) - mu0)**2 / (kappa0 + n)
        sigma2_cur = 1.0 / rng.gamma(a_n, 1.0 / b_n)
        sigma_cur = np.sqrt(max(sigma2_cur, 0.01))

        samples["mu"][it] = mu_cur
        samples["sigma"][it] = sigma_cur

    return {k: v[burn_in:] for k, v in samples.items()}


def simulate_mixture_gibbs(log_rv_train, log_rv_test, acf_params, gibbs_posterior,
                            n_sim=1000, Q_type="empirical", rng=None, last_val=None,
                            bootstrap_params=None):
    """Simulate mixture SF-Harris with Gibbs posterior for mu/sigma.

    ACF parameters drawn from bootstrap (which varies them properly).
    mu and sigma drawn from Gibbs posterior (proper Bayesian joint uncertainty).
    This combines the best of both: Bayesian marginal uncertainty + ACF parameter variation.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_test = len(log_rv_test)
    R = acf_params["R"]
    var_train = np.var(log_rv_train)

    # Empirical data for Q
    emp_data = log_rv_train - np.mean(log_rv_train)  # centered

    n_post = len(gibbs_posterior["mu"])
    n_boot = len(bootstrap_params) if bootstrap_params else 0
    start_val = last_val if last_val is not None else np.mean(log_rv_train)
    simulated = np.empty((n_sim, n_test))

    for s in range(n_sim):
        # Draw mu/sigma from Gibbs posterior (proper Bayesian joint uncertainty)
        idx = rng.integers(0, n_post)
        sim_mu = gibbs_posterior["mu"][idx]
        sim_sigma = gibbs_posterior["sigma"][idx]

        # Draw ACF parameters from bootstrap (captures ACF variation)
        if n_boot > 0:
            bp = bootstrap_params[rng.integers(0, n_boot)]
            sim_w1, sim_alpha1 = bp["w1"], bp["alpha1"]
            sim_w2, sim_alpha2 = bp["w2"], bp["alpha2"]
        else:
            sim_w1, sim_alpha1 = acf_params["w1"], acf_params["alpha1"]
            sim_w2, sim_alpha2 = acf_params["w2"], acf_params["alpha2"]

        sim_w_norm = np.array([sim_w1, sim_w2])
        sim_w_norm = sim_w_norm / np.sum(sim_w_norm)
        sim_alphas = np.array([sim_alpha1, sim_alpha2])

        # Variance decomposition using Gibbs sigma
        sim_sigma_Q = sim_sigma * np.sqrt(min(R, 0.999))
        sim_sigma_obs = sim_sigma * np.sqrt(max(1 - min(R, 0.999), 0.001))

        current_val = start_val

        for i in range(n_test):
            t = 0.0
            while t < 1.0:
                component = rng.choice(len(sim_w_norm), p=sim_w_norm)
                rho = sim_alphas[component]
                wait = rng.exponential(1.0 / rho)
                t += wait
                if t < 1.0:
                    if Q_type == "empirical":
                        current_val = sim_mu + rng.choice(emp_data)
                    elif Q_type == "normal":
                        current_val = rng.normal(sim_mu, sim_sigma_Q)
                    else:
                        current_val = rng.normal(sim_mu, sim_sigma_Q)

            simulated[s, i] = current_val + rng.normal(0, sim_sigma_obs)

    return simulated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Mixture SF-Harris Table 3: Double Exponential ACF + GIG-Q")
    print("Key improvement: r(h) = w1*exp(-a1*h) + w2*exp(-a2*h) captures long memory")
    print("=" * 70)

    anzarut_coverage = {0.25: 25, 0.50: 51, 0.75: 75, 0.85: 84, 0.90: 89, 0.95: 93}
    prob_levels = [0.25, 0.50, 0.75, 0.85, 0.90, 0.95]

    df = load_ibm_data()
    print(f"Loaded {len(df)} 1-minute bars")

    # ====================================================================
    # PIPELINE A: Calendar bars
    # ====================================================================
    print(f"\n{'='*70}")
    print("CALENDAR BARS")
    print("=" * 70)

    returns_15min = compute_15min_returns(df)
    daily_rv_cal = compute_daily_rv(returns_15min)
    log_rv_cal = np.log(daily_rv_cal.values)
    valid_cal = np.isfinite(log_rv_cal)
    log_rv_cal_clean = log_rv_cal[valid_cal]
    valid_dates_cal = daily_rv_cal.index[valid_cal]

    n_total = len(log_rv_cal_clean)
    split_idx = int(n_total * 0.8)
    train_cal = log_rv_cal_clean[:split_idx]
    test_cal = log_rv_cal_clean[split_idx:]

    print(f"  Train: {len(train_cal)} days, Test: {len(test_cal)} days")
    print(f"  Train: mean={np.mean(train_cal):.4f}, std={np.std(train_cal):.4f}, "
          f"skew={stats.skew(train_cal):.2f}, kurt={stats.kurtosis(train_cal):.2f}")

    # Fit mixture ACF
    cal_params = mixture_mom_estimate(train_cal)
    print(f"\n  Mixture ACF fit (calendar):")
    print(f"    w1={cal_params['w1']:.3f}, a1={cal_params['alpha1']:.3f} (half-life={np.log(2)/cal_params['alpha1']:.1f}d)")
    print(f"    w2={cal_params['w2']:.3f}, a2={cal_params['alpha2']:.4f} (half-life={np.log(2)/cal_params['alpha2']:.0f}d)")
    print(f"    R²={cal_params['r_squared']:.4f}")
    print(f"    R (signal ratio)={cal_params['R']:.4f}")
    print(f"    sigma_Q={cal_params['sigma_Q']:.4f}, sigma_obs={cal_params['sigma_obs']:.4f}")

    # ACF comparison
    emp_acf = cal_params["acf"]
    print(f"\n  ACF comparison (calendar):")
    print(f"    {'Lag':>4}  {'Empirical':>10}  {'Mixture':>10}  {'Single':>10}")
    for h in [1, 2, 5, 10, 20]:
        mix_acf = cal_params["R"] * (cal_params["w1"] * np.exp(-cal_params["alpha1"] * h) +
                                       cal_params["w2"] * np.exp(-cal_params["alpha2"] * h))
        single_acf = cal_params["R"] * np.exp(-cal_params["alpha_single"] * h)
        print(f"    {h:>4}  {emp_acf[h]:>10.4f}  {mix_acf:>10.4f}  {single_acf:>10.4f}")

    # Bootstrap for parameter uncertainty
    print(f"\n  Bootstrapping parameters (200 resamples)...")
    rng_boot_cal = np.random.default_rng(42)
    cal_boot = bootstrap_mixture(train_cal, n_boot=200, rng=rng_boot_cal)
    print(f"  Got {len(cal_boot)} bootstrap estimates")

    # GIG parameters for calendar bars
    rv_train_cal = np.exp(train_cal)
    gig_params_cal = fit_gig_moments(rv_train_cal)
    print(f"  GIG parameters (calendar): lam={gig_params_cal[0]:.2f}, "
          f"kappa={gig_params_cal[1]:.2f}, eta={gig_params_cal[2]:.6f}")

    # Hybrid Gibbs: ACF fixed at MoM, mu/sigma from conjugate posterior
    print(f"\n  Running hybrid Gibbs sampler (3000 iterations, mu/sigma only)...")
    rng_gibbs_cal = np.random.default_rng(789)
    cal_gibbs = gibbs_mixture(train_cal, cal_params, n_iter=3000, burn_in=1000, rng=rng_gibbs_cal)
    print(f"    mu:  mean={np.mean(cal_gibbs['mu']):.4f}, std={np.std(cal_gibbs['mu']):.4f}")
    print(f"    sigma: mean={np.mean(cal_gibbs['sigma']):.4f}, std={np.std(cal_gibbs['sigma']):.4f}")

    # Simulate with different Q distributions (bootstrap)
    n_sim = 1000
    n_test_cal = len(test_cal)
    Q_types = ["normal", "t", "empirical", "gig"]
    cal_simulations = {}

    for Q_type in Q_types:
        print(f"\n  Simulating {n_sim} trajectories ({Q_type} Q, bootstrap)...")
        rng_sim = np.random.default_rng(123)
        cal_simulations[Q_type] = simulate_mixture(
            train_cal, test_cal, cal_params, n_sim=n_sim,
            Q_type=Q_type, Q_df=5, rng=rng_sim,
            last_val=train_cal[-1], bootstrap_params=cal_boot
        )

    # Simulate with Gibbs posterior + empirical Q
    print(f"\n  Simulating {n_sim} trajectories (Gibbs+bootstrap + empirical Q)...")
    rng_gibbs_sim = np.random.default_rng(321)
    cal_simulations["gibbs_empirical"] = simulate_mixture_gibbs(
        train_cal, test_cal, cal_params, cal_gibbs,
        n_sim=n_sim, Q_type="empirical", rng=rng_gibbs_sim,
        last_val=train_cal[-1], bootstrap_params=cal_boot
    )

    # Simulate with Gibbs posterior + normal Q
    print(f"  Simulating {n_sim} trajectories (Gibbs+bootstrap + normal Q)...")
    rng_gibbs_norm = np.random.default_rng(322)
    cal_simulations["gibbs_normal"] = simulate_mixture_gibbs(
        train_cal, test_cal, cal_params, cal_gibbs,
        n_sim=n_sim, Q_type="normal", rng=rng_gibbs_norm,
        last_val=train_cal[-1], bootstrap_params=cal_boot
    )

    # ====================================================================
    # PIPELINE B: Dollar bars
    # ====================================================================
    print(f"\n{'='*70}")
    print("DOLLAR BARS")
    print("=" * 70)

    train_end_date = valid_dates_cal[split_idx - 1].date() if hasattr(valid_dates_cal[split_idx-1], 'date') else valid_dates_cal[split_idx-1]
    df_train = df[df.index.date <= train_end_date]
    train_threshold = df_train["dollar_volume"].resample("15min").sum().mean()

    dollar_bars = build_dollar_bars(df, dollar_threshold=train_threshold)
    dol_rv = compute_daily_rv(dollar_bars)
    log_rv_dol = np.log(dol_rv.values)
    valid_dol = np.isfinite(log_rv_dol)
    log_rv_dol_clean = log_rv_dol[valid_dol]
    dol_dates = dol_rv.index[valid_dol]

    dol_train_mask = np.array([d.date() <= train_end_date for d in dol_dates])
    train_dol = log_rv_dol_clean[dol_train_mask]
    test_dol = log_rv_dol_clean[~dol_train_mask]

    print(f"  Train: {len(train_dol)} days, Test: {len(test_dol)} days")
    print(f"  Train: mean={np.mean(train_dol):.4f}, std={np.std(train_dol):.4f}, "
          f"skew={stats.skew(train_dol):.2f}, kurt={stats.kurtosis(train_dol):.2f}")

    dol_params = mixture_mom_estimate(train_dol)
    print(f"\n  Mixture ACF fit (dollar):")
    print(f"    w1={dol_params['w1']:.3f}, a1={dol_params['alpha1']:.3f} (half-life={np.log(2)/dol_params['alpha1']:.1f}d)")
    print(f"    w2={dol_params['w2']:.3f}, a2={dol_params['alpha2']:.4f} (half-life={np.log(2)/dol_params['alpha2']:.0f}d)")
    print(f"    R²={dol_params['r_squared']:.4f}")
    print(f"    R (signal ratio)={dol_params['R']:.4f}")
    print(f"    sigma_Q={dol_params['sigma_Q']:.4f}, sigma_obs={dol_params['sigma_obs']:.4f}")

    # Bootstrap for dollar bars
    print(f"\n  Bootstrapping parameters (200 resamples)...")
    rng_boot_dol = np.random.default_rng(42)
    dol_boot = bootstrap_mixture(train_dol, n_boot=200, rng=rng_boot_dol)
    print(f"  Got {len(dol_boot)} bootstrap estimates")

    # Hybrid Gibbs for dollar bars
    print(f"\n  Running hybrid Gibbs sampler (3000 iterations, mu/sigma only)...")
    rng_gibbs_dol = np.random.default_rng(789)
    dol_gibbs = gibbs_mixture(train_dol, dol_params, n_iter=3000, burn_in=1000, rng=rng_gibbs_dol)
    print(f"    mu:  mean={np.mean(dol_gibbs['mu']):.4f}, std={np.std(dol_gibbs['mu']):.4f}")
    print(f"    sigma: mean={np.mean(dol_gibbs['sigma']):.4f}, std={np.std(dol_gibbs['sigma']):.4f}")

    # GIG parameters for dollar bars
    rv_train_dol = np.exp(train_dol)
    gig_params_dol = fit_gig_moments(rv_train_dol)

    # Simulate (bootstrap)
    n_test_dol = len(test_dol)
    dol_simulations = {}

    for Q_type in Q_types:
        print(f"\n  Simulating {n_sim} trajectories ({Q_type} Q, bootstrap)...")
        rng_sim = np.random.default_rng(456)
        dol_simulations[Q_type] = simulate_mixture(
            train_dol, test_dol, dol_params, n_sim=n_sim,
            Q_type=Q_type, Q_df=5, rng=rng_sim,
            last_val=train_dol[-1], bootstrap_params=dol_boot
        )

    # Simulate with Gibbs posterior + empirical Q
    print(f"\n  Simulating {n_sim} trajectories (Gibbs+bootstrap + empirical Q)...")
    rng_gibbs_sim = np.random.default_rng(654)
    dol_simulations["gibbs_empirical"] = simulate_mixture_gibbs(
        train_dol, test_dol, dol_params, dol_gibbs,
        n_sim=n_sim, Q_type="empirical", rng=rng_gibbs_sim,
        last_val=train_dol[-1], bootstrap_params=dol_boot
    )

    # Simulate with Gibbs posterior + normal Q
    print(f"  Simulating {n_sim} trajectories (Gibbs+bootstrap + normal Q)...")
    rng_gibbs_norm = np.random.default_rng(655)
    dol_simulations["gibbs_normal"] = simulate_mixture_gibbs(
        train_dol, test_dol, dol_params, dol_gibbs,
        n_sim=n_sim, Q_type="normal", rng=rng_gibbs_norm,
        last_val=train_dol[-1], bootstrap_params=dol_boot
    )

    # ====================================================================
    # TABLE 3: Coverage comparison
    # ====================================================================
    print(f"\n{'='*70}")
    print("TABLE 3: Coverage Validation")
    print("=" * 70)

    print(f"\n  Anzarut's Table 3 (GIG-Harris + Gibbs-b):")
    print(f"  {'p':>6}  {'Anzarut':>10}  {'Ideal':>6}")
    print(f"  {'-'*28}")
    for p in prob_levels:
        print(f"  {p:>6.2f}  {anzarut_coverage[p]:>9.0f}%  {p*100:>5.0f}%")

    # Calendar bars results
    cal_coverages = {}
    cal_aads = {}

    print(f"\n  Calendar bars — Mixture SF-Harris:")
    all_Q_types = Q_types + ["gibbs_empirical", "gibbs_normal"]
    for Q_type in all_Q_types:
        cov = compute_coverage(test_cal, cal_simulations[Q_type], prob_levels)
        cal_coverages[Q_type] = cov
        aad = np.mean([abs(cov[p] - p * 100) for p in prob_levels])
        cal_aads[Q_type] = aad

        if Q_type == "gibbs_empirical":
            label = "Gibbs+Emp"
        elif Q_type == "gibbs_normal":
            label = "Gibbs+Norm"
        elif Q_type == "t":
            label = "Student-t"
        else:
            label = Q_type.upper()
        print(f"\n  Calendar + Mixture + {label}:")
        print(f"  {'p':>6}  {'Coverage':>10}  {'Ideal':>6}  {'Dev':>6}")
        print(f"  {'-'*34}")
        for p in prob_levels:
            dev = cov[p] - p * 100
            print(f"  {p:>6.2f}  {cov[p]:>9.1f}%  {p*100:>5.0f}%  {dev:>+5.1f}pp")
        print(f"  AAD = {aad:.1f}pp")

    # Dollar bars results
    dol_coverages = {}
    dol_aads = {}

    print(f"\n  Dollar bars — Mixture SF-Harris:")
    for Q_type in all_Q_types:
        cov = compute_coverage(test_dol, dol_simulations[Q_type], prob_levels)
        dol_coverages[Q_type] = cov
        aad = np.mean([abs(cov[p] - p * 100) for p in prob_levels])
        dol_aads[Q_type] = aad

        if Q_type == "gibbs_empirical":
            label = "Gibbs+Emp"
        elif Q_type == "gibbs_normal":
            label = "Gibbs+Norm"
        elif Q_type == "t":
            label = "Student-t"
        else:
            label = Q_type.upper()
        print(f"\n  Dollar + Mixture + {label}:")
        print(f"  {'p':>6}  {'Coverage':>10}  {'Ideal':>6}  {'Dev':>6}")
        print(f"  {'-'*34}")
        for p in prob_levels:
            dev = cov[p] - p * 100
            print(f"  {p:>6.2f}  {cov[p]:>9.1f}%  {p*100:>5.0f}%  {dev:>+5.1f}pp")
        print(f"  AAD = {aad:.1f}pp")

    # ====================================================================
    # Side-by-side comparison
    # ====================================================================
    print(f"\n{'='*70}")
    print("SIDE-BY-SIDE COMPARISON — BEST CONFIGURATIONS")
    print("=" * 70)

    print(f"\n  {'p':>6}  {'Anzarut':>8}  {'Dol+Emp':>8}  {'Dol+Gibbs':>10}  {'Cal+Emp':>8}  {'Cal+Gibbs':>10}  {'Ideal':>6}")
    print(f"  {'-'*62}")
    for p in prob_levels:
        anz = anzarut_coverage[p]
        dol_emp = dol_coverages["empirical"].get(p, 0)
        dol_gibbs = dol_coverages["gibbs_empirical"].get(p, 0)
        cal_emp = cal_coverages["empirical"].get(p, 0)
        cal_gibbs = cal_coverages["gibbs_empirical"].get(p, 0)
        print(f"  {p:>6.2f}  {anz:>7.0f}%  {dol_emp:>7.1f}%  {dol_gibbs:>9.1f}%  {cal_emp:>7.1f}%  {cal_gibbs:>9.1f}%  {p*100:>5.0f}%")

    print(f"\n  AAD comparison (lower is better):")
    print(f"    Anzarut (GIG+Gibbs, intraday):  0.8pp")
    print(f"    Basic single exp (FFBS):        4.6pp")
    print(f"    Basic single exp (MoM):         5.2pp")
    print(f"    ---")
    for Q_type in all_Q_types:
        if Q_type == "gibbs_empirical":
            label = "Gibbs+Emp"
        elif Q_type == "gibbs_normal":
            label = "Gibbs+Norm"
        elif Q_type == "t":
            label = "Student-t"
        else:
            label = Q_type.upper()
        print(f"    Mixture Cal {label:>10}: {cal_aads[Q_type]:.1f}pp")
    print(f"    ---")
    for Q_type in all_Q_types:
        if Q_type == "gibbs_empirical":
            label = "Gibbs+Emp"
        elif Q_type == "gibbs_normal":
            label = "Gibbs+Norm"
        elif Q_type == "t":
            label = "Student-t"
        else:
            label = Q_type.upper()
        print(f"    Mixture Dol {label:>10}: {dol_aads[Q_type]:.1f}pp")

    # ====================================================================
    # Distribution diagnostics
    # ====================================================================
    print(f"\n{'='*70}")
    print("DISTRIBUTION DIAGNOSTICS")
    print("=" * 70)

    print(f"\n  Calendar bars - log-RV distribution:")
    print(f"    Actual:  mean={np.mean(test_cal):.3f}, std={np.std(test_cal):.3f}, "
          f"skew={stats.skew(test_cal):.2f}, kurt={stats.kurtosis(test_cal):.2f}")
    for Q_type in all_Q_types:
        flat = cal_simulations[Q_type].flatten()
        if Q_type == "gibbs_empirical":
            label = "Gibbs+Emp"
        elif Q_type == "gibbs_normal":
            label = "Gibbs+Norm"
        elif Q_type == "t":
            label = "Student-t"
        else:
            label = Q_type.upper()
        print(f"    {label:>10}: mean={np.mean(flat):.3f}, std={np.std(flat):.3f}, "
              f"skew={stats.skew(flat):.2f}, kurt={stats.kurtosis(flat):.2f}")

    print(f"\n  Dollar bars - log-RV distribution:")
    print(f"    Actual:  mean={np.mean(test_dol):.3f}, std={np.std(test_dol):.3f}, "
          f"skew={stats.skew(test_dol):.2f}, kurt={stats.kurtosis(test_dol):.2f}")
    for Q_type in all_Q_types:
        flat = dol_simulations[Q_type].flatten()
        if Q_type == "gibbs_empirical":
            label = "Gibbs+Emp"
        elif Q_type == "gibbs_normal":
            label = "Gibbs+Norm"
        elif Q_type == "t":
            label = "Student-t"
        else:
            label = Q_type.upper()
        print(f"    {label:>10}: mean={np.mean(flat):.3f}, std={np.std(flat):.3f}, "
              f"skew={stats.skew(flat):.2f}, kurt={stats.kurtosis(flat):.2f}")

    # ====================================================================
    # Save results
    # ====================================================================
    rows = []
    for bar_type, coverages, aads in [("calendar", cal_coverages, cal_aads), ("dollar", dol_coverages, dol_aads)]:
        for Q_type in all_Q_types:
            for p in prob_levels:
                rows.append({
                    "bar_type": bar_type, "Q_type": Q_type, "p": p,
                    "coverage": coverages[Q_type].get(p, 0), "ideal": p * 100,
                })

    table3_df = pd.DataFrame(rows)
    table3_df.to_csv(OUTPUT_DIR / "table3_mixture_coverage.csv", index=False)
    print(f"\n  Results saved to {OUTPUT_DIR / 'table3_mixture_coverage.csv'}")