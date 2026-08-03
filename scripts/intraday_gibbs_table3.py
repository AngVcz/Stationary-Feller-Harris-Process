"""Daily-level FFBS Gibbs sampler for SF-Harris SV model.

The 15-min FFBS approach failed because individual squared returns are too noisy
(sigma_obs ~ 2.2) for reliable state identification — the sampler can't distinguish
stays from observation noise, inflating sigma_Q to ~5.

This version uses DAILY log-RV as observations, where the observation noise is
sigma_obs ~ sqrt(2/26) ~ 0.28 (log-chi-squared variability from averaging 26
intraday returns). The much better signal-to-noise ratio (sigma_Q/sigma_obs ~ 3)
allows the FFBS to properly identify stays vs jumps.

Model:
  Latent state: log(IV_t) — piecewise-constant SF-Harris process
  Observation:  log(RV_t) = log(IV_t) + epsilon_t, epsilon_t ~ N(0, sigma_obs^2)
  Transition:   P(log(IV_t) = log(IV_{t-1})) = exp(-alpha)  (stay)
                P(log(IV_t) ~ Q) = 1-exp(-alpha)              (jump)
  Q:            log(IV) ~ Normal(mu_Q, sigma_Q^2)

Following Anzarut Section 4.2 + 5:
  1. Compute daily RV from 15-min returns
  2. Apply periodicity adjustment (calendar bars)
  3. Fit SF-Harris with FFBS Gibbs sampler at daily level
  4. Simulate forward (SF-Harris + observation noise)
  5. Compute Table 3 coverage
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

DATA_PATH = Path(r"C:\Users\angve\OneDrive\Desktop\Servicio\Libros\SF-Harris\IBM.txt")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"

# Trading day: 9:30-16:00 = 6.5 hours = 390 minutes = 26 intervals of 15 min
DT = 15.0 / 390.0
N_INTERVALS = 26


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


def estimate_periodicity(returns_15min):
    records = []
    for date, group in returns_15min.groupby(returns_15min.index.date):
        rv_daily = np.sum(group**2)
        n = len(group)
        for idx, val in group.items():
            time_str = idx.strftime("%H:%M")
            records.append({"date": date, "time": time_str, "rv": val**2, "rv_daily": rv_daily, "n": n})
    rv_df = pd.DataFrame(records)
    rv_df["rv_daily_mean"] = rv_df["rv_daily"] / rv_df["n"]
    rv_df["ratio"] = rv_df["rv"] / rv_df["rv_daily_mean"]
    periodicity = rv_df.groupby("time")["ratio"].mean()
    periodicity = periodicity / periodicity.mean()
    return periodicity


# ---------------------------------------------------------------------------
# FFBS Gibbs sampler (daily level)
# ---------------------------------------------------------------------------
def gibbs_sf_harris_daily(log_rv, n_iter=2000, burn_in=500, rng=None, K=50):
    """Gibbs sampler for SF-Harris using discretized FFBS at the DAILY level.

    Model:
      States x_t: log(IV_t) — piecewise-constant SF-Harris (discretized into K bins)
      Observations y_t: log(RV_t) — noisy estimates of log(IV_t)
      Transition:  P(x_t = x_{t-1}) = exp(-alpha), P(x_t ~ Q) = 1-exp(-alpha)
      Observation: y_t | x_t ~ Normal(x_t, sigma_obs^2)
      Q:           x ~ Normal(mu_Q, sigma_Q^2)

    sigma_obs is initialized at sqrt(2/N_INTERVALS) ~ 0.28, reflecting the
    log-chi-squared variability in daily RV from averaging 26 intraday returns.
    """
    if rng is None:
        rng = np.random.default_rng()

    y = log_rv.copy()
    n = len(y)

    # Discretize state space into K bins
    y_lo, y_hi = np.percentile(y, [1, 99])
    pad = (y_hi - y_lo) * 1.0
    bin_edges = np.linspace(y_lo - pad, y_hi + pad, K + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Initialize parameters
    mu_Q = np.mean(y)
    sigma_Q = np.std(y)
    alpha = -np.log(max(np.corrcoef(y[:-1], y[1:])[0, 1], 0.01))  # init from ACF(1)
    sigma_obs = np.sqrt(2.0 / N_INTERVALS)  # ~0.28 for 26 intervals

    print(f"  Init: mu_Q={mu_Q:.4f}, sigma_Q={sigma_Q:.4f}, alpha={alpha:.4f}, "
          f"sigma_obs={sigma_obs:.4f}, P(stay)={np.exp(-alpha):.4f}")

    # Pre-compute z-score matrix (constant across iterations)
    z_matrix = y[:, np.newaxis] - bin_centers[np.newaxis, :]  # n x K

    samples = {
        "alpha": np.empty(n_iter),
        "mu_Q": np.empty(n_iter),
        "sigma_Q": np.empty(n_iter),
        "sigma_obs": np.empty(n_iter),
        "n_jumps": np.empty(n_iter, dtype=int),
    }

    for it in range(n_iter):
        # ---- Build transition matrix ----
        Q_log = stats.norm.logpdf(bin_centers, mu_Q, sigma_Q)
        Q_probs = np.exp(Q_log - np.max(Q_log))
        Q_probs /= Q_probs.sum()

        p_stay = np.exp(-alpha)
        p_jump = 1 - p_stay
        T = np.tile(p_jump * Q_probs, (K, 1))  # K x K
        np.fill_diagonal(T, T.diagonal() + p_stay)

        # ---- Pre-compute observation log-likelihoods ----
        log_liks = -0.5 * (z_matrix / sigma_obs) ** 2 - np.log(sigma_obs) - 0.5 * np.log(2 * np.pi)

        # ---- Forward filtering ----
        log_fwd = np.log(Q_probs + 1e-300) + log_liks[0]
        log_fwd -= np.max(log_fwd)
        fwd = np.exp(log_fwd)
        fwd /= fwd.sum()

        fwd_messages = np.empty((n, K))
        fwd_messages[0] = fwd

        for t in range(1, n):
            pred = fwd @ T
            log_new = np.log(pred + 1e-300) + log_liks[t]
            log_new -= np.max(log_new)
            fwd = np.exp(log_new)
            s = fwd.sum()
            fwd = fwd / s if s > 1e-300 else Q_probs.copy()
            fwd_messages[t] = fwd

        # ---- Backward sampling ----
        states = np.empty(n, dtype=int)
        states[n - 1] = rng.choice(K, p=fwd_messages[n - 1])

        for t in range(n - 2, -1, -1):
            bw = fwd_messages[t] * T[:, states[t + 1]]
            s = bw.sum()
            bw = bw / s if s > 1e-300 else fwd_messages[t]
            states[t] = rng.choice(K, p=bw)

        sampled_vals = bin_centers[states]

        # Count jumps
        is_jump = np.ones(n, dtype=bool)
        for t in range(1, n):
            is_jump[t] = (states[t] != states[t - 1])
        n_jumps = int(np.sum(is_jump))

        # ---- Sample alpha ----
        a_post = n_jumps + 1.0
        b_post = float(n) + 0.1
        alpha = rng.gamma(a_post, 1.0 / b_post)
        alpha = np.clip(alpha, 0.001, 10.0)

        # ---- Sample Q parameters ----
        jump_vals = sampled_vals[is_jump]
        if len(jump_vals) > 5:
            n_j = len(jump_vals)
            mean_j = np.mean(jump_vals)
            var_j = np.var(jump_vals) + 1e-10

            kappa0, a0 = 0.01, 3.0
            kappa_n = kappa0 + n_j
            mu_n = (kappa0 * mu_Q + n_j * mean_j) / kappa_n
            a_n = a0 + n_j / 2.0
            b_n = sigma_Q**2 + 0.5 * (n_j * var_j + kappa0 * n_j * (mean_j - mu_Q)**2 / kappa_n)

            sigma2_Q = 1.0 / rng.gamma(a_n, 1.0 / b_n)
            sigma_Q = np.sqrt(max(sigma2_Q, 0.01))
            mu_Q = rng.normal(mu_n, sigma_Q / np.sqrt(kappa_n))

        # ---- Sample sigma_obs ----
        residuals = y - sampled_vals
        a_res = n / 2.0 + 1.0
        b_res = np.sum(residuals**2) / 2.0 + 0.1
        sigma_obs = np.sqrt(1.0 / rng.gamma(a_res, 1.0 / b_res))
        sigma_obs = max(sigma_obs, 0.01)

        # Store
        samples["alpha"][it] = alpha
        samples["mu_Q"][it] = mu_Q
        samples["sigma_Q"][it] = sigma_Q
        samples["sigma_obs"][it] = sigma_obs
        samples["n_jumps"][it] = n_jumps

        if it % 200 == 0 or it == n_iter - 1:
            ps = np.exp(-alpha)
            seg = 1 / (1 - ps) if ps < 1 else float('inf')
            print(f"    iter {it:>4}/{n_iter}: alpha={alpha:.4f} P(stay)={ps:.4f} "
                  f"n_jumps={n_jumps}/{n} sigma_Q={sigma_Q:.4f} sigma_obs={sigma_obs:.4f} "
                  f"avg_seg={seg:.1f}d")

    return {k: v[burn_in:] for k, v in samples.items()}


# ---------------------------------------------------------------------------
# Predictive simulation (daily level)
# ---------------------------------------------------------------------------
def simulate_daily_sf_harris(n_days, posterior, last_log_iv, rng=None):
    """Simulate SF-Harris at daily level with observation noise.

    1. Generate latent log(IV) using SF-Harris (stay/jump process)
    2. Add observation noise: log(RV) = log(IV) + epsilon
    """
    if rng is None:
        rng = np.random.default_rng()

    n_post = len(posterior["alpha"])
    idx = rng.integers(0, n_post)

    alpha = posterior["alpha"][idx]
    mu_Q = posterior["mu_Q"][idx]
    sigma_Q = posterior["sigma_Q"][idx]
    sigma_obs = posterior["sigma_obs"][idx]

    p_stay = np.exp(-alpha)
    daily_log_rv = np.empty(n_days)

    current_log_iv = last_log_iv

    for d in range(n_days):
        # SF-Harris transition
        if rng.uniform() >= p_stay:
            current_log_iv = rng.normal(mu_Q, sigma_Q)
        # Observation noise
        epsilon = rng.normal(0, sigma_obs)
        daily_log_rv[d] = current_log_iv + epsilon

    return daily_log_rv


def compute_coverage(actual_log_rv, simulated_log_rvs, prob_levels):
    n_obs = len(actual_log_rv)
    min_len = min(n_obs, simulated_log_rvs.shape[1])
    actual = actual_log_rv[:min_len]
    sims = simulated_log_rvs[:, :min_len]

    coverage = {}
    for p in prob_levels:
        lower = np.quantile(sims, (1 - p) / 2, axis=0)
        upper = np.quantile(sims, (1 + p) / 2, axis=0)
        inside = np.sum((actual >= lower) & (actual <= upper))
        coverage[p] = inside / min_len * 100

    return coverage


# ---------------------------------------------------------------------------
# Method-of-moments estimation
# ---------------------------------------------------------------------------
def mom_estimate(log_rv, n_lags=10):
    """Estimate SF-Harris parameters from ACF and variance.

    Key insight: ACF(h) = P(stay)^h * R, where R = sigma_Q^2 / (sigma_Q^2 + sigma_obs^2)
    So P(stay) = ACF(2)/ACF(1), and R = ACF(1)/P(stay) = ACF(1)^2/ACF(2)

    Returns alpha, mu_Q, sigma_Q, sigma_obs, p_stay, acf_values
    """
    mu = np.mean(log_rv)
    var = np.var(log_rv)

    # Compute ACF
    acf = np.zeros(n_lags + 1)
    acf[0] = 1.0
    mean = np.mean(log_rv)
    for h in range(1, n_lags + 1):
        acf[h] = np.mean((log_rv[:-h] - mean) * (log_rv[h:] - mean)) / var

    # P(stay) from ACF ratio
    p_stay = np.clip(acf[2] / acf[1] if acf[1] > 0 else 0.5, 0.01, 0.99)
    alpha = -np.log(p_stay)

    # R = signal variance ratio
    R = acf[1] / p_stay if p_stay > 0 else 0.5
    R = np.clip(R, 0.01, 0.99)

    # Variance decomposition
    sigma_Q_sq = var * R
    sigma_obs_sq = var * (1 - R)

    # Ensure positive
    if sigma_Q_sq < 0.001:
        sigma_Q_sq = 0.001
    if sigma_obs_sq < 0.001:
        sigma_obs_sq = 0.001

    sigma_Q = np.sqrt(sigma_Q_sq)
    sigma_obs = np.sqrt(sigma_obs_sq)

    return alpha, mu, sigma_Q, sigma_obs, p_stay, acf


def simulate_daily_mom(n_days, alpha, mu_Q, sigma_Q, sigma_obs, p_stay, last_val, rng=None, nu=np.inf):
    """Simulate SF-Harris at daily level with method-of-moments parameters.

    Supports Student-t Q distribution for heavier tails (nu < inf).
    Student-t variance correction: scale = sigma_Q * sqrt((nu-2)/nu)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Student-t variance correction
    if nu < np.inf and nu > 2:
        scale = sigma_Q * np.sqrt((nu - 2) / nu)
    else:
        scale = sigma_Q

    daily_log_rv = np.empty(n_days)
    current = last_val

    for d in range(n_days):
        if rng.uniform() >= p_stay:
            if nu < np.inf and nu > 2:
                current = mu_Q + scale * rng.standard_t(nu)
            else:
                current = rng.normal(mu_Q, sigma_Q)
        daily_log_rv[d] = current + rng.normal(0, sigma_obs)

    return daily_log_rv


def bootstrap_mom_estimate(log_rv, n_boot=500, rng=None):
    """Bootstrap method-of-moments estimates for parameter uncertainty."""
    if rng is None:
        rng = np.random.default_rng()

    n = len(log_rv)
    boot_params = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_sample = log_rv[idx]
        try:
            alpha, mu, sigma_Q, sigma_obs, p_stay, acf = mom_estimate(boot_sample)
            boot_params.append((alpha, mu, sigma_Q, sigma_obs, p_stay))
        except Exception:
            continue

    return boot_params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Daily-Level FFBS Gibbs Sampler - SF-Harris SV Model")
    print("Observation: log(RV_t) = log(IV_t) + noise")
    print("=" * 70)

    anzarut_coverage = {0.25: 25, 0.50: 51, 0.75: 75, 0.85: 84, 0.90: 89, 0.95: 93}
    prob_levels = [0.25, 0.50, 0.75, 0.85, 0.90, 0.95]

    df = load_ibm_data()
    print(f"Loaded {len(df)} 1-minute bars")

    # ====================================================================
    # PIPELINE A: Calendar bars
    # ====================================================================
    print(f"\n{'='*70}")
    print("CALENDAR BARS (daily-level FFBS Gibbs)")
    print("=" * 70)

    returns_15min = compute_15min_returns(df)
    print(f"\n  15-min returns: {len(returns_15min)}")

    # Daily RV from 15-min returns
    daily_rv_cal = returns_15min.groupby(returns_15min.index.date).apply(lambda x: np.sum(x**2))
    daily_rv_cal.index = pd.to_datetime(daily_rv_cal.index)
    log_rv_cal = np.log(daily_rv_cal.values)

    # Remove any inf/nan
    valid = np.isfinite(log_rv_cal)
    log_rv_cal_clean = log_rv_cal[valid]
    valid_dates = daily_rv_cal.index[valid]

    print(f"  Daily log-RV: {len(log_rv_cal_clean)} days")
    print(f"  Mean: {np.mean(log_rv_cal_clean):.4f}, Std: {np.std(log_rv_cal_clean):.4f}")
    print(f"  Skew: {stats.skew(log_rv_cal_clean):.2f}, Kurt: {stats.kurtosis(log_rv_cal_clean):.2f}")

    # ACF
    acf1 = np.corrcoef(log_rv_cal_clean[:-1], log_rv_cal_clean[1:])[0, 1]
    print(f"  ACF(1): {acf1:.4f}")

    # 80/20 split
    n_days_total = len(log_rv_cal_clean)
    split_idx = int(n_days_total * 0.8)
    train_log_rv = log_rv_cal_clean[:split_idx]
    test_log_rv = log_rv_cal_clean[split_idx:]
    print(f"  Train: {split_idx} days, Test: {n_days_total - split_idx} days")

    # FFBS Gibbs sampler
    print(f"\n  Running daily-level FFBS Gibbs sampler (2000 iterations, K=50 bins)...")
    rng_cal = np.random.default_rng(42)
    cal_posterior = gibbs_sf_harris_daily(
        train_log_rv, n_iter=2000, burn_in=500, rng=rng_cal, K=50
    )

    print(f"\n  Posterior summary (after burn-in):")
    for key in ["alpha", "mu_Q", "sigma_Q", "sigma_obs"]:
        vals = cal_posterior[key]
        print(f"    {key:>12}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, "
              f"2.5%={np.percentile(vals, 2.5):.4f}, 97.5%={np.percentile(vals, 97.5):.4f}")

    p_stay_mean = np.mean(np.exp(-cal_posterior["alpha"]))
    avg_jumps = np.mean(cal_posterior["n_jumps"])
    p_jump = 1 - p_stay_mean
    avg_seg = 1 / p_jump if p_jump > 0 else float('inf')
    print(f"    P(stay per day): {p_stay_mean:.4f}")
    print(f"    Avg jumps: {avg_jumps:.0f} / {len(train_log_rv)} days")
    print(f"    Avg segment length: {avg_seg:.1f} days")

    # Simulate
    n_test = len(test_log_rv)
    n_sim = 2000
    print(f"\n  Simulating {n_sim} trajectories for {n_test} test days...")
    rng_sim = np.random.default_rng(123)
    cal_simulated_log_rv = np.empty((n_sim, n_test))

    for s in range(n_sim):
        cal_simulated_log_rv[s] = simulate_daily_sf_harris(
            n_test, cal_posterior, last_log_iv=train_log_rv[-1], rng=rng_sim
        )

    cal_coverage = compute_coverage(test_log_rv, cal_simulated_log_rv, prob_levels)

    # ====================================================================
    # PIPELINE B: Dollar bars (daily-level FFBS)
    # ====================================================================
    print(f"\n{'='*70}")
    print("DOLLAR BARS (daily-level FFBS Gibbs)")
    print("=" * 70)

    # Build dollar bars
    train_end_date = valid_dates[split_idx - 1].date() if hasattr(valid_dates[split_idx-1], 'date') else valid_dates[split_idx-1]
    df_train = df[df.index.date <= train_end_date]
    train_threshold = df_train["dollar_volume"].resample("15min").sum().mean()

    dollar_bars = []
    cum_dv = 0.0
    first_close = None
    for idx, row in df.iterrows():
        if first_close is None:
            first_close = row["close"]
        cum_dv += row["dollar_volume"]
        if cum_dv >= train_threshold:
            ret = np.log(row["close"] / first_close)
            dollar_bars.append({"datetime": idx, "return": ret})
            cum_dv = 0.0
            first_close = None

    dol_df = pd.DataFrame(dollar_bars).set_index("datetime") if dollar_bars else pd.DataFrame()
    dol_returns = dol_df["return"]

    # Daily RV from dollar bars
    dol_daily_rv = dol_returns.groupby(dol_returns.index.date).apply(lambda x: np.sum(x**2))
    dol_daily_rv.index = pd.to_datetime(dol_daily_rv.index)
    log_rv_dol = np.log(dol_daily_rv.values)

    # Remove inf/nan
    valid_dol = np.isfinite(log_rv_dol)
    log_rv_dol_clean = log_rv_dol[valid_dol]

    # Train/test split (same dates as calendar)
    dol_dates = dol_daily_rv.index[valid_dol]
    dol_train_mask = np.array([d.date() <= train_end_date for d in dol_dates])
    dol_train_log_rv = log_rv_dol_clean[dol_train_mask]
    dol_test_log_rv = log_rv_dol_clean[~dol_train_mask]

    print(f"  Dollar bar daily log-RV: {len(log_rv_dol_clean)} days")
    print(f"  Mean: {np.mean(log_rv_dol_clean):.4f}, Std: {np.std(log_rv_dol_clean):.4f}")
    print(f"  Train: {len(dol_train_log_rv)} days, Test: {len(dol_test_log_rv)} days")

    # FFBS Gibbs sampler for dollar bars
    print(f"\n  Running daily-level FFBS Gibbs sampler (2000 iterations, K=50 bins)...")
    rng_dol = np.random.default_rng(42)
    dol_posterior = gibbs_sf_harris_daily(
        dol_train_log_rv, n_iter=2000, burn_in=500, rng=rng_dol, K=50
    )

    print(f"\n  Posterior summary (after burn-in):")
    for key in ["alpha", "mu_Q", "sigma_Q", "sigma_obs"]:
        vals = dol_posterior[key]
        print(f"    {key:>12}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}")

    p_stay_dol = np.mean(np.exp(-dol_posterior["alpha"]))
    avg_jumps_dol = np.mean(dol_posterior["n_jumps"])
    print(f"    P(stay per day): {p_stay_dol:.4f}")
    print(f"    Avg jumps: {avg_jumps_dol:.0f} / {len(dol_train_log_rv)} days")

    # Simulate
    n_test_dol = len(dol_test_log_rv)
    rng_dsim = np.random.default_rng(456)
    dol_simulated_log_rv = np.empty((n_sim, n_test_dol))

    for s in range(n_sim):
        dol_simulated_log_rv[s] = simulate_daily_sf_harris(
            n_test_dol, dol_posterior, last_log_iv=dol_train_log_rv[-1], rng=rng_dsim
        )

    dol_coverage = compute_coverage(dol_test_log_rv, dol_simulated_log_rv, prob_levels)

    # ====================================================================
    # PIPELINE C: Method-of-moments (ACF-constrained)
    # ====================================================================
    print(f"\n{'='*70}")
    print("METHOD-OF-MOMENTS (ACF-constrained parameters)")
    print("=" * 70)
    print("  Key: P(stay) = ACF(2)/ACF(1) forces the model to match observed ACF")

    # Calendar bars - MoM
    cal_mom = mom_estimate(train_log_rv)
    cal_alpha_mom, cal_mu_mom, cal_sigmaQ_mom, cal_sigmaObs_mom, cal_pstay_mom, cal_acf = cal_mom
    print(f"\n  Calendar bars MoM parameters:")
    print(f"    ACF(1)={cal_acf[1]:.4f}, ACF(2)={cal_acf[2]:.4f}, ACF(5)={cal_acf[5]:.4f}")
    print(f"    P(stay)={cal_pstay_mom:.4f}, alpha={cal_alpha_mom:.4f}")
    print(f"    mu_Q={cal_mu_mom:.4f}, sigma_Q={cal_sigmaQ_mom:.4f}, sigma_obs={cal_sigmaObs_mom:.4f}")
    print(f"    Signal ratio R={cal_sigmaQ_mom**2/(cal_sigmaQ_mom**2+cal_sigmaObs_mom**2):.4f}")

    # Variance scaling: ensure sigma_Q^2 + sigma_obs^2 matches data variance
    cal_var = np.var(train_log_rv)
    cal_model_var = cal_sigmaQ_mom**2 + cal_sigmaObs_mom**2
    cal_var_scale = np.sqrt(cal_var / cal_model_var) if cal_model_var > 0 else 1.0
    cal_sigmaQ_scaled = cal_sigmaQ_mom * cal_var_scale
    cal_sigmaObs_scaled = cal_sigmaObs_mom * cal_var_scale
    print(f"    Variance scaling: {cal_var_scale:.4f} (model_var={cal_model_var:.4f}, data_var={cal_var:.4f})")
    print(f"    Scaled: sigma_Q={cal_sigmaQ_scaled:.4f}, sigma_obs={cal_sigmaObs_scaled:.4f}")

    # Student-t degrees of freedom (from excess kurtosis of training data)
    cal_kurt = stats.kurtosis(train_log_rv)
    cal_nu = max(4 + 6 / max(cal_kurt, 0.1), 4.5)  # nu >= 4.5 for finite kurtosis
    print(f"    Excess kurtosis: {cal_kurt:.2f}, Student-t nu: {cal_nu:.1f}")

    # Bootstrap for parameter uncertainty
    print(f"\n  Bootstrapping MoM parameters (500 resamples)...")
    rng_boot_cal = np.random.default_rng(789)
    cal_boot_params = bootstrap_mom_estimate(train_log_rv, n_boot=500, rng=rng_boot_cal)
    print(f"  Got {len(cal_boot_params)} bootstrap estimates")

    # Simulate with bootstrap parameter uncertainty + variance scaling + Student-t Q
    n_test = len(test_log_rv)
    n_sim_mom = 2000
    rng_mom_cal = np.random.default_rng(321)
    cal_mom_simulated = np.empty((n_sim_mom, n_test))
    cal_mom_sim_t = np.empty((n_sim_mom, n_test))  # Student-t version

    for s in range(n_sim_mom):
        # Draw bootstrap parameters
        bp = cal_boot_params[rng_mom_cal.integers(0, len(cal_boot_params))]
        alpha_b, mu_b, sigmaQ_b, sigmaObs_b, pstay_b = bp
        # Variance scaling per bootstrap sample
        boot_var = sigmaQ_b**2 + sigmaObs_b**2
        if boot_var > 0:
            vscale = np.sqrt(cal_var / boot_var)
            sigmaQ_b = sigmaQ_b * vscale
            sigmaObs_b = sigmaObs_b * vscale
        # Gaussian Q
        cal_mom_simulated[s] = simulate_daily_mom(
            n_test, alpha_b, mu_b, sigmaQ_b, sigmaObs_b, pstay_b,
            last_val=train_log_rv[-1], rng=rng_mom_cal
        )
        # Student-t Q
        cal_mom_sim_t[s] = simulate_daily_mom(
            n_test, alpha_b, mu_b, sigmaQ_b, sigmaObs_b, pstay_b,
            last_val=train_log_rv[-1], rng=rng_mom_cal, nu=cal_nu
        )

    cal_mom_coverage = compute_coverage(test_log_rv, cal_mom_simulated, prob_levels)
    cal_mom_t_coverage = compute_coverage(test_log_rv, cal_mom_sim_t, prob_levels)

    # Diagnostics
    cal_mom_flat = cal_mom_simulated.flatten()
    cal_mom_t_flat = cal_mom_sim_t.flatten()
    print(f"\n  Calendar MoM (Gaussian Q) - log-RV distribution:")
    print(f"    Actual:  mean={np.mean(test_log_rv):.3f}, std={np.std(test_log_rv):.3f}")
    print(f"    Simulated: mean={np.mean(cal_mom_flat):.3f}, std={np.std(cal_mom_flat):.3f}")
    print(f"\n  Calendar MoM (Student-t Q, nu={cal_nu:.1f}) - log-RV distribution:")
    print(f"    Simulated: mean={np.mean(cal_mom_t_flat):.3f}, std={np.std(cal_mom_t_flat):.3f}")

    # Dollar bars - MoM
    dol_mom = mom_estimate(dol_train_log_rv)
    dol_alpha_mom, dol_mu_mom, dol_sigmaQ_mom, dol_sigmaObs_mom, dol_pstay_mom, dol_acf = dol_mom
    print(f"\n  Dollar bars MoM parameters:")
    print(f"    ACF(1)={dol_acf[1]:.4f}, ACF(2)={dol_acf[2]:.4f}")
    print(f"    P(stay)={dol_pstay_mom:.4f}, alpha={dol_alpha_mom:.4f}")
    print(f"    mu_Q={dol_mu_mom:.4f}, sigma_Q={dol_sigmaQ_mom:.4f}, sigma_obs={dol_sigmaObs_mom:.4f}")

    # Variance scaling for dollar bars
    dol_var = np.var(dol_train_log_rv)
    dol_model_var = dol_sigmaQ_mom**2 + dol_sigmaObs_mom**2
    dol_var_scale = np.sqrt(dol_var / dol_model_var) if dol_model_var > 0 else 1.0
    dol_sigmaQ_scaled = dol_sigmaQ_mom * dol_var_scale
    dol_sigmaObs_scaled = dol_sigmaObs_mom * dol_var_scale
    dol_kurt = stats.kurtosis(dol_train_log_rv)
    dol_nu = max(4 + 6 / max(dol_kurt, 0.1), 4.5)
    print(f"    Variance scaling: {dol_var_scale:.4f}")
    print(f"    Scaled: sigma_Q={dol_sigmaQ_scaled:.4f}, sigma_obs={dol_sigmaObs_scaled:.4f}")
    print(f"    Excess kurtosis: {dol_kurt:.2f}, Student-t nu: {dol_nu:.1f}")

    rng_boot_dol = np.random.default_rng(789)
    dol_boot_params = bootstrap_mom_estimate(dol_train_log_rv, n_boot=500, rng=rng_boot_dol)
    print(f"  Got {len(dol_boot_params)} bootstrap estimates")

    n_test_dol = len(dol_test_log_rv)
    rng_mom_dol = np.random.default_rng(654)
    dol_mom_simulated = np.empty((n_sim_mom, n_test_dol))
    dol_mom_sim_t = np.empty((n_sim_mom, n_test_dol))

    for s in range(n_sim_mom):
        bp = dol_boot_params[rng_mom_dol.integers(0, len(dol_boot_params))]
        alpha_b, mu_b, sigmaQ_b, sigmaObs_b, pstay_b = bp
        # Variance scaling per bootstrap sample
        boot_var = sigmaQ_b**2 + sigmaObs_b**2
        if boot_var > 0:
            vscale = np.sqrt(dol_var / boot_var)
            sigmaQ_b = sigmaQ_b * vscale
            sigmaObs_b = sigmaObs_b * vscale
        dol_mom_simulated[s] = simulate_daily_mom(
            n_test_dol, alpha_b, mu_b, sigmaQ_b, sigmaObs_b, pstay_b,
            last_val=dol_train_log_rv[-1], rng=rng_mom_dol
        )
        dol_mom_sim_t[s] = simulate_daily_mom(
            n_test_dol, alpha_b, mu_b, sigmaQ_b, sigmaObs_b, pstay_b,
            last_val=dol_train_log_rv[-1], rng=rng_mom_dol, nu=dol_nu
        )

    dol_mom_coverage = compute_coverage(dol_test_log_rv, dol_mom_simulated, prob_levels)
    dol_mom_t_coverage = compute_coverage(dol_test_log_rv, dol_mom_sim_t, prob_levels)

    dol_mom_flat = dol_mom_simulated.flatten()
    dol_mom_t_flat = dol_mom_sim_t.flatten()
    print(f"\n  Dollar MoM (Gaussian Q) - log-RV distribution:")
    print(f"    Actual:  mean={np.mean(dol_test_log_rv):.3f}, std={np.std(dol_test_log_rv):.3f}")
    print(f"    Simulated: mean={np.mean(dol_mom_flat):.3f}, std={np.std(dol_mom_flat):.3f}")
    print(f"\n  Dollar MoM (Student-t Q, nu={dol_nu:.1f}) - log-RV distribution:")
    print(f"    Simulated: mean={np.mean(dol_mom_t_flat):.3f}, std={np.std(dol_mom_t_flat):.3f}")

    # ====================================================================
    # Diagnostics
    # ====================================================================
    print(f"\n{'='*70}")
    print("DISTRIBUTION DIAGNOSTICS")
    print("=" * 70)

    print(f"\n  Calendar bars - log-RV distribution:")
    print(f"    Actual:  mean={np.mean(test_log_rv):.3f}, std={np.std(test_log_rv):.3f}, "
          f"skew={stats.skew(test_log_rv):.2f}, kurt={stats.kurtosis(test_log_rv):.2f}")
    cal_flat = cal_simulated_log_rv.flatten()
    print(f"    FFBS:     mean={np.mean(cal_flat):.3f}, std={np.std(cal_flat):.3f}, "
          f"skew={stats.skew(cal_flat):.2f}, kurt={stats.kurtosis(cal_flat):.2f}")
    print(f"    MoM:      mean={np.mean(cal_mom_flat):.3f}, std={np.std(cal_mom_flat):.3f}, "
          f"skew={stats.skew(cal_mom_flat):.2f}, kurt={stats.kurtosis(cal_mom_flat):.2f}")
    print(f"    MoM-t:    mean={np.mean(cal_mom_t_flat):.3f}, std={np.std(cal_mom_t_flat):.3f}, "
          f"skew={stats.skew(cal_mom_t_flat):.2f}, kurt={stats.kurtosis(cal_mom_t_flat):.2f}")

    print(f"\n  Dollar bars - log-RV distribution:")
    print(f"    Actual:  mean={np.mean(dol_test_log_rv):.3f}, std={np.std(dol_test_log_rv):.3f}, "
          f"skew={stats.skew(dol_test_log_rv):.2f}, kurt={stats.kurtosis(dol_test_log_rv):.2f}")
    dol_flat = dol_simulated_log_rv.flatten()
    print(f"    FFBS:     mean={np.mean(dol_flat):.3f}, std={np.std(dol_flat):.3f}, "
          f"skew={stats.skew(dol_flat):.2f}, kurt={stats.kurtosis(dol_flat):.2f}")
    print(f"    MoM:      mean={np.mean(dol_mom_flat):.3f}, std={np.std(dol_mom_flat):.3f}, "
          f"skew={stats.skew(dol_mom_flat):.2f}, kurt={stats.kurtosis(dol_mom_flat):.2f}")
    print(f"    MoM-t:    mean={np.mean(dol_mom_t_flat):.3f}, std={np.std(dol_mom_t_flat):.3f}, "
          f"skew={stats.skew(dol_mom_t_flat):.2f}, kurt={stats.kurtosis(dol_mom_t_flat):.2f}")

    # ====================================================================
    # TABLE 3
    # ====================================================================
    print(f"\n{'='*70}")
    print("TABLE 3: Coverage Validation")
    print("=" * 70)

    print(f"\n  Anzarut's Table 3 (GIG-Harris + Gibbs-b):")
    print(f"  {'p':>6}  {'Anzarut':>10}  {'Ideal':>6}")
    print(f"  {'-'*28}")
    for p in prob_levels:
        print(f"  {p:>6.2f}  {anzarut_coverage[p]:>9.0f}%  {p*100:>5.0f}%")

    print(f"\n  Calendar + FFBS Gibbs:")
    print(f"  {'p':>6}  {'Coverage':>10}  {'Ideal':>6}  {'Dev':>6}")
    print(f"  {'-'*34}")
    for p in prob_levels:
        cov = cal_coverage.get(p, 0)
        dev = cov - p * 100
        print(f"  {p:>6.2f}  {cov:>9.1f}%  {p*100:>5.0f}%  {dev:>+5.1f}pp")

    print(f"\n  Calendar + MoM (ACF-constrained, Gaussian Q):")
    print(f"  {'p':>6}  {'Coverage':>10}  {'Ideal':>6}  {'Dev':>6}")
    print(f"  {'-'*34}")
    for p in prob_levels:
        cov = cal_mom_coverage.get(p, 0)
        dev = cov - p * 100
        print(f"  {p:>6.2f}  {cov:>9.1f}%  {p*100:>5.0f}%  {dev:>+5.1f}pp")

    print(f"\n  Calendar + MoM (Student-t Q, nu={cal_nu:.1f}):")
    print(f"  {'p':>6}  {'Coverage':>10}  {'Ideal':>6}  {'Dev':>6}")
    print(f"  {'-'*34}")
    for p in prob_levels:
        cov = cal_mom_t_coverage.get(p, 0)
        dev = cov - p * 100
        print(f"  {p:>6.2f}  {cov:>9.1f}%  {p*100:>5.0f}%  {dev:>+5.1f}pp")

    print(f"\n  Dollar + FFBS Gibbs:")
    print(f"  {'p':>6}  {'Coverage':>10}  {'Ideal':>6}  {'Dev':>6}")
    print(f"  {'-'*34}")
    for p in prob_levels:
        cov = dol_coverage.get(p, 0)
        dev = cov - p * 100
        print(f"  {p:>6.2f}  {cov:>9.1f}%  {p*100:>5.0f}%  {dev:>+5.1f}pp")

    print(f"\n  Dollar + MoM (Gaussian Q):")
    print(f"  {'p':>6}  {'Coverage':>10}  {'Ideal':>6}  {'Dev':>6}")
    print(f"  {'-'*34}")
    for p in prob_levels:
        cov = dol_mom_coverage.get(p, 0)
        dev = cov - p * 100
        print(f"  {p:>6.2f}  {cov:>9.1f}%  {p*100:>5.0f}%  {dev:>+5.1f}pp")

    print(f"\n  Dollar + MoM (Student-t Q, nu={dol_nu:.1f}):")
    print(f"  {'p':>6}  {'Coverage':>10}  {'Ideal':>6}  {'Dev':>6}")
    print(f"  {'-'*34}")
    for p in prob_levels:
        cov = dol_mom_t_coverage.get(p, 0)
        dev = cov - p * 100
        print(f"  {p:>6.2f}  {cov:>9.1f}%  {p*100:>5.0f}%  {dev:>+5.1f}pp")

    # Side-by-side comparison
    print(f"\n  Side-by-side comparison:")
    print(f"  {'p':>6}  {'Anzarut':>8}  {'Cal_FFBS':>9}  {'Cal_MoM':>8}  {'Cal_t':>7}  "
          f"{'Dol_FFBS':>9}  {'Dol_MoM':>8}  {'Dol_t':>7}  {'Ideal':>6}")
    print(f"  {'-'*75}")
    for p in prob_levels:
        print(f"  {p:>6.2f}  {anzarut_coverage[p]:>7.0f}%  {cal_coverage.get(p,0):>8.1f}%  "
              f"{cal_mom_coverage.get(p,0):>7.1f}%  {cal_mom_t_coverage.get(p,0):>6.1f}%  "
              f"{dol_coverage.get(p,0):>8.1f}%  {dol_mom_coverage.get(p,0):>7.1f}%  "
              f"{dol_mom_t_coverage.get(p,0):>6.1f}%  {p*100:>5.0f}%")

    # AAD
    anz_aad = np.mean([abs(anzarut_coverage[p] - p * 100) for p in prob_levels])
    cal_aad = np.mean([abs(cal_coverage.get(p, 0) - p * 100) for p in prob_levels])
    cal_mom_aad = np.mean([abs(cal_mom_coverage.get(p, 0) - p * 100) for p in prob_levels])
    cal_mom_t_aad = np.mean([abs(cal_mom_t_coverage.get(p, 0) - p * 100) for p in prob_levels])
    dol_aad = np.mean([abs(dol_coverage.get(p, 0) - p * 100) for p in prob_levels])
    dol_mom_aad = np.mean([abs(dol_mom_coverage.get(p, 0) - p * 100) for p in prob_levels])
    dol_mom_t_aad = np.mean([abs(dol_mom_t_coverage.get(p, 0) - p * 100) for p in prob_levels])
    print(f"\n  Mean absolute deviation (AAD) from ideal:")
    print(f"    Anzarut:      {anz_aad:.1f}pp")
    print(f"    Cal FFBS:     {cal_aad:.1f}pp")
    print(f"    Cal MoM:      {cal_mom_aad:.1f}pp")
    print(f"    Cal MoM-t:    {cal_mom_t_aad:.1f}pp")
    print(f"    Dol FFBS:     {dol_aad:.1f}pp")
    print(f"    Dol MoM:      {dol_mom_aad:.1f}pp")
    print(f"    Dol MoM-t:    {dol_mom_t_aad:.1f}pp")

    # Save
    table3_df = pd.DataFrame({
        "p": prob_levels,
        "anzarut": [anzarut_coverage[p] for p in prob_levels],
        "calendar_ffbs": [cal_coverage.get(p, 0) for p in prob_levels],
        "calendar_mom": [cal_mom_coverage.get(p, 0) for p in prob_levels],
        "calendar_mom_t": [cal_mom_t_coverage.get(p, 0) for p in prob_levels],
        "dollar_ffbs": [dol_coverage.get(p, 0) for p in prob_levels],
        "dollar_mom": [dol_mom_coverage.get(p, 0) for p in prob_levels],
        "dollar_mom_t": [dol_mom_t_coverage.get(p, 0) for p in prob_levels],
        "ideal": [p * 100 for p in prob_levels],
    })
    table3_df.to_csv(OUTPUT_DIR / "table3_intraday_gibbs_coverage.csv", index=False)
    print(f"\n  Results saved to {OUTPUT_DIR / 'table3_intraday_gibbs_coverage.csv'}")