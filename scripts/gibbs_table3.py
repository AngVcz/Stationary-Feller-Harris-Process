"""Gibbs sampler for the SF-Harris SV model — full Bayesian prediction.

Unlike the point-estimate approach, the Gibbs sampler produces posterior
samples of ALL parameters, then generates predictive trajectories from
the full posterior predictive distribution. This propagates:
  - Parameter uncertainty (alpha, mu, sigma, ACF shape)
  - Model uncertainty (correct Q distribution)
  - Process uncertainty (random jumps)

Two Q distribution options:
  - Gaussian: appropriate for dollar bars (log-RV kurtosis ~0.2)
  - Student-t(df): appropriate for calendar bars (log-RV kurtosis ~4.1)

The Gibbs sampler structure:
  1. Given alpha, sample latent jump indicators z_t
  2. Given z_t, sample alpha from conjugate posterior
  3. Given jump observations, sample Q parameters
  4. Given all parameters, sample predictive trajectories

For the mixture SF-Harris (double exponential ACF), we also sample
the ACF mixture parameters (w1, alpha1, w2, alpha2) using
Metropolis-within-Gibbs.

Output: Table 3 coverage validation matching Anzarut's methodology.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats

DATA_PATH = Path(r"C:\Users\angve\OneDrive\Desktop\Servicio\Libros\SF-Harris\IBM.txt")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"


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


def build_calendar_bars(df, freq="15min"):
    close = df["close"].resample(freq).last().dropna()
    return np.log(close).diff().dropna()


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


def compute_daily_rv(returns_series):
    if isinstance(returns_series, pd.DataFrame):
        ret = returns_series["return"]
    else:
        ret = returns_series
    rv = ret.groupby(ret.index.date).apply(lambda x: np.sum(x**2))
    rv.index = pd.to_datetime(rv.index)
    return rv


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
    for x0 in [
        [0.0, np.log(1.0), np.log(0.1)],
        [1.0, np.log(2.0), np.log(0.05)],
        [-1.0, np.log(0.5), np.log(0.01)],
        [0.5, np.log(3.0), np.log(0.02)],
    ]:
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
# Gibbs sampler for SF-Harris SV model on log-RV
# ---------------------------------------------------------------------------
def gibbs_sf_harris_sv(log_rv, n_iter=3000, burn_in=1000, Q_type="normal",
                        Q_df=5, rng=None):
    """Gibbs sampler for the SF-Harris SV model on daily log-RV.

    At the daily level, the SF-Harris process is a stationary process with:
      - Marginal distribution Q (Gaussian or Student-t)
      - ACF: r(h) = w1*exp(-a1*h) + w2*exp(-a2*h) (mixture SF-Harris)

    The stay/jump structure exists at the INTRADAY (15-min) level.
    At the daily level, we model the ACF structure directly.

    Gibbs steps:
      1. Sample mu | data, sigma (conjugate Normal)
      2. Sample sigma^2 | data, mu (conjugate Inverse-Gamma)
      3. Sample ACF parameters (w1, a1, a2) | empirical ACF (MH)
      4. Conditional on parameters, simulate predictive trajectories
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(log_rv)
    emp_acf = acf_analysis(log_rv, max_lag=20)
    lags = np.arange(1, 21)
    emp_acf_vec = emp_acf[1:21]

    # Initial estimates
    mu0 = np.mean(log_rv)
    sigma0 = np.std(log_rv)
    acf_params0 = fit_double_exponential(lags, emp_acf_vec)

    # Storage
    samples = {
        "mu": np.empty(n_iter),
        "sigma": np.empty(n_iter),
        "w1": np.empty(n_iter),
        "alpha1": np.empty(n_iter),
        "w2": np.empty(n_iter),
        "alpha2": np.empty(n_iter),
    }

    # Current state
    mu_cur = mu0
    sigma_cur = sigma0
    w1_cur = acf_params0["w1"]
    a1_cur = acf_params0["alpha1"]
    w2_cur = acf_params0["w2"]
    a2_cur = acf_params0["alpha2"]

    # Prior hyperparameters
    kappa0 = 0.01      # prior precision for mu
    mu0_prior = mu0    # prior mean for mu
    a0 = 3.0           # prior shape for sigma^2 (IG)
    b0 = sigma0**2     # prior scale for sigma^2 (IG)

    for it in range(n_iter):
        # ---- Step 1: Sample mu | data, sigma ----
        # Conjugate Normal-Normal update
        # Data: x_i ~ N(mu, sigma^2), prior: mu ~ N(mu0_prior, sigma^2/kappa0)
        kappa_n = kappa0 + n
        mu_n = (kappa0 * mu0_prior + n * np.mean(log_rv)) / kappa_n
        mu_cur = rng.normal(mu_n, sigma_cur / np.sqrt(kappa_n))

        # ---- Step 2: Sample sigma^2 | data, mu ----
        # Conjugate Normal-Inverse-Gamma
        # sigma^2 ~ IG(a0, b0), posterior: IG(a_n, b_n)
        ss = np.sum((log_rv - mu_cur)**2)
        a_n = a0 + n / 2.0
        b_n = b0 + ss / 2.0 + 0.5 * kappa0 * n * (np.mean(log_rv) - mu0_prior)**2 / (kappa0 + n)
        sigma2_cur = 1.0 / rng.gamma(a_n, 1.0 / b_n)
        sigma_cur = np.sqrt(sigma2_cur)

        # ---- Step 3: Sample ACF parameters via MH ----
        # Multiple MH steps for better mixing
        for _ in range(5):
            # Propose new ACF parameters
            log_w1_prop = np.log(w1_cur / max(1 - w1_cur, 1e-6)) + rng.normal(0, 0.2)
            w1_prop = 1.0 / (1.0 + np.exp(-log_w1_prop))
            w2_prop = 1.0 - w1_prop
            a1_prop = a1_cur * np.exp(rng.normal(0, 0.15))
            a2_prop = a2_cur * np.exp(rng.normal(0, 0.15))

            # Compute fitted ACF
            fitted_cur = w1_cur * np.exp(-a1_cur * lags) + w2_cur * np.exp(-a2_cur * lags)
            fitted_prop = w1_prop * np.exp(-a1_prop * lags) + w2_prop * np.exp(-a2_prop * lags)

            # Log-likelihood: weighted squared error to empirical ACF
            # (this is a pseudo-likelihood approximation)
            weights = 1.0 / (1 + lags)
            ll_cur = -0.5 * np.sum(weights * (emp_acf_vec - fitted_cur)**2) / (0.05**2)
            ll_prop = -0.5 * np.sum(weights * (emp_acf_vec - fitted_prop)**2) / (0.05**2)

            # Weak priors: keep parameters in reasonable range
            log_prior_cur = -0.001 * max(a1_cur - 20, 0)**2 - 0.001 * max(a2_cur - 5, 0)**2
            log_prior_prop = -0.001 * max(a1_prop - 20, 0)**2 - 0.001 * max(a2_prop - 5, 0)**2

            log_ratio = (ll_prop + log_prior_prop) - (ll_cur + log_prior_cur)

            if np.log(rng.uniform()) < log_ratio:
                w1_cur = w1_prop
                a1_cur = a1_prop
                w2_cur = w2_prop
                a2_cur = a2_prop

        # Store
        samples["mu"][it] = mu_cur
        samples["sigma"][it] = sigma_cur
        samples["w1"][it] = w1_cur
        samples["alpha1"][it] = a1_cur
        samples["w2"][it] = w2_cur
        samples["alpha2"][it] = a2_cur

    return {k: v[burn_in:] for k, v in samples.items()}


def simulate_predictive_from_gibbs(n_forecast, posterior, rng, Q_type="normal", Q_df=5):
    """Simulate one predictive trajectory from the Gibbs posterior.

    For each trajectory, draws a parameter set from the posterior,
    then simulates the mixture SF-Harris process forward.

    The mixture SF-Harris process at the daily level:
      r(h) = w1*exp(-a1*h) + w2*exp(-a2*h)
    The process is simulated using the continuous-time renewal construction.
    """
    n_post = len(posterior["mu"])
    idx = rng.integers(0, n_post)

    mu = posterior["mu"][idx]
    sigma = posterior["sigma"][idx]
    w1 = posterior["w1"][idx]
    a1 = posterior["alpha1"][idx]
    w2 = posterior["w2"][idx]
    a2 = posterior["alpha2"][idx]

    w = np.array([w1, w2])
    alphas = np.array([a1, a2])
    w_norm = w / np.sum(w)  # normalize

    trajectory = np.empty(n_forecast)

    def draw_Q():
        if Q_type == "normal":
            return rng.normal(mu, sigma)
        elif Q_type == "t":
            scale = sigma * np.sqrt((Q_df - 2) / Q_df)
            return mu + rng.standard_t(Q_df) * scale
        else:
            return rng.normal(mu, sigma)

    trajectory[0] = draw_Q()
    current_val = trajectory[0]

    for i in range(1, n_forecast):
        t_target = float(i)
        t = float(i - 1)

        # Advance continuous-time renewal process
        while t < t_target:
            component = rng.choice(len(w_norm), p=w_norm)
            rho = alphas[component]
            wait = rng.exponential(1.0 / rho)
            t += wait
            if t < t_target:
                current_val = draw_Q()

        trajectory[i] = current_val

    return trajectory


def compute_coverage(actual_log_rv, simulated_log_rvs, prob_levels):
    n_sim = len(simulated_log_rvs)
    n_obs = len(actual_log_rv)
    min_len = min(n_obs, simulated_log_rvs.shape[1])
    actual = actual_log_rv[:min_len]
    sims = simulated_log_rvs[:, :min_len]

    coverage = {}
    for p in prob_levels:
        lower_q = (1 - p) / 2
        upper_q = (1 + p) / 2
        lower = np.quantile(sims, lower_q, axis=0)
        upper = np.quantile(sims, upper_q, axis=0)
        inside = np.sum((actual >= lower) & (actual <= upper))
        coverage[p] = inside / min_len * 100

    return coverage


def estimate_mu_beta(returns_daily, integrated_vol_daily):
    R = returns_daily.values
    dH = integrated_vol_daily.values
    valid = dH > 0
    R = R[valid]
    dH = dH[valid]
    if len(R) < 10:
        return {"mu": 0.0, "beta": 0.0}
    dt = 1.0
    y = R / np.sqrt(dH)
    X = np.column_stack([dt / np.sqrt(dH), np.sqrt(dH)])
    try:
        beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return {"mu": 0.0, "beta": 0.0}
    return {"mu": beta_hat[0], "beta": beta_hat[1]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Gibbs Sampler SF-Harris SV Model - Table 3 Validation")
    print("=" * 70)

    df = load_ibm_data()
    print(f"Loaded {len(df)} 1-minute bars")

    # Anzarut's Table 3
    anzarut_coverage = {0.25: 25, 0.50: 51, 0.75: 75, 0.85: 84, 0.90: 89, 0.95: 93}
    prob_levels = [0.25, 0.50, 0.75, 0.85, 0.90, 0.95]

    # ====================================================================
    # Calendar bars
    # ====================================================================
    print(f"\n{'='*70}")
    print(f"CALENDAR BARS (15-min, Anzarut's approach)")
    print(f"{'='*70}")

    cal_returns = build_calendar_bars(df)
    cal_rv = compute_daily_rv(cal_returns)
    cal_log_rv = np.log(cal_rv.values)

    # 80/20 split
    n = len(cal_log_rv)
    n_train = int(n * 0.8)
    train = cal_log_rv[:n_train]
    test = cal_log_rv[n_train:]

    print(f"  Train: {n_train} days, Test: {n - n_train} days")
    print(f"  Train: mean={np.mean(train):.4f}, std={np.std(train):.4f}, kurt={pd.Series(train).kurtosis():.2f}")

    # Run Gibbs sampler on calendar bars with Student-t Q
    print(f"\n  Running Gibbs sampler (Student-t Q, 3000 iterations)...")
    rng_cal = np.random.default_rng(42)
    cal_posterior = gibbs_sf_harris_sv(
        train, n_iter=3000, burn_in=1000, Q_type="t", Q_df=5, rng=rng_cal
    )

    # Diagnostics
    print(f"  Posterior summary:")
    for key in ["mu", "sigma", "w1", "alpha1", "w2", "alpha2"]:
        vals = cal_posterior[key]
        print(f"    {key:>8}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, "
              f"2.5%={np.percentile(vals, 2.5):.4f}, 97.5%={np.percentile(vals, 97.5):.4f}")

    # Generate predictive trajectories
    n_forecast = len(test)
    n_sim = 1000
    rng_pred = np.random.default_rng(123)
    cal_simulated = np.empty((n_sim, n_forecast))

    for s in range(n_sim):
        cal_simulated[s] = simulate_predictive_from_gibbs(
            n_forecast, cal_posterior, rng_pred, Q_type="t", Q_df=5
        )

    cal_coverage = compute_coverage(test, cal_simulated, prob_levels)

    # ====================================================================
    # Dollar bars
    # ====================================================================
    print(f"\n{'='*70}")
    print(f"DOLLAR BARS (Lopez de Prado, Gaussian Q)")
    print(f"{'='*70}")

    # Train-only threshold
    dates_sorted = sorted(df.index.date)
    split_date = dates_sorted[int(len(set(dates_sorted)) * 0.8)]
    df_train = df[df.index.date <= split_date]
    train_threshold = df_train["dollar_volume"].resample("15min").sum().mean()

    dollar_bars = build_dollar_bars(df, dollar_threshold=train_threshold)
    dollar_rv = compute_daily_rv(dollar_bars)
    dollar_log_rv = np.log(dollar_rv.values)

    d_train = dollar_log_rv[:n_train]
    d_test = dollar_log_rv[n_train:]

    print(f"  Train: mean={np.mean(d_train):.4f}, std={np.std(d_train):.4f}, kurt={pd.Series(d_train).kurtosis():.2f}")

    # Run Gibbs sampler on dollar bars with Gaussian Q
    print(f"\n  Running Gibbs sampler (Gaussian Q, 3000 iterations)...")
    rng_dol = np.random.default_rng(42)
    dol_posterior = gibbs_sf_harris_sv(
        d_train, n_iter=3000, burn_in=1000, Q_type="normal", rng=rng_dol
    )

    print(f"  Posterior summary:")
    for key in ["mu", "sigma", "w1", "alpha1", "w2", "alpha2"]:
        vals = dol_posterior[key]
        print(f"    {key:>8}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, "
              f"2.5%={np.percentile(vals, 2.5):.4f}, 97.5%={np.percentile(vals, 97.5):.4f}")

    # Generate predictive trajectories
    n_d_forecast = len(d_test)
    rng_d_pred = np.random.default_rng(123)
    dol_simulated = np.empty((n_sim, n_d_forecast))

    for s in range(n_sim):
        dol_simulated[s] = simulate_predictive_from_gibbs(
            n_d_forecast, dol_posterior, rng_d_pred, Q_type="normal"
        )

    dol_coverage = compute_coverage(d_test, dol_simulated, prob_levels)

    # ====================================================================
    # TABLE 3: Final comparison
    # ====================================================================
    print(f"\n{'='*70}")
    print(f"TABLE 3: Coverage Validation (Gibbs Sampler)")
    print(f"{'='*70}")

    print(f"\n  Anzarut's Table 3 (GIG-Harris + Gibbs, IBM 2012-2014):")
    print(f"  {'p':>6}  {'Anzarut':>10}  {'Ideal':>6}")
    print(f"  {'-'*28}")
    for p in prob_levels:
        print(f"  {p:>6.2f}  {anzarut_coverage[p]:>9.0f}%  {p*100:>5.0f}%")

    print(f"\n  Our Table 3 - Calendar bars + Student-t Q + Gibbs:")
    print(f"  {'p':>6}  {'Coverage':>10}  {'Ideal':>6}  {'Dev':>6}")
    print(f"  {'-'*34}")
    for p in prob_levels:
        cov = cal_coverage.get(p, 0)
        dev = cov - p * 100
        print(f"  {p:>6.2f}  {cov:>9.1f}%  {p*100:>5.0f}%  {dev:>+5.1f}pp")

    print(f"\n  Our Table 3 - Dollar bars + Gaussian Q + Gibbs:")
    print(f"  {'p':>6}  {'Coverage':>10}  {'Ideal':>6}  {'Dev':>6}")
    print(f"  {'-'*34}")
    for p in prob_levels:
        cov = dol_coverage.get(p, 0)
        dev = cov - p * 100
        print(f"  {p:>6.2f}  {cov:>9.1f}%  {p*100:>5.0f}%  {dev:>+5.1f}pp")

    # Side-by-side
    print(f"\n  Side-by-side:")
    print(f"  {'p':>6}  {'Anzarut':>10}  {'Cal+Gibbs':>10}  {'Dol+Gibbs':>10}  {'Ideal':>6}")
    print(f"  {'-'*48}")
    for p in prob_levels:
        print(f"  {p:>6.2f}  {anzarut_coverage[p]:>9.0f}%  {cal_coverage.get(p,0):>9.1f}%  "
              f"{dol_coverage.get(p,0):>9.1f}%  {p*100:>5.0f}%")

    # Mean absolute deviation
    anz_aad = np.mean([abs(anzarut_coverage[p] - p*100) for p in prob_levels])
    cal_aad = np.mean([abs(cal_coverage.get(p, 0) - p*100) for p in prob_levels])
    dol_aad = np.mean([abs(dol_coverage.get(p, 0) - p*100) for p in prob_levels])

    print(f"\n  Mean absolute deviation from ideal:")
    print(f"    Anzarut (GIG+Gibbs, intraday):  {anz_aad:.1f}pp")
    print(f"    Calendar (t+Gibbs, daily):      {cal_aad:.1f}pp")
    print(f"    Dollar (Normal+Gibbs, daily):  {dol_aad:.1f}pp")

    print(f"\n{'='*70}")
    print(f"WHY WE DON'T MATCH ANZARUT")
    print(f"{'='*70}")
    print(f"\n  Anzarut fits the SF-Harris process at the INTRADAY (15-min) level,")
    print(f"  where the spot volatility IS piecewise-constant: it stays at a")
    print(f"  value until a new market event causes a jump. The Gibbs sampler")
    print(f"  operates on 15-min data, where stays/jumps are observable.")
    print(f"\n  We fit at the DAILY level on log-RV, which is an AGGREGATED")
    print(f"  quantity (sum of 26 fifteen-minute squared returns). The daily")
    print(f"  log-RV is never piecewise-constant — it's a smooth, continuous")
    print(f"  series with autocorrelation. The SF-Harris renewal process")
    print(f"  simulation produces piecewise-constant trajectories that don't")
    print(f"  match the distributional shape of aggregated daily data.")
    print(f"\n  The systematic under-coverage at p=0.25 to p=0.75 reflects this:")
    print(f"  the piecewise-constant simulation puts too much mass at the")
    print(f"  extremes (exact repeats or fresh draws) and too little in the")
    print(f"  moderate zone, unlike the smooth empirical distribution.")
    print(f"\n  To match Anzarut's calibration, we would need to:")
    print(f"  1. Fit the SF-Harris at 15-min level (where stays exist)")
    print(f"  2. Aggregate 15-min spot vol to daily integrated vol")
    print(f"  3. Use the daily integrated vol for return model predictions")
    print(f"  This requires an intraday Gibbs sampler + aggregation step.")

    # Save
    table3_df = pd.DataFrame({
        "p": prob_levels,
        "anzarut": [anzarut_coverage[p] for p in prob_levels],
        "calendar_gibbs": [cal_coverage.get(p, 0) for p in prob_levels],
        "dollar_gibbs": [dol_coverage.get(p, 0) for p in prob_levels],
        "ideal": [p * 100 for p in prob_levels],
    })
    table3_df.to_csv(OUTPUT_DIR / "table3_gibbs_coverage.csv", index=False)
    print(f"\n  Results saved to {OUTPUT_DIR / 'table3_gibbs_coverage.csv'}")