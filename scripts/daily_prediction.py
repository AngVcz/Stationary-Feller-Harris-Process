"""Daily return prediction using SF-Harris volatility + emission function.

Full SV model chain:
1. SF-Harris predicts spot volatility H_t (15-min or dollar-bar level)
2. Integrate to get daily variance: tau* = sum(H_i * dt) per day
3. Emission function: R_daily | tau* ~ N(mu + beta*tau*, tau*)
4. Coverage validation of daily return predictions

This extends our intraday volatility model (0.2-0.5pp AAD) to next-day
return predictions, addressing the aggregation gap (10pp at daily level).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from anzarut_replication import (
    load_ibm_data,
    compute_15min_returns,
    detect_and_remove_jumps,
    estimate_periodicity,
    estimate_alpha,
    gibbs_gig_harris,
    simulate_predictive_sf_harris,
    compute_coverage,
)

from benchmark_sv_models import (
    fit_garch,
    fit_egarch,
    simulate_garch,
    simulate_egarch,
    fit_heston,
    simulate_heston,
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


def estimate_mu_beta(daily_returns, daily_rv):
    """Estimate mu and beta from the emission function.

    R_i = mu + beta * H*_i + sqrt(H*_i) * Z_i
    where H*_i ≈ RV_i (daily realized variance).

    This is a weighted least squares regression:
    R_i / sqrt(RV_i) = mu / sqrt(RV_i) + beta * sqrt(RV_i) + Z_i
    """
    rv = daily_rv[daily_rv > 0]  # filter zero RV
    ret = daily_returns.loc[rv.index]

    # Weighted regression: R_i = mu + beta * RV_i + error
    # Simple OLS first
    X = np.column_stack([np.ones(len(rv)), rv.values])
    y = ret.values

    # OLS: (X'X)^{-1} X'y
    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    mu_hat, beta_hat = beta_ols

    # Residual variance
    residuals = y - X @ beta_ols
    sigma2_hat = np.sum(residuals**2) / (len(y) - 2)

    # Conjugate Bayesian posterior (Anzarut Section 4.4.1)
    # Prior: mu ~ N(0, 100), beta ~ N(0, 100)  (vague)
    # Posterior: see Anzarut eqs on p.56-57
    H_star = rv.values
    H_star_inv = 1.0 / H_star
    R = ret.values

    # Sufficient statistics (Anzarut notation)
    n = len(R)
    R_bar = np.mean(R)
    R2_bar = np.mean(R * H_star)
    H_bar = np.mean(H_star)
    H2_bar = np.mean(H_star_inv)

    # Prior hyperparameters (vague)
    m_mu, s2_mu = 0.0, 100.0
    m_beta, s2_beta = 0.0, 100.0
    h = 1.0  # 1 day

    # Posterior (Anzarut p.57)
    A = h**2 * H2_bar + 1.0 / s2_mu
    B = h * R2_bar + m_mu / s2_mu
    C = h
    D = H_bar + 1.0 / s2_beta
    E = R_bar + m_beta / s2_beta
    F = A * D - C**2

    mu_post_mean = (D * B - C * E) / F
    mu_post_var = D / F
    beta_post_mean = (E * A - B * C) / F
    beta_post_var = A / F

    return {
        "mu_hat": mu_hat, "beta_hat": beta_hat,
        "mu_post_mean": mu_post_mean, "mu_post_var": mu_post_var,
        "beta_post_mean": beta_post_mean, "beta_post_var": beta_post_var,
        "sigma2_hat": sigma2_hat,
        "n": n,
    }


def simulate_daily_returns(mu_beta, sim_log_spot, test_idx, rv_df, periodicity, n_sim,
                           close_daily, rng=None):
    """Simulate daily returns from simulated volatility trajectories.

    For each simulated volatility trajectory:
    1. Aggregate 15-min spot vol to daily integrated variance: tau* = sum(exp(log_spot) * f * dt)
    2. Generate daily return: R ~ N(mu + beta*tau*, tau*)

    Args:
        close_daily: pre-computed daily close prices (avoids reloading data)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Day assignments (use Timestamp for alignment)
    test_dates = pd.to_datetime([d.date() if hasattr(d, 'date') else pd.Timestamp(d).date()
                                  for d in test_idx])
    unique_days = sorted(set(test_dates))
    n_days = len(unique_days)
    day_map = {d: i for i, d in enumerate(unique_days)}
    day_assign = np.array([day_map[d] for d in test_dates])

    # Periodicity factors
    time_strings = [d.strftime("%H:%M") for d in test_idx]
    f_t = np.array([periodicity.get(t, 1.0) for t in time_strings])

    n_test = min(sim_log_spot.shape[1], len(day_assign))

    # Aggregate simulated 15-min spot vol to daily integrated variance
    # tau* = sum(exp(log_spot_i) * f_i * dt) where dt = 1/26 (15 min)
    dt = 1.0 / 26.0  # 15 min in units of 1 day
    exp_sim = np.exp(sim_log_spot[:, :n_test])  # (n_sim, n_test)
    weighted = exp_sim * f_t[np.newaxis, :n_test] * dt  # (n_sim, n_test)

    # Day assignment matrix
    day_matrix = np.zeros((n_test, n_days))
    for i in range(n_test):
        day_matrix[i, day_assign[i]] = 1.0

    tau_star = weighted @ day_matrix  # (n_sim, n_days) — daily integrated variance

    # Simulate daily returns using emission function
    # R | tau* ~ N(mu + beta*tau*, tau*)
    mu = mu_beta["mu_post_mean"]
    beta = mu_beta["beta_post_mean"]

    mean_returns = mu + beta * tau_star  # (n_sim, n_days)
    std_returns = np.sqrt(np.maximum(tau_star, 1e-20))  # (n_sim, n_days)

    sim_returns = mean_returns + std_returns * rng.normal(size=(n_sim, n_days))

    # Actual daily RV from 15-min data (test period)
    test_rv_df = rv_df.loc[test_idx[0]:].copy()
    test_rv_df["date"] = pd.to_datetime(test_rv_df.index.date)
    daily_rv = test_rv_df.groupby("date")["rv_15min"].sum()

    # Actual daily log-returns from price data
    daily_log_returns = np.log(close_daily).diff().dropna()

    # Align: find dates that exist in both daily_rv and daily_log_returns
    common_dates = daily_rv.index.intersection(daily_log_returns.index)
    common_dates = common_dates[common_dates.isin(unique_days)]  # only test period

    actual_returns = daily_log_returns.loc[common_dates].values
    actual_rv = daily_rv.loc[common_dates].values

    # Map common dates to day indices in tau_star
    day_indices = [day_map[d] for d in common_dates if d in day_map]
    n_valid = len(day_indices)

    # Extract simulated returns/RV for valid days only
    sim_returns_valid = sim_returns[:, day_indices]
    tau_star_valid = tau_star[:, day_indices]

    return sim_returns_valid, actual_returns, tau_star_valid, actual_rv, n_valid, common_dates


if __name__ == "__main__":
    print("=" * 70)
    print("DAILY RETURN PREDICTION: SF-Harris Volatility + Emission Function")
    print("Y_t = mu*dt + beta*tau*_t + B_{tau*_t}")
    print("=" * 70)

    # ==================================================================
    # STEP 1: Load data
    # ==================================================================
    print("\n--- Step 1: Load and preprocess data ---")
    df = load_ibm_data()
    print(f"  Loaded {len(df)} minute bars")

    returns = compute_15min_returns(df)
    print(f"  {len(returns)} 15-min return observations")

    returns_clean = detect_and_remove_jumps(returns, n_passes=2, top_pct=0.001)
    print(f"  Clean returns: {len(returns_clean)}")

    periodicity = estimate_periodicity(returns)
    print(f"  Periodicity range: {periodicity.min():.2f} to {periodicity.max():.2f}")

    # ==================================================================
    # STEP 2: Compute 15-min spot volatility
    # ==================================================================
    print("\n--- Step 2: Compute 15-min spot volatility ---")
    rv_df = returns_clean.to_frame("return")
    rv_df["rv_15min"] = rv_df["return"]**2
    rv_df["time"] = rv_df.index.strftime("%H:%M")
    rv_df["f_t"] = rv_df["time"].map(periodicity)
    rv_df["f_t"] = rv_df["f_t"].fillna(1.0)
    rv_df["rv_adj"] = rv_df["rv_15min"] / rv_df["f_t"]
    rv_df["log_spot"] = np.log(rv_df["rv_adj"].clip(lower=1e-20))

    log_spot = rv_df["log_spot"].replace([np.inf, -np.inf], np.nan).dropna()
    log_spot_vals = log_spot.values

    n_15 = len(log_spot_vals)
    split_15 = int(n_15 * 0.8)
    train_log = log_spot_vals[:split_15]
    test_log = log_spot_vals[split_15:]
    train_idx = log_spot.index[:split_15]
    test_idx = log_spot.index[split_15:]

    print(f"  Train: {split_15}, Test: {n_15 - split_15}")

    # ==================================================================
    # STEP 3: Estimate mu and beta (emission function parameters)
    # ==================================================================
    print("\n--- Step 3: Estimate emission function parameters ---")

    # Compute daily RV from 15-min data (training period)
    train_rv_df = rv_df.loc[train_idx[0]:train_idx[-1]].copy()
    train_rv_df["date"] = pd.to_datetime(train_rv_df.index.date)
    daily_rv_train = train_rv_df.groupby("date")["rv_15min"].sum()

    # Compute daily log-returns (training period)
    close_daily = df["close"].resample("D").last().dropna()
    train_end_date = pd.Timestamp(train_idx[-1].date())
    close_train = close_daily[:train_end_date]

    daily_returns_train = np.log(close_train).diff().dropna()

    # Align dates (both are now Timestamps)
    common_dates = daily_rv_train.index.intersection(daily_returns_train.index)
    print(f"  Daily RV dates: {len(daily_rv_train)}, Daily return dates: {len(daily_returns_train)}, Common: {len(common_dates)}")
    rv_aligned = daily_rv_train.loc[common_dates].values
    ret_aligned = daily_returns_train.loc[common_dates].values

    # Filter out zero/negative RV
    valid = rv_aligned > 0
    rv_valid = rv_aligned[valid]
    ret_valid = ret_aligned[valid]

    # Estimate mu and beta
    mu_beta = estimate_mu_beta(
        pd.Series(ret_valid, index=common_dates[valid]),
        pd.Series(rv_valid, index=common_dates[valid])
    )

    print(f"  mu:    {mu_beta['mu_hat']:.6f} (posterior mean: {mu_beta['mu_post_mean']:.6f})")
    print(f"  beta:  {mu_beta['beta_hat']:.6f} (posterior mean: {mu_beta['beta_post_mean']:.6f})")
    print(f"  sigma2: {mu_beta['sigma2_hat']:.6f}")
    print(f"  N days: {mu_beta['n']}")

    # ==================================================================
    # STEP 4: Gibbs sampler on 15-min series
    # ==================================================================
    print("\n--- Step 4: Gibbs sampler (eps=0.1, best from before) ---")
    alpha_est = estimate_alpha(train_log)
    rng_gibbs = np.random.default_rng(42)
    gibbs_result = gibbs_gig_harris(train_log, alpha_init=alpha_est["alpha_acf"],
                                      epsilon=0.1, n_iter=5000, burn_in=2000, rng=rng_gibbs)

    # ==================================================================
    # STEP 5: Simulate 15-min volatility trajectories
    # ==================================================================
    print("\n--- Step 5: Simulate 15-min volatility trajectories ---")
    n_sim = 2000
    rng_sim = np.random.default_rng(123)
    sim_log = simulate_predictive_sf_harris(
        train_log, test_log, gibbs_result, gibbs_result["alpha"],
        Q_type="empirical", n_sim=n_sim, rng=rng_sim
    )
    print(f"  Simulated {n_sim} trajectories of {len(test_log)} 15-min intervals each")

    # Compute daily close prices (shared across steps)
    close_daily = df["close"].resample("D").last().dropna()

    # ==================================================================
    # STEP 6: Generate daily returns using emission function
    # ==================================================================
    print("\n--- Step 6: Generate daily returns from emission function ---")
    print(f"  Emission: R | tau* ~ N(mu + beta*tau*, tau*)")
    print(f"  mu={mu_beta['mu_post_mean']:.6f}, beta={mu_beta['beta_post_mean']:.6f}")

    sim_returns, actual_returns, tau_star, actual_rv, n_valid, common_dates = \
        simulate_daily_returns(mu_beta, sim_log, test_idx, rv_df, periodicity, n_sim,
                              close_daily=close_daily, rng=np.random.default_rng(456))

    # ==================================================================
    # STEP 7: Coverage validation of daily returns
    # ==================================================================
    print(f"\n--- Step 7: Daily return coverage validation ---")
    print(f"  {n_valid} valid test days with both returns and volatility data")

    if n_valid > 0:
        # Coverage of daily returns
        cov_returns = compute_coverage(actual_returns, sim_returns, PROB_LEVELS)
        aad_returns = np.mean([abs(cov_returns[p] - p*100) for p in PROB_LEVELS])

        print(f"\n  Daily return coverage (emission function):")
        print(f"  {'p':>6}  {'Coverage':>10}  {'Ideal':>6}  {'Dev':>6}")
        print(f"  {'-'*34}")
        for p in PROB_LEVELS:
            dev = cov_returns[p] - p*100
            print(f"  {p:>6.2f}  {cov_returns[p]:>9.1f}%  {p*100:>5.0f}%  {dev:>+5.1f}pp")
        print(f"  AAD = {aad_returns:.1f}pp")

        # Coverage of daily log-RV (volatility only)
        log_rv_actual = np.log(actual_rv)
        log_rv_sim = np.log(np.maximum(tau_star, 1e-20))
        cov_rv = compute_coverage(log_rv_actual, log_rv_sim, PROB_LEVELS)
        aad_rv = np.mean([abs(cov_rv[p] - p*100) for p in PROB_LEVELS])

        print(f"\n  Daily log-RV coverage (volatility only):")
        print(f"  {'p':>6}  {'Coverage':>10}  {'Ideal':>6}  {'Dev':>6}")
        print(f"  {'-'*34}")
        for p in PROB_LEVELS:
            dev = cov_rv[p] - p*100
            print(f"  {p:>6.2f}  {cov_rv[p]:>9.1f}%  {p*100:>5.0f}%  {dev:>+5.1f}pp")
        print(f"  AAD = {aad_rv:.1f}pp")
    else:
        print(f"  Not enough valid data for coverage (n_valid={n_valid})")
        aad_returns = None
        aad_rv = None

    # ==================================================================
    # STEP 8: Dollar bar version
    # ==================================================================
    print("\n" + "=" * 70)
    print("DOLLAR BAR VERSION")
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

    # Estimate mu and beta for dollar bars
    # For dollar bars, "daily" = sum of N bars per day where N varies
    # Simpler approach: use bar-level returns and RV directly
    dol_rv_train = dol_train**2
    mu_beta_dol = estimate_mu_beta(
        pd.Series(dol_train),
        pd.Series(dol_rv_train)
    )

    print(f"  mu:    {mu_beta_dol['mu_hat']:.6f} (posterior: {mu_beta_dol['mu_post_mean']:.6f})")
    print(f"  beta:  {mu_beta_dol['beta_hat']:.6f} (posterior: {mu_beta_dol['beta_post_mean']:.6f})")

    # Gibbs on dollar bars
    dol_alpha = estimate_alpha(dol_train_spot)
    rng_gibbs_dol = np.random.default_rng(55)
    gibbs_dol = gibbs_gig_harris(dol_train_spot, alpha_init=dol_alpha["alpha_acf"],
                                    epsilon=0.1, n_iter=5000, burn_in=2000, rng=rng_gibbs_dol)

    # Simulate
    rng_sim_dol = np.random.default_rng(321)
    sim_dol = simulate_predictive_sf_harris(
        dol_train_spot, dol_test_spot, gibbs_dol, gibbs_dol["alpha"],
        Q_type="empirical", n_sim=n_sim, rng=rng_sim_dol
    )

    # For dollar bars: each bar's return ~ N(mu + beta*RV_bar, RV_bar)
    # where RV_bar = exp(log_spot_bar)
    sim_rv_dol = np.exp(sim_dol)  # simulated RV for each bar
    mu_dol = mu_beta_dol["mu_post_mean"]
    beta_dol = mu_beta_dol["beta_post_mean"]

    mean_ret_dol = mu_dol + beta_dol * sim_rv_dol
    std_ret_dol = np.sqrt(np.maximum(sim_rv_dol, 1e-20))
    rng_emission = np.random.default_rng(789)
    sim_returns_dol = mean_ret_dol + std_ret_dol * rng_emission.normal(size=sim_rv_dol.shape)

    # Coverage: compare against actual bar returns
    cov_dol_returns = compute_coverage(dol_test, sim_returns_dol, PROB_LEVELS)
    aad_dol_returns = np.mean([abs(cov_dol_returns[p] - p*100) for p in PROB_LEVELS])

    print(f"\n  Dollar bar return coverage (emission function):")
    print(f"  {'p':>6}  {'Coverage':>10}  {'Ideal':>6}  {'Dev':>6}")
    print(f"  {'-'*34}")
    for p in PROB_LEVELS:
        dev = cov_dol_returns[p] - p*100
        print(f"  {p:>6.2f}  {cov_dol_returns[p]:>9.1f}%  {p*100:>5.0f}%  {dev:>+5.1f}pp")
    print(f"  AAD = {aad_dol_returns:.1f}pp")

    # Also coverage of bar-level volatility (log RV)
    cov_dol_rv = compute_coverage(dol_test_spot, sim_dol, PROB_LEVELS)
    aad_dol_rv = np.mean([abs(cov_dol_rv[p] - p*100) for p in PROB_LEVELS])

    print(f"\n  Dollar bar volatility coverage (log RV, from before):")
    print(f"  AAD = {aad_dol_rv:.1f}pp")

    # ==================================================================
    # STEP 9: Direct daily SF-Harris model
    # ==================================================================
    # Instead of aggregating 15-min predictions, fit SF-Harris directly
    # on daily realized variance (RV from 15-min returns aggregated per day).
    # This avoids the aggregation problem entirely.
    # ==================================================================
    print("\n" + "=" * 70)
    print("DIRECT DAILY SF-HARRIS MODEL")
    print("Fit SF-Harris on daily log-RV directly, then predict daily returns")
    print("=" * 70)

    # Compute daily RV from 15-min returns
    rv_df_daily = rv_df.copy()
    rv_df_daily["date"] = pd.to_datetime(rv_df_daily.index.date)
    daily_rv_all = rv_df_daily.groupby("date")["rv_15min"].sum()
    daily_rv_all = daily_rv_all[daily_rv_all > 0]

    # Daily log-returns from close prices
    daily_log_ret = np.log(close_daily).diff().dropna()

    # Align dates
    common_daily = daily_rv_all.index.intersection(daily_log_ret.index)
    daily_rv_aligned = daily_rv_all.loc[common_daily].values
    daily_ret_aligned = daily_log_ret.loc[common_daily].values

    # Daily log spot volatility
    daily_log_spot = np.log(daily_rv_aligned)
    daily_log_spot = daily_log_spot[np.isfinite(daily_log_spot)]

    print(f"  Daily observations: {len(daily_rv_aligned)}")
    print(f"  Daily log-RV: mean={daily_log_spot.mean():.4f}, std={daily_log_spot.std():.4f}")

    # Train/test split (80/20)
    n_daily = len(daily_log_spot)
    split_d = int(n_daily * 0.8)
    train_daily_log = daily_log_spot[:split_d]
    test_daily_log = daily_log_spot[split_d:]

    print(f"  Train: {split_d}, Test: {n_daily - split_d}")

    # Estimate alpha for daily series
    daily_alpha = estimate_alpha(train_daily_log)
    print(f"  Daily alpha (ACF): {daily_alpha['alpha_acf']:.4f}")

    # Gibbs sampler on daily log-RV
    print("\n  Running Gibbs sampler on daily log-RV...")
    rng_gibbs_daily = np.random.default_rng(777)
    gibbs_daily = gibbs_gig_harris(train_daily_log, alpha_init=daily_alpha["alpha_acf"],
                                     epsilon=0.1, n_iter=5000, burn_in=2000, rng=rng_gibbs_daily)

    # Simulate daily volatility trajectories
    n_sim_daily = 2000
    rng_sim_daily = np.random.default_rng(888)
    sim_daily_log = simulate_predictive_sf_harris(
        train_daily_log, test_daily_log, gibbs_daily, gibbs_daily["alpha"],
        Q_type="empirical", n_sim=n_sim_daily, rng=rng_sim_daily
    )
    print(f"  Simulated {n_sim_daily} daily volatility trajectories of {len(test_daily_log)} days each")

    # Coverage of daily log-RV (volatility prediction)
    cov_daily_rv = compute_coverage(test_daily_log, sim_daily_log, PROB_LEVELS)
    aad_daily_rv = np.mean([abs(cov_daily_rv[p] - p*100) for p in PROB_LEVELS])

    print(f"\n  Daily log-RV coverage (direct daily SF-Harris):")
    print(f"  {'p':>6}  {'Coverage':>10}  {'Ideal':>6}  {'Dev':>6}")
    print(f"  {'-'*34}")
    for p in PROB_LEVELS:
        dev = cov_daily_rv[p] - p*100
        print(f"  {p:>6.2f}  {cov_daily_rv[p]:>9.1f}%  {p*100:>5.0f}%  {dev:>+5.1f}pp")
    print(f"  AAD = {aad_daily_rv:.1f}pp")

    # Estimate emission function parameters on daily data
    train_daily_rv = daily_rv_aligned[:split_d]
    train_daily_ret = daily_ret_aligned[:split_d]

    # Need to align — training period only
    train_dates_aligned = common_daily[:split_d]
    mu_beta_daily = estimate_mu_beta(
        pd.Series(train_daily_ret, index=train_dates_aligned),
        pd.Series(train_daily_rv, index=train_dates_aligned)
    )

    print(f"\n  Daily emission function parameters:")
    print(f"  mu:    {mu_beta_daily['mu_hat']:.6f} (posterior: {mu_beta_daily['mu_post_mean']:.6f})")
    print(f"  beta:  {mu_beta_daily['beta_hat']:.6f} (posterior: {mu_beta_daily['beta_post_mean']:.6f})")

    # Simulate daily returns from daily volatility trajectories
    # R | tau* ~ N(mu + beta*tau*, tau*)
    # where tau* = exp(log_spot) for each daily observation
    sim_daily_rv = np.exp(sim_daily_log)  # (n_sim, n_test) simulated daily RV
    mu_d = mu_beta_daily["mu_post_mean"]
    beta_d = mu_beta_daily["beta_post_mean"]

    mean_daily_ret = mu_d + beta_d * sim_daily_rv
    std_daily_ret = np.sqrt(np.maximum(sim_daily_rv, 1e-20))
    rng_emit_daily = np.random.default_rng(999)
    sim_daily_returns = mean_daily_ret + std_daily_ret * rng_emit_daily.normal(size=sim_daily_rv.shape)

    # Actual test returns
    test_daily_ret = daily_ret_aligned[split_d:n_daily]
    # Align: make sure test_daily_ret length matches sim_daily_returns columns
    n_test_daily = min(len(test_daily_ret), sim_daily_returns.shape[1])

    cov_daily_ret = compute_coverage(test_daily_ret[:n_test_daily],
                                      sim_daily_returns[:, :n_test_daily], PROB_LEVELS)
    aad_daily_ret = np.mean([abs(cov_daily_ret[p] - p*100) for p in PROB_LEVELS])

    print(f"\n  Daily return coverage (direct daily model):")
    print(f"  {'p':>6}  {'Coverage':>10}  {'Ideal':>6}  {'Dev':>6}")
    print(f"  {'-'*34}")
    for p in PROB_LEVELS:
        dev = cov_daily_ret[p] - p*100
        print(f"  {p:>6.2f}  {cov_daily_ret[p]:>9.1f}%  {p*100:>5.0f}%  {dev:>+5.1f}pp")
    print(f"  AAD = {aad_daily_ret:.1f}pp")

    # ==================================================================
    # STEP 10: Also try with eps=1e-5 for daily
    # ==================================================================
    print("\n--- Daily SF-Harris with eps=1e-5 ---")
    gibbs_daily2 = gibbs_gig_harris(train_daily_log, alpha_init=daily_alpha["alpha_acf"],
                                      epsilon=1e-5, n_iter=5000, burn_in=2000,
                                      rng=np.random.default_rng(111))
    sim_daily_log2 = simulate_predictive_sf_harris(
        train_daily_log, test_daily_log, gibbs_daily2, gibbs_daily2["alpha"],
        Q_type="empirical", n_sim=n_sim_daily, rng=np.random.default_rng(222)
    )
    cov_daily_rv2 = compute_coverage(test_daily_log, sim_daily_log2, PROB_LEVELS)
    aad_daily_rv2 = np.mean([abs(cov_daily_rv2[p] - p*100) for p in PROB_LEVELS])

    sim_daily_rv2 = np.exp(sim_daily_log2)
    mean_daily_ret2 = mu_d + beta_d * sim_daily_rv2
    std_daily_ret2 = np.sqrt(np.maximum(sim_daily_rv2, 1e-20))
    sim_daily_returns2 = mean_daily_ret2 + std_daily_ret2 * np.random.default_rng(333).normal(size=sim_daily_rv2.shape)

    cov_daily_ret2 = compute_coverage(test_daily_ret[:n_test_daily],
                                       sim_daily_returns2[:, :n_test_daily], PROB_LEVELS)
    aad_daily_ret2 = np.mean([abs(cov_daily_ret2[p] - p*100) for p in PROB_LEVELS])

    print(f"  Daily log-RV coverage (eps=1e-5): AAD = {aad_daily_rv2:.1f}pp")
    print(f"  Daily return coverage (eps=1e-5): AAD = {aad_daily_ret2:.1f}pp")

    # ==================================================================
    # STEP 11: Daily benchmarks (GARCH, EGARCH, Heston)
    # ==================================================================
    # Compare our daily SF-Harris + Emission (2.5pp) against standard
    # SV models at the daily level, using the same train/test split.
    # ==================================================================
    print("\n" + "=" * 70)
    print("DAILY BENCHMARKS: GARCH, EGARCH, Heston")
    print("Same 80/20 split as direct daily SF-Harris model")
    print("=" * 70)

    # Use daily log-returns (same data as SF-Harris daily)
    train_daily_ret_data = daily_ret_aligned[:split_d]
    test_daily_ret_data = daily_ret_aligned[split_d:n_daily]
    n_test_daily_ret = len(test_daily_ret_data)

    # --- GARCH(1,1) daily ---
    print("\n  --- GARCH(1,1) daily ---")
    res_garch_daily = fit_garch(train_daily_ret_data, p=1, q=1, dist="normal")
    print(f"  omega={res_garch_daily.params['omega']:.6f}, "
          f"alpha={res_garch_daily.params['alpha[1]']:.6f}, "
          f"beta={res_garch_daily.params['beta[1]']:.6f}")
    print(f"  Persistence: {res_garch_daily.params['alpha[1]'] + res_garch_daily.params['beta[1]']:.4f}")

    rng_garch_daily = np.random.default_rng(101)
    _, sim_garch_daily_ret = simulate_garch(res_garch_daily, n_test_daily_ret, n_sim_daily,
                                             rng=rng_garch_daily)
    # Align length
    min_len_g = min(n_test_daily_ret, sim_garch_daily_ret.shape[1])
    cov_garch_daily_ret = compute_coverage(test_daily_ret_data[:min_len_g],
                                             sim_garch_daily_ret[:, :min_len_g], PROB_LEVELS)
    aad_garch_daily_ret = np.mean([abs(cov_garch_daily_ret[p] - p*100) for p in PROB_LEVELS])

    # Also compute volatility coverage for GARCH
    sim_garch_daily_var = np.log(np.maximum(sim_garch_daily_ret[:, :min_len_g]**2, 1e-20))
    test_daily_log_spot_g = test_daily_log[:min_len_g]
    cov_garch_daily_rv = compute_coverage(test_daily_log_spot_g, sim_garch_daily_var, PROB_LEVELS)
    aad_garch_daily_rv = np.mean([abs(cov_garch_daily_rv[p] - p*100) for p in PROB_LEVELS])

    print(f"\n  GARCH(1,1) daily return coverage:")
    print(f"  {'p':>6}  {'Coverage':>10}  {'Ideal':>6}  {'Dev':>6}")
    print(f"  {'-'*34}")
    for p in PROB_LEVELS:
        dev = cov_garch_daily_ret[p] - p*100
        print(f"  {p:>6.2f}  {cov_garch_daily_ret[p]:>9.1f}%  {p*100:>5.0f}%  {dev:>+5.1f}pp")
    print(f"  AAD (return) = {aad_garch_daily_ret:.1f}pp, AAD (vol) = {aad_garch_daily_rv:.1f}pp")

    # --- EGARCH(1,1) daily ---
    print("\n  --- EGARCH(1,1) daily ---")
    res_egarch_daily = fit_egarch(train_daily_ret_data, p=1, q=1, dist="normal")
    print(f"  omega={res_egarch_daily.params['omega']:.6f}, "
          f"alpha={res_egarch_daily.params['alpha[1]']:.6f}, "
          f"beta={res_egarch_daily.params['beta[1]']:.6f}")

    rng_egarch_daily = np.random.default_rng(202)
    _, sim_egarch_daily_ret = simulate_egarch(res_egarch_daily, n_test_daily_ret, n_sim_daily,
                                                rng=rng_egarch_daily)
    min_len_e = min(n_test_daily_ret, sim_egarch_daily_ret.shape[1])
    cov_egarch_daily_ret = compute_coverage(test_daily_ret_data[:min_len_e],
                                             sim_egarch_daily_ret[:, :min_len_e], PROB_LEVELS)
    aad_egarch_daily_ret = np.mean([abs(cov_egarch_daily_ret[p] - p*100) for p in PROB_LEVELS])

    sim_egarch_daily_var = np.log(np.maximum(sim_egarch_daily_ret[:, :min_len_e]**2, 1e-20))
    test_daily_log_spot_e = test_daily_log[:min_len_e]
    cov_egarch_daily_rv = compute_coverage(test_daily_log_spot_e, sim_egarch_daily_var, PROB_LEVELS)
    aad_egarch_daily_rv = np.mean([abs(cov_egarch_daily_rv[p] - p*100) for p in PROB_LEVELS])

    print(f"\n  EGARCH(1,1) daily return coverage:")
    print(f"  {'p':>6}  {'Coverage':>10}  {'Ideal':>6}  {'Dev':>6}")
    print(f"  {'-'*34}")
    for p in PROB_LEVELS:
        dev = cov_egarch_daily_ret[p] - p*100
        print(f"  {p:>6.2f}  {cov_egarch_daily_ret[p]:>9.1f}%  {p*100:>5.0f}%  {dev:>+5.1f}pp")
    print(f"  AAD (return) = {aad_egarch_daily_ret:.1f}pp, AAD (vol) = {aad_egarch_daily_rv:.1f}pp")

    # --- Heston daily ---
    print("\n  --- Heston daily ---")
    heston_daily = fit_heston(train_daily_ret_data, dt=1.0)
    if heston_daily is not None:
        print(f"  kappa={heston_daily['kappa']:.4f}, "
              f"theta={heston_daily['theta']:.6f}, "
              f"sigma_v={heston_daily['sigma_v']:.4f}, "
              f"rho={heston_daily['rho']:.4f}")

        V0_daily = np.var(train_daily_ret_data)
        rng_heston_daily = np.random.default_rng(303)
        _, sim_heston_daily_ret = simulate_heston(heston_daily, V0_daily, n_test_daily_ret,
                                                    dt=1.0, n_sim=n_sim_daily, rng=rng_heston_daily)
        min_len_h = min(n_test_daily_ret, sim_heston_daily_ret.shape[1])
        cov_heston_daily_ret = compute_coverage(test_daily_ret_data[:min_len_h],
                                                  sim_heston_daily_ret[:, :min_len_h], PROB_LEVELS)
        aad_heston_daily_ret = np.mean([abs(cov_heston_daily_ret[p] - p*100) for p in PROB_LEVELS])

        sim_heston_daily_var = np.log(np.maximum(sim_heston_daily_ret[:, :min_len_h]**2, 1e-20))
        test_daily_log_spot_h = test_daily_log[:min_len_h]
        cov_heston_daily_rv = compute_coverage(test_daily_log_spot_h, sim_heston_daily_var, PROB_LEVELS)
        aad_heston_daily_rv = np.mean([abs(cov_heston_daily_rv[p] - p*100) for p in PROB_LEVELS])

        print(f"\n  Heston daily return coverage:")
        print(f"  {'p':>6}  {'Coverage':>10}  {'Ideal':>6}  {'Dev':>6}")
        print(f"  {'-'*34}")
        for p in PROB_LEVELS:
            dev = cov_heston_daily_ret[p] - p*100
            print(f"  {p:>6.2f}  {cov_heston_daily_ret[p]:>9.1f}%  {p*100:>5.0f}%  {dev:>+5.1f}pp")
        print(f"  AAD (return) = {aad_heston_daily_ret:.1f}pp, AAD (vol) = {aad_heston_daily_rv:.1f}pp")
    else:
        print("  Heston fit failed for daily data")
        aad_heston_daily_ret = None
        aad_heston_daily_rv = None

    # ==================================================================
    # FINAL COMPARISON
    # ==================================================================
    sep = "=" * 70
    print(f"\n{sep}")
    print("FINAL COMPARISON - DAILY RETURN PREDICTION")
    print(sep)
    print("  Our model: Daily SF-Harris + Emission function")
    print("  Y_t | tau* ~ N(mu + beta*tau*, tau*)")
    print("  Benchmarks: GARCH(1,1), EGARCH(1,1), Heston (all daily)")
    print(f"\n  {'Model':>60}  {'Quantity':>10}  {'Scale':>6}  {'AAD':>6}")
    print(f"  {'-'*88}")

    # Our models
    print(f"  {'--- Our models:':>60}")
    print(f"  {'Daily SF-Harris + Emission (eps=1e-5)':>60}  {'return':>10}  {'daily':>6}  {aad_daily_ret2:>5.1f}pp")
    print(f"  {'Daily SF-Harris + Emission (eps=0.1)':>60}  {'return':>10}  {'daily':>6}  {aad_daily_ret:>5.1f}pp")
    print(f"  {'Daily SF-Harris (eps=1e-5)':>60}  {'daily RV':>10}  {'daily':>6}  {aad_daily_rv2:>5.1f}pp")
    print(f"  {'Daily SF-Harris (eps=0.1)':>60}  {'daily RV':>10}  {'daily':>6}  {aad_daily_rv:>5.1f}pp")

    # Daily benchmarks - returns
    print("\n  --- Daily benchmarks (return coverage):")
    print(f"  {'GARCH(1,1) daily':>60}  {'return':>10}  {'daily':>6}  {aad_garch_daily_ret:>5.1f}pp")
    print(f"  {'EGARCH(1,1) daily':>60}  {'return':>10}  {'daily':>6}  {aad_egarch_daily_ret:>5.1f}pp")
    if aad_heston_daily_ret is not None:
        print(f"  {'Heston daily':>60}  {'return':>10}  {'daily':>6}  {aad_heston_daily_ret:>5.1f}pp")

    # Daily benchmarks - volatility
    print("\n  --- Daily benchmarks (volatility coverage):")
    print(f"  {'GARCH(1,1) daily':>60}  {'daily RV':>10}  {'daily':>6}  {aad_garch_daily_rv:>5.1f}pp")
    print(f"  {'EGARCH(1,1) daily':>60}  {'daily RV':>10}  {'daily':>6}  {aad_egarch_daily_rv:>5.1f}pp")
    if aad_heston_daily_rv is not None:
        print(f"  {'Heston daily':>60}  {'daily RV':>10}  {'daily':>6}  {aad_heston_daily_rv:>5.1f}pp")

    # Intraday reference
    print("\n  --- Intraday reference (Table 3):")
    print(f"  {'SF-Harris Gibbs (eps=0.1)':>60}  {'volatility':>10}  {'15min':>6}  {'0.3pp':>6}")
    print(f"  {'SF-Harris Gibbs (eps=0.1)':>60}  {'volatility':>10}  {'dollar':>6}  {'0.2pp':>6}")
    print(f"  {'GARCH(1,1) intraday':>60}  {'volatility':>10}  {'15min':>6}  {'3.4pp':>6}")
    print(f"  {'Anzarut (GIG+Gibbs)':>60}  {'volatility':>10}  {'15min':>6}  {'0.8pp':>6}")

    # Emission function parameters
    print(f"\n  Emission function: Y_t = mu*dt + beta*tau*_t + B_{{tau*_t}}")
    print(f"  Daily model: mu={mu_beta_daily['mu_post_mean']:.6f}, beta={mu_beta_daily['beta_post_mean']:.6f}")

    # Save results
    results = {
        "model": [
            "Daily_SF-Harris_Emission_eps1e-5",
            "Daily_SF-Harris_Emission_eps0.1",
            "Daily_SF-Harris_RV_eps1e-5",
            "Daily_SF-Harris_RV_eps0.1",
            "GARCH_daily_return",
            "EGARCH_daily_return",
            "GARCH_daily_RV",
            "EGARCH_daily_RV",
        ],
        "quantity": [
            "return", "return", "daily_RV", "daily_RV",
            "return", "return", "daily_RV", "daily_RV",
        ],
        "aad": [
            aad_daily_ret2, aad_daily_ret, aad_daily_rv2, aad_daily_rv,
            aad_garch_daily_ret, aad_egarch_daily_ret,
            aad_garch_daily_rv, aad_egarch_daily_rv,
        ],
    }
    if aad_heston_daily_ret is not None:
        results["model"].extend(["Heston_daily_return", "Heston_daily_RV"])
        results["quantity"].extend(["return", "daily_RV"])
        results["aad"].extend([aad_heston_daily_ret, aad_heston_daily_rv])

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_DIR / "daily_prediction_results.csv", index=False)
    print(f"\n  Results saved to {OUTPUT_DIR / 'daily_prediction_results.csv'}")

