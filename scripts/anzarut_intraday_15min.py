"""Anzarut intraday replication: 15-min pipeline with epsilon threshold.

Key insight: At 15-min intraday level, consecutive spot vol values are
similar enough that epsilon threshold identifies proper stay/jump structure.
At daily level, everything changes between observations, so epsilon gives
0% stays. At 15-min level, we expect many "stays" (piecewise constant paths).

Pipeline:
1. Load IBM 1-minute data -> 15-min returns
2. Jump detection via bipower variation
3. Periodicity adjustment (divide out intraday U-shape)
4. Compute 15-min spot volatility (raw and log)
5. Test epsilon thresholds at 15-min level (both scales)
6. Gibbs sampler on 15-min series with epsilon
7. Simulate 15-min trajectories from posterior predictive
8. Aggregate to daily for Table 3 coverage validation
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
    fit_jump_thresholds,
    split_returns_by_date,
    estimate_periodicity,
    estimate_alpha,
    gibbs_gig_harris,
    simulate_predictive_sf_harris,
    compute_coverage,
)

DATA_PATH = Path(r"C:\Users\angve\OneDrive\Desktop\Servicio\Libros\SF-Harris\IBM.txt")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"

ANZARUT_TABLE3 = {0.25: 25, 0.50: 51, 0.75: 75, 0.85: 84, 0.90: 89, 0.95: 93}
PROB_LEVELS = [0.25, 0.50, 0.75, 0.85, 0.90, 0.95]


def compute_15min_spot_volatility(returns_clean, periodicity):
    """Compute 15-min spot volatility (raw and log, periodicity-adjusted).

    Spot volatility: H_{t,i} = r_{t,i}^2 / f(t_i)
    Raw: rv_adj = r^2 / f(t)
    Log: log_spot = log(rv_adj)
    """
    rv_df = returns_clean.to_frame("return")
    rv_df["rv_15min"] = rv_df["return"]**2
    rv_df["time"] = rv_df.index.strftime("%H:%M")
    rv_df["f_t"] = rv_df["time"].map(periodicity)
    rv_df["f_t"] = rv_df["f_t"].fillna(1.0)
    rv_df["rv_adj"] = rv_df["rv_15min"] / rv_df["f_t"]
    rv_df["log_spot"] = np.log(rv_df["rv_adj"].clip(lower=1e-20))

    # Clean: remove NaN/inf from log
    mask = np.isfinite(rv_df["log_spot"])
    log_spot = rv_df.loc[mask, "log_spot"]

    return log_spot, rv_df


def aggregate_to_daily(sim_15, test_idx, rv_df, periodicity, n_sim, prob_levels):
    """Aggregate 15-min simulated trajectories to daily level.

    For adjusted: daily RV = sum(exp(sim_log_spot) * f_t) per day
    For unadjusted: daily RV = sum(exp(sim_log_spot)) per day

    Compare against observed daily log-RV for coverage.
    """
    # Build day assignments for each test 15-min interval
    test_dates = [d.date() if hasattr(d, 'date') else pd.Timestamp(d).date()
                  for d in test_idx]
    unique_days = sorted(set(test_dates))
    n_days = len(unique_days)
    day_map = {d: i for i, d in enumerate(unique_days)}
    day_assign = np.array([day_map[d] for d in test_dates])

    # Periodicity factors for each test 15-min interval
    time_strings = [d.strftime("%H:%M") for d in test_idx]
    f_t = np.array([periodicity.get(t, 1.0) for t in time_strings])

    n_test = min(sim_15.shape[1], len(day_assign))

    # Day assignment matrix: (n_test, n_days), one-hot
    day_matrix = np.zeros((n_test, n_days))
    for i in range(n_test):
        day_matrix[i, day_assign[i]] = 1.0

    # Number of intervals per day (for coverage alignment)
    intervals_per_day = day_matrix.sum(axis=0)

    # Actual daily log-RV from 15-min data
    test_rv_df = rv_df.loc[test_idx[0]:].copy()
    test_rv_df["date"] = test_rv_df.index.date

    daily_rv_adj_actual = test_rv_df.groupby("date")["rv_adj"].sum()
    daily_rv_unadj_actual = test_rv_df.groupby("date")["rv_15min"].sum()

    log_rv_adj_daily = np.log(daily_rv_adj_actual.values)
    log_rv_unadj_daily = np.log(daily_rv_unadj_actual.values)

    valid_adj = np.isfinite(log_rv_adj_daily)
    valid_unadj = np.isfinite(log_rv_unadj_daily)

    log_rv_adj_daily = log_rv_adj_daily[valid_adj]
    log_rv_unadj_daily = log_rv_unadj_daily[valid_unadj]

    # Vectorized aggregation: (n_sim, n_test) -> (n_sim, n_days)
    exp_sim = np.exp(sim_15[:, :n_test])  # (n_sim, n_test)
    weighted = exp_sim * f_t[np.newaxis, :n_test]  # (n_sim, n_test)

    daily_sum_adj = weighted @ day_matrix   # (n_sim, n_days)
    daily_sum_unadj = exp_sim @ day_matrix  # (n_sim, n_days)

    sim_daily_adj = np.log(np.maximum(daily_sum_adj, 1e-20))
    sim_daily_unadj = np.log(np.maximum(daily_sum_unadj, 1e-20))

    # Coverage
    n_valid = min(len(log_rv_adj_daily), n_days, len(log_rv_unadj_daily))

    cov_adj = compute_coverage(log_rv_adj_daily[:n_valid], sim_daily_adj[:, :n_valid], prob_levels)
    cov_unadj = compute_coverage(log_rv_unadj_daily[:n_valid], sim_daily_unadj[:, :n_valid], prob_levels)

    aad_adj = np.mean([abs(cov_adj[p] - p*100) for p in prob_levels])
    aad_unadj = np.mean([abs(cov_unadj[p] - p*100) for p in prob_levels])

    return cov_adj, aad_adj, cov_unadj, aad_unadj, n_valid


def gibbs_raw_scale(raw_data, epsilon=1e-5, n_iter=5000, burn_in=2000, rng=None):
    """Gibbs sampler for SF-Harris on RAW (not log) scale with epsilon threshold.

    At 15-min level, raw spot volatility values are ~1e-6.
    epsilon = 1e-5 on raw scale means consecutive values within 1e-5
    are classified as "stays" — this should capture the piecewise constant
    structure of intraday volatility.

    Posterior:
      z_i | alpha: Bernoulli(1 - exp(-alpha)) with epsilon threshold
      alpha | z: Gamma conjugate update
      mu, sigma: Normal-Inverse-Gamma on log-space (converted back)
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(raw_data)
    x = raw_data.copy()

    # Work in log space for mu/sigma estimation (raw values are positive but skewed)
    log_x = np.log(x[x > 0])
    mu_init = np.mean(log_x)
    sigma_init = np.std(log_x)

    # Initial alpha from ACF of log data
    log_series = np.log(x)
    mu_log = np.mean(log_series)
    var_log = np.var(log_series)
    if var_log > 1e-10:
        rho1 = np.mean((log_series[:-1] - mu_log) * (log_series[1:] - mu_log)) / var_log
        alpha_init = -np.log(max(rho1, 0.01))
    else:
        alpha_init = 1.0

    # Initialize z using epsilon threshold on RAW scale
    diffs = np.abs(np.diff(x))
    z = (diffs >= epsilon).astype(int)
    n_stays = np.sum(z == 0)
    n_jumps = np.sum(z == 1)
    print(f"  epsilon (raw scale) = {epsilon:.0e}: stays={n_stays}/{len(diffs)} "
          f"({n_stays/len(diffs)*100:.1f}%), jumps={n_jumps}/{len(diffs)} "
          f"({n_jumps/len(diffs)*100:.1f}%)")

    alpha = alpha_init
    mu = mu_init
    sigma = sigma_init

    # Prior hyperparameters
    c_alpha = 0.01
    mu0 = mu_init
    kappa0_mu = 0.01
    a0 = 3.0
    b0 = sigma_init**2

    # Storage
    samples = {
        "alpha": np.empty(n_iter),
        "mu": np.empty(n_iter),
        "sigma": np.empty(n_iter),
        "n_jumps": np.empty(n_iter),
    }

    for it in range(n_iter):
        # Step 1: Update z_i using epsilon threshold on RAW scale
        diffs = np.abs(np.diff(x))
        z = (diffs >= epsilon).astype(int)

        # Step 2: Update alpha (conjugate Gamma)
        m = np.sum(z)
        jump_indices = np.where(z == 1)[0]
        if len(jump_indices) > 0:
            inter_arrivals = np.diff(np.concatenate([[-1], jump_indices]))
            total_inter_arrival = np.sum(inter_arrivals)
        else:
            total_inter_arrival = n
        alpha = rng.gamma(m + 1, 1.0 / (total_inter_arrival + c_alpha))

        # Step 3: Update mu, sigma (Normal-Inverse-Gamma on log space)
        log_vals = np.log(x[x > 0])
        kappa_n = kappa0_mu + len(log_vals)
        mu_n = (kappa0_mu * mu0 + len(log_vals) * np.mean(log_vals)) / kappa_n
        mu = rng.normal(mu_n, sigma / np.sqrt(kappa_n))

        ss = np.sum((log_vals - mu)**2)
        a_n = a0 + len(log_vals) / 2
        b_n = b0 + ss / 2
        sigma2 = 1.0 / rng.gamma(a_n, 1.0 / b_n)
        sigma = np.sqrt(max(sigma2, 1e-6))

        samples["alpha"][it] = alpha
        samples["mu"][it] = mu
        samples["sigma"][it] = sigma
        samples["n_jumps"][it] = m

    result = {k: v[burn_in:] for k, v in samples.items()}

    p_stay_mean = np.exp(-np.mean(result["alpha"]))
    print(f"  Posterior means (raw scale Gibbs):")
    print(f"    alpha:    {np.mean(result['alpha']):.3f} (std={np.std(result['alpha']):.3f})")
    print(f"    mu (log): {np.mean(result['mu']):.4f} (std={np.std(result['mu']):.4f})")
    print(f"    sigma:    {np.mean(result['sigma']):.4f} (std={np.std(result['sigma']):.4f})")
    print(f"    P(stay):  {p_stay_mean:.4f}")
    print(f"    n_jumps:  {np.mean(result['n_jumps']):.1f}/{n-1} (mean)")

    return result


def simulate_raw_sf_harris(raw_train, raw_test, gibbs_post, epsilon=1e-5,
                             n_sim=2000, rng=None):
    """Simulate SF-Harris on RAW scale with epsilon threshold.

    Transition kernel:
      P(x_{t+1} = x_t) = exp(-alpha)  (stay)
      P(x_{t+1} ~ Q) = 1 - exp(-alpha)  (jump)

    Q is empirical: resample from centered training data.
    Stay means x_{t+1} = x_t exactly (within epsilon on raw scale).
    """
    if rng is None:
        rng = np.random.default_rng()

    n_test = len(raw_test)
    n_post = len(gibbs_post["alpha"])

    # Empirical Q: resample from training data (centered)
    emp_data = raw_train - np.mean(raw_train)

    simulated = np.empty((n_sim, n_test))
    last_val = raw_train[-1]

    for s in range(n_sim):
        idx = rng.integers(0, n_post)
        alpha = gibbs_post["alpha"][idx]
        mu = gibbs_post["mu"][idx]  # in log space
        sigma = gibbs_post["sigma"][idx]

        p_stay = np.exp(-alpha)
        current = last_val

        for i in range(n_test):
            if rng.random() < p_stay:
                pass  # stay at current value
            else:
                # Jump: draw from empirical Q
                # Resample from centered training data, add mu offset (converted to raw)
                jump_val = np.exp(mu) + rng.choice(emp_data)
                current = max(jump_val, 1e-20)  # ensure positive

            simulated[s, i] = current

    return simulated


if __name__ == "__main__":
    print("=" * 70)
    print("ANZARUT INTRADAY REPLICATION: 15-min Pipeline")
    print("SF-Harris + Gibbs + epsilon at 15-min level")
    print("=" * 70)

    # ==================================================================
    # STEP 1: Load and preprocess data
    # ==================================================================
    print("\n--- Step 1: Load and preprocess data ---")
    df = load_ibm_data()
    print(f"  Loaded {len(df)} minute bars")

    returns = compute_15min_returns(df)
    print(f"  {len(returns)} 15-min return observations")

    # --- leak-free split: cut by date BEFORE cleaning; fit jump thresholds on TRAIN, apply to both ---
    train_returns, test_returns = split_returns_by_date(returns, train_frac=0.8)
    thresholds = fit_jump_thresholds(train_returns, n_passes=2, top_pct=0.001)
    train_clean = detect_and_remove_jumps(train_returns, fixed_thresholds=thresholds)
    test_clean = detect_and_remove_jumps(test_returns, fixed_thresholds=thresholds)
    print(f"  Clean returns: train={len(train_clean)}, test={len(test_clean)}")

    # ==================================================================
    # STEP 2: Periodicity adjustment
    # ==================================================================
    print("\n--- Step 2: Periodicity adjustment ---")
    # Estimate on TRAIN clean returns (leak-free; also fixes the raw-not-clean inconsistency)
    periodicity = estimate_periodicity(train_clean)
    print(f"  Periodicity range: {periodicity.min():.2f} to {periodicity.max():.2f}")

    # ==================================================================
    # STEP 3: Compute 15-min spot volatility (per window, TRAIN-fitted periodicity)
    # ==================================================================
    print("\n--- Step 3: Compute 15-min spot volatility ---")
    log_spot_train, rv_df_train = compute_15min_spot_volatility(train_clean, periodicity)
    log_spot_test, rv_df_test = compute_15min_spot_volatility(test_clean, periodicity)

    train_log = log_spot_train.values
    test_log = log_spot_test.values
    train_raw = np.exp(train_log)
    test_raw = np.exp(test_log)
    train_idx = log_spot_train.index
    test_idx = log_spot_test.index

    print(f"  15-min log spot vol: train={len(train_log)}, test={len(test_log)}")
    print(f"  Train log:  mean={train_log.mean():.4f}, std={train_log.std():.4f}, "
          f"skew={stats.skew(train_log):.2f}, kurt={stats.kurtosis(train_log):.2f}")
    print(f"  Test  log:  mean={test_log.mean():.4f}, std={test_log.std():.4f}, "
          f"skew={stats.skew(test_log):.2f}, kurt={stats.kurtosis(test_log):.2f}")

    # ==================================================================
    # STEP 4: Test epsilon thresholds at 15-min level (train window)
    # ==================================================================
    print("\n--- Step 4: Testing epsilon thresholds (train window) ---")

    # Log scale
    diffs_log = np.abs(np.diff(train_log))
    print(f"\n  LOG scale (log spot volatility):")
    print(f"    Typical diff: mean={diffs_log.mean():.4f}, median={np.median(diffs_log):.4f}")
    print(f"    Diff range: [{diffs_log.min():.6f}, {diffs_log.max():.4f}]")

    for eps_label, eps_val in [("1e-5 (Anzarut)", 1e-5), ("1e-4", 1e-4),
                                 ("1e-3", 1e-3), ("1e-2", 0.01), ("0.1", 0.1)]:
        n_stay = np.sum(diffs_log < eps_val)
        n_total = len(diffs_log)
        print(f"    eps={eps_label}: stays={n_stay}/{n_total} ({n_stay/n_total*100:.1f}%)")

    # Raw scale
    diffs_raw = np.abs(np.diff(train_raw))
    print(f"\n  RAW scale (spot volatility):")
    print(f"    Typical diff: mean={diffs_raw.mean():.2e}, median={np.median(diffs_raw):.2e}")
    print(f"    Diff range: [{diffs_raw.min():.2e}, {diffs_raw.max():.2e}]")

    for eps_label, eps_val in [("1e-5 (Anzarut)", 1e-5), ("1e-6", 1e-6),
                                 ("1e-7", 1e-7), ("1e-8", 1e-8), ("1e-10", 1e-10)]:
        n_stay = np.sum(diffs_raw < eps_val)
        n_total = len(diffs_raw)
        print(f"    eps={eps_label}: stays={n_stay}/{n_total} ({n_stay/n_total*100:.1f}%)")

    # ==================================================================
    # STEP 5: Train/test split (already done leak-free above)
    # ==================================================================
    print("\n--- Step 5: Train/test split (by date, before cleaning) ---")
    n_15 = len(train_log) + len(test_log)
    print(f"  Total: {n_15}, Train: {len(train_log)}, Test: {len(test_log)}")
    print(f"  Train log: mean={train_log.mean():.4f}, std={train_log.std():.4f}")
    print(f"  Test  log: mean={test_log.mean():.4f}, std={test_log.std():.4f}")

    # ==================================================================
    # STEP 5b: Dollar bar comparison (no periodicity needed)
    # ==================================================================
    print("\n" + "=" * 70)
    print("STEP 5b: Dollar bar pipeline (no periodicity adjustment)")
    print("=" * 70)

    # Build dollar bars from minute data
    avg_dv_per_15min = df["dollar_volume"].resample("15min").sum().mean()
    dollar_threshold = avg_dv_per_15min

    cum_dv = 0.0
    first_close = None
    dol_returns = []
    for idx, row in df.iterrows():
        if first_close is None:
            first_close = row["close"]
        cum_dv += row["dollar_volume"]
        if cum_dv >= dollar_threshold:
            ret = np.log(row["close"] / first_close)
            dol_returns.append({"datetime": idx, "return": ret, "dollar_volume": cum_dv})
            cum_dv = 0.0
            first_close = None

    dol_df = pd.DataFrame(dol_returns).set_index("datetime") if dol_returns else pd.DataFrame()
    dol_returns_series = dol_df["return"]
    print(f"  Dollar bars: {len(dol_returns_series)} observations (threshold=${dollar_threshold:,.0f})")
    print(f"  Avg bars/day: {len(dol_returns_series) / len(dol_returns_series.index.to_series().dt.date.unique()):.1f}")

    # Dollar bar spot volatility (no periodicity adjustment needed)
    dol_rv = dol_returns_series**2  # squared returns = spot variance proxy
    dol_log_spot = np.log(dol_rv.clip(lower=1e-20))
    dol_log_spot_series = dol_log_spot.replace([np.inf, -np.inf], np.nan).dropna()

    # --- leak-free: split dollar bars at the SAME temporal boundary; fit outlier bounds on TRAIN, drop from both ---
    dol_cutoff = pd.Timestamp(train_returns.index.date.max())  # last train day (inclusive)
    dol_train_full = dol_log_spot_series[dol_log_spot_series.index <= dol_cutoff].values
    dol_test_full = dol_log_spot_series[dol_log_spot_series.index > dol_cutoff].values

    # Outlier bounds fit on TRAIN only; drop outliers from BOTH windows (leak-free cleaning,
    # same drop semantics as the original — train/test are independent arrays so dropping is safe)
    dol_q01 = np.percentile(dol_train_full, 0.5)
    dol_q99 = np.percentile(dol_train_full, 99.5)
    dol_train = dol_train_full[(dol_train_full > dol_q01) & (dol_train_full < dol_q99)]
    dol_test = dol_test_full[(dol_test_full > dol_q01) & (dol_test_full < dol_q99)]
    dol_log_spot_clean = dol_train  # for the diagnostics print below

    print(f"  After outlier removal (bounds fit on train): train={len(dol_train)}, test={len(dol_test)}")
    print(f"  Dollar bar log spot (train): mean={dol_train.mean():.4f}, "
          f"std={dol_train.std():.4f}, "
          f"skew={stats.skew(dol_train):.2f}, "
          f"kurt={stats.kurtosis(dol_train):.2f}")

    print(f"  Dollar bar train: {len(dol_train)}, test: {len(dol_test)}")

    # Epsilon thresholds for dollar bars
    dol_diffs = np.abs(np.diff(dol_train))
    print(f"\n  Dollar bar epsilon test (log scale):")
    for eps_label, eps_val in [("1e-5", 1e-5), ("1e-4", 1e-4), ("1e-3", 1e-3),
                                 ("1e-2", 0.01), ("0.1", 0.1)]:
        n_stay = np.sum(dol_diffs < eps_val)
        print(f"    eps={eps_label}: stays={n_stay}/{len(dol_diffs)} ({n_stay/len(dol_diffs)*100:.1f}%)")

    # Gibbs sampler on dollar bars
    print(f"\n  Dollar bar Gibbs sampler (eps=0.1, best from calendar bars):")
    dol_alpha = estimate_alpha(dol_train)
    print(f"  ACF-based alpha: {dol_alpha['alpha_acf']:.3f}, rho1={dol_alpha['rho1']:.4f}")

    rng_gibbs_dol = np.random.default_rng(55)
    gibbs_dol = gibbs_gig_harris(dol_train, alpha_init=dol_alpha["alpha_acf"],
                                   epsilon=0.1, n_iter=5000, burn_in=2000, rng=rng_gibbs_dol)

    # Simulate dollar bar trajectories
    n_sim_dol = 2000
    rng_sim_dol = np.random.default_rng(321)
    sim_dol = simulate_predictive_sf_harris(
        dol_train, dol_test, gibbs_dol, gibbs_dol["alpha"],
        Q_type="empirical", n_sim=n_sim_dol, rng=rng_sim_dol
    )
    cov_dol = compute_coverage(dol_test, sim_dol, PROB_LEVELS)
    aad_dol = np.mean([abs(cov_dol[p] - p*100) for p in PROB_LEVELS])

    print(f"\n  Dollar bar coverage (eps=0.1):")
    print(f"  {'p':>6}  {'Coverage':>10}  {'Ideal':>6}  {'Dev':>6}")
    print(f"  {'-'*34}")
    for p in PROB_LEVELS:
        dev = cov_dol[p] - p*100
        print(f"  {p:>6.2f}  {cov_dol[p]:>9.1f}%  {p*100:>5.0f}%  {dev:>+5.1f}pp")
    print(f"  AAD = {aad_dol:.1f}pp")

    # ==================================================================
    # STEP 6: Gibbs sampler on 15-min series
    # ==================================================================
    print("\n" + "=" * 70)
    print("STEP 6: Gibbs sampler on 15-min series")
    print("=" * 70)

    # --- Method A: Log scale with epsilon (as in anzarut_replication.py) ---
    print("\n  [A] Log-scale Gibbs (eps=1e-5 on log differences) ---")
    alpha_est_log = estimate_alpha(train_log)
    print(f"  ACF-based alpha: {alpha_est_log['alpha_acf']:.3f}, rho1={alpha_est_log['rho1']:.4f}")

    rng_gibbs_log = np.random.default_rng(42)
    gibbs_log = gibbs_gig_harris(train_log, alpha_init=alpha_est_log["alpha_acf"],
                                   epsilon=1e-5, n_iter=5000, burn_in=2000, rng=rng_gibbs_log)

    # --- Method B: Raw-scale Gibbs with epsilon ---
    print("\n  [B] Raw-scale Gibbs (eps=1e-5 on raw differences) ---")
    gibbs_raw = gibbs_raw_scale(train_raw, epsilon=1e-5, n_iter=5000, burn_in=2000,
                                 rng=np.random.default_rng(43))

    # --- Method C: Log-scale Gibbs with larger epsilon (0.1 on log scale) ---
    print("\n  [C] Log-scale Gibbs (eps=0.1 on log differences) ---")
    rng_gibbs_log2 = np.random.default_rng(44)
    gibbs_log_e01 = gibbs_gig_harris(train_log, alpha_init=alpha_est_log["alpha_acf"],
                                       epsilon=0.1, n_iter=5000, burn_in=2000, rng=rng_gibbs_log2)

    # ==================================================================
    # STEP 7: Simulate 15-min trajectories
    # ==================================================================
    print("\n" + "=" * 70)
    print("STEP 7: Simulating 15-min posterior predictive trajectories")
    print("=" * 70)

    n_sim = 2000

    # Method A: Log-scale simulation
    print(f"\n  [A] Simulating {n_sim} trajectories (log-scale, eps=1e-5)...")
    rng_sim_a = np.random.default_rng(123)
    sim_log = simulate_predictive_sf_harris(
        train_log, test_log, gibbs_log, gibbs_log["alpha"],
        Q_type="empirical", n_sim=n_sim, rng=rng_sim_a
    )
    cov_log_15 = compute_coverage(test_log, sim_log, PROB_LEVELS)
    aad_log_15 = np.mean([abs(cov_log_15[p] - p*100) for p in PROB_LEVELS])

    print(f"\n  [A] 15-min coverage (log-scale, eps=1e-5):")
    print(f"  {'p':>6}  {'Coverage':>10}  {'Ideal':>6}  {'Dev':>6}")
    print(f"  {'-'*34}")
    for p in PROB_LEVELS:
        dev = cov_log_15[p] - p*100
        print(f"  {p:>6.2f}  {cov_log_15[p]:>9.1f}%  {p*100:>5.0f}%  {dev:>+5.1f}pp")
    print(f"  AAD = {aad_log_15:.1f}pp")

    # Method C: Log-scale with larger epsilon
    print(f"\n  [C] Simulating {n_sim} trajectories (log-scale, eps=0.1)...")
    rng_sim_c = np.random.default_rng(789)
    sim_log_e01 = simulate_predictive_sf_harris(
        train_log, test_log, gibbs_log_e01, gibbs_log_e01["alpha"],
        Q_type="empirical", n_sim=n_sim, rng=rng_sim_c
    )
    cov_log_e01_15 = compute_coverage(test_log, sim_log_e01, PROB_LEVELS)
    aad_log_e01_15 = np.mean([abs(cov_log_e01_15[p] - p*100) for p in PROB_LEVELS])

    print(f"\n  [C] 15-min coverage (log-scale, eps=0.1):")
    print(f"  {'p':>6}  {'Coverage':>10}  {'Ideal':>6}  {'Dev':>6}")
    print(f"  {'-'*34}")
    for p in PROB_LEVELS:
        dev = cov_log_e01_15[p] - p*100
        print(f"  {p:>6.2f}  {cov_log_e01_15[p]:>9.1f}%  {p*100:>5.0f}%  {dev:>+5.1f}pp")
    print(f"  AAD = {aad_log_e01_15:.1f}pp")

    # Method B: Raw-scale simulation
    print(f"\n  [B] Simulating {n_sim} trajectories (raw-scale, eps=1e-5)...")
    rng_sim_b = np.random.default_rng(456)
    sim_raw = simulate_raw_sf_harris(
        train_raw, test_raw, gibbs_raw, epsilon=1e-5, n_sim=n_sim, rng=rng_sim_b
    )
    # Convert raw predictions to log for comparison with log test data
    sim_raw_log = np.log(np.maximum(sim_raw, 1e-20))
    cov_raw_15 = compute_coverage(test_log, sim_raw_log, PROB_LEVELS)
    aad_raw_15 = np.mean([abs(cov_raw_15[p] - p*100) for p in PROB_LEVELS])

    print(f"\n  [B] 15-min coverage (raw-scale, eps=1e-5, converted to log):")
    print(f"  {'p':>6}  {'Coverage':>10}  {'Ideal':>6}  {'Dev':>6}")
    print(f"  {'-'*34}")
    for p in PROB_LEVELS:
        dev = cov_raw_15[p] - p*100
        print(f"  {p:>6.2f}  {cov_raw_15[p]:>9.1f}%  {p*100:>5.0f}%  {dev:>+5.1f}pp")
    print(f"  AAD = {aad_raw_15:.1f}pp")

    # ==================================================================
    # STEP 8: Aggregate to daily level for Table 3
    # ==================================================================
    print("\n" + "=" * 70)
    print("STEP 8: Aggregate 15-min predictions to daily level")
    print("=" * 70)

    # Method A: log-scale simulation aggregated to daily
    print(f"\n  [A] Aggregating log-scale simulations to daily...")
    cov_log_adj, aad_log_adj, cov_log_unadj, aad_log_unadj, n_valid = aggregate_to_daily(
        sim_log, test_idx, rv_df_test, periodicity, n_sim, PROB_LEVELS
    )

    print(f"  Daily aggregated (adjusted, log-scale, eps=1e-5):")
    print(f"  {n_valid} valid test days")
    print(f"  {'p':>6}  {'Coverage':>10}  {'Ideal':>6}  {'Dev':>6}")
    print(f"  {'-'*34}")
    for p in PROB_LEVELS:
        dev = cov_log_adj[p] - p*100
        print(f"  {p:>6.2f}  {cov_log_adj[p]:>9.1f}%  {p*100:>5.0f}%  {dev:>+5.1f}pp")
    print(f"  AAD = {aad_log_adj:.1f}pp")

    print(f"\n  Daily aggregated (unadjusted, log-scale, eps=1e-5):")
    print(f"  {'p':>6}  {'Coverage':>10}  {'Ideal':>6}  {'Dev':>6}")
    print(f"  {'-'*34}")
    for p in PROB_LEVELS:
        dev = cov_log_unadj[p] - p*100
        print(f"  {p:>6.2f}  {cov_log_unadj[p]:>9.1f}%  {p*100:>5.0f}%  {dev:>+5.1f}pp")
    print(f"  AAD = {aad_log_unadj:.1f}pp")

    # Method C: log-scale eps=0.1 aggregated to daily
    print(f"\n  [C] Aggregating log-scale (eps=0.1) simulations to daily...")
    cov_e01_adj, aad_e01_adj, cov_e01_unadj, aad_e01_unadj, _ = aggregate_to_daily(
        sim_log_e01, test_idx, rv_df_test, periodicity, n_sim, PROB_LEVELS
    )

    print(f"  Daily aggregated (adjusted, log-scale, eps=0.1):")
    print(f"  {'p':>6}  {'Coverage':>10}  {'Ideal':>6}  {'Dev':>6}")
    print(f"  {'-'*34}")
    for p in PROB_LEVELS:
        dev = cov_e01_adj[p] - p*100
        print(f"  {p:>6.2f}  {cov_e01_adj[p]:>9.1f}%  {p*100:>5.0f}%  {dev:>+5.1f}pp")
    print(f"  AAD = {aad_e01_adj:.1f}pp")

    # ==================================================================
    # STEP 9: Final comparison
    # ==================================================================
    anz_aad = np.mean([abs(ANZARUT_TABLE3[p] - p*100) for p in PROB_LEVELS])

    print(f"\n{'='*70}")
    print("FINAL COMPARISON — 15-MIN INTRADAY PIPELINE")
    print("=" * 70)
    print(f"\n  Anzarut's Table 3 (GIG+Gibbs, 15-min): AAD = {anz_aad:.1f}pp")
    print(f"\n  {'Method':>55}  {'AAD':>6}")
    print(f"  {'-'*65}")
    print(f"  {'--- 15-min level:':>55}")
    print(f"  {'[A] Log-scale Gibbs (eps=1e-5) at 15-min':>55}  {aad_log_15:>5.1f}pp")
    print(f"  {'[C] Log-scale Gibbs (eps=0.1) at 15-min':>55}  {aad_log_e01_15:>5.1f}pp")
    print(f"  {'[B] Raw-scale Gibbs (eps=1e-5) at 15-min':>55}  {aad_raw_15:>5.1f}pp")
    print(f"  {'--- Daily aggregated:':>55}")
    print(f"  {'[A] Log-scale (eps=1e-5) daily aggregate (adj)':>55}  {aad_log_adj:>5.1f}pp")
    print(f"  {'[A] Log-scale (eps=1e-5) daily aggregate (unadj)':>55}  {aad_log_unadj:>5.1f}pp")
    print(f"  {'[C] Log-scale (eps=0.1) daily aggregate (adj)':>55}  {aad_e01_adj:>5.1f}pp")
    print(f"  {'--- Dollar bars (no periodicity):':>55}")
    print(f"  {'[D] Dollar bar Gibbs (eps=0.1) at bar level':>55}  {aad_dol:>5.1f}pp")
    print(f"  {'--- Previous best results (daily level):':>55}")
    print(f"  {'Mixture-Harris + Emp + Boot (dollar bars)':>55}  {'2.2pp':>6}")
    print(f"  {'Mixture-Harris + Emp + Boot (calendar)':>55}  {'3.2pp':>6}")
    print(f"  {'SF-Harris + Gibbs (daily, eps=1e-5)':>55}  {'12.8pp':>6}")

    # Distribution diagnostics
    print(f"\n{'='*70}")
    print("DISTRIBUTION DIAGNOSTICS")
    print("=" * 70)

    print(f"\n  15-min log spot vol (test):")
    print(f"    mean={test_log.mean():.4f}, std={test_log.std():.4f}, "
          f"skew={stats.skew(test_log):.2f}, kurt={stats.kurtosis(test_log):.2f}")

    flat_log = sim_log.flatten()
    print(f"  15-min sim (log-scale, eps=1e-5):")
    print(f"    mean={flat_log.mean():.4f}, std={flat_log.std():.4f}, "
          f"skew={stats.skew(flat_log):.2f}, kurt={stats.kurtosis(flat_log):.2f}")

    flat_e01 = sim_log_e01.flatten()
    print(f"  15-min sim (log-scale, eps=0.1):")
    print(f"    mean={flat_e01.mean():.4f}, std={flat_e01.std():.4f}, "
          f"skew={stats.skew(flat_e01):.2f}, kurt={stats.kurtosis(flat_e01):.2f}")

    # ACF comparison
    acf_test = np.zeros(6)
    acf_test[0] = 1.0
    mu_test = np.mean(test_log)
    var_test = np.var(test_log)
    for h in range(1, 6):
        acf_test[h] = np.mean((test_log[:-h] - mu_test) * (test_log[h:] - mu_test)) / var_test

    acf_sim = np.zeros(6)
    acf_sim[0] = 1.0
    flat_mean = flat_log.mean()
    flat_var = flat_log.var()
    for h in range(1, 6):
        acf_sim[h] = np.mean((flat_log[:-h] - flat_mean) * (flat_log[h:] - flat_mean)) / flat_var

    print(f"\n  ACF comparison (15-min level):")
    print(f"  {'Lag':>6}  {'Test':>8}  {'Sim(A)':>8}  {'Sim(C)':>8}")
    for h in range(6):
        print(f"  {h:>6}  {acf_test[h]:>8.4f}  {acf_sim[h]:>8.4f}  ", end="")
        acf_c = np.zeros(6)
        acf_c[0] = 1.0
        mu_c = flat_e01.mean()
        var_c = flat_e01.var()
        for h2 in range(1, 6):
            acf_c[h2] = np.mean((flat_e01[:-h2] - mu_c) * (flat_e01[h2:] - mu_c)) / var_c
        if h == 0:
            print(f"{1.0:>8.4f}")
        elif h <= 5:
            print(f"{acf_c[h]:>8.4f}")

    # Save results
    results = {
        "method": ["log_eps1e5_15min", "log_eps0.1_15min", "raw_eps1e5_15min",
                    "log_eps1e5_daily_adj", "log_eps1e5_daily_unadj", "log_eps0.1_daily_adj"],
        "aad": [aad_log_15, aad_log_e01_15, aad_raw_15,
                aad_log_adj, aad_log_unadj, aad_e01_adj],
    }
    for p in PROB_LEVELS:
        results[f"cov_{p}"] = [cov_log_15.get(p, 0), cov_log_e01_15.get(p, 0), cov_raw_15.get(p, 0),
                                cov_log_adj.get(p, 0), cov_log_unadj.get(p, 0), cov_e01_adj.get(p, 0)]

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_DIR / "anzarut_intraday_15min_results.csv", index=False)
    print(f"\n  Results saved to {OUTPUT_DIR / 'anzarut_intraday_15min_results.csv'}")