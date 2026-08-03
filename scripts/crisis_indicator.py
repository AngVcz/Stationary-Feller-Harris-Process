"""SF-Harris Crisis Indicator: Predictive Distribution Surprise.

Core idea: SF-Harris gives a predictive distribution for log(RV).
When realized volatility consistently falls in the extreme tails of this
distribution, the model is "surprised" — signaling potential regime change.

Metrics:
1. PIT (Probability Integral Transform): where actual RV falls in the predictive CDF
2. Surprise index: |PIT - 0.5| or -2*log(min(PIT, 1-PIT))
3. Rolling surprise: accumulation of model surprise over recent window

Evaluation: AUC for detecting known crisis periods, compared to VIX baselines.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from anzarut_replication import (
    load_ibm_data, compute_15min_returns, detect_and_remove_jumps,
    estimate_alpha, gibbs_gig_harris, simulate_predictive_sf_harris
)

VIX_PATH = Path(r"C:\Users\angve\Downloads\VIX_History.csv")

# Crisis periods (market stress events)
CRISIS_PERIODS = {
    "LTCM":     ("1998-08-01", "1998-10-31"),
    "Dot-com":  ("2000-03-01", "2002-10-31"),
    "GFC":      ("2007-10-01", "2009-03-31"),
    "Flash":    ("2010-05-01", "2010-05-31"),
    "Euro":     ("2011-08-01", "2011-10-31"),
    "Taper":    ("2013-05-01", "2013-06-30"),
    "Oil":      ("2015-08-01", "2016-02-28"),
    "Volma":    ("2018-02-01", "2018-02-28"),
    "COVID":    ("2020-02-19", "2020-04-30"),
    "Rate hike":("2022-01-01", "2022-06-30"),
}

def is_crisis(date, crisis_periods):
    for name, (start, end) in crisis_periods.items():
        if pd.Timestamp(start) <= date <= pd.Timestamp(end):
            return True
    return False

def compute_pit(actual, simulations):
    """Probability Integral Transform: fraction of sims <= actual value."""
    n_sim = simulations.shape[0]
    pit = np.sum(simulations <= actual.reshape(1, -1), axis=0) / n_sim
    return pit

def surprise_index(pit):
    """Transform PIT to surprise: -2*log(min(PIT, 1-PIT)).
    Under uniformity, this ~ Exp(2) if model is correct.
    Large values = model surprise."""
    pit_clipped = np.clip(pit, 1e-10, 1 - 1e-10)
    return -2 * np.log(np.minimum(pit_clipped, 1 - pit_clipped))

def rolling_surprise(surprise, window):
    """Rolling average of surprise over past `window` days."""
    s = pd.Series(surprise)
    return s.rolling(window, min_periods=1).mean().values

print("=" * 70)
print("SF-HARRIS CRISIS INDICATOR: PREDICTIVE DISTRIBUTION SURPRISE")
print("=" * 70)

# ==================================================================
# 1. Load data and compute daily RV
# ==================================================================
print("\n--- Loading data ---")
df = load_ibm_data(start_date="1998-01-01", end_date="2026-12-31")
returns = compute_15min_returns(df)
returns_clean = detect_and_remove_jumps(returns, n_passes=2, top_pct=0.001)

rv_df = returns_clean.to_frame("return")
rv_df["rv_15min"] = rv_df["return"]**2
rv_df["date"] = pd.to_datetime(rv_df.index.date)
daily_rv_all = rv_df.groupby("date")["rv_15min"].sum()
daily_rv_all = daily_rv_all[daily_rv_all > 0]

close_daily = df["close"].resample("D").last().dropna()
daily_log_ret = np.log(close_daily).diff().dropna()

common = daily_rv_all.index.intersection(daily_log_ret.index)
daily_rv = daily_rv_all.loc[common].values
daily_ret = daily_log_ret.loc[common].values
daily_log_spot = np.log(daily_rv)
daily_log_spot = daily_log_spot[np.isfinite(daily_log_spot)]
dates = common[np.isfinite(daily_rv[common <= daily_rv_all.index[-1]])]
# Re-derive dates properly
valid_mask = np.isfinite(np.log(daily_rv_all.loc[common].values))
dates_all = common[valid_mask]
daily_rv_v = daily_rv_all.loc[dates_all].values
daily_ret_v = daily_log_ret.loc[dates_all].values
daily_log_spot_v = np.log(daily_rv_v)

n = len(daily_log_spot_v)
print(f"  Total days: {n} ({dates_all[0].strftime('%Y-%m-%d')} to {dates_all[-1].strftime('%Y-%m-%d')})")

# ==================================================================
# 2. Fit SF-Harris on pre-GFC data (1998-2007)
# ==================================================================
print("\n--- Fitting SF-Harris on pre-GFC data (1998-2007) ---")
train_end = pd.Timestamp("2007-12-31")
train_mask = dates_all <= train_end
train_log = daily_log_spot_v[train_mask]
test_mask = dates_all > train_end
test_log = daily_log_spot_v[test_mask]
test_dates = dates_all[test_mask]
test_rv = daily_rv_v[test_mask]
test_ret = daily_ret_v[test_mask]

print(f"  Train: {len(train_log)} days (up to {train_end.strftime('%Y-%m-%d')})")
print(f"  Test:  {len(test_log)} days ({test_dates[0].strftime('%Y-%m-%d')} to {test_dates[-1].strftime('%Y-%m-%d')})")

daily_alpha = estimate_alpha(train_log)
rng_gibbs = np.random.default_rng(111)
gibbs_result = gibbs_gig_harris(train_log, alpha_init=daily_alpha["alpha_acf"],
                                  epsilon=1e-5, n_iter=5000, burn_in=2000, rng=rng_gibbs)

print(f"  alpha = {float(np.mean(gibbs_result['alpha'])):.4f}")

# ==================================================================
# 3. Generate predictive distributions
# ==================================================================
print("\n--- Generating predictive distributions (2000 simulations) ---")
rng_sim = np.random.default_rng(222)
n_sim = 2000
n_test = len(test_log)
sim_log = simulate_predictive_sf_harris(
    train_log, test_log, gibbs_result, gibbs_result["alpha"],
    Q_type="empirical", n_sim=n_sim, rng=rng_sim
)
n_test_actual = min(n_test, sim_log.shape[1])
sim_rv = np.exp(sim_log[:, :n_test_actual])
test_rv_aligned = test_rv[:n_test_actual]
test_ret_aligned = test_ret[:n_test_actual]
test_dates_aligned = test_dates[:n_test_actual]

print(f"  Predictive distribution shape: {sim_rv.shape}")
print(f"  Test days: {n_test_actual}")

# ==================================================================
# 4. Compute PIT and surprise
# ==================================================================
print("\n--- Computing PIT and surprise metrics ---")

# PIT for log(RV) - where does actual log(RV) fall in the predictive CDF?
test_log_rv = np.log(test_rv_aligned)
sim_log_rv = np.log(sim_rv)

pit = compute_pit(test_log_rv, sim_log_rv)
surprise = surprise_index(pit)

# Also compute PIT for RV itself (not log)
pit_rv = compute_pit(test_rv_aligned, sim_rv)
surprise_rv = surprise_index(pit_rv)

# Rolling surprise
surprise_5 = rolling_surprise(surprise, 5)
surprise_10 = rolling_surprise(surprise, 10)
surprise_21 = rolling_surprise(surprise, 21)

# Mark crisis days
crisis_mask = np.array([is_crisis(d, CRISIS_PERIODS) for d in test_dates_aligned])
n_crisis = np.sum(crisis_mask)
n_normal = np.sum(~crisis_mask)

print(f"  Crisis days: {n_crisis} ({n_crisis/n_test_actual:.1%})")
print(f"  Normal days:  {n_normal} ({n_normal/n_test_actual:.1%})")

# ==================================================================
# 5. PIT uniformity test
# ==================================================================
print(f"\n{'='*70}")
print("PIT UNIFORMITY ANALYSIS")
print("=" * 70)

# Overall PIT statistics
print(f"\n  Overall PIT statistics:")
print(f"    Mean PIT:   {np.mean(pit):.4f} (uniform = 0.50)")
print(f"    Std PIT:    {np.std(pit):.4f} (uniform = {1/np.sqrt(12):.4f})")
print(f"    Skew PIT:   {stats.skew(pit):.4f} (uniform = 0)")
print(f"    Kurt PIT:   {stats.kurtosis(pit):.4f} (uniform = -1.2)")

# Kolmogorov-Smirnov test for uniformity
ks_stat, ks_p = stats.kstest(pit, 'uniform')
print(f"    KS test:    stat={ks_stat:.4f}, p={ks_p:.4f}")

# PIT distribution: fraction in each decile
deciles = np.linspace(0, 1, 11)
pit_hist, _ = np.histogram(pit, bins=deciles)
print(f"\n  PIT decile distribution (should be ~10% each):")
print(f"  {'Decile':>8}  {'Count':>6}  {'Pct':>6}  {'Ideal':>6}")
for i in range(10):
    pct = pit_hist[i] / len(pit) * 100
    print(f"  [{deciles[i]:.1f}-{deciles[i+1]:.1f}]  {pit_hist[i]:>6}  {pct:>5.1f}%  10.0%")

# PIT by crisis vs normal
print(f"\n  PIT during crisis vs normal days:")
print(f"    Crisis mean PIT:  {np.mean(pit[crisis_mask]):.4f}")
print(f"    Normal mean PIT:  {np.mean(pit[~crisis_mask]):.4f}")
print(f"    Crisis mean surprise:  {np.mean(surprise[crisis_mask]):.4f}")
print(f"    Normal mean surprise:  {np.mean(surprise[~crisis_mask]):.4f}")

# ==================================================================
# 6. Surprise index as crisis predictor
# ==================================================================
print(f"\n{'='*70}")
print("CRISIS PREDICTION: AUC ANALYSIS")
print("=" * 70)

# Compute AUC for each surprise metric predicting crisis days
metrics = {
    "PIT": pit,
    "Surprise (raw)": surprise,
    "Surprise 5d": surprise_5,
    "Surprise 10d": surprise_10,
    "Surprise 21d": surprise_21,
}

# Also compute VIX-based baselines if available
try:
    vix_df = pd.read_csv(VIX_PATH)
    vix_df.columns = [c.strip().lower() for c in vix_df.columns]
    vix_df["date"] = pd.to_datetime(vix_df["date"], format="%m/%d/%Y")
    vix_df = vix_df.set_index("date").sort_index()
    vix_daily_var = (vix_df["close"] / 100) ** 2 / 252

    vix_aligned = vix_daily_var.reindex(test_dates_aligned)
    vix_valid = vix_aligned.notna()
    vix_level = vix_aligned.dropna().values

    # VIX change
    vix_change = np.diff(vix_level, prepend=vix_level[0])

    # Rolling VIX level
    vix_rolling = pd.Series(vix_level).rolling(21, min_periods=1).mean().values

    metrics["VIX level"] = np.full(n_test_actual, np.nan)
    metrics["VIX level"][:len(vix_level)] = vix_level
    metrics["VIX change"] = np.full(n_test_actual, np.nan)
    metrics["VIX change"][:len(vix_change)] = vix_change
    metrics["VIX 21d avg"] = np.full(n_test_actual, np.nan)
    metrics["VIX 21d avg"][:len(vix_rolling)] = vix_rolling
    has_vix = True
except Exception:
    print("  [VIX data not available, skipping VIX baselines]")
    has_vix = False

# RV-based baselines
rv_level = test_rv_aligned
rv_change = np.diff(rv_level, prepend=rv_level[0])
rv_rolling = pd.Series(rv_level).rolling(21, min_periods=1).mean().values

metrics["RV level"] = rv_level
metrics["RV change"] = rv_change
metrics["RV 21d avg"] = rv_rolling

print(f"\n  AUC for detecting crisis days (higher = better):")
print(f"  {'Metric':>20}  {'AUC':>6}  {'p-val':>8}")
print(f"  {'-'*40}")

for name, values in metrics.items():
    valid = np.isfinite(values) & np.isfinite(crisis_mask.astype(float))
    if np.sum(valid) < 100:
        continue
    v = values[valid]
    c = crisis_mask[valid]
    if np.std(v) < 1e-10 or np.sum(c) < 10:
        continue
    try:
        auc = stats.roc_auc_score(c, v)
        # For PIT, lower values might indicate crisis (model surprised)
        # For surprise, higher values indicate crisis
        # Adjust direction: we want AUC > 0.5 if metric increases during crisis
        if name == "PIT" and auc < 0.5:
            auc = 1 - auc  # reverse: extreme PIT (both tails) matters
        p_val = stats.mannwhitneyu(v[c], v[~c])[1]
        print(f"  {name:>20}  {auc:>6.3f}  {p_val:>8.4f}")
    except Exception:
        pass

# ==================================================================
# 7. Per-crisis analysis
# ==================================================================
print(f"\n{'='*70}")
print("PER-CRISIS SURPRISE ANALYSIS")
print("=" * 70)

for name, (start, end) in CRISIS_PERIODS.items():
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    mask = (test_dates_aligned >= start_ts) & (test_dates_aligned <= end_ts)
    if np.sum(mask) < 3:
        continue
    crisis_pit = pit[mask]
    crisis_surprise = surprise[mask]
    crisis_surprise_5 = surprise_5[mask]
    n_days = np.sum(mask)

    # Pre-crisis: 21 trading days before crisis start
    pre_start = start_ts - pd.Timedelta(days=31)
    pre_mask = (test_dates_aligned >= pre_start) & (test_dates_aligned < start_ts)
    pre_n = np.sum(pre_mask)

    print(f"\n  {name} ({start} to {end}, {n_days} days):")
    print(f"    Mean PIT:       {np.mean(crisis_pit):.4f} (vs 0.50 uniform)")
    print(f"    Mean surprise:  {np.mean(crisis_surprise):.4f}")
    print(f"    Mean surprise-5d: {np.mean(crisis_surprise_5):.4f}")
    print(f"    % extreme (PIT<0.05 or >0.95): {np.mean((crisis_pit<0.05)|(crisis_pit>0.95)):.1%}")
    print(f"    % very extreme (PIT<0.01 or >0.99): {np.mean((crisis_pit<0.01)|(crisis_pit>0.99)):.1%}")
    if pre_n > 5:
        print(f"    Pre-crisis (21d before) surprise: {np.mean(surprise_5[pre_mask]):.4f}")
        print(f"    Pre-crisis vs overall: {np.mean(surprise_5[pre_mask]):.4f} vs {np.mean(surprise_5):.4f}")

# ==================================================================
# 8. Threshold analysis for exit signal
# ==================================================================
print(f"\n{'='*70}")
print("THRESHOLD ANALYSIS: EXIT SIGNAL PERFORMANCE")
print("=" * 70)

# Use rolling 5-day surprise as the signal
# For each threshold, compute: true positive rate, false positive rate, precision
print(f"\n  Using rolling 5-day surprise as exit signal:")
print(f"  Threshold  TPR     FPR    Precision  F1     Lead time")
print(f"  {'-'*65}")

for threshold in np.percentile(surprise_5, [50, 60, 70, 75, 80, 85, 90, 95]):
    signal = surprise_5 >= threshold
    tp = np.sum(signal & crisis_mask)
    fp = np.sum(signal & ~crisis_mask)
    fn = np.sum(~signal & crisis_mask)
    tn = np.sum(~signal & ~crisis_mask)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * prec * tpr / (prec + tpr) if (prec + tpr) > 0 else 0

    # Average lead time: for each crisis day, how many days before was the signal on?
    pctile = np.sum(surprise_5 <= threshold) / len(surprise_5) * 100
    print(f"  P{pctile:.0f} ({threshold:.2f})  {tpr:.3f}  {fpr:.3f}  {prec:.3f}     {f1:.3f}")

# ==================================================================
# 9. Pre-crisis signal analysis
# ==================================================================
print(f"\n{'='*70}")
print("PRE-CRISIS LEAD ANALYSIS: DOES SURPRISE INCREASE BEFORE CRISIS?")
print("=" * 70)

# For each crisis, compute surprise in the N days before
for window_name, window in [("5d", 5), ("10d", 10), ("21d", 21)]:
    print(f"\n  Surprise in {window_name} BEFORE each crisis:")
    for name, (start, end) in CRISIS_PERIODS.items():
        start_ts = pd.Timestamp(start)
        pre_start = start_ts - pd.Timedelta(days=int(window * 1.5))
        pre_mask = (test_dates_aligned >= pre_start) & (test_dates_aligned < start_ts)
        if np.sum(pre_mask) < 3:
            continue
        pre_surprise = np.mean(surprise_5[pre_mask])
        overall_surprise = np.mean(surprise_5)
        ratio = pre_surprise / overall_surprise if overall_surprise > 0 else 0
        print(f"    {name:>10}: {pre_surprise:.3f} (vs overall {overall_surprise:.3f}, ratio={ratio:.2f}x)")

# ==================================================================
# 10. Visualization data summary
# ==================================================================
print(f"\n{'='*70}")
print("TIMESERIES SUMMARY (for plotting)")
print("=" * 70)

# Output key series for plotting
summary_df = pd.DataFrame({
    "date": test_dates_aligned,
    "rv": test_rv_aligned,
    "log_rv": test_log_rv,
    "pit": pit,
    "surprise": surprise,
    "surprise_5d": surprise_5,
    "surprise_21d": surprise_21,
    "crisis": crisis_mask.astype(int),
})

if has_vix:
    vix_for_df = vix_daily_var.reindex(test_dates_aligned).values
    summary_df["vix_var"] = vix_for_df

# Print crisis periods for reference
print(f"\n  Crisis periods defined:")
for name, (start, end) in CRISIS_PERIODS.items():
    mask = (test_dates_aligned >= pd.Timestamp(start)) & (test_dates_aligned <= pd.Timestamp(end))
    print(f"    {name:>10}: {start} to {end} ({np.sum(mask)} days)")

# Summary stats
print(f"\n  Key summary:")
print(f"    Surprise during crisis:   {np.mean(surprise[crisis_mask]):.3f}")
print(f"    Surprise during normal:   {np.mean(surprise[~crisis_mask]):.3f}")
print(f"    Ratio:                    {np.mean(surprise[crisis_mask])/np.mean(surprise[~crisis_mask]):.2f}x")
print(f"    PIT mean during crisis:  {np.mean(pit[crisis_mask]):.4f}")
print(f"    PIT mean during normal:  {np.mean(pit[~crisis_mask]):.4f}")
print(f"    % extreme PIT during crisis: {np.mean((pit[crisis_mask]<0.05)|(pit[crisis_mask]>0.95)):.1%}")
print(f"    % extreme PIT during normal: {np.mean((pit[~crisis_mask]<0.05)|(pit[~crisis_mask]>0.95)):.1%}")