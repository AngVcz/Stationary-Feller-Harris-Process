"""SF-Harris Crisis Indicator v2: Rolling-Window with Adaptive PIT.

Key fix: v1 trained on pre-GFC data and tested on 2008-2026.
The model was miscalibrated (PIT mean=0.36, not 0.50) because it was
frozen in 2007 expectations.

v2 uses an expanding window with annual refitting. At each year Y:
  - Fit SF-Harris on all data up to Y-1
  - Compute PIT for days in year Y
  - This gives a locally-adapted PIT that should be more uniform

Also adds: z-score based surprise (less sensitive to distribution shape
than PIT), and VRP-based surprise (VIX vs SF-Harris expected variance).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from anzarut_replication import (
    load_ibm_data, compute_15min_returns, detect_and_remove_jumps,
    estimate_alpha, gibbs_gig_harris, simulate_predictive_sf_harris
)

VIX_PATH = Path(r"C:\Users\angve\Downloads\VIX_History.csv")

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

print("=" * 70)
print("SF-HARRIS CRISIS INDICATOR v2: ROLLING-WINDOW ADAPTIVE PIT")
print("=" * 70)

# ==================================================================
# 1. Load data
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
valid = np.isfinite(daily_log_spot)
dates_all = common[valid]
daily_rv_v = daily_rv[valid]
daily_ret_v = daily_ret[valid]
daily_log_spot_v = daily_log_spot[valid]

n = len(daily_log_spot_v)
print(f"  Total days: {n} ({dates_all[0].strftime('%Y-%m-%d')} to {dates_all[-1].strftime('%Y-%m-%d')})")

# ==================================================================
# 2. Expanding window with annual refit
# ==================================================================
print("\n--- Rolling window: expanding, annual refit ---")
print("  Train on [start, Y-1], compute PIT for year Y")

# Minimum training window
MIN_TRAIN = 1000  # ~4 years
REFIT_YEAR = 1

# Build year groups
years = pd.Series([d.year for d in dates_all])
unique_years = sorted(years.unique())

all_pit = np.full(n, np.nan)
all_surprise = np.full(n, np.nan)
all_zscore = np.full(n, np.nan)
all_median = np.full(n, np.nan)
all_iqr = np.full(n, np.nan)
all_tau = np.full(n, np.nan)
all_actual_log_rv = daily_log_spot_v.copy()

processed = 0

for yi, year in enumerate(unique_years):
    year_mask = years == year
    year_idx = np.where(year_mask & (np.arange(n) >= MIN_TRAIN))[0]
    if len(year_idx) == 0:
        continue

    # Train on all data before this year
    train_end = year_idx[0]
    train_log = daily_log_spot_v[:train_end]

    if len(train_log) < MIN_TRAIN:
        continue

    # Test indices for this year
    test_idx = year_idx

    # Fit SF-Harris on training data
    try:
        alpha_est = estimate_alpha(train_log)
        rng_gibbs = np.random.default_rng(1000 + yi)
        gibbs = gibbs_gig_harris(train_log, alpha_init=alpha_est["alpha_acf"],
                                   epsilon=1e-5, n_iter=3000, burn_in=1000, rng=rng_gibbs)
    except Exception as e:
        print(f"  Year {year}: Gibbs failed ({e}), skipping")
        continue

    # Generate predictive distribution for test days
    test_log = daily_log_spot_v[test_idx]
    rng_sim = np.random.default_rng(2000 + yi)
    try:
        sim_log = simulate_predictive_sf_harris(
            train_log, test_log, gibbs, gibbs["alpha"],
            Q_type="empirical", n_sim=1000, rng=rng_sim
        )
    except Exception as e:
        print(f"  Year {year}: Simulation failed ({e}), skipping")
        continue

    n_test_y = min(len(test_idx), sim_log.shape[1])
    sim_rv_y = np.exp(sim_log[:, :n_test_y])

    # Compute PIT, z-score, and tau for this year
    for j in range(n_test_y):
        idx = test_idx[j]
        actual = daily_log_spot_v[idx]
        sims = sim_log[:, j]

        # PIT: fraction of sims <= actual
        pit = np.mean(sims <= actual)
        all_pit[idx] = pit

        # Surprise: -2*log(min(PIT, 1-PIT))
        pit_c = np.clip(pit, 1e-10, 1 - 1e-10)
        all_surprise[idx] = -2 * np.log(min(pit_c, 1 - pit_c))

        # z-score: (actual - median) / IQR
        med = np.median(sims)
        iqr = np.percentile(sims, 75) - np.percentile(sims, 25)
        all_zscore[idx] = (actual - med) / iqr if iqr > 1e-10 else 0.0
        all_median[idx] = med
        all_iqr[idx] = iqr
        all_tau[idx] = np.median(sim_rv_y[:, j])

    processed += n_test_y
    print(f"  Year {year}: {n_test_y} days processed, train={train_end} days")

print(f"\n  Total processed: {processed} days")

# ==================================================================
# 3. Valid mask
# ==================================================================
valid_mask = np.isfinite(all_pit)
pit = all_pit[valid_mask]
surprise = all_surprise[valid_mask]
zscore = all_zscore[valid_mask]
dates_valid = dates_all[valid_mask]
rv_valid = daily_rv_v[valid_mask]
crisis_mask = np.array([is_crisis(d, CRISIS_PERIODS) for d in dates_valid])
n_valid = len(pit)

print(f"  Valid days: {n_valid}")
print(f"  Crisis days: {np.sum(crisis_mask)} ({np.mean(crisis_mask):.1%})")

# ==================================================================
# 4. PIT uniformity (should be much better with rolling window)
# ==================================================================
print(f"\n{'='*70}")
print("PIT UNIFORMITY (ROLLING WINDOW)")
print("=" * 70)

print(f"\n  PIT statistics:")
print(f"    Mean:   {np.mean(pit):.4f} (uniform = 0.50)")
print(f"    Std:    {np.std(pit):.4f} (uniform = {1/np.sqrt(12):.4f})")
ks_stat, ks_p = stats.kstest(pit, 'uniform')
print(f"    KS test: stat={ks_stat:.4f}, p={ks_p:.4f}")

deciles = np.linspace(0, 1, 11)
pit_hist, _ = np.histogram(pit, bins=deciles)
print(f"\n  PIT decile distribution (should be ~10% each):")
print(f"  {'Decile':>10}  {'Count':>6}  {'Pct':>6}  {'Ideal':>6}")
for i in range(10):
    pct = pit_hist[i] / len(pit) * 100
    print(f"  [{deciles[i]:.1f}-{deciles[i+1]:.1f}]  {pit_hist[i]:>6}  {pct:>5.1f}%  10.0%")

# ==================================================================
# 5. Crisis vs normal comparison
# ==================================================================
print(f"\n{'='*70}")
print("CRISIS vs NORMAL COMPARISON")
print("=" * 70)

print(f"\n  {'Metric':>25}  {'Crisis':>10}  {'Normal':>10}  {'Ratio':>8}")
print(f"  {'-'*60}")
for name, vals in [("PIT", pit), ("Surprise", surprise), ("|z-score|", np.abs(zscore))]:
    c = vals[crisis_mask]
    n = vals[~crisis_mask]
    print(f"  {name:>25}  {np.mean(c):>10.4f}  {np.mean(n):>10.4f}  {np.mean(c)/np.mean(n) if np.mean(n) > 0 else 0:>8.2f}x")

# Mann-Whitney U test
u_stat, u_p = stats.mannwhitneyu(pit[crisis_mask], pit[~crisis_mask], alternative='two-sided')
print(f"\n  Mann-Whitney U (PIT, crisis vs normal): U={u_stat:.0f}, p={u_p:.4f}")

# AUC for crisis detection
for name, vals in [("PIT", pit), ("Surprise", surprise), ("|z-score|", np.abs(zscore))]:
    valid_auc = np.isfinite(vals) & np.isfinite(crisis_mask.astype(float))
    if np.sum(valid_auc) > 100:
        try:
            auc = roc_auc_score(crisis_mask[valid_auc], vals[valid_auc])
            # For PIT: both tails matter, so also try |PIT - 0.5|
            if name == "PIT":
                auc_abs = roc_auc_score(crisis_mask[valid_auc], np.abs(pit[valid_auc] - 0.5))
                print(f"  AUC ({name}): {auc:.3f}")
                print(f"  AUC (|PIT-0.5|): {auc_abs:.3f}")
            else:
                print(f"  AUC ({name}): {auc:.3f}")
        except Exception:
            pass

# ==================================================================
# 6. Per-crisis analysis
# ==================================================================
print(f"\n{'='*70}")
print("PER-CRISIS ANALYSIS (ROLLING WINDOW)")
print("=" * 70)

for name, (start, end) in CRISIS_PERIODS.items():
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    mask = (dates_valid >= start_ts) & (dates_valid <= end_ts)
    if np.sum(mask) < 3:
        continue

    crisis_pit = pit[mask]
    crisis_surprise = surprise[mask]
    crisis_zscore = zscore[mask]
    n_days = np.sum(mask)

    # Pre-crisis: 21 trading days before
    pre_start = start_ts - pd.Timedelta(days=31)
    pre_mask = (dates_valid >= pre_start) & (dates_valid < start_ts)
    pre_n = np.sum(pre_mask)

    # Post-crisis: 21 trading days after
    post_end = end_ts + pd.Timedelta(days=31)
    post_mask = (dates_valid > end_ts) & (dates_valid <= post_end)
    post_n = np.sum(post_mask)

    print(f"\n  {name} ({start} to {end}, {n_days} days):")
    print(f"    PIT:    {np.mean(crisis_pit):.4f} (uniform=0.50)")
    print(f"    z-score: {np.mean(crisis_zscore):.3f}")
    print(f"    Surprise: {np.mean(crisis_surprise):.3f}")
    print(f"    % extreme PIT: {np.mean((crisis_pit<0.05)|(crisis_pit>0.95)):.1%}")
    if pre_n > 5:
        print(f"    Pre-crisis PIT:    {np.mean(pit[pre_mask]):.4f}")
        print(f"    Pre-crisis z-score: {np.mean(zscore[pre_mask]):.3f}")
        print(f"    Pre-crisis surprise: {np.mean(surprise[pre_mask]):.3f}")
    if post_n > 5:
        print(f"    Post-crisis PIT:   {np.mean(pit[post_mask]):.4f}")
        print(f"    Post-crisis z-score: {np.mean(zscore[post_mask]):.3f}")

# ==================================================================
# 7. Rolling surprise as exit signal
# ==================================================================
print(f"\n{'='*70}")
print("ROLLING SURPRISE AS EXIT SIGNAL")
print("=" * 70)

# Rolling averages
surprise_5 = pd.Series(surprise).rolling(5, min_periods=1).mean().values
surprise_10 = pd.Series(surprise).rolling(10, min_periods=1).mean().values
surprise_21 = pd.Series(surprise).rolling(21, min_periods=1).mean().values
zscore_5 = pd.Series(np.abs(zscore)).rolling(5, min_periods=1).mean().values
zscore_21 = pd.Series(np.abs(zscore)).rolling(21, min_periods=1).mean().values
pit_abs_5 = pd.Series(np.abs(pit - 0.5)).rolling(5, min_periods=1).mean().values
pit_abs_21 = pd.Series(np.abs(pit - 0.5)).rolling(21, min_periods=1).mean().values

# Rolling RV for baseline
rv_21 = pd.Series(rv_valid).rolling(21, min_periods=1).mean().values
rv_std_21 = pd.Series(rv_valid).rolling(21, min_periods=1).std().values

# Load VIX for baseline
try:
    vix_df = pd.read_csv(VIX_PATH)
    vix_df.columns = [c.strip().lower() for c in vix_df.columns]
    vix_df["date"] = pd.to_datetime(vix_df["date"], format="%m/%d/%Y")
    vix_df = vix_df.set_index("date").sort_index()
    vix_level = vix_df["close"].reindex(dates_valid)
    vix_valid = vix_level.notna()
    vix_21 = vix_level.rolling(21, min_periods=1).mean()
    vix_std = vix_level.rolling(21, min_periods=1).std()
    has_vix = True
except Exception:
    has_vix = False

print(f"\n  AUC for detecting crisis days:")
print(f"  {'Metric':>30}  {'AUC':>6}")
print(f"  {'-'*42}")

metrics_to_test = {
    "PIT (raw)": pit,
    "|PIT - 0.5|": np.abs(pit - 0.5),
    "|PIT - 0.5| 5d avg": pit_abs_5,
    "|PIT - 0.5| 21d avg": pit_abs_21,
    "Surprise (raw)": surprise,
    "Surprise 5d": surprise_5,
    "Surprise 21d": surprise_21,
    "|z-score|": np.abs(zscore),
    "|z-score| 5d": zscore_5,
    "|z-score| 21d": zscore_21,
    "RV level": rv_valid,
    "RV 21d avg": rv_21,
}

if has_vix:
    metrics_to_test["VIX level"] = vix_level.values
    metrics_to_test["VIX 21d avg"] = vix_21.values

for name, values in metrics_to_test.items():
    v = np.array(values, dtype=float)
    valid = np.isfinite(v)
    if np.sum(valid) < 100:
        continue
    c = crisis_mask[valid]
    vv = v[valid]
    if np.sum(c) < 10 or np.std(vv) < 1e-10:
        continue
    try:
        auc = roc_auc_score(c, vv)
        # If AUC < 0.5, the metric decreases during crisis (reverse direction)
        if auc < 0.5:
            auc = 1 - auc
            direction = "↓crisis"
        else:
            direction = "↑crisis"
        print(f"  {name:>30}  {auc:.3f}  {direction}")
    except Exception:
        pass

# ==================================================================
# 8. Pre-crisis lead: does surprise increase BEFORE crisis?
# ==================================================================
print(f"\n{'='*70}")
print("PRE-CRISIS LEAD ANALYSIS (ROLLING WINDOW)")
print("=" * 70)

for window_name, window in [("5d", 5), ("10d", 10), ("21d", 21)]:
    print(f"\n  |z-score| in {window_name} BEFORE each crisis:")
    overall_abs_z = np.mean(np.abs(zscore))
    for name, (start, end) in CRISIS_PERIODS.items():
        start_ts = pd.Timestamp(start)
        pre_start = start_ts - pd.Timedelta(days=int(window * 1.5))
        pre_mask = (dates_valid >= pre_start) & (dates_valid < start_ts)
        if np.sum(pre_mask) < 3:
            continue
        pre_z = np.mean(np.abs(zscore[pre_mask]))
        print(f"    {name:>10}: {pre_z:.3f} (vs overall {overall_abs_z:.3f}, ratio={pre_z/overall_abs_z:.2f}x)")

# ==================================================================
# 9. Threshold analysis
# ==================================================================
print(f"\n{'='*70}")
print("THRESHOLD ANALYSIS: |z-score| 5d as EXIT SIGNAL")
print("=" * 70)

best_signal = zscore_5
for thresh_name, thresh_vals in [("surprise_5d", surprise_5), ("|z-score|_5d", zscore_5),
                                   ("|PIT-0.5|_5d", pit_abs_5)]:
    print(f"\n  {thresh_name}:")
    print(f"  {'Threshold':>12}  {'TPR':>6}  {'FPR':>6}  {'Prec':>6}  {'F1':>6}")
    print(f"  {'-'*46}")
    for pctile in [50, 60, 70, 75, 80, 85, 90, 95]:
        threshold = np.nanpercentile(thresh_vals, pctile)
        signal = thresh_vals >= threshold
        valid_s = np.isfinite(thresh_vals)
        s = signal & valid_s
        c = crisis_mask & valid_s
        tp = np.sum(s & c)
        fp = np.sum(s & ~c)
        fn = np.sum(~s & c)
        tn = np.sum(~s & ~c)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2*prec*tpr/(prec+tpr) if (prec+tpr) > 0 else 0
        print(f"  P{pctile:>2} ({threshold:>6.2f})  {tpr:>6.3f}  {fpr:>6.3f}  {prec:>6.3f}  {f1:>6.3f}")

# ==================================================================
# 10. Combined signal: SF-Harris surprise + VIX
# ==================================================================
if has_vix:
    print(f"\n{'='*70}")
    print("COMBINED SIGNAL: SF-HARRIS SURPRISE + VIX")
    print("=" * 70)

    vix_v = vix_level.values
    valid_combo = np.isfinite(vix_v) & np.isfinite(zscore_5)
    if np.sum(valid_combo) > 100:
        # Normalize both signals
        z5_norm = (zscore_5[valid_combo] - np.nanmean(zscore_5[valid_combo])) / np.nanstd(zscore_5[valid_combo])
        vix_norm = (vix_v[valid_combo] - np.nanmean(vix_v[valid_combo])) / np.nanstd(vix_v[valid_combo])
        combo = 0.5 * z5_norm + 0.5 * vix_norm
        crisis_v = crisis_mask[valid_combo]

        auc_z = roc_auc_score(crisis_v, np.abs(zscore_5[valid_combo]))
        auc_v = roc_auc_score(crisis_v, vix_v[valid_combo])
        auc_c = roc_auc_score(crisis_v, combo)

        print(f"  AUC |z-score| 5d:  {auc_z:.3f}")
        print(f"  AUC VIX level:    {auc_v:.3f}")
        print(f"  AUC Combined:      {auc_c:.3f}")

        # VRP signal
        vix_daily_var = (vix_v / 100) ** 2 / 252
        rv_rolling_21 = pd.Series(rv_valid).rolling(21, min_periods=1).mean().values
        vrp = vix_daily_var - rv_rolling_21
        valid_vrp = np.isfinite(vrp) & np.isfinite(crisis_mask.astype(float))
        if np.sum(valid_vrp) > 100:
            auc_vrp = roc_auc_score(crisis_mask[valid_vrp], vrp[valid_vrp])
            print(f"  AUC VRP (VIX - RV_21d): {auc_vrp:.3f}")

print(f"\n{'='*70}")
print("SUMMARY")
print("=" * 70)
print(f"\n  Rolling-window PIT analysis (expanding window, annual refit)")
print(f"  PIT mean: {np.mean(pit):.4f} (uniform = 0.50)")
print(f"  KS test p-value: {ks_p:.4f}")
print(f"  Crisis PIT mean: {np.mean(pit[crisis_mask]):.4f}")
print(f"  Normal PIT mean: {np.mean(pit[~crisis_mask]):.4f}")
print(f"  Crisis |z-score|: {np.mean(np.abs(zscore[crisis_mask])):.3f}")
print(f"  Normal |z-score|: {np.mean(np.abs(zscore[~crisis_mask])):.3f}")
print(f"  Ratio (crisis/normal): {np.mean(np.abs(zscore[crisis_mask]))/np.mean(np.abs(zscore[~crisis_mask])):.2f}x")