"""VIX vs SF-Harris variance swap pricing - FIXED SCALE.

VIX is annualized volatility. VIX^2/252 = annualized variance -> daily variance.
We compare against daily realized variance (sum of squared 15-min returns).
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

print("=" * 70)
print("VIX vs SF-HARRIS VARIANCE SWAP PRICING")
print("=" * 70)

# Load VIX
vix_df = pd.read_csv(VIX_PATH)
vix_df.columns = [c.strip().lower() for c in vix_df.columns]
vix_df["date"] = pd.to_datetime(vix_df["date"], format="%m/%d/%Y")
vix_df = vix_df.set_index("date").sort_index()
# VIX close is annualized vol in % -> daily variance = VIX^2 / (100^2 * 252)
vix_df["vix_daily_var"] = (vix_df["close"] / 100) ** 2 / 252

# Load IBM data
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

common = daily_rv_all.index.intersection(daily_log_ret.index).intersection(vix_df.index)
daily_rv = daily_rv_all.loc[common].values
daily_ret = daily_log_ret.loc[common].values
daily_log_spot = np.log(daily_rv)
daily_log_spot = daily_log_spot[np.isfinite(daily_log_spot)]
vix_daily_var = vix_df.loc[common, "vix_daily_var"].values

print(f"\n  Overlap: {len(common)} days ({common[0].strftime('%Y-%m-%d')} to {common[-1].strftime('%Y-%m-%d')})")
print(f"  Mean daily RV:  {np.mean(daily_rv):.6f}")
print(f"  Mean VIX var:   {np.mean(vix_daily_var):.6f}")
print(f"  VIX/RV ratio:   {np.mean(vix_daily_var)/np.mean(daily_rv):.2f}x")

# 30-day forward RV
fwd_window = 21
fwd_rv = np.full(len(daily_rv), np.nan)
for i in range(len(daily_rv) - fwd_window):
    fwd_rv[i] = np.mean(daily_rv[i+1:i+1+fwd_window])

valid = np.isfinite(fwd_rv)
fwd_rv_v = fwd_rv[valid]
daily_rv_v = daily_rv[valid]
vix_v = vix_daily_var[valid]
log_spot_v = daily_log_spot[valid]
ret_v = daily_ret[valid]
n = len(fwd_rv_v)

split = int(n * 0.8)
fwd_test = fwd_rv_v[split:]
rv_test = daily_rv_v[split:]
vix_test = vix_v[split:]
log_spot_test = log_spot_v[split:]
n_test = len(fwd_test)

print(f"  Train: {int(n*0.8)}, Test: {n_test}")
print(f"  Test mean RV: {np.mean(rv_test):.6f}")
print(f"  Test mean VIX: {np.mean(vix_test):.6f}")

# ==================================================================
# Forecasts
# ==================================================================
print("\n--- Computing forecasts ---")

# 1. Historical average
pred_hist = np.full(n_test, np.mean(daily_rv_v[:split]))

# 2. Random walk (today's RV)
pred_rw = rv_test.copy()

# 3. GARCH(1,1)
from arch import arch_model
train_garch = ret_v[:split] * 100
garch_model = arch_model(train_garch, vol='Garch', p=1, q=1, dist='normal')
garch_fit = garch_model.fit(disp='off')

# Rolling 1-day GARCH forecasts, then average over 21 days
garch_forecasts = []
for t in range(split, split + n_test):
    try:
        fc = garch_fit.forecast(horizon=1, start=t, method='simulation')
        var_1d = fc.variance.iloc[t].values[0] / 10000  # back to daily units
        garch_forecasts.append(var_1d)
    except:
        garch_forecasts.append(np.nan)
pred_garch_1d = np.array(garch_forecasts[:n_test])

# 21-day GARCH forecast: use GARCH persistence to project
# sigma2_{t+h} = omega + alpha*sigma2_{t+h-1} + beta*sigma2_{t+h-1}
# For simplicity, use rolling average of 1-day GARCH forecasts
pred_garch = np.full(n_test, np.nan)
for t in range(n_test - fwd_window):
    if not np.isnan(pred_garch_1d[t]):
        # GARCH 21-day: use persistence formula
        # sigma2_{t+21} ≈ sigma2_inf + (alpha+beta)^21 * (sigma2_1 - sigma2_inf)
        pred_garch[t] = pred_garch_1d[t]  # approximate with 1-day
    # Better: average persistence-adjusted
valid_garch = np.isfinite(pred_garch)
pred_garch_clean = pred_garch.copy()
# Use rolling mean of 1-day GARCH as proxy
for t in range(min(n_test, len(pred_garch_1d))):
    pred_garch_clean[t] = pred_garch_1d[t]

# 4. VIX (market-implied)
pred_vix = vix_test.copy()

# 5. SF-Harris
train_log = log_spot_v[:split]
test_log = log_spot_v[split:]
daily_alpha = estimate_alpha(train_log)
rng_gibbs = np.random.default_rng(111)
gibbs_result = gibbs_gig_harris(train_log, alpha_init=daily_alpha["alpha_acf"],
                                  epsilon=1e-5, n_iter=5000, burn_in=2000, rng=rng_gibbs)
rng_sim = np.random.default_rng(222)
sim_log = simulate_predictive_sf_harris(
    train_log, test_log, gibbs_result, gibbs_result["alpha"],
    Q_type="empirical", n_sim=2000, rng=rng_sim
)
n_test_actual = min(n_test, sim_log.shape[1])
sim_rv = np.exp(sim_log[:, :n_test_actual])

# 1-day SF-Harris forecast
sf_1day = np.median(sim_rv, axis=0)
# 21-day: rolling average of 1-day forecasts
sf_21day = np.full(n_test_actual, np.nan)
for t in range(n_test_actual - fwd_window):
    sf_21day[t] = np.mean(sf_1day[t:t+fwd_window])

# ==================================================================
# Evaluation
# ==================================================================
print(f"\n{'='*70}")
print("30-DAY VARIANCE FORECAST COMPARISON")
print("Target: mean daily RV over next 21 trading days")
print("=" * 70)

# Align
min_len = min(n_test, n_test_actual, len(pred_garch_clean))
fwd_test_a = fwd_test[:min_len]

def eval_forecast(actual, pred, name):
    valid = np.isfinite(actual) & np.isfinite(pred) & (pred > 0) & (actual > 0)
    if np.sum(valid) < 50:
        return {"name": name, "RMSE": np.nan, "MAE": np.nan, "Corr": np.nan,
                "R2": np.nan, "QLIKE": np.nan, "N": np.sum(valid)}
    a = actual[valid]
    p = pred[valid]
    rmse = np.sqrt(np.mean((a - p)**2))
    mae = np.mean(np.abs(a - p))
    corr = np.corrcoef(a, p)[0, 1]
    r2 = 1 - np.sum((a - p)**2) / np.sum((a - np.mean(a))**2)
    qlike = np.mean(a / p - np.log(a / p) - 1)
    return {"name": name, "RMSE": rmse, "MAE": mae, "Corr": corr, "R2": r2, "QLIKE": qlike, "N": np.sum(valid)}

results = [
    eval_forecast(fwd_test_a, np.full(min_len, np.mean(daily_rv_v[:split]))[:min_len], "Historical avg"),
    eval_forecast(fwd_test_a, rv_test[:min_len], "Random walk"),
    eval_forecast(fwd_test_a, pred_garch_clean[:min_len], "GARCH(1,1)"),
    eval_forecast(fwd_test_a, vix_test[:min_len], "VIX (market)"),
    eval_forecast(fwd_test_a, sf_21day[:min_len], "SF-Harris"),
]

print(f"\n  {'Model':>15}  {'RMSE':>10}  {'MAE':>10}  {'Corr':>7}  {'R2':>7}  {'QLIKE':>8}  {'N':>5}")
print(f"  {'-'*70}")
for r in results:
    if not np.isnan(r['RMSE']):
        print(f"  {r['name']:>15}  {r['RMSE']:>10.6f}  {r['MAE']:>10.6f}  {r['Corr']:>7.4f}  "
              f"{r['R2']:>7.4f}  {r['QLIKE']:>8.4f}  {r['N']:>5}")

# VIX as variance swap
print(f"\n{'='*70}")
print("VARIANCE SWAP ANALYSIS")
print("=" * 70)

# VIX variance risk premium
vix_corr = np.corrcoef(vix_v[:len(daily_rv_v)], daily_rv_v)[0, 1]
print(f"\n  VIX vs contemporaneous RV:  corr = {vix_corr:.4f}")

# VIX vs forward RV
valid_fwd = np.isfinite(fwd_rv_v) & np.isfinite(vix_v[:len(fwd_rv_v)])
if np.sum(valid_fwd) > 100:
    corr_fwd = np.corrcoef(fwd_rv_v[valid_fwd], vix_v[:len(fwd_rv_v)][valid_fwd])[0, 1]
    print(f"  VIX vs 21-day forward RV:   corr = {corr_fwd:.4f}")

# Variance risk premium
vrp = vix_v[:len(daily_rv_v)] - daily_rv_v[:len(vix_v)]
print(f"\n  Variance Risk Premium (VIX_var - actual RV):")
print(f"    Mean VRP: {np.mean(vrp):.6f} ({np.mean(vrp)/np.mean(daily_rv_v[:len(vrp)])*100:.1f}% of RV)")
print(f"    Std VRP:  {np.std(vrp):.6f}")
print(f"    t-stat:   {stats.ttest_1samp(vrp, 0)[0]:.2f} (p={stats.ttest_1samp(vrp, 0)[1]:.4f})")
print(f"    VRP > 0:  {np.mean(vrp > 0):.1%} of days")

# SF-Harris vs VIX
print(f"\n  SF-Harris vs VIX as variance forecaster:")
sf_corr = np.corrcoef(sf_1day, daily_rv_v[split:split+len(sf_1day)])[0, 1]
print(f"    SF-Harris corr with RV: {sf_corr:.4f}")
print(f"    VIX corr with RV:      {vix_corr:.4f}")
print(f"    SF-Harris gives PHYSICAL measure (objective P)")
print(f"    VIX gives RISK-NEUTRAL measure (market Q)")
print(f"    The gap is the variance risk premium (VRP).")
print(f"    For option pricing, you need Q, not P.")