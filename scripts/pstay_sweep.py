"""Sweep P(stay) values for the coin flip model on log-RV.

Tests:
1. Fixed P(stay) from 0.0 to 0.99 — how does AAD change?
2. Time-varying P(stay) — higher persistence in calm periods, lower in volatile
3. P(stay) as function of recent volatility
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from anzarut_replication import (
    load_ibm_data, compute_15min_returns, detect_and_remove_jumps,
    estimate_periodicity, compute_coverage,
)

PROB_LEVELS = [0.25, 0.50, 0.75, 0.85, 0.90, 0.95]
N_SIM = 3000

# Fixed P_STAY values to sweep
P_STAY_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.95, 0.97, 0.99]

print("=" * 70)
print("P(STAY) SWEEP: HOW DOES THE COIN FLIP MASS POINT AFFECT COVERAGE?")
print("=" * 70)
print()

# =============================================================================
# 1. Load data
# =============================================================================
print("Loading IBM data...")
df = load_ibm_data()
returns = compute_15min_returns(df)
returns_clean = detect_and_remove_jumps(returns, n_passes=2, top_pct=0.001)
periodicity = estimate_periodicity(returns)

rv_df = returns_clean.to_frame("return")
rv_df["rv_15min"] = rv_df["return"] ** 2
rv_df["time"] = rv_df.index.strftime("%H:%M")
rv_df["f_t"] = rv_df["time"].map(periodicity).fillna(1.0)
rv_df["rv_adj"] = rv_df["rv_15min"] / rv_df["f_t"]
rv_df["log_spot"] = np.log(rv_df["rv_adj"].clip(lower=1e-20))

mask = np.isfinite(rv_df["log_spot"])
log_spot = rv_df.loc[mask, "log_spot"].values
raw_returns = rv_df.loc[mask, "return"].values

n = len(log_spot)
split = int(n * 0.8)
train_logrv = log_spot[:split]
test_logrv = log_spot[split:]
train_ret = raw_returns[:split]
test_ret = raw_returns[split:]
n_test = len(test_logrv)

print(f"  15-min observations: {n}")
print(f"  Train: {split}, Test: {n_test}")

# =============================================================================
# 2. Sweep P(stay) for log-RV coverage (coin flip with obs noise)
# =============================================================================
print("\n" + "=" * 70)
print("FIXED P(STAY) SWEEP — LOG-RV COVERAGE")
print("=" * 70)

# Observation noise estimate
sigma_obs = np.std(train_logrv[1:] - train_logrv[:-1]) / np.sqrt(2)

rng = np.random.default_rng(42)
results_logrv = {}

for p_stay in P_STAY_VALUES:
    scenarios = np.zeros((N_SIM, n_test))
    for t in range(n_test):
        idx = split + t
        stay_mask = rng.random(N_SIM) < p_stay
        jump_draws = rng.choice(train_logrv, size=N_SIM)
        tau = np.where(stay_mask, log_spot[idx - 1], jump_draws)
        scenarios[:, t] = rng.normal(tau, sigma_obs, size=N_SIM)

    cov = compute_coverage(test_logrv, scenarios, PROB_LEVELS)
    aad = np.mean([abs(cov[p] - p * 100) for p in PROB_LEVELS])
    results_logrv[p_stay] = {"coverage": cov, "aad": aad}

    dev_str = "  ".join([f"p={p:.2f}:dev={cov[p]-p*100:+.1f}" for p in [0.25, 0.50, 0.75, 0.95]])
    print(f"  P(stay)={p_stay:.2f}: AAD={aad:.2f}pp  |  {dev_str}")

# =============================================================================
# 3. Sweep P(stay) WITHOUT observation noise
# =============================================================================
print("\n" + "=" * 70)
print("FIXED P(STAY) SWEEP — LOG-RV COVERAGE (NO OBS NOISE)")
print("=" * 70)

rng2 = np.random.default_rng(42)
results_logrv_nonoise = {}

for p_stay in P_STAY_VALUES:
    scenarios = np.zeros((N_SIM, n_test))
    for t in range(n_test):
        idx = split + t
        stay_mask = rng2.random(N_SIM) < p_stay
        jump_draws = rng2.choice(train_logrv, size=N_SIM)
        scenarios[:, t] = np.where(stay_mask, log_spot[idx - 1], jump_draws)

    cov = compute_coverage(test_logrv, scenarios, PROB_LEVELS)
    aad = np.mean([abs(cov[p] - p * 100) for p in PROB_LEVELS])
    results_logrv_nonoise[p_stay] = {"coverage": cov, "aad": aad}

    print(f"  P(stay)={p_stay:.2f}: AAD={aad:.2f}pp")

# =============================================================================
# 4. Time-varying P(stay)
# =============================================================================
print("\n" + "=" * 70)
print("TIME-VARYING P(STAY) — LOG-RV COVERAGE")
print("=" * 70)

# Compute rolling volatility for time-varying P(stay)
window = 26  # ~6.5 hours of 15-min bars
rolling_std = pd.Series(log_spot).rolling(window, min_periods=1).std().values
rolling_mean = pd.Series(log_spot).rolling(window, min_periods=1).mean().values

# Normalize volatility regime
vol_regime = np.where(np.isfinite(rolling_std), rolling_std, np.nan)
vol_regime_norm = (vol_regime - np.nanmean(vol_regime[split:])) / np.nanstd(vol_regime[split:])

# Time-varying models
tv_models = {}

# Model 1: P(stay) = 0.88 baseline, adjust ±0.1 based on volatility
# Low vol -> high P(stay), High vol -> low P(stay)
p_stay_vol = np.clip(0.88 - 0.1 * vol_regime_norm, 0.1, 0.99)
tv_models["P=0.88±0.1·vol"] = p_stay_vol

# Model 2: More aggressive adjustment
p_stay_vol2 = np.clip(0.88 - 0.2 * vol_regime_norm, 0.1, 0.99)
tv_models["P=0.88±0.2·vol"] = p_stay_vol2

# Model 3: Time-of-day P(stay)
# Higher persistence at open/close, lower in middle of day
hours = rv_df.index.hour
minutes = rv_df.index.minute
time_of_day = hours + minutes / 60
# U-shape: high at open (9.5) and close (16), low at midday (12-13)
p_stay_tod = 0.88 + 0.05 * np.cos(2 * np.pi * (time_of_day - 12) / 6.5)
p_stay_tod = np.clip(p_stay_tod, 0.1, 0.99)
tv_models["P=0.88+0.05·cos(tod)"] = p_stay_tod

# Model 4: P(stay) from ACF at different lags
# Use rolling ACF(1) as P(stay)
rolling_acf = np.full(n, np.nan)
for t in range(window, n):
    segment = log_spot[t-window:t]
    if np.std(segment) > 0:
        rolling_acf[t] = np.corrcoef(segment[:-1], segment[1:])[0, 1]
    else:
        rolling_acf[t] = 0.5

rolling_acf = np.where(np.isfinite(rolling_acf), rolling_acf, 0.5)
p_stay_acf = np.clip(rolling_acf, 0.01, 0.99)
tv_models["P=rolling_ACF(1)"] = p_stay_acf

rng3 = np.random.default_rng(42)
results_tv = {}

for name, p_stay_arr in tv_models.items():
    scenarios = np.zeros((N_SIM, n_test))
    for t in range(n_test):
        idx = split + t
        p = p_stay_arr[idx]
        stay_mask = rng3.random(N_SIM) < p
        jump_draws = rng3.choice(train_logrv, size=N_SIM)
        tau = np.where(stay_mask, log_spot[idx - 1], jump_draws)
        scenarios[:, t] = rng.normal(tau, sigma_obs, size=N_SIM)

    cov = compute_coverage(test_logrv, scenarios, PROB_LEVELS)
    aad = np.mean([abs(cov[p] - p * 100) for p in PROB_LEVELS])
    results_tv[name] = {"coverage": cov, "aad": aad}
    print(f"  {name}: AAD={aad:.2f}pp")
    for p in PROB_LEVELS:
        print(f"    p={p:.2f}: cov={cov[p]:.1f}% (dev={cov[p]-p*100:+.1f}pp)")

# =============================================================================
# 5. Also sweep P(stay) for RETURN prediction (similar-vol coin flip)
# =============================================================================
print("\n" + "=" * 70)
print("FIXED P(STAY) SWEEP — RETURN COVERAGE (similar-vol coin flip)")
print("=" * 70)

# Build volatility buckets
n_buckets = 20
logrv_min, logrv_max = train_logrv.min(), train_logrv.max()
bucket_edges = np.linspace(logrv_min, logrv_max, n_buckets + 1)
bucket_indices = np.digitize(train_logrv, bucket_edges) - 1
bucket_indices = np.clip(bucket_indices, 0, n_buckets - 1)

bucket_returns = {}
for b in range(n_buckets):
    mask_b = bucket_indices == b
    if mask_b.sum() > 0:
        bucket_returns[b] = raw_returns[:split][mask_b]
    else:
        bucket_returns[b] = train_ret

rng4 = np.random.default_rng(42)
results_ret = {}

for p_stay in P_STAY_VALUES:
    scenarios = np.zeros((N_SIM, n_test))
    for t in range(n_test):
        idx = split + t
        stay_mask = rng4.random(N_SIM) < p_stay
        current_bucket = np.clip(np.digitize(log_spot[idx-1], bucket_edges) - 1, 0, n_buckets-1)
        similar_returns = bucket_returns[current_bucket]
        stay_draws = rng4.choice(similar_returns, size=N_SIM)
        jump_draws = rng4.choice(train_ret, size=N_SIM)
        scenarios[:, t] = np.where(stay_mask, stay_draws, jump_draws)

    cov = compute_coverage(test_ret, scenarios, PROB_LEVELS)
    aad = np.mean([abs(cov[p] - p * 100) for p in PROB_LEVELS])
    results_ret[p_stay] = {"coverage": cov, "aad": aad}
    print(f"  P(stay)={p_stay:.2f}: AAD={aad:.2f}pp")

# =============================================================================
# 6. Summary table
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY: P(STAY) SWEEP")
print("=" * 70)

print("\n--- Log-RV coverage (with obs noise) ---")
print(f"  {'P(stay)':>8}  {'AAD':>8}  {'vs P=0':>8}  {'vs P=0.88':>10}  {'vs HistSim':>10}")
print("-" * 55)
histsim_aad_logrv = results_logrv[0.0]["aad"]
for p_stay in P_STAY_VALUES:
    aad = results_logrv[p_stay]["aad"]
    vs_p0 = f"{aad - histsim_aad_logrv:+.2f}"
    vs_p88 = f"{aad - results_logrv[0.88]['aad']:+.2f}"
    vs_hist = f"{aad - 0.52:+.2f}"  # HistSim reference
    best_p = min(results_logrv, key=lambda k: results_logrv[k]["aad"])
    marker = " <-- best" if p_stay == best_p else ""
    print(f"  {p_stay:>8.2f}  {aad:>6.2f}pp  {vs_p0:>8}  {vs_p88:>10}  {vs_hist:>10}{marker}")

print("\n--- Log-RV coverage (no obs noise) ---")
print(f"  {'P(stay)':>8}  {'AAD':>8}")
print("-" * 25)
for p_stay in P_STAY_VALUES:
    aad = results_logrv_nonoise[p_stay]["aad"]
    print(f"  {p_stay:>8.2f}  {aad:>6.2f}pp")

print("\n--- Return coverage (similar-vol coin flip) ---")
print(f"  {'P(stay)':>8}  {'AAD':>8}  {'vs P=0':>8}")
print("-" * 30)
ret_histsim_aad = results_ret[0.0]["aad"]
for p_stay in P_STAY_VALUES:
    aad = results_ret[p_stay]["aad"]
    vs_p0 = f"{aad - ret_histsim_aad:+.2f}"
    best_ret_p = min(results_ret, key=lambda k: results_ret[k]["aad"])
    marker = " <-- best" if p_stay == best_ret_p else ""
    print(f"  {p_stay:>8.2f}  {aad:>6.2f}pp  {vs_p0:>8}{marker}")

print("\n--- Time-varying P(stay) ---")
print(f"  {'Model':>25}  {'AAD':>8}")
print("-" * 40)
for name, res in sorted(results_tv.items(), key=lambda x: x[1]["aad"]):
    print(f"  {name:>25}  {res['aad']:>6.2f}pp")
print(f"  {'Fixed P=0.88 (reference)':>25}  {results_logrv[0.88]['aad']:>6.2f}pp")
print(f"  {'HistSim (P=0, reference)':>25}  {results_logrv[0.0]['aad']:>6.2f}pp")

# =============================================================================
# 7. Verdict
# =============================================================================
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)

best_logrv_p = min(results_logrv, key=lambda k: results_logrv[k]["aad"])
best_logrv_aad = results_logrv[best_logrv_p]["aad"]
best_ret_p = min(results_ret, key=lambda k: results_ret[k]["aad"])
best_ret_aad = results_ret[best_ret_p]["aad"]

print(f"\nBest P(stay) for log-RV: {best_logrv_p:.2f} -> AAD={best_logrv_aad:.2f}pp")
print(f"Best P(stay) for returns: {best_ret_p:.2f} -> AAD={best_ret_aad:.2f}pp")
print(f"HistSim (P=0) for log-RV: {results_logrv[0.0]['aad']:.2f}pp")
print(f"HistSim (P=0) for returns: {results_ret[0.0]['aad']:.2f}pp")

if best_logrv_p == 0.0:
    print("\nP(stay)=0 (pure HistSim) is optimal — the mass point HURTS coverage.")
    print("The coin flip model cannot improve on simple empirical resampling.")
elif best_logrv_aad < results_logrv[0.0]["aad"] - 0.5:
    print(f"\nOptimal P(stay)={best_logrv_p:.2f} improves over HistSim by "
          f"{results_logrv[0.0]['aad'] - best_logrv_aad:.2f}pp.")
else:
    print(f"\nP(stay) has minimal effect on coverage.")
    print(f"The mass point doesn't help or hurt much for the coin flip model.")

print(f"\nKey comparisons:")
print(f"  Gibbs (mass point + denoising): 0.2pp")
print(f"  HistSim (P=0, no mass point):    {results_logrv[0.0]['aad']:.2f}pp")
print(f"  Coin flip P=0.88 (mass point):  {results_logrv[0.88]['aad']:.2f}pp")
print(f"  Best coin flip P={best_logrv_p:.2f}:           {best_logrv_aad:.2f}pp")

# =============================================================================
# 8. Plot
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel 1: AAD vs P(stay) for log-RV
ax = axes[0]
p_values = sorted(results_logrv.keys())
aad_values = [results_logrv[p]["aad"] for p in p_values]
aad_nonoise = [results_logrv_nonoise[p]["aad"] for p in p_values]

ax.plot(p_values, aad_values, 'o-', color='steelblue', linewidth=2, markersize=6, label='With obs noise')
ax.plot(p_values, aad_nonoise, 's--', color='coral', linewidth=1.5, markersize=5, label='No obs noise')
ax.axhline(y=0.52, color='gray', linestyle=':', alpha=0.7, label='HistSim (0.52pp)')
ax.axhline(y=0.2, color='navy', linestyle=':', alpha=0.7, label='Gibbs (0.2pp)')
ax.axvline(x=0.88, color='green', linestyle='--', alpha=0.5, label='P(stay)=0.88')
ax.set_xlabel('P(stay)')
ax.set_ylabel('AAD (pp, lower is better)')
ax.set_title('Log-RV Coverage AAD vs P(stay)')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.set_ylim(0, min(max(aad_values + aad_nonoise) * 1.1, 60))

# Panel 2: AAD vs P(stay) for returns
ax = axes[1]
ret_aad = [results_ret[p]["aad"] for p in p_values]
ax.plot(p_values, ret_aad, 'o-', color='seagreen', linewidth=2, markersize=6)
ax.axhline(y=2.91, color='gray', linestyle=':', alpha=0.7, label='HistSim returns (2.91pp)')
ax.axvline(x=0.88, color='green', linestyle='--', alpha=0.5, label='P(stay)=0.88')
ax.set_xlabel('P(stay)')
ax.set_ylabel('AAD (pp, lower is better)')
ax.set_title('Return Coverage AAD vs P(stay)')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Panel 3: Coverage profiles for select P(stay) values
ax = axes[2]
for p_stay in [0.0, 0.5, 0.88, 0.95]:
    cov = results_logrv[p_stay]["coverage"]
    ideal = [p * 100 for p in PROB_LEVELS]
    deviations = [cov[p] - p * 100 for p in PROB_LEVELS]
    ax.plot(PROB_LEVELS, deviations, 'o-', label=f'P(stay)={p_stay:.2f}', linewidth=1.5)

ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_xlabel('Probability level')
ax.set_ylabel('Coverage deviation (pp)')
ax.set_title('Coverage Profile: P(stay) Sweep (log-RV)')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('pstay_sweep.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved to pstay_sweep.png")