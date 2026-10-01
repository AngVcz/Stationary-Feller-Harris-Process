"""Coverage decomposition: what drives the AAD?

Is it empirical Q doing all the work, or does SF-Harris contribute?

Models tested:
1. SF-Harris + empirical Q (full model)
2. iid bootstrap from training returns (no SF-Harris, no tau*)
3. Bootstrap scaled by tau* (empirical shape + variance scaling)
4. N(0, sigma^2) constant (no tau*, Gaussian)
5. N(0, tau*) Gaussian tails (tau* but no empirical shape)
6. iid bootstrap from TEST returns (oracle distribution)
7. N(0, actual RV) oracle variance + Gaussian
8. Bootstrap scaled by actual RV (oracle everything)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from anzarut_replication import (
    load_ibm_data, compute_15min_returns, detect_and_remove_jumps,
    fit_jump_thresholds, split_returns_by_date,
    estimate_alpha, gibbs_gig_harris, simulate_predictive_sf_harris, compute_coverage
)

PROB_LEVELS = [0.25, 0.50, 0.75, 0.85, 0.90, 0.95]

df = load_ibm_data(start_date="1998-01-01", end_date="2026-12-31")
returns = compute_15min_returns(df)

# --- leak-free split: cut by date BEFORE cleaning; fit jump thresholds on TRAIN, apply to both ---
train_returns, test_returns = split_returns_by_date(returns, train_frac=0.8)
thresholds = fit_jump_thresholds(train_returns, n_passes=2, top_pct=0.001)
train_clean = detect_and_remove_jumps(train_returns, fixed_thresholds=thresholds)
test_clean = detect_and_remove_jumps(test_returns, fixed_thresholds=thresholds)


def _daily_rv(ret):
    """Daily realized variance (sum of r^2 per day) from an intraday return series."""
    rdf = ret.to_frame("return")
    rdf["rv_15min"] = rdf["return"]**2
    rdf["date"] = pd.to_datetime(rdf.index.date)
    return rdf.groupby("date")["rv_15min"].sum()


daily_rv_train = _daily_rv(train_clean)
daily_rv_train = daily_rv_train[daily_rv_train > 0]
daily_rv_test = _daily_rv(test_clean)
daily_rv_test = daily_rv_test[daily_rv_test > 0]

# close-to-close daily returns, cut at the SAME temporal boundary as the intraday split
close_daily = df["close"].resample("D").last().dropna()
daily_log_ret = np.log(close_daily).diff().dropna()
cutoff = pd.Timestamp(train_returns.index.date.max())  # last train day (inclusive)

# train window: realized variance aligned with its close-to-close return
common_train = daily_rv_train.index.intersection(daily_log_ret.index)
train_log = np.log(daily_rv_train.loc[common_train].values)
train_ret = daily_log_ret.loc[common_train].values
_finite = np.isfinite(train_log)
train_log, train_ret = train_log[_finite], train_ret[_finite]

# test window: same alignment (close-to-close return must fall AFTER the cutoff)
test_ret_idx = daily_log_ret.index[daily_log_ret.index > cutoff]
common_test = daily_rv_test.index.intersection(test_ret_idx)
test_log = np.log(daily_rv_test.loc[common_test].values)
test_ret = daily_log_ret.loc[common_test].values
_finite_t = np.isfinite(test_log)
test_log, test_ret = test_log[_finite_t], test_ret[_finite_t]
daily_rv = daily_rv_test.loc[common_test].values  # oracle realized variance (test)
n_test = len(test_log)

# SF-Harris
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
test_ret = test_ret[:n_test_actual]
tau_median = np.median(sim_rv, axis=0)
train_std = np.std(train_ret)
train_var = np.var(train_ret)

print("=" * 70)
print("COVERAGE DECOMPOSITION: WHAT DRIVES THE AAD?")
print("Is it empirical Q doing all the work?")
print("=" * 70)

# Model 1: SF-Harris + empirical Q
rng = np.random.default_rng(42)
sim1 = np.sqrt(np.maximum(sim_rv, 1e-20)) * rng.normal(size=sim_rv.shape)
cov1 = compute_coverage(test_ret, sim1, PROB_LEVELS)
aad1 = np.mean([abs(cov1[p] - p*100) for p in PROB_LEVELS])

# Model 2: iid bootstrap from training returns (no SF-Harris, no tau*)
rng2 = np.random.default_rng(43)
sim2 = rng2.choice(train_ret, size=(2000, n_test_actual), replace=True)
cov2 = compute_coverage(test_ret, sim2, PROB_LEVELS)
aad2 = np.mean([abs(cov2[p] - p*100) for p in PROB_LEVELS])

# Model 3: Bootstrap scaled by tau* (empirical shape + variance adaptation)
rng3 = np.random.default_rng(44)
sim3 = np.empty((2000, n_test_actual))
for s in range(2000):
    boot = rng3.choice(train_ret, size=n_test_actual, replace=True)
    for t in range(n_test_actual):
        sim3[s, t] = boot[t] * np.sqrt(sim_rv[s, t] / train_var)
cov3 = compute_coverage(test_ret, sim3, PROB_LEVELS)
aad3 = np.mean([abs(cov3[p] - p*100) for p in PROB_LEVELS])

# Model 4: N(0, sigma^2) constant variance
rng4 = np.random.default_rng(45)
sim4 = rng4.normal(0, train_std, size=(2000, n_test_actual))
cov4 = compute_coverage(test_ret, sim4, PROB_LEVELS)
aad4 = np.mean([abs(cov4[p] - p*100) for p in PROB_LEVELS])

# Model 5: N(0, tau*) Gaussian tails (same as model 1 - just for clarity)
# Already computed as sim1
aad5 = aad1  # N(0, tau*) with Gaussian IS model 1

# Model 6: iid bootstrap from TEST returns (oracle distribution)
rng6 = np.random.default_rng(47)
sim6 = rng6.choice(test_ret, size=(2000, n_test_actual), replace=True)
cov6 = compute_coverage(test_ret, sim6, PROB_LEVELS)
aad6 = np.mean([abs(cov6[p] - p*100) for p in PROB_LEVELS])

# Model 7: N(0, actual RV) oracle variance + Gaussian
rng7 = np.random.default_rng(48)
test_rv_aligned = daily_rv[:n_test_actual]
sim7 = np.sqrt(np.maximum(test_rv_aligned, 1e-20)) * rng7.normal(size=(2000, n_test_actual))
cov7 = compute_coverage(test_ret, sim7, PROB_LEVELS)
aad7 = np.mean([abs(cov7[p] - p*100) for p in PROB_LEVELS])

# Model 8: Bootstrap scaled by actual RV (oracle variance + empirical shape)
rng8 = np.random.default_rng(49)
sim8 = np.empty((2000, n_test_actual))
for t in range(n_test_actual):
    boot_t = rng8.choice(train_ret, size=2000, replace=True)
    sim8[:, t] = (boot_t / train_std) * np.sqrt(max(test_rv_aligned[t], 1e-20))
cov8 = compute_coverage(test_ret, sim8, PROB_LEVELS)
aad8 = np.mean([abs(cov8[p] - p*100) for p in PROB_LEVELS])

# Print results
print("\n  Model                                             AAD      Uses")
print("  " + "-" * 85)
results = [
    ("1. SF-Harris + empirical Q (OURS)", aad1, "tau* + empirical shape"),
    ("2. iid bootstrap (no tau*, no shape)", aad2, "just training dist"),
    ("3. Bootstrap x tau* (variance adapts)", aad3, "tau* + empirical shape"),
    ("4. N(0, sigma2) constant Gaussian", aad4, "just Gaussian"),
    ("5. N(0, tau*) Gaussian", aad5, "tau* + Gaussian tails"),
    ("6. Oracle: iid bootstrap from TEST", aad6, "oracle distribution"),
    ("7. Oracle: N(0, actual RV)", aad7, "oracle var + Gaussian"),
    ("8. Oracle: bootstrap x actual RV", aad8, "oracle var + emp shape"),
]
for name, aad, uses in results:
    print(f"  {name:>45}  {aad:>5.1f}pp  {uses:>25}")

print(f"\n  Per-probability coverage:")
print(f"  {'p':>6}  {'Ideal':>6}  {'SF-H+Q':>8}  {'Boot':>8}  {'N(0,s2)':>8}  {'N(0,tau*)':>10}  {'Oracle':>8}")
print(f"  " + "-" * 56)
for p in PROB_LEVELS:
    print(f"  {p:>6.2f}  {p*100:>5.0f}%  {cov1[p]:>7.1f}%  {cov2[p]:>7.1f}%  {cov4[p]:>7.1f}%  {cov1[p]:>9.1f}%  {cov7[p]:>7.1f}%")

# Decomposition
print(f"\n{'='*70}")
print("DECOMPOSITION: HOW MUCH DOES EACH COMPONENT CONTRIBUTE?")
print("=" * 70)
print(f"\n  Total AAD improvement over N(0, sigma^2): {aad4:.1f} -> {aad1:.1f}pp = {aad4-aad1:.1f}pp")
print(f"\n  Breaking it down:")
print(f"    tau* (variance adaptation):    N(0,s2)={aad4:.1f} -> N(0,tau*)={aad5:.1f}pp = {aad4-aad5:.1f}pp improvement")
print(f"    Empirical shape:               N(0,tau*)={aad5:.1f} -> SF+Q={aad1:.1f}pp = {aad5-aad1:.1f}pp improvement")
print(f"    iid bootstrap (no adaptation): {aad2:.1f}pp")
print(f"\n  Key insight:")
if aad2 < aad4:
    print(f"    iid bootstrap ({aad2:.1f}pp) BEATS constant Gaussian ({aad4:.1f}pp)")
    print(f"    -> Empirical tails matter MORE than variance adaptation")
else:
    print(f"    N(0,sigma2) ({aad4:.1f}pp) beats iid bootstrap ({aad2:.1f}pp)")
    print(f"    -> Variance adaptation matters more than empirical tails")

if abs(aad1 - aad3) < 0.5:
    print(f"    SF-Harris ({aad1:.1f}pp) ~= Bootstrap x tau* ({aad3:.1f}pp)")
    print(f"    -> SF-Harris temporal structure adds NOTHING beyond tau* + empirical shape")
else:
    print(f"    SF-Harris ({aad1:.1f}pp) vs Bootstrap x tau* ({aad3:.1f}pp)")
    print(f"    -> SF-Harris temporal structure adds {aad3-aad1:.1f}pp")

print(f"\n  Oracle bounds:")
print(f"    Best possible (oracle var + emp shape): {aad8:.1f}pp")
print(f"    Oracle var + Gaussian: {aad7:.1f}pp")
print(f"    Oracle distribution (no var adapt): {aad6:.1f}pp")