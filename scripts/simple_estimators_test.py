"""Simple Variance Estimators vs Gibbs Sampler at 15-min Intraday Level.

Tests whether simpler variance estimators can match the 0.2pp AAD achieved
by the full SF-Harris Gibbs pipeline, proving (or disproving) that the
Harris chain dynamics are necessary.

Three simple estimators replace the Gibbs sampler:
1. EWMA of log(RV): tau*_t = lambda * log(RV_t) + (1-lambda) * tau*_{t-1}
2. Rolling mean of log(RV): tau*_t = mean(log(RV_{t-W:t}))
3. Naive persistence: tau*_t = log(RV_{t-1})  (yesterday's variance)

Each uses the same prediction formula:
  log(H_{t+1}) | tau*_t ~ N(tau*_t, sigma^2)  with sigma^2 estimated from residuals

Plus two baselines:
4. iid bootstrap (Historical Simulation at 15-min)
5. Full SF-Harris Gibbs (from existing results)

Evaluation: coverage AAD at standard probability levels on 15-min spot volatility.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from anzarut_replication import (
    load_ibm_data, compute_15min_returns, detect_and_remove_jumps,
    estimate_periodicity, compute_coverage,
)

PROB_LEVELS = [0.25, 0.50, 0.75, 0.85, 0.90, 0.95]
ANZARUT_TABLE3 = {0.25: 25, 0.50: 51, 0.75: 75, 0.85: 84, 0.90: 89, 0.95: 93}

print("=" * 70)
print("SIMPLE VARIANCE ESTIMATORS vs GIBBS SAMPLER")
print("=" * 70)
print()
print("Testing: Can simpler variance estimators match the Gibbs sampler's 0.2pp AAD?")
print("If yes, the Harris chain dynamics are not necessary — only variance adaptation matters.")
print()

# =============================================================================
# 1. Load data and compute 15-min spot volatility
# =============================================================================
print("Loading IBM data...")
df = load_ibm_data()
returns = compute_15min_returns(df)
returns_clean = detect_and_remove_jumps(returns, n_passes=2, top_pct=0.001)
periodicity = estimate_periodicity(returns)

# Compute spot volatility
rv_df = returns_clean.to_frame("return")
rv_df["rv_15min"] = rv_df["return"] ** 2
rv_df["time"] = rv_df.index.strftime("%H:%M")
rv_df["f_t"] = rv_df["time"].map(periodicity).fillna(1.0)
rv_df["rv_adj"] = rv_df["rv_15min"] / rv_df["f_t"]
rv_df["log_spot"] = np.log(rv_df["rv_adj"].clip(lower=1e-20))

mask = np.isfinite(rv_df["log_spot"])
log_spot = rv_df.loc[mask, "log_spot"].values
print(f"  15-min observations: {len(log_spot)}")
print(f"  Log spot vol: mean={log_spot.mean():.4f}, std={log_spot.std():.4f}")

# =============================================================================
# 2. Train/test split (80/20)
# =============================================================================
n = len(log_spot)
split = int(n * 0.8)
train = log_spot[:split]
test = log_spot[split:]
print(f"  Train: {split}, Test: {n - split}")

# =============================================================================
# 3. Simple variance estimators
# =============================================================================
print("\n" + "=" * 70)
print("COMPUTING VARIANCE ESTIMATES")
print("=" * 70)

# 3a. EWMA of log(RV)
EWMA_LAMBDAS = [0.05, 0.1, 0.2, 0.3, 0.5]
ewma_estimates = {}
for lam in EWMA_LAMBDAS:
    tau = np.full(n, np.nan)
    tau[0] = train[0]
    for t in range(1, n):
        tau[t] = lam * log_spot[t - 1] + (1 - lam) * tau[t - 1]
    ewma_estimates[lam] = tau
    print(f"  EWMA(lambda={lam}): mean={np.nanmean(tau[split:]):.4f}, std={np.nanstd(tau[split:]):.4f}")

# 3b. Rolling mean of log(RV)
ROLLING_WINDOWS = [6, 12, 26, 52]  # 1.5h, 3h, 6.5h, 13h (in 15-min bars)
rolling_estimates = {}
for w in ROLLING_WINDOWS:
    tau = pd.Series(log_spot).rolling(w, min_periods=1).mean().values
    rolling_estimates[w] = tau
    print(f"  Rolling(window={w}): mean={np.nanmean(tau[split:]):.4f}, std={np.nanstd(tau[split:]):.4f}")

# 3c. Naive persistence: tau*_t = log(RV_{t-1})
tau_persist = np.roll(log_spot, 1)
tau_persist[0] = log_spot[0]
print(f"  Persistence: mean={np.nanmean(tau_persist[split:]):.4f}, std={np.nanstd(tau_persist[split:]):.4f}")

# =============================================================================
# 4. Coverage prediction using each estimator
# =============================================================================
print("\n" + "=" * 70)
print("COVERAGE PREDICTION: tau*_t -> predict log(H_{t+1})")
print("=" * 70)
print()
print("Prediction model: log(H_{t+1}) | tau*_t ~ N(tau*_t, sigma^2)")
print("where sigma^2 is the residual variance estimated from training data.")
print()


def predict_and_evaluate(tau_estimates, train_data, test_data, prob_levels, n_sim=3000, rng_seed=42):
    """Evaluate coverage using a variance estimate sequence.

    For each test point t:
      1. Use tau*_t as the predicted mean of log(H_{t+1})
      2. Estimate sigma^2 from residuals on training data
      3. Generate scenarios: log(H_{t+1}) ~ N(tau*_t, sigma^2)
      4. Compute coverage at each probability level
    """
    # Estimate residual variance from training data
    # Residual = log(RV_t) - tau*_t for training period
    train_tau = tau_estimates[:len(train_data)]
    train_clean = train_data[~np.isnan(train_tau)]
    train_tau_clean = train_tau[~np.isnan(train_tau)]
    # Align lengths
    min_len = min(len(train_clean), len(train_tau_clean))
    residuals = train_clean[:min_len] - train_tau_clean[:min_len]
    sigma2 = np.var(residuals)
    sigma = np.sqrt(sigma2) if sigma2 > 0 else 1e-6

    rng = np.random.default_rng(rng_seed)
    test_tau = tau_estimates[len(train_data):len(train_data) + len(test_data)]

    # For each test point, compute coverage
    coverage = {p: 0.0 for p in prob_levels}
    n_test = len(test_data)

    # Generate scenarios for all test points at once
    # For each test point, generate n_sim scenarios
    for t in range(n_test):
        if np.isnan(test_tau[t]):
            continue
        # Predictive distribution: N(tau*_t, sigma^2)
        scenarios = rng.normal(test_tau[t], sigma, size=n_sim)

        # Coverage: fraction of scenarios <= test_data[t]
        for p in prob_levels:
            coverage[p] += np.mean(scenarios <= test_data[t])

    # Average over test points
    n_valid = sum(1 for t in range(n_test) if not np.isnan(test_tau[t]))
    for p in prob_levels:
        coverage[p] = coverage[p] / n_valid * 100

    # Compute AAD
    aad = np.mean([abs(coverage[p] - p * 100) for p in prob_levels])

    return coverage, aad, sigma


# Also test with Empirical Q (bootstrap from training residuals)
def predict_with_empirical_q(tau_estimates, train_data, test_data, prob_levels, n_sim=3000, rng_seed=42):
    """Evaluate coverage using variance estimate + empirical residual bootstrap.

    For each test point t:
      1. Use tau*_t as the predicted mean
      2. Bootstrap residuals from training data
      3. Generate scenarios: log(H_{t+1}) = tau*_t + bootstrapped_residual
      4. Compute coverage
    """
    train_tau = tau_estimates[:len(train_data)]
    train_clean = train_data[~np.isnan(train_tau)]
    train_tau_clean = train_tau[~np.isnan(train_tau)]
    min_len = min(len(train_clean), len(train_tau_clean))
    residuals = train_clean[:min_len] - train_tau_clean[:min_len]

    rng = np.random.default_rng(rng_seed)
    test_tau = tau_estimates[len(train_data):len(train_data) + len(test_data)]

    coverage = {p: 0.0 for p in prob_levels}
    n_test = len(test_data)

    for t in range(n_test):
        if np.isnan(test_tau[t]):
            continue
        # Bootstrap residuals
        idx = rng.integers(0, len(residuals), size=n_sim)
        scenarios = test_tau[t] + residuals[idx]

        for p in prob_levels:
            coverage[p] += np.mean(scenarios <= test_data[t])

    n_valid = sum(1 for t in range(n_test) if not np.isnan(test_tau[t]))
    for p in prob_levels:
        coverage[p] = coverage[p] / n_valid * 100

    aad = np.mean([abs(coverage[p] - p * 100) for p in prob_levels])
    return coverage, aad


# Also: pure Historical Simulation (no variance adaptation)
def predict_hist_sim(train_data, test_data, prob_levels, n_sim=3000, rng_seed=42):
    """Pure iid bootstrap from training data. No variance adaptation."""
    rng = np.random.default_rng(rng_seed)
    coverage = {p: 0.0 for p in prob_levels}
    n_test = len(test_data)

    for t in range(n_test):
        idx = rng.integers(0, len(train_data), size=n_sim)
        scenarios = train_data[idx]

        for p in prob_levels:
            coverage[p] += np.mean(scenarios <= test_data[t])

    for p in prob_levels:
        coverage[p] = coverage[p] / n_test * 100

    aad = np.mean([abs(coverage[p] - p * 100) for p in prob_levels])
    return coverage, aad


# =============================================================================
# 5. Run all models
# =============================================================================
results = []

# --- EWMA models ---
print("\n--- EWMA models ---")
for lam in EWMA_LAMBDAS:
    cov, aad, sigma = predict_and_evaluate(ewma_estimates[lam], train, test, PROB_LEVELS)
    results.append(("EWMA_Gauss", f"lam={lam}", cov, aad, sigma))
    cov_eq, aad_eq = predict_with_empirical_q(ewma_estimates[lam], train, test, PROB_LEVELS)
    results.append(("EWMA_EmpQ", f"lam={lam}", cov_eq, aad_eq, None))
    print(f"  EWMA(lam={lam}): AAD_Gauss={aad:.2f}pp, AAD_EmpQ={aad_eq:.2f}pp, sigma={sigma:.4f}")

# --- Rolling mean models ---
print("\n--- Rolling mean models ---")
for w in ROLLING_WINDOWS:
    cov, aad, sigma = predict_and_evaluate(rolling_estimates[w], train, test, PROB_LEVELS)
    results.append(("RollMean_Gauss", f"w={w}", cov, aad, sigma))
    cov_eq, aad_eq = predict_with_empirical_q(rolling_estimates[w], train, test, PROB_LEVELS)
    results.append(("RollMean_EmpQ", f"w={w}", cov_eq, aad_eq, None))
    print(f"  RollMean(w={w}): AAD_Gauss={aad:.2f}pp, AAD_EmpQ={aad_eq:.2f}pp, sigma={sigma:.4f}")

# --- Persistence model ---
print("\n--- Persistence model ---")
cov, aad, sigma = predict_and_evaluate(tau_persist, train, test, PROB_LEVELS)
results.append(("Persist_Gauss", "tau*=RV_{t-1}", cov, aad, sigma))
cov_eq, aad_eq = predict_with_empirical_q(tau_persist, train, test, PROB_LEVELS)
results.append(("Persist_EmpQ", "tau*=RV_{t-1}", cov_eq, aad_eq, None))
print(f"  Persistence: AAD_Gauss={aad:.2f}pp, AAD_EmpQ={aad_eq:.2f}pp, sigma={sigma:.4f}")

# --- Historical Simulation (no adaptation) ---
print("\n--- Historical Simulation baseline ---")
cov_hist, aad_hist = predict_hist_sim(train, test, PROB_LEVELS)
results.append(("HistSim", "iid bootstrap", cov_hist, aad_hist, None))
print(f"  HistSim: AAD={aad_hist:.2f}pp")

# --- Anzarut reference ---
anz_aad = np.mean([abs(ANZARUT_TABLE3[p] - p * 100) for p in PROB_LEVELS])

# =============================================================================
# 6. Summary table
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY: SIMPLE VARIANCE ESTIMATORS vs GIBBS SAMPLER")
print("=" * 70)
print()
print(f"{'Model':>20}  {'Param':>15}  {'AAD_Gauss':>10}  {'AAD_EmpQ':>10}  {'sigma':>8}")
print("-" * 70)

for model, param, cov, aad, sigma in results:
    sigma_str = f"{sigma:.4f}" if sigma is not None else "N/A"
    # Find EmpQ result for same model/param
    empq_aad = "—"
    for m2, p2, c2, a2, s2 in results:
        if m2 == model.replace("_Gauss", "_EmpQ") and p2 == param:
            empq_aad = f"{a2:.2f}pp"
            break
    if "_Gauss" in model:
        print(f"{model:>20}  {param:>15}  {aad:>8.2f}pp  {empq_aad:>10}  {sigma_str:>8}")

print("-" * 70)
print(f"{'HistSim':>20}  {'iid bootstrap':>15}  {'—':>10}  {aad_hist:>8.2f}pp  {'N/A':>8}")
print(f"{'Gibbs (eps=0.1)':>20}  {'dollar bars':>15}  {'—':>10}  {'0.2pp':>10}  {'ref':>8}")
print(f"{'Gibbs (eps=1e-5)':>20}  {'15min':>15}  {'—':>10}  {'0.5pp':>10}  {'ref':>8}")
print(f"{'Anzarut (GIG)':>20}  {'15min':>15}  {'—':>10}  {'0.8pp':>10}  {'ref':>8}")

# =============================================================================
# 7. Detailed coverage table for best simple model
# =============================================================================
print("\n" + "=" * 70)
print("DETAILED COVERAGE: BEST SIMPLE MODELS vs GIBBS vs HISTSIM")
print("=" * 70)

# Find best EWMA and Rolling mean (Gaussian version)
best_ewma = min([(model, param, aad) for model, param, cov, aad, sigma in results
                  if model == "EWMA_Gauss"], key=lambda x: x[2])
best_roll = min([(model, param, aad) for model, param, cov, aad, sigma in results
                  if model == "RollMean_Gauss"], key=lambda x: x[2])

# Find best EWMA and Rolling mean (EmpQ version)
best_ewma_eq = min([(model, param, aad) for model, param, cov, aad, sigma in results
                     if model == "EWMA_EmpQ"], key=lambda x: x[2])
best_roll_eq = min([(model, param, aad) for model, param, cov, aad, sigma in results
                     if model == "RollMean_EmpQ"], key=lambda x: x[2])

# Get full coverage for best models
best_ewma_lam = best_ewma[1].split("=")[1]
best_ewma_cov = dict((p, v) for m, p, c, v, s in results
                      if m == "EWMA_Gauss" and p == best_ewma[1])

best_roll_w = best_roll[1].split("=")[1]
best_roll_cov = dict((p, v) for m, p, c, v, s in results
                      if m == "RollMean_Gauss" and p == best_roll[1])

# Persistence
persist_cov = dict((p, v) for m, p, c, v, s in results
                    if m == "Persist_Gauss")

print(f"\n{'p':>6}  {'Ideal':>6}  {'EWMA':>8}  {'RollMean':>10}  {'Persist':>10}  {'HistSim':>10}  {'Anzarut':>10}")
print("-" * 65)
for p in PROB_LEVELS:
    ewma_c = best_ewma_cov.get(p, 0)
    roll_c = best_roll_cov.get(p, 0)
    pers_c = persist_cov.get(p, 0)
    hist_c = cov_hist.get(p, 0)
    anz_c = ANZARUT_TABLE3[p]
    print(f"{p:>6.2f}  {p*100:>5.0f}%  {ewma_c:>7.1f}%  {roll_c:>9.1f}%  {pers_c:>9.1f}%  {hist_c:>9.1f}%  {anz_c:>9.0f}%")

print(f"\n  AAD:  EWMA={best_ewma[2]:.2f}pp  RollMean={best_roll[2]:.2f}pp  "
      f"Persist={persist_cov.get('aad', 0):.2f}pp  HistSim={aad_hist:.2f}pp  Anzarut=0.8pp")

# =============================================================================
# 8. Fill the comparison table
# =============================================================================
print("\n" + "=" * 70)
print("COMPLETE COMPARISON TABLE")
print("=" * 70)
print()
print(f"{'Vehicle for tau*':>30}  {'Method':>20}  {'AAD (15-min)':>13}  {'Source':>15}")
print("-" * 80)

# Collect all results
all_results = {}
for model, param, cov, aad, sigma in results:
    key = f"{model}({param})"
    all_results[key] = (model, param, aad, sigma)

# Sort by AAD
sorted_results = sorted(all_results.items(), key=lambda x: x[1][2])

print(f"{'Harris chain (Gibbs, eps=0.1)':>30}  {'dollar bars':>20}  {'0.2pp':>13}  {'our best':>15}")
print(f"{'Harris chain (Gibbs, eps=0.1)':>30}  {'15min bars':>20}  {'0.3pp':>13}  {'our result':>15}")
print(f"{'Harris chain (Gibbs, eps=1e-5)':>30}  {'15min bars':>20}  {'0.5pp':>13}  {'our result':>15}")
print(f"{'GIG Q (Anzarut)':>30}  {'15min bars':>20}  {'0.8pp':>13}  {'reference':>15}")
print()

for key, (model, param, aad, sigma) in sorted_results:
    sigma_str = f", sigma={sigma:.4f}" if sigma else ""
    print(f"  {key:>28}  {aad:>18.2f}pp{sigma_str}")

print()
print(f"  {'HistSim (no tau*)':>28}  {aad_hist:>18.2f}pp")
print(f"  {'GARCH(1,1)':>28}  {'3.4pp':>18}")
print(f"  {'N(0, sigma^2) constant':>28}  {'9.4pp':>18}")

# =============================================================================
# 9. Plot
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Panel 1: AAD comparison bar chart
ax = axes[0]
model_names = []
aad_values = []
colors = []

reference_models = [
    ("Gibbs (dollar)", 0.2, "steelblue"),
    ("Gibbs (15min)", 0.3, "royalblue"),
    ("Anzarut (GIG)", 0.8, "navy"),
]

# Add best simple models
for model, param, cov, aad, sigma in results:
    if "_Gauss" in model and aad < 5.0:  # only show reasonable models
        short_name = model.replace("_Gauss", "").replace("RollMean", "Roll")
        model_names.append(f"{short_name}({param})")
        aad_values.append(aad)
        colors.append("seagreen" if "EWMA" in model else "lightgreen")

model_names.extend(["HistSim", "Gibbs (dollar)", "Gibbs (15min)", "Anzarut (GIG)"])
aad_values.extend([aad_hist, 0.2, 0.3, 0.8])
colors.extend(["gray", "steelblue", "royalblue", "navy"])

y_pos = np.arange(len(model_names))
ax.barh(y_pos, aad_values, color=colors, height=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(model_names, fontsize=9)
ax.set_xlabel('AAD (pp, lower is better)')
ax.set_title('Coverage AAD: Simple Estimators vs Gibbs Sampler')
ax.axvline(x=0.2, color='steelblue', linewidth=1, linestyle='--', alpha=0.5, label='Best Gibbs')
ax.axvline(x=0.8, color='navy', linewidth=1, linestyle='--', alpha=0.5, label='Anzarut')
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis='x')

# Panel 2: tau* over time for different estimators
ax = axes[1]
t_range = np.arange(split, min(split + 500, n))  # show 500 test points
ax.plot(t_range, log_spot[split:split + 500], color='black', linewidth=0.3, alpha=0.5, label='Actual log(RV)')

best_lam = float(best_ewma[1].split("=")[1])
ax.plot(t_range, ewma_estimates[best_lam][split:split + 500], color='seagreen', linewidth=1, alpha=0.8,
        label=f'EWMA(lam={best_lam})')

best_w = int(best_roll[1].split("=")[1])
ax.plot(t_range, rolling_estimates[best_w][split:split + 500], color='coral', linewidth=1, alpha=0.8,
        label=f'RollMean(w={best_w})')

ax.plot(t_range, tau_persist[split:split + 500], color='gold', linewidth=0.5, alpha=0.5, label='Persistence')

ax.set_xlabel('Time (15-min bars)')
ax.set_ylabel('log(RV)')
ax.set_title('Variance Estimators vs Actual log(RV)')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('simple_estimators_comparison.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved to simple_estimators_comparison.png")

# =============================================================================
# 10. Verdict
# =============================================================================
print("\n" + "=" * 70)
print("VERDICT: IS THE HARRIS CHAIN NECESSARY?")
print("=" * 70)
print()

best_simple_aad = min([aad for model, param, cov, aad, sigma in results])
best_simple_name = min([(model, param, aad) for model, param, cov, aad, sigma in results],
                       key=lambda x: x[2])

print(f"Best simple estimator: {best_simple_name[0]}({best_simple_name[1]}) = {best_simple_name[2]:.2f}pp")
print(f"Gibbs sampler (dollar bars): 0.2pp")
print(f"Gibbs sampler (15min bars): 0.3pp")
print(f"Anzarut (GIG+Gibbs): 0.8pp")
print(f"HistSim (no tau*): {aad_hist:.2f}pp")
print()

if best_simple_name[2] < 0.5:
    print("The simple estimator MATCHES or BEATS the Gibbs sampler.")
    print("The Harris chain dynamics are NOT necessary for coverage prediction.")
    print("Variance adaptation (knowing current volatility) is what matters,")
    print("and a simple EWMA or rolling mean provides it just as well.")
elif best_simple_name[2] < 1.0:
    print("The simple estimator is close to but WORSE than the Gibbs sampler.")
    print(f"Gap: {best_simple_name[2] - 0.2:.1f}pp vs 0.2pp from Gibbs.")
    print("The Harris chain provides better variance adaptation than simple estimators,")
    print("but the gap is small enough that practical implications are limited.")
else:
    print("The simple estimator is significantly WORSE than the Gibbs sampler.")
    print(f"Gap: {best_simple_name[2] - 0.2:.1f}pp vs 0.2pp from Gibbs.")
    print("The Harris chain provides substantial improvement over simple variance estimators.")