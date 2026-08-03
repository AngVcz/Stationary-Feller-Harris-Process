"""Simple Variance Estimators vs Gibbs Sampler at DAILY Level.

Same comparison as simple_estimators_test.py but for daily returns.
At the daily level, σ=2.27 (much noisier than 15-min σ=1.04),
so simple estimators might perform differently.

Tests:
1. EWMA of daily RV (various lambda)
2. Rolling mean of daily RV (various windows)
3. Naive persistence (yesterday's RV)
4. Historical Simulation (no tau*)
5. N(0, sigma^2) constant
6. Gibbs sampler (from existing results: 2.5pp)

Evaluation: coverage AAD of daily returns at standard probability levels.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROB_LEVELS = [0.25, 0.50, 0.75, 0.85, 0.90, 0.95]
TICKER = 'SPY'
WINDOW = 252
START_DATE = '2005-01-01'
END_DATE = '2026-05-01'

print("=" * 70)
print("SIMPLE VARIANCE ESTIMATORS vs HARRIS CHAIN — DAILY LEVEL")
print("=" * 70)
print()
print("At daily level: sigma(log RV) = 2.27 (noisy)")
print("At 15-min level: sigma(log RV) = 1.04 (less noisy)")
print("Question: does variance adaptation matter at daily level?")
print()

# =============================================================================
# 1. Download daily data
# =============================================================================
print("Downloading SPY daily data...")
prices = yf.download(TICKER, start=START_DATE, end=END_DATE)['Close']
if isinstance(prices, pd.DataFrame):
    prices = prices[TICKER]
returns = np.log(prices / prices.shift(1)).dropna()
rv_daily = returns.values ** 2  # daily realized variance
log_rv_daily = np.log(np.clip(rv_daily, 1e-20, None))
dates = returns.index

print(f"  Data: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
print(f"  Trading days: {len(returns)}")
print(f"  log(RV): mean={log_rv_daily.mean():.4f}, std={log_rv_daily.std():.4f}")

# =============================================================================
# 2. Train/test split (80/20)
# =============================================================================
n = len(returns)
split = int(n * 0.8)
train_ret = returns.values[:split]
test_ret = returns.values[split:]
train_logrv = log_rv_daily[:split]
test_logrv = log_rv_daily[split:]
train_rv = rv_daily[:split]
test_rv = rv_daily[split:]

print(f"  Train: {split}, Test: {n - split}")
print(f"  Train returns: mean={train_ret.mean()*252:.4f}, vol={train_ret.std()*np.sqrt(252):.4f}")
print(f"  Test  returns: mean={test_ret.mean()*252:.4f}, vol={test_ret.std()*np.sqrt(252):.4f}")

# =============================================================================
# 3. Simple variance estimators on daily log(RV)
# =============================================================================
print("\n" + "=" * 70)
print("COMPUTING DAILY VARIANCE ESTIMATES")
print("=" * 70)

# EWMA of daily RV (various lambda)
EWMA_LAMBDAS = [0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
ewma_estimates = {}
for lam in EWMA_LAMBDAS:
    tau = np.full(n, np.nan)
    tau[0] = log_rv_daily[0]
    for t in range(1, n):
        tau[t] = lam * log_rv_daily[t - 1] + (1 - lam) * tau[t - 1]
    ewma_estimates[lam] = tau
    valid = ~np.isnan(tau[split:])
    print(f"  EWMA(lam={lam}): mean={np.nanmean(tau[split:]):.4f}, "
          f"std={np.nanstd(tau[split:]):.4f}, "
          f"var_captured={np.nanstd(tau[split:])**2 / np.var(log_rv_daily[split:])**0.5:.1%}")

# Rolling mean of daily RV (various windows)
ROLLING_WINDOWS = [5, 10, 21, 42, 63, 126, 252]
rolling_estimates = {}
for w in ROLLING_WINDOWS:
    tau = pd.Series(log_rv_daily).rolling(w, min_periods=1).mean().values
    rolling_estimates[w] = tau
    print(f"  Rolling(w={w}): mean={np.nanmean(tau[split:]):.4f}, "
          f"std={np.nanstd(tau[split:]):.4f}")

# Naive persistence: tau*_t = log(RV_{t-1})
tau_persist = np.roll(log_rv_daily, 1)
tau_persist[0] = log_rv_daily[0]
print(f"  Persistence: mean={np.nanmean(tau_persist[split:]):.4f}, "
      f"std={np.nanstd(tau_persist[split:]):.4f}")

# GARCH(1,1) variance estimate
from arch import arch_model
print("\n  Fitting GARCH(1,1)...")
am = arch_model(returns.values[:split] * 100, vol='Garch', p=1, q=1, mean='Zero')
res = am.fit(disp='off')
garch_var = res.conditional_volatility / 100  # scale back
# Extend to full sample
garch_forecast_var = np.full(n, np.nan)
# Use rolling GARCH for test period
garch_var_full = np.full(n, np.nan)
for t in range(split, n):
    am_t = arch_model(returns.values[:t] * 100, vol='Garch', p=1, q=1, mean='Zero')
    try:
        res_t = am_t.fit(disp='off')
        fc = res_t.forecast(horizon=1)
        garch_var_full[t] = np.sqrt(fc.variance.values[-1, 0]) / 100
    except:
        garch_var_full[t] = garch_var_full[t-1] if t > split else np.nan
    if t % 252 == 0:
        print(f"    GARCH fitting {dates[t].strftime('%Y-%m-%d')}...")

garch_logrv = np.log(garch_var_full ** 2 + 1e-20)
print(f"  GARCH: mean={np.nanmean(garch_logrv[split:]):.4f}, "
      f"std={np.nanstd(garch_logrv[split:]):.4f}")

# =============================================================================
# 4. Coverage evaluation functions
# =============================================================================
def predict_and_evaluate_daily(tau_estimates, train_returns, test_returns, prob_levels,
                                n_sim=5000, rng_seed=42):
    """Evaluate coverage of daily returns using variance estimate.

    Prediction: r_{t+1} | tau*_t ~ N(0, exp(tau*_t))
    where tau*_t is the log-variance estimate.
    """
    rng = np.random.default_rng(rng_seed)
    test_tau = tau_estimates[split:split + len(test_returns)]

    coverage = {p: 0.0 for p in prob_levels}
    n_valid = 0

    for t in range(len(test_returns)):
        if np.isnan(test_tau[t]):
            continue
        # tau*_t is log(variance), so variance = exp(tau*_t)
        var_t = np.exp(test_tau[t])
        scenarios = rng.normal(0, np.sqrt(var_t), size=n_sim)

        for p in prob_levels:
            coverage[p] += np.mean(scenarios <= test_returns[t])
        n_valid += 1

    for p in prob_levels:
        coverage[p] = coverage[p] / n_valid * 100

    aad = np.mean([abs(coverage[p] - p * 100) for p in prob_levels])
    return coverage, aad


def predict_with_empirical_q_daily(tau_estimates, train_returns, test_returns, prob_levels,
                                     n_sim=5000, rng_seed=42):
    """Evaluate coverage using variance estimate + empirical residual bootstrap.

    Prediction: r_{t+1} = sqrt(exp(tau*_t)) * residual
    where residual is bootstrapped from standardized training residuals.
    """
    train_tau = tau_estimates[:split]
    train_rv_est = np.exp(train_tau)
    # Standardize: residual = r / sqrt(RV_estimate)
    valid = ~np.isnan(train_tau) & (train_rv_est > 0)
    residuals = train_returns[:split][valid] / np.sqrt(train_rv_est[valid])

    rng = np.random.default_rng(rng_seed)
    test_tau = tau_estimates[split:split + len(test_returns)]

    coverage = {p: 0.0 for p in prob_levels}
    n_valid = 0

    for t in range(len(test_returns)):
        if np.isnan(test_tau[t]):
            continue
        var_t = np.exp(test_tau[t])
        idx = rng.integers(0, len(residuals), size=n_sim)
        scenarios = np.sqrt(var_t) * residuals[idx]

        for p in prob_levels:
            coverage[p] += np.mean(scenarios <= test_returns[t])
        n_valid += 1

    for p in prob_levels:
        coverage[p] = coverage[p] / n_valid * 100

    aad = np.mean([abs(coverage[p] - p * 100) for p in prob_levels])
    return coverage, aad


def predict_hist_sim_daily(train_returns, test_returns, prob_levels, n_sim=5000, rng_seed=42):
    """Pure iid bootstrap from training data. No variance adaptation."""
    rng = np.random.default_rng(rng_seed)
    coverage = {p: 0.0 for p in prob_levels}

    for t in range(len(test_returns)):
        idx = rng.integers(0, len(train_returns), size=n_sim)
        scenarios = train_returns[idx]
        for p in prob_levels:
            coverage[p] += np.mean(scenarios <= test_returns[t])

    for p in prob_levels:
        coverage[p] = coverage[p] / len(test_returns) * 100

    aad = np.mean([abs(coverage[p] - p * 100) for p in prob_levels])
    return coverage, aad


def predict_constant_daily(train_returns, test_returns, prob_levels, n_sim=5000, rng_seed=42):
    """N(0, sigma^2) constant — no variance adaptation."""
    rng = np.random.default_rng(rng_seed)
    mu = np.mean(train_returns)
    sigma = np.std(train_returns)
    coverage = {p: 0.0 for p in prob_levels}

    for t in range(len(test_returns)):
        scenarios = rng.normal(mu, sigma, size=n_sim)
        for p in prob_levels:
            coverage[p] += np.mean(scenarios <= test_returns[t])

    for p in prob_levels:
        coverage[p] = coverage[p] / len(test_returns) * 100

    aad = np.mean([abs(coverage[p] - p * 100) for p in prob_levels])
    return coverage, aad

# =============================================================================
# 5. Run all models
# =============================================================================
print("\n" + "=" * 70)
print("RUNNING ALL MODELS")
print("=" * 70)

results = []

# EWMA models
print("\n--- EWMA models (daily) ---")
for lam in EWMA_LAMBDAS:
    cov, aad = predict_and_evaluate_daily(ewma_estimates[lam], train_ret, test_ret, PROB_LEVELS)
    results.append(("EWMA_N(0,tau)", f"lam={lam}", cov, aad))
    cov_eq, aad_eq = predict_with_empirical_q_daily(ewma_estimates[lam], train_ret, test_ret, PROB_LEVELS)
    results.append(("EWMA_EmpQ", f"lam={lam}", cov_eq, aad_eq))
    print(f"  EWMA(lam={lam}): AAD_N(0,tau)={aad:.2f}pp, AAD_EmpQ={aad_eq:.2f}pp")

# Rolling mean models
print("\n--- Rolling mean models (daily) ---")
for w in ROLLING_WINDOWS:
    cov, aad = predict_and_evaluate_daily(rolling_estimates[w], train_ret, test_ret, PROB_LEVELS)
    results.append(("RollMean_N(0,tau)", f"w={w}", cov, aad))
    cov_eq, aad_eq = predict_with_empirical_q_daily(rolling_estimates[w], train_ret, test_ret, PROB_LEVELS)
    results.append(("RollMean_EmpQ", f"w={w}", cov_eq, aad_eq))
    print(f"  RollMean(w={w}): AAD_N(0,tau)={aad:.2f}pp, AAD_EmpQ={aad_eq:.2f}pp")

# Persistence
print("\n--- Persistence model (daily) ---")
cov, aad = predict_and_evaluate_daily(tau_persist, train_ret, test_ret, PROB_LEVELS)
results.append(("Persist_N(0,tau)", "RV_{t-1}", cov, aad))
cov_eq, aad_eq = predict_with_empirical_q_daily(tau_persist, train_ret, test_ret, PROB_LEVELS)
results.append(("Persist_EmpQ", "RV_{t-1}", cov_eq, aad_eq))
print(f"  Persistence: AAD_N(0,tau)={aad:.2f}pp, AAD_EmpQ={aad_eq:.2f}pp")

# GARCH(1,1)
print("\n--- GARCH(1,1) model (daily) ---")
try:
    cov, aad = predict_and_evaluate_daily(garch_logrv, train_ret, test_ret, PROB_LEVELS)
    results.append(("GARCH_N(0,tau)", "GARCH(1,1)", cov, aad))
    print(f"  GARCH(1,1): AAD_N(0,tau)={aad:.2f}pp")
except Exception as e:
    print(f"  GARCH N(0,tau) failed: {e}")
    results.append(("GARCH_N(0,tau)", "GARCH(1,1)", {p: 0 for p in PROB_LEVELS}, 99.0))

try:
    cov_eq, aad_eq = predict_with_empirical_q_daily(garch_logrv, train_ret, test_ret, PROB_LEVELS)
    results.append(("GARCH_EmpQ", "GARCH(1,1)", cov_eq, aad_eq))
    print(f"  GARCH(1,1): AAD_EmpQ={aad_eq:.2f}pp")
except Exception as e:
    print(f"  GARCH EmpQ failed: {e}")
    results.append(("GARCH_EmpQ", "GARCH(1,1)", {p: 0 for p in PROB_LEVELS}, 99.0))

# Hist.Sim
print("\n--- Historical Simulation (daily) ---")
cov_hist, aad_hist = predict_hist_sim_daily(train_ret, test_ret, PROB_LEVELS)
results.append(("HistSim", "iid bootstrap", cov_hist, aad_hist))
print(f"  HistSim: AAD={aad_hist:.2f}pp")

# Constant
print("\n--- Constant N(0, sigma^2) (daily) ---")
cov_const, aad_const = predict_constant_daily(train_ret, test_ret, PROB_LEVELS)
results.append(("Constant", "N(0,sigma^2)", cov_const, aad_const))
print(f"  Constant: AAD={aad_const:.2f}pp")

# =============================================================================
# 6. Summary table
# =============================================================================
print("\n" + "=" * 70)
print("COMPLETE COMPARISON TABLE — DAILY LEVEL")
print("=" * 70)
print()
print(f"{'Vehicle for tau*':>30}  {'Method':>20}  {'AAD_N(0,tau)':>14}  {'AAD_EmpQ':>10}")
print("-" * 80)

# Group results
for model_base in ["EWMA", "RollMean", "Persist", "GARCH"]:
    for variant in ["N(0,tau)", "EmpQ"]:
        model = f"{model_base}_{variant}"
        model_results = [(param, cov, aad) for m, param, cov, aad in results if m == model]
        if model_results:
            for param, cov, aad in sorted(model_results, key=lambda x: x[2]):
                print(f"  {model:>28}({param})  {aad:>18.2f}pp")

# Baselines
print()
print(f"  {'HistSim (no tau*)':>28}  {'iid bootstrap':>20}  {'—':>14}  {aad_hist:>8.2f}pp")
print(f"  {'Constant N(0,sigma^2)':>28}  {'no adaptation':>20}  {'—':>14}  {aad_const:>8.2f}pp")
print()
print(f"  {'--- Reference results (from previous experiments):':>68}")
print(f"  {'N(0, tau*) from Harris chain':>28}  {'daily returns':>20}  {'2.0pp':>14}  {'—':>10}")
print(f"  {'iid bootstrap':>28}  {'daily returns':>20}  {'2.0pp':>14}  {'2.0pp':>10}")
print(f"  {'SF-Harris + Emission':>28}  {'daily returns':>20}  {'2.5pp':>14}  {'—':>10}")
print(f"  {'GARCH(1,1) from 15-min':>28}  {'15min vol':>20}  {'3.4pp':>14}  {'—':>10}")
print(f"  {'Harris chain (Gibbs)':>28}  {'15min vol':>20}  {'0.2pp':>14}  {'—':>10}")
print(f"  {'Anzarut (GIG+Gibbs)':>28}  {'15min vol':>20}  {'0.8pp':>14}  {'—':>10}")

# =============================================================================
# 7. Detailed coverage for best models
# =============================================================================
print("\n" + "=" * 70)
print("DETAILED COVERAGE — BEST DAILY MODELS")
print("=" * 70)

# Find best daily models
best_daily = sorted(
    [(model, param, aad) for model, param, cov, aad in results],
    key=lambda x: x[2]
)[:10]

print(f"\nTop 10 daily models by AAD:")
for i, (model, param, aad) in enumerate(best_daily, 1):
    print(f"  {i}. {model}({param}): {aad:.2f}pp")

# Detailed coverage for select models
print(f"\n{'p':>6}  {'Ideal':>6}  {'Best EWMA':>10}  {'Best Roll':>10}  "
      f"{'Persist':>10}  {'GARCH':>10}  {'HistSim':>10}  {'Constant':>10}")

# Get best EWMA and Rolling
best_ewma_entry = min([(m, p, c, a) for m, p, c, a in results if m == "EWMA_N(0,tau)"], key=lambda x: x[3])
best_roll_entry = min([(m, p, c, a) for m, p, c, a in results if m == "RollMean_N(0,tau)"], key=lambda x: x[3])
persist_entry = [(m, p, c, a) for m, p, c, a in results if m == "Persist_N(0,tau)"][0]
garch_entry = [(m, p, c, a) for m, p, c, a in results if m == "GARCH_N(0,tau)"][0]

for p in PROB_LEVELS:
    print(f"{p:>6.2f}  {p*100:>5.0f}%  "
          f"{best_ewma_entry[2][p]:>9.1f}%  "
          f"{best_roll_entry[2][p]:>9.1f}%  "
          f"{persist_entry[2][p]:>9.1f}%  "
          f"{garch_entry[2][p]:>9.1f}%  "
          f"{cov_hist[p]:>9.1f}%  "
          f"{cov_const[p]:>9.1f}%")

print(f"\n  AAD:  EWMA={best_ewma_entry[3]:.2f}pp  "
      f"RollMean={best_roll_entry[3]:.2f}pp  "
      f"Persist={persist_entry[3]:.2f}pp  "
      f"GARCH={garch_entry[3]:.2f}pp  "
      f"HistSim={aad_hist:.2f}pp  "
      f"Constant={aad_const:.2f}pp")

# =============================================================================
# 8. Plot
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Panel 1: AAD bar chart
ax = axes[0]
model_names = []
aad_values = []
colors_list = []

for model, param, cov, aad in sorted(results, key=lambda x: x[3]):
    if aad < 15:  # only show reasonable models
        short = model.replace("_N(0,tau)", "G").replace("_EmpQ", "E")
        model_names.append(f"{short}({param})")
        aad_values.append(aad)
        if "EWMA" in model:
            colors_list.append("seagreen")
        elif "RollMean" in model:
            colors_list.append("lightgreen")
        elif "Persist" in model:
            colors_list.append("gold")
        elif "GARCH" in model:
            colors_list.append("coral")
        else:
            colors_list.append("gray")

# Add reference lines
model_names.extend(["HistSim", "Constant", "N(0,tau*)", "iid boot", "Gibbs(15m)", "Gibbs($)"])
aad_values.extend([aad_hist, aad_const, 2.0, 2.0, 0.3, 0.2])
colors_list.extend(["gray", "lightgray", "steelblue", "steelblue", "royalblue", "navy"])

y_pos = np.arange(len(model_names))
ax.barh(y_pos, aad_values, color=colors_list, height=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(model_names, fontsize=8)
ax.set_xlabel('AAD (pp, lower is better)')
ax.set_title('Daily Coverage AAD: Simple Estimators vs Harris Chain')
ax.axvline(x=2.0, color='steelblue', linewidth=1, linestyle='--', alpha=0.5, label='N(0,tau*)=2.0pp')
ax.axvline(x=0.2, color='navy', linewidth=1, linestyle='--', alpha=0.5, label='Gibbs(15m)=0.2pp')
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis='x')
ax.set_xlim(0, max(aad_values) * 1.1)

# Panel 2: tau* over time
ax = axes[1]
t_range = np.arange(split, min(split + 500, n))
ax.plot(t_range, log_rv_daily[split:split + 500], color='black', linewidth=0.3, alpha=0.5, label='Actual log(RV)')

# Best EWMA
best_lam = float(best_ewma_entry[1].split("=")[1])
ax.plot(t_range, ewma_estimates[best_lam][split:split + 500], color='seagreen', linewidth=1, alpha=0.8,
        label=f'EWMA(lam={best_lam})')

# Best rolling mean
best_w = int(best_roll_entry[1].split("=")[1])
ax.plot(t_range, rolling_estimates[best_w][split:split + 500], color='coral', linewidth=1, alpha=0.8,
        label=f'RollMean(w={best_w})')

ax.plot(t_range, tau_persist[split:split + 500], color='gold', linewidth=0.5, alpha=0.5, label='Persistence')

ax.set_xlabel('Time (days)')
ax.set_ylabel('log(RV)')
ax.set_title('Daily Variance Estimators vs Actual log(RV)')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('simple_estimators_daily_comparison.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved to simple_estimators_daily_comparison.png")

# =============================================================================
# 9. Verdict
# =============================================================================
print("\n" + "=" * 70)
print("VERDICT: DAILY LEVEL")
print("=" * 70)
print()

best_daily_aad = min([aad for _, _, _, aad in results])
best_daily_model = min([(model, param, aad) for model, param, cov, aad in results], key=lambda x: x[2])

print(f"Best simple daily estimator: {best_daily_model[0]}({best_daily_model[1]}) = {best_daily_model[2]:.2f}pp")
print(f"HistSim (no tau*): {aad_hist:.2f}pp")
print(f"Constant N(0,sigma^2): {aad_const:.2f}pp")
print()
print("Reference results:")
print(f"  N(0, tau*) from Harris chain: 2.0pp (daily returns)")
print(f"  iid bootstrap: 2.0pp (daily returns)")
print(f"  GARCH(1,1): {garch_entry[3]:.2f}pp (daily returns)")
print(f"  Gibbs sampler (15-min): 0.3pp (intraday vol)")
print(f"  Gibbs sampler (dollar): 0.2pp (intraday vol)")
print()

if best_daily_model[2] < 2.0:
    print("SIMPLE ESTIMATORS BEAT N(0,tau*) at daily level!")
    print("This confirms that at daily scale, variance adaptation from")
    print("the Harris chain adds nothing beyond simple estimators.")
elif best_daily_model[2] < 3.0:
    print("Simple estimators are comparable to N(0,tau*) at daily level.")
    print("The Harris chain's tau* provides similar adaptation to EWMA/rolling mean.")
    print("At daily scale, sigma=2.27 is too noisy for any variance estimator to help much.")
else:
    print("Simple estimators are WORSE than N(0,tau*) at daily level.")
    print("But all daily models (2-10pp) are far worse than 15-min Gibbs (0.2pp).")
    print("The daily noise (sigma=2.27) dominates regardless of the estimator.")