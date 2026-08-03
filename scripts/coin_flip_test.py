"""Coin Flip Model: 0.88 * current + 0.12 * Q_empirical.

Tests the simplest possible version of the mixture:
  - 88%: stay at current level (or resample from similar-volatility returns)
  - 12%: draw from Q (past data)

No Gibbs, no Harris chain, no Gaussian emission. Just a coin flip + empirical data.
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
P_STAY = 0.88  # from epsilon calibration at 15-min

print("=" * 70)
print("COIN FLIP MODEL: 0.88 * current + 0.12 * Q_empirical")
print("=" * 70)
print()
print("Simplest version of the Harris chain: a coin flip + empirical data.")
print("No Gibbs, no MCMC, no Gaussian emission.")
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
# 2. Coin flip models for VOLATILITY (log-RV) prediction
# =============================================================================
print("\n" + "=" * 70)
print("VOLATILITY PREDICTION (log-RV coverage)")
print("=" * 70)

rng = np.random.default_rng(42)

# --- Model A: Coin flip on log(RV) levels ---
# log(RV_{t+1}) = log(RV_t) with prob 0.88, or draw from Q with prob 0.12
# Then add observation noise: log(H_{t+1}) ~ N(log(RV_{t+1}), sigma^2)
print("\n  A: Coin flip on log(RV) levels (stay at current or jump to past)")
sigma_obs = np.std(train_logrv[1:] - train_logrv[:-1]) / np.sqrt(2)  # observation noise
coinflip_lv_scenarios = np.zeros((N_SIM, n_test))
for t in range(n_test):
    idx = split + t
    stay_mask = rng.random(N_SIM) < P_STAY
    jump_draws = rng.choice(train_logrv, size=N_SIM)
    tau = np.where(stay_mask, log_spot[idx - 1], jump_draws)
    coinflip_lv_scenarios[:, t] = rng.normal(tau, sigma_obs, size=N_SIM)

# --- Model B: Coin flip on log(RV) levels, NO observation noise ---
# Direct: log(RV_{t+1}) = log(RV_t) with prob 0.88, or draw from Q with prob 0.12
print("  B: Coin flip on log(RV) levels (no obs noise)")
coinflip_lv_nonoise = np.zeros((N_SIM, n_test))
for t in range(n_test):
    idx = split + t
    stay_mask = rng.random(N_SIM) < P_STAY
    jump_draws = rng.choice(train_logrv, size=N_SIM)
    coinflip_lv_nonoise[:, t] = np.where(stay_mask, log_spot[idx - 1], jump_draws)

# --- Model C: Coin flip with EWMA for current level ---
# EWMA(lambda=0.1) for current, then stay/jump
print("  C: Coin flip with EWMA current level")
lam = 0.1
tau_ewma = np.full(n, np.nan)
tau_ewma[0] = log_spot[0]
for t in range(1, n):
    tau_ewma[t] = lam * log_spot[t - 1] + (1 - lam) * tau_ewma[t - 1]

coinflip_ewma = np.zeros((N_SIM, n_test))
for t in range(n_test):
    idx = split + t
    stay_mask = rng.random(N_SIM) < P_STAY
    jump_draws = rng.choice(train_logrv, size=N_SIM)
    tau = np.where(stay_mask, tau_ewma[idx], jump_draws)
    coinflip_ewma[:, t] = rng.normal(tau, sigma_obs, size=N_SIM)

# --- Model D: Coin flip with Kalman posterior for current level ---
print("  D: Coin flip with Kalman posterior (from previous experiment)")
# Reuse Kalman filter from kalman_vs_gibbs.py
from scipy import optimize

class KalmanLogVol:
    def __init__(self, rho, mu, Q, R):
        self.rho, self.mu, self.Q, self.R = rho, mu, Q, R
    def filter(self, y):
        n = len(y)
        m, P = np.zeros(n), np.zeros(n)
        m[0] = self.mu
        P[0] = self.Q / (1 - self.rho**2) if abs(self.rho) < 1 else self.Q
        for t in range(1, n):
            m_pred = self.rho * m[t-1] + (1-self.rho)*self.mu
            P_pred = self.rho**2 * P[t-1] + self.Q
            K = P_pred / (P_pred + self.R)
            m[t] = m_pred + K * (y[t] - m_pred)
            P[t] = (1 - K) * P_pred
        return m, P

# Fit Kalman with rho forced to 0.88
mu_init = np.mean(train_logrv)
residuals_88 = train_logrv[1:] - 0.88 * train_logrv[:-1] - (1-0.88)*mu_init
Q_88 = np.var(residuals_88) * 0.5
R_88 = np.var(train_logrv) * 0.1

def neg_ll(params):
    mu, lQ, lR = params
    Q, R = np.exp(lQ), np.exp(lR)
    kf = KalmanLogVol(0.88, mu, Q, R)
    m, P = kf.filter(log_spot)
    S = np.zeros(n); ll = 0
    m_pred = np.zeros(n); P_pred = np.zeros(n)
    m_pred[0] = m[0]; P_pred[0] = P[0]
    for t in range(1, n):
        m_pred[t] = 0.88*m[t-1]+0.12*mu
        P_pred[t] = 0.88**2*P[t-1]+Q
    S = P_pred + R
    valid = S > 0
    ll = np.sum(-0.5*np.log(2*np.pi*S[valid]) - 0.5*(log_spot[valid]-m_pred[valid])**2/S[valid])
    return -ll

x0 = [mu_init, np.log(Q_88), np.log(R_88)]
res = optimize.minimize(neg_ll, x0, method='L-BFGS-B')
mu_kf, Q_kf, R_kf = res.x[0], np.exp(res.x[1]), np.exp(res.x[2])

kf = KalmanLogVol(0.88, mu_kf, Q_kf, R_kf)
m_kf, P_kf = kf.filter(log_spot)

coinflip_kalman = np.zeros((N_SIM, n_test))
for t in range(n_test):
    idx = split + t
    stay_mask = rng.random(N_SIM) < P_STAY
    jump_draws = rng.choice(train_logrv, size=N_SIM)
    tau_current = rng.normal(m_kf[idx], np.sqrt(P_kf[idx]), size=N_SIM)
    tau = np.where(stay_mask, tau_current, jump_draws)
    coinflip_kalman[:, t] = rng.normal(tau, np.sqrt(R_kf), size=N_SIM)

# --- Model E: Historical Simulation (reference) ---
print("  E: Historical Simulation (iid from past)")
hist_scenarios = np.zeros((N_SIM, n_test))
for t in range(n_test):
    idx = rng.integers(0, len(train_logrv), size=N_SIM)
    hist_scenarios[:, t] = train_logrv[idx]

# --- Model F: Pure persistence (reference) ---
print("  F: Persistence (tau* = last observation + noise)")
persist_scenarios = np.zeros((N_SIM, n_test))
sigma_persist = np.std(train_logrv[1:] - train_logrv[:-1]) / np.sqrt(2)
for t in range(n_test):
    idx = split + t
    persist_scenarios[:, t] = rng.normal(log_spot[idx-1], sigma_persist, size=N_SIM)

# =============================================================================
# 3. Evaluate volatility coverage
# =============================================================================
print("\n" + "=" * 70)
print("VOLATILITY COVERAGE (log-RV)")
print("=" * 70)

vol_models = {
    "A: CoinFlip raw": coinflip_lv_scenarios,
    "B: CoinFlip no-noise": coinflip_lv_nonoise,
    "C: CoinFlip EWMA": coinflip_ewma,
    "D: CoinFlip Kalman": coinflip_kalman,
    "E: HistSim": hist_scenarios,
    "F: Persistence": persist_scenarios,
}

vol_results = {}
for name, scenarios in vol_models.items():
    cov = compute_coverage(test_logrv, scenarios, PROB_LEVELS)
    aad = np.mean([abs(cov[p] - p*100) for p in PROB_LEVELS])
    vol_results[name] = {"coverage": cov, "aad": aad}
    print(f"\n  {name}: AAD = {aad:.2f}pp")
    for p in PROB_LEVELS:
        dev = cov[p] - p*100
        print(f"    p={p:.2f}: cov={cov[p]:.1f}% (dev={dev:+.1f}pp)")

# =============================================================================
# 4. Coin flip models for RETURN prediction
# =============================================================================
print("\n" + "=" * 70)
print("RETURN PREDICTION (daily-like coverage of returns)")
print("=" * 70)

# For return prediction, we need to convert volatility levels to returns.
# Instead of N(0, sqrt(exp(tau))), use empirical returns at similar volatility.

# Build volatility buckets for conditional resampling
n_buckets = 20
logrv_min, logrv_max = train_logrv.min(), train_logrv.max()
bucket_edges = np.linspace(logrv_min, logrv_max, n_buckets + 1)
bucket_indices = np.digitize(train_logrv, bucket_edges) - 1
bucket_indices = np.clip(bucket_indices, 0, n_buckets - 1)

# Create bucket-to-returns mapping
bucket_returns = {}
for b in range(n_buckets):
    mask_b = bucket_indices == b
    if mask_b.sum() > 0:
        bucket_returns[b] = raw_returns[:split][mask_b]
    else:
        bucket_returns[b] = train_ret  # fallback

# --- Model G: Coin flip on returns (conditional resampling) ---
print("\n  G: Coin flip on returns (88% from similar-vol, 12% from all)")
coinflip_ret_scenarios = np.zeros((N_SIM, n_test))
for t in range(n_test):
    idx = split + t
    stay_mask = rng.random(N_SIM) < P_STAY
    # For "stay": resample from returns at similar volatility
    current_bucket = np.clip(np.digitize(log_spot[idx-1], bucket_edges) - 1, 0, n_buckets-1)
    similar_returns = bucket_returns[current_bucket]
    stay_draws = rng.choice(similar_returns, size=N_SIM)
    # For "jump": resample from all returns
    jump_draws = rng.choice(train_ret, size=N_SIM)
    coinflip_ret_scenarios[:, t] = np.where(stay_mask, stay_draws, jump_draws)

# --- Model H: Coin flip on returns (current return as stay) ---
print("  H: Coin flip on returns (88% repeat last, 12% from all)")
coinflip_ret_repeat = np.zeros((N_SIM, n_test))
for t in range(n_test):
    idx = split + t
    stay_mask = rng.random(N_SIM) < P_STAY
    jump_draws = rng.choice(train_ret, size=N_SIM)
    coinflip_ret_repeat[:, t] = np.where(stay_mask, raw_returns[idx-1], jump_draws)

# --- Model I: N(0, sqrt(exp(tau))) emission (Gibbs-style) ---
print("  I: Coin flip + N(0, sqrt(exp(tau))) emission")
coinflip_emission = np.zeros((N_SIM, n_test))
for t in range(n_test):
    idx = split + t
    stay_mask = rng.random(N_SIM) < P_STAY
    jump_draws = rng.choice(train_logrv, size=N_SIM)
    tau = np.where(stay_mask, log_spot[idx-1], jump_draws)
    coinflip_emission[:, t] = rng.normal(0, np.sqrt(np.exp(tau)), size=N_SIM)

# --- Model J: HistSim on returns ---
print("  J: HistSim on returns (iid from past)")
hist_ret = np.zeros((N_SIM, n_test))
for t in range(n_test):
    idx = rng.integers(0, len(train_ret), size=N_SIM)
    hist_ret[:, t] = train_ret[idx]

# Evaluate return coverage
ret_models = {
    "G: CoinFlip similar-vol": coinflip_ret_scenarios,
    "H: CoinFlip repeat-last": coinflip_ret_repeat,
    "I: CoinFlip + emission": coinflip_emission,
    "J: HistSim returns": hist_ret,
}

ret_results = {}
for name, scenarios in ret_models.items():
    cov = compute_coverage(test_ret, scenarios, PROB_LEVELS)
    aad = np.mean([abs(cov[p] - p*100) for p in PROB_LEVELS])
    ret_results[name] = {"coverage": cov, "aad": aad}
    print(f"\n  {name}: AAD = {aad:.2f}pp")
    for p in PROB_LEVELS:
        dev = cov[p] - p*100
        print(f"    p={p:.2f}: cov={cov[p]:.1f}% (dev={dev:+.1f}pp)")

# =============================================================================
# 5. Summary
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY: COIN FLIP vs GIBBS vs HISTSIM")
print("=" * 70)
print()
print("VOLATILITY (log-RV) coverage:")
print(f"  {'Model':>30}  {'AAD':>8}  {'vs Gibbs':>10}  {'vs HistSim':>12}")
print("-" * 65)
for name, res in sorted(vol_results.items(), key=lambda x: x[1]["aad"]):
    aad = res["aad"]
    vs_gibbs = f"{aad/0.2:.1f}x" if aad < 10 else f">{aad:.0f}pp"
    vs_hist = f"{aad - 0.48:+.2f}pp"
    print(f"  {name:>30}  {aad:>6.2f}pp  {vs_gibbs:>10}  {vs_hist:>12}")

print()
print("RETURN coverage:")
print(f"  {'Model':>30}  {'AAD':>8}  {'vs HistSim':>12}")
print("-" * 55)
for name, res in sorted(ret_results.items(), key=lambda x: x[1]["aad"]):
    aad = res["aad"]
    vs_hist = f"{aad - ret_results['J: HistSim returns']['aad']:+.2f}pp"
    print(f"  {name:>30}  {aad:>6.2f}pp  {vs_hist:>12}")

print()
print("Reference:")
print(f"  Gibbs (dollar bars): 0.2pp (volatility)")
print(f"  Gibbs (15min bars):  0.3pp (volatility)")
print(f"  Anzarut (GIG+Gibbs): 0.8pp (volatility)")
print()

best_vol = min(vol_results.items(), key=lambda x: x[1]["aad"])
best_ret = min(ret_results.items(), key=lambda x: x[1]["aad"])
print(f"Best coin flip (volatility): {best_vol[0]} = {best_vol[1]['aad']:.2f}pp")
print(f"Best coin flip (returns):    {best_ret[0]} = {best_ret[1]['aad']:.2f}pp")
print()

if best_vol[1]['aad'] < 1.0:
    print("COIN FLIP MATCHES GIBBS SAMPLER!")
    print("The Harris chain is unnecessary — a simple coin flip + empirical data suffices.")
elif best_vol[1]['aad'] < 2.0:
    print("Coin flip is close to Gibbs but not as good.")
    print("The mass point helps, but the Gibbs posterior denoising adds value.")
elif best_vol[1]['aad'] < 5.0:
    print("Coin flip is better than simple estimators but far from Gibbs.")
    print("The mass point helps, but denoising and emission function matter.")
else:
    print("Coin flip is no better than simple estimators.")
    print("The mass point alone doesn't help without proper denoising.")