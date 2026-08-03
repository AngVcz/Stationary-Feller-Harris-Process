"""Kalman Filter vs Gibbs Sampler vs Simple Estimators at 15-min Level.

Tests whether an AR(1) state-space model with Kalman filter inference
matches the 0.2pp AAD achieved by the SF-Harris Gibbs sampler.

Key question: Is Bayesian posterior inference over tau* sufficient,
or do we need the Harris chain specifically?

Models tested:
1. Kalman filter (AR(1) state-space, Gaussian prediction)
2. Kalman + Empirical residuals (hybrid: posterior + bootstrap)
3. Kalman + Student-t innovations (heavy tails)
4. Gibbs sampler (reference: 0.2-0.3pp)
5. EWMA (reference: ~26pp)
6. Historical Simulation (reference: ~2.0pp)
7. Persistence (reference: ~27pp)

If Kalman ≈ Gibbs → Harris chain unnecessary, only posterior matters.
If Kalman >> Gibbs → Harris chain prediction mechanism matters.
If Kalman ≈ EWMA → AR(1) model provides no benefit over point estimates.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats, optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from anzarut_replication import (
    load_ibm_data, compute_15min_returns, detect_and_remove_jumps,
    estimate_periodicity, compute_coverage,
)

PROB_LEVELS = [0.25, 0.50, 0.75, 0.85, 0.90, 0.95]
N_SIM = 3000

print("=" * 70)
print("KALMAN FILTER vs GIBBS SAMPLER vs SIMPLE ESTIMATORS")
print("=" * 70)
print()
print("Key question: Does an AR(1) + Kalman filter match the 0.2pp AAD")
print("of the Gibbs sampler? If yes, the Harris chain is unnecessary.")
print()

# =============================================================================
# 1. Load data and compute 15-min spot volatility
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
# 3. Kalman filter implementation
# =============================================================================
class KalmanLogVol:
    """Kalman filter for log-volatility AR(1) state-space model.

    State:     tau_t = rho * tau_{t-1} + (1-rho) * mu + eta_t,  eta_t ~ N(0, Q)
    Observation: y_t = tau_t + eps_t,  eps_t ~ N(0, R)
    """

    def __init__(self, rho, mu, Q, R):
        self.rho = rho
        self.mu = mu
        self.Q = Q
        self.R = R

    def filter(self, y):
        """Run Kalman filter. Returns filtered/predicted means and variances."""
        n = len(y)
        m = np.zeros(n)
        P = np.zeros(n)
        m_pred = np.zeros(n)
        P_pred = np.zeros(n)
        log_lik = 0.0

        # Initialize from stationary distribution
        if abs(self.rho) < 1:
            m[0] = self.mu
            P[0] = self.Q / (1 - self.rho**2)
        else:
            m[0] = y[0]
            P[0] = self.Q

        for t in range(n):
            if t > 0:
                m_pred[t] = self.rho * m[t-1] + (1 - self.rho) * self.mu
                P_pred[t] = self.rho**2 * P[t-1] + self.Q
            else:
                m_pred[t] = m[0]
                P_pred[t] = P[0]

            # Update
            S = P_pred[t] + self.R
            K = P_pred[t] / S
            m[t] = m_pred[t] + K * (y[t] - m_pred[t])
            P[t] = (1 - K) * P_pred[t]

            # Log-likelihood
            log_lik += -0.5 * np.log(2 * np.pi * S) - 0.5 * (y[t] - m_pred[t])**2 / S

        return m, P, m_pred, P_pred, log_lik

    def smooth(self, y, m, P):
        """Rauch-Tung-Striebel smoother. Returns smoothed means and variances."""
        n = len(y)
        m_s = np.zeros(n)
        P_s = np.zeros(n)

        m_s[-1] = m[-1]
        P_s[-1] = P[-1]

        for t in range(n - 2, -1, -1):
            m_pred_next = self.rho * m[t] + (1 - self.rho) * self.mu
            P_pred_next = self.rho**2 * P[t] + self.Q
            J = self.rho * P[t] / P_pred_next
            m_s[t] = m[t] + J * (m_s[t+1] - m_pred_next)
            P_s[t] = P[t] + J**2 * (P_s[t+1] - P_pred_next)

        return m_s, P_s


def kalman_mle(y, rho_init=0.9):
    """Estimate Kalman filter parameters via MLE."""
    mu_init = np.mean(y)
    # Initial guess for Q and R
    rho_val = rho_init
    residuals = y[1:] - rho_val * y[:-1] - (1 - rho_val) * mu_init
    Q_init = np.var(residuals) * 0.5
    R_init = np.var(y) * 0.1

    def neg_log_lik(params):
        rho = np.tanh(params[0])
        mu = params[1]
        Q = np.exp(params[2])
        R = np.exp(params[3])
        if Q < 1e-10 or R < 1e-10 or abs(rho) > 0.999:
            return 1e10
        kf = KalmanLogVol(rho, mu, Q, R)
        try:
            _, _, _, _, ll = kf.filter(y)
            return -ll
        except Exception:
            return 1e10

    x0 = [np.arctanh(rho_init), mu_init, np.log(Q_init), np.log(R_init)]
    result = optimize.minimize(neg_log_lik, x0, method='L-BFGS-B',
                               options={'maxiter': 1000})

    rho_mle = np.tanh(result.x[0])
    mu_mle = result.x[1]
    Q_mle = np.exp(result.x[2])
    R_mle = np.exp(result.x[3])

    return rho_mle, mu_mle, Q_mle, R_mle, -result.fun


# =============================================================================
# 4. Fit Kalman filter on training data
# =============================================================================
print("\n" + "=" * 70)
print("FITTING KALMAN FILTER (AR(1) STATE-SPACE MODEL)")
print("=" * 70)

print("\nEstimating parameters via MLE on training data...")
rho_mle, mu_mle, Q_mle, R_mle, ll_train = kalman_mle(train, rho_init=0.88)

print(f"  MLE parameters:")
print(f"    rho = {rho_mle:.4f}  (AR(1) coefficient, equiv. P(stay))")
print(f"    mu  = {mu_mle:.4f}  (long-term mean of log-vol)")
print(f"    Q   = {Q_mle:.6f}  (state noise variance, process)")
print(f"    R   = {R_mle:.6f}  (observation noise variance)")
print(f"    log-likelihood = {ll_train:.2f}")

# Implied P(stay) from AR(1) model
p_stay_ar1 = rho_mle
acf1_data = np.corrcoef(train[:-1], train[1:])[0, 1]
print(f"\n  AR(1) rho = {rho_mle:.4f} (equiv. P(stay) in Harris chain)")
print(f"  Data ACF(1) = {acf1_data:.4f}")
print(f"  Harris chain P(stay) = 0.88 (from epsilon calibration)")
print(f"  sigma_obs = {np.sqrt(R_mle):.4f} (observation noise std)")
print(f"  sigma_state = {np.sqrt(Q_mle):.4f} (process noise std)")

# =============================================================================
# 5. Run Kalman filter on all data (train + test)
# =============================================================================
print("\nRunning Kalman filter on all data...")
kf = KalmanLogVol(rho_mle, mu_mle, Q_mle, R_mle)
m, P, m_pred, P_pred, ll_all = kf.filter(log_spot)

print(f"  Filtered log-likelihood (full): {ll_all:.2f}")

# Smoothed estimates (using all data)
m_smooth, P_smooth = kf.smooth(log_spot, m, P)
print(f"  Smoothed tau* mean: {m_smooth[split:].mean():.4f}")
print(f"  Smoothed tau* std:  {m_smooth[split:].std():.4f}")
print(f"  Actual log(RV) std: {log_spot[split:].std():.4f}")
print(f"  Signal-to-noise ratio: Q/R = {Q_mle/R_mle:.2f}")

# =============================================================================
# 6. Generate predictive scenarios
# =============================================================================
print("\n" + "=" * 70)
print("GENERATING PREDICTIVE SCENARIOS")
print("=" * 70)

rng = np.random.default_rng(42)
n_test = len(test)

# --- Model 1: Kalman filter (Gaussian prediction) ---
print("\n  Model 1: Kalman-Gauss (AR(1) + Gaussian prediction)")
kalman_gauss_scenarios = np.zeros((N_SIM, n_test))
kalman_gauss_pit = np.zeros(n_test)

for t in range(n_test):
    idx = split + t
    pred_mean = m_pred[idx]
    pred_var = P_pred[idx] + R_mle
    # Draw scenarios from predictive distribution
    kalman_gauss_scenarios[:, t] = rng.normal(pred_mean, np.sqrt(pred_var), size=N_SIM)
    # PIT value (analytical)
    kalman_gauss_pit[t] = stats.norm.cdf(log_spot[idx], pred_mean, np.sqrt(pred_var))

# --- Model 2: Kalman + Empirical residuals ---
print("  Model 2: Kalman-EmpQ (AR(1) posterior + bootstrap residuals)")
# Compute standardized residuals from training data
train_pred_mean = m_pred[:split]
train_pred_var = P_pred[:split] + R_mle
train_residuals = (train[:split] - train_pred_mean) / np.sqrt(train_pred_var)
train_residuals = train_residuals[np.isfinite(train_residuals)]

kalman_empq_scenarios = np.zeros((N_SIM, n_test))
for t in range(n_test):
    idx = split + t
    # Draw tau from posterior predictive
    tau_draws = rng.normal(m_pred[idx], np.sqrt(P_pred[idx]), size=N_SIM)
    # Draw residuals from empirical distribution
    eps_draws = rng.choice(train_residuals, size=N_SIM)
    kalman_empq_scenarios[:, t] = tau_draws + eps_draws * np.sqrt(R_mle)

# --- Model 3: Kalman + Student-t innovations ---
print("  Model 3: Kalman-Student (AR(1) + Student-t innovations)")
# Estimate degrees of freedom from residuals
t_params = stats.t.fit(train_residuals)
df_mle = t_params[0]
print(f"    Estimated Student-t df: {df_mle:.1f}")

kalman_student_scenarios = np.zeros((N_SIM, n_test))
for t in range(n_test):
    idx = split + t
    # Draw tau from posterior predictive
    tau_draws = rng.normal(m_pred[idx], np.sqrt(P_pred[idx]), size=N_SIM)
    # Draw Student-t innovations
    t_draws = rng.standard_t(df_mle, size=N_SIM)
    kalman_student_scenarios[:, t] = tau_draws + t_draws * np.sqrt(R_mle)

# --- Model 4: Kalman + Harris transition ---
print("  Model 4: Kalman-Harris (Kalman posterior + Harris transition)")
# Use Kalman posterior for tau*, but Harris chain transition for prediction
p_stay = 0.88  # from epsilon calibration
Q_emp = train  # empirical jump distribution

kalman_harris_scenarios = np.zeros((N_SIM, n_test))
for t in range(n_test):
    idx = split + t
    # Draw tau from Kalman posterior (denoised estimate)
    tau_current = rng.normal(m[idx], np.sqrt(P[idx]), size=N_SIM)
    # Harris transition: stay with prob p_stay, jump with prob 1-p_stay
    stay_mask = rng.random(N_SIM) < p_stay
    jump_draws = rng.choice(Q_emp, size=N_SIM)
    tau_next = np.where(stay_mask, tau_current, jump_draws)
    # Add observation noise
    kalman_harris_scenarios[:, t] = tau_next + rng.normal(0, np.sqrt(R_mle), size=N_SIM)

# --- Model 5: EWMA (for comparison) ---
print("  Model 5: EWMA (lambda=0.1, point estimate)")
lam = 0.1
tau_ewma = np.full(n, np.nan)
tau_ewma[0] = log_spot[0]
for t in range(1, n):
    tau_ewma[t] = lam * log_spot[t-1] + (1-lam) * tau_ewma[t-1]

# Estimate residual variance from training data
residuals_ewma = train[:split] - tau_ewma[1:split+1]
sigma_ewma = np.std(residuals_ewma[np.isfinite(residuals_ewma)])

ewma_scenarios = np.zeros((N_SIM, n_test))
for t in range(n_test):
    idx = split + t
    ewma_scenarios[:, t] = rng.normal(tau_ewma[idx], sigma_ewma, size=N_SIM)

# --- Model 6: EWMA + Empirical Q ---
print("  Model 6: EWMA-EmpQ (EWMA point estimate + bootstrap residuals)")
# Residuals from EWMA model
valid_train_ewma = ~np.isnan(tau_ewma[1:split+1])
ewma_resid = (train[:split] - tau_ewma[1:split+1])[valid_train_ewma] / sigma_ewma

ewma_empq_scenarios = np.zeros((N_SIM, n_test))
for t in range(n_test):
    idx = split + t
    tau_est = tau_ewma[idx]
    eps_draws = rng.choice(ewma_resid, size=N_SIM)
    ewma_empq_scenarios[:, t] = tau_est + eps_draws * sigma_ewma

# --- Model 7: Historical Simulation ---
print("  Model 7: Historical Simulation (iid bootstrap)")
hist_scenarios = np.zeros((N_SIM, n_test))
for t in range(n_test):
    idx = rng.integers(0, len(train), size=N_SIM)
    hist_scenarios[:, t] = train[idx]

# --- Model 8: Persistence ---
print("  Model 8: Persistence (tau* = log(RV_{t-1}))")
tau_persist = np.roll(log_spot, 1)
tau_persist[0] = log_spot[0]
residuals_persist = train[1:split] - tau_persist[2:split+1]
sigma_persist = np.std(residuals_persist[np.isfinite(residuals_persist)])

persist_scenarios = np.zeros((N_SIM, n_test))
for t in range(n_test):
    idx = split + t
    persist_scenarios[:, t] = rng.normal(tau_persist[idx], sigma_persist, size=N_SIM)

# --- Model 9: Kalman with forced rho=0.88 (matches Harris P(stay)) ---
print("  Model 9: Kalman-Gauss-rho88 (AR(1) rho=0.88, MLE for Q,R)")
# Use MLE for Q and R but force rho=0.88 (matching Harris P(stay))
# Re-estimate Q and R with rho fixed at 0.88
def neg_log_lik_forced_rho(params, y, rho_fixed):
    mu = params[0]
    Q = np.exp(params[1])
    R = np.exp(params[2])
    if Q < 1e-10 or R < 1e-10:
        return 1e10
    kf = KalmanLogVol(rho_fixed, mu, Q, R)
    try:
        _, _, _, _, ll = kf.filter(y)
        return -ll
    except Exception:
        return 1e10

x0_rho88 = [mu_mle, np.log(Q_mle), np.log(R_mle)]
result_rho88 = optimize.minimize(neg_log_lik_forced_rho, x0_rho88, args=(train, 0.88),
                                  method='L-BFGS-B')
mu_rho88 = result_rho88.x[0]
Q_rho88 = np.exp(result_rho88.x[1])
R_rho88 = np.exp(result_rho88.x[2])

print(f"    Forced rho=0.88, mu={mu_rho88:.4f}, Q={Q_rho88:.6f}, R={R_rho88:.6f}")

kf_rho88 = KalmanLogVol(0.88, mu_rho88, Q_rho88, R_rho88)
m88, P88, m_pred88, P_pred88, ll88 = kf_rho88.filter(log_spot)

kalman_rho88_scenarios = np.zeros((N_SIM, n_test))
for t in range(n_test):
    idx = split + t
    pred_mean = m_pred88[idx]
    pred_var = P_pred88[idx] + R_rho88
    kalman_rho88_scenarios[:, t] = rng.normal(pred_mean, np.sqrt(pred_var), size=N_SIM)

# --- Model 10: Kalman + Harris transition with rho=0.88 ---
print("  Model 10: Kalman-Harris-rho88 (Kalman posterior rho=0.88 + Harris transition)")
kalman_harris88_scenarios = np.zeros((N_SIM, n_test))
for t in range(n_test):
    idx = split + t
    # Draw tau from Kalman posterior (with rho=0.88)
    tau_current = rng.normal(m88[idx], np.sqrt(P88[idx]), size=N_SIM)
    # Harris transition: stay with prob 0.88, jump from Q
    stay_mask = rng.random(N_SIM) < 0.88
    jump_draws = rng.choice(train, size=N_SIM)
    tau_next = np.where(stay_mask, tau_current, jump_draws)
    # Add observation noise
    kalman_harris88_scenarios[:, t] = tau_next + rng.normal(0, np.sqrt(R_rho88), size=N_SIM)

# =============================================================================
# 7. Coverage evaluation
# =============================================================================
print("\n" + "=" * 70)
print("COVERAGE EVALUATION")
print("=" * 70)

def compute_aad(coverage, prob_levels):
    """Compute average absolute deviation from ideal coverage."""
    return np.mean([abs(coverage[p] - p * 100) for p in prob_levels])


models = {
    "Kalman-Gauss": kalman_gauss_scenarios,
    "Kalman-EmpQ": kalman_empq_scenarios,
    "Kalman-Student": kalman_student_scenarios,
    "Kalman-Harris": kalman_harris_scenarios,
    "Kalman-rho88": kalman_rho88_scenarios,
    "Kalman-Harris88": kalman_harris88_scenarios,
    "EWMA(lam=0.1)": ewma_scenarios,
    "EWMA-EmpQ": ewma_empq_scenarios,
    "HistSim": hist_scenarios,
    "Persistence": persist_scenarios,
}

results = {}
for name, scenarios in models.items():
    cov = compute_coverage(test, scenarios, PROB_LEVELS)
    aad = compute_aad(cov, PROB_LEVELS)
    results[name] = {"coverage": cov, "aad": aad}
    print(f"\n  {name}: AAD = {aad:.2f}pp")
    for p in PROB_LEVELS:
        dev = cov[p] - p * 100
        print(f"    p={p:.2f}: cov={cov[p]:.1f}% (dev={dev:+.1f}pp)")

# Analytical PIT for Kalman (exact, no simulation noise)
kalman_gauss_analytical_aad = np.mean([abs(stats.norm.cdf(stats.norm.ppf(p)) - p) * 100
                                        for p in PROB_LEVELS])
# Actually compute from PIT values
kalman_pit_coverage = {}
for p in PROB_LEVELS:
    kalman_pit_coverage[p] = np.mean(kalman_gauss_pit[split:] <= p) * 100
kalman_analytical_aad = np.mean([abs(kalman_pit_coverage[p] - p * 100) for p in PROB_LEVELS])
print(f"\n  Kalman-Gauss (analytical PIT): AAD = {kalman_analytical_aad:.2f}pp")

# =============================================================================
# 8. Comparison table
# =============================================================================
print("\n" + "=" * 70)
print("COMPARISON TABLE — 15-MIN LEVEL")
print("=" * 70)
print()
print(f"{'Model':>25}  {'AAD':>8}  {'vs Gibbs':>10}  {'Posterior?':>10}  {'Mechanism':>20}")
print("-" * 80)

reference = {
    "Gibbs (dollar)": 0.2,
    "Gibbs (15min)": 0.3,
    "Anzarut (GIG)": 0.8,
    "GARCH(1,1)": 3.4,
}

for name, res in sorted(results.items(), key=lambda x: x[1]["aad"]):
    aad = res["aad"]
    vs_gibbs = f"{aad/0.2:.1f}x" if aad < 10 else f">{aad:.0f}pp"
    has_posterior = "Yes" if "Kalman" in name else "No"
    mechanism = "AR(1)" if "Kalman" in name else ("EWMA" if "EWMA" in name else
                  ("Harris" if "Harris" in name else ("iid" if "Hist" in name else "persist")))
    print(f"  {name:>23}  {aad:>6.2f}pp  {vs_gibbs:>10}  {has_posterior:>10}  {mechanism:>20}")

print()
print("  --- Reference results (from previous experiments): ---")
print(f"  {'Gibbs (dollar)':>23}  {'0.20pp':>8}  {'1.0x':>10}  {'Yes':>10}  {'Harris':>20}")
print(f"  {'Gibbs (15min)':>23}  {'0.30pp':>8}  {'1.5x':>10}  {'Yes':>10}  {'Harris':>20}")
print(f"  {'Anzarut (GIG)':>23}  {'0.80pp':>8}  {'4.0x':>10}  {'Yes':>10}  {'Harris':>20}")
print(f"  {'GARCH(1,1)':>23}  {'3.40pp':>8}  {'17x':>10}  {'Yes':>10}  {'AR-like':>20}")

# =============================================================================
# 9. Detailed coverage comparison
# =============================================================================
print("\n" + "=" * 70)
print("DETAILED COVERAGE — BEST MODELS")
print("=" * 70)

best_kalman = min([(name, res) for name, res in results.items() if "Kalman" in name],
                  key=lambda x: x[1]["aad"])

print(f"\nBest Kalman model: {best_kalman[0]} (AAD = {best_kalman[1]['aad']:.2f}pp)")
print()

print(f"{'p':>6}  {'Ideal':>6}  {'Kalman':>8}  {'EWMA':>8}  {'HistSim':>8}  {'Persist':>8}")
print("-" * 50)
for p in PROB_LEVELS:
    k_c = results[best_kalman[0]]["coverage"][p]
    e_c = results["EWMA(lam=0.1)"]["coverage"][p]
    h_c = results["HistSim"]["coverage"][p]
    p_c = results["Persistence"]["coverage"][p]
    print(f"{p:>6.2f}  {p*100:>5.0f}%  {k_c:>7.1f}%  {e_c:>7.1f}%  {h_c:>7.1f}%  {p_c:>7.1f}%")

print(f"\n  AAD:  Kalman={best_kalman[1]['aad']:.2f}pp  "
      f"EWMA={results['EWMA(lam=0.1)']['aad']:.2f}pp  "
      f"HistSim={results['HistSim']['aad']:.2f}pp  "
      f"Persist={results['Persistence']['aad']:.2f}pp")

# =============================================================================
# 10. Comparison of Kalman posterior vs actual log(RV)
# =============================================================================
print("\n" + "=" * 70)
print("KALMAN POSTERIOR QUALITY")
print("=" * 70)
print()

# Denoising effect: how much does the Kalman filter smooth the data?
print("Denoising: Kalman posterior vs raw log(RV)")
print(f"  Raw log(RV) std (test):    {test.std():.4f}")
print(f"  Kalman posterior std (test): {np.std(m[split:split+n_test]):.4f}")
print(f"  Kalman predicted std (test): {np.std(m_pred[split:split+n_test]):.4f}")
print(f"  Smoothing ratio: {np.std(m[split:split+n_test]) / test.std():.4f}")
print(f"  (Lower = more denoising)")

# Innovation analysis
innovations = log_spot[1:] - m_pred[1:]
innov_std = np.std(innovations[split:])
print(f"\n  Innovation std (test): {innov_std:.4f}")
print(f"  Expected std (sqrt(Q+R)): {np.sqrt(Q_mle + R_mle):.4f}")
print(f"  Ratio: {innov_std / np.sqrt(Q_mle + R_mle):.4f}")

# Posterior uncertainty
print(f"\n  Mean posterior variance (test): {np.mean(P[split:split+n_test]):.6f}")
print(f"  Mean predicted variance (test): {np.mean(P_pred[split:split+n_test]):.6f}")
print(f"  Observation noise R: {R_mle:.6f}")
print(f"  Ratio P/(P+R): {np.mean(P[split:split+n_test]) / np.mean(P[split:split+n_test] + R_mle):.4f}")
print(f"  (This is the Kalman gain — higher = more weight on observation)")

# =============================================================================
# 11. Plot
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Panel 1: AAD comparison bar chart
ax = axes[0, 0]
model_names = []
aad_values = []
colors = []
for name, res in sorted(results.items(), key=lambda x: x[1]["aad"]):
    model_names.append(name)
    aad_values.append(res["aad"])
    if "Kalman" in name:
        colors.append("steelblue")
    elif "EWMA" in name:
        colors.append("seagreen")
    elif "Hist" in name:
        colors.append("gray")
    else:
        colors.append("gold")

# Add reference lines
model_names.extend(["Gibbs (dollar)", "Gibbs (15min)", "Anzarut (GIG)"])
aad_values.extend([0.2, 0.3, 0.8])
colors.extend(["navy", "royalblue", "darkblue"])

y_pos = np.arange(len(model_names))
ax.barh(y_pos, aad_values, color=colors, height=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(model_names, fontsize=8)
ax.set_xlabel('AAD (pp, lower is better)')
ax.set_title('Coverage AAD: Kalman Filter vs Gibbs Sampler vs Simple Estimators')
ax.axvline(x=0.2, color='navy', linewidth=1, linestyle='--', alpha=0.5, label='Gibbs (dollar)')
ax.axvline(x=2.0, color='gray', linewidth=1, linestyle='--', alpha=0.5, label='HistSim')
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis='x')
ax.set_xlim(0, min(max(aad_values) * 1.1, 30))

# Panel 2: Kalman posterior vs actual log(RV)
ax = axes[0, 1]
t_range = np.arange(split, min(split + 500, n))
ax.plot(t_range, log_spot[split:split+500], color='black', linewidth=0.3, alpha=0.5,
        label='Actual log(RV)')
ax.plot(t_range, m[split:split+500], color='steelblue', linewidth=1, alpha=0.8,
        label='Kalman posterior mean')
ax.fill_between(t_range,
                m[split:split+500] - 2*np.sqrt(P[split:split+500]),
                m[split:split+500] + 2*np.sqrt(P[split:split+500]),
                color='steelblue', alpha=0.2, label='95% CI')
ax.plot(t_range, tau_ewma[split:split+500], color='seagreen', linewidth=0.8, alpha=0.6,
        label='EWMA(lam=0.1)')
ax.set_xlabel('Time (15-min bars)')
ax.set_ylabel('log(RV)')
ax.set_title('Kalman Posterior vs Actual log(RV)')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Panel 3: PIT histogram for Kalman filter
ax = axes[1, 0]
bins = np.linspace(0, 1, 21)
pit_hist, _ = np.histogram(kalman_gauss_pit[split:], bins=bins)
expected = len(kalman_gauss_pit[split:]) / 20
ax.bar(bins[:-1], pit_hist / expected, width=0.04, color='steelblue', alpha=0.7,
       label='Kalman-Gauss')
ax.axhline(y=1.0, color='black', linewidth=0.5, linestyle='--', label='Uniform')
ax.set_xlabel('PIT Value')
ax.set_ylabel('Density (relative to uniform)')
ax.set_title('PIT Histogram (Uniform = well-calibrated)')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# KS test for PIT uniformity
ks_stat, ks_p = stats.kstest(kalman_gauss_pit[split:], 'uniform')
ax.text(0.02, 0.95, f'KS stat={ks_stat:.4f}, p={ks_p:.4f}',
        transform=ax.transAxes, fontsize=9, verticalalignment='top')

# Panel 4: Predictive distribution width over time
ax = axes[1, 1]
t_range2 = np.arange(split, min(split + 500, n))
pred_std_kalman = np.sqrt(P_pred[split:split+500] + R_mle)
pred_std_ewma = np.full(500, sigma_ewma)
pred_std_gibbs_ref = np.full(500, 1.04)  # approximate sigma from Gibbs

ax.plot(t_range2, pred_std_kalman, color='steelblue', linewidth=1, label='Kalman pred. std')
ax.axhline(y=np.mean(pred_std_kalman), color='steelblue', linewidth=0.5, linestyle='--')
ax.axhline(y=sigma_ewma, color='seagreen', linewidth=0.5, linestyle='--', label=f'EWMA sigma={sigma_ewma:.2f}')
ax.set_xlabel('Time (15-min bars)')
ax.set_ylabel('Predictive std (log-vol)')
ax.set_title('Predictive Distribution Width Over Time')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('kalman_vs_gibbs_comparison.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved to kalman_vs_gibbs_comparison.png")

# =============================================================================
# 12. Verdict
# =============================================================================
print("\n" + "=" * 70)
print("VERDICT: IS THE HARRIS CHAIN NECESSARY?")
print("=" * 70)
print()

best_kalman_name = min([(name, res["aad"]) for name, res in results.items() if "Kalman" in name],
                        key=lambda x: x[1])
best_all_name = min([(name, res["aad"]) for name, res in results.items()],
                    key=lambda x: x[1])
best_kalman_aad = best_kalman_name[1]
best_all_aad = best_all_name[1]

print(f"Best Kalman model: {best_kalman_name[0]} = {best_kalman_aad:.2f}pp")
print(f"Best overall model: {best_all_name[0]} = {best_all_aad:.2f}pp")
print(f"EWMA (point estimate): {results['EWMA(lam=0.1)']['aad']:.2f}pp")
print(f"Historical Simulation: {results['HistSim']['aad']:.2f}pp")
print(f"Persistence: {results['Persistence']['aad']:.2f}pp")
print()
print("Reference: Gibbs sampler = 0.2pp (dollar bars), 0.3pp (15min bars)")
print()

if best_kalman_aad < 1.0:
    print("KALMAN FILTER MATCHES OR APPROACHES GIBBS SAMPLER!")
    print("The AR(1) + Kalman posterior provides similar coverage to the Harris chain.")
    print("Conclusion: Only Bayesian posterior inference over tau* matters.")
    print("The specific Harris chain mechanism (stay/jump) is NOT necessary.")
elif best_kalman_aad < 5.0:
    print("KALMAN FILTER IS BETTER THAN SIMPLE ESTIMATORS BUT WORSE THAN GIBBS.")
    print(f"Gap: Kalman={best_kalman_aad:.2f}pp, Gibbs=0.2pp, EWMA={results['EWMA(lam=0.1)']['aad']:.2f}pp")
    print("The posterior over tau* helps (vs point estimates),")
    print("but the Harris chain prediction mechanism adds further improvement.")
elif best_kalman_aad > 10.0:
    print("KALMAN FILTER IS NO BETTER THAN SIMPLE ESTIMATORS.")
    print("The AR(1) model with Gaussian assumptions cannot capture the")
    print("heavy-tailed volatility dynamics at 15-min scale.")
    print("The Harris chain's empirical Q and mixed prediction are essential.")
else:
    print(f"KALMAN FILTER AAD = {best_kalman_aad:.2f}pp")
    print(f"Gap vs Gibbs: {best_kalman_aad - 0.2:.1f}pp")
    print(f"Gap vs EWMA: {results['EWMA(lam=0.1)']['aad'] - best_kalman_aad:.1f}pp")
    if best_kalman_aad < results['EWMA(lam=0.1)']['aad'] * 0.5:
        print("The Kalman posterior provides significant improvement over point estimates,")
        print("but is still far from the Gibbs sampler.")
    else:
        print("The Kalman posterior provides modest improvement over point estimates.")

# Key comparison: does the Kalman-Harris hybrid (Kalman posterior + Harris transition) help?
kalman_gauss_aad = results["Kalman-Gauss"]["aad"]
kalman_harris_aad = results["Kalman-Harris"]["aad"]
kalman_empq_aad = results["Kalman-EmpQ"]["aad"]

print("\n" + "=" * 70)
print("MECHANISM DECOMPOSITION")
print("=" * 70)
print()
print("All models use the SAME Kalman posterior over tau*.")
print("Difference is only in the prediction mechanism:")
print()
print(f"  Kalman-Gauss  (AR(1) transition + Gaussian noise): {kalman_gauss_aad:.2f}pp")
print(f"  Kalman-EmpQ   (AR(1) transition + empirical residuals): {kalman_empq_aad:.2f}pp")
print(f"  Kalman-Student (AR(1) transition + Student-t noise): {results['Kalman-Student']['aad']:.2f}pp")
print(f"  Kalman-Harris (AR(1) posterior + Harris transition): {kalman_harris_aad:.2f}pp")
print()
diff_gauss_harris = abs(kalman_gauss_aad - kalman_harris_aad)
diff_gauss_empq = abs(kalman_gauss_aad - kalman_empq_aad)
print(f"  Difference: Gauss vs Harris = {diff_gauss_harris:.2f}pp")
print(f"  Difference: Gauss vs EmpQ  = {diff_gauss_empq:.2f}pp")
print()
if diff_gauss_harris < 0.5:
    print("  Harris transition adds < 0.5pp over Gaussian transition.")
    print("  The prediction mechanism doesn't matter — only the posterior over tau* matters.")
elif diff_gauss_harris < 2.0:
    print("  Harris transition adds some improvement, but not dramatic.")
    print("  Both posterior inference AND prediction mechanism contribute.")
else:
    print("  Harris transition adds significant improvement.")
    print("  The specific prediction mechanism (Harris vs Gaussian) matters.")