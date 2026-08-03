"""Fast sweep of P(stay) in the Gibbs transition kernel.

Vectorized simulation: all N_SIM trajectories simulated simultaneously.
Runs the Gibbs sampler once, then sweeps P(stay) in the prediction step.
"""
import sys
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from anzarut_replication import (
    load_ibm_data, compute_15min_returns, detect_and_remove_jumps,
    estimate_periodicity, gibbs_gig_harris, compute_coverage,
)

PROB_LEVELS = [0.25, 0.50, 0.75, 0.85, 0.90, 0.95]
N_SIM = 2000
P_STAY_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.88, 0.9, 0.95, 0.99]

print("=" * 70)
print("P(STAY) SWEEP IN GIBBS TRANSITION (vectorized)")
print("=" * 70)

# Load data
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

n = len(log_spot)
split = int(n * 0.8)
train_logrv = log_spot[:split]
test_logrv = log_spot[split:]
n_test = len(test_logrv)

print(f"  Train: {split}, Test: {n_test}")

# Run Gibbs ONCE
print("\nRunning Gibbs (eps=0.1)...")
rng_gibbs = np.random.default_rng(42)
gibbs = gibbs_gig_harris(train_logrv, epsilon=0.1, n_iter=2000, burn_in=500, rng=rng_gibbs)

alpha_post = gibbs["alpha"]
mu_post = gibbs["mu"]
sigma_post = gibbs["sigma"]
p_stay_gibbs = np.exp(-np.mean(alpha_post))

print(f"  Posterior P(stay) = exp(-alpha) = {p_stay_gibbs:.4f}")
print(f"  Posterior mu = {np.mean(mu_post):.4f}")
print(f"  Posterior sigma = {np.mean(sigma_post):.4f}")

emp_data = train_logrv - np.mean(train_logrv)
emp_mean = np.mean(train_logrv)
mu_mean = np.mean(mu_post)
sigma_mean = np.mean(sigma_post)

# =========================================================================
# Vectorized simulation function
# =========================================================================
def simulate_harris_vectorized(p_stay, n_sim, n_test, emp_data, mu, sigma,
                                  start_val, rng_seed=123, obs_noise=True):
    """Vectorized Harris chain simulation with overridden P(stay)."""
    rng = np.random.default_rng(rng_seed)
    simulated = np.zeros((n_sim, n_test))
    current = np.full(n_sim, start_val)

    for i in range(n_test):
        stay_mask = rng.random(n_sim) < p_stay
        jump_draws = rng.choice(emp_data, size=n_sim) + mu
        current = np.where(stay_mask, current, jump_draws)
        if obs_noise:
            simulated[:, i] = current + rng.normal(0, sigma, size=n_sim)
        else:
            simulated[:, i] = current

    return simulated

def simulate_harris_gibbs_uncertainty(p_stay, n_sim, n_test, emp_data,
                                         mu_post, sigma_post, start_val, rng_seed=123):
    """Harris chain with mu/sigma drawn from Gibbs posterior each trajectory."""
    rng = np.random.default_rng(rng_seed)
    simulated = np.zeros((n_sim, n_test))

    for s in range(n_sim):
        idx = rng.integers(0, len(mu_post))
        mu_draw = mu_post[idx]
        sigma_draw = sigma_post[idx]
        current = start_val

        for i in range(n_test):
            if rng.random() < p_stay:
                pass
            else:
                current = rng.choice(emp_data) + mu_draw
            simulated[s, i] = current + rng.normal(0, sigma_draw)

    return simulated

# =========================================================================
# Sweep P(stay)
# =========================================================================
print("\n" + "=" * 70)
print("APPROACH A: Fixed params + override P(stay) + obs noise (vectorized)")
print("=" * 70)

results_a = {}
for p_stay in P_STAY_VALUES:
    sim = simulate_harris_vectorized(p_stay, N_SIM, n_test, emp_data, mu_mean, sigma_mean,
                                       train_logrv[-1], obs_noise=True)
    cov = compute_coverage(test_logrv, sim, PROB_LEVELS)
    aad = np.mean([abs(cov[p] - p * 100) for p in PROB_LEVELS])
    results_a[p_stay] = {"coverage": cov, "aad": aad}
    print(f"  P(stay)={p_stay:.2f}: AAD={aad:.2f}pp")

print("\n" + "=" * 70)
print("APPROACH B: Fixed params + override P(stay), NO obs noise")
print("=" * 70)

results_b = {}
for p_stay in P_STAY_VALUES:
    sim = simulate_harris_vectorized(p_stay, N_SIM, n_test, emp_data, mu_mean, sigma_mean,
                                       train_logrv[-1], obs_noise=False)
    cov = compute_coverage(test_logrv, sim, PROB_LEVELS)
    aad = np.mean([abs(cov[p] - p * 100) for p in PROB_LEVELS])
    results_b[p_stay] = {"coverage": cov, "aad": aad}
    print(f"  P(stay)={p_stay:.2f}: AAD={aad:.2f}pp")

print("\n" + "=" * 70)
print("APPROACH C: Gibbs posterior uncertainty + override P(stay) (slow)")
print("=" * 70)

results_c = {}
for p_stay in P_STAY_VALUES:
    sim = simulate_harris_gibbs_uncertainty(p_stay, N_SIM, n_test, emp_data,
                                              mu_post, sigma_post, train_logrv[-1])
    cov = compute_coverage(test_logrv, sim, PROB_LEVELS)
    aad = np.mean([abs(cov[p] - p * 100) for p in PROB_LEVELS])
    results_c[p_stay] = {"coverage": cov, "aad": aad}
    print(f"  P(stay)={p_stay:.2f}: AAD={aad:.2f}pp")

# =========================================================================
# Coin flip reference (no Gibbs, from previous experiment)
# =========================================================================
coinflip_nonoise = {0.0: 0.42, 0.10: 0.70, 0.20: 1.75, 0.30: 4.43,
                    0.40: 8.12, 0.50: 13.69, 0.60: 20.18, 0.70: 27.61,
                    0.80: 37.90, 0.85: 45.50, 0.88: 50.49,
                    0.90: 54.57, 0.95: 65.39, 0.99: 69.97}

coinflip_noise = {0.0: 18.51, 0.10: 18.50, 0.20: 18.37, 0.30: 17.89,
                  0.40: 17.35, 0.50: 16.53, 0.60: 15.61, 0.70: 14.49,
                  0.80: 13.06, 0.85: 12.18, 0.88: 11.66,
                  0.90: 11.25, 0.95: 10.12, 0.99: 9.22}

# =========================================================================
# Summary
# =========================================================================
print("\n" + "=" * 70)
print("SUMMARY: P(STAY) IN GIBBS TRANSITION vs COIN FLIP")
print("=" * 70)

print(f"\nGibbs posterior P(stay) = {p_stay_gibbs:.4f}")
print(f"\n{'P(stay)':>8}  {'Gibbs+noise':>12}  {'Gibbs no-n':>12}  {'Gibbs uncert':>13}  {'CF+noise':>10}  {'CF no-n':>10}")
print("-" * 70)
for p in P_STAY_VALUES:
    a = f"{results_a[p]['aad']:.2f}"
    b = f"{results_b[p]['aad']:.2f}"
    c = f"{results_c[p]['aad']:.2f}"
    cf_n = f"{coinflip_noise[p]:.2f}" if p in coinflip_noise else "N/A"
    cf_nn = f"{coinflip_nonoise[p]:.2f}" if p in coinflip_nonoise else "N/A"
    print(f"  {p:>6.2f}  {a:>10}pp  {b:>10}pp  {c:>11}pp  {cf_n:>8}pp  {cf_nn:>8}pp")

best_a = min(results_a, key=lambda k: results_a[k]["aad"])
best_b = min(results_b, key=lambda k: results_b[k]["aad"])
best_c = min(results_c, key=lambda k: results_c[k]["aad"])

print(f"\nOptimal P(stay):")
print(f"  A (fixed params + noise): P={best_a:.2f}, AAD={results_a[best_a]['aad']:.2f}pp")
print(f"  B (fixed params, no noise): P={best_b:.2f}, AAD={results_b[best_b]['aad']:.2f}pp")
print(f"  C (Gibbs uncertainty + noise): P={best_c:.2f}, AAD={results_c[best_c]['aad']:.2f}pp")

print(f"\nGibbs posterior P(stay)={p_stay_gibbs:.4f}")
print(f"  A at Gibbs P: {results_a.get(round(p_stay_gibbs, 2), results_a[0.4])['aad']:.2f}pp")
print(f"  B at Gibbs P: {results_b.get(round(p_stay_gibbs, 2), results_b[0.4])['aad']:.2f}pp")

print(f"\nKey comparisons:")
print(f"  Gibbs eps=0.1 (full pipeline):       0.2pp (reference)")
print(f"  HistSim (P=0, no Gibbs):             0.42pp (no noise) / 18.51pp (with noise)")
print(f"  Coin flip P=0.88 (no Gibbs):         50.49pp (no noise) / 11.66pp (with noise)")

# =========================================================================
# Plot
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Panel 1: AAD vs P(stay)
ax = axes[0]
p_vals = sorted(results_a.keys())

ax.plot(p_vals, [results_a[p]["aad"] for p in p_vals], 'o-', color='steelblue',
        linewidth=2, markersize=6, label='A: Gibbs params + noise')
ax.plot(p_vals, [results_b[p]["aad"] for p in p_vals], 's--', color='coral',
        linewidth=1.5, markersize=5, label='B: Gibbs params, no noise')
ax.plot(p_vals, [results_c[p]["aad"] for p in p_vals], '^-', color='seagreen',
        linewidth=1.5, markersize=5, label='C: Gibbs uncertainty + noise')

# Coin flip reference (with noise)
valid_cf = [(p, coinflip_noise[p]) for p in p_vals if p in coinflip_noise and coinflip_noise[p] < 20]
ax.plot([p for p, v in valid_cf], [v for p, v in valid_cf], 'D:',
        color='gray', linewidth=1, markersize=4, label='Coin flip + noise (no Gibbs)')

ax.axhline(y=0.2, color='navy', linestyle=':', alpha=0.7, label='Gibbs eps=0.1 (0.2pp)')
ax.axhline(y=0.52, color='gray', linestyle=':', alpha=0.5, label='HistSim (0.52pp)')
ax.axvline(x=0.88, color='red', linestyle='--', alpha=0.3, label='P=0.88 (ACF)')
ax.axvline(x=p_stay_gibbs, color='orange', linestyle='--', alpha=0.5,
           label=f'P={p_stay_gibbs:.2f} (Gibbs posterior)')

ax.set_xlabel('P(stay)')
ax.set_ylabel('AAD (pp, lower is better)')
ax.set_title('P(stay) Sweep: Gibbs Transition vs Coin Flip')
ax.legend(fontsize=7, loc='upper left')
ax.grid(alpha=0.3)
ax.set_ylim(0, 25)

# Panel 2: Coverage profiles for select P(stay)
ax = axes[1]
for p_stay in [0.0, p_stay_gibbs, 0.88]:
    if p_stay in results_b:
        cov = results_b[p_stay]["coverage"]
        deviations = [cov[p] - p * 100 for p in PROB_LEVELS]
        label = f'P={p_stay:.2f}' if p_stay != 0.38 else 'P=0.38 (Gibbs posterior)'
        ax.plot(PROB_LEVELS, deviations, 'o-', label=label, linewidth=1.5)

ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_xlabel('Probability level')
ax.set_ylabel('Coverage deviation (pp)')
ax.set_title('Coverage Profile: Gibbs Transition with different P(stay)')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('gibbs_pstay_sweep.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved to gibbs_pstay_sweep.png")