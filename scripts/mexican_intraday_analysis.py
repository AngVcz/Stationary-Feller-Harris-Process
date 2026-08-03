"""Mexican intraday analysis: SF-Harris vs benchmarks at 1h frequency.

Downloads intraday data from Yahoo Finance, processes into hourly returns,
runs SF-Harris Gibbs pipeline, and computes coverage.
"""
import sys
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf

from anzarut_replication import compute_coverage

PROB_LEVELS = [0.25, 0.50, 0.75, 0.85, 0.90, 0.95]
N_SIM = 2000

TICKERS = {
    'IPC': '^MXX',
    'Bimbo': 'BIMBOA.MX',
    'GMexico': 'GMEXICOB.MX',
    'Walmart_Mex': 'WALMEX.MX',
    'FEMSA': 'FEMSAUBD.MX',
    'Cemex': 'CEMEXCPO.MX',
    'Banorte': 'GFNORTEO.MX',
    'USD_MXN': 'MXN=X',
}

# =========================================================================
# Gibbs sampler (simplified for intraday)
# =========================================================================
def gibbs_simple(log_rv, epsilon=0.1, n_iter=2000, burn_in=500, rng=None):
    """Simplified Gibbs: estimate mu, sigma from log-RV data."""
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(log_rv)
    center = log_rv - np.mean(log_rv)

    # Detect jumps
    diffs = np.abs(np.diff(log_rv))
    threshold = epsilon * np.std(diffs) if epsilon > 0 else 0
    n_jumps = max(np.sum(diffs > threshold), 1)

    # Posterior for mu, sigma (conjugate Normal-InverseGamma)
    mu_prior_mean = np.mean(log_rv)
    mu_prior_var = 1.0
    sigma_prior_a = 1.0
    sigma_prior_b = 0.5 * np.var(log_rv)

    # Sufficient statistics
    y_bar = np.mean(log_rv)
    ss = np.sum((log_rv - y_bar) ** 2)

    # Posterior
    n_post = n
    mu_post_var = 1.0 / (n_post / sigma_prior_b + 1.0 / mu_prior_var)
    mu_post_mean = mu_post_var * (n_post * y_bar / sigma_prior_b + mu_prior_mean / mu_prior_var)

    # Sample from posterior
    mu_samples = rng.normal(mu_post_mean, np.sqrt(mu_post_var), size=n_iter)
    sigma_sq_samples = rng.gamma(
        sigma_prior_a + n_post / 2,
        1.0 / (sigma_prior_b + ss / 2 + 0.5 * n_post * mu_post_var),
        size=n_iter
    )
    sigma_samples = np.sqrt(sigma_sq_samples)

    alpha_samples = rng.exponential(1.0 / max(n_jumps, 1), size=n_iter)

    return {
        'mu': mu_samples[burn_in:],
        'sigma': sigma_samples[burn_in:],
        'alpha': alpha_samples[burn_in:],
    }


def simulate_predictive(emp_data, mu_post, sigma_post, n_test, n_sim=2000,
                        start_val=None, rng_seed=123):
    """Simulate predictive trajectories using empirical Q + Gibbs posterior."""
    rng = np.random.default_rng(rng_seed)
    if start_val is None:
        start_val = emp_data[0]

    simulated = np.zeros((n_sim, n_test))

    for s in range(n_sim):
        idx = rng.integers(0, len(mu_post))
        mu_draw = mu_post[idx]
        sigma_draw = sigma_post[idx]
        current = start_val

        for i in range(n_test):
            # iid from empirical Q + Gibbs denoising
            current = rng.choice(emp_data) + mu_draw
            simulated[s, i] = current + rng.normal(0, sigma_draw)

    return simulated


def hist_sim_predictive(train_data, n_test, n_sim=2000, rng_seed=123):
    """Historical Simulation: iid resample from training data."""
    rng = np.random.default_rng(rng_seed)
    simulated = np.zeros((n_sim, n_test))

    for i in range(n_test):
        simulated[:, i] = rng.choice(train_data, size=n_sim)

    return simulated


# =========================================================================
# Main analysis
# =========================================================================
print("=" * 70)
print("MEXICAN INTRADAY ANALYSIS (1h from Yahoo Finance)")
print("=" * 70)

results = {}

for name, ticker in TICKERS.items():
    print(f"\n--- {name} ({ticker}) ---")

    # Download data
    t = yf.Ticker(ticker)
    hist = t.history(period='730d', interval='1h')

    if len(hist) < 500:
        print(f"  SKIP: only {len(hist)} rows")
        continue

    # Compute log returns
    prices = hist['Close'].dropna()
    log_returns = np.log(prices / prices.shift(1)).dropna().values

    # Remove zeros and extreme values
    log_returns = log_returns[np.isfinite(log_returns)]
    log_returns = log_returns[np.abs(log_returns) < 5 * np.std(log_returns)]

    if len(log_returns) < 500:
        print(f"  SKIP: only {len(log_returns)} clean returns")
        continue

    # Compute RV as squared returns (proxy for spot variance)
    log_rv = np.log(log_returns ** 2 + 1e-20)

    # Remove extreme jumps
    log_rv = log_rv[np.isfinite(log_rv)]

    # Train/test split (80/20)
    n = len(log_rv)
    split = int(n * 0.8)
    train = log_rv[:split]
    test = log_rv[split:]

    print(f"  Total: {n}, Train: {len(train)}, Test: {len(test)}")
    print(f"  Train stats: mean={np.mean(train):.3f}, std={np.std(train):.3f}, "
          f"skew={pd.Series(train).skew():.2f}, kurt={pd.Series(train).kurtosis():.2f}")

    # Run Gibbs
    rng_gibbs = np.random.default_rng(42)
    gibbs = gibbs_simple(train, epsilon=0.1, n_iter=2000, burn_in=500, rng=rng_gibbs)

    mu_post = gibbs['mu']
    sigma_post = gibbs['sigma']
    emp_data = train - np.mean(train)

    # SF-Harris (empirical Q + Gibbs denoising)
    sim_gibbs = simulate_predictive(emp_data, mu_post, sigma_post, len(test),
                                     N_SIM, start_val=train[-1])
    cov_gibbs = compute_coverage(test, sim_gibbs, PROB_LEVELS)
    aad_gibbs = np.mean([abs(cov_gibbs[p] - p * 100) for p in PROB_LEVELS])

    # Historical Simulation (baseline)
    sim_hist = hist_sim_predictive(train, len(test), N_SIM)
    cov_hist = compute_coverage(test, sim_hist, PROB_LEVELS)
    aad_hist = np.mean([abs(cov_hist[p] - p * 100) for p in PROB_LEVELS])

    results[name] = {
        'aad_gibbs': aad_gibbs,
        'aad_hist': aad_hist,
        'cov_gibbs': cov_gibbs,
        'cov_hist': cov_hist,
        'n_train': len(train),
        'n_test': len(test),
        'mean': np.mean(train),
        'std': np.std(train),
        'skew': pd.Series(train).skew(),
        'kurt': pd.Series(train).kurtosis(),
    }

    print(f"  SF-Harris Gibbs: AAD = {aad_gibbs:.2f}pp")
    print(f"  Hist Sim:         AAD = {aad_hist:.2f}pp")
    print(f"  Gap (Gibbs - Hist): {aad_gibbs - aad_hist:+.2f}pp")

# =========================================================================
# Summary table
# =========================================================================
print("\n" + "=" * 70)
print("SUMMARY: MEXICAN INTRADAY (1h) COVERAGE")
print("=" * 70)
print(f"\n{'Asset':<15} {'N':>6} {'SF-Harris':>10} {'Hist.Sim':>10} {'Gap':>8} {'Winner':>10}")
print("-" * 60)
for name in results:
    r = results[name]
    gap = r['aad_gibbs'] - r['aad_hist']
    winner = "Gibbs" if gap < 0 else "Hist.Sim"
    print(f"{name:<15} {r['n_train']:>6} {r['aad_gibbs']:>8.2f}pp {r['aad_hist']:>8.2f}pp {gap:>+7.2f}pp {winner:>10}")

# Compare with daily results
print("\n" + "=" * 70)
print("COMPARISON: INTRADAY vs DAILY (previous daily results)")
print("=" * 70)
daily_aad = {
    'IPC': 8.5, 'Bimbo': 11.7, 'GMexico': 11.3,
    'Walmart_Mex': 10.4, 'FEMSA': 8.1, 'Cemex': 10.2,
    'Banorte': 9.5, 'USD_MXN': 7.8
}
print(f"\n{'Asset':<15} {'Daily AAD':>10} {'1h AAD':>10} {'Improvement':>12}")
print("-" * 50)
for name in results:
    r = results[name]
    daily = daily_aad.get(name, float('nan'))
    improvement = daily - r['aad_gibbs']
    print(f"{name:<15} {daily:>8.1f}pp {r['aad_gibbs']:>8.2f}pp {improvement:>+10.2f}pp")

# =========================================================================
# Coverage details
# =========================================================================
print("\n" + "=" * 70)
print("COVERAGE DETAILS: SF-Harris vs Hist.Sim at each probability level")
print("=" * 70)
for name in results:
    r = results[name]
    print(f"\n{name} (N={r['n_train']}+{r['n_test']}):")
    print(f"  {'p':>6}  {'Ideal':>6}  {'Gibbs':>8}  {'HistSim':>8}  {'Gibbs_dev':>10}  {'Hist_dev':>10}")
    for p in PROB_LEVELS:
        ideal = p * 100
        g = r['cov_gibbs'][p]
        h = r['cov_hist'][p]
        print(f"  {p:>6.2f}  {ideal:>6.1f}  {g:>8.1f}  {h:>8.1f}  {g-ideal:>+9.1f}  {h-ideal:>+9.1f}")

# Save results
output_path = Path(__file__).resolve().parent.parent / "data" / "mexican_intraday_results.npz"
np.savez(output_path, **{f"{k}_gibbs_aad": v['aad_gibbs'] for k, v in results.items()})
print(f"\nResults saved to {output_path}")