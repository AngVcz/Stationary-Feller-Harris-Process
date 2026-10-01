"""Verify simulate_predictive_sf_harris_vec matches the loop within MC noise.

Runs old (per-path loop) vs vec (vectorized over n_sim) on the two headline
configs (calendar 15-min eps1e-5, dollar ~24/dia eps0.1) with 3 seed pairs each.
AADs must agree within MC noise (~0.05 pp). Not bit-identical (draw order
differs), same distribution.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from anzarut_replication import (
    load_ibm_data, compute_15min_returns, split_returns_by_date,
    estimate_periodicity, estimate_alpha, gibbs_gig_harris,
    simulate_predictive_sf_harris, simulate_predictive_sf_harris_vec,
    compute_coverage, build_dollar_bars_rolling,
)
from anzarut_intraday_15min import compute_15min_spot_volatility, PROB_LEVELS
import pandas as pd

PROB = [0.25, 0.50, 0.75, 0.85, 0.90, 0.95]
CUTOFF = pd.Timestamp("2014-05-28")
SEEDS = [(42, 123), (7, 7), (99, 99)]


def aad(cov):
    return float(np.mean([abs(cov[p] - p * 100) for p in PROB]))


def calendar_pair(gseed, sseed):
    df = load_ibm_data()
    returns = compute_15min_returns(df)
    tr, te = split_returns_by_date(returns, train_frac=0.8)
    period = estimate_periodicity(tr)
    lt, _ = compute_15min_spot_volatility(tr, period)
    et, _ = compute_15min_spot_volatility(te, period)
    tl, tev = lt.values, et.values
    a = estimate_alpha(tl)
    g = gibbs_gig_harris(tl, alpha_init=a["alpha_acf"], epsilon=1e-5,
                         n_iter=5000, burn_in=2000, rng=np.random.default_rng(gseed))
    sim_old = simulate_predictive_sf_harris(tl, tev, g, g["alpha"], Q_type="empirical",
                                            n_sim=2000, rng=np.random.default_rng(sseed))
    sim_vec = simulate_predictive_sf_harris_vec(tl, tev, g, g["alpha"], Q_type="empirical",
                                                n_sim=2000, rng=np.random.default_rng(sseed))
    return aad(compute_coverage(tev, sim_old, PROB)), aad(compute_coverage(tev, sim_vec, PROB))


def dollar_pair(gseed, sseed):
    df = load_ibm_data()
    dol = build_dollar_bars_rolling(df, lookback_days=30, bars_per_day=26)
    log = np.log((dol ** 2).clip(lower=1e-20)).replace([np.inf, -np.inf], np.nan).dropna()
    tr = log[log.index <= CUTOFF].values
    te = log[log.index > CUTOFF].values
    a = estimate_alpha(tr)
    g = gibbs_gig_harris(tr, alpha_init=a["alpha_acf"], epsilon=0.1,
                         n_iter=5000, burn_in=2000, rng=np.random.default_rng(gseed))
    sim_old = simulate_predictive_sf_harris(tr, te, g, g["alpha"], Q_type="empirical",
                                            n_sim=2000, rng=np.random.default_rng(sseed))
    sim_vec = simulate_predictive_sf_harris_vec(tr, te, g, g["alpha"], Q_type="empirical",
                                                n_sim=2000, rng=np.random.default_rng(sseed))
    return aad(compute_coverage(te, sim_old, PROB)), aad(compute_coverage(te, sim_vec, PROB))


print("=" * 60)
print("VERIFY simulate_predictive_sf_harris_vec vs loop")
print("=" * 60)
print(f"  {'config':>14} {'seeds':>10} {'AAD old':>9} {'AAD vec':>9} {'Delta':>8}")
print("  " + "-" * 54)
for gs, ss in SEEDS:
    t0 = time.perf_counter()
    o, v = calendar_pair(gs, ss)
    print(f"  {'calendar 15m':>14} {f'{gs}/{ss}':>10} {o:9.3f} {v:9.3f} {v-o:+8.3f}"
          f"   [{time.perf_counter()-t0:.0f}s]")
for gs, ss in SEEDS:
    t0 = time.perf_counter()
    o, v = dollar_pair(gs, ss)
    print(f"  {'dollar ~24/d':>14} {f'{gs}/{ss}':>10} {o:9.3f} {v:9.3f} {v-o:+8.3f}"
          f"   [{time.perf_counter()-t0:.0f}s]")