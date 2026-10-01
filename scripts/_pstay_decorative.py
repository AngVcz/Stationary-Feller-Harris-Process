"""Is p_stay decorative for coverage? Compare fitted alpha (p_stay~0.368) vs
forced p_stay=0 (always jump -> iid Q) vs forced p_stay=1 (always stay) on the
plateau cell cal 26/d 0.08%."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from anzarut_replication import (
    load_ibm_data, split_returns_by_date, estimate_periodicity, estimate_alpha,
    gibbs_gig_harris, simulate_predictive_sf_harris_vec, compute_coverage,
)
from anzarut_intraday_15min import compute_15min_spot_volatility

PROB = [0.25, 0.50, 0.75, 0.85, 0.90, 0.95]
df = load_ibm_data()
PCT = 0.08
close = df["close"].resample("15min").last().dropna()
ret = np.log(close).diff().dropna()
tr, te = split_returns_by_date(ret, train_frac=0.8)
ar = np.abs(tr.values)
thr = np.percentile(ar, 100 - PCT) if PCT > 0 else np.inf
trc = tr[ar < thr] if PCT > 0 else tr
tec = te[np.abs(te.values) < thr] if PCT > 0 else te
period = estimate_periodicity(trc)
lt, _ = compute_15min_spot_volatility(trc, period)
et, _ = compute_15min_spot_volatility(tec, period)
tl, tev = lt.values, et.values
a = estimate_alpha(tl)
g = gibbs_gig_harris(tl, alpha_init=a["alpha_acf"], epsilon=1e-5,
                     n_iter=5000, burn_in=2000, rng=np.random.default_rng(42))


def aad(sim):
    m = min(len(tev), sim.shape[1])
    cov = compute_coverage(tev[:m], sim[:, :m], PROB)
    return float(np.mean([abs(cov[p] - p * 100) for p in PROB]))


def cov_per_p(sim):
    m = min(len(tev), sim.shape[1])
    return compute_coverage(tev[:m], sim[:, :m], PROB)


n_post = len(g["alpha"])
RNG = np.random.default_rng(123)
# (a) fitted alpha
sim_fit = simulate_predictive_sf_harris_vec(tl, tev, g, g["alpha"], Q_type="empirical", n_sim=2000, rng=np.random.default_rng(123))
# (b) forced p_stay=0  (alpha -> huge)
alpha_zero = np.full(n_post, 100.0)
sim_p0 = simulate_predictive_sf_harris_vec(tl, tev, g, alpha_zero, Q_type="empirical", n_sim=2000, rng=np.random.default_rng(123))
# (c) forced p_stay=1 (alpha -> -inf, never jump, constant = x_0)
alpha_one = np.full(n_post, -1e6)
sim_p1 = simulate_predictive_sf_harris_vec(tl, tev, g, alpha_one, Q_type="empirical", n_sim=2000, rng=np.random.default_rng(123))

print(f"fitted alpha mean={np.mean(g['alpha']):.3f}  -> p_stay={np.mean(np.exp(-g['alpha'])):.3f}")
print(f"n_test={len(tev)}\n")
print(f"{'case':>14} {'AAD':>7}   per-p coverage (ideal in brackets)")
for name, sim in [("fitted p_stay~0.37", sim_fit), ("p_stay=0 (iid Q)", sim_p0), ("p_stay=1 (const x0)", sim_p1)]:
    c = cov_per_p(sim)
    pp = "  ".join(f"{p:.2f}:{c[p]:4.1f}" for p in PROB)
    ideal = "  ".join(f"{p*100:4.0f}" for p in PROB)
    print(f"{name:>16} {aad(sim):>6.3f}   {pp}")
print(f"{'ideal':>16} {'':>7}   {ideal}")