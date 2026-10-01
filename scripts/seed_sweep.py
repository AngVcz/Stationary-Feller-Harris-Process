"""Seed/MC stability sweep on the top-tier cells.

Vary the Gibbs seed and the simulate seed across N runs; report the spread of
AAD (cleaned test, standard metric) per cell. A cell whose AAD rank flips or
whose std is large across seeds is MC-fragile. Reference: the grid used seeds
42/123 (cal) / 55/321 (dol); here we vary seeds to get the MC distribution.

Top-tier (from robustness analysis): cal 13/d 1%, cal 13/d 5%, cal 26/d 1%,
dol 12.41/d 10%. NOTE: dol 12.41/d 10% is an artifact of test trimming (see
clean_artifact_check.py) — included to show its seed-stability nonetheless.
"""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from anzarut_replication import (
    load_ibm_data, split_returns_by_date, estimate_periodicity, estimate_alpha,
    gibbs_gig_harris, simulate_predictive_sf_harris_vec, compute_coverage,
    build_dollar_bars_rolling,
)
from anzarut_intraday_15min import compute_15min_spot_volatility

PROB = [0.25, 0.50, 0.75, 0.85, 0.90, 0.95]
CUTOFF = pd.Timestamp("2014-05-28")
N_SEEDS = 8
CELLS = [
    {"name": "cal 13/d 1%",  "kind": "cal", "minutes": 30, "pct": 1.0},
    {"name": "cal 13/d 5%",  "kind": "cal", "minutes": 30, "pct": 5.0},
    {"name": "cal 26/d 1%",  "kind": "cal", "minutes": 15, "pct": 1.0},
    {"name": "dol 12.41/d 10%", "kind": "dol", "target": 13, "pct": 10.0},
]
df = load_ibm_data()


def aad(cov):
    return float(np.mean([abs(cov[p] - p * 100) for p in PROB]))


def cal_aad(cell, gseed, sseed):
    close = df["close"].resample(f"{cell['minutes']}min").last().dropna()
    returns = np.log(close).diff().dropna()
    tr_ret, te_ret = split_returns_by_date(returns, train_frac=0.8)
    pct = cell["pct"]
    ar_tr = np.abs(tr_ret.values)
    thr = np.percentile(ar_tr, 100 - pct) if pct > 0 else np.inf
    trc = tr_ret[ar_tr < thr] if pct > 0 else tr_ret
    tec = te_ret[np.abs(te_ret.values) < thr] if pct > 0 else te_ret
    period = estimate_periodicity(trc)
    lt, _ = compute_15min_spot_volatility(trc, period)
    et, _ = compute_15min_spot_volatility(tec, period)
    tl, tev = lt.values, et.values
    a = estimate_alpha(tl)
    g = gibbs_gig_harris(tl, alpha_init=a["alpha_acf"], epsilon=1e-5,
                         n_iter=5000, burn_in=2000, rng=np.random.default_rng(gseed))
    sim = simulate_predictive_sf_harris_vec(tl, tev, g, g["alpha"], Q_type="empirical",
                                            n_sim=2000, rng=np.random.default_rng(sseed))
    return aad(compute_coverage(tev, sim, PROB))


def dol_aad(cell, gseed, sseed):
    dol = build_dollar_bars_rolling(df, lookback_days=30, bars_per_day=cell["target"])
    log = np.log((dol ** 2).clip(lower=1e-20)).replace([np.inf, -np.inf], np.nan).dropna()
    tr_full = log[log.index <= CUTOFF].values
    te_full = log[log.index > CUTOFF].values
    pct = cell["pct"]
    qhi = np.percentile(tr_full, 100 - pct) if pct > 0 else np.inf
    tr = tr_full[tr_full < qhi] if pct > 0 else tr_full
    te = te_full[te_full < qhi] if pct > 0 else te_full
    a = estimate_alpha(tr)
    g = gibbs_gig_harris(tr, alpha_init=a["alpha_acf"], epsilon=0.1,
                         n_iter=5000, burn_in=2000, rng=np.random.default_rng(gseed))
    sim = simulate_predictive_sf_harris_vec(tr, te, g, g["alpha"], Q_type="empirical",
                                            n_sim=2000, rng=np.random.default_rng(sseed))
    return aad(compute_coverage(te, sim, PROB))


print("=" * 78)
print(f"Seed/MC stability sweep — N={N_SEEDS} seed pairs per cell (AAD on cleaned test)")
print("seeds: gibbs=42+i, sim=123+i (cal) / gibbs=55+i, sim=321+i (dol)")
print("=" * 78)
print(f"  {'cell':>18} {'mean':>7} {'std':>7} {'min':>7} {'max':>7} {'range':>7}  per-seed AAD")
for cell in CELLS:
    aads = []
    for i in range(N_SEEDS):
        if cell["kind"] == "cal":
            a = cal_aad(cell, 42 + i, 123 + i)
        else:
            a = dol_aad(cell, 55 + i, 321 + i)
        aads.append(a)
    arr = np.array(aads)
    traj = " ".join(f"{x:.3f}" for x in aads)
    print(f"  {cell['name']:>18} {arr.mean():>7.3f} {arr.std():>7.3f} "
          f"{arr.min():>7.3f} {arr.max():>7.3f} {arr.max()-arr.min():>7.3f}  {traj}")