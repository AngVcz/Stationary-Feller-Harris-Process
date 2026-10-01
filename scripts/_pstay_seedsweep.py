"""Seed sweep over p_stay for the 5 plateau models.
Captures the MC noise (forward-simulate layer) by varying the sim seed across
N_SEEDS runs per (model, p_stay); Gibbs fit once per model (seed 42) and held fixed,
so the sweep isolates simulate noise -- consistent with the grid's mc_se (batch
of one sim). Reports AAD mean +/- std per cell.

Logs incrementally to docs/pstay_seed_sweep_log.txt so partial results survive.
"""
import sys, io, contextlib, math, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from anzarut_replication import (
    load_ibm_data, split_returns_by_date, estimate_periodicity, estimate_alpha,
    gibbs_gig_harris, simulate_predictive_sf_harris_vec, compute_coverage,
)
from anzarut_intraday_15min import compute_15min_spot_volatility
from grid_is_oos import calendar_bars_n_per_day

PROB = [0.25, 0.50, 0.75, 0.85, 0.90, 0.95]
PCT = 0.08
PSTAYS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
N_SEEDS = 8
NSIM = 2000
MODELS = [
    (8, "groupby", "~49min"),
    (13, "resample30", "30min"),
    (26, "resample15", "15min"),
    (78, "resample5", "5min"),
    # 390 (1-min) omitted: ~29 min for 8 seeds, exceeds 10-min timeout.
    # Its table row was already flat (0.566/0.541/0.522/0.557/0.531/0.570).
]
df = load_ibm_data()
LOG = Path("docs/pstay_seed_sweep_log.txt")
LOG.write_text("")  # reset


def log(msg):
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def aad(sim, tev):
    m = min(len(tev), sim.shape[1])
    cov = compute_coverage(tev[:m], sim[:, :m], PROB)
    return float(np.mean([abs(cov[p] - p * 100) for p in PROB]))


def build_log_spot(bpd, kind):
    if kind.startswith("resample"):
        minutes = int(kind.replace("resample", ""))
        close = df["close"].resample(f"{minutes}min").last().dropna()
    else:
        close = calendar_bars_n_per_day(df["close"], bpd)
    ret = np.log(close).diff().dropna()
    tr, te = split_returns_by_date(ret, train_frac=0.8)
    ar_tr, ar_te = np.abs(tr.values), np.abs(te.values)
    thr = np.percentile(ar_tr, 100 - PCT)
    trc, tec = tr[ar_tr < thr], te[ar_te < thr]
    period = estimate_periodicity(trc)
    lt, _ = compute_15min_spot_volatility(trc, period)
    et, _ = compute_15min_spot_volatility(tec, period)
    return lt.values, et.values


results = {}
for bpd, kind, label in MODELS:
    t0 = time.perf_counter()
    tl, tev = build_log_spot(bpd, kind)
    a = estimate_alpha(tl)
    with contextlib.redirect_stdout(io.StringIO()):
        g = gibbs_gig_harris(tl, alpha_init=a["alpha_acf"], epsilon=1e-5,
                             n_iter=5000, burn_in=2000, rng=np.random.default_rng(42))
    n_post = len(g["alpha"])
    log(f"\n## bpd={bpd} {label}  n_test={len(tev)}  fitted_pstay={np.mean(np.exp(-g['alpha'])):.3f}  [fit {time.perf_counter()-t0:.0f}s]")
    results[bpd] = {}
    for ps in PSTAYS:
        alpha_post = np.full(n_post, 100.0) if ps == 0.0 else np.full(n_post, -math.log(ps))
        aads = []
        for i in range(N_SEEDS):
            sim = simulate_predictive_sf_harris_vec(tl, tev, g, alpha_post, Q_type="empirical",
                                                    n_sim=NSIM, rng=np.random.default_rng(123 + i))
            aads.append(aad(sim, tev))
        arr = np.array(aads)
        results[bpd][ps] = (arr.mean(), arr.std(ddof=1), arr)
        log(f"  p_stay={ps:<3}  mean={arr.mean():.3f}  std={arr.std(ddof=1):.3f}  "
            f"min={arr.min():.3f} max={arr.max():.3f}  | {' '.join(f'{x:.2f}' for x in aads)}")

# summary table
log("\n" + "=" * 90)
log(f"{'bpd':>4} {'bar':>7} " + " ".join(f"p={ps:<3}mean/std".rjust(11) for ps in PSTAYS))
for bpd, _, label in MODELS:
    cells = []
    for ps in PSTAYS:
        mu, sd, _ = results[bpd][ps]
        cells.append(f"{mu:.3f}±{sd:.3f}")
    log(f"{bpd:>4} {label:>7} " + " ".join(f"{c:>11}" for c in cells))
# average row
log(f"{'avg':>4} {'':>7} " + " ".join(f"{np.mean([results[b][ps][0] for b in [m[0] for m in MODELS]]):.3f}±{np.mean([results[b][ps][1] for b in [m[0] for m in MODELS]]):.3f}".rjust(11) for ps in PSTAYS))
log("\n(std = MC noise across 8 sim seeds; if |col diff| < ~2*mean_std -> p_stay decorative)")