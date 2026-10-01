"""Walk-forward recalibration of SF-Harris (incl. Q-emp) vs static fit-once.

Prediction tiles the timeline in non-overlapping L-day blocks (recalibrate every L
days, strictly causal). The CALIBRATION window per block is set by cal_mode:
  block      : cal = previous L days              (1:1, the original spec)
  expanding  : cal = all history [0, t)           (grows each block)
  rolling    : cal = trailing W=600 days [t-W, t) (fixed ~train-size window)
L in {30,100,252}. Comparison is on the SAME post-CUTOFF test bars as the static
baseline (fit once on pre-CUTOFF, predict all post-CUTOFF), so AAD_static and
AAD_walk are evaluated on identical post-CUTOFF bars.

Cells: calendar 15-min/0.08% (headline, baseline 0.547) and 30-min/1% (argmin, 0.208).
Prints comparison table + per-block AAD trajectory.
"""
import sys, math, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from anzarut_replication import (
    load_ibm_data, estimate_periodicity, estimate_alpha,
    gibbs_gig_harris, simulate_predictive_sf_harris_vec,
)
from anzarut_intraday_15min import compute_15min_spot_volatility

PROB = [0.25, 0.50, 0.75, 0.85, 0.90, 0.95]
CUTOFF = pd.Timestamp("2014-05-28")
LS = [30, 100, 252]
W_ROLL = 600
SCHEMES = ["expanding"]
CELLS = [(15, 0.08, "15min/0.08% (headline)"),
         (30, 1.0, "30min/1% (argmin cal)")]

df = load_ibm_data()


def aad_from_counts(counts):
    return float(np.mean([abs(c[0] / max(c[1], 1) * 100 - p * 100) for p, c in counts.items()]))


def fit_predict_counts(cal_ret, pred_ret, pct):
    """Fit SF-Harris on cal_ret (calendar eps=1e-5, seeds 42/123), predict pred_ret.
    Returns coverage counts {p:(inside,n)} on pred and n_pred."""
    ar_cal = np.abs(cal_ret.values)
    thr = np.percentile(ar_cal, 100 - pct) if pct > 0 else np.inf
    trc = cal_ret[ar_cal < thr] if pct > 0 else cal_ret
    if len(trc) < 50:
        raise ValueError(f"cal too short after clean ({len(trc)})")
    ar_pr = np.abs(pred_ret.values)
    tec = pred_ret[ar_pr < thr] if pct > 0 else pred_ret
    period = estimate_periodicity(trc)
    lt, _ = compute_15min_spot_volatility(trc, period)
    et, _ = compute_15min_spot_volatility(tec, period)
    tl, tev = lt.values, et.values
    a = estimate_alpha(tl)
    g = gibbs_gig_harris(tl, alpha_init=a["alpha_acf"], epsilon=1e-5,
                         n_iter=5000, burn_in=2000, rng=np.random.default_rng(42))
    sim = simulate_predictive_sf_harris_vec(tl, tev, g, g["alpha"], Q_type="empirical",
                                            n_sim=2000, rng=np.random.default_rng(123))
    m = min(len(tev), sim.shape[1])
    tev_a = tev[:m]; sim_a = sim[:, :m]
    counts = {}
    for p in PROB:
        lo = np.quantile(sim_a, (1 - p) / 2, axis=0)
        hi = np.quantile(sim_a, (1 + p) / 2, axis=0)
        counts[p] = (int(np.sum((tev_a >= lo) & (tev_a <= hi))), m)
    return counts, m, float(np.exp(-np.mean(g["alpha"])))


def get_returns(minutes):
    close = df["close"].resample(f"{minutes}min").last().dropna()
    ret = np.log(close).diff().dropna()
    days = sorted(ret.index.normalize().unique())
    by_day = {d: ret[ret.index.normalize() == d] for d in days}
    return ret, days, by_day


def static_baseline(days, by_day, pct):
    cal_ret = pd.concat([by_day[d] for d in days if d <= CUTOFF])
    pred_ret = pd.concat([by_day[d] for d in days if d > CUTOFF])
    counts, m, ps = fit_predict_counts(cal_ret, pred_ret, pct)
    return aad_from_counts(counts), m, ps


def walk_forward(days, by_day, pct, L, scheme):
    N = len(days)
    pooled = {p: [0, 0] for p in PROB}
    block_aads = []
    n_blocks = 0
    # expanding-from-600: anchor fixed so the FIRST post-CUTOFF block has 600 cal days,
    # then the window grows from that anchor (differs from rolling, which stays at 600).
    anchor = None
    if scheme == "expanding":
        b0 = 0
        while (b0 + 1) * L < N:
            ps0 = (b0 + 1) * L
            pdays0 = days[ps0:ps0 + L]
            if len(pd.concat([by_day[d] for d in pdays0]).loc[lambda s: s.index > CUTOFF]) >= 10:
                anchor = max(0, ps0 - W_ROLL)
                break
            b0 += 1
    b = 0
    while (b + 1) * L < N:
        pred_start = (b + 1) * L
        if scheme == "block":
            cal_start = max(0, pred_start - L)
        elif scheme == "expanding":
            cal_start = anchor if anchor is not None else 0
        else:  # rolling
            cal_start = max(0, pred_start - W_ROLL)
        cal_days = days[cal_start:pred_start]
        pred_days = days[pred_start:pred_start + L]
        pred_ret = pd.concat([by_day[d] for d in pred_days])
        pred_post = pred_ret[pred_ret.index > CUTOFF]
        if len(pred_post) < 10:
            b += 1
            continue
        if len(cal_days) < 5:
            b += 1
            continue
        cal_ret = pd.concat([by_day[d] for d in cal_days])
        try:
            counts, m, ps = fit_predict_counts(cal_ret, pred_post, pct)
        except Exception as e:
            block_aads.append(("err", str(e)[:45]))
            b += 1
            continue
        for p in PROB:
            pooled[p][0] += counts[p][0]
            pooled[p][1] += counts[p][1]
        block_aads.append((f"b{b} cal={cal_days[0].date()}..{cal_days[-1].date()}({len(cal_days)}d) "
                           f"n={m} ps={ps:.3f}", round(aad_from_counts(counts), 3)))
        n_blocks += 1
        b += 1
    return aad_from_counts(pooled), n_blocks, pooled, block_aads


print("=" * 96)
print("Walk-forward recalibration vs static fit-once — same post-CUTOFF bars")
print("Refit periodicity+Gibbs(alpha/mu/sigma)+Q-emp per block. cal_mode: block(prev L) / expanding / rolling600.")
print("=" * 96)
for minutes, pct, label in CELLS:
    ret, days, by_day = get_returns(minutes)
    t0 = time.perf_counter()
    aad_st, n_st, ps_st = static_baseline(days, by_day, pct)
    print(f"\n## {label}  ({minutes}min bars, clean {pct}%)")
    print(f"   STATIC  fit-once pre-CUTOFF: AAD={aad_st:.3f}  n_pred={n_st}  p_stay={ps_st:.3f}  [{time.perf_counter()-t0:.0f}s]")
    for scheme in SCHEMES:
        tag = {"block": "block(prev L)", "expanding": f"expanding(anchor600)", "rolling": f"rolling-{W_ROLL}d"}[scheme]
        print(f"   -- cal={scheme}  ({tag})")
        print(f"   {'L':>4} {'n_blocks':>8} {'n_pred':>7} {'AAD_walk':>9} {'AAD_stat':>9} {'delta':>7}  per-block AAD")
        for L in LS:
            t1 = time.perf_counter()
            aad_wf, nb, pooled, ba = walk_forward(days, by_day, pct, L, scheme)
            n_wf = pooled[PROB[0]][1]
            delta = aad_wf - aad_st
            traj = " ".join(f"{v:.2f}" if isinstance(v, float) else f"[{v[1]}]" for _, v in ba)
            print(f"   {L:>4} {nb:>8} {n_wf:>7} {aad_wf:>9.3f} {aad_st:>9.3f} {delta:>+7.3f}  {traj}  [{time.perf_counter()-t1:.0f}s]")