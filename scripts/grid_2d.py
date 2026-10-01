"""2D grid: AAD over (bars/day x top-removal %) for calendar and dollar.

Same top-removal mechanism for both (drop the largest-|r| bars, threshold from
train applied to train+test). Each construction uses its NATIVE downstream
pipeline: calendar eps=1e-5 + f_t (seeds 42/123); dollar eps=0.1 (seeds 55/321).
Uses the vectorized simulate (simulate_predictive_sf_harris_vec).

Dumps docs/grid_2d.json and prints RESULT_JSON.
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from anzarut_replication import (
    load_ibm_data, split_returns_by_date, estimate_periodicity, estimate_alpha,
    gibbs_gig_harris, simulate_predictive_sf_harris_vec, compute_coverage,
    build_dollar_bars_rolling,
)
from anzarut_intraday_15min import compute_15min_spot_volatility, PROB_LEVELS

PROB = [0.25, 0.50, 0.75, 0.85, 0.90, 0.95]
CUTOFF = pd.Timestamp("2014-05-28")
PCTS = [0.0, 0.08, 0.5, 1.0, 2.5, 5.0]
CAL_MINUTES = [30, 15, 5, 3, 1]          # bars/day 13, 26, 78, 130, 390
DOL_TARGETS = [13, 26, 40, 101, 200]     # -> ~8.5, ~17, 24, 54, 92 /day

df = load_ibm_data()


def aad(cov):
    return float(np.mean([abs(cov[p] - p * 100) for p in PROB]))


def calendar_grid():
    out = []
    for minutes in CAL_MINUTES:
        close = df["close"].resample(f"{minutes}min").last().dropna()
        returns = np.log(close).diff().dropna()
        tr_ret, te_ret = split_returns_by_date(returns, train_frac=0.8)
        ar_tr, ar_te = np.abs(tr_ret.values), np.abs(te_ret.values)
        bpd = round(390.0 / minutes, 1)
        for pct in PCTS:
            t0 = time.perf_counter()
            if pct == 0.0:
                trc, tec = tr_ret, te_ret
            else:
                thr = np.percentile(ar_tr, 100 - pct)
                trc, tec = tr_ret[ar_tr < thr], te_ret[ar_te < thr]
            period = estimate_periodicity(trc)
            lt, _ = compute_15min_spot_volatility(trc, period)
            et, _ = compute_15min_spot_volatility(tec, period)
            tl, tev = lt.values, et.values
            a = estimate_alpha(tl)
            g = gibbs_gig_harris(tl, alpha_init=a["alpha_acf"], epsilon=1e-5,
                                 n_iter=5000, burn_in=2000, rng=np.random.default_rng(42))
            sim = simulate_predictive_sf_harris_vec(tl, tev, g, g["alpha"], Q_type="empirical",
                                                    n_sim=2000, rng=np.random.default_rng(123))
            cov = compute_coverage(tev, sim, PROB)
            rec = {"bpd": bpd, "minutes": minutes, "pct": pct, "aad": aad(cov),
                   "p_stay": float(np.exp(-np.mean(g["alpha"]))),
                   "n_train": len(tl), "n_test": len(tev)}
            out.append(rec)
            print(f"  CAL {minutes:>3}min ({bpd:>5.1f}/d)  pct={pct:>4}%  AAD={rec['aad']:.3f}"
                  f"  p_stay={rec['p_stay']:.3f}  [{time.perf_counter()-t0:.0f}s]", flush=True)
    return out


def dollar_grid():
    out = []
    for target in DOL_TARGETS:
        t0 = time.perf_counter()
        dol = build_dollar_bars_rolling(df, lookback_days=30, bars_per_day=target)
        ndays = len(dol.index.to_series().dt.date.unique())
        bpd = len(dol) / max(ndays, 1)
        log = np.log((dol ** 2).clip(lower=1e-20)).replace([np.inf, -np.inf], np.nan).dropna()
        tr_full = log[log.index <= CUTOFF].values
        te_full = log[log.index > CUTOFF].values
        print(f"  DOL target={target:>4} -> {bpd:.1f}/d  build={time.perf_counter()-t0:.0f}s", flush=True)
        for pct in PCTS:
            t1 = time.perf_counter()
            if pct == 0.0:
                tr, te = tr_full, te_full
            else:
                qhi = np.percentile(tr_full, 100 - pct)
                tr = tr_full[tr_full < qhi]
                te = te_full[te_full < qhi]
            a = estimate_alpha(tr)
            g = gibbs_gig_harris(tr, alpha_init=a["alpha_acf"], epsilon=0.1,
                                 n_iter=5000, burn_in=2000, rng=np.random.default_rng(55))
            sim = simulate_predictive_sf_harris_vec(tr, te, g, g["alpha"], Q_type="empirical",
                                                    n_sim=2000, rng=np.random.default_rng(321))
            cov = compute_coverage(te, sim, PROB)
            rec = {"bpd": round(bpd, 2), "target": target, "pct": pct, "aad": aad(cov),
                   "p_stay": float(np.exp(-np.mean(g["alpha"]))),
                   "n_train": len(tr), "n_test": len(te)}
            out.append(rec)
            print(f"      pct={pct:>4}%  AAD={rec['aad']:.3f}  p_stay={rec['p_stay']:.3f}"
                  f"  n={len(tr)}/{len(te)}  [{time.perf_counter()-t1:.0f}s]", flush=True)
    return out


if __name__ == "__main__":
    print("=== CALENDAR GRID ===", flush=True)
    cal = calendar_grid()
    print("=== DOLLAR GRID ===", flush=True)
    dol = dollar_grid()
    out = {"pcts": PCTS, "calendar": cal, "dollar": dol}
    Path("docs").mkdir(exist_ok=True)
    with open("docs/grid_2d.json", "w") as f:
        json.dump(out, f, indent=1)
    print("RESULT_JSON " + json.dumps(out))