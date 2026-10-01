"""Isolar si la diferencia dollar-vs-calendar es SOLO la fraccion de cleaning.

Dollar bars rolling (~24/dia, leak-free, matching calendar 15-min ~26/dia),
mismo pipeline que dollar_fair_compare.py, barriendo SOLO la fraccion removida
del extremo superior (saltos grandes), mismo mecanismo que el BNS top-quantile
del calendar. Reporta AAD y % removido para comparar contra calendar 15-min
(0.08% removido -> AAD 0.521 clean / 0.460 no-clean).
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from anzarut_replication import (
    load_ibm_data, build_dollar_bars_rolling, estimate_alpha,
    gibbs_gig_harris, simulate_predictive_sf_harris, compute_coverage,
)
from anzarut_intraday_15min import PROB_LEVELS

PROB = [0.25, 0.50, 0.75, 0.85, 0.90, 0.95]
CUTOFF = pd.Timestamp("2014-05-28")

df = load_ibm_data()
dol = build_dollar_bars_rolling(df, lookback_days=30, bars_per_day=26)
ndays = len(dol.index.to_series().dt.date.unique())
log = np.log((dol ** 2).clip(lower=1e-20)).replace([np.inf, -np.inf], np.nan).dropna()
tr_full = log[log.index <= CUTOFF].values
te_full = log[log.index > CUTOFF].values


def aad_of(clean_mode):
    """clean_mode: 'none' | ('top', pct)  -> drop top pct of train by log-r2 value."""
    if clean_mode == "none":
        tr, te = tr_full, te_full
    else:
        _, pct = clean_mode
        qhi = np.percentile(tr_full, 100 - pct)
        tr = tr_full[tr_full < qhi]
        te = te_full[te_full < qhi]
    n_rm_tr = len(tr_full) - len(tr)
    n_rm_te = len(te_full) - len(te)
    alpha = estimate_alpha(tr)
    g = gibbs_gig_harris(tr, alpha_init=alpha["alpha_acf"], epsilon=0.1,
                         n_iter=5000, burn_in=2000, rng=np.random.default_rng(55))
    sim = simulate_predictive_sf_harris(tr, te, g, g["alpha"], Q_type="empirical",
                                        n_sim=2000, rng=np.random.default_rng(321))
    cov = compute_coverage(te, sim, PROB)
    aad = float(np.mean([abs(cov[p] - p * 100) for p in PROB]))
    pct_rm = 100 * (n_rm_tr + n_rm_te) / (len(tr_full) + len(te_full))
    return {"aad": aad, "pct_removed": pct_rm,
            "n_train": len(tr), "n_test": len(te),
            "n_rm_train": n_rm_tr, "n_rm_test": n_rm_te,
            "p_stay": float(np.exp(-np.mean(g["alpha"])))}


modes = [
    ("none", "none"),
    ("top0.08%", ("top", 0.08)),   # match calendar's 0.08%
    ("top0.5%",  ("top", 0.5)),
    ("top1%",    ("top", 1.0)),
    ("top2.5%",  ("top", 2.5)),
    ("top5%",    ("top", 5.0)),
]
out = {"bars_per_day": len(dol) / ndays, "cutoff": str(CUTOFF.date()),
       "n_full_train": len(tr_full), "n_full_test": len(te_full),
       "calendar_15min_ref": {"aad_clean": 0.521, "aad_noclean": 0.460, "pct_removed": 0.08},
       "arms": {}}
for name, mode in modes:
    r = aad_of(mode)
    out["arms"][name] = r
    print(f"  {name:>10}: AAD={r['aad']:.3f}  removed={r['pct_removed']:.2f}%  "
          f"p_stay={r['p_stay']:.3f}  n_tr/te={r['n_train']}/{r['n_test']}")
print("RESULT_JSON " + json.dumps(out))