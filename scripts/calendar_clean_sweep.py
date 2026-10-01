"""Calendar 15-min sweep al Mismo % de removal que dollar_clean_sweep.

Top-removal por |return| (mismo criterio que el dollar: tirar las barras de
mayor |r|). Pipeline nativo del calendar: estimate_periodicity(train limpio) ->
compute_15min_spot_volatility -> Gibbs eps=1e-5 rng=42 -> sim rng=123.
Referencia: ablation run(clean=True/False) = 0.521 / 0.460 (BNS top 0.1%).
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from anzarut_replication import (
    load_ibm_data, compute_15min_returns, split_returns_by_date,
    estimate_periodicity, estimate_alpha, gibbs_gig_harris,
    simulate_predictive_sf_harris, compute_coverage,
)
from anzarut_intraday_15min import compute_15min_spot_volatility, PROB_LEVELS

PROB = [0.25, 0.50, 0.75, 0.85, 0.90, 0.95]

df = load_ibm_data()
returns = compute_15min_returns(df)
train_ret, test_ret = split_returns_by_date(returns, train_frac=0.8)
ar_tr, ar_te = np.abs(train_ret.values), np.abs(test_ret.values)


def aad_of(clean_mode):
    if clean_mode == "none":
        trc, tec = train_ret, test_ret
        n_rm = 0
    else:
        pct = clean_mode
        thr = np.percentile(ar_tr, 100 - pct)
        mtr = ar_tr < thr
        mte = ar_te < thr
        trc, tec = train_ret[mtr], test_ret[mte]
        n_rm = (~mtr).sum() + (~mte).sum()
    period = estimate_periodicity(trc)
    log_tr, _ = compute_15min_spot_volatility(trc, period)
    log_te, _ = compute_15min_spot_volatility(tec, period)
    tl, te = log_tr.values, log_te.values
    alpha = estimate_alpha(tl)
    g = gibbs_gig_harris(tl, alpha_init=alpha["alpha_acf"], epsilon=1e-5,
                         n_iter=5000, burn_in=2000, rng=np.random.default_rng(42))
    sim = simulate_predictive_sf_harris(tl, te, g, g["alpha"], Q_type="empirical",
                                        n_sim=2000, rng=np.random.default_rng(123))
    cov = compute_coverage(te, sim, PROB)
    aad = float(np.mean([abs(cov[p] - p * 100) for p in PROB]))
    n_tot = len(train_ret) + len(test_ret)
    return {"aad": aad, "pct_removed": 100 * n_rm / n_tot,
            "n_train": len(tl), "n_test": len(te),
            "p_stay": float(np.exp(-np.mean(g["alpha"])))}


modes = [
    ("none", "none"),
    ("top0.08%", 0.08),   # ~ BNS calendar headline
    ("top0.5%",  0.5),
    ("top1%",    1.0),
    ("top2.5%",  2.5),
    ("top5%",    5.0),
]
out = {"construction": "calendar_15min", "train_frac": 0.8,
       "n_full_train": len(train_ret), "n_full_test": len(test_ret),
       "bns_ref": {"aad_clean": 0.521, "aad_noclean": 0.460, "pct_removed": 0.08},
       "arms": {}}
for name, mode in modes:
    r = aad_of(mode)
    out["arms"][name] = r
    print(f"  {name:>10}: AAD={r['aad']:.3f}  removed={r['pct_removed']:.2f}%  "
          f"p_stay={r['p_stay']:.3f}  n_tr/te={r['n_train']}/{r['n_test']}")
print("RESULT_JSON " + json.dumps(out))