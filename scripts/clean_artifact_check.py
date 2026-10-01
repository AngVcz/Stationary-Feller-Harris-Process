"""Is the '%-cleaning improves AAD' trend stable or a test-trimming artifact?

Standard pipeline cleans BOTH train and test with the same threshold (removes the
top-pct |r| bars from each). Those removed test bars are exactly the volatile
outliers the model miscalibrates -> removing them makes AAD drop *by construction*,
not because the model got better.

Decisive check: fit on cleaned train, but evaluate coverage on the FULL test (no
test cleaning). If AAD_full stays flat/rises with pct while AAD_clean drops, the
trend is an artifact of trimming the test set.

Cells: dollar target=13 (12.41/d) and calendar 30-min (13/d) at pct in {0,1,5,10}.
"""
import sys, math, json
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
PCTS = [0.0, 1.0, 5.0, 10.0]
df = load_ibm_data()


def aad(cov):
    return float(np.mean([abs(cov[p] - p * 100) for p in PROB]))


def calendar_run(pct, eval_full):
    close = df["close"].resample("30min").last().dropna()
    returns = np.log(close).diff().dropna()
    tr_ret, te_ret = split_returns_by_date(returns, train_frac=0.8)
    ar_tr = np.abs(tr_ret.values)
    thr = np.percentile(ar_tr, 100 - pct) if pct > 0 else np.inf
    trc = tr_ret[ar_tr < thr] if pct > 0 else tr_ret
    te_eval = te_ret if eval_full else (te_ret[np.abs(te_ret.values) < thr] if pct > 0 else te_ret)
    period = estimate_periodicity(trc)
    lt, _ = compute_15min_spot_volatility(trc, period)
    et, _ = compute_15min_spot_volatility(te_eval, period)
    tl, tev = lt.values, et.values
    a = estimate_alpha(tl)
    g = gibbs_gig_harris(tl, alpha_init=a["alpha_acf"], epsilon=1e-5,
                         n_iter=5000, burn_in=2000, rng=np.random.default_rng(42))
    sim = simulate_predictive_sf_harris_vec(tl, tev, g, g["alpha"], Q_type="empirical",
                                            n_sim=2000, rng=np.random.default_rng(123))
    return aad(compute_coverage(tev, sim, PROB)), len(tev), len(te_ret)


def dollar_run(pct, eval_full):
    dol = build_dollar_bars_rolling(df, lookback_days=30, bars_per_day=13)
    log = np.log((dol ** 2).clip(lower=1e-20)).replace([np.inf, -np.inf], np.nan).dropna()
    tr_full = log[log.index <= CUTOFF].values
    te_full = log[log.index > CUTOFF].values
    qhi = np.percentile(tr_full, 100 - pct) if pct > 0 else np.inf
    tr = tr_full[tr_full < qhi] if pct > 0 else tr_full
    te = te_full if eval_full else (te_full[te_full < qhi] if pct > 0 else te_full)
    a = estimate_alpha(tr)
    g = gibbs_gig_harris(tr, alpha_init=a["alpha_acf"], epsilon=0.1,
                         n_iter=5000, burn_in=2000, rng=np.random.default_rng(55))
    sim = simulate_predictive_sf_harris_vec(tr, te, g, g["alpha"], Q_type="empirical",
                                            n_sim=2000, rng=np.random.default_rng(321))
    return aad(compute_coverage(te, sim, PROB)), len(te), len(te_full)


print("=" * 90)
print("Cleaning artifact check: AAD on CLEANED test (standard) vs FULL test (no test trimming)")
print("If AAD_full does NOT improve with pct while AAD_clean drops -> artifact of test trimming.")
print("=" * 90)
for name, fn in [("cal 30min(13/d)", calendar_run), ("dol 13(12.41/d)", dollar_run)]:
    print(f"\n## {name}")
    print(f"  {'pct':>5} {'n_clean':>8} {'n_full':>8} {'AAD_clean':>10} {'AAD_full':>10} {'drop(clean)':>12} {'drop(full)':>11}")
    base_clean = base_full = None
    for pct in PCTS:
        aad_c, n_c, n_f = fn(pct, eval_full=False)
        aad_f, _, _ = fn(pct, eval_full=True)
        if pct == 0.0:
            base_clean, base_full = aad_c, aad_f
        dc = aad_c - base_clean
        df_ = aad_f - base_full
        print(f"  {pct:>5.1f} {n_c:>8} {n_f:>8} {aad_c:>10.3f} {aad_f:>10.3f} {dc:>+12.3f} {df_:>+11.3f}")