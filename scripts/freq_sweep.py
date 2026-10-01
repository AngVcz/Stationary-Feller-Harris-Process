"""Frequency sweep: map AAD vs bars/day for dollar and calendar pipelines.

Holds each pipeline constant and varies only the bar frequency, to separate a
*frequency artifact* (finer bars -> tighter emission noise -> better coverage)
from a *genuine activity-sampling benefit* of dollar bars.

  dollar  : build_dollar_bars_rolling(bars_per_day) -> log(r^2), eps0.1, seeds 55/321
            [same pipeline as dollar_fair_compare.py]
  calendar: returns at M-min -> clean -> periodicity(train) -> spot vol -> log,
            eps1e-5, seeds 42/123  [same pipeline as anzarut_intraday_15min.py headline]

Both split at the same date cutoff (2014-05-28, train_frac=0.8) so frequencies are
compared on identical train/test windows.

Usage: python freq_sweep.py <dollar|calendar> <int>
  dollar   : arg = bars_per_day target  (13, 26, 50, 101, ...)
  calendar : arg = minutes per bar      (15, 5, 30, ...)
Prints: RESULT_JSON {config, clean:{...}, no_clean:{...}}
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from anzarut_replication import (
    load_ibm_data, build_dollar_bars_rolling, detect_and_remove_jumps,
    fit_jump_thresholds, split_returns_by_date, estimate_periodicity,
    estimate_alpha, gibbs_gig_harris, simulate_predictive_sf_harris, compute_coverage,
)
from anzarut_intraday_15min import compute_15min_spot_volatility

PROB_LEVELS = [0.25, 0.50, 0.75, 0.85, 0.90, 0.95]
CUTOFF = pd.Timestamp("2014-05-28")


def _aad(cov):
    return float(np.mean([abs(cov[p] - p * 100) for p in PROB_LEVELS]))


def _levels(cov):
    return [{"p": p, "coverage": cov[p], "deviation": cov[p] - p * 100} for p in PROB_LEVELS]


def run_dollar(bars_per_day, clean):
    df = load_ibm_data()
    dol = build_dollar_bars_rolling(df, lookback_days=30, bars_per_day=bars_per_day)
    log = np.log((dol ** 2).clip(lower=1e-20)).replace([np.inf, -np.inf], np.nan).dropna()
    tr_full = log[log.index <= CUTOFF].values
    te_full = log[log.index > CUTOFF].values
    if clean:
        q01, q99 = np.percentile(tr_full, 0.5), np.percentile(tr_full, 99.5)
        tr = tr_full[(tr_full > q01) & (tr_full < q99)]
        te = te_full[(te_full > q01) & (te_full < q99)]
    else:
        tr, te = tr_full, te_full
    alpha = estimate_alpha(tr)
    gibbs = gibbs_gig_harris(tr, alpha_init=alpha["alpha_acf"], epsilon=0.1,
                             n_iter=5000, burn_in=2000, rng=np.random.default_rng(55))
    sim = simulate_predictive_sf_harris(tr, te, gibbs, gibbs["alpha"],
                                        Q_type="empirical", n_sim=2000,
                                        rng=np.random.default_rng(321))
    cov = compute_coverage(te, sim, PROB_LEVELS)
    span_days = max((dol.index[-1] - dol.index[0]).days, 1)
    return {"aad": _aad(cov), "levels": _levels(cov), "n_train": len(tr), "n_test": len(te),
            "n_rm_train": len(tr_full) - len(tr), "n_rm_test": len(te_full) - len(te),
            "bars_total": int(len(dol)), "bars_per_day": len(dol) / span_days,
            "alpha_post": float(np.mean(gibbs["alpha"])),
            "p_stay": float(np.exp(-np.mean(gibbs["alpha"])))}


def run_calendar(minutes, clean):
    df = load_ibm_data()
    close = df["close"].resample(f"{minutes}min").last().dropna()
    returns = np.log(close).diff().dropna()
    train_ret, test_ret = split_returns_by_date(returns, train_frac=0.8)
    if clean:
        thr = fit_jump_thresholds(train_ret, n_passes=2, top_pct=0.001)
        train_c = detect_and_remove_jumps(train_ret, fixed_thresholds=thr)
        test_c = detect_and_remove_jumps(test_ret, fixed_thresholds=thr)
    else:
        train_c = detect_and_remove_jumps(train_ret, fixed_thresholds=[])
        test_c = detect_and_remove_jumps(test_ret, fixed_thresholds=[])
    period = estimate_periodicity(train_c)
    log_train, _ = compute_15min_spot_volatility(train_c, period)
    log_test, _ = compute_15min_spot_volatility(test_c, period)
    tr, te = log_train.values, log_test.values
    alpha = estimate_alpha(tr)
    gibbs = gibbs_gig_harris(tr, alpha_init=alpha["alpha_acf"], epsilon=1e-5,
                             n_iter=5000, burn_in=2000, rng=np.random.default_rng(42))
    sim = simulate_predictive_sf_harris(tr, te, gibbs, gibbs["alpha"],
                                        Q_type="empirical", n_sim=2000,
                                        rng=np.random.default_rng(123))
    cov = compute_coverage(te, sim, PROB_LEVELS)
    return {"aad": _aad(cov), "levels": _levels(cov), "n_train": len(tr), "n_test": len(te),
            "n_rm_train": len(train_ret) - len(train_c), "n_rm_test": len(test_ret) - len(test_c),
            "bars_per_day": round(390.0 / minutes, 1),
            "alpha_post": float(np.mean(gibbs["alpha"])),
            "p_stay": float(np.exp(-np.mean(gibbs["alpha"])))}


if __name__ == "__main__":
    kind = sys.argv[1]
    target = int(sys.argv[2])
    out = {"config": {"kind": kind, "target": target}}
    if kind == "dollar":
        for cl in (True, False):
            out["clean" if cl else "no_clean"] = run_dollar(target, cl)
    else:
        for cl in (True, False):
            out["clean" if cl else "no_clean"] = run_calendar(target, cl)
    print("RESULT_JSON " + json.dumps(out))