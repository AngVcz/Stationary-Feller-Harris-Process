"""Ablation (per-quantile): SF-Harris headlines — cleaning ON vs OFF, per-level coverage.

Extension of ablation_jumpcleaning.py. Identical seeds and pipeline settings:
  15-min : Gibbs seed=42, sim seed=123, n_iter=5000, n_sim=2000, eps=1e-5
  dollar : Gibbs seed=55, sim seed=321, n_iter=5000, n_sim=2000, eps=0.1

New: report per-level empirical coverage for 7 quantile levels
  [0.25, 0.50, 0.75, 0.85, 0.90, 0.95, 0.99]  (0.99 added)
with signed deviation = coverage - nominal(p*100).

Headline AAD over the 6 STANDARD levels [0.25..0.95] is reported as aad6
(unchanged logic, must match known headlines:
  15min CLEAN ~0.52, 15min NO-CLEAN ~0.46,
  dollar CLEAN ~0.30, dollar NO-CLEAN ~1.66).
aad7 = mean(|coverage[p]-p*100|) over all 7 levels (incl 0.99).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from anzarut_replication import (
    load_ibm_data, compute_15min_returns, detect_and_remove_jumps,
    fit_jump_thresholds, split_returns_by_date, estimate_periodicity,
    estimate_alpha, gibbs_gig_harris, simulate_predictive_sf_harris, compute_coverage,
)
from anzarut_intraday_15min import compute_15min_spot_volatility, PROB_LEVELS

# 6 standard headline levels (unchanged) + 0.99
PROB_LEVELS_7 = [0.25, 0.50, 0.75, 0.85, 0.90, 0.95, 0.99]
STANDARD_6 = [p for p in PROB_LEVELS_7 if p != 0.99]  # == PROB_LEVELS


def run(clean: bool):
    df = load_ibm_data()
    returns = compute_15min_returns(df)
    train_ret, test_ret = split_returns_by_date(returns, train_frac=0.8)

    if clean:
        thr = fit_jump_thresholds(train_ret, n_passes=2, top_pct=0.001)
        train_c = detect_and_remove_jumps(train_ret, fixed_thresholds=thr)
        test_c = detect_and_remove_jumps(test_ret, fixed_thresholds=thr)
    else:
        train_c = detect_and_remove_jumps(train_ret, fixed_thresholds=[])  # no-op
        test_c = detect_and_remove_jumps(test_ret, fixed_thresholds=[])     # no-op
    n_rm_tr = len(train_ret) - len(train_c)
    n_rm_te = len(test_ret) - len(test_c)

    period = estimate_periodicity(train_c)
    log_train, _ = compute_15min_spot_volatility(train_c, period)
    log_test, _ = compute_15min_spot_volatility(test_c, period)
    tl, te = log_train.values, log_test.values

    alpha = estimate_alpha(tl)
    gibbs = gibbs_gig_harris(tl, alpha_init=alpha["alpha_acf"], epsilon=1e-5,
                             n_iter=5000, burn_in=2000, rng=np.random.default_rng(42))
    sim = simulate_predictive_sf_harris(tl, te, gibbs, gibbs["alpha"],
                                        Q_type="empirical", n_sim=2000,
                                        rng=np.random.default_rng(123))
    cov = compute_coverage(te, sim, PROB_LEVELS_7)
    aad6 = np.mean([abs(cov[p] - p * 100) for p in STANDARD_6])
    aad7 = np.mean([abs(cov[p] - p * 100) for p in PROB_LEVELS_7])
    return {
        "cov": cov, "aad6": aad6, "aad7": aad7,
        "alpha_acf": alpha["alpha_acf"], "rho1": alpha["rho1"],
        "alpha_post": float(np.mean(gibbs["alpha"])), "p_stay": float(np.exp(-np.mean(gibbs["alpha"]))),
        "sigma_obs": float(np.mean(gibbs["sigma_obs"])) if "sigma_obs" in gibbs else float("nan"),
        "train_log_std": float(tl.std()), "test_log_std": float(te.std()),
        "n_train": len(tl), "n_test": len(te), "n_rm_train": n_rm_tr, "n_rm_test": n_rm_te,
    }


def run_dollar(clean: bool):
    """Dollar-bar headline (the 0.3pp BEST). Cleaning = outlier drop by percentile
    bounds 0.5/99.5 fit on train. OFF = keep all bars. Seeds: Gibbs 55, sim 321."""
    import pandas as pd
    df = load_ibm_data()
    returns = compute_15min_returns(df)
    train_ret, _ = split_returns_by_date(returns, train_frac=0.8)
    dol_cutoff = pd.Timestamp(train_ret.index.date.max())

    avg_dv = df["dollar_volume"].resample("15min").sum().mean()
    cum_dv, first_close, recs = 0.0, None, []
    for idx_, row in df.iterrows():
        if first_close is None:
            first_close = row["close"]
        cum_dv += row["dollar_volume"]
        if cum_dv >= avg_dv:
            recs.append({"datetime": idx_, "return": np.log(row["close"] / first_close)})
            cum_dv, first_close = 0.0, None
    dol = pd.DataFrame(recs).set_index("datetime")["return"]
    dol_log = np.log((dol ** 2).clip(lower=1e-20)).replace([np.inf, -np.inf], np.nan).dropna()

    tr_full = dol_log[dol_log.index <= dol_cutoff].values
    te_full = dol_log[dol_log.index > dol_cutoff].values
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
    cov = compute_coverage(te, sim, PROB_LEVELS_7)
    aad6 = np.mean([abs(cov[p] - p * 100) for p in STANDARD_6])
    aad7 = np.mean([abs(cov[p] - p * 100) for p in PROB_LEVELS_7])
    return {
        "cov": cov, "aad6": aad6, "aad7": aad7,
        "alpha_acf": alpha["alpha_acf"],
        "alpha_post": float(np.mean(gibbs["alpha"])),
        "p_stay": float(np.exp(-np.mean(gibbs["alpha"]))),
        "n_train": len(tr), "n_test": len(te),
        "n_rm_train": len(tr_full) - len(tr), "n_rm_test": len(te_full) - len(te),
    }


def levels_table(cov):
    """Build per-level rows: p, nominal, coverage, deviation."""
    rows = []
    for p in PROB_LEVELS_7:
        nominal = p * 100
        coverage = float(cov[p])
        deviation = coverage - nominal
        rows.append({"p": p, "nominal": nominal, "coverage": coverage, "deviation": deviation})
    return rows


if __name__ == "__main__":
    print("=" * 78)
    print("ABLATION (per-quantile): SF-Harris headlines — cleaning ON vs OFF")
    print(f"Levels: {PROB_LEVELS_7}  (0.99 added; aad6 over 6 standard, aad7 over all 7)")
    print("=" * 78)

    on = run(clean=True)
    off = run(clean=False)
    don = run_dollar(clean=True)
    doff = run_dollar(clean=False)

    arms = [
        ("15min CLEAN", on),
        ("15min NO-CLEAN", off),
        ("dollar CLEAN", don),
        ("dollar NO-CLEAN", doff),
    ]

    for name, res in arms:
        print(f"\n--- {name} ---")
        print(f"  aad6 = {res['aad6']:.4f}   aad7 = {res['aad7']:.4f}")
        print(f"  {'p':>6} {'nominal':>8} {'coverage':>10} {'dev(pp)':>10}")
        for r in levels_table(res["cov"]):
            print(f"  {r['p']:>6.2f} {r['nominal']:>8.2f} {r['coverage']:>10.2f} {r['deviation']:>+10.2f}")

    print("\nHeadline check (aad6):")
    for name, res in arms:
        print(f"  {name:>18}: aad6 = {res['aad6']:.4f}, aad7 = {res['aad7']:.4f}")

    # Emit a machine-readable JSON-ish block for downstream capture.
    import json
    out = []
    for name, res in arms:
        out.append({
            "name": name,
            "aad6": float(res["aad6"]),
            "aad7": float(res["aad7"]),
            "levels": levels_table(res["cov"]),
        })
    print("\n___JSON_BEGIN___")
    print(json.dumps(out, indent=2))
    print("___JSON_END___")