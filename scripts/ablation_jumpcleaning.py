"""Ablation: does jump-cleaning earn its predictive power on the SF-Harris 15-min headline?

Replays Method A of anzarut_intraday_15min.py (log-scale Gibbs, eps=1e-5, the 0.5pp
headline) with identical seeds, toggling ONLY the jump-cleaning step:
  ON  -> fit_jump_thresholds(train) + detect_and_remove_jumps(train/test, fixed)
  OFF -> detect_and_remove_jumps(..., fixed_thresholds=[])  (no-op: returns unchanged)

Everything else (periodicity from train, spot vol, alpha init, Gibbs seed=42, sim
seed=123, n_iter=5000, n_sim=2000, prob levels) is bit-identical between arms.
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
    cov = compute_coverage(te, sim, PROB_LEVELS)
    aad = np.mean([abs(cov[p] - p * 100) for p in PROB_LEVELS])
    return {
        "aad": aad, "alpha_acf": alpha["alpha_acf"], "rho1": alpha["rho1"],
        "alpha_post": float(np.mean(gibbs["alpha"])), "p_stay": float(np.exp(-np.mean(gibbs["alpha"]))),
        "sigma_obs": float(np.mean(gibbs["sigma_obs"])) if "sigma_obs" in gibbs else float("nan"),
        "train_log_std": float(tl.std()), "test_log_std": float(te.std()),
        "n_train": len(tl), "n_test": len(te), "n_rm_train": n_rm_tr, "n_rm_test": n_rm_te,
    }


def run_dollar(clean: bool):
    """Dollar-bar headline (the 0.3pp BEST). Cleaning here = outlier drop by percentile
    bounds 0.5/99.5 fit on train (lines 421-424 of anzarut_intraday_15min.py). OFF = keep
    all bars. Same seeds as the pipeline: Gibbs 55, sim 321."""
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
    cov = compute_coverage(te, sim, PROB_LEVELS)
    aad = np.mean([abs(cov[p] - p * 100) for p in PROB_LEVELS])
    return {"aad": aad, "alpha_acf": alpha["alpha_acf"],
            "alpha_post": float(np.mean(gibbs["alpha"])),
            "p_stay": float(np.exp(-np.mean(gibbs["alpha"]))),
            "n_train": len(tr), "n_test": len(te),
            "n_rm_train": len(tr_full) - len(tr), "n_rm_test": len(te_full) - len(te)}


if __name__ == "__main__":
    print("=" * 72)
    print("ABLATION: SF-Harris headlines — cleaning ON vs OFF (identical seeds)")
    print("=" * 72)
    on = run(clean=True)
    off = run(clean=False)

    rows = [
        ("AAD (pp)", on["aad"], off["aad"]),
        ("alpha (ACF init)", on["alpha_acf"], off["alpha_acf"]),
        ("alpha (Gibbs post mean)", on["alpha_post"], off["alpha_post"]),
        ("P(stay) post", on["p_stay"], off["p_stay"]),
        ("sigma_obs post", on["sigma_obs"], off["sigma_obs"]),
        ("train log-std", on["train_log_std"], off["train_log_std"]),
        ("test log-std", on["test_log_std"], off["test_log_std"]),
        ("n removed (train/test)", f"{on['n_rm_train']}/{on['n_rm_test']}", f"{off['n_rm_train']}/{off['n_rm_test']}"),
        ("n obs (train/test)", f"{on['n_train']}/{on['n_test']}", f"{off['n_train']}/{off['n_test']}"),
    ]
    print(f"\n  {'metric':>24}  {'CLEAN':>14}  {'NO-CLEAN':>14}")
    print("  " + "-" * 58)
    for name, a, b in rows:
        print(f"  {name:>24}  {str(round(a,4) if isinstance(a,float) else a):>14}  {str(round(b,4) if isinstance(b,float) else b):>14}")

    d = off["aad"] - on["aad"]
    print(f"\n  Delta AAD (OFF - ON) = {d:+.2f}pp")
    if on["aad"] > 0:
        print(f"  Relativo: {(off['aad']/on['aad'] - 1)*100:+.1f}%  (NO-CLEAN {'peor' if d>0 else 'mejor' if d<0 else 'igual'})")

    print("\n" + "=" * 72)
    print("DOLLAR-BAR headline (the 0.3pp BEST) — outlier-removal ON vs OFF")
    print("=" * 72)
    don = run_dollar(clean=True)
    doff = run_dollar(clean=False)
    drows = [
        ("AAD (pp)", don["aad"], doff["aad"]),
        ("alpha (ACF init)", don["alpha_acf"], doff["alpha_acf"]),
        ("alpha (Gibbs post)", don["alpha_post"], doff["alpha_post"]),
        ("P(stay) post", don["p_stay"], doff["p_stay"]),
        ("n removed (train/test)", f"{don['n_rm_train']}/{don['n_rm_test']}", f"{doff['n_rm_train']}/{doff['n_rm_test']}"),
        ("n obs (train/test)", f"{don['n_train']}/{don['n_test']}", f"{doff['n_train']}/{doff['n_test']}"),
    ]
    print(f"\n  {'metric':>24}  {'CLEAN':>14}  {'NO-CLEAN':>14}")
    print("  " + "-" * 58)
    for name, a, b in drows:
        print(f"  {name:>24}  {str(round(a,4) if isinstance(a,float) else a):>14}  {str(round(b,4) if isinstance(b,float) else b):>14}")
    dd = doff["aad"] - don["aad"]
    print(f"\n  Delta AAD (OFF - ON) = {dd:+.2f}pp")
    if don["aad"] > 0:
        print(f"  Relativo: {(doff['aad']/don['aad'] - 1)*100:+.1f}%  (NO-CLEAN {'peor' if dd>0 else 'mejor' if dd<0 else 'igual'})")