"""One arm of the leak-vs-constant test. Usage: python _dilution_arm.py <leaky|trainonly|causal>"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from anzarut_replication import (
    load_ibm_data, build_dollar_bars_rolling, estimate_alpha, gibbs_gig_harris,
    simulate_predictive_sf_harris, compute_coverage,
)

PROB_LEVELS = [0.25, 0.50, 0.75, 0.85, 0.90, 0.95]
CUTOFF = pd.Timestamp("2014-05-28")


def bars_from_threshold(df, threshold):
    cum, first_close, recs = 0.0, None, []
    for idx, row in df.iterrows():
        if first_close is None:
            first_close = row["close"]
        cum += row["dollar_volume"]
        if cum >= threshold:
            recs.append({"datetime": idx, "return": np.log(row["close"] / first_close)})
            cum, first_close = 0.0, None
    return pd.DataFrame(recs).set_index("datetime")["return"]


def run(dol, threshold):
    log = np.log((dol ** 2).clip(lower=1e-20)).replace([np.inf, -np.inf], np.nan).dropna()
    tr_full = log[log.index <= CUTOFF].values
    te_full = log[log.index > CUTOFF].values
    q01, q99 = np.percentile(tr_full, 0.5), np.percentile(tr_full, 99.5)
    tr = tr_full[(tr_full > q01) & (tr_full < q99)]
    te = te_full[(te_full > q01) & (te_full < q99)]
    alpha = estimate_alpha(tr)
    g = gibbs_gig_harris(tr, alpha_init=alpha["alpha_acf"], epsilon=0.1, n_iter=5000,
                         burn_in=2000, rng=np.random.default_rng(55))
    sim = simulate_predictive_sf_harris(tr, te, g, g["alpha"], Q_type="empirical",
                                        n_sim=2000, rng=np.random.default_rng(321))
    cov = compute_coverage(te, sim, PROB_LEVELS)
    aad = float(np.mean([abs(cov[p] - p * 100) for p in PROB_LEVELS]))
    levels = [{"p": p, "coverage": float(cov[p]), "deviation": float(cov[p] - p * 100)} for p in PROB_LEVELS]
    span_days = max((dol.index[-1] - dol.index[0]).days, 1)
    return {"threshold": float(threshold), "bpd": len(dol) / span_days, "aad": aad,
            "levels": levels,
            "abs_r_median": float(np.median(np.abs(dol.values))),
            "std_logr2": float(tr.std()), "n_train": len(tr), "n_test": len(te),
            "tr_mean": float(tr.mean()), "te_mean": float(te.mean())}


df = load_ibm_data()
arm = sys.argv[1]
if arm == "leaky":
    thr = float(df["dollar_volume"].resample("15min").sum().mean())
    out = run(bars_from_threshold(df, thr), thr)
elif arm == "trainonly":
    train_df = df[df.index <= CUTOFF]
    thr = float(train_df["dollar_volume"].resample("15min").sum().mean())
    out = run(bars_from_threshold(df, thr), thr)
elif arm == "causal":
    thr = float(df["dollar_volume"].groupby(df.index.normalize()).sum()
                .shift(1).rolling(30, min_periods=1).mean().mean() / 130)
    out = run(build_dollar_bars_rolling(df, lookback_days=30, bars_per_day=130), thr)
elif arm.startswith("const:"):
    # honest constant-threshold dollar bars: threshold fit on TRAIN only =
    # mean train daily dollar volume / target_bars_per_day. Leak-free + homoscedastic.
    target = int(arm.split(":")[1])
    train_daily = df[df.index <= CUTOFF]["dollar_volume"].groupby(df[df.index <= CUTOFF].index.normalize()).sum()
    thr = float(train_daily.mean() / target)
    out = run(bars_from_threshold(df, thr), thr)
else:
    out = {"error": "unknown arm " + arm}
out["arm"] = arm
print("RESULT_JSON " + json.dumps(out))