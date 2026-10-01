"""Verify what the OLD-dollar 0.30pp headline actually was.

Three constructions, identical pipeline downstream (log(r^2), eps=0.1, Gibbs 5000/2000,
seeds 55/321, winsorize 0.5/99.5 on train), same cutoff 2014-05-28:

  LEAKY    : threshold = full-series resample('15min').sum().mean()  [OLD ablation]
             -> fit on train+test -> TRAIN/TEST LEAK through bar boundaries.
  TRAINONLY: threshold = TRAIN-period resample('15min').sum().mean() [leak-free, constant]
             -> isolates the leak: same constant-threshold construction, no test info.
  CAUSAL   : build_dollar_bars_rolling(target chosen to match frequency)  [leak-free, adaptive]

Decision:
  - if TRAINONLY AAD ~= LEAKY (0.30)  -> the 0.30 is a CONSTANT-threshold property, NOT a leak.
  - if TRAINONLY AAD ~= CAUSAL (0.83) -> the 0.30 WAS the leak (test volume informing the bar
    boundaries), and the leak-fix is what raised AAD to the honest ~0.8 at this frequency.
Also reports bars/day, |r|, std(log r^2) to confirm none are degenerate.
"""
import sys
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


def aad_and_diag(name, dol, threshold):
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
    span_days = max((dol.index[-1] - dol.index[0]).days, 1)
    abs_r = np.abs(dol.values)
    print("\n=== " + name + " ===")
    print("  threshold ($/bar):  " + f"{threshold:,.0f}")
    print("  bars / day:         " + f"{len(dol)/span_days:.1f}")
    print("  |r| median:         " + f"{np.median(abs_r):.2e}")
    print("  std(log r^2):       " + f"{tr.std():.3f}")
    print("  AAD clean (pp):     " + f"{aad:.4f}")
    return {"name": name, "threshold": threshold, "bpd": len(dol)/span_days,
            "aad": aad, "abs_r_median": float(np.median(abs_r)), "std_logr2": float(tr.std())}


df = load_ibm_data()
train_df = df[df.index <= CUTOFF]

thr_leaky = float(df["dollar_volume"].resample("15min").sum().mean())
thr_train = float(train_df["dollar_volume"].resample("15min").sum().mean())
# causal target to land near the ~55-70/day band
causal = build_dollar_bars_rolling(df, lookback_days=30, bars_per_day=130)
thr_causal = float(df["dollar_volume"].groupby(df.index.normalize()).sum()
                   .shift(1).rolling(30, min_periods=1).mean().mean() / 130)

print("=" * 64)
print("WHAT WAS THE 0.30? leak vs constant-threshold (identical pipeline)")
print("=" * 64)
leaky = aad_and_diag("LEAKY (full-series threshold)", bars_from_threshold(df, thr_leaky), thr_leaky)
trainonly = aad_and_diag("TRAINONLY (train-only constant)", bars_from_threshold(df, thr_train), thr_train)
caus = aad_and_diag("CAUSAL (rolling, leak-free)", causal, thr_causal)

print("\n" + "=" * 64)
print("VERDICT")
print("=" * 64)
print(f"  {'arm':>32}  {'thr $':>12}  {'bpd':>6}  {'AAD':>7}")
for r in (leaky, trainonly, caus):
    print(f"  {r['name']:>32}  {r['threshold']:>12,.0f}  {r['bpd']:>6.1f}  {r['aad']:>7.3f}")
gap_leaky_train = leaky["aad"] - trainonly["aad"]
print("\n  LEAKY - TRAINONLY AAD = " + f"{gap_leaky_train:+.3f}pp")
if abs(gap_leaky_train) < 0.15:
    verdict = "TRAINONLY ~= LEAKY -> the 0.30 is a CONSTANT-threshold property, NOT a leak."
elif abs(trainonly["aad"] - caus["aad"]) < 0.15:
    verdict = "TRAINONLY ~= CAUSAL -> the 0.30 WAS the leak; leak-free (train-only or causal) gives the honest ~" + f"{(trainonly['aad']+caus['aad'])/2:.2f}."
else:
    verdict = "ambiguous: TRAINONLY sits between LEAKY and CAUSAL -> partly leak, partly constancy/frequency."
print("  " + verdict)