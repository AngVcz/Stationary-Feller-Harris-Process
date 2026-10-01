"""Structural diagnostics for the constant-vs-adaptive threshold mechanism (no Gibbs).

Why does a leak-free CONSTANT threshold give AAD 0.515 while a leak-free ADAPTIVE
(causal rolling) gives 1.099, at the same ~67 bars/day? Measure the structural
differences that could drive it:
  - volume trend over 2012-2014 (constant threshold -> bar frequency co-varies with
    volume -> test log(r^2) shifts relative to train; adaptive -> frequency ~constant,
    return magnitude ~stationary).
  - bar-count vs daily-volume coupling (constant ~1, adaptive ~0).
  - train vs test mean/std of log(r^2) (regime shift each construction induces).
  - lag-1 autocorr of log(r^2) (stay/jump persistence proxy).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from anzarut_replication import load_ibm_data, build_dollar_bars_rolling

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


def per_day_bars(dol):
    s = pd.Series(1, index=dol.index).groupby(dol.index.normalize()).count()
    return s


def stats(name, dol):
    log = np.log((dol ** 2).clip(lower=1e-20)).replace([np.inf, -np.inf], np.nan).dropna()
    tr = log[log.index <= CUTOFF]
    te = log[log.index > CUTOFF]
    # winsorize like the pipeline (0.5/99.5 on train)
    q01, q99 = np.percentile(tr.values, 0.5), np.percentile(tr.values, 99.5)
    trw = tr[(tr > q01) & (tr < q99)]
    tew = te[(te > q01) & (te < q99)]
    # per-day bar count vs daily dollar volume
    nb = per_day_bars(dol)
    daily_dv = dol.index.to_series().groupby(dol.index.normalize()).apply(
        lambda ix: float("nan"))  # placeholder, real daily_dv computed from df below
    ac1 = trw.autocorr(1) if len(trw) > 2 else float("nan")
    print("\n=== " + name + " ===")
    print("  train mean/std log(r^2): " + f"{trw.mean():.3f} / {trw.std():.3f}  (n={len(trw)})")
    print("  test  mean/std log(r^2): " + f"{tew.mean():.3f} / {tew.std():.3f}  (n={len(tew)})")
    print("  |test-train mean shift|:  " + f"{abs(tew.mean()-trw.mean()):.3f}  (in train-std units: {abs(tew.mean()-trw.mean())/trw.std():.2f})")
    print("  lag-1 autocorr (train):    " + f"{ac1:.3f}")
    return {"tr_mean": float(trw.mean()), "tr_std": float(trw.std()),
            "te_mean": float(tew.mean()), "te_std": float(tew.std()),
            "shift": float(abs(tew.mean() - trw.mean())),
            "ac1": float(ac1), "bars_per_day": float(len(dol) / max((dol.index[-1]-dol.index[0]).days, 1))}


df = load_ibm_data()
daily_dv = df["dollar_volume"].groupby(df.index.normalize()).sum()
print("=" * 64)
print("VOLUME TREND 2012-2014 (daily dollar volume)")
print("=" * 64)
y2012 = daily_dv[daily_dv.index.year == 2012].mean()
y2013 = daily_dv[daily_dv.index.year == 2013].mean()
y2014 = daily_dv[daily_dv.index.year == 2014].mean()
print("  mean daily $ volume:  2012=" + f"{y2012:,.0f}" + "  2013=" + f"{y2013:,.0f}" + "  2014=" + f"{y2014:,.0f}")
print("  2014/2012 ratio:      " + f"{y2014/y2012:.2f}x")
# train (<=CUTOFF 2014-05-28) vs test daily volume
train_dv = daily_dv[daily_dv.index <= CUTOFF].mean()
test_dv = daily_dv[daily_dv.index > CUTOFF].mean()
print("  train mean / test mean $ vol: " + f"{train_dv:,.0f} / {test_dv:,.0f}  (test/train={test_dv/train_dv:.2f}x)")

thr_const = float(df[df.index <= CUTOFF]["dollar_volume"].resample("15min").sum().mean())
dol_const = bars_from_threshold(df, thr_const)
dol_caus = build_dollar_bars_rolling(df, lookback_days=30, bars_per_day=130)

# bar-count vs daily-volume coupling
nb_const = per_day_bars(dol_const)
nb_caus = per_day_bars(dol_caus)
nb_const = nb_const.reindex(daily_dv.index, fill_value=0)
nb_caus = nb_caus.reindex(daily_dv.index, fill_value=0)
corr_const = float(np.corrcoef(nb_const.values, daily_dv.values)[0, 1])
corr_caus = float(np.corrcoef(nb_caus.values, daily_dv.values)[0, 1])
print("\n  corr(daily bar count, daily $ volume):")
print("    constant threshold: " + f"{corr_const:.3f}")
print("    adaptive threshold:  " + f"{corr_caus:.3f}")

print("\n" + "=" * 64)
print("log(r^2) STRUCTURE: constant vs adaptive")
print("=" * 64)
s_const = stats("CONSTANT (trainonly, leak-free)", dol_const)
s_caus = stats("ADAPTIVE (causal, leak-free)", dol_caus)

print("\n" + "=" * 64)
print("READING")
print("=" * 64)
print("If volume trends and constant-threshold bar count tracks it (corr~1), the")
print("constant test log(r^2) should show a larger train->test mean shift than adaptive.")
print("Shift in train-std units > 0.5 means a regime shift the train-fit Q cannot bracket ->")
print("but here constant has LOWER AAD, so the story is not simply 'more shift = worse'.")
print("Compare shift_const vs shift_caus and ac1_const vs ac1_caus.")