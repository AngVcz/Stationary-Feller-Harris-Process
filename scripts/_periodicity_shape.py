"""f(t_i) shape for the 5 plateau bar sizes (calendar, 0.08% clean).
Question: same f across the 5? U-shape or decreasing?"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from anzarut_replication import load_ibm_data, split_returns_by_date, estimate_periodicity

df = load_ibm_data()
PCT = 0.08
BPD_MIN = [(8, 49), (13, 30), (26, 15), (78, 5), (390, 1)]  # bpd -> approx minutes (390/bpd)

for bpd, minutes in BPD_MIN:
    close = df["close"].resample(f"{minutes}min").last().dropna()
    ret = np.log(close).diff().dropna()
    tr, te = split_returns_by_date(ret, train_frac=0.8)
    ar = np.abs(tr.values)
    thr = np.percentile(ar, 100 - PCT) if PCT > 0 else np.inf
    trc = tr[ar < thr] if PCT > 0 else tr
    f = estimate_periodicity(trc)
    print(f"\n## bpd={bpd} (~{minutes}min)  n_bins={len(f)}  clean={PCT}%")
    # print first, mid, last few bins to see shape
    idx = list(f.index)
    picks = idx if len(idx) <= 14 else idx[:3] + ["..."] + idx[len(idx)//2-1:len(idx)//2+1] + ["..."] + idx[-3:]
    for t in picks:
        if t == "...":
            print("     ...")
        else:
            print(f"   {t}  f={f[t]:.3f}")
    # open vs midday vs close summary
    if len(idx) >= 4:
        o, c = f.iloc[0], f.iloc[-1]
        m = f.iloc[len(f)//2]
        print(f"   -> open={o:.3f}  midday={m:.3f}  close={c:.3f}  (open/mid={o/m:.2f}, close/mid={c/m:.2f})")