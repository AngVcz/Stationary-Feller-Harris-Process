"""Cross-tab: exit forward-return sign (+10d) vs price above/below EMA50/EMA100 at exit.
Reads docs/strategy_{AND,u15}.json (exits are indices into the BURN slice) and recomputes
EMA on the full close aligned to rv trading-day index. BURN must match the strategy script."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from anzarut_replication import load_ibm_data

BURN = 252
HORIZON = 10   # +10d forward return defines the sign


def main():
    df = load_ibm_data()
    rv = pd.read_csv(Path("data/ibm_daily_rv.csv"))
    rv["date"] = pd.to_datetime(rv["date"])
    rv = rv.sort_values("date").set_index("date")
    close = df["close"].resample("D").last().dropna()
    close.index = pd.to_datetime(close.index).normalize()
    close = close.reindex(rv.index)
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema100 = close.ewm(span=100, adjust=False).mean()

    for mode in ["AND", "u15"]:
        d = json.load(open(f"docs/strategy_{mode}.json"))
        c = d["close"]
        rows = []
        for i in d["exits"]:
            j = BURN + i                # full-sample index
            px = c[i]                   # exit-day close (slice)
            fr = c[i + HORIZON] / px - 1 if i + HORIZON < len(c) and c[i + HORIZON] is not None else None
            if fr is None:
                continue
            above50 = bool(close.iloc[j] > ema50.iloc[j])
            above100 = bool(close.iloc[j] > ema100.iloc[j])
            rows.append((d["date"][i], px, fr, above50, above100))
        pos = [r for r in rows if r[2] > 0]
        neg = [r for r in rows if r[2] <= 0]
        print(f"\n=== {mode}  (n={len(rows)} exits, sign by +{HORIZON}d forward) ===")
        print(f"  POSITIVOS forward (n={len(pos)}): "
              f"arriba EMA50={sum(r[3] for r in pos)}, abajo EMA50={len(pos)-sum(r[3] for r in pos)}  |  "
              f"arriba EMA100={sum(r[4] for r in pos)}, abajo EMA100={len(pos)-sum(r[4] for r in pos)}")
        print(f"  NEGATIVOS forward (n={len(neg)}): "
              f"arriba EMA50={sum(r[3] for r in neg)}, abajo EMA50={len(neg)-sum(r[3] for r in neg)}  |  "
              f"arriba EMA100={sum(r[4] for r in neg)}, abajo EMA100={len(neg)-sum(r[4] for r in neg)}")
        det = "; ".join(f"{r[0]} {r[2]*100:+.1f}% E50{'^' if r[3] else 'v'} E100{'^' if r[4] else 'v'}" for r in rows)
        print(f"  detalle: {det}")


if __name__ == "__main__":
    main()