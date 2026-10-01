"""Strategy full-sample (2012-2014): entry Close>MA200, exit on AND crisis signal,
re-buy 60 trading days after each exit (cooldown). EXPANDING-window yardstick (OOS
everywhere): u_daily uses mean/std of log-RV up to t-1; u15_outfrac uses the ecdf of
cleaned 15-min log-spot up to t-1 (refit every 21 days); q85 of outfrac is expanding.
f (periodicity) fit once on the burn-in year (stable seasonal). Burn-in 252d for MA200."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from scipy.stats import norm
from anzarut_replication import load_ibm_data, estimate_periodicity
from anzarut_intraday_15min import compute_15min_spot_volatility

BURN = 252          # trading days burn-in (MA200 + yardstick)
COOLDOWN = 15       # trading days after an exit before re-entry
PCT = 0.08          # 15-min cleaning (top 0.08% |returns| trimmed from yardstick)
REFIT = 21          # ecdf refit cadence (trading days)
# EXIT mode: "AND" = u_daily>0.8 AND u15_outfrac>q85 ; "u15" = u15_outfrac>q85 ; "udaily" = u_daily>0.8
EXIT = sys.argv[1] if len(sys.argv) > 1 else "AND"
OUT = Path(f"docs/strategy_{EXIT}.json")


def main():
    df = load_ibm_data()
    # ---- daily close + MA200 + RV, all aligned on rv trading-day index ----
    rv = pd.read_csv(Path("data/ibm_daily_rv.csv"))
    rv["date"] = pd.to_datetime(rv["date"])
    rv = rv.sort_values("date").set_index("date")
    dates = rv.index
    close = df["close"].resample("D").last().dropna()
    close.index = pd.to_datetime(close.index).normalize()
    close = close.reindex(dates)
    ma200 = close.rolling(200).mean()
    # RSI(14) Wilder on close
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rsi14 = 100 - 100 / (1 + avg_gain / avg_loss)
    log_rv = rv["log_rv"]
    # expanding μ/σ for u_daily (mean/std of log-RV up to t-1 -> shift 1)
    mu_t = log_rv.expanding().mean().shift(1)
    sig_t = log_rv.expanding().std().shift(1)
    u_daily = pd.Series(norm.cdf((log_rv - mu_t) / sig_t), index=dates)

    # ---- 15-min: f from burn-in year, all-bar raw log-spot, cleaning threshold ----
    close15 = df["close"].resample("15min").last().dropna()
    ret15 = np.log(close15).diff().dropna()
    ret15.index = pd.to_datetime(ret15.index).normalize()
    burn_dates = set(dates[:BURN])
    burn_ret = ret15[ret15.index.isin(burn_dates)]
    period = estimate_periodicity(burn_ret)
    _, rvdf = compute_15min_spot_volatility(ret15, period)
    bar_date = pd.to_datetime(rvdf.index).normalize()
    logspot = rvdf["log_spot"].values.astype(float)
    ar = np.abs(rvdf["return"].values)
    thr = np.percentile(ar, 100 - PCT)          # global cleaning (robust trim; minor lookahead)
    finite = np.isfinite(logspot)
    clean_mask = (ar < thr) & finite
    raw_mask = finite
    # group bars by date
    raw_by_date, clean_by_date = {}, {}
    for d, v, cm in zip(bar_date, logspot, clean_mask):
        if np.isfinite(v):
            raw_by_date.setdefault(d, []).append(float(v))
            if cm:
                clean_by_date.setdefault(d, []).append(float(v))

    n = len(dates)
    u15_outfrac = np.full(n, np.nan)
    cum_clean = []
    sorted_cum = np.array([])
    last_refit = -1
    for idx in range(BURN, n):
        d = dates[idx]
        # refit sorted ecdf every REFIT days (bars strictly before d)
        if idx - last_refit >= REFIT:
            cum_clean = np.concatenate([np.array(clean_by_date.get(dd, []))
                                        for dd in dates[:idx]])
            sorted_cum = np.sort(cum_clean)
            last_refit = idx
        bars = raw_by_date.get(d, [])
        if bars and sorted_cum.size:
            u = np.searchsorted(sorted_cum, np.array(bars), side="right") / sorted_cum.size
            u15_outfrac[idx] = float(np.mean(u > 0.95))

    # ---- expanding q85 of outfrac history -> exit signal (per EXIT mode) ----
    exit_sig = np.zeros(n, dtype=bool)
    hist = []
    for idx in range(BURN, n):
        v = u15_outfrac[idx]
        if len(hist) >= 30:
            q85 = np.percentile(hist, 85)
            if np.isfinite(v):
                ud = u_daily.iloc[idx] > 0.8
                below = pd.notna(close.iloc[idx]) and pd.notna(ma200.iloc[idx]) and \
                        close.iloc[idx] < ma200.iloc[idx]
                if EXIT == "AND":
                    fire = v > q85 and ud
                elif EXIT == "u15":
                    fire = v > q85
                elif EXIT == "udaily":
                    fire = ud
                elif EXIT == "udaily_sma200":
                    fire = ud and below
                elif EXIT == "and_sma200":
                    fire = v > q85 and ud and below
                elif EXIT == "u15_sma200":
                    fire = v > q85 and below
                elif EXIT == "and_rsi80":
                    fire = v > q85 and ud and pd.notna(rsi14.iloc[idx]) and rsi14.iloc[idx] > 80
                elif EXIT == "rsi80":
                    fire = pd.notna(rsi14.iloc[idx]) and rsi14.iloc[idx] > 80
                elif EXIT == "udaily_rsi80":
                    fire = ud and pd.notna(rsi14.iloc[idx]) and rsi14.iloc[idx] > 80
                else:
                    fire = False
                if fire:
                    exit_sig[idx] = True
        if np.isfinite(v):
            hist.append(v)

    # ---- position sim on [BURN, n) ----
    pos = np.zeros(n, dtype=int)
    state = 0
    cooldown_until = 0
    entries, exits = [], []
    for idx in range(BURN, n):
        c = close.iloc[idx]; m = ma200.iloc[idx]
        if state == 0:
            if idx >= cooldown_until and pd.notna(m) and pd.notna(c) and c > m:
                state = 1; entries.append(idx)
        else:
            if bool(exit_sig[idx]):
                state = 0; exits.append(idx)
                cooldown_until = idx + COOLDOWN
                pos[idx] = 1; continue
        pos[idx] = state

    # ---- returns over [BURN, n) ----
    s = slice(BURN, n)
    ret = close.iloc[BURN:].pct_change().fillna(0).values
    strat_ret = pos[BURN:] * ret
    bh = float(np.prod(1 + ret) - 1)
    st = float(np.prod(1 + strat_ret) - 1)
    n_in = int(pos[BURN:].sum())

    out = {
        "mode": EXIT,
        "burn_in": BURN, "cooldown": COOLDOWN, "n_used": int(n - BURN),
        "date0": str(dates[BURN].date()), "date1": str(dates[-1].date()),
        "n_in": n_in, "n_entries": len(entries), "n_exits": len(exits),
        "ret_buyhold": round(bh, 4), "ret_strategy": round(st, 4),
        "date": [str(d.date()) for d in dates[BURN:]],
        "close": [round(float(x), 2) if pd.notna(x) else None for x in close.values[BURN:]],
        "ma200": [round(float(x), 2) if pd.notna(x) else None for x in ma200.values[BURN:]],
        "pos": [int(x) for x in pos[BURN:]],
        # indices relative to the BURN slice for the chart
        "entries": [i - BURN for i in entries],
        "exits": [i - BURN for i in exits],
        "exit_dates": [str(dates[i].date()) for i in exits],
    }
    OUT.write_text(json.dumps(out), encoding="utf-8")
    print(f"burn={BURN} used={n-BURN}d ({dates[BURN].date()}->{dates[-1].date()}) "
          f"in={n_in} entries={len(entries)} exits={len(exits)}")
    print(f"ret  buy&hold={bh*100:.1f}%  strategy={st*100:.1f}%")
    print(f"exit dates: {out['exit_dates']}")
    print(f"json -> {OUT} ({OUT.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()