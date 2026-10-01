"""PIT (u_t) predictive power: market direction + crisis, horizons 1..60d, AND signals.

Signals (both OOS, train yardstick, evaluated on test):
  u_daily_t = Phi((log RV_t - mu)/sigma)          # closed form, mu/sig = train log-RV
  u15_max_t = max over day t of ecdf_train(log_spot_bar)   # intraday vol-spike PIT
  u15_mean_t, u15_outfrac_t (>0.95)               # daily aggregates of 15-min PIT

Yardsticks fit on TRAIN (CUTOFF 80/20), evaluated on TEST bars (raw, incl. jumps):
the 15-min ecdf is from the CLEANED train plateau (26/d, pct=0.08) so intraday
spike bars in the raw test land at u~1.0 (above the cleaned support) -> u15_max
captures intraday crises the cleaning would otherwise remove.

Outcomes (forward, horizon h in {1,5,10,30,60}):
  r_fwd[h]   = log return at t+h              (direction)
  cum[h]     = sum r_{t+1..t+h}               (cumulative direction)
  rvlog[h]   = log RV at t+h                  (vol clustering)
  maxRV[h]   = max RV_{t+1..t+h}             (crisis spike)
  crisis[h]  = maxRV[h] > q90(train RV)      (crisis flag)

Reported: Pearson + Spearman corr (signal vs outcome), conditional P(crisis) and
mean r_fwd by u-bucket, AND/OR combos of u_daily & u15_max. OOS (test) primary,
full-sample u_daily reference.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr, spearmanr, fisher_exact
from anzarut_replication import (
    load_ibm_data, split_returns_by_date, estimate_periodicity,
)
from anzarut_intraday_15min import compute_15min_spot_volatility

CUTOFF = pd.Timestamp("2014-05-28")
HORIZONS = [1, 2, 3, 5, 10, 30, 60]
PCT = 0.08          # plateau cleaning (26/d, 0.08%)
OUT = Path("docs/pit_predictive.json")


def ecdf_fn(train_sorted):
    n = len(train_sorted)
    def f(x):
        return np.searchsorted(train_sorted, x, side="right") / n
    return f


def corr_pack(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 5 or np.std(x) == 0 or np.std(y) == 0:
        return {"n": int(n), "r": None, "rho": None, "p": None}
    r, p = pearsonr(x, y)
    rho = spearmanr(x, y).correlation
    return {"n": int(n), "r": round(float(r), 4), "rho": round(float(rho), 4),
            "p": round(float(p), 4)}


def main():
    df = load_ibm_data()
    # ---- daily RV + close-to-close returns ----
    rv = pd.read_csv(Path("data/ibm_daily_rv.csv"))
    rv["date"] = pd.to_datetime(rv["date"])
    rv = rv.sort_values("date").set_index("date")
    daily_close = df["close"].resample("D").last().dropna()
    daily_ret = np.log(daily_close).diff().dropna()
    daily_ret.index = pd.to_datetime(daily_ret.index).normalize()
    panel = rv.copy()
    panel["ret"] = daily_ret.reindex(panel.index)
    panel = panel.dropna(subset=["ret", "log_rv"])
    # ---- daily u (train yardstick) ----
    tr_mask = panel.index <= CUTOFF
    mu, sig = panel.loc[tr_mask, "log_rv"].mean(), panel.loc[tr_mask, "log_rv"].std()
    panel["u_daily"] = norm.cdf((panel["log_rv"] - mu) / sig)
    q90 = float(np.percentile(panel.loc[tr_mask, "rv"], 90))
    q90_full = float(np.percentile(panel["rv"], 90))

    # ---- 15-min u (train cleaned ecdf, evaluated on RAW test bars) ----
    close15 = df["close"].resample("15min").last().dropna()
    ret15 = np.log(close15).diff().dropna()
    tr_ret, te_ret = split_returns_by_date(ret15, train_frac=0.8)
    ar_tr, ar_te = np.abs(tr_ret.values), np.abs(te_ret.values)
    thr = np.percentile(ar_tr, 100 - PCT)
    trc = tr_ret[ar_tr < thr]
    period = estimate_periodicity(trc)
    lt, _ = compute_15min_spot_volatility(trc, period)          # train CLEANED log-spot
    et_raw, _ = compute_15min_spot_volatility(te_ret, period)   # test RAW log-spot (all bars)
    train_ls = np.sort(lt.values)
    u_bar = ecdf_fn(train_ls)(et_raw.values)                    # per test bar PIT
    et_u = pd.Series(u_bar, index=pd.to_datetime(et_raw.index).normalize())
    day_grp = et_u.groupby(level=0)
    u15_max = day_grp.max()
    u15_mean = day_grp.mean()
    u15_outfrac = day_grp.apply(lambda s: float((s > 0.95).mean()))
    panel["u15_max"] = u15_max.reindex(panel.index)
    panel["u15_mean"] = u15_mean.reindex(panel.index)
    panel["u15_outfrac"] = u15_outfrac.reindex(panel.index)

    # ---- forward outcomes ----
    for h in HORIZONS:
        panel[f"r_fwd{h}"] = panel["ret"].shift(-h)
        panel[f"cum{h}"] = panel["ret"].rolling(h).sum().shift(-h)
        panel[f"rvlog{h}"] = panel["log_rv"].shift(-h)
        panel[f"maxRV{h}"] = panel["rv"].rolling(h).max().shift(-h)
        panel[f"crisis{h}"] = (panel[f"maxRV{h}"] > q90).astype(float)

    test = panel[~tr_mask].copy()
    signals = ["u_daily", "u15_max", "u15_mean", "u15_outfrac"]
    out_cols = {h: [f"r_fwd{h}", f"cum{h}", f"rvlog{h}", f"maxRV{h}", f"crisis{h}"]
                for h in HORIZONS}

    def corr_table(df_, sigs):
        tab = {}
        for s in sigs:
            row = {}
            for h in HORIZONS:
                row[str(h)] = {o: corr_pack(df_[s].values, df_[o].values)
                               for o in out_cols[h]}
            tab[s] = row
        return tab

    corr_oos = corr_table(test, signals)
    corr_full = corr_table(panel, ["u_daily"])

    # ---- conditional by bucket (OOS) ----
    def bucket_stats(df_, col, h):
        d = df_.copy()
        d["bkt"] = pd.cut(d[col], [-0.1, 0.2, 0.8, 1.1], labels=["low<0.2", "mid", "high>0.8"])
        res = {}
        for b in ["low<0.2", "mid", "high>0.8"]:
            sub = d[d["bkt"] == b]
            res[b] = {
                "n": int(len(sub)),
                "P_crisis": {str(hh): round(float(sub[f"crisis{hh}"].mean()), 3)
                             for hh in HORIZONS},
                "mean_r_fwd1": round(float(sub["r_fwd1"].mean()), 5),
                "P_rneg_fwd1": round(float((sub["r_fwd1"] < 0).mean()), 3),
            }
        res["base_P_crisis"] = {str(hh): round(float(d[f"crisis{hh}"].mean()), 3)
                                for hh in HORIZONS}
        res["base_mean_r_fwd1"] = round(float(d["r_fwd1"].mean()), 5)
        res["base_P_rneg"] = round(float((d["r_fwd1"] < 0).mean()), 3)
        return res

    cond_uD = bucket_stats(test, "u_daily", 1)
    cond_u15 = bucket_stats(test, "u15_max", 1)

    # ---- AND / OR combos of u_daily & u15_max (OOS) ----
    def combo(df_, a_hi, a_lo, b_hi, b_lo):
        d = df_.dropna(subset=["u_daily", "u15_max"])
        ma = d["u_daily"] > 0.8; ma_lo = d["u_daily"] < 0.2
        mb = d["u15_max"] > 0.8; mb_lo = d["u15_max"] < 0.2
        if a_hi and b_hi:   sel = ma & mb
        elif a_hi and b_lo: sel = ma & mb_lo
        elif a_lo and b_hi: sel = ma_lo & mb
        elif a_lo and b_lo: sel = ma_lo & mb_lo
        elif a_hi:          sel = ma
        elif b_hi:          sel = mb
        else:               sel = pd.Series(True, index=d.index)
        sub = d[sel]
        if len(sub) < 3:
            return {"n": int(len(sub))}
        return {
            "n": int(len(sub)),
            "P_crisis": {str(hh): round(float(sub[f"crisis{hh}"].mean()), 3)
                         for hh in HORIZONS},
            "mean_r_fwd1": round(float(sub["r_fwd1"].mean()), 5),
            "P_rneg_fwd1": round(float((sub["r_fwd1"] < 0).mean()), 3),
        }
    # ---- AND/OR combos: u_daily (selective >0.8) vs u15 intraday-spike flag ----
    # u15_max>0.8 is degenerate (max of ~26 PITs ~always >0.8); use u15_outfrac
    # at its own 85th pct so the intraday flag selects ~15% of days (matches uD_hi).
    d = test.dropna(subset=["u_daily", "u15_outfrac"]).copy()
    thr_out = float(np.percentile(d["u15_outfrac"], 85))
    d["uD"] = d["u_daily"] > 0.8
    d["u15"] = d["u15_outfrac"] > thr_out

    def combo_stats(sel):
        sub = d[sel]
        if len(sub) < 3:
            return {"n": int(len(sub))}
        return {
            "n": int(len(sub)),
            "P_crisis": {str(hh): round(float(sub[f"crisis{hh}"].mean()), 3)
                         for hh in HORIZONS},
            "mean_r_fwd1": round(float(sub["r_fwd1"].mean()), 5),
            "P_rneg_fwd1": round(float((sub["r_fwd1"] < 0).mean()), 3),
        }
    combos = {
        "thresh_u15_outfrac": round(thr_out, 4),
        "uD_hi": combo_stats(d["uD"]),
        "u15_spike": combo_stats(d["u15"]),
        "AND_uD_hi_u15_spike": combo_stats(d["uD"] & d["u15"]),
        "OR_uD_hi_u15_spike": combo_stats(d["uD"] | d["u15"]),
        "uD_hi_NOT_u15": combo_stats(d["uD"] & ~d["u15"]),     # daily spike, intraday calm
        "u15_NOT_uD_hi": combo_stats(~d["uD"] & d["u15"]),     # intraday spike, daily calm
        "NEITHER": combo_stats(~d["uD"] & ~d["u15"]),
    }

    # ---- Fisher exact 2x2: (flag high vs rest) x (crisis vs not), per horizon ----
    def fisher(flag, h):
        cr = test[f"crisis{h}"].fillna(0).astype(int)
        a = flag & (cr == 1); b = flag & (cr == 0)
        c = (~flag) & (cr == 1); e = (~flag) & (cr == 0)
        tbl = [[int(a.sum()), int(b.sum())], [int(c.sum()), int(e.sum())]]
        if min(tbl[0]) == 0 and min(tbl[1]) == 0:
            return {"or": None, "p": None, "tbl": tbl}
        OR, p = fisher_exact(tbl, alternative="greater")
        return {"or": round(float(OR), 3), "p": round(float(p), 4), "tbl": tbl}

    pvals = {}
    for h in HORIZONS:
        pvals[str(h)] = {
            "u_daily_high": fisher(test["u_daily"] > 0.8, h),
            "u15_outfrac_spike": fisher(d["u15"].reindex(test.index).fillna(False), h),
            "AND": fisher((d["uD"] & d["u15"]).reindex(test.index).fillna(False), h),
            "base_P_crisis": round(float(test[f"crisis{h}"].mean()), 3),
        }

    out = {
        "n_train": int(tr_mask.sum()), "n_test": int((~tr_mask).sum()),
        "cutoff": str(CUTOFF.date()),
        "mu": round(float(mu), 5), "sigma": round(float(sig), 5),
        "q90_train": round(q90, 6), "q90_full": round(q90_full, 6),
        "horizons": HORIZONS,
        "corr_oos": corr_oos, "corr_full_daily": corr_full,
        "cond_u_daily": cond_uD, "cond_u15_max": cond_u15,
        "combos_uD_u15": combos,
        "fisher_pvals": pvals,
        "series": {
            "date": [str(d.date()) for d in panel.index],
            "u_daily": [round(float(x), 4) if np.isfinite(x) else None
                        for x in panel["u_daily"]],
            "u15_max": [round(float(x), 4) if np.isfinite(x) else None
                        for x in panel["u15_max"]],
            "log_rv": [round(float(x), 4) for x in panel["log_rv"]],
            "ret": [round(float(x), 5) for x in panel["ret"]],
            "is_test": [bool(x) for x in (~tr_mask)],
        },
    }
    OUT.write_text(json.dumps(out), encoding="utf-8")
    # ---- console summary ----
    print(f"n_train={out['n_train']} n_test={out['n_test']} mu={mu:.4f} sig={sig:.4f} "
          f"q90_tr={q90:.5f} q90_full={q90_full:.5f}")
    print("\n== OOS corr (test) signal vs outcome ==")
    for s in signals:
        print(f"  [{s}]")
        for h in HORIZONS:
            c = corr_oos[s][str(h)]
            print(f"    h={h:>2}  r_fwd r={c['r_fwd'+str(h)]['r']}  "
                  f"cum r={c['cum'+str(h)]['r']}  rvlog r={c['rvlog'+str(h)]['r']}  "
                  f"maxRV r={c['maxRV'+str(h)]['r']}  crisis r={c['crisis'+str(h)]['r']}")
    print("\n== crisis prob by u_daily bucket (OOS) ==")
    for b in ["low<0.2", "mid", "high>0.8"]:
        b1 = cond_uD[b]
        print(f"  {b:>8} n={b1['n']:>3}  P_crisis h1={b1['P_crisis']['1']} "
              f"h30={b1['P_crisis']['30']} h60={b1['P_crisis']['60']}  "
              f"mean_r1={b1['mean_r_fwd1']}")
    print(f"  base   P_crisis h1={cond_uD['base_P_crisis']['1']} "
          f"h30={cond_uD['base_P_crisis']['30']} h60={cond_uD['base_P_crisis']['60']}")
    print(f"\n== AND/OR combos (u_daily>0.8 & u15_outfrac>q85={combos['thresh_u15_outfrac']}) ==")
    for k, v in combos.items():
        if k.startswith("thresh"):
            continue
        if v and v.get("n", 0) >= 1:
            pc = v.get("P_crisis", {})
            print(f"  {k:<22} n={v['n']:>3}  P_crisis h1={pc.get('1')} "
                  f"h30={pc.get('30')} h60={pc.get('60')}  mean_r1={v.get('mean_r_fwd1')}")
    print(f"\n== Fisher exact 2x2 (flag high vs rest) x (crisis vs not), alt='greater' ==")
    print(f"{'h':>3} {'base':>6} | {'u_daily_hi OR/p':>16} | {'u15_outfrac OR/p':>16} | {'AND OR/p':>14}")
    for h in HORIZONS:
        p = pvals[str(h)]
        def fmt(x): return f"{x['or']}/{x['p']}" if x["or"] is not None else "n/a"
        print(f"{h:>3} {p['base_P_crisis']:>6.3f} | {fmt(p['u_daily_high']):>16} | "
              f"{fmt(p['u15_outfrac_spike']):>16} | {fmt(p['AND']):>14}")
    print(f"\njson -> {OUT} ({OUT.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()