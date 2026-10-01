"""Grid IS vs OOS: extend %removal to 8/10 and compute train(IS)+test(OOS) AAD per cell.

Same native pipelines as grid_2d.py + grid_2d_ext.py (calendar eps=1e-5 + f_t, seeds
42/123; dollar eps=0.1, seeds 55/321). Per cell: Gibbs once, then simulate over TEST
(aad_te = OOS, same as grid_2d) AND over TRAIN (aad_tr = IS — forward-sim a path of
len L_TR_CAP from train[-1], coverage vs a random subsample of train realized; caps
memory on the 1-min/3-min cells whose train is 78k-234k). Bootstrap B over test
indices with fixed sim quantiles -> 95% CI on aad_te (observation-sampling noise);
batch-of-sims std -> MC SE (noise from n_sim=2000 draws). Combined SE = quad-sum.

Dumps docs/grid_is_oos.json and prints ANALYSIS: top-10 by train AAD with their test
rank (does the train order persist OOS?), Spearman rank corr (train vs test), and
whether the OOS-best cell is statistically separable from the runner-up.
"""
import sys, json, time, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from anzarut_replication import (
    load_ibm_data, split_returns_by_date, estimate_periodicity, estimate_alpha,
    gibbs_gig_harris, simulate_predictive_sf_harris_vec, compute_coverage,
    build_dollar_bars_rolling,
)
from anzarut_intraday_15min import compute_15min_spot_volatility

PROB = [0.25, 0.50, 0.75, 0.85, 0.90, 0.95]
CUTOFF = pd.Timestamp("2014-05-28")
PCTS = [0.0, 0.08, 0.5, 1.0, 2.5, 5.0, 8.0, 10.0]
CAL_MINUTES = [30, 15, 5, 3, 1]      # -> 13, 26, 78, 130, 390 bpd
CAL_N = [8, 5, 1]                    # coarse via groupby-date (positional split)
DOL_TARGETS = [13, 26, 40, 101, 200, 8, 5, 1]
L_TR_CAP = 20000                    # ponytail: cap IS simulate length (1-min train=234k -> OOM otherwise)
B_BOOT = 1000

df = load_ibm_data()


def aad(cov):
    return float(np.mean([abs(cov[p] - p * 100) for p in PROB]))


def aad_from_bands(act, bands):
    s = 0.0
    for p in PROB:
        lo, hi = bands[p]
        s += abs(np.mean((act >= lo) & (act <= hi)) * 100 - p * 100)
    return s / len(PROB)


def boot_ci(tev, sim, B=B_BOOT, rng=None):
    """95% CI on AAD from resampling test observations (sim quantiles fixed)."""
    if rng is None:
        rng = np.random.default_rng(7)
    n = len(tev)
    m = min(n, sim.shape[1])
    act = tev[:m]
    bands = {p: (np.quantile(sim[:, :m], (1 - p) / 2, axis=0),
                 np.quantile(sim[:, :m], (1 + p) / 2, axis=0)) for p in PROB}
    out = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, m, m)
        out[b] = aad_from_bands(act[idx], {p: (bands[p][0][idx], bands[p][1][idx]) for p in PROB})
    return np.percentile(out, [2.5, 50, 97.5])


def mc_se(tev, sim, nbatch=4):
    """MC SE on AAD: std of per-batch AADs / sqrt(nbatch) (noise from n_sim draws)."""
    m = min(len(tev), sim.shape[1])
    sims = sim[:, :m]
    bs = np.array_split(np.arange(sims.shape[0]), nbatch)
    aads = [aad(compute_coverage(tev[:m], sims[b], PROB)) for b in bs]
    return float(np.std(aads, ddof=1) / np.sqrt(nbatch))


def calendar_bars_n_per_day(close, n_bpd):
    """N equal-time bars/trading-day (groupby date, positional array_split)."""
    idx, vals = [], []
    for _, s in close.groupby(close.index.normalize()):
        s = s.dropna()
        if len(s) < n_bpd:
            continue
        for ch in np.array_split(s, n_bpd):
            if len(ch):
                idx.append(ch.index[-1])
                vals.append(ch.iloc[-1])
    return pd.Series(vals, index=pd.DatetimeIndex(idx)).sort_index()


def cal_cell(returns, bpd, pct):
    tr_ret, te_ret = split_returns_by_date(returns, train_frac=0.8)
    ar_tr, ar_te = np.abs(tr_ret.values), np.abs(te_ret.values)
    if pct == 0.0:
        trc, tec = tr_ret, te_ret
    else:
        thr = np.percentile(ar_tr, 100 - pct)
        trc, tec = tr_ret[ar_tr < thr], te_ret[ar_te < thr]
    period = estimate_periodicity(trc)
    lt, _ = compute_15min_spot_volatility(trc, period)
    et, _ = compute_15min_spot_volatility(tec, period)
    tl, tev = lt.values, et.values
    a = estimate_alpha(tl)
    g = gibbs_gig_harris(tl, alpha_init=a["alpha_acf"], epsilon=1e-5,
                         n_iter=5000, burn_in=2000, rng=np.random.default_rng(42))
    sim_te = simulate_predictive_sf_harris_vec(tl, tev, g, g["alpha"], Q_type="empirical",
                                               n_sim=2000, rng=np.random.default_rng(123))
    # IS: simulate a path of len L_TR_CAP, compare to a random train subsample
    rng_tr = np.random.default_rng(2024)
    L = min(len(tl), L_TR_CAP)
    sub = rng_tr.choice(len(tl), L, replace=False)
    tl_sub = tl[sub]
    sim_tr = simulate_predictive_sf_harris_vec(tl, tl_sub, g, g["alpha"], Q_type="empirical",
                                               n_sim=2000, rng=np.random.default_rng(123))
    cov_te = compute_coverage(tev, sim_te, PROB)
    cov_tr = compute_coverage(tl_sub, sim_tr, PROB)
    ci = boot_ci(tev, sim_te)
    return {"bpd": bpd, "pct": pct, "aad_te": aad(cov_te), "aad_tr": aad(cov_tr),
            "ci_lo": float(ci[0]), "ci_med": float(ci[1]), "ci_hi": float(ci[2]),
            "mc_se": mc_se(tev, sim_te),
            "p_stay": float(np.exp(-np.mean(g["alpha"]))),
            "n_train": len(tl), "n_test": len(tev)}


def dol_cell(tr_full, te_full, bpd, pct):
    if pct == 0.0:
        tr, te = tr_full, te_full
    else:
        qhi = np.percentile(tr_full, 100 - pct)
        tr, te = tr_full[tr_full < qhi], te_full[te_full < qhi]
    a = estimate_alpha(tr)
    g = gibbs_gig_harris(tr, alpha_init=a["alpha_acf"], epsilon=0.1,
                         n_iter=5000, burn_in=2000, rng=np.random.default_rng(55))
    sim_te = simulate_predictive_sf_harris_vec(tr, te, g, g["alpha"], Q_type="empirical",
                                               n_sim=2000, rng=np.random.default_rng(321))
    rng_tr = np.random.default_rng(2024)
    L = min(len(tr), L_TR_CAP)
    sub = rng_tr.choice(len(tr), L, replace=False)
    tr_sub = tr[sub]
    sim_tr = simulate_predictive_sf_harris_vec(tr, tr_sub, g, g["alpha"], Q_type="empirical",
                                               n_sim=2000, rng=np.random.default_rng(321))
    cov_te = compute_coverage(te, sim_te, PROB)
    cov_tr = compute_coverage(tr_sub, sim_tr, PROB)
    ci = boot_ci(te, sim_te)
    return {"bpd": round(bpd, 2), "pct": pct, "aad_te": aad(cov_te), "aad_tr": aad(cov_tr),
            "ci_lo": float(ci[0]), "ci_med": float(ci[1]), "ci_hi": float(ci[2]),
            "mc_se": mc_se(te, sim_te),
            "p_stay": float(np.exp(-np.mean(g["alpha"]))),
            "n_train": len(tr), "n_test": len(te)}


def calendar_grid():
    out = []
    # base: 13, 26, 78, 130, 390 bpd via resample
    for minutes in CAL_MINUTES:
        close = df["close"].resample(f"{minutes}min").last().dropna()
        returns = np.log(close).diff().dropna()
        bpd = round(390.0 / minutes, 1)
        for pct in PCTS:
            t0 = time.perf_counter()
            try:
                rec = cal_cell(returns, bpd, pct)
            except Exception as e:
                rec = {"bpd": bpd, "pct": pct, "error": repr(e)}
            out.append(rec)
            print(f"  CAL {minutes:>3}min ({bpd:>5.1f}/d) pct={pct:>4}% "
                  f"IS={rec.get('aad_tr')} OOS={rec.get('aad_te')} "
                  f"[{time.perf_counter()-t0:.0f}s]", flush=True)
    # coarse: 8, 5, 1 bpd via groupby-date
    for n in CAL_N:
        bars = calendar_bars_n_per_day(df["close"], n)
        returns = np.log(bars).diff().dropna()
        bpd = float(n)
        for pct in PCTS:
            t0 = time.perf_counter()
            try:
                rec = cal_cell(returns, bpd, pct)
            except Exception as e:
                rec = {"bpd": bpd, "pct": pct, "error": repr(e)}
            out.append(rec)
            print(f"  CAL n={n} ({bpd:>4.0f}/d) pct={pct:>4}% "
                  f"IS={rec.get('aad_tr')} OOS={rec.get('aad_te')} "
                  f"[{time.perf_counter()-t0:.0f}s]", flush=True)
    return out


def dollar_grid():
    out = []
    for target in DOL_TARGETS:
        t0 = time.perf_counter()
        dol = build_dollar_bars_rolling(df, lookback_days=30, bars_per_day=target)
        ndays = len(dol.index.to_series().dt.date.unique())
        bpd = len(dol) / max(ndays, 1)
        log = np.log((dol ** 2).clip(lower=1e-20)).replace([np.inf, -np.inf], np.nan).dropna()
        tr_full = log[log.index <= CUTOFF].values
        te_full = log[log.index > CUTOFF].values
        print(f"  DOL target={target:>4} -> {bpd:.2f}/d build={time.perf_counter()-t0:.0f}s", flush=True)
        for pct in PCTS:
            t1 = time.perf_counter()
            try:
                rec = dol_cell(tr_full, te_full, bpd, pct)
            except Exception as e:
                rec = {"bpd": round(bpd, 2), "pct": pct, "error": repr(e)}
            out.append(rec)
            print(f"      pct={pct:>4}% IS={rec.get('aad_tr')} OOS={rec.get('aad_te')} "
                  f"[{time.perf_counter()-t1:.0f}s]", flush=True)
    return out


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    return float(np.dot(rx, ry) / (math.sqrt((rx ** 2).sum()) * math.sqrt((ry ** 2).sum())))


def norm_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def analysis(cal, dol):
    cells = [(r, "cal") for r in cal if "error" not in r] + \
            [(r, "dol") for r in dol if "error" not in r]
    n = len(cells)
    by_tr = sorted(cells, key=lambda x: x[0]["aad_tr"])
    by_te = sorted(cells, key=lambda x: x[0]["aad_te"])
    te_rank = {id(r): i + 1 for i, (r, _) in enumerate(by_te)}
    tr_rank = {id(r): i + 1 for i, (r, _) in enumerate(by_tr)}

    print("\n" + "=" * 78)
    print("ANALYSIS — IS (train) vs OOS (test) AAD, 128 cells (8 bpd x 8 pct x 2 constr)")
    print("=" * 78)
    print(f"\nTop-10 by TRAIN AAD (IS champion) -> their TEST rank (does order persist?):")
    print(f"  {'#':>2} {'constr':>5} {'bpd':>7} {'pct':>6} {'AAD_tr':>7} {'AAD_te':>7} {'te_rank':>7} {'CI95%(te)':>14}")
    persist = 0
    for i, (r, c) in enumerate(by_tr[:10]):
        ci = f"[{r['ci_lo']:.2f},{r['ci_hi']:.2f}]"
        print(f"  {i+1:>2} {c:>5} {r['bpd']:>7} {r['pct']:>6} {r['aad_tr']:>7.3f} "
              f"{r['aad_te']:>7.3f} {te_rank[id(r)]:>7} {ci:>14}")
        if te_rank[id(r)] <= 10:
            persist += 1
    print(f"\n  {persist}/10 of the train-top-10 stay in the test-top-10.")

    rho = spearman([r["aad_tr"] for r, _ in cells], [r["aad_te"] for r, _ in cells])
    z = rho * math.sqrt(n - 1)
    pval = 2 * (1 - norm_cdf(abs(z)))
    print(f"\nSpearman rank corr (train vs test AAD, all {n} cells): rho={rho:.3f} "
          f"(z={z:.2f}, p~{pval:.2g})")

    print(f"\nTop-10 by TEST AAD (OOS) -> their TRAIN rank:")
    print(f"  {'#':>2} {'constr':>5} {'bpd':>7} {'pct':>6} {'AAD_te':>7} {'AAD_tr':>7} {'tr_rank':>7} {'CI95%(te)':>14} {'mc_se':>6}")
    for i, (r, c) in enumerate(by_te[:10]):
        ci = f"[{r['ci_lo']:.2f},{r['ci_hi']:.2f}]"
        print(f"  {i+1:>2} {c:>5} {r['bpd']:>7} {r['pct']:>6} {r['aad_te']:>7.3f} "
              f"{r['aad_tr']:>7.3f} {tr_rank[id(r)]:>7} {ci:>14} {r['mc_se']:>6.3f}")

    # is the OOS-best statistically the best? combined SE = sqrt(boot_se^2 + mc_se^2)
    def se(r):
        boot = (r["ci_hi"] - r["ci_lo"]) / (2 * 1.96)
        return math.sqrt(boot ** 2 + r["mc_se"] ** 2)
    best = by_te[0][0]
    second = by_te[1][0]
    diff = second["aad_te"] - best["aad_te"]
    se_diff = math.sqrt(se(best) ** 2 + se(second) ** 2)
    z_diff = diff / se_diff if se_diff > 0 else 0
    p_diff = 2 * (1 - norm_cdf(abs(z_diff)))
    # how many cells have CI overlapping the best (ci_lo < best.ci_hi)?
    overlap = sum(1 for r, _ in cells if r is not best and r["ci_lo"] < best["ci_hi"])
    print(f"\nIs the OOS-best statistically the best?")
    print(f"  best: {best['bpd']}/{best['pct']}% AAD_te={best['aad_te']:.3f} "
          f"CI=[{best['ci_lo']:.2f},{best['ci_hi']:.2f}] mc_se={best['mc_se']:.3f} combined_se={se(best):.3f}")
    print(f"  2nd:  {second['bpd']}/{second['pct']}% AAD_te={second['aad_te']:.3f} "
          f"CI=[{second['ci_lo']:.2f},{second['ci_hi']:.2f}] combined_se={se(second):.3f}")
    print(f"  gap(best,2nd)={diff:.3f}pp  SE(diff)={se_diff:.3f}  z={z_diff:.2f}  p~{p_diff:.2g}")
    print(f"  cells whose 95% CI overlaps the best's CI: {overlap}/{n-1}")
    print(f"  -> {'NOT ' if (p_diff > 0.05 or overlap > 0) else ''}statistically separable at 5%.")
    print("\n(note: CI is observation-sampling only; mc_se is n_sim noise; combined_se=quad-sum. "
          "CIs are per-cell independent, not paired — cells have different test sets after cleaning.)")


if __name__ == "__main__":
    print("=== CALENDAR GRID (IS+OOS) ===", flush=True)
    cal = calendar_grid()
    print("=== DOLLAR GRID (IS+OOS) ===", flush=True)
    dol = dollar_grid()
    out = {"pcts": PCTS, "calendar": cal, "dollar": dol}
    Path("docs").mkdir(exist_ok=True)
    with open("docs/grid_is_oos.json", "w") as f:
        json.dump(out, f, indent=1)
    analysis(cal, dol)
    print("\nWROTE docs/grid_is_oos.json")