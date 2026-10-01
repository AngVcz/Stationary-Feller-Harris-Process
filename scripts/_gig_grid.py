"""GIG grid: calendar arm with Q_type='normal' (parametric Gaussian jump, sigma
from the Normal-Inverse-Gamma / GIG posterior) instead of Q_type='empirical'.
Mirrors the Q-emp plateau search so we can find the GIG analogue of the flat tied
region (cal bpd x pct, OOS AAD).

ponytail: no boot_ci (1000x6 quantiles) -- plateau ID only needs aad_te + mc_se.
Logs incrementally to docs/gig_grid_log.txt + docs/gig_grid.json.
390 (1-min) omitted in this run: ~15-25 min/cell x pct -> hours; run separately
if the coarse plateau needs the 1-min anchor. The Q-emp 390 row was tied/flat.
"""
import sys, io, json, time, contextlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from anzarut_replication import (
    load_ibm_data, split_returns_by_date, estimate_periodicity, estimate_alpha,
    gibbs_gig_harris, simulate_predictive_sf_harris_vec, compute_coverage,
)
from anzarut_intraday_15min import compute_15min_spot_volatility
from grid_is_oos import calendar_bars_n_per_day

PROB = [0.25, 0.50, 0.75, 0.85, 0.90, 0.95]
PCTS = [0.0, 0.08, 0.5, 1.0, 2.5, 5.0]
CAL_MINUTES = [30, 15, 5, 3]      # -> 13, 26, 78, 130 bpd
CAL_N = [8]                       # coarse via groupby-date
L_TR_CAP = 20000
df = load_ibm_data()
LOG = Path("docs/gig_grid_log.txt")
LOG.write_text("")


def log(msg):
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def aad(cov):
    return float(np.mean([abs(cov[p] - p * 100) for p in PROB]))


def mc_se(tev, sim, nbatch=4):
    m = min(len(tev), sim.shape[1])
    sims = sim[:, :m]
    bs = np.array_split(np.arange(sims.shape[0]), nbatch)
    aads = [aad(compute_coverage(tev[:m], sims[b], PROB)) for b in bs]
    return float(np.std(aads, ddof=1) / np.sqrt(nbatch))


def cal_cell_gig(returns, bpd, pct):
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
    with contextlib.redirect_stdout(io.StringIO()):
        g = gibbs_gig_harris(tl, alpha_init=a["alpha_acf"], epsilon=1e-5,
                             n_iter=5000, burn_in=2000, rng=np.random.default_rng(42))
    sim_te = simulate_predictive_sf_harris_vec(tl, tev, g, g["alpha"], Q_type="normal",
                                               n_sim=2000, rng=np.random.default_rng(123))
    rng_tr = np.random.default_rng(2024)
    L = min(len(tl), L_TR_CAP)
    sub = rng_tr.choice(len(tl), L, replace=False)
    tl_sub = tl[sub]
    sim_tr = simulate_predictive_sf_harris_vec(tl, tl_sub, g, g["alpha"], Q_type="normal",
                                               n_sim=2000, rng=np.random.default_rng(123))
    cov_te = compute_coverage(tev, sim_te, PROB)
    cov_tr = compute_coverage(tl_sub, sim_tr, PROB)
    return {"bpd": bpd, "pct": pct, "aad_te": aad(cov_te), "aad_tr": aad(cov_tr),
            "mc_se": mc_se(tev, sim_te),
            "p_stay": float(np.exp(-np.mean(g["alpha"]))),
            "sigma_mean": float(np.mean(g["sigma"])),
            "mu_mean": float(np.mean(g["mu"])),
            "n_train": len(tl), "n_test": len(tev)}


def run():
    out = []
    for minutes in CAL_MINUTES:
        close = df["close"].resample(f"{minutes}min").last().dropna()
        returns = np.log(close).diff().dropna()
        bpd = round(390.0 / minutes, 1)
        for pct in PCTS:
            t0 = time.perf_counter()
            try:
                rec = cal_cell_gig(returns, bpd, pct)
            except Exception as e:
                rec = {"bpd": bpd, "pct": pct, "error": repr(e)}
            out.append(rec)
            log(f"  CAL {minutes:>3}min ({bpd:>5.1f}/d) pct={pct:>4}% "
                f"IS={rec.get('aad_tr')} OOS={rec.get('aad_te')} "
                f"mc_se={rec.get('mc_se')} p_stay={rec.get('p_stay')} "
                f"[{time.perf_counter()-t0:.0f}s]")
            Path("docs/gig_grid.json").write_text(json.dumps({"pcts": PCTS, "calendar": out}, indent=1), encoding="utf-8")
    for n in CAL_N:
        bars = calendar_bars_n_per_day(df["close"], n)
        returns = np.log(bars).diff().dropna()
        bpd = float(n)
        for pct in PCTS:
            t0 = time.perf_counter()
            try:
                rec = cal_cell_gig(returns, bpd, pct)
            except Exception as e:
                rec = {"bpd": bpd, "pct": pct, "error": repr(e)}
            out.append(rec)
            log(f"  CAL n={n} ({bpd:>4.0f}/d) pct={pct:>4}% "
                f"IS={rec.get('aad_tr')} OOS={rec.get('aad_te')} "
                f"mc_se={rec.get('mc_se')} p_stay={rec.get('p_stay')} "
                f"[{time.perf_counter()-t0:.0f}s]")
            Path("docs/gig_grid.json").write_text(json.dumps({"pcts": PCTS, "calendar": out}, indent=1), encoding="utf-8")
    log("\nDONE. cells=" + str(len(out)))
    return out


if __name__ == "__main__":
    run()