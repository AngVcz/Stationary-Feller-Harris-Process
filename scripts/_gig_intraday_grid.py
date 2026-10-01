"""GIG intraday plateau: same calendar bpd x pct grid as the Q-emp plateau, but
Q = Generalized Inverse Gaussian (GIGQ, positive support) fit by MLE to the
POSITIVE spot variance H = r^2/f per cell. State = H (positive); alpha = ACF of
log H. Simulate SF-Harris(n_test, GIGQ, alpha, n_sim=2000), coverage AAD vs H_test.

This is the real 'Inversa Gaussiana Generalizada' (not the Gaussian Q_type='normal'
branch, which is a plain Normal and catastrophically mis-specified for log chi^2_1).

Logs incrementally to docs/gig_intraday_log.txt + docs/gig_intraday_grid.json.
390 (1-min) omitted: MLE + 43M GIG draws/cell too slow for the batch; Q-emp 390
row was tied/flat anyway.
"""
import sys, io, json, time, contextlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from scipy.stats import geninvgauss
from anzarut_replication import (
    load_ibm_data, split_returns_by_date, estimate_periodicity, estimate_alpha,
    compute_coverage,
)
from anzarut_intraday_15min import compute_15min_spot_volatility
from grid_is_oos import calendar_bars_n_per_day
from fit_sf_harris_ibm import fit_gig_mle

PROB = [0.25, 0.50, 0.75, 0.85, 0.90, 0.95]
PCTS = [0.0, 0.08, 0.5, 1.0, 2.5, 5.0]
CAL_MINUTES = [30, 15, 5, 3]      # -> 13, 26, 78, 130 bpd
CAL_N = [8]                       # coarse via groupby-date
NSIM = 2000
df = load_ibm_data()
LOG = Path("docs/gig_intraday_log.txt")
LOG.write_text("")


def log(msg):
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def aad(cov):
    return float(np.mean([abs(cov[p] - p * 100) for p in PROB]))


def simulate_gig_vec(H_train, H_test, alpha, lam, kappa, eta, scale_factor, n_sim, rng):
    """SF-Harris kernel with GIG jump. State positive (H units)."""
    n_test = len(H_test)
    p_stay = np.exp(-alpha)
    simulated = np.empty((n_sim, n_test))
    current = np.full(n_sim, float(H_train[-1]))
    for i in range(n_test):
        u = rng.random(n_sim)
        jump = u >= p_stay
        n_jump = int(jump.sum())
        if n_jump:
            z = geninvgauss.rvs(lam, kappa, size=n_jump, random_state=rng)
            current[jump] = (z / eta) / scale_factor  # unscale to H units
        simulated[:, i] = current
    return simulated


def cell_gig(returns, bpd, pct):
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
    tl, tev_log = lt.values, et.values
    H_train = np.exp(tl)          # positive spot variance
    H_test = np.exp(tev_log)
    H_train = H_train[np.isfinite(H_train)]
    H_test = H_test[np.isfinite(H_test)]
    # alpha from ACF of log H (consistent with fit_sf_harris / grid)
    alpha = float(estimate_alpha(tl)["alpha_acf"])
    p_stay = float(np.exp(-alpha))
    # GIG MLE on rescaled H (mean ~ 1) for numerical stability
    sf = 1.0 / max(np.mean(H_train), 1e-12)
    with contextlib.redirect_stdout(io.StringIO()):
        gp = fit_gig_mle(H_train * sf, method="de")
    lam, kappa, eta = gp["lam"], gp["kappa"], gp["eta"]
    sim = simulate_gig_vec(H_train, H_test, alpha, lam, kappa, eta, sf, NSIM,
                           rng=np.random.default_rng(123))
    cov = compute_coverage(H_test, sim, PROB)
    # also Q-emp on same H for direct comparison (monotone-equivariant: = Q-emp on log H)
    emp = H_train - np.mean(H_train)
    n_emp = emp.size
    rng2 = np.random.default_rng(123)
    ps2 = p_stay
    sim_q = np.empty((NSIM, len(H_test)))
    cur = np.full(NSIM, float(H_train[-1]))
    for i in range(len(H_test)):
        u = rng2.random(NSIM); jmp = u >= ps2
        if jmp.any():
            cur[jmp] = np.mean(H_train) + emp[rng2.integers(0, n_emp, int(jmp.sum()))]
        sim_q[:, i] = cur
    cov_q = compute_coverage(H_test, sim_q, PROB)
    return {"bpd": bpd, "pct": pct,
            "aad_te": aad(cov), "aad_qemp": aad(cov_q),
            "p_stay": p_stay, "alpha": alpha,
            "lam": float(lam), "kappa": float(kappa), "eta": float(eta),
            "n_train": int(len(H_train)), "n_test": int(len(H_test)),
            "cov": {str(p): float(cov[p]) for p in PROB}}


def run():
    out = []
    for minutes in CAL_MINUTES:
        close = df["close"].resample(f"{minutes}min").last().dropna()
        returns = np.log(close).diff().dropna()
        bpd = round(390.0 / minutes, 1)
        for pct in PCTS:
            t0 = time.perf_counter()
            try:
                rec = cell_gig(returns, bpd, pct)
            except Exception as e:
                rec = {"bpd": bpd, "pct": pct, "error": repr(e)}
            out.append(rec)
            log(f"  CAL {minutes:>3}min ({bpd:>5.1f}/d) pct={pct:>4}% "
                f"GIG_AAD={rec.get('aad_te')} Qemp_AAD={rec.get('aad_qemp')} "
                f"p_stay={rec.get('p_stay')} lam={rec.get('lam')} "
                f"[{time.perf_counter()-t0:.0f}s]")
            Path("docs/gig_intraday_grid.json").write_text(
                json.dumps({"pcts": PCTS, "calendar": out}, indent=1), encoding="utf-8")
    for n in CAL_N:
        bars = calendar_bars_n_per_day(df["close"], n)
        returns = np.log(bars).diff().dropna()
        bpd = float(n)
        for pct in PCTS:
            t0 = time.perf_counter()
            try:
                rec = cell_gig(returns, bpd, pct)
            except Exception as e:
                rec = {"bpd": bpd, "pct": pct, "error": repr(e)}
            out.append(rec)
            log(f"  CAL n={n} ({bpd:>4.0f}/d) pct={pct:>4}% "
                f"GIG_AAD={rec.get('aad_te')} Qemp_AAD={rec.get('aad_qemp')} "
                f"p_stay={rec.get('p_stay')} lam={rec.get('lam')} "
                f"[{time.perf_counter()-t0:.0f}s]")
            Path("docs/gig_intraday_grid.json").write_text(
                json.dumps({"pcts": PCTS, "calendar": out}, indent=1), encoding="utf-8")
    log("\nDONE intraday. cells=" + str(len(out)))
    return out


if __name__ == "__main__":
    run()