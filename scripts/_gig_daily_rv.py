"""GIG on daily realized variance (Anzarut Sec 4 formulation) -- OOS coverage.

State = daily RV (positive). Fit GIG MLE to TRAIN RV (rescaled), alpha = ACF of
log-RV train. Simulate SF-Harris(len(test), GIGQ, alpha, n_sim=2000), coverage
AAD vs TEST RV. Also Q-emp on the same daily RV for direct GIG-vs-Q comparison,
and a log-normal Q reference (the other Anzarut Sec 4 marginal).

This is a single series (no bpd x pct grid) -- the paper's native GIG object.
"""
import sys, io, json, contextlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from scipy.stats import geninvgauss
from anzarut_replication import compute_coverage
from fit_sf_harris_ibm import fit_gig_mle

PROB = [0.25, 0.50, 0.75, 0.85, 0.90, 0.95]
CUTOFF = pd.Timestamp("2014-05-28")
NSIM = 2000
OUT = Path("docs/gig_daily_rv.json")


def aad(cov):
    return float(np.mean([abs(cov[p] - p * 100) for p in PROB]))


def acf1(x):
    x = np.asarray(x, float); x = x - x.mean(); v = np.var(x)
    if v <= 0:
        return 0.0
    return float(np.mean(x[:-1] * x[1:]) / v)


def sim_gig(rv_train, rv_test, alpha, lam, kappa, eta, sf, rng):
    p_stay = np.exp(-alpha); n = len(rv_test)
    sim = np.empty((NSIM, n)); cur = np.full(NSIM, float(rv_train[-1]))
    for i in range(n):
        u = rng.random(NSIM); jmp = u >= p_stay
        if jmp.any():
            z = geninvgauss.rvs(lam, kappa, size=int(jmp.sum()), random_state=rng)
            cur[jmp] = (z / eta) / sf
        sim[:, i] = cur
    return sim


def sim_emp(rv_train, rv_test, alpha, rng):
    p_stay = np.exp(-alpha); n = len(rv_test)
    emp = rv_train - rv_train.mean(); ne = emp.size
    sim = np.empty((NSIM, n)); cur = np.full(NSIM, float(rv_train[-1]))
    for i in range(n):
        u = rng.random(NSIM); jmp = u >= p_stay
        if jmp.any():
            cur[jmp] = rv_train.mean() + emp[rng.integers(0, ne, int(jmp.sum()))]
        sim[:, i] = cur
    return sim


def sim_lognormal(rv_train, rv_test, alpha, mu, sigma, rng):
    p_stay = np.exp(-alpha); n = len(rv_test)
    sim = np.empty((NSIM, n)); cur = np.full(NSIM, float(rv_train[-1]))
    for i in range(n):
        u = rng.random(NSIM); jmp = u >= p_stay
        if jmp.any():
            cur[jmp] = rng.lognormal(mu, sigma, int(jmp.sum()))
        sim[:, i] = cur
    return sim


def main():
    df = pd.read_csv(Path("data/ibm_daily_rv.csv"))
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    train = df[df.index <= CUTOFF]
    test = df[df.index > CUTOFF]
    rv_tr = train["rv"].values; rv_te = test["rv"].values
    log_rv_tr = train["log_rv"].values
    n_tr, n_te = len(rv_tr), len(rv_te)
    alpha = -np.log(max(acf1(log_rv_tr), 1e-3))
    p_stay = float(np.exp(-alpha))
    # GIG MLE on rescaled train RV
    sf = 1.0 / max(np.mean(rv_tr), 1e-12)
    with contextlib.redirect_stdout(io.StringIO()):
        gp = fit_gig_mle(rv_tr * sf, method="de")
    lam, kappa, eta = gp["lam"], gp["kappa"], gp["eta"]
    # log-normal params on train RV
    mu_ln = float(np.mean(np.log(rv_tr))); sig_ln = float(np.std(np.log(rv_tr)))
    sim_g = sim_gig(rv_tr, rv_te, alpha, lam, kappa, eta, sf, np.random.default_rng(123))
    sim_e = sim_emp(rv_tr, rv_te, alpha, np.random.default_rng(123))
    sim_l = sim_lognormal(rv_tr, rv_te, alpha, mu_ln, sig_ln, np.random.default_rng(123))
    cov_g = compute_coverage(rv_te, sim_g, PROB)
    cov_e = compute_coverage(rv_te, sim_e, PROB)
    cov_l = compute_coverage(rv_te, sim_l, PROB)
    # bands for plotting (subsample ~120 days)
    bidx = np.linspace(0, n_te - 1, min(120, n_te)).astype(int)
    def bands(sim):
        return {"lo95": np.quantile(sim[:, bidx], 0.025, axis=0).tolist(),
                "hi95": np.quantile(sim[:, bidx], 0.975, axis=0).tolist(),
                "lo50": np.quantile(sim[:, bidx], 0.25, axis=0).tolist(),
                "hi50": np.quantile(sim[:, bidx], 0.75, axis=0).tolist()}
    # sample paths (60 x first 80 days)
    pidx = np.random.default_rng(7).choice(NSIM, 60, replace=False)
    out = {
        "n_train": int(n_tr), "n_test": int(n_te), "cutoff": str(CUTOFF.date()),
        "alpha": alpha, "p_stay": p_stay,
        "gig": {"lam": float(lam), "kappa": float(kappa), "eta": float(eta),
                "aad": aad(cov_g), "cov": {str(p): float(cov_g[p]) for p in PROB}},
        "qemp": {"aad": aad(cov_e), "cov": {str(p): float(cov_e[p]) for p in PROB}},
        "lognormal": {"mu": mu_ln, "sigma": sig_ln,
                      "aad": aad(cov_l), "cov": {str(p): float(cov_l[p]) for p in PROB}},
        "PROB": PROB,
        "rv_test": rv_te.tolist(),
        "band_idx": bidx.tolist(),
        "bands_gig": bands(sim_g),
        "bands_qemp": bands(sim_e),
        "paths_gig": sim_g[pidx, :min(80, n_te)].tolist(),
    }
    OUT.write_text(json.dumps(out), encoding="utf-8")
    print(f"n_train={n_tr} n_test={n_te} alpha={alpha:.3f} p_stay={p_stay:.3f}")
    print(f"GIG lam={lam:.3f} kappa={kappa:.3f} eta={eta:.3f}")
    print(f"AAD  GIG={aad(cov_g):.3f}  Qemp={aad(cov_e):.3f}  LogN={aad(cov_l):.3f}")
    print("cov GIG  :", " ".join(f"{p}:{cov_g[p]:.1f}" for p in PROB))
    print("cov Qemp :", " ".join(f"{p}:{cov_e[p]:.1f}" for p in PROB))
    print("cov LogN :", " ".join(f"{p}:{cov_l[p]:.1f}" for p in PROB))
    print(f"json -> {OUT} ({OUT.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()