"""Compare simplified model (p_stay=0, iid-Q) vs fitted (p_stay~0.378) on the
plateau headline cell cal 26/d 0.08% (15-min). Same Gibbs posterior + same Q-emp;
only p_stay differs. Dumps paths, bands, coverage to JSON for the HTML."""
import sys, io, contextlib, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from anzarut_replication import (
    load_ibm_data, split_returns_by_date, estimate_periodicity, estimate_alpha,
    gibbs_gig_harris, simulate_predictive_sf_harris_vec, compute_coverage,
)
from anzarut_intraday_15min import compute_15min_spot_volatility

PROB = [0.25, 0.50, 0.75, 0.85, 0.90, 0.95]
PCT = 0.08
NSIM = 2000
df = load_ibm_data()
close = df["close"].resample("15min").last().dropna()
ret = np.log(close).diff().dropna()
tr, te = split_returns_by_date(ret, train_frac=0.8)
ar_tr, ar_te = np.abs(tr.values), np.abs(te.values)
thr = np.percentile(ar_tr, 100 - PCT)
trc, tec = tr[ar_tr < thr], te[ar_te < thr]
period = estimate_periodicity(trc)
lt, _ = compute_15min_spot_volatility(trc, period)
et, _ = compute_15min_spot_volatility(tec, period)
tl, tev = lt.values, et.values
a = estimate_alpha(tl)
with contextlib.redirect_stdout(io.StringIO()):
    g = gibbs_gig_harris(tl, alpha_init=a["alpha_acf"], epsilon=1e-5,
                         n_iter=5000, burn_in=2000, rng=np.random.default_rng(42))
n_post = len(g["alpha"])
fitted_pstay = float(np.mean(np.exp(-g["alpha"])))
mu_mean = float(np.mean(g["mu"]))

# Model A: p_stay=0 (iid-Q) ; Model B: fitted alpha
alpha_A = np.full(n_post, 100.0)
alpha_B = g["alpha"]
sim_A = simulate_predictive_sf_harris_vec(tl, tev, g, alpha_A, Q_type="empirical", n_sim=NSIM, rng=np.random.default_rng(123))
sim_B = simulate_predictive_sf_harris_vec(tl, tev, g, alpha_B, Q_type="empirical", n_sim=NSIM, rng=np.random.default_rng(123))


def aad(sim):
    m = min(len(tev), sim.shape[1])
    return float(np.mean([abs(compute_coverage(tev[:m], sim[:, :m], PROB)[p] - p * 100) for p in PROB]))


def covp(sim):
    m = min(len(tev), sim.shape[1])
    return {p: float(compute_coverage(tev[:m], sim[:, :m], PROB)[p]) for p in PROB}


AAD_A, AAD_B = aad(sim_A), aad(sim_B)
cov_A, cov_B = covp(sim_A), covp(sim_B)
m = min(len(tev), sim_A.shape[1])

# path viz: 100 sample paths x first 150 bars
NPATH = 100
WIN = min(150, m)
path_idx = np.random.default_rng(7).choice(NSIM, NPATH, replace=False)
paths_A = sim_A[path_idx, :WIN].tolist()
paths_B = sim_B[path_idx, :WIN].tolist()

# band viz: subsample ~130 indices across full test, bands at p=0.95 and 0.50
NB = 130
bidx = np.linspace(0, m - 1, NB).astype(int)


def bands(sim, p):
    lo = np.quantile(sim[:, :m][:, bidx], (1 - p) / 2, axis=0)
    hi = np.quantile(sim[:, :m][:, bidx], (1 + p) / 2, axis=0)
    return lo.tolist(), hi.tolist()


lo95_A, hi95_A = bands(sim_A, 0.95)
hi95_B_lo, hi95_B_hi = bands(sim_B, 0.95)
lo50_A, hi50_A = bands(sim_A, 0.50)
lo50_B, hi50_B = bands(sim_B, 0.50)
obs = tev[bidx].tolist()

out = {
    "n_test": int(m), "n_train": int(len(tl)), "nsim": NSIM,
    "fitted_pstay": fitted_pstay, "mu_mean": mu_mean,
    "alpha_mean": float(np.mean(g["alpha"])), "sigma_mean": float(np.mean(g["sigma"])),
    "AAD": {"pstay0": AAD_A, "fitted": AAD_B},
    "cov_pstay0": {str(k): v for k, v in cov_A.items()},
    "cov_fitted": {str(k): v for k, v in cov_B.items()},
    "PROB": PROB,
    "path_window": WIN,
    "paths_pstay0": paths_A, "paths_fitted": paths_B,
    "band_idx": bidx.tolist(),
    "obs": obs,
    "bands_pstay0": {"lo95": lo95_A, "hi95": hi95_A, "lo50": lo50_A, "hi50": hi50_A},
    "bands_fitted": {"lo95": hi95_B_lo, "hi95": hi95_B_hi, "lo50": lo50_B, "hi50": hi50_B},
}
Path("docs/pstay_compare.json").write_text(json.dumps(out), encoding="utf-8")
print(f"fitted_pstay={fitted_pstay:.3f}  AAD pstay0={AAD_A:.3f}  AAD fitted={AAD_B:.3f}")
print(f"cov pstay0: " + " ".join(f"{p}:{cov_A[p]:.1f}" for p in PROB))
print(f"cov fitted: " + " ".join(f"{p}:{cov_B[p]:.1f}" for p in PROB))
print(f"json written ({len(json.dumps(out))//1024} KB)")