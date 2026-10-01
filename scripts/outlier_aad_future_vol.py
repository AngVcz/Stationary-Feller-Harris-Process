"""Outlier presence & daily AAD error vs future volatility (1,2,3,7-day ahead).

Headline calendar spec: 15-min bars, 0.08% top-|r| cleaning, eps=1e-5, seeds 42/123.
Fit on train (pre-CUTOFF), simulate over test. Per test day t:
  out_max(t)    = max |r| of RAW 15-min returns on day t (continuous outlier score)
  out_share90   = fraction of day-t raw bars above train 90th-pct |r|
  out_count     = fraction of day-t raw bars above the 0.08% cleaning threshold
  aad_day(t)    = coverage AAD on day t's CLEANED test bars only (model calibration error)
  RV(t)         = std of raw 15-min returns on day t (realized volatility)
Relationship with RV(t+h), h in {1,2,3,7}: Pearson + Spearman (raw) and partial
(OLS RV(t+h) ~ predictor + RV(t); reports coef, t, p). Controlling RV(t) isolates
whether the predictor adds info beyond today's vol (volatility clustering).

 Prints ANALYSIS. Self-check asserts at end.
"""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from anzarut_replication import (
    load_ibm_data, split_returns_by_date, estimate_periodicity, estimate_alpha,
    gibbs_gig_harris, simulate_predictive_sf_harris_vec,
)
from anzarut_intraday_15min import compute_15min_spot_volatility

PROB = [0.25, 0.50, 0.75, 0.85, 0.90, 0.95]
PCT_CLEAN = 0.08
MINUTES = 15
HS = [1, 2, 3, 7]

df = load_ibm_data()
close = df["close"].resample(f"{MINUTES}min").last().dropna()
returns = np.log(close).diff().dropna()
tr_ret, te_ret = split_returns_by_date(returns, train_frac=0.8)

# --- fit headline model on cleaned train, simulate over cleaned test ---
ar_tr = np.abs(tr_ret.values)
thr_clean = np.percentile(ar_tr, 100 - PCT_CLEAN)   # 0.08% cleaning threshold
q90 = np.percentile(ar_tr, 90)
ar_te = np.abs(te_ret.values)
trc = tr_ret[ar_tr < thr_clean]
tec = te_ret[ar_te < thr_clean]
period = estimate_periodicity(trc)
lt, _ = compute_15min_spot_volatility(trc, period)
et, _ = compute_15min_spot_volatility(tec, period)
tl, tev = lt.values, et.values
a = estimate_alpha(tl)
g = gibbs_gig_harris(tl, alpha_init=a["alpha_acf"], epsilon=1e-5,
                     n_iter=5000, burn_in=2000, rng=np.random.default_rng(42))
sim = simulate_predictive_sf_harris_vec(tl, tev, g, g["alpha"], Q_type="empirical",
                                        n_sim=2000, rng=np.random.default_rng(123))

# per-bar predictive bands (aligned to tev / et.index)
m = min(len(tev), sim.shape[1])
tev_a = tev[:m]
sim_a = sim[:, :m]
bands = {p: (np.quantile(sim_a, (1 - p) / 2, axis=0),
             np.quantile(sim_a, (1 + p) / 2, axis=0)) for p in PROB}
clean_dates = et.index[:m].normalize().values          # date of each cleaned test bar


def daily_aad_err(idx):
    """Coverage AAD on a subset of test-bar indices (one day's cleaned bars)."""
    if len(idx) < 2:
        return np.nan
    s = 0.0
    for p in PROB:
        lo, hi = bands[p]
        cov = np.mean((tev_a[idx] >= lo[idx]) & (tev_a[idx] <= hi[idx])) * 100
        s += abs(cov - p * 100)
    return s / len(PROB)


# --- per-day quantities ---
# cleaned-test coverage error (by date of cleaned bars)
pos_by_date = {}
for i, d in enumerate(clean_dates):
    pos_by_date.setdefault(d, []).append(i)
aad_day = {d: daily_aad_err(np.array(idx)) for d, idx in pos_by_date.items()}

# raw-test returns: outlier presence + RV, by date
raw_dates = te_ret.index.normalize().values
rvals = te_ret.values
raw_by_date = {}
for i, d in enumerate(raw_dates):
    raw_by_date.setdefault(d, []).append(i)
out_max = {}; out_share90 = {}; out_count = {}; rv = {}
for d, idx in raw_by_date.items():
    idx = np.array(idx)
    ar = np.abs(rvals[idx])
    out_max[d] = float(ar.max())
    out_share90[d] = float(np.mean(ar > q90))
    out_count[d] = float(np.mean(ar >= thr_clean))
    rv[d] = float(np.std(rvals[idx]))

# align on the intersection of days (sorted)
days = sorted(set(aad_day) & set(raw_by_date))
n = len(days)
A = np.array([aad_day[d] for d in days])
OM = np.array([out_max[d] for d in days])
OS = np.array([out_share90[d] for d in days])
OC = np.array([out_count[d] for d in days])
RV = np.array([rv[d] for d in days])


# --- stats helpers (no scipy) ---
def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return np.nan
    rx = np.argsort(np.argsort(x[m])); ry = np.argsort(np.argsort(y[m]))
    return float(np.corrcoef(rx, ry)[0, 1])


def norm_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def partial(y, x, z):
    """OLS y ~ 1 + x + z; return (coef on x, se, t, 2-sided p via normal approx)."""
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[m], y[m], z[m]
    X = np.column_stack([np.ones(len(x)), x, z])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    sigma2 = float(resid @ resid) / (n - k)
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = math.sqrt(max(cov[1, 1], 0))
    t = beta[1] / se if se > 0 else 0.0
    p = 2 * (1 - norm_cdf(abs(t)))
    return float(beta[1]), se, t, p


print("=" * 80)
print(f"Outlier presence & daily AAD error vs FUTURE realized volatility")
print(f"Headline spec: 15-min bars, 0.08% top-|r| clean, eps=1e-5, seeds 42/123, n_sim=2000")
print(f"Test days: {n}  |  thr_clean(0.08%)={thr_clean:.5f}  q90|r|={q90:.5f}")
print("=" * 80)

preds = [("out_max", OM), ("out_share90", OS), ("out_count", OC), ("aad_day", A)]
print(f"\n{'predictor(t)':>14} {'h':>3} {'r(RVth)':>8} {'rho(RVth)':>9} "
      f"{'beta|RV(t)':>10} {'t':>7} {'p':>7}")
print("-" * 64)
for name, x in preds:
    for h in HS:
        y = np.full(n, np.nan)
        y[:n - h] = RV[h:]            # RV(t+h)
        r = pearson(x, y)
        rho = spearman(x, y)
        b, se, t, p = partial(y, x, RV)
        star = "*" if p < 0.05 else " "
        print(f"{name:>14} {h:>3} {r:>8.3f} {rho:>9.3f} {b:>10.4f} {t:>7.2f} {p:>7.3f}{star}")

# baseline: RV(t) -> RV(t+h) for reference (volatility clustering benchmark)
print("-" * 64)
print("[benchmark] RV(t) -> RV(t+h):")
for h in HS:
    y = np.full(n, np.nan); y[:n - h] = RV[h:]
    r = pearson(RV, y); rho = spearman(RV, y)
    print(f"{'RV(t)':>14} {h:>3} {r:>8.3f} {rho:>9.3f}")

print("\nLegend: r=Pearson(raw), rho=Spearman(raw), beta|RV(t)=partial coef controlling today's vol,")
print("  *=p<0.05 (normal approx, ~n={} obs after the {}-day shift).".format(n - max(HS), max(HS)))
print("\nReading: a positive beta|RV(t) means the predictor forecasts HIGHER future vol")
print("beyond today's vol; a negative beta means it forecasts LOWER future vol.")

# ======================================================================
# Relative (scale-free) daily error: predictive z-score in IQR units.
# z_i = (tev_i - predictive_median_i) / predictive_IQR_i  -> invariant to the
# day's vol level (normalized by the model's OWN band width that day). A high
# mean|z| = realized values sit far from the predictive center *in band-width
# units* = genuine miscalibration, NOT just a wide day. Distinguishes wide/thin.
# ======================================================================
med = np.median(sim_a, axis=0)
iqr = np.quantile(sim_a, 0.75, axis=0) - np.quantile(sim_a, 0.25, axis=0)
iqr = np.where(iqr > 1e-12, iqr, np.nan)
z = (tev_a - med) / iqr
rel_err = {}
for d, idx in pos_by_date.items():
    idx = np.array(idx)
    rel_err[d] = float(np.mean(np.abs(z[idx]))) if len(idx) >= 2 else np.nan
RE = np.array([rel_err.get(d, np.nan) for d in days])
# model's own per-day band width (IQR of the predictive, averaged) -> a 'wide day' per the model
wday = {}
for d, idx in pos_by_date.items():
    idx = np.array(idx)
    wday[d] = float(np.nanmean(iqr[idx])) if len(idx) >= 2 else np.nan
W = np.array([wday.get(d, np.nan) for d in days])

print("\n" + "=" * 80)
print("WIDTH-CONFOUND CHECK: corr(predictor(t), RV(t))  -- is the error a width proxy?")
print(f"  aad_day     : r={pearson(A, RV):+.3f}  rho={spearman(A, RV):+.3f}")
print(f"  rel_err(z)  : r={pearson(RE, RV):+.3f}  rho={spearman(RE, RV):+.3f}  <- scale-free target")
print(f"  out_share90 : r={pearson(OS, RV):+.3f}")
print(f"  model_width : r={pearson(W, RV):+.3f}  (predictive IQR, the model's notion of width)")
print("  rel_err ~ 0 with RV(t) => it is NOT a width proxy (good); aad_day != 0 => partly one.")

print(f"\n{'predictor(t)':>14} {'h':>3} {'r(RVth)':>8} {'rho(RVth)':>9} {'beta|RV(t)':>10} {'t':>7} {'p':>7}")
print("-" * 64)
for name, x in [("rel_err(z)", RE), ("model_width", W)]:
    for h in HS:
        y = np.full(n, np.nan); y[:n - h] = RV[h:]
        r = pearson(x, y); rho = spearman(x, y)
        b, se, t, p = partial(y, x, RV)
        star = "*" if p < 0.05 else " "
        print(f"{name:>14} {h:>3} {r:>8.3f} {rho:>9.3f} {b:>10.4f} {t:>7.2f} {p:>7.3f}{star}")

# stratify by today's width (median split of RV(t)): relationship WITHIN thin/wide days
print("\n" + "=" * 80)
print("Stratified by today's width (RV(t) median split) -- rel_err(t) -> RV(t+h), within stratum")
cut = np.median(RV)
wide = RV >= cut
print(f"  cut (median RV) = {cut:.5f}; thin n={int((~wide).sum())}, wide n={int(wide.sum())}")
for label, mask in [("thin", ~wide), ("wide", wide)]:
    line = f"  [{label:>4}]"
    for h in HS:
        yfull = np.full(n, np.nan); yfull[:n - h] = RV[h:]
        x = RE[mask]; y = yfull[mask]
        r = pearson(x, y); rho = spearman(x, y)
        line += f"  h={h}: r={r:+.2f} rho={rho:+.2f}"
    print(line)
print("  (if the relationship survives WITHIN each stratum, it is not a width artifact)")

# ---- self-check ----
assert n > 50, f"too few test days ({n})"
assert np.isfinite(A).mean() > 0.95
assert np.isfinite(OM).all()
assert np.isfinite(RE).mean() > 0.95, "rel_err mostly finite"
assert all(pearson(RV[:n - h], RV[h:]) >= 0 for h in HS if (n - h) > 5), "RV autocorr sanity"
print("\nSELF-CHECK OK")