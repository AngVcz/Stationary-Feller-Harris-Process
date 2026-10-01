"""Plateau p_stay sweep table.
Rows: 5 plateau models (cal bpd {8,13,26,78,390} x 0.08% clean) + average.
Cols: AAD at forced p_stay in {0, 0.1, 0.2, 0.3, 0.4, 0.5}.
Gibbs fit once per model (seed 42); p_stay forced by overriding alpha_posterior =
-log(p_stay) (mu/sigma and Q-emp from the fit held fixed -> isolates p_stay)."""
import sys, io, contextlib, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from anzarut_replication import (
    load_ibm_data, split_returns_by_date, estimate_periodicity, estimate_alpha,
    gibbs_gig_harris, simulate_predictive_sf_harris_vec, compute_coverage,
)
from anzarut_intraday_15min import compute_15min_spot_volatility
from grid_is_oos import calendar_bars_n_per_day

PROB = [0.25, 0.50, 0.75, 0.85, 0.90, 0.95]
PCT = 0.08
PSTAYS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
MODELS = [
    (8, "groupby", "~49min"),
    (13, "resample30", "30min"),
    (26, "resample15", "15min"),
    (78, "resample5", "5min"),
    (390, "resample1", "1min"),
]
df = load_ibm_data()


def aad(sim, tev):
    m = min(len(tev), sim.shape[1])
    cov = compute_coverage(tev[:m], sim[:, :m], PROB)
    return float(np.mean([abs(cov[p] - p * 100) for p in PROB]))


def build_log_spot(bpd, kind):
    if kind.startswith("resample"):
        minutes = int(kind.replace("resample", ""))
        close = df["close"].resample(f"{minutes}min").last().dropna()
    else:
        close = calendar_bars_n_per_day(df["close"], bpd)
    ret = np.log(close).diff().dropna()
    tr, te = split_returns_by_date(ret, train_frac=0.8)
    ar_tr, ar_te = np.abs(tr.values), np.abs(te.values)
    thr = np.percentile(ar_tr, 100 - PCT)
    trc, tec = tr[ar_tr < thr], te[ar_te < thr]
    period = estimate_periodicity(trc)
    lt, _ = compute_15min_spot_volatility(trc, period)
    et, _ = compute_15min_spot_volatility(tec, period)
    return lt.values, et.values


rows = []
for bpd, kind, label in MODELS:
    tl, tev = build_log_spot(bpd, kind)
    a = estimate_alpha(tl)
    with contextlib.redirect_stdout(io.StringIO()):
        g = gibbs_gig_harris(tl, alpha_init=a["alpha_acf"], epsilon=1e-5,
                             n_iter=5000, burn_in=2000, rng=np.random.default_rng(42))
    n_post = len(g["alpha"])
    fitted_pstay = float(np.mean(np.exp(-g["alpha"])))
    aads = {}
    for ps in PSTAYS:
        alpha_post = np.full(n_post, 100.0) if ps == 0.0 else np.full(n_post, -math.log(ps))
        sim = simulate_predictive_sf_harris_vec(tl, tev, g, alpha_post, Q_type="empirical",
                                                n_sim=2000, rng=np.random.default_rng(123))
        aads[ps] = aad(sim, tev)
    # fitted reference
    sim_f = simulate_predictive_sf_harris_vec(tl, tev, g, g["alpha"], Q_type="empirical",
                                              n_sim=2000, rng=np.random.default_rng(123))
    aads["fit"] = aad(sim_f, tev)
    rows.append({"bpd": bpd, "bar": label, "n_test": len(tev), "pstay_fit": fitted_pstay, **aads})

# print table
hdr = f"{'bpd':>4} {'bar':>7} {'n_test':>6} {'p_stay=0':>9} {'0.1':>6} {'0.2':>6} {'0.3':>6} {'0.4':>6} {'0.5':>6} {'fitted':>7}"
print(hdr)
print("-" * len(hdr))
col_sum = {ps: 0.0 for ps in PSTAYS}
col_sum_fit = 0.0
for r in rows:
    print(f"{r['bpd']:>4} {r['bar']:>7} {r['n_test']:>6} "
          f"{r[0.0]:>9.3f} {r[0.1]:>6.3f} {r[0.2]:>6.3f} {r[0.3]:>6.3f} {r[0.4]:>6.3f} {r[0.5]:>6.3f} {r['fit']:>7.3f}")
    for ps in PSTAYS:
        col_sum[ps] += r[ps]
    col_sum_fit += r["fit"]
n = len(rows)
avg = f"{'avg':>4} {'':>7} {'':>6} " + " ".join(f"{col_sum[ps]/n:>9.3f}" if ps == 0.0 else f"{col_sum[ps]/n:>6.3f}" for ps in PSTAYS) + f" {col_sum_fit/n:>7.3f}"
print("-" * len(hdr))
print(avg)
print(f"\n(fitted p_stay per model ~0.37; 'fitted' col = AAD at the Gibbs-estimated alpha)")