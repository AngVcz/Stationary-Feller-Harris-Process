"""h=1 sharpness test for P(stay): rolling one-step-ahead CRPS + pinball on log(RV).

Decision test: if CRPS/pinball are flat across P(stay) in {0, 0.38, 0.88}, the
Harris chain can be defaulted to P(stay)=0 (pure empirical bootstrap) with no
measurable loss in sharpness either -- closing the gap the AAD sweep left open
(AAD is calibration only; see Handout.md section 6, option (d)).

Design (strictly causal, one step ahead, evaluated at every test bar):
  predictive(t) = P * delta_{log_spot[t-1]} + (1-P) * Bootstrap(log_spot[:t])
  - anchor = previous noisy observation (what the real pipeline uses)
  - jumps = expanding-window empirical bootstrap. No MCMC needed: with emp-Q
    the Gibbs posterior over mu is analytic and tight (sigma/sqrt(n), n~13k),
    and sigma is unused in the empirical-Q predictive.
  - common random numbers across P values (same jump draws + same uniforms),
    so the P comparison is paired.

Data prep identical to gibbs_pstay_sweep.py (15-min calendar bars, full-series
jump cleaning + periodicity -- the known leak is common to every P, so the
P-comparison is valid; absolute CRPS levels are provisional).

P=1.0 (pure persistence point forecast) included as reference.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from anzarut_replication import (
    load_ibm_data, compute_15min_returns, detect_and_remove_jumps,
    estimate_periodicity, OUTPUT_DIR,
)

N_SIM = 2000
P_VALUES = [0.0, 0.38, 0.88, 1.0, "E[Q]", "combo"]
PIN_LEVELS = [0.05, 0.25, 0.50, 0.75, 0.95]
COV_LEVELS = [0.25, 0.50, 0.75, 0.85, 0.90, 0.95]
SEED = 123


def crps_sample(x, y):
    """CRPS of ensemble x vs observation y: mean|x-y| - 0.5*mean|x-x'| (sort trick)."""
    xs = np.sort(x)
    n = xs.size
    k = np.arange(1, n + 1)
    return np.mean(np.abs(xs - y)) - np.sum((2 * k - n - 1) * xs) / (n * n)


def _selfcheck():
    rng = np.random.default_rng(0)
    x = rng.normal(size=60)
    brute = np.mean(np.abs(x - 0.3)) - np.mean(np.abs(x[:, None] - x[None, :])) / 2
    assert abs(crps_sample(x, 0.3) - brute) < 1e-10, "CRPS sort-trick mismatch"


def main():
    _selfcheck()

    # --- data prep: identical to gibbs_pstay_sweep.py ---
    df = load_ibm_data()
    returns = compute_15min_returns(df)
    returns_clean = detect_and_remove_jumps(returns, n_passes=2, top_pct=0.001)
    periodicity = estimate_periodicity(returns)

    rv_df = returns_clean.to_frame("return")
    rv_df["rv_15min"] = rv_df["return"] ** 2
    rv_df["time"] = rv_df.index.strftime("%H:%M")
    rv_df["f_t"] = rv_df["time"].map(periodicity).fillna(1.0)
    rv_df["rv_adj"] = rv_df["rv_15min"] / rv_df["f_t"]
    rv_df["log_spot"] = np.log(rv_df["rv_adj"].clip(lower=1e-20))
    log_spot = rv_df.loc[np.isfinite(rv_df["log_spot"]), "log_spot"].values

    n = len(log_spot)
    split = int(n * 0.8)
    y = log_spot[split:]
    n_test = len(y)
    print(f"Bars: {n} total, {split} train, {n_test} test (rolling h=1 evaluation)")

    # --- common random numbers: anchor + bootstrap draws + uniforms per test bar ---
    rng = np.random.default_rng(SEED)
    last = np.empty(n_test)
    mu_hist = np.empty(n_test)
    jump_draws = np.empty((n_test, N_SIM))
    u = np.empty((n_test, N_SIM))
    for i in range(n_test):
        t = split + i
        last[i] = log_spot[t - 1]
        mu_hist[i] = log_spot[:t].mean()
        jump_draws[i] = rng.choice(log_spot[:t], size=N_SIM)
        u[i] = rng.random(N_SIM)

    # optimal point-forecast weight on the current deviation = lag-1 autocorr (train only)
    a = log_spot[:split]
    rho1 = float(np.corrcoef(a[:-1], a[1:])[0, 1])
    print(f"Train lag-1 autocorr of log(r^2): rho1 = {rho1:.3f} "
          f"(-> optimal point combo: {(1 - rho1):.2f}*E[Q] + {rho1:.2f}*(current deviation))")

    results = {}
    crps_by_p = {}
    print(f"\n{'entry':>8}  {'CRPS':>8}  {'AAD(h=1)':>9}  " +
          "  ".join(f"pin{int(q * 100):02d}" for q in PIN_LEVELS))
    for p in P_VALUES:
        if isinstance(p, str):
            if p == "E[Q]":
                point = mu_hist
            else:  # combo: conditional-mean point forecast, user's E[x_{t+1}|x_t] formula
                point = mu_hist + rho1 * (last - mu_hist)
            draws = np.broadcast_to(point[:, None], (n_test, N_SIM))
        else:
            draws = np.where(u < p, last[:, None], jump_draws)

        crps = np.array([crps_sample(draws[i], y[i]) for i in range(n_test)])
        crps_by_p[p] = crps

        pin = {}
        for q in PIN_LEVELS:
            q_hat = np.quantile(draws, q, axis=1)
            pin[q] = np.where(y >= q_hat, q * (y - q_hat), (1 - q) * (q_hat - y)).mean()

        cov_dev = []
        for c in COV_LEVELS:
            lo = np.quantile(draws, (1 - c) / 2, axis=1)
            hi = np.quantile(draws, (1 + c) / 2, axis=1)
            inside = ((y >= lo) & (y <= hi)).mean()
            cov_dev.append(abs(inside - c) * 100)

        results[p] = {"crps": crps.mean(), "pin": pin, "aad_h1": float(np.mean(cov_dev))}
        print(f"{str(p):>8}  {crps.mean():>8.4f}  {results[p]['aad_h1']:>7.2f}pp  " +
              "  ".join(f"{pin[q]:>.4f}" for q in PIN_LEVELS))

    # --- paired diffs vs P=0 (common random numbers -> same draws underneath) ---
    print("\nPaired CRPS diffs vs P=0 (per test bar):")
    for p in P_VALUES[1:]:
        if isinstance(p, str):
            continue
        d = crps_by_p[p] - crps_by_p[0.0]
        # ponytail: naive SE -- consecutive bars are autocorrelated, so treat
        # significance by effect size, not by this SE
        print(f"  P={p:.2f}: dCRPS={d.mean():+.4f} (naive SE {d.std(ddof=1) / np.sqrt(n_test):.4f})")

    numeric_p = [p for p in P_VALUES if not isinstance(p, str)]
    best = min(numeric_p, key=lambda p: results[p]["crps"])
    print(f"\nBest CRPS: P={best:.2f} ({results[best]['crps']:.4f})  |  P=0 reference: {results[0.0]['crps']:.4f}")
    if best == 0.0:
        print("Verdict: no P(stay)>0 beats the pure bootstrap at h=1 -> defaulting to 0 is safe.")
    else:
        print(f"Verdict: P={best:.2f} has lower CRPS -- check the paired diff before defaulting to 0.")

    out = OUTPUT_DIR / "pstay_sharpness_h1.csv"
    with open(out, "w", encoding="utf-8") as f:
        f.write("entry,crps,aad_h1," + ",".join(f"pin_{int(q*100)}" for q in PIN_LEVELS) + "\n")
        for p in P_VALUES:
            f.write(f"{p},{results[p]['crps']:.6f},{results[p]['aad_h1']:.4f}," +
                    ",".join(f"{results[p]['pin'][q]:.6f}" for q in PIN_LEVELS) + "\n")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()