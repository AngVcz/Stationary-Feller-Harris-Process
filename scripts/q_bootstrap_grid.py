"""Calibration-window grid for the essential model (empirical-Q bootstrap).

Answers three questions for the reduced model (bootstrap of log(r^2)):
  1. Minimum calibration time (days) per frequency.
  2. Rolling (fixed W) vs expanding (all past) window.
  3. Does it replicate at 4h / 1d, or is it 15-min-only?

Design: per-bar log(r^2) at each frequency, RAW (no jump cleaning, no
periodicity -- same prep for every frequency, no leak). Rolling-origin h=1
evaluation on the last 20% of the sample, split by DATE on the finest
(15-min) grid (~2014-05-28, matching the Handout convention). Predictive =
bootstrap of the calibration window strictly before each test bar.
Metrics: CRPS + AAD(h=1) at the six Anzarut levels.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from anzarut_replication import load_ibm_data, OUTPUT_DIR

FREQS = ["15min", "1h", "4h", "1d"]
WINDOWS_D = [5, 10, 21, 63, 126, 252, 504]
N_SIM = 1000
LEVELS = [0.25, 0.50, 0.75, 0.85, 0.90, 0.95]
SEED = 7


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

    df = load_ibm_data()
    close15 = df["close"].resample("15min").last().dropna()
    split_ts = close15.index[int(len(close15) * 0.8)]
    print(f"15-min bars: {len(close15)}, split date: {split_ts.date()}")

    rows = []
    for freq in FREQS:
        cl = close15.resample(freq).last().dropna()
        ret = np.log(cl).diff().dropna()
        ls = np.log(np.maximum(ret**2, 1e-20))
        ls = ls[np.isfinite(ls)]

        dates = ls.index.date
        uniq, counts = np.unique(dates, return_counts=True)
        bpd = int(np.median(counts))  # empirical bars per trading day

        test_idx = np.where(ls.index >= split_ts)[0]
        n_test = len(test_idx)
        vals = ls.values
        print(f"\n=== {freq}: {len(vals)} bars (~{bpd}/day), {n_test} test bars ===")
        print(f"{'window':>10}  {'bars':>6}  {'CRPS':>8}  {'AAD(h=1)':>9}")

        configs = [("expanding", None)] + [(f"roll {d:>3}d", d * bpd) for d in WINDOWS_D]
        for name, W in configs:
            rng = np.random.default_rng(SEED)
            draws = np.empty((n_test, N_SIM))
            for j, t in enumerate(test_idx):
                lo = 0 if W is None else max(0, t - W)
                draws[j] = rng.choice(vals[lo:t], size=N_SIM)

            y = vals[test_idx]
            crps = np.array([crps_sample(draws[j], y[j]) for j in range(n_test)])

            aad = 0.0
            for p in LEVELS:
                qlo = np.quantile(draws, (1 - p) / 2, axis=1)
                qhi = np.quantile(draws, (1 + p) / 2, axis=1)
                cov = np.mean((y >= qlo) & (y <= qhi))
                aad += abs(cov - p)
            aad = aad / len(LEVELS) * 100

            wbars = "-" if W is None else f"{W:>6}"
            print(f"{name:>10}  {wbars:>6}  {crps.mean():>8.4f}  {aad:>7.2f}pp")
            rows.append((freq, name, crps.mean(), aad))

    # --- summary: best window per freq, rolling vs expanding, min days ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for freq in FREQS:
        fr = [r for r in rows if r[0] == freq]
        best = min(fr, key=lambda r: r[2])
        exp = next(r for r in fr if r[1] == "expanding")
        tol = 0.05
        min_days = next((r for r in fr if r[1] != "expanding" and r[2] <= best[2] + tol), None)
        print(f"{freq:>6}: best={best[1]} (CRPS {best[2]:.3f}, AAD {best[3]:.2f}pp) | "
              f"expanding CRPS {exp[2]:.3f} | min-days within {tol}: "
              f"{min_days[1] if min_days else 'none'}")

    out = OUTPUT_DIR / "q_bootstrap_grid.csv"
    with open(out, "w", encoding="utf-8") as f:
        f.write("freq,window,crps,aad_h1\n")
        for freq, name, crps, aad in rows:
            f.write(f"{freq},{name},{crps:.6f},{aad:.4f}\n")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()