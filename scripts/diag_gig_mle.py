"""Diagnostic: How well does _mle_gig recover known GIG parameters?

Draws i.i.d. samples directly from GIGQ (no SF-Harris process) and
estimates parameters via _mle_gig.  This isolates GIG MLE difficulty
from the SF-Harris estimation pipeline.

For each parameter set and sample size n, reports:
  - Mean absolute error for lam, kappa, eta
  - Mean KL divergence between true and estimated GIG

Parameter sets cover the simulation study ranges (and beyond):
  - moderate:        lam=1.0,  kappa=5.0,  eta=1.5  (typical)
  - extreme_neg_lam: lam=-3.0, kappa=0.5,  eta=0.5  (extreme lambda)
  - large_kappa:    lam=2.0,  kappa=30.0, eta=2.0  (high concentration)
  - small_kappa:    lam=1.0,  kappa=0.5,  eta=1.0  (near-boundary)
  - large_eta:      lam=1.0,  kappa=5.0,  eta=3.5  (large scale)
"""
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sf_harris.distributions import GIGQ
from src.sf_harris.estimation import _mle_gig

# ---------------------------------------------------------------------------
# Parameter sets
# ---------------------------------------------------------------------------
PARAM_SETS = {
    "moderate":        {"lam": 1.0,  "kappa": 5.0,  "eta": 1.5},
    "extreme_neg_lam": {"lam": -3.0, "kappa": 0.5,  "eta": 0.5},
    "large_kappa":    {"lam": 2.0,  "kappa": 30.0, "eta": 2.0},
    "small_kappa":    {"lam": 1.0,  "kappa": 0.5,  "eta": 1.0},
    "large_eta":      {"lam": 1.0,  "kappa": 5.0,  "eta": 3.5},
}

SAMPLE_SIZES = [20, 50, 100, 500, 1000]
N_REPLICATIONS = 50


def run_diagnostic(seed: int = 42):
    """Run the full diagnostic and print a summary table."""
    rng_master = np.random.default_rng(seed)

    # Collect results: results[pname][n] = dict of arrays
    results = {}
    for pname, params in PARAM_SETS.items():
        results[pname] = {n: {"lam_err": [], "kappa_err": [], "eta_err": [], "kl": []}
                          for n in SAMPLE_SIZES}

    for pname, params in PARAM_SETS.items():
        Q_true = GIGQ(lam=params["lam"], kappa=params["kappa"], eta=params["eta"])
        print(f"\n--- Parameter set: {pname} "
              f"(lam={params['lam']}, kappa={params['kappa']}, eta={params['eta']}) ---",
              flush=True)
        print(f"    Using method='nm' for speed", flush=True)

        for n in SAMPLE_SIZES:
            t0 = time.time()
            n_fail = 0

            for rep in range(N_REPLICATIONS):
                if (rep + 1) % 10 == 0:
                    print(f"    n={n}, rep {rep+1}/{N_REPLICATIONS}", flush=True)
                rep_rng = np.random.default_rng(rng_master.integers(0, 2**62))

                # Draw n i.i.d. samples directly from GIGQ
                try:
                    samples = Q_true.sample_n(n, rep_rng)
                except Exception:
                    n_fail += 1
                    results[pname][n]["lam_err"].append(abs(params["lam"]))
                    results[pname][n]["kappa_err"].append(abs(params["kappa"]))
                    results[pname][n]["eta_err"].append(abs(params["eta"]))
                    results[pname][n]["kl"].append(10.0)
                    continue

                # Estimate via _mle_gig (use "nm" for speed in diagnostic)
                try:
                    est = _mle_gig(samples, method="nm")
                    lam_hat = est["lam"]
                    kappa_hat = est["kappa"]
                    eta_hat = est["eta"]
                except Exception:
                    n_fail += 1
                    results[pname][n]["lam_err"].append(abs(params["lam"]))
                    results[pname][n]["kappa_err"].append(abs(params["kappa"]))
                    results[pname][n]["eta_err"].append(abs(params["eta"]))
                    results[pname][n]["kl"].append(10.0)
                    continue

                # Compute errors
                lam_err = abs(params["lam"] - lam_hat)
                kappa_err = abs(params["kappa"] - kappa_hat)
                eta_err = abs(params["eta"] - eta_hat)

                # Compute KL divergence
                try:
                    Q_est = GIGQ(lam=lam_hat, kappa=kappa_hat, eta=eta_hat)
                    kl = Q_true.kl_divergence(Q_est)
                    kl = max(kl, 0.0)
                except Exception:
                    kl = 10.0

                results[pname][n]["lam_err"].append(lam_err)
                results[pname][n]["kappa_err"].append(kappa_err)
                results[pname][n]["eta_err"].append(eta_err)
                results[pname][n]["kl"].append(kl)

            elapsed = time.time() - t0

            # Compute means
            m_lam = np.mean(results[pname][n]["lam_err"])
            m_kap = np.mean(results[pname][n]["kappa_err"])
            m_eta = np.mean(results[pname][n]["eta_err"])
            m_kl  = np.mean(results[pname][n]["kl"])
            print(f"  n={n:>4d}: MAE(lam)={m_lam:.4f}  MAE(kappa)={m_kap:.4f}  "
                  f"MAE(eta)={m_eta:.4f}  KL={m_kl:.4f}  "
                  f"(fails={n_fail}, {elapsed:.1f}s)", flush=True)

    # Print summary table
    print("\n" + "=" * 120)
    print("SUMMARY TABLE: GIG MLE Recovery Diagnostic")
    print("=" * 120)
    header = (f"{'param_set':<20s} {'n':>5s}  {'MAE_lam':>10s}  {'MAE_kappa':>10s}  "
              f"{'MAE_eta':>10s}  {'KL_div':>10s}")
    print(header)
    print("-" * 120)

    for pname, params in PARAM_SETS.items():
        for n in SAMPLE_SIZES:
            m_lam = np.mean(results[pname][n]["lam_err"])
            m_kap = np.mean(results[pname][n]["kappa_err"])
            m_eta = np.mean(results[pname][n]["eta_err"])
            m_kl  = np.mean(results[pname][n]["kl"])
            print(f"{pname:<20s} {n:>5d}  {m_lam:>10.4f}  {m_kap:>10.4f}  "
                  f"{m_eta:>10.4f}  {m_kl:>10.4f}")
        print()

    # Also print KL at n=100 explicitly for comparison with simulation E_Q
    print("\n" + "=" * 80)
    print("KEY COMPARISON: KL divergence at each n, vs simulation E_Q at k=100")
    print("=" * 80)
    print(f"{'param_set':<20s}  ", end="")
    for n in SAMPLE_SIZES:
        print(f"{'n='+str(n):>10s}  ", end="")
    print()
    print("-" * 80)
    for pname in PARAM_SETS:
        print(f"{pname:<20s}  ", end="")
        for n in SAMPLE_SIZES:
            m_kl = np.mean(results[pname][n]["kl"])
            print(f"{m_kl:>10.4f}  ", end="")
        print()

    # Print relative parameter errors
    print("\n" + "=" * 80)
    print("RELATIVE ERRORS: MAE/|true_param| (median over replications)")
    print("=" * 80)
    for pname, params in PARAM_SETS.items():
        print(f"\n  {pname} (true: lam={params['lam']}, kappa={params['kappa']}, eta={params['eta']})")
        for n in SAMPLE_SIZES:
            rel_lam = np.median(results[pname][n]["lam_err"]) / max(abs(params["lam"]), 0.01)
            rel_kap = np.median(results[pname][n]["kappa_err"]) / max(abs(params["kappa"]), 0.01)
            rel_eta = np.median(results[pname][n]["eta_err"]) / max(abs(params["eta"]), 0.01)
            print(f"    n={n:>4d}: rel_lam={rel_lam:.4f}  rel_kappa={rel_kap:.4f}  rel_eta={rel_eta:.4f}")

    return results


if __name__ == "__main__":
    results = run_diagnostic()