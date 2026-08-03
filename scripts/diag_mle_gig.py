"""Diagnostic: Does _mle_gig recover known GIG parameters?

Simulate directly from GIG(lambda, kappa, eta) — no SF-Harris process —
and check if _mle_gig recovers the parameters and if the KL divergence
between true and estimated distributions is small.

This isolates the GIG MLE problem from the SF-Harris estimation pipeline.
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sf_harris.distributions import GIGQ
from src.sf_harris.estimation import _mle_gig, _get_distinct_values


def diagnose_mle_gig(
    true_params: tuple,
    sample_sizes: tuple = (50, 100, 500, 1000),
    n_reps: int = 50,
    seed: int = 42,
):
    """Diagnose _mle_gig with known GIG parameters.

    Args:
        true_params: (lam, kappa, eta) tuple.
        sample_sizes: Number of direct i.i.d. samples from GIG.
        n_reps: Number of replications per sample size.
        seed: Random seed.
    """
    lam_t, kappa_t, eta_t = true_params
    Q_true = GIGQ(lam=lam_t, kappa=kappa_t, eta=eta_t)

    print(f"\n{'='*70}")
    print(f"Diagnosing _mle_gig: GIG(lam={lam_t}, kappa={kappa_t}, eta={eta_t})")
    print(f"  chi = kappa*eta = {kappa_t*eta_t:.4f}, psi = kappa/eta = {kappa_t/eta_t:.4f}")
    print(f"  omega = kappa = {kappa_t:.4f}")
    print(f"  True mean = {Q_true._mean():.4f}, True var = {Q_true._var():.4f}" if hasattr(Q_true, '_mean') else "")
    print(f"{'='*70}")

    for n in sample_sizes:
        rng = np.random.default_rng(seed)
        kl_vals = []
        lam_errs = []
        kappa_errs = []
        eta_errs = []
        param_list = []

        for rep in range(n_reps):
            # Draw n i.i.d. samples directly from GIG (no SF-Harris)
            samples = Q_true.sample_n(n, rng)

            # Estimate GIG parameters
            est = _mle_gig(samples)
            lam_hat = est["lam"]
            kappa_hat = est["kappa"]
            eta_hat = est["eta"]

            # Compute KL divergence
            Q_est = GIGQ(lam=lam_hat, kappa=kappa_hat, eta=eta_hat)
            kl = Q_true.kl_divergence(Q_est)
            kl = max(kl, 0.0)

            kl_vals.append(kl)
            lam_errs.append(abs(lam_hat - lam_t))
            kappa_errs.append(abs(kappa_hat - kappa_t))
            eta_errs.append(abs(eta_hat - eta_t))
            param_list.append((lam_hat, kappa_hat, eta_hat))

        mean_kl = np.mean(kl_vals)
        mean_lam_err = np.mean(lam_errs)
        mean_kappa_err = np.mean(kappa_errs)
        mean_eta_err = np.mean(eta_errs)

        print(f"\n  n={n}: KL={mean_kl:.4f}  |lam_err|={mean_lam_err:.4f}  "
              f"|kappa_err|={mean_kappa_err:.4f}  |eta_err|={mean_eta_err:.4f}")

        # Show parameter estimates for first 5 reps
        print(f"  First 5 estimates (lam, kappa, eta):")
        for i in range(min(5, n_reps)):
            l, k, e = param_list[i]
            print(f"    rep {i+1}: lam={l:.3f}, kappa={k:.3f}, eta={e:.3f}  (true: {lam_t}, {kappa_t}, {eta_t})")

        # Show median and quartiles of KL
        print(f"  KL quartiles: Q25={np.percentile(kl_vals, 25):.4f}  "
              f"Q50={np.percentile(kl_vals, 50):.4f}  Q75={np.percentile(kl_vals, 75):.4f}  "
              f"max={np.max(kl_vals):.4f}")


def main():
    # Parameter regimes from simulation study ranges:
    # alpha ~ U(0,30), lam ~ U(-5,5), kappa ~ U(0.1,50), eta ~ U(0.1,4)

    print("GIG MLE Diagnostic: Independent recovery from known parameters")
    print("=" * 70)
    print("Simulating directly from GIG (no SF-Harris) and checking _mle_gig recovery.")
    print()

    # 1. Moderate parameters (should work well)
    diagnose_mle_gig(true_params=(1.0, 5.0, 1.5), sample_sizes=(50, 100, 500, 1000), n_reps=50)

    # 2. Large kappa (known to be hard — high concentration)
    diagnose_mle_gig(true_params=(2.0, 30.0, 2.0), sample_sizes=(50, 100, 500, 1000), n_reps=50)

    # 3. Small kappa (should work well)
    diagnose_mle_gig(true_params=(1.0, 0.5, 1.0), sample_sizes=(50, 100, 500, 1000), n_reps=50)

    # 4. Negative lambda (skewed distribution)
    diagnose_mle_gig(true_params=(-2.0, 2.0, 1.0), sample_sizes=(50, 100, 500, 1000), n_reps=50)

    # 5. Extreme: very large kappa
    diagnose_mle_gig(true_params=(1.0, 50.0, 3.0), sample_sizes=(50, 100, 500, 1000), n_reps=50)

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()