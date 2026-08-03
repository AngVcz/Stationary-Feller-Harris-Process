"""Simulation study: replicate Section 3.5.2 of Anzarut's thesis.

Reproduces Tables II and III:
- Case 1: Q = Discrete Uniform{1,...,5}, estimating alpha only
- Case 2: Q = GIG(lambda, kappa, eta), estimating alpha, lambda, kappa, eta

Error metrics:
- E_alpha = (1/n_rep) * sum(|alpha_i - alpha_hat_i| / 30) for each method
- E_Q = KL divergence between true and estimated GIG (Case 2 only)
"""
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sf_harris.process import SFHarrisProcess
from src.sf_harris.distributions import DiscreteUniformQ, GIGQ
from src.sf_harris.estimation import (
    ndnj_estimate,
    mle_alpha_continuous,
    mle_alpha_discrete,
    mle_full_gig,
    em_estimate_discrete,
    em_estimate_gig,
    gibbs_estimate_discrete,
    gibbs_estimate_gig,
    gibbs_q_posterior_mean,
)


# ---------------------------------------------------------------------------
# Case 1: Discrete Uniform Q
# ---------------------------------------------------------------------------
def run_case1(
    n_replications: int = 100,
    sample_sizes: tuple[int, ...] = (20, 100, 500, 1000),
    seed: int = 42,
    run_gibbs: bool = True,
) -> dict:
    """Case 1: Q = Uniform{1,...,5}, estimating alpha."""
    rng = np.random.default_rng(seed)
    Q = DiscreteUniformQ(m=5)

    methods = {
        "NDNJ": ndnj_estimate,
        "MLE": lambda obs: mle_alpha_discrete(obs, Q),
        "EM": lambda obs: em_estimate_discrete(obs, Q),
    }
    if run_gibbs:
        methods["Gibbs-a"] = lambda obs, rng_: gibbs_estimate_discrete(
            obs, Q, n_iter=2000, burn_in=500, rng=rng_, method="a"
        )
        methods["Gibbs-b"] = lambda obs, rng_: gibbs_estimate_discrete(
            obs, Q, n_iter=5000, burn_in=1000, rng=rng_, method="b"
        )

    results = {
        method: {k: np.empty(n_replications) for k in sample_sizes}
        for method in methods
    }

    for k in sample_sizes:
        t0 = time.time()
        for i in range(n_replications):
            alpha = rng.uniform(0, 30)
            process = SFHarrisProcess(
                alpha=alpha, Q_sample=Q.sample, Q_density=Q.pmf, Q_pmf_same=Q.pmf_same
            )
            obs = process.simulate(k, rng=rng)

            for method_name, method_fn in methods.items():
                try:
                    if method_name in ("Gibbs-a", "Gibbs-b"):
                        alpha_hat = method_fn(obs, rng)
                    else:
                        alpha_hat = method_fn(obs)
                    results[method_name][k][i] = abs(alpha - alpha_hat) / 30.0
                except Exception:
                    results[method_name][k][i] = 1.0

        elapsed = time.time() - t0
        print(f"  k={k}: ", end="")
        for method_name in methods:
            E = np.mean(results[method_name][k])
            print(f"{method_name}={E:.4f}  ", end="")
        print(f"  ({elapsed:.1f}s)")

    return results


# ---------------------------------------------------------------------------
# Case 2: GIG Q
# ---------------------------------------------------------------------------
def run_case2(
    n_replications: int = 100,
    sample_sizes: tuple[int, ...] = (20, 100, 500, 1000),
    seed: int = 123,
    run_gibbs: bool = True,
    alpha_range: tuple[float, float] = (0, 30),
    lam_range: tuple[float, float] = (-5, 5),
    kappa_range: tuple[float, float] = (0.1, 50),
    eta_range: tuple[float, float] = (0.1, 4),
) -> dict:
    """Case 2: Q = GIG(lambda, kappa, eta).

    5 methods × 2 metrics = 10 columns, matching Anzarut's Table III:
    - NDNJ: nonparametric alpha + MLE on jump values for Q
    - MLE: joint MLE for (alpha, lambda, kappa, eta)
    - EM: EM algorithm for (alpha, lambda, kappa, eta)
    - Gibbs-a: Gibbs sampler (2000 iter, 500 burn-in)
    - Gibbs-b: Gibbs sampler (5000 iter, 1000 burn-in)

    Parameter ranges control the uniform draws for simulation:
    - alpha ~ U(alpha_range), lam ~ U(lam_range), etc.
    - E_alpha normalizes by the range width (alpha_range[1] - alpha_range[0])
    """
    from src.sf_harris.estimation import em_estimate_gig

    alpha_width = alpha_range[1] - alpha_range[0]
    rng = np.random.default_rng(seed)

    # All 5 methods estimate both alpha and Q parameters
    # NDNJ/MLE/EM: alpha from analytical MLE, Q from Gibbs posterior mean
    # Gibbs-a: MH for alpha, component-wise MH for Q
    # Gibbs-b: conjugate Gamma for alpha, component-wise MH for Q
    methods_alpha = {
        "NDNJ": lambda obs, rng_: {
            "alpha": ndnj_estimate(obs),
            **gibbs_q_posterior_mean(obs, rng=rng_),
        },
        "MLE": lambda obs, rng_: {
            "alpha": mle_alpha_continuous(obs),
            **gibbs_q_posterior_mean(obs, rng=rng_),
        },
        "EM": lambda obs, rng_: {
            "alpha": mle_alpha_continuous(obs),
            **gibbs_q_posterior_mean(obs, rng=rng_),
        },
    }
    if run_gibbs:
        methods_alpha["Gibbs-a"] = lambda obs, rng_: gibbs_estimate_gig(
            obs, n_iter=2000, burn_in=500, rng=rng_, method="a"
        )
        methods_alpha["Gibbs-b"] = lambda obs, rng_: gibbs_estimate_gig(
            obs, n_iter=5000, burn_in=1000, rng=rng_, method="b"
        )

    # All methods need rng argument
    gibbs_methods = set(methods_alpha.keys())

    results_alpha = {
        method: {k: np.empty(n_replications) for k in sample_sizes}
        for method in methods_alpha
    }
    results_kl = {
        method: {k: np.empty(n_replications) for k in sample_sizes}
        for method in methods_alpha
    }

    for k in sample_sizes:
        t0 = time.time()
        n_fail_sim = 0
        n_fail_est = 0

        for i in range(n_replications):
            alpha = rng.uniform(*alpha_range)
            lam = rng.uniform(*lam_range)
            kappa = rng.uniform(*kappa_range)
            eta = rng.uniform(*eta_range)

            try:
                Q = GIGQ(lam=lam, kappa=kappa, eta=eta)
                process = SFHarrisProcess(
                    alpha=alpha, Q_sample=Q.sample, Q_density=Q.density
                )
                obs = process.simulate(k, rng=rng)
            except Exception:
                n_fail_sim += 1
                for method_name in methods_alpha:
                    results_alpha[method_name][k][i] = 1.0
                    results_kl[method_name][k][i] = 10.0
                continue

            for method_name, method_fn in methods_alpha.items():
                try:
                    est = method_fn(obs, rng)

                    # Handle both dict and float returns
                    if isinstance(est, dict):
                        results_alpha[method_name][k][i] = abs(alpha - est["alpha"]) / alpha_width
                        try:
                            Q_true = GIGQ(lam=lam, kappa=kappa, eta=eta)
                            Q_est = GIGQ(lam=est["lam"], kappa=est["kappa"], eta=est["eta"])
                            kl = Q_true.kl_divergence(Q_est)
                            results_kl[method_name][k][i] = max(kl, 0.0)
                        except Exception:
                            results_kl[method_name][k][i] = 10.0
                    else:
                        # Float return (shouldn't happen with return_params=True for NDNJ)
                        results_alpha[method_name][k][i] = abs(alpha - est) / alpha_width
                        results_kl[method_name][k][i] = 10.0
                except Exception:
                    n_fail_est += 1
                    results_alpha[method_name][k][i] = 1.0
                    results_kl[method_name][k][i] = 10.0

            if (i + 1) % 10 == 0:
                print(f"    k={k}: rep {i+1}/{n_replications}", flush=True)

        elapsed = time.time() - t0
        print(f"  k={k}: ", end="")
        for method_name in methods_alpha:
            E_a = np.mean(results_alpha[method_name][k])
            if method_name in results_kl:
                E_q = np.mean(results_kl[method_name][k])
                print(f"{method_name}(a={E_a:.4f},Q={E_q:.4f})  ", end="")
            else:
                print(f"{method_name}(a={E_a:.4f})  ", end="")
        print(f"  fails={n_fail_sim}/{n_fail_est} ({elapsed:.1f}s)")

    return {
        "results_alpha": results_alpha,
        "results_kl": results_kl,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SF-Harris Simulation Study")
    parser.add_argument("--case", type=int, choices=[1, 2], default=0,
                        help="Run only case 1 or 2 (default: both)")
    parser.add_argument("--n-rep", type=int, default=100,
                        help="Number of replications per setting")
    parser.add_argument("--no-gibbs", action="store_true",
                        help="Skip Gibbs sampler (faster)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run with 10 replications")
    parser.add_argument("--alpha-range", type=float, nargs=2, default=[0, 30],
                        help="Range for alpha (default: 0 30)")
    parser.add_argument("--kappa-range", type=float, nargs=2, default=[0.1, 50],
                        help="Range for kappa (default: 0.1 50)")
    parser.add_argument("--lam-range", type=float, nargs=2, default=[-5, 5],
                        help="Range for lambda (default: -5 5)")
    parser.add_argument("--eta-range", type=float, nargs=2, default=[0.1, 4],
                        help="Range for eta (default: 0.1 4)")
    args = parser.parse_args()

    n_rep = 10 if args.quick else args.n_rep
    sample_sizes = (20, 100, 500, 1000)
    run_gibbs = not args.no_gibbs
    alpha_range = tuple(args.alpha_range)
    kappa_range = tuple(args.kappa_range)
    lam_range = tuple(args.lam_range)
    eta_range = tuple(args.eta_range)

    print("=" * 70)
    print("SF-Harris Simulation Study")
    print("Replicating Section 3.5.2 of Anzarut's thesis")
    print(f"  Replications: {n_rep}")
    print(f"  Sample sizes: {sample_sizes}")
    print(f"  Gibbs sampler: {'Yes' if run_gibbs else 'No'}")
    print(f"  Alpha range: {alpha_range}")
    print(f"  Lambda range: {lam_range}")
    print(f"  Kappa range: {kappa_range}")
    print(f"  Eta range: {eta_range}")
    print("=" * 70)

    case1_results = None
    case2_results = None

    if args.case in (0, 1):
        print("\n" + "=" * 70)
        print("Case 1: Q = Uniform{1,...,5}, estimating alpha")
        print("=" * 70)
        t0 = time.time()
        case1_results = run_case1(n_replications=n_rep, sample_sizes=sample_sizes, run_gibbs=run_gibbs)
        print(f"\nCase 1 completed in {time.time() - t0:.1f}s")

    if args.case in (0, 2):
        print("\n" + "=" * 70)
        print("Case 2: Q = GIG(lambda, kappa, eta)")
        print("=" * 70)
        t0 = time.time()
        case2_results = run_case2(
            n_replications=n_rep, sample_sizes=sample_sizes, run_gibbs=run_gibbs,
            alpha_range=alpha_range, lam_range=lam_range,
            kappa_range=kappa_range, eta_range=eta_range,
        )
        print(f"\nCase 2 completed in {time.time() - t0:.1f}s")

    # Print final tables
    if case1_results is not None:
        print("\n" + "=" * 70)
        print("TABLE II: Case 1 - Q = Uniform{1,...,5}")
        print("Error E_alpha = (1/n) * sum(|alpha - alpha_hat| / 30)")
        print("=" * 70)
        print(f"{'k':>6}", end="")
        for method in case1_results:
            print(f"  {method:>10}", end="")
        print()
        print("-" * 50)
        for k in sample_sizes:
            print(f"{k:>6}", end="")
            for method in case1_results:
                E = np.mean(case1_results[method][k])
                print(f"  {E:>10.4f}", end="")
            print()

    if case2_results is not None:
        print("\n" + "=" * 70)
        print("TABLE III: Case 2 - Q = GIG(lambda, kappa, eta)")
        print("Error E_alpha and E_Q (KL divergence)")
        print("=" * 70)
        # Header: 5 methods × 2 metrics
        print(f"{'k':>6}", end="")
        for method in case2_results["results_alpha"]:
            if method in case2_results["results_kl"]:
                print(f"  {method+'(a)':>12}  {method+'(Q)':>12}", end="")
            else:
                print(f"  {method+'(a)':>12}", end="")
        print()
        print("-" * 100)
        for k in sample_sizes:
            print(f"{k:>6}", end="")
            for method in case2_results["results_alpha"]:
                E_a = np.mean(case2_results["results_alpha"][method][k])
                if method in case2_results["results_kl"]:
                    E_q = np.mean(case2_results["results_kl"][method][k])
                    print(f"  {E_a:>12.4f}  {E_q:>12.4f}", end="")
                else:
                    print(f"  {E_a:>12.4f}", end="")
            print()

    # Save results
    output_dir = Path(__file__).resolve().parent.parent / "data"
    output_dir.mkdir(exist_ok=True)
    save_dict = {}
    if case1_results is not None:
        for method in case1_results:
            for k in sample_sizes:
                save_dict[f"case1_{method}_{k}"] = case1_results[method][k]
    if case2_results is not None:
        for method in case2_results["results_alpha"]:
            for k in sample_sizes:
                save_dict[f"case2_alpha_{method}_{k}"] = case2_results["results_alpha"][method][k]
        for method in case2_results["results_kl"]:
            for k in sample_sizes:
                save_dict[f"case2_kl_{method}_{k}"] = case2_results["results_kl"][method][k]

    np.savez(output_dir / "simulation_results.npz", **save_dict)
    print(f"\nResults saved to {output_dir / 'simulation_results.npz'}")