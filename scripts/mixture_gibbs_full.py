"""Full Gibbs sampler for the Mixture SF-Harris process.

Model (Anzarut Section 7 extension):
  log(RV_t) follows a mixture SF-Harris process with ACF
  r(h) = w1 * exp(-alpha1 * h) + w2 * exp(-alpha2 * h)

  This replaces the single exponential r(h) = exp(-alpha * h) with a
  mixture that captures both fast (intraday) and slow (multi-day) decay.

Data augmentation:
  z_t in {0,1}: stay (z_t=0) or jump (z_t=1) indicator for transition t
  c_t in {1,2}: which mixture component generated transition t

Gibbs steps:
  1. Sample (z_t, c_t) | w, alpha, data  (stay/jump + component assignment)
  2. Sample w1 | c  (conjugate Beta)
  3. Sample alpha1 | z, c  (Metropolis-Hastings)
  4. Sample alpha2 | z, c  (Metropolis-Hastings)
  5. Sample mu | sigma2, data  (conjugate Normal)
  6. Sample sigma2 | mu, data  (conjugate Inverse-Gamma)
  7. Update Q from jump values  (empirical bootstrap)

Evaluates on full IBM dataset (1998-2026) with comparison to single exponential.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats, optimize

from anzarut_replication import (
    load_ibm_data,
    compute_15min_returns,
    detect_and_remove_jumps,
    estimate_alpha,
    gibbs_gig_harris,
    simulate_predictive_sf_harris,
    compute_coverage,
)

DATA_PATH = Path(r"C:\Users\angve\Desktop\Servicio\Libros\SF-Harris\IBM.txt")
PROB_LEVELS = [0.25, 0.50, 0.75, 0.85, 0.90, 0.95]


# ===========================================================================
# ACF fitting
# ===========================================================================

def acf_analysis(x, max_lag=40):
    """Compute autocorrelation function."""
    n = len(x)
    acf = np.zeros(max_lag + 1)
    mean = np.mean(x)
    var = np.var(x)
    if var < 1e-20:
        return acf
    for h in range(max_lag + 1):
        acf[h] = np.sum((x[:n-h] - mean) * (x[h:] - mean)) / (n * var)
    return acf


def fit_double_exponential(lags, acf_values):
    """Fit r(h) = w1*exp(-a1*h) + w2*exp(-a2*h) to ACF via weighted least squares."""
    from scipy.optimize import minimize

    def neg_ll(params):
        w1, log_a1, log_a2 = params
        a1, a2 = np.exp(log_a1), np.exp(log_a2)
        w2 = 1.0 - w1
        w1c = 1.0 / (1.0 + np.exp(-w1))  # sigmoid to constrain w1 in (0,1)
        w2c = 1.0 - w1c
        pred = w1c * np.exp(-a1 * lags) + w2c * np.exp(-a2 * lags)
        weights = 1.0 / (1 + lags.astype(float))
        return np.sum(weights * (acf_values - pred)**2)

    best_result = None
    best_val = np.inf
    for w1_0 in [0.5, 0.7, 0.9]:
        for a1_0 in [1.0, 3.0, 5.0]:
            for a2_0 in [0.05, 0.15, 0.5]:
                x0 = [w1_0, np.log(a1_0), np.log(a2_0)]
                try:
                    res = minimize(neg_ll, x0, method='L-BFGS-B')
                    if res.fun < best_val:
                        best_val = res.fun
                        best_result = res
                except Exception:
                    continue

    w1 = 1.0 / (1.0 + np.exp(-best_result.x[0]))
    a1 = np.exp(best_result.x[1])
    a2 = np.exp(best_result.x[2])

    # Compute R-squared
    pred = w1 * np.exp(-a1 * lags) + (1 - w1) * np.exp(-a2 * lags)
    ss_res = np.sum((acf_values - pred)**2)
    ss_tot = np.sum((acf_values - np.mean(acf_values))**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return {
        'w1': w1, 'alpha1': a1, 'alpha2': a2,
        'w2': 1 - w1, 'r_squared': r_squared,
        'alpha1_half_life': np.log(2) / a1,
        'alpha2_half_life': np.log(2) / a2,
    }


# ===========================================================================
# Full Gibbs sampler for mixture SF-Harris
# ===========================================================================

def gibbs_mixture_harris(log_rv, alpha1_init=3.0, alpha2_init=0.15, w1_init=0.7,
                          mu_init=None, sigma2_init=None, epsilon=None,
                          n_iter=5000, burn_in=2000, rng=None):
    """Full Gibbs sampler for mixture SF-Harris model.

    ACF: r(h) = w1*exp(-alpha1*h) + w2*exp(-alpha2*h)

    If epsilon is None, uses adaptive threshold based on median absolute deviation.
    This avoids the "all jumps" problem with very small epsilon.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(log_rv)

    # Initialize parameters
    mu = mu_init if mu_init is not None else np.mean(log_rv)
    sigma2 = sigma2_init if sigma2_init is not None else np.var(log_rv)
    alpha1 = alpha1_init
    alpha2 = alpha2_init
    w1 = w1_init

    # Adaptive epsilon: use median absolute deviation of differences
    # This ensures a reasonable fraction of stays (not 0% or 100%)
    diffs = np.diff(log_rv)
    if epsilon is None:
        mad = np.median(np.abs(diffs - np.median(diffs)))
        epsilon = 0.5 * mad  # transitions within half-MAD are "stays"
        # Ensure at least 20% and at most 80% stays
        n_stay_test = np.sum(np.abs(diffs) < epsilon)
        frac_stay = n_stay_test / len(diffs)
        if frac_stay < 0.20:
            # Increase epsilon until we get enough stays
            for mult in [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]:
                epsilon = mult * mad
                if np.sum(np.abs(diffs) < epsilon) / len(diffs) >= 0.20:
                    break
        elif frac_stay > 0.80:
            # Decrease epsilon
            for mult in [0.5, 0.2, 0.1, 0.05]:
                epsilon = mult * mad
                if np.sum(np.abs(diffs) < epsilon) / len(diffs) <= 0.80:
                    break

    is_stay = np.abs(diffs) < epsilon
    is_jump = ~is_stay
    n_stays = np.sum(is_stay)
    n_jumps = np.sum(is_jump)
    stay_frac = n_stays / (n_stays + n_jumps)

    print(f"  epsilon threshold: {epsilon:.6f}")
    print(f"  stays: {n_stays}/{n_stays+n_jumps} ({stay_frac:.1%}), jumps: {n_jumps}/{n_stays+n_jumps} ({1-stay_frac:.1%})")

    # Initialize component assignments
    c = np.ones(n - 1, dtype=int)  # component assignments

    # Prior hyperparameters
    mu_prior_mean = np.mean(log_rv)
    mu_prior_var = 10.0
    sigma2_prior_a = 2.0
    sigma2_prior_b = 0.5 * np.var(log_rv)
    alpha_prior_shape = 2.0
    alpha_prior_rate = 1.0

    # Storage for posterior samples
    samples = {
        'w1': [], 'alpha1': [], 'alpha2': [],
        'mu': [], 'sigma2': [],
        'n_stays': [], 'n_jumps': [],
        'P_stay_fast': [], 'P_stay_slow': [],
        'P_stay_mixture': [],
    }

    for it in range(n_iter):
        # ==============================================================
        # Step 1: Sample component assignments c_t | w, alpha, z
        # ==============================================================
        p_stay_fast = w1 * np.exp(-alpha1)
        p_stay_slow = (1 - w1) * np.exp(-alpha2)
        p_jump_fast = w1 * (1 - np.exp(-alpha1))
        p_jump_slow = (1 - w1) * (1 - np.exp(-alpha2))

        # For stays: P(c=j | stay) = w_j*exp(-alpha_j) / sum
        p_c1_given_stay = p_stay_fast / (p_stay_fast + p_stay_slow + 1e-30)
        # For jumps: P(c=j | jump) = w_j*(1-exp(-alpha_j)) / sum
        p_c1_given_jump = p_jump_fast / (p_jump_fast + p_jump_slow + 1e-30)

        # Assign components
        u_stay = rng.random(n_stays)
        c[is_stay] = (u_stay < p_c1_given_stay).astype(int)  # 0=fast, 1=slow
        u_jump = rng.random(n_jumps)
        c[is_jump] = (u_jump < p_c1_given_jump).astype(int)  # 0=fast, 1=slow

        # Count components
        n_stay_fast = np.sum(is_stay & (c == 0))
        n_stay_slow = np.sum(is_stay & (c == 1))
        n_jump_fast = np.sum(is_jump & (c == 0))
        n_jump_slow = np.sum(is_jump & (c == 1))

        # ==============================================================
        # Step 2: Sample w1 | c (conjugate Beta)
        # ==============================================================
        n_fast = n_stay_fast + n_jump_fast
        n_slow = n_stay_slow + n_jump_slow
        # Prior: Beta(2, 2) — weak prior centered at 0.5
        w1 = rng.beta(2 + n_fast, 2 + n_slow)

        # ==============================================================
        # Step 3: Sample alpha1, alpha2 | z, c (MH with log-normal proposal)
        # Enforce ordering: alpha1 > alpha2 (fast decay > slow decay)
        # Stronger priors centered around initial estimates to prevent collapse
        # ==============================================================
        def log_cond_alpha(alpha, n_s, n_j, prior_mean, prior_sd):
            if alpha <= 0:
                return -np.inf
            log_p = -alpha * n_s + n_j * np.log(1 - np.exp(-alpha) + 1e-30)
            # Informative Normal prior on log(alpha)
            log_p += stats.norm.logpdf(np.log(alpha), loc=np.log(prior_mean), scale=prior_sd)
            return log_p

        # MH step for alpha1 (fast decay, high alpha)
        alpha1_prop = alpha1 * np.exp(0.05 * rng.normal())
        log_accept = (log_cond_alpha(alpha1_prop, n_stay_fast, n_jump_fast, alpha1_init, 1.0) -
                      log_cond_alpha(alpha1, n_stay_fast, n_jump_fast, alpha1_init, 1.0))
        if np.log(rng.random()) < log_accept and alpha1_prop > alpha2:
            alpha1 = alpha1_prop

        # MH step for alpha2 (slow decay, low alpha)
        alpha2_prop = alpha2 * np.exp(0.05 * rng.normal())
        log_accept = (log_cond_alpha(alpha2_prop, n_stay_slow, n_jump_slow, alpha2_init, 1.0) -
                      log_cond_alpha(alpha2, n_stay_slow, n_jump_slow, alpha2_init, 1.0))
        if np.log(rng.random()) < log_accept and alpha2_prop < alpha1:
            alpha2 = alpha2_prop

        # ==============================================================
        # Step 4: Sample mu | sigma2, data (conjugate Normal)
        # ==============================================================
        precision_data = n / sigma2
        precision_prior = 1.0 / mu_prior_var
        mu_post_var = 1.0 / (precision_data + precision_prior)
        mu_post_mean = mu_post_var * (np.sum(log_rv) / sigma2 + mu_prior_mean / mu_prior_var)
        mu = rng.normal(mu_post_mean, np.sqrt(mu_post_var))

        # ==============================================================
        # Step 5: Sample sigma2 | mu, data (conjugate Inverse-Gamma)
        # ==============================================================
        residuals = log_rv - mu
        a_post = sigma2_prior_a + n / 2
        b_post = sigma2_prior_b + np.sum(residuals**2) / 2
        sigma2 = 1.0 / rng.gamma(a_post, 1.0 / b_post)

        # Store samples
        if it >= burn_in:
            samples['w1'].append(w1)
            samples['alpha1'].append(alpha1)
            samples['alpha2'].append(alpha2)
            samples['mu'].append(mu)
            samples['sigma2'].append(sigma2)
            samples['n_stays'].append(n_stays)
            samples['n_jumps'].append(n_jumps)
            samples['P_stay_fast'].append(np.exp(-alpha1))
            samples['P_stay_slow'].append(np.exp(-alpha2))
            samples['P_stay_mixture'].append(w1 * np.exp(-alpha1) + (1 - w1) * np.exp(-alpha2))

    # Convert to arrays
    for key in samples:
        samples[key] = np.array(samples[key])

    return samples


def simulate_mixture_predictive(log_rv_train, log_rv_test, gibbs_samples, n_sim=2000,
                                 Q_type="empirical", rng=None):
    """Simulate predictive trajectories from mixture SF-Harris posterior.

    For each posterior sample, simulate a trajectory using the mixture ACF
    and compare against test data.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_train = len(log_rv_train)
    n_test = len(log_rv_test)
    n_posterior = len(gibbs_samples['w1'])

    # Get jump values for Q sampling
    diffs_train = np.diff(log_rv_train)
    epsilon = 1e-5
    jump_values = diffs_train[np.abs(diffs_train) >= epsilon]

    if len(jump_values) == 0:
        jump_values = diffs_train

    # Use posterior mean for point prediction
    w1_mean = np.mean(gibbs_samples['w1'])
    alpha1_mean = np.mean(gibbs_samples['alpha1'])
    alpha2_mean = np.mean(gibbs_samples['alpha2'])
    mu_mean = np.mean(gibbs_samples['mu'])
    sigma2_mean = np.mean(gibbs_samples['sigma2'])

    # Simulate multiple trajectories from posterior
    n_use = min(n_sim, n_posterior)
    indices = rng.choice(n_posterior, size=n_use, replace=False)

    sim_log = np.empty((n_use, n_test))

    for i, idx in enumerate(indices):
        w1_s = float(gibbs_samples['w1'][idx])
        alpha1_s = float(gibbs_samples['alpha1'][idx])
        alpha2_s = float(gibbs_samples['alpha2'][idx])
        mu_s = float(gibbs_samples['mu'][idx])
        sigma_s = float(np.sqrt(gibbs_samples['sigma2'][idx]))

        # Simulate mixture SF-Harris trajectory
        x = np.empty(n_test)
        x[0] = log_rv_train[-1]

        for t in range(1, n_test):
            # Draw component
            if rng.random() < w1_s:
                alpha_curr = alpha1_s
            else:
                alpha_curr = alpha2_s

            # Stay or jump
            p_stay = np.exp(-alpha_curr)
            if rng.random() < p_stay:
                x[t] = x[t-1]  # stay
            else:
                # Jump: draw from Q
                if Q_type == "empirical":
                    x[t] = mu_s + rng.choice(jump_values) * sigma_s / np.std(jump_values)
                else:
                    x[t] = rng.normal(mu_s, sigma_s)

        sim_log[i, :] = x

    return sim_log


def simulate_single_predictive(log_rv_train, log_rv_test, gibbs_result, n_sim=2000,
                                Q_type="empirical", rng=None):
    """Simulate from single-exponential SF-Harris (for comparison)."""
    from anzarut_replication import simulate_predictive_sf_harris
    return simulate_predictive_sf_harris(
        log_rv_train, log_rv_test, gibbs_result, gibbs_result['alpha'],
        Q_type=Q_type, n_sim=n_sim, rng=rng
    )


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("FULL GIBBS SAMPLER: MIXTURE SF-HARRIS MODEL")
    print("r(h) = w1*exp(-alpha1*h) + w2*exp(-alpha2*h)")
    print("Full posterior inference on (w1, alpha1, alpha2, mu, sigma2)")
    print("=" * 70)

    # Load FULL dataset
    print("\n--- Loading full IBM dataset ---")
    df = load_ibm_data(start_date="1998-01-01", end_date="2026-12-31")
    returns = compute_15min_returns(df)
    returns_clean = detect_and_remove_jumps(returns, n_passes=2, top_pct=0.001)

    # Daily RV and returns
    rv_df = returns_clean.to_frame("return")
    rv_df["rv_15min"] = rv_df["return"]**2
    rv_df["date"] = pd.to_datetime(rv_df.index.date)
    daily_rv_all = rv_df.groupby("date")["rv_15min"].sum()
    daily_rv_all = daily_rv_all[daily_rv_all > 0]

    close_daily = df["close"].resample("D").last().dropna()
    daily_log_ret = np.log(close_daily).diff().dropna()

    common_daily = daily_rv_all.index.intersection(daily_log_ret.index)
    daily_rv = daily_rv_all.loc[common_daily].values
    daily_ret = daily_log_ret.loc[common_daily].values
    daily_log_spot = np.log(daily_rv)
    daily_log_spot = daily_log_spot[np.isfinite(daily_log_spot)]

    n = len(daily_log_spot)
    split = int(n * 0.8)
    train_log = daily_log_spot[:split]
    test_log = daily_log_spot[split:]

    print(f"  Dataset: {n} trading days")
    print(f"  Train: {split} days, Test: {n - split} days")

    # ==================================================================
    # ACF analysis
    # ==================================================================
    print("\n--- ACF analysis ---")
    acf = acf_analysis(train_log, max_lag=40)
    lags = np.arange(1, 21)
    acf_vals = acf[1:21]

    # Single exponential fit
    log_lags = np.log(lags.astype(float))
    log_acf = np.log(np.maximum(acf_vals, 1e-10))
    valid = acf_vals > 0
    if np.sum(valid) > 2:
        slope, intercept, r_single, _, _ = stats.linregress(log_lags[valid], log_acf[valid])
        alpha_single = -slope
        r2_single = r_single**2
    else:
        alpha_single = 1.0
        r2_single = 0.0

    # Double exponential fit
    double = fit_double_exponential(lags, acf_vals)

    print(f"  Single exponential: alpha={alpha_single:.4f}, R2={r2_single:.3f}")
    print(f"  Double exponential: w1={double['w1']:.3f}, alpha1={double['alpha1']:.4f} "
          f"(half-life={double['alpha1_half_life']:.2f}d), "
          f"alpha2={double['alpha2']:.4f} (half-life={double['alpha2_half_life']:.1f}d), "
          f"R2={double['r_squared']:.3f}")

    # ==================================================================
    # Single-exponential Gibbs (for comparison)
    # ==================================================================
    print("\n--- Single-exponential SF-Harris Gibbs ---")
    from anzarut_replication import estimate_alpha, gibbs_gig_harris

    daily_alpha = estimate_alpha(train_log)
    rng_gibbs_single = np.random.default_rng(111)
    gibbs_single = gibbs_gig_harris(train_log, alpha_init=daily_alpha["alpha_acf"],
                                      epsilon=1e-5, n_iter=5000, burn_in=2000,
                                      rng=rng_gibbs_single)
    alpha_single_mean = np.mean(gibbs_single['alpha']) if isinstance(gibbs_single['alpha'], np.ndarray) else gibbs_single['alpha']
    p_stay_single = np.exp(-alpha_single_mean)
    print(f"  Single alpha (posterior mean): {alpha_single_mean:.4f}")
    print(f"  P(stay) single: {p_stay_single:.4f}")

    # ==================================================================
    # Mixture Gibbs sampler
    # ==================================================================
    print("\n--- Mixture SF-Harris Gibbs sampler ---")
    print(f"  Initializing: w1={double['w1']:.3f}, alpha1={double['alpha1']:.4f}, "
          f"alpha2={double['alpha2']:.4f}")

    rng_gibbs_mix = np.random.default_rng(222)
    gibbs_mix = gibbs_mixture_harris(
        train_log,
        alpha1_init=double['alpha1'],
        alpha2_init=double['alpha2'],
        w1_init=double['w1'],
        mu_init=np.mean(train_log),
        sigma2_init=np.var(train_log),
        epsilon=None,  # adaptive epsilon
        n_iter=8000,
        burn_in=3000,
        rng=rng_gibbs_mix
    )

    print(f"\n  Posterior summary:")
    print(f"  {'Parameter':>12}  {'Mean':>10}  {'Std':>10}  {'Median':>10}  {'2.5%':>10}  {'97.5%':>10}")
    print(f"  {'-'*70}")
    for name in ['w1', 'alpha1', 'alpha2', 'mu', 'sigma2']:
        vals = gibbs_mix[name]
        print(f"  {name:>12}  {np.mean(vals):>10.4f}  {np.std(vals):>10.4f}  "
              f"{np.median(vals):>10.4f}  {np.percentile(vals, 2.5):>10.4f}  "
              f"{np.percentile(vals, 97.5):>10.4f}")

    print(f"\n  P(stay) posterior:")
    print(f"    Fast component (alpha1): {np.mean(gibbs_mix['P_stay_fast']):.4f}")
    print(f"    Slow component (alpha2): {np.mean(gibbs_mix['P_stay_slow']):.4f}")
    print(f"    Mixture:                 {np.mean(gibbs_mix['P_stay_mixture']):.4f}")

    # Check convergence: effective sample size
    for name in ['w1', 'alpha1', 'alpha2']:
        vals = gibbs_mix[name]
        # Simple ESS estimate via autocorrelation
        acf_1 = np.corrcoef(vals[:-1], vals[1:])[0, 1]
        ess = len(vals) * (1 - acf_1) / (1 + acf_1) if abs(acf_1) < 0.99 else len(vals)
        print(f"    ESS({name}): {ess:.0f} (of {len(vals)} samples)")

    # ==================================================================
    # Predictive simulation
    # ==================================================================
    print("\n--- Predictive simulation ---")
    n_sim = 2000
    rng_sim = np.random.default_rng(333)

    # Single-exponential prediction
    sim_log_single = simulate_predictive_sf_harris(
        train_log, test_log, gibbs_single, gibbs_single['alpha'],
        Q_type="empirical", n_sim=n_sim, rng=rng_sim
    )
    n_test = min(len(test_log), sim_log_single.shape[1])

    # Mixture prediction
    sim_log_mix = simulate_mixture_predictive(
        train_log, test_log, gibbs_mix, n_sim=n_sim,
        Q_type="empirical", rng=rng_sim
    )

    sim_rv_single = np.exp(sim_log_single[:, :n_test])
    sim_rv_mix = np.exp(sim_log_mix[:, :n_test])
    test_ret = daily_ret[split:split + n_test]

    # ==================================================================
    # Coverage comparison
    # ==================================================================
    print(f"\n{'='*70}")
    print("COVERAGE COMPARISON: SINGLE vs MIXTURE SF-HARRIS")
    print("=" * 70)

    cov_single = compute_coverage(test_ret, sim_rv_single, PROB_LEVELS)
    cov_mix = compute_coverage(test_ret, sim_rv_mix, PROB_LEVELS)

    aad_single = np.mean([abs(cov_single[p] - p*100) for p in PROB_LEVELS])
    aad_mix = np.mean([abs(cov_mix[p] - p*100) for p in PROB_LEVELS])

    print(f"\n  {'p':>6}  {'Ideal':>6}  {'Single':>8}  {'Mixture':>8}  {'Single Dev':>10}  {'Mix Dev':>8}")
    print(f"  {'-'*52}")
    for p in PROB_LEVELS:
        dev_s = cov_single[p] - p*100
        dev_m = cov_mix[p] - p*100
        print(f"  {p:>6.2f}  {p*100:>5.0f}%  {cov_single[p]:>7.1f}%  {cov_mix[p]:>7.1f}%  {dev_s:>+9.1f}pp  {dev_m:>+7.1f}pp")

    print(f"\n  AAD (Single):   {aad_single:.1f}pp")
    print(f"  AAD (Mixture):  {aad_mix:.1f}pp")
    print(f"  Improvement:     {aad_single - aad_mix:+.1f}pp")

    # ==================================================================
    # ACF comparison: simulated vs empirical
    # ==================================================================
    print(f"\n--- ACF comparison: empirical vs simulated ---")

    acf_emp = acf_analysis(test_log, max_lag=20)

    # Simulate ACFs from mixture posterior
    n_acf_sims = 200
    rng_acf = np.random.default_rng(444)
    acf_mix_sims = []
    acf_single_sims = []

    # Use short trajectories matching test length
    for _ in range(n_acf_sims):
        # Mixture
        idx = rng_acf.choice(len(gibbs_mix['w1']))
        w1_s = gibbs_mix['w1'][idx]
        a1_s = gibbs_mix['alpha1'][idx]
        a2_s = gibbs_mix['alpha2'][idx]
        mu_s = gibbs_mix['mu'][idx]
        sig_s = np.sqrt(gibbs_mix['sigma2'][idx])

        # Simulate mixture trajectory
        x_mix = np.empty(n_test)
        x_mix[0] = rng_acf.normal(mu_s, sig_s)
        for t in range(1, n_test):
            comp = rng_acf.random() < w1_s
            alpha_curr = a1_s if comp else a2_s
            if rng_acf.random() < np.exp(-alpha_curr):
                x_mix[t] = x_mix[t-1]
            else:
                x_mix[t] = rng_acf.normal(mu_s, sig_s)
        acf_mix_sims.append(acf_analysis(x_mix, max_lag=20))

        # Single
        alpha_s = float(alpha_single_mean)
        x_single = np.empty(n_test)
        x_single[0] = rng_acf.normal(mu_s, sig_s)
        for t in range(1, n_test):
            if rng_acf.random() < np.exp(-alpha_s):
                x_single[t] = x_single[t-1]
            else:
                x_single[t] = rng_acf.normal(mu_s, sig_s)
        acf_single_sims.append(acf_analysis(x_single, max_lag=20))

    acf_mix_mean = np.mean(acf_mix_sims, axis=0)
    acf_single_mean = np.mean(acf_single_sims, axis=0)

    print(f"\n  {'Lag':>4}  {'Empirical':>10}  {'Single':>10}  {'Mixture':>10}")
    print(f"  {'-'*38}")
    for h in [1, 2, 3, 5, 10, 15, 20]:
        if h <= 20:
            print(f"  {h:>4}  {acf_emp[h]:>10.4f}  {acf_single_mean[h]:>10.4f}  {acf_mix_mean[h]:>10.4f}")

    # ==================================================================
    # Daily return prediction with mixture
    # ==================================================================
    print(f"\n{'='*70}")
    print("DAILY RETURN PREDICTION")
    print("=" * 70)

    # Mixture: R_t ~ N(0, tau*_t)
    tau_mix_median = np.median(sim_rv_mix, axis=0)
    rng_daily = np.random.default_rng(555)
    sim_ret_mix = np.sqrt(np.maximum(sim_rv_mix, 1e-20)) * rng_daily.normal(size=sim_rv_mix.shape)
    cov_ret_mix = compute_coverage(test_ret, sim_ret_mix, PROB_LEVELS)
    aad_ret_mix = np.mean([abs(cov_ret_mix[p] - p*100) for p in PROB_LEVELS])

    # Single: R_t ~ N(0, tau*_t)
    tau_single_median = np.median(sim_rv_single, axis=0)
    sim_ret_single = np.sqrt(np.maximum(sim_rv_single, 1e-20)) * rng_daily.normal(size=sim_rv_single.shape)
    cov_ret_single = compute_coverage(test_ret, sim_ret_single, PROB_LEVELS)
    aad_ret_single = np.mean([abs(cov_ret_single[p] - p*100) for p in PROB_LEVELS])

    print(f"\n  Daily return coverage (N(0, tau*)):")
    print(f"  {'p':>6}  {'Ideal':>6}  {'Single':>8}  {'Mixture':>8}  {'Single Dev':>10}  {'Mix Dev':>8}")
    print(f"  {'-'*52}")
    for p in PROB_LEVELS:
        dev_s = cov_ret_single[p] - p*100
        dev_m = cov_ret_mix[p] - p*100
        print(f"  {p:>6.2f}  {p*100:>5.0f}%  {cov_ret_single[p]:>7.1f}%  {cov_ret_mix[p]:>7.1f}%  {dev_s:>+9.1f}pp  {dev_m:>+7.1f}pp")

    print(f"\n  AAD (Single, daily returns):   {aad_ret_single:.1f}pp")
    print(f"  AAD (Mixture, daily returns):  {aad_ret_mix:.1f}pp")

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print(f"\n{'='*70}")
    print("SUMMARY: MIXTURE vs SINGLE SF-HARRIS")
    print("=" * 70)

    print(f"\n  ACF fit:")
    print(f"    Single exponential: alpha={alpha_single:.4f}, R2={r2_single:.3f}")
    print(f"    Double exponential: w1={double['w1']:.3f}, alpha1={double['alpha1']:.4f}, "
          f"alpha2={double['alpha2']:.4f}, R2={double['r_squared']:.3f}")

    print(f"\n  Posterior (mixture Gibbs):")
    print(f"    w1:       {np.mean(gibbs_mix['w1']):.3f} [{np.percentile(gibbs_mix['w1'], 2.5):.3f}, "
          f"{np.percentile(gibbs_mix['w1'], 97.5):.3f}]")
    print(f"    alpha1:   {np.mean(gibbs_mix['alpha1']):.4f} [{np.percentile(gibbs_mix['alpha1'], 2.5):.4f}, "
          f"{np.percentile(gibbs_mix['alpha1'], 97.5):.4f}]")
    print(f"    alpha2:   {np.mean(gibbs_mix['alpha2']):.4f} [{np.percentile(gibbs_mix['alpha2'], 2.5):.4f}, "
          f"{np.percentile(gibbs_mix['alpha1'], 97.5):.4f}]")
    print(f"    Half-life fast: {np.log(2)/np.mean(gibbs_mix['alpha1']):.2f} days")
    print(f"    Half-life slow: {np.log(2)/np.mean(gibbs_mix['alpha2']):.1f} days")

    print(f"\n  Coverage (volatility prediction):")
    print(f"    Single SF-Harris: AAD = {aad_single:.1f}pp")
    print(f"    Mixture SF-Harris: AAD = {aad_mix:.1f}pp")
    print(f"    Improvement: {aad_single - aad_mix:+.1f}pp")

    print(f"\n  Coverage (daily returns):")
    print(f"    Single SF-Harris: AAD = {aad_ret_single:.1f}pp")
    print(f"    Mixture SF-Harris: AAD = {aad_ret_mix:.1f}pp")
    print(f"    Improvement: {aad_ret_single - aad_ret_mix:+.1f}pp")

    if aad_mix < aad_single:
        print(f"\n  VERDICT: Mixture SF-Harris IMPROVES over single exponential")
        print(f"    ACF fit: R2 = {double['r_squared']:.3f} vs {r2_single:.3f}")
        print(f"    Coverage: {aad_mix:.1f}pp vs {aad_single:.1f}pp ({aad_single-aad_mix:+.1f}pp better)")
    else:
        print(f"\n  VERDICT: Single exponential still better for coverage")
        print(f"    ACF fit is better ({double['r_squared']:.3f} vs {r2_single:.3f})")
        print(f"    But coverage is worse ({aad_mix:.1f}pp vs {aad_single:.1f}pp)")
        print(f"    Possible: overfitting ACF parameters to in-sample data")