"""Anzarut replication: Full GIG-Harris SV pipeline (Sections 4-5, Appendices B-C).

Follows Anzarut's exact methodology:
1. Load IBM 1-minute data → 15-min returns
2. Jump detection via bipower variation (remove top 0.1%)
3. Compute daily realized variance from cleaned returns
4. Periodicity adjustment (divide out intraday U-shape, Appendix C.2)
5. Fit GIG-Harris SV model (single exponential ACF, GIG-Q)
6. Gibbs sampler for full posterior (α, λ, κ, η, μ, β)
7. Posterior predictive intervals
8. Coverage validation (Table 3)

Target: replicate Table 3 (25, 51, 75, 84, 89, 93 at p = 0.25, ..., 0.95)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import kv as bessel_kv
from scipy import stats

from src.sf_harris.distributions import GIGQ

DATA_PATH = Path(r"C:\Users\angve\OneDrive\Desktop\Servicio\Libros\SF-Harris\IBM.txt")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"


# ===========================================================================
# 1. Data loading and bar construction
# ===========================================================================
def load_ibm_data(start_date="2012-01-01", end_date="2014-12-31"):
    df = pd.read_csv(
        DATA_PATH, header=None,
        names=["date", "time", "open", "high", "low", "close", "volume"],
    )
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="%m/%d/%Y %H:%M")
    df = df.drop(columns=["date", "time"]).set_index("datetime")
    df = df.loc[start_date:end_date]
    df["dollar_volume"] = df["close"] * df["volume"]
    return df


def compute_15min_returns(df):
    close_15min = df["close"].resample("15min").last().dropna()
    return np.log(close_15min).diff().dropna()


def build_dollar_bars_rolling(df, lookback_days=30, bars_per_day=26):
    """Dollar bars with a CAUSAL rolling threshold -> leak-free and ~bars_per_day/day.

    Each trading day's threshold = trailing ``lookback_days`` trading days' average
    dollars per active 15-min block = (mean trailing daily dollar volume) / bars_per_day.
    Calibrated from trading-hours-only daily dollar volume (groupby date), so the
    overnight/weekend empty bins that diluted the old global
    ``resample('15min').sum().mean()`` threshold (~5.4x too small -> ~101 bars/day) never
    enter the average.

    Causal: day d's threshold uses daily dollar volume of days [d-lookback, d-1] only
    (shift(1)), so bar boundaries never look at the future -> leak-free and adaptive to
    volume drift. The first trading day (no history) is skipped; min_periods=1 then
    expands the warmup until ``lookback_days`` fill.
    """
    daily_dv = df["dollar_volume"].groupby(df.index.normalize()).sum()
    avg_daily = daily_dv.shift(1).rolling(lookback_days, min_periods=1).mean()
    thr_map = (avg_daily / bars_per_day).to_dict()

    cum = 0.0
    first_close = None
    recs = []
    cur_day = None
    thr = np.nan
    for idx, row in df.iterrows():
        d = idx.normalize()
        if d != cur_day:
            cur_day = d
            t = thr_map.get(d, np.nan)
            if not np.isnan(t):
                thr = t
        if np.isnan(thr):
            continue  # warmup: no causal threshold yet
        if first_close is None:
            first_close = row["close"]
        cum += row["dollar_volume"]
        if cum >= thr:
            recs.append({"datetime": idx, "return": np.log(row["close"] / first_close)})
            cum = 0.0
            first_close = None
    if not recs:
        return pd.Series(dtype=float)
    return pd.DataFrame(recs).set_index("datetime")["return"]


# ===========================================================================
# 2. Jump detection via bipower variation (Anzarut Section 4.2)
# ===========================================================================
def bipower_variation(returns):
    """BPV = (pi/2) * sum(|r_t| * |r_{t-1}|) — robust to jumps."""
    abs_ret = np.abs(returns.values)
    if len(abs_ret) < 2:
        return 0.0
    return (np.pi / 2) * np.sum(abs_ret[1:] * abs_ret[:-1])


def fit_jump_thresholds(returns, n_passes=2, top_pct=0.001):
    """Fit jump-detection thresholds on `returns` (the TRAIN set), per pass.

    Same per-pass logic as detect_and_remove_jumps in fit mode — the threshold is a
    single global scalar per pass (the top ``top_pct`` quantile of the positive jump
    measures over the current cleaned series) — but returns the list of thresholds
    instead of the cleaned series. Apply them later to any series with
    ``detect_and_remove_jumps(returns, fixed_thresholds=thresholds)``.

    Returns a list[float] of length <= n_passes (shorter if an early break triggers).
    """
    cleaned = returns.copy()
    thresholds = []
    for _ in range(n_passes):
        all_jump_measures = np.maximum(
            cleaned.values**2 - (np.pi / 2) * np.abs(cleaned.values) * np.abs(
                np.concatenate([cleaned.values[1:], [0]])
            ),
            0,
        )
        positive = all_jump_measures[all_jump_measures > 0]
        if len(positive) == 0:
            break
        threshold = np.quantile(positive, 1 - top_pct)
        thresholds.append(threshold)
        jump_mask = all_jump_measures > threshold
        if not np.any(jump_mask):
            break
        cleaned = cleaned.drop(cleaned.index[jump_mask])
    return thresholds


def detect_and_remove_jumps(returns, n_passes=2, top_pct=0.001, fixed_thresholds=None):
    """Anzarut's jump detection: flag top 0.1% intervals by RV-BPV jump measure, drop.

    Run n_passes times (Anzarut found 2 passes sufficient). The threshold is a single
    global scalar per pass.

    Leak-free split: pass ``fixed_thresholds`` (a list[float] from ``fit_jump_thresholds``
    fit on TRAIN only) to apply pre-fit thresholds to a held-out series without
    re-quantiling on it. ``fixed_thresholds=None`` (default) fits+applies on the same
    series — backward compatible with every existing caller.
    """
    cleaned = returns.copy()

    # ponytail: fit mode is just apply-on-self with the thresholds fit would produce;
    # detect_and_remove_jumps(x) is bit-identical to fit_jump_thresholds(x) then apply.
    if fixed_thresholds is None:
        fixed_thresholds = fit_jump_thresholds(returns, n_passes=n_passes, top_pct=top_pct)

    for threshold in fixed_thresholds:
        all_jump_measures = np.maximum(
            cleaned.values**2 - (np.pi / 2) * np.abs(cleaned.values) * np.abs(
                np.concatenate([cleaned.values[1:], [0]])
            ),
            0,
        )
        jump_mask = all_jump_measures > threshold
        if not np.any(jump_mask):
            break
        cleaned = cleaned.drop(cleaned.index[jump_mask])

    n_removed = len(returns) - len(cleaned)
    print(f"  Jump detection removed {n_removed} returns ({n_removed/len(returns)*100:.2f}%)")
    return cleaned


def split_returns_by_date(returns, train_frac=0.8):
    """Split an intraday return series by a single temporal boundary.

    Sorts the unique trading days, assigns the first ``train_frac`` of them to train
    and the rest to test, and partitions the 15-min returns accordingly. Guarantees one
    shared temporal cut for intraday returns, daily realized variance and close-to-close
    returns — call this BEFORE any cleaning so the test window never informs the jump
    threshold or the periodicity. The boundary (last train day) is
    ``train.index.date.max()``.
    """
    unique_dates = pd.Index(sorted(set(returns.index.date)))
    n_train = int(len(unique_dates) * train_frac)
    train_dates = set(unique_dates[:n_train])
    day = pd.Series(returns.index.date, index=returns.index)
    train = returns[day.isin(train_dates).values]
    test = returns[~day.isin(train_dates).values]
    return train, test


# ===========================================================================
# 3. Daily realized variance
# ===========================================================================
def compute_daily_rv(returns):
    """Compute daily realized variance from intraday returns."""
    if isinstance(returns, pd.DataFrame):
        ret = returns["return"]
    else:
        ret = returns
    rv = ret.groupby(ret.index.date).apply(lambda x: np.sum(x**2))
    rv.index = pd.to_datetime(rv.index)
    return rv


# ===========================================================================
# 4. Periodicity adjustment (Anzarut Appendix C.2)
# ===========================================================================
def estimate_periodicity(returns):
    """Estimate periodicity function f(t) following Anzarut Appendix C.2.

    f(t) = E[RV_15min(t) / RV_daily_mean(t)]
    Normalized so that E[f(t)] = 1.
    """
    rv_df = returns.to_frame("return")
    rv_df["rv_15min"] = rv_df["return"]**2
    rv_df["date"] = rv_df.index.date
    rv_df["time"] = rv_df.index.strftime("%H:%M")

    # Daily RV
    daily_rv = rv_df.groupby("date")["rv_15min"].sum()
    rv_df["rv_daily"] = rv_df["date"].map(daily_rv)
    n_per_day = rv_df.groupby("date")["rv_15min"].transform("count")
    rv_df["rv_daily_mean"] = rv_df["rv_daily"] / n_per_day

    # Ratio: intraday RV / daily mean
    rv_df["ratio"] = rv_df["rv_15min"] / rv_df["rv_daily_mean"]

    # Average ratio by time-of-day
    periodicity = rv_df.groupby("time")["ratio"].mean()

    # Normalize: mean over day = 1
    periodicity = periodicity / periodicity.mean()

    return periodicity


def adjust_for_periodicity(returns, periodicity):
    """Remove periodic component: divide each return by sqrt(f(t))."""
    time_map = returns.index.strftime("%H:%M")
    f_t = np.array([periodicity.get(t, 1.0) for t in time_map])

    # Adjusted returns: r_adj = r / sqrt(f(t))
    adjusted = returns / np.sqrt(f_t)
    return adjusted


# ===========================================================================
# 5. GIG-Harris SV model estimation
# ===========================================================================
def fit_gig_mle(rv_data):
    """Fit GIG distribution to RV data via maximum likelihood.

    GIG(lam, kappa, eta): f(x) proportional to x^{lam-1} * exp{-(kappa/2)(eta*x + 1/(eta*x))}

    We parameterize and optimize in log-space for numerical stability.
    """
    x = rv_data[rv_data > 0]
    n = len(x)
    log_x = np.log(x)

    def neg_log_lik(params):
        lam = params[0]
        log_kappa = params[1]
        log_eta = params[2]
        kappa = np.exp(log_kappa)
        eta = np.exp(log_eta)

        # GIG log-likelihood
        try:
            log_kl = np.log(bessel_kv(lam, kappa)) if kappa > 0 else 50.0
        except Exception:
            return 1e10

        ll = (n * lam * np.log(eta)
              - n * log_kl
              + (lam - 1) * np.sum(log_x)
              - (kappa / 2) * (eta * np.sum(1.0/x) + np.sum(x) / eta))

        return -ll

    # Method-of-moments initialization
    mean_x = np.mean(x)
    var_x = np.var(x)
    kappa_init = mean_x**2 / var_x if var_x > 0 else 1.0
    eta_init = kappa_init / mean_x

    best_result = None
    best_nll = np.inf

    starts = [
        [0.0, np.log(kappa_init), np.log(eta_init)],
        [-0.5, np.log(kappa_init), np.log(eta_init)],
        [1.0, np.log(kappa_init), np.log(eta_init)],
        [0.0, np.log(5.0), np.log(1.0)],
        [-1.0, np.log(10.0), np.log(0.5)],
    ]

    for x0 in starts:
        try:
            result = minimize(neg_log_lik, x0=x0, method="Nelder-Mead",
                            options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8})
            if result.fun < best_nll:
                best_nll = result.fun
                best_result = result
        except Exception:
            continue

    if best_result is None:
        return {"lam": 0.0, "kappa": kappa_init, "eta": eta_init, "converged": False}

    lam = best_result.x[0]
    kappa = np.exp(best_result.x[1])
    eta = np.exp(best_result.x[2])

    return {"lam": lam, "kappa": kappa, "eta": eta, "converged": best_result.success}


def estimate_alpha(log_rv):
    """Estimate alpha (jump rate) from log-RV using MLE for continuous data.

    For continuous Q (GIG), every observation is a "change" from the previous.
    The MLE for alpha is: alpha_hat = -log(p_stay), where p_stay is estimated
    by comparing how often consecutive observations are "close enough" to be
    considered the same state.

    Anzarut uses Gibbs sampling for alpha. We also compute the simple
    autocorrelation-based estimate: alpha = -log(ACF(1)).
    """
    # ACF-based estimate: alpha = -log(rho(1))
    mu = np.mean(log_rv)
    var = np.var(log_rv)
    if var < 1e-10:
        return {"alpha_acf": 1.0}

    rho1 = np.mean((log_rv[:-1] - mu) * (log_rv[1:] - mu)) / var
    alpha_acf = -np.log(max(rho1, 0.01))

    return {"alpha_acf": alpha_acf, "rho1": rho1}


# ===========================================================================
# 6. Gibbs sampler for GIG-Harris SV model (Anzarut Sections 3 & B.1)
# ===========================================================================
def gibbs_gig_harris(log_rv, alpha_init=None, epsilon=1e-5, n_iter=5000, burn_in=2000, rng=None):
    """Full Gibbs sampler for the GIG-Harris SV model with ε threshold.

    KEY: Anzarut's thesis (p.66) uses ε ≈ 10⁻⁵ to identify "stays" vs "jumps":
    when |x_i - x_{i-1}| < ε, treat as a stay (z_i = 0); otherwise a jump (z_i = 1).
    This is essential for continuous Q (GIG) — without it, every observation is a "jump"
    and α → ∞ (no persistence).

    Gibbs steps:
      1. z_i | α: Bernoulli with ε threshold
      2. α | z: Gamma (Gibbs-b conjugate update)
      3. μ, σ²: Normal-Inverse-Gamma conjugate
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(log_rv)
    x = log_rv.copy()

    # Initial estimates
    mu_init = np.mean(x)
    sigma_init = np.std(x)

    if alpha_init is None:
        alpha_init = estimate_alpha(x)["alpha_acf"]

    # Initialize z using ε threshold
    diffs = np.abs(np.diff(x))
    z = (diffs >= epsilon).astype(int)  # 1 = jump, 0 = stay
    n_stays = np.sum(z == 0)
    n_jumps = np.sum(z == 1)
    print(f"  epsilon threshold: {epsilon:.0e}, stays: {n_stays}/{len(diffs)} ({n_stays/len(diffs)*100:.1f}%), "
          f"jumps: {n_jumps}/{len(diffs)} ({n_jumps/len(diffs)*100:.1f}%)")

    # Initialize
    alpha = alpha_init
    mu = mu_init
    sigma = sigma_init

    # Prior hyperparameters
    c_alpha = 0.01   # rate for α prior ~ Exp(c)
    mu0 = mu_init
    kappa0_mu = 0.01
    a0 = 3.0
    b0 = sigma_init**2

    # Storage
    samples = {
        "alpha": np.empty(n_iter),
        "mu": np.empty(n_iter),
        "sigma": np.empty(n_iter),
        "n_jumps": np.empty(n_iter),
    }

    print(f"  Running Gibbs sampler: {n_iter} iterations, {burn_in} burn-in")
    print(f"  Initial: alpha={alpha:.3f}, mu={mu:.4f}, sigma={sigma:.4f}")

    for it in range(n_iter):
        # --- Step 1: Update z_i (jump indicators) using ε threshold ---
        # P(z_i=1 | α, x_i, x_{i-1}) = (1 - e^{-α}) * Q(x_i) / [(1-e^{-α})*Q(x_i) + e^{-α}*δ(x_i = x_{i-1})]
        # Simplification with ε: if |x_i - x_{i-1}| < ε, z_i = 0 (stay); else z_i = 1 (jump)
        diffs = np.abs(np.diff(x))
        z = (diffs >= epsilon).astype(int)
        # Note: z has length n-1 (for observations x_1, ..., x_{n-1})

        # --- Step 2: Update alpha (Gibbs-b: conjugate Gamma) ---
        # m = number of jumps
        m = np.sum(z)
        # Inter-arrival times: consecutive indices where z=1
        jump_indices = np.where(z == 1)[0]
        if len(jump_indices) > 0:
            # Times between consecutive jumps (in units of observations)
            inter_arrivals = np.diff(np.concatenate([[-1], jump_indices]))
            total_inter_arrival = np.sum(inter_arrivals)
        else:
            total_inter_arrival = n

        # alpha | z ~ Gamma(m+1, total_inter_arrival + c)
        alpha = rng.gamma(m + 1, 1.0 / (total_inter_arrival + c_alpha))

        # --- Step 3: Update mu and sigma (conjugate Normal-Inverse-Gamma) ---
        kappa_n = kappa0_mu + n
        mu_n = (kappa0_mu * mu0 + n * np.mean(x)) / kappa_n
        mu = rng.normal(mu_n, sigma / np.sqrt(kappa_n))

        ss = np.sum((x - mu)**2)
        a_n = a0 + n / 2
        b_n = b0 + ss / 2
        sigma2 = 1.0 / rng.gamma(a_n, 1.0 / b_n)
        sigma = np.sqrt(max(sigma2, 1e-6))

        samples["alpha"][it] = alpha
        samples["mu"][it] = mu
        samples["sigma"][it] = sigma
        samples["n_jumps"][it] = m

    # Discard burn-in
    result = {k: v[burn_in:] for k, v in samples.items()}

    p_stay_mean = np.exp(-np.mean(result["alpha"]))
    print(f"  Posterior means:")
    print(f"    alpha:    {np.mean(result['alpha']):.3f} (std={np.std(result['alpha']):.3f})")
    print(f"    mu:       {np.mean(result['mu']):.4f} (std={np.std(result['mu']):.4f})")
    print(f"    sigma:    {np.mean(result['sigma']):.4f} (std={np.std(result['sigma']):.4f})")
    print(f"    P(stay):  {p_stay_mean:.4f} (mean)")
    print(f"    n_jumps:  {np.mean(result['n_jumps']):.1f}/{n-1} (mean)")

    return result


# ===========================================================================
# 7. Posterior predictive simulation (SF-Harris transition kernel)
# ===========================================================================
def simulate_predictive_sf_harris(log_rv_train, log_rv_test, gibbs_posterior,
                                   alpha_posterior, Q_type="empirical",
                                   n_sim=2000, rng=None):
    """Simulate SF-Harris posterior predictive trajectories.

    Uses the actual SF-Harris transition kernel:
      P(x_{t+1} | x_t) = e^{-alpha} * delta(x_{t+1} = x_t) + (1 - e^{-alpha}) * Q(x_{t+1})

    This is a MIXTURE: with prob e^{-alpha} stay at current value, with prob
    1-e^{-alpha} draw a new value from Q.

    For continuous Q, the "stay" component means a point mass at x_t,
    which creates wider prediction intervals than the Normal approximation.

    Q_type: "empirical" uses resampled training data, "normal" uses Gaussian.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_test = len(log_rv_test)
    n_post = len(alpha_posterior)

    # Centered training data for empirical Q
    emp_data = log_rv_train - np.mean(log_rv_train)

    simulated = np.empty((n_sim, n_test))
    last_val = log_rv_train[-1]

    for s in range(n_sim):
        # Draw from posterior
        idx = rng.integers(0, n_post)
        alpha = alpha_posterior[idx]
        mu = gibbs_posterior["mu"][idx]
        sigma = gibbs_posterior["sigma"][idx]

        p_stay = np.exp(-alpha)

        # Simulate forward using SF-Harris transition kernel
        current = last_val
        for i in range(n_test):
            if rng.random() < p_stay:
                # Stay at current value
                pass
            else:
                # Draw new value from Q
                if Q_type == "empirical":
                    current = mu + rng.choice(emp_data)
                else:  # normal
                    current = rng.normal(mu, sigma)

            simulated[s, i] = current

    return simulated


def simulate_predictive_sf_harris_vec(log_rv_train, log_rv_test, gibbs_posterior,
                                      alpha_posterior, Q_type="empirical",
                                      n_sim=2000, rng=None):
    """Vectorized-over-n_sim version of simulate_predictive_sf_harris.

    Same SF-Harris transition kernel and same (n_sim, n_test) output, but the
    n_sim paths advance in lockstep: `current` is a length-n_sim vector and each
    of the n_test sequential steps is one vectorized numpy op. ~30x faster than
    the per-path Python loop. Not bit-identical to the loop (draw order differs)
    but identical in distribution.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_test = len(log_rv_test)
    n_post = len(alpha_posterior)
    emp = np.asarray(log_rv_train - np.mean(log_rv_train))
    n_emp = emp.size

    idx = rng.integers(0, n_post, size=n_sim)
    alpha = np.asarray(alpha_posterior)[idx]
    mu = np.asarray(gibbs_posterior["mu"])[idx]
    sigma = np.asarray(gibbs_posterior["sigma"])[idx]
    p_stay = np.exp(-alpha)                 # (n_sim,)

    simulated = np.empty((n_sim, n_test))
    current = np.full(n_sim, float(log_rv_train[-1]))
    for i in range(n_test):
        u = rng.random(n_sim)
        jump = u >= p_stay                    # loop stays when u < p_stay
        n_jump = int(jump.sum())
        if n_jump:
            if Q_type == "empirical":
                q = emp[rng.integers(0, n_emp, size=n_jump)]
                current[jump] = mu[jump] + q
            else:  # normal
                current[jump] = rng.normal(mu[jump], sigma[jump])
        simulated[:, i] = current
    return simulated


def simulate_predictive_mixture(log_rv_train, log_rv_test, params,
                                  bootstrap_params=None, Q_type="empirical",
                                  n_sim=2000, rng=None):
    """Simulate mixture SF-Harris predictive trajectories.

    Uses the continuous-time renewal construction with double exponential ACF.
    Draws parameters from bootstrap for uncertainty propagation.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_test = len(log_rv_test)
    mu = params["mu"]
    sigma_Q = params["sigma_Q"]
    sigma_obs = params["sigma_obs"]
    R = params["R"]
    w1, alpha1, w2, alpha2 = params["w1"], params["alpha1"], params["w2"], params["alpha2"]
    w_mix = np.array([w1, w2])
    w_norm = w_mix / np.sum(w_mix)
    alphas = np.array([alpha1, alpha2])

    emp_data = log_rv_train - mu if Q_type == "empirical" else None
    start_val = log_rv_train[-1]

    simulated = np.empty((n_sim, n_test))

    for s in range(n_sim):
        # Optionally draw from bootstrap parameter uncertainty
        if bootstrap_params is not None:
            bp = bootstrap_params[rng.integers(0, len(bootstrap_params))]
            sim_mu = bp["mu"]
            sim_sigma_Q = bp["sigma_Q"]
            sim_sigma_obs = bp["sigma_obs"]
            sim_w1, sim_alpha1 = bp["w1"], bp["alpha1"]
            sim_w2, sim_alpha2 = bp["w2"], bp["alpha2"]
            sim_w_norm = np.array([sim_w1, sim_w2])
            sim_w_norm = sim_w_norm / np.sum(sim_w_norm)
            sim_alphas = np.array([sim_alpha1, sim_alpha2])
            sim_R = bp["R"]
            # Variance scaling
            sim_model_var = sim_sigma_Q**2 + sim_sigma_obs**2
            sim_data_var = np.var(log_rv_train)
            if sim_model_var > 0:
                vscale = np.sqrt(sim_data_var / sim_model_var)
                sim_sigma_Q *= vscale
                sim_sigma_obs *= vscale
        else:
            sim_mu, sim_sigma_Q, sim_sigma_obs = mu, sigma_Q, sigma_obs
            sim_w_norm, sim_alphas = w_norm, alphas
            sim_R = R

        current_val = start_val

        for i in range(n_test):
            t = 0.0
            while t < 1.0:
                component = rng.choice(len(sim_w_norm), p=sim_w_norm)
                rho = sim_alphas[component]
                wait = rng.exponential(1.0 / rho)
                t += wait
                if t < 1.0:
                    if Q_type == "empirical":
                        current_val = sim_mu + rng.choice(emp_data)
                    else:
                        current_val = rng.normal(sim_mu, sim_sigma_Q)

            simulated[s, i] = current_val + rng.normal(0, sim_sigma_obs)

    return simulated


# ===========================================================================
# 8. Coverage validation (Table 3)
# ===========================================================================
def compute_coverage(actual, simulated, prob_levels):
    n_obs = len(actual)
    min_len = min(n_obs, simulated.shape[1])
    act = actual[:min_len]
    sims = simulated[:, :min_len]

    coverage = {}
    for p in prob_levels:
        lower = np.quantile(sims, (1 - p) / 2, axis=0)
        upper = np.quantile(sims, (1 + p) / 2, axis=0)
        inside = np.sum((act >= lower) & (act <= upper))
        coverage[p] = inside / min_len * 100
    return coverage


# ===========================================================================
# Main pipeline
# ===========================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("ANZARUT REPLICATION: Full GIG-Harris SV Pipeline")
    print("Sections 4-5, Appendices B-C")
    print("=" * 70)

    ANZARUT_TABLE3 = {0.25: 25, 0.50: 51, 0.75: 75, 0.85: 84, 0.90: 89, 0.95: 93}
    prob_levels = [0.25, 0.50, 0.75, 0.85, 0.90, 0.95]

    # ==================================================================
    # STEP 1: Load IBM 1-minute data
    # ==================================================================
    print("\n--- Step 1: Load IBM data ---")
    df = load_ibm_data()
    print(f"  Loaded {len(df)} minute bars")

    # ==================================================================
    # STEP 2: Compute 15-minute returns
    # ==================================================================
    print("\n--- Step 2: Compute 15-minute returns ---")
    returns = compute_15min_returns(df)
    print(f"  {len(returns)} 15-min return observations")

    # ==================================================================
    # STEP 3: Jump detection — leak-free split: cut by date BEFORE cleaning,
    # fit jump thresholds on TRAIN, apply to both windows
    # ==================================================================
    print("\n--- Step 3: Jump detection (split-by-date, fit on train) ---")
    train_returns, test_returns = split_returns_by_date(returns, train_frac=0.8)
    thresholds = fit_jump_thresholds(train_returns, n_passes=2, top_pct=0.001)
    train_clean = detect_and_remove_jumps(train_returns, fixed_thresholds=thresholds)
    test_clean = detect_and_remove_jumps(test_returns, fixed_thresholds=thresholds)
    print(f"  Clean returns: train={len(train_clean)}, test={len(test_clean)} (from {len(returns)})")

    # ==================================================================
    # STEP 4: Periodicity adjustment — estimate on TRAIN clean (leak-free;
    # also fixes the prior inconsistency of estimating on raw, un-cleaned returns)
    # ==================================================================
    print("\n--- Step 4: Periodicity adjustment ---")
    periodicity = estimate_periodicity(train_clean)

    # Show periodicity pattern
    morning_vals = [v for t, v in periodicity.items() if t < "10:00"]
    midday_vals = [v for t, v in periodicity.items() if "11:00" <= t < "14:00"]
    afternoon_vals = [v for t, v in periodicity.items() if t >= "15:00"]

    print(f"  Periodicity range: {periodicity.min():.2f} to {periodicity.max():.2f}")
    if morning_vals and midday_vals and afternoon_vals:
        print(f"  Morning avg: {np.mean(morning_vals):.2f}, "
              f"Midday avg: {np.mean(midday_vals):.2f}, "
              f"Afternoon avg: {np.mean(afternoon_vals):.2f}")
        if np.mean(morning_vals) > np.mean(midday_vals) and np.mean(afternoon_vals) > np.mean(midday_vals):
            print("  -> U-shape detected (high at open/close)")

    # Adjust returns for periodicity (leak-free: periodicity fit on train, applied to both)
    train_adjusted = adjust_for_periodicity(train_clean, periodicity)
    test_adjusted = adjust_for_periodicity(test_clean, periodicity)

    # ==================================================================
    # STEP 5: Daily realized variance — per window, then concat train+test so the
    # downstream [:split_idx]/[split_idx:] slicing stays a clean temporal cut
    # ==================================================================
    print("\n--- Step 5: Daily realized variance ---")

    def _window_log_rv(clean_ret, adj_ret):
        """Daily log-RV (unadjusted and adjusted) for one window, aligned by date."""
        rv_u = compute_daily_rv(clean_ret)
        rv_a = compute_daily_rv(adj_ret)
        common = rv_u.index.intersection(rv_a.index)
        log_u = np.log(rv_u.loc[common].values)
        log_a = np.log(rv_a.loc[common].values)
        fin = np.isfinite(log_u) & np.isfinite(log_a)
        return log_u[fin], log_a[fin]

    log_u_train, log_a_train = _window_log_rv(train_clean, train_adjusted)
    log_u_test, log_a_test = _window_log_rv(test_clean, test_adjusted)

    log_rv_unadj_final = np.concatenate([log_u_train, log_u_test])
    log_rv_adj_final = np.concatenate([log_a_train, log_a_test])
    split_idx = len(log_u_train)  # train/test boundary (train days come first chronologically)
    n_total = len(log_rv_unadj_final)

    # Two versions: unadjusted and periodicity-adjusted
    print(f"  Total days: {n_total} (train={split_idx}, test={n_total - split_idx})")
    print(f"  Unadjusted: mean={np.mean(log_rv_unadj_final):.4f}, std={np.std(log_rv_unadj_final):.4f}, "
          f"skew={stats.skew(log_rv_unadj_final):.2f}, kurt={stats.kurtosis(log_rv_unadj_final):.2f}")
    print(f"  Adjusted:   mean={np.mean(log_rv_adj_final):.4f}, std={np.std(log_rv_adj_final):.4f}, "
          f"skew={stats.skew(log_rv_adj_final):.2f}, kurt={stats.kurtosis(log_rv_adj_final):.2f}")

    # ==================================================================
    # STEP 6: Fit GIG distribution
    # ==================================================================
    print("\n--- Step 6: GIG distribution fit ---")
    rv_train = np.exp(log_rv_adj_final[:split_idx])
    gig_fit = fit_gig_mle(rv_train)
    print(f"  GIG parameters: lam={gig_fit['lam']:.3f}, "
          f"kappa={gig_fit['kappa']:.3f}, eta={gig_fit['eta']:.6f}")
    print(f"  Converged: {gig_fit['converged']}")

    # ==================================================================
    # STEP 7: Estimate alpha (jump rate)
    # ==================================================================
    print("\n--- Step 7: Alpha estimation ---")
    train_adj = log_rv_adj_final[:split_idx]
    test_adj = log_rv_adj_final[split_idx:]

    train_unadj = log_rv_unadj_final[:split_idx]
    test_unadj = log_rv_unadj_final[split_idx:]

    alpha_est = estimate_alpha(train_adj)
    print(f"  ACF-based alpha: {alpha_est['alpha_acf']:.3f}")
    print(f"  ACF(1) = rho1: {alpha_est['rho1']:.4f}")

    # ==================================================================
    # STEP 8: Gibbs sampler
    # ==================================================================
    print("\n--- Step 8: Gibbs sampler ---")
    # Anzarut's thesis uses ε ≈ 10⁻⁵ for stay/jump identification
    epsilon = 1e-5

    rng_gibbs = np.random.default_rng(42)
    gibbs_adj = gibbs_gig_harris(train_adj, alpha_init=alpha_est["alpha_acf"],
                                  epsilon=epsilon, n_iter=5000, burn_in=2000, rng=rng_gibbs)

    rng_gibbs2 = np.random.default_rng(43)
    gibbs_unadj = gibbs_gig_harris(train_unadj, alpha_init=estimate_alpha(train_unadj)["alpha_acf"],
                                     epsilon=epsilon, n_iter=5000, burn_in=2000, rng=rng_gibbs2)

    # ==================================================================
    # STEP 9: Posterior predictive simulation (proper SF-Harris kernel)
    # ==================================================================
    print("\n--- Step 9: Posterior predictive simulation ---")
    n_sim = 2000

    results = {}

    # --- Method A: SF-Harris + Gibbs (single exp ACF) + empirical Q ---
    print(f"  [A] SF-Harris + Gibbs + Empirical Q (adjusted)...")
    rng_sim = np.random.default_rng(123)
    sim_a_adj = simulate_predictive_sf_harris(
        train_adj, test_adj, gibbs_adj, gibbs_adj["alpha"],
        Q_type="empirical", n_sim=n_sim, rng=rng_sim)
    results["sf_harris_adj_emp"] = compute_coverage(test_adj, sim_a_adj, prob_levels)

    print(f"  [A] SF-Harris + Gibbs + Empirical Q (unadjusted)...")
    rng_sim2 = np.random.default_rng(456)
    sim_a_unadj = simulate_predictive_sf_harris(
        train_unadj, test_unadj, gibbs_unadj, gibbs_unadj["alpha"],
        Q_type="empirical", n_sim=n_sim, rng=rng_sim2)
    results["sf_harris_unadj_emp"] = compute_coverage(test_unadj, sim_a_unadj, prob_levels)

    # --- Method B: SF-Harris + Gibbs (single exp ACF) + normal Q ---
    print(f"  [B] SF-Harris + Gibbs + Normal Q (adjusted)...")
    rng_sim3 = np.random.default_rng(789)
    sim_b_adj = simulate_predictive_sf_harris(
        train_adj, test_adj, gibbs_adj, gibbs_adj["alpha"],
        Q_type="normal", n_sim=n_sim, rng=rng_sim3)
    results["sf_harris_adj_norm"] = compute_coverage(test_adj, sim_b_adj, prob_levels)

    # --- Method C: Mixture-Harris + bootstrap (periodicity-adjusted) ---
    from mixture_table3 import mixture_mom_estimate, bootstrap_mixture

    print(f"  [C] Mixture-Harris + Empirical Q + Bootstrap (adjusted)...")
    adj_params = mixture_mom_estimate(train_adj)
    print(f"      Mixture ACF (adj): w1={adj_params['w1']:.3f}, a1={adj_params['alpha1']:.3f}, "
          f"w2={adj_params['w2']:.3f}, a2={adj_params['alpha2']:.4f}, R²={adj_params['r_squared']:.4f}")

    rng_boot_adj = np.random.default_rng(42)
    adj_boot = bootstrap_mixture(train_adj, n_boot=200, rng=rng_boot_adj)

    sim_c_adj = simulate_predictive_mixture(
        train_adj, test_adj, adj_params, bootstrap_params=adj_boot,
        Q_type="empirical", n_sim=n_sim, rng=np.random.default_rng(111))
    results["mixture_adj_emp"] = compute_coverage(test_adj, sim_c_adj, prob_levels)

    # --- Method D: Mixture-Harris + bootstrap (unadjusted) ---
    print(f"  [D] Mixture-Harris + Empirical Q + Bootstrap (unadjusted)...")
    unadj_params = mixture_mom_estimate(train_unadj)
    print(f"      Mixture ACF (unadj): w1={unadj_params['w1']:.3f}, a1={unadj_params['alpha1']:.3f}, "
          f"w2={unadj_params['w2']:.3f}, a2={unadj_params['alpha2']:.4f}, R²={unadj_params['r_squared']:.4f}")

    rng_boot_unadj = np.random.default_rng(42)
    unadj_boot = bootstrap_mixture(train_unadj, n_boot=200, rng=rng_boot_unadj)

    sim_d_unadj = simulate_predictive_mixture(
        train_unadj, test_unadj, unadj_params, bootstrap_params=unadj_boot,
        Q_type="empirical", n_sim=n_sim, rng=np.random.default_rng(222))
    results["mixture_unadj_emp"] = compute_coverage(test_unadj, sim_d_unadj, prob_levels)

    # ==================================================================
    # STEP 10: Coverage validation (Table 3)
    # ==================================================================
    print(f"\n{'='*70}")
    print("TABLE 3: Coverage Validation — Full Pipeline Comparison")
    print("=" * 70)

    print(f"\n  Anzarut's Table 3 (GIG-Harris + Gibbs-b, intraday):")
    print(f"  {'p':>6}  {'Anzarut':>10}  {'Ideal':>6}")
    print(f"  {'-'*28}")
    for p in prob_levels:
        print(f"  {p:>6.2f}  {ANZARUT_TABLE3[p]:>9.0f}%  {p*100:>5.0f}%")
    anz_aad = np.mean([abs(ANZARUT_TABLE3[p] - p*100) for p in prob_levels])
    print(f"  AAD = {anz_aad:.1f}pp")

    method_labels = {
        "sf_harris_adj_emp": "SF-Harris + Gibbs + Emp (adjusted)",
        "sf_harris_unadj_emp": "SF-Harris + Gibbs + Emp (unadjusted)",
        "sf_harris_adj_norm": "SF-Harris + Gibbs + Normal (adjusted)",
        "mixture_adj_emp": "Mixture-Harris + Emp + Boot (adjusted)",
        "mixture_unadj_emp": "Mixture-Harris + Emp + Boot (unadjusted)",
    }

    aads = {}
    for key, label in method_labels.items():
        if key in results:
            cov = results[key]
            aad = np.mean([abs(cov[p] - p*100) for p in prob_levels])
            aads[key] = aad

            print(f"\n  {label}:")
            print(f"  {'p':>6}  {'Coverage':>10}  {'Ideal':>6}  {'Dev':>6}")
            print(f"  {'-'*34}")
            for p in prob_levels:
                dev = cov[p] - p*100
                print(f"  {p:>6.2f}  {cov[p]:>9.1f}%  {p*100:>5.0f}%  {dev:>+5.1f}pp")
            print(f"  AAD = {aad:.1f}pp")

    # Comparison table
    print(f"\n{'='*70}")
    print("COMPARISON — ALL METHODS")
    print("=" * 70)
    print(f"\n  {'Method':>42}  {'AAD':>6}")
    print(f"  {'-'*52}")
    print(f"  {'Anzarut (GIG+Gibbs, intraday)':>42}  {anz_aad:>5.1f}pp")
    for key, label in method_labels.items():
        if key in aads:
            print(f"  {label:>42}  {aads[key]:>5.1f}pp")
    print(f"  {'Mixture-Harris + Emp (dollar, previous)':>42}  {'2.2pp':>6}")
    print(f"  {'Mixture-Harris + Emp (calendar, previous)':>42}  {'3.2pp':>6}")

    # ==================================================================
    # STEP 11: Intraday pipeline (Anzarut's exact approach)
    # Anzarut works with spot volatility at 15-min intervals, not daily log-RV.
    # The ε threshold makes sense at 15-min granularity (many consecutive
    # spot vol values stay at the same level), not at daily granularity.
    # ==================================================================
    print(f"\n{'='*70}")
    print("INTRADAY PIPELINE (Anzarut's exact approach)")
    print("Working with 15-min spot volatility + epsilon threshold")
    print("=" * 70)

    # Compute 15-min realized variance per interval, per window (leak-free: periodicity
    # fit on train, applied to both; daily RV = sum of periodicity-adjusted 15-min RV)
    def _15min_daily_log_rv(clean_ret):
        rv = clean_ret.to_frame("return")
        rv["rv"] = rv["return"]**2
        rv["time"] = rv.index.strftime("%H:%M")
        rv["f_t"] = rv["time"].map(periodicity).fillna(1.0)
        rv["rv_adj"] = rv["rv"] / rv["f_t"]
        daily = rv.groupby(rv.index.date)["rv_adj"].sum()
        log_rv = np.log(daily.values)
        return log_rv[np.isfinite(log_rv)]

    train_15 = _15min_daily_log_rv(train_clean)
    test_15 = _15min_daily_log_rv(test_clean)
    log_rv_15_adj = np.concatenate([train_15, test_15])
    n_15 = len(log_rv_15_adj)
    split_15 = len(train_15)

    print(f"\n  Periodicity-adjusted daily log-RV:")
    print(f"    Total: {n_15} days, Train: {split_15}, Test: {n_15 - split_15}")
    print(f"    Train: mean={np.mean(train_15):.4f}, std={np.std(train_15):.4f}, "
          f"skew={stats.skew(train_15):.2f}, kurt={stats.kurtosis(train_15):.2f}")

    # Now apply Gibbs sampler with epsilon threshold at different scales
    # Test epsilon thresholds at the daily level
    # At daily level, typical diff is ~0.5, so epsilon needs to be larger
    for eps_label, eps_val in [("1e-5 (thesis)", 1e-5), ("0.01", 0.01), ("0.1", 0.1), ("0.5 (data-scale)", 0.5)]:
        diffs = np.abs(np.diff(train_15))
        n_stay = np.sum(diffs < eps_val)
        n_jump = np.sum(diffs >= eps_val)
        print(f"\n  epsilon={eps_label}: stays={n_stay}/{len(diffs)} ({n_stay/len(diffs)*100:.1f}%), "
              f"jumps={n_jump}/{len(diffs)} ({n_jump/len(diffs)*100:.1f}%)")

    # Use epsilon=0.5 as a data-driven threshold (roughly 1 std of daily diff)
    eps_daily = 0.5  # ~1 std of daily log-RV differences
    print(f"\n  Using epsilon={eps_daily} for daily data")

    rng_gibbs_eps = np.random.default_rng(55)
    gibbs_adj_eps = gibbs_gig_harris(train_15, alpha_init=estimate_alpha(train_15)["alpha_acf"],
                                      epsilon=eps_daily, n_iter=5000, burn_in=2000, rng=rng_gibbs_eps)

    # Also try with NDNJ-style alpha estimation
    diffs_train = np.abs(np.diff(train_15))
    n_stay_eps = np.sum(diffs_train < eps_daily)
    alpha_ndnj = -np.log(max(n_stay_eps / len(diffs_train), 0.01))
    print(f"  NDNJ alpha (eps={eps_daily}): {alpha_ndnj:.3f}, P(stay)={np.exp(-alpha_ndnj):.4f}")

    # Simulate with epsilon-aware Gibbs
    print(f"\n  Simulating with epsilon-aware Gibbs (eps={eps_daily})...")
    rng_sim_eps = np.random.default_rng(999)
    sim_eps_adj = simulate_predictive_sf_harris(
        train_15, test_15, gibbs_adj_eps, gibbs_adj_eps["alpha"],
        Q_type="empirical", n_sim=n_sim, rng=rng_sim_eps)
    cov_eps_adj = compute_coverage(test_15, sim_eps_adj, prob_levels)
    aad_eps_adj = np.mean([abs(cov_eps_adj[p] - p*100) for p in prob_levels])

    print(f"\n  SF-Harris + Gibbs (eps={eps_daily}) + Empirical Q (adjusted):")
    print(f"  {'p':>6}  {'Coverage':>10}  {'Ideal':>6}  {'Dev':>6}")
    print(f"  {'-'*34}")
    for p in prob_levels:
        dev = cov_eps_adj[p] - p*100
        print(f"  {p:>6.2f}  {cov_eps_adj[p]:>9.1f}%  {p*100:>5.0f}%  {dev:>+5.1f}pp")
    print(f"  AAD = {aad_eps_adj:.1f}pp")
    print(f"\n{'='*70}")
    print("DISTRIBUTION DIAGNOSTICS")
    print("=" * 70)

    print(f"\n  Test set (adjusted):  mean={np.mean(test_adj):.3f}, std={np.std(test_adj):.3f}, "
          f"skew={stats.skew(test_adj):.2f}, kurt={stats.kurtosis(test_adj):.2f}")
    print(f"  Test set (unadj):    mean={np.mean(test_unadj):.3f}, std={np.std(test_unadj):.3f}, "
          f"skew={stats.skew(test_unadj):.2f}, kurt={stats.kurtosis(test_unadj):.2f}")

    if "sf_harris_adj_emp" in results:
        flat = sim_a_adj.flatten()
        print(f"  Sim (SF-Harris adj): mean={np.mean(flat):.3f}, std={np.std(flat):.3f}, "
              f"skew={stats.skew(flat):.2f}, kurt={stats.kurtosis(flat):.2f}")
    if "mixture_adj_emp" in results:
        flat = sim_c_adj.flatten()
        print(f"  Sim (Mixture adj):   mean={np.mean(flat):.3f}, std={np.std(flat):.3f}, "
              f"skew={stats.skew(flat):.2f}, kurt={stats.kurtosis(flat):.2f}")

    # Posterior diagnostics
    print(f"\n  Posterior diagnostics (adjusted):")
    print(f"    alpha:  mean={np.mean(gibbs_adj['alpha']):.3f}, "
          f"median={np.median(gibbs_adj['alpha']):.3f}, "
          f"std={np.std(gibbs_adj['alpha']):.3f}")
    print(f"    alpha 95% CI: [{np.percentile(gibbs_adj['alpha'], 2.5):.3f}, "
          f"{np.percentile(gibbs_adj['alpha'], 97.5):.3f}]")
    print(f"    mu:     mean={np.mean(gibbs_adj['mu']):.4f}, std={np.std(gibbs_adj['mu']):.4f}")
    print(f"    sigma:  mean={np.mean(gibbs_adj['sigma']):.4f}, std={np.std(gibbs_adj['sigma']):.4f}")

    # Save results
    rows = []
    for key, label in method_labels.items():
        if key in results:
            for p in prob_levels:
                rows.append({
                    "method": key, "label": label, "p": p,
                    "coverage": results[key][p], "ideal": p*100,
                    "deviation": results[key][p] - p*100,
                    "aad": aads[key],
                })
    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUT_DIR / "anzarut_replication_coverage.csv", index=False)
    print(f"\n  Results saved to {OUTPUT_DIR / 'anzarut_replication_coverage.csv'}")