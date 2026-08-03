"""Mexican Financial Data Analysis for SF-Harris Final Deliverable.

Downloads Mexican financial data, fits SF-Harris and benchmark models,
computes coverage, VaR, and ES metrics, and saves results for the notebook.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from arch import arch_model
from scripts.anzarut_replication import (
    estimate_alpha, gibbs_gig_harris, simulate_predictive_sf_harris, compute_coverage
)

RESULTS_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "mexican_results.pkl"

# Mexican assets to analyze
MEXICAN_TICKERS = {
    "IPC": "^MXX",
    "Bimbo": "BIMBOA.MX",
    "Grupo_Mexico": "GMEXICOB.MX",
    "Walmart_Mex": "WALMEX.MX",
    "FEMSA": "FEMSAUBD.MX",
    "Cemex": "CX",
    "Banorte": "GFNORTEO.MX",
    "USD_MXN": "MXN=X",
}

PROB_LEVELS = [0.25, 0.50, 0.75, 0.85, 0.90, 0.95]
TRAIN_FRAC = 0.8
N_SIM = 2000
GIBBS_ITER = 3000
GIBBS_BURN = 1000


def download_data():
    """Download daily data for all Mexican assets."""
    import yfinance as yf
    data = {}
    for name, ticker in MEXICAN_TICKERS.items():
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="max")
            if len(hist) > 500:
                # Use adjusted close for total return
                prices = hist["Close"]
                # Filter to reasonable date range
                prices = prices[prices.index >= "2000-01-01"]
                data[name] = prices
                print(f"  {name:>15}: {len(prices)} days")
        except Exception as e:
            print(f"  {name:>15}: ERROR - {e}")
    return data


def compute_daily_returns(prices):
    """Compute log returns from price series."""
    returns = np.log(prices).diff().dropna()
    return returns


def compute_realized_volatility(returns, window=21):
    """Rolling realized volatility (annualized)."""
    return returns.rolling(window).std() * np.sqrt(252)


def fit_sf_harris(log_rv_series, n_sim=N_SIM, gibbs_iter=GIBBS_ITER, gibbs_burn=GIBBS_BURN):
    """Fit SF-Harris model on log realized volatility series."""
    log_rv = log_rv_series.values
    n = len(log_rv)
    split = int(n * TRAIN_FRAC)
    train = log_rv[:split]
    test = log_rv[split:]

    # Estimate alpha and fit Gibbs
    alpha_est = estimate_alpha(train)
    rng_gibbs = np.random.default_rng(42)
    gibbs = gibbs_gig_harris(train, alpha_init=alpha_est["alpha_acf"],
                               epsilon=1e-5, n_iter=gibbs_iter, burn_in=gibbs_burn, rng=rng_gibbs)

    # Generate predictive distribution
    rng_sim = np.random.default_rng(123)
    sim_log = simulate_predictive_sf_harris(train, test, gibbs, gibbs["alpha"],
                                              Q_type="empirical", n_sim=n_sim, rng=rng_sim)
    n_test = min(len(test), sim_log.shape[1])
    sim_rv = np.exp(sim_log[:, :n_test])

    # Emission: returns ~ N(0, tau*)
    rng_emit = np.random.default_rng(456)
    sim_ret = np.sqrt(np.maximum(sim_rv, 1e-20)) * rng_emit.normal(size=sim_rv.shape)

    return {
        "train": train,
        "test": test[:n_test],
        "sim_rv": sim_rv,
        "sim_ret": sim_ret,
        "alpha": gibbs["alpha"],
        "p_stay": np.exp(-float(np.mean(gibbs["alpha"]))),
        "mu": gibbs["mu"],
        "sigma": gibbs["sigma"],
        "split": split,
        "n_test": n_test,
    }


def fit_garch(returns_series, n_sim=N_SIM):
    """Fit GARCH(1,1) model."""
    ret = returns_series.values * 100  # scale for arch package
    n = len(ret)
    split = int(n * TRAIN_FRAC)
    train = ret[:split]
    test = ret[split:]

    model = arch_model(train, vol='Garch', p=1, q=1, dist='normal')
    fit = model.fit(disp='off')

    # Rolling 1-day forecasts
    forecasts = []
    for t in range(len(test)):
        try:
            fc = fit.forecast(horizon=1, start=split + t, method='simulation')
            var_1d = fc.variance.iloc[split + t].values[0] / 10000
            forecasts.append(var_1d)
        except Exception:
            forecasts.append(np.nan)

    pred_var = np.array(forecasts[:len(test)])
    valid = np.isfinite(pred_var) & (pred_var > 0)

    # Generate simulation paths for coverage
    rng = np.random.default_rng(789)
    sim_ret_garch = np.empty((n_sim, len(test)))
    for t in range(len(test)):
        if valid[t]:
            sim_ret_garch[:, t] = rng.normal(0, np.sqrt(pred_var[t]), size=n_sim)
        else:
            sim_ret_garch[:, t] = rng.normal(0, np.std(train / 100), size=n_sim)

    return {
        "pred_var": pred_var,
        "sim_ret": sim_ret_garch,
        "valid": valid,
        "params": {"omega": fit.params.get("omega", 0),
                    "alpha": fit.params.get("alpha[1]", 0),
                    "beta": fit.params.get("beta[1]", 0)},
    }


def compute_var_es(actual_returns, sim_returns, var_levels=[0.95, 0.99]):
    """Compute VaR and ES from simulation paths."""
    results = {}
    for level in var_levels:
        # VaR: quantile of the simulated distribution
        var_values = np.percentile(sim_returns, (1 - level) * 100, axis=0)
        es_values = np.mean(np.where(sim_returns <= var_values[np.newaxis, :], sim_returns, np.nan), axis=0)

        # Fix NaN in ES (when no simulations below VaR)
        es_values = np.where(np.isnan(es_values), var_values, es_values)

        # Violations: actual return below VaR
        violations = actual_returns < var_values
        violation_rate = np.mean(violations)
        expected_rate = 1 - level

        results[f"VaR_{level}"] = var_values
        results[f"ES_{level}"] = es_values
        results[f"violations_{level}"] = violations
        results[f"violation_rate_{level}"] = violation_rate
        results[f"expected_rate_{level}"] = expected_rate

    return results


def kupiec_test(violations, level, n_total):
    """Kupiec unconditional coverage test."""
    n_violations = np.sum(violations)
    p_hat = n_violations / n_total if n_total > 0 else 0
    p = 1 - level
    if p_hat == 0 or p_hat == 1:
        return {"lr": np.nan, "p_value": np.nan, "reject": np.nan}
    lr = -2 * (n_total * np.log(p) + n_violations * np.log(p_hat / p + 1e-10) +
                (n_total - n_violations) * np.log((1 - p_hat) / (1 - p) + 1e-10))
    # Simplified: LR = -2 * ln(H0/H1)
    n = n_total
    x = n_violations
    if x * (n - x) == 0:
        return {"lr": np.nan, "p_value": np.nan, "reject": np.nan}
    try:
        lr_stat = -2 * np.log((1 - p)**(n - x) * p**x / ((1 - p_hat)**(n - x) * p_hat**x))
        p_value = 1 - stats.chi2.cdf(lr_stat, 1)
    except Exception:
        lr_stat = np.nan
        p_value = np.nan
    return {"lr": lr_stat, "p_value": p_value, "reject": p_value < 0.05 if not np.isnan(p_value) else np.nan}


def run_full_analysis():
    """Run the complete analysis and save results."""
    print("=" * 70)
    print("MEXICAN FINANCIAL DATA ANALYSIS")
    print("=" * 70)

    # Step 1: Download data
    print("\n--- Downloading data ---")
    import yfinance as yf
    price_data = download_data()

    all_results = {}

    for name, prices in price_data.items():
        print(f"\n{'='*70}")
        print(f"  Processing: {name}")
        print(f"{'='*70}")

        # Compute returns
        returns = compute_daily_returns(prices)
        ret = returns.values
        n = len(ret)
        split = int(n * TRAIN_FRAC)

        # Compute daily realized variance from squared returns
        daily_rv = ret ** 2
        daily_log_rv = np.log(np.maximum(daily_rv, 1e-20))
        daily_log_rv = daily_log_rv[np.isfinite(daily_log_rv)]

        # Align returns with log RV
        n_common = min(len(ret), len(daily_log_rv))

        print(f"  Total days: {n}, Train: {split}, Test: {n - split}")

        # Step 2: Fit SF-Harris
        print(f"  Fitting SF-Harris...")
        try:
            sf_result = fit_sf_harris(pd.Series(daily_log_rv))
            sf_result["returns_test"] = ret[split:split + sf_result["n_test"]]
            sf_result["rv_test"] = daily_rv[split:split + sf_result["n_test"]]
            print(f"    alpha={float(np.mean(sf_result['alpha'])):.4f}, P(stay)={sf_result['p_stay']:.4f}")
        except Exception as e:
            print(f"    SF-Harris failed: {e}")
            continue

        # Step 3: Fit GARCH
        print(f"  Fitting GARCH(1,1)...")
        try:
            garch_result = fit_garch(pd.Series(ret))
            garch_result["returns_test"] = ret[split:]
            garch_result["n_test"] = len(ret) - split
        except Exception as e:
            print(f"    GARCH failed: {e}")
            garch_result = None

        # Step 4: Coverage comparison
        print(f"  Computing coverage...")
        n_test = sf_result["n_test"]
        test_ret = sf_result["returns_test"][:n_test]

        # SF-Harris coverage
        sf_cov = compute_coverage(test_ret, sf_result["sim_ret"][:, :n_test], PROB_LEVELS)
        sf_aad = np.mean([abs(sf_cov[p] - p * 100) for p in PROB_LEVELS])

        # GARCH coverage
        if garch_result is not None:
            garch_cov = compute_coverage(test_ret, garch_result["sim_ret"][:, :n_test], PROB_LEVELS)
            garch_aad = np.mean([abs(garch_cov[p] - p * 100) for p in PROB_LEVELS])
        else:
            garch_cov = {p: np.nan for p in PROB_LEVELS}
            garch_aad = np.nan

        # Historical simulation (iid bootstrap from training)
        rng_boot = np.random.default_rng(999)
        train_ret = ret[:split]
        sim_hist = rng_boot.choice(train_ret, size=(N_SIM, n_test), replace=True)
        hist_cov = compute_coverage(test_ret, sim_hist, PROB_LEVELS)
        hist_aad = np.mean([abs(hist_cov[p] - p * 100) for p in PROB_LEVELS])

        # Normal with constant variance
        rng_norm = np.random.default_rng(888)
        sim_norm = rng_norm.normal(0, np.std(train_ret), size=(N_SIM, n_test))
        norm_cov = compute_coverage(test_ret, sim_norm, PROB_LEVELS)
        norm_aad = np.mean([abs(norm_cov[p] - p * 100) for p in PROB_LEVELS])

        # Step 5: VaR and ES
        print(f"  Computing VaR/ES...")
        sf_var = compute_var_es(test_ret, sf_result["sim_ret"][:, :n_test])
        if garch_result is not None:
            garch_var = compute_var_es(test_ret, garch_result["sim_ret"][:, :n_test])
        else:
            garch_var = {}
        hist_var = compute_var_es(test_ret, sim_hist)
        norm_var = compute_var_es(test_ret, sim_norm)

        # Kupiec tests
        for level in [0.95, 0.99]:
            for model_name, var_dict in [("SF-Harris", sf_var), ("GARCH", garch_var),
                                          ("Historical", hist_var), ("Normal", norm_var)]:
                if f"violations_{level}" in var_dict:
                    kupiec = kupiec_test(var_dict[f"violations_{level}"], level, n_test)
                    var_dict[f"kupiec_{level}"] = kupiec

        # Step 6: Descriptive statistics
        desc = {
            "mean": np.mean(ret),
            "std": np.std(ret),
            "skew": stats.skew(ret),
            "kurt": stats.kurtosis(ret),
            "min": np.min(ret),
            "max": np.max(ret),
            "n": n,
        }

        # Save results
        all_results[name] = {
            "sf_harris": {
                "coverage": sf_cov,
                "aad": sf_aad,
                "alpha": float(np.mean(sf_result["alpha"])),
                "p_stay": sf_result["p_stay"],
                "var_es": sf_var,
                "sim_ret": sf_result["sim_ret"][:, :n_test],
            },
            "garch": {
                "coverage": garch_cov,
                "aad": garch_aad,
                "var_es": garch_var if garch_result else None,
                "params": garch_result["params"] if garch_result else None,
            } if garch_result else None,
            "historical": {
                "coverage": hist_cov,
                "aad": hist_aad,
                "var_es": hist_var,
            },
            "normal": {
                "coverage": norm_cov,
                "aad": norm_aad,
                "var_es": norm_var,
            },
            "descriptive": desc,
            "returns": ret,
            "test_returns": test_ret,
            "n_total": n,
            "n_test": n_test,
            "split": split,
            "dates": returns.index,
        }

        print(f"  Results:")
        print(f"    SF-Harris AAD: {sf_aad:.1f}pp")
        print(f"    GARCH AAD:     {garch_aad:.1f}pp" if garch_result else "    GARCH AAD:     N/A")
        print(f"    Historical AAD: {hist_aad:.1f}pp")
        print(f"    Normal AAD:    {norm_aad:.1f}pp")

    # Save results
    print(f"\n--- Saving results to {RESULTS_PATH} ---")
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "wb") as f:
        pickle.dump(all_results, f)
    print(f"  Saved {len(all_results)} assets")

    return all_results


if __name__ == "__main__":
    results = run_full_analysis()