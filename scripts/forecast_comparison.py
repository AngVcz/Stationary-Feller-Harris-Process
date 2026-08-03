"""Full SF-Harris model fit and forecast comparison: dollar bars vs calendar bars.

For each bar type:
1. Build bars from IBM 1-minute data
2. Compute daily log-realized variance
3. Fit marginal distribution (log-normal)
4. Fit ACF model (double exponential = mixture SF-Harris)
5. Generate out-of-sample forecasts using rolling window
6. Compare forecast accuracy (RMSE, MAE, QL, log-score)

Forecast model: optimal linear predictor under stationarity.
  E[log(RV_{t+h}) | log(RV_t)] = mu + r(h) * (log(RV_t) - mu)
where r(h) = w1*exp(-a1*h) + w2*exp(-a2*h) is the mixture ACF.

Also benchmarks:
  - Naive: forecast = mu (historical mean)
  - Random walk: forecast = log(RV_t) (yesterday's value)
  - AR(1): forecast = mu + phi*(log(RV_t) - mu), phi = ACF(1)
  - HAR (Corsi): forecast from heterogeneous autoregressive model
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.optimize import minimize

DATA_PATH = Path(r"C:\Users\angve\OneDrive\Desktop\Servicio\Libros\SF-Harris\IBM.txt")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------------------
# Data loading and bar construction
# ---------------------------------------------------------------------------
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


def build_calendar_bars(df, freq="15min"):
    close = df["close"].resample(freq).last().dropna()
    returns = np.log(close).diff().dropna()
    return returns


def build_dollar_bars(df, dollar_threshold=None):
    if dollar_threshold is None:
        avg_dv_per_15min = df["dollar_volume"].resample("15min").sum().mean()
        dollar_threshold = avg_dv_per_15min

    returns = []
    cum_dv = 0.0
    first_close = None

    for idx, row in df.iterrows():
        if first_close is None:
            first_close = row["close"]
        cum_dv += row["dollar_volume"]

        if cum_dv >= dollar_threshold:
            ret = np.log(row["close"] / first_close)
            returns.append({"datetime": idx, "return": ret})
            cum_dv = 0.0
            first_close = None

    result = pd.DataFrame(returns)
    if len(result) > 0:
        result = result.set_index("datetime")
    return result


# ---------------------------------------------------------------------------
# RV and analysis functions
# ---------------------------------------------------------------------------
def compute_daily_rv(returns_series):
    if isinstance(returns_series, pd.DataFrame):
        ret = returns_series["return"]
    else:
        ret = returns_series
    rv = ret.groupby(ret.index.date).apply(lambda x: np.sum(x**2))
    rv.index = pd.to_datetime(rv.index)
    return rv


def acf_analysis(x, max_lag=40):
    n = len(x)
    acf_values = np.zeros(max_lag + 1)
    acf_values[0] = 1.0
    mean = np.mean(x)
    var = np.var(x)
    if var == 0:
        return acf_values
    for h in range(1, max_lag + 1):
        acf_values[h] = np.mean((x[:-h] - mean) * (x[h:] - mean)) / var
    return acf_values


def fit_double_exponential(lags, acf):
    def neg_log_lik(params):
        log_w1, log_alpha1, log_alpha2 = params
        w1 = 1.0 / (1.0 + np.exp(-log_w1))
        w2 = 1.0 - w1
        alpha1 = np.exp(log_alpha1)
        alpha2 = np.exp(log_alpha2)
        fitted = w1 * np.exp(-alpha1 * lags) + w2 * np.exp(-alpha2 * lags)
        weights = 1.0 / (1 + lags)
        return np.sum(weights * (acf - fitted)**2)

    best_result = None
    best_nll = np.inf
    for x0 in [
        [0.0, np.log(1.0), np.log(0.1)],
        [1.0, np.log(2.0), np.log(0.05)],
        [-1.0, np.log(0.5), np.log(0.01)],
        [0.5, np.log(3.0), np.log(0.02)],
    ]:
        try:
            result = minimize(neg_log_lik, x0=x0, method="L-BFGS-B",
                            bounds=[(-5, 5), (-5, 5), (-5, 5)],
                            options={"maxiter": 1000})
            if result.fun < best_nll:
                best_nll = result.fun
                best_result = result
        except Exception:
            continue

    if best_result is None:
        return {"w1": 0.5, "alpha1": 1.0, "w2": 0.5, "alpha2": 0.1, "r_squared": 0}

    w1 = 1.0 / (1.0 + np.exp(-best_result.x[0]))
    w2 = 1.0 - w1
    alpha1 = np.exp(best_result.x[1])
    alpha2 = np.exp(best_result.x[2])

    fitted = w1 * np.exp(-alpha1 * lags) + w2 * np.exp(-alpha2 * lags)
    valid = acf > 0
    ss_res = np.sum((acf[valid] - fitted[valid])**2)
    ss_tot = np.sum((acf[valid] - np.mean(acf[valid]))**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return {"w1": w1, "alpha1": alpha1, "w2": w2, "alpha2": alpha2, "r_squared": r_squared}


# ---------------------------------------------------------------------------
# Forecasting models
# ---------------------------------------------------------------------------
def forecast_sf_harris(log_rv, acf_params, h=1):
    """Mixture SF-Harris forecast: mu + r(h) * (x_t - mu)."""
    mu = np.mean(log_rv[:-h])
    x_t = log_rv[-h - 1] if h <= len(log_rv) - 1 else log_rv[-1]
    w1, a1 = acf_params["w1"], acf_params["alpha1"]
    w2, a2 = acf_params["w2"], acf_params["alpha2"]
    r_h = w1 * np.exp(-a1 * h) + w2 * np.exp(-a2 * h)
    return mu + r_h * (x_t - mu)


def forecast_naive(log_rv, h=1):
    """Historical mean forecast."""
    return np.mean(log_rv[:-h])


def forecast_rw(log_rv, h=1):
    """Random walk: use last observation."""
    return log_rv[-1]


def forecast_ar1(log_rv, h=1):
    """AR(1) forecast: mu + phi^h * (x_t - mu), phi = ACF(1)."""
    mu = np.mean(log_rv[:-h])
    acf = acf_analysis(log_rv[:-h], max_lag=1)
    phi = acf[1]
    x_t = log_rv[-1]
    return mu + phi**h * (x_t - mu)


def forecast_har(log_rv, h=1):
    """HAR (Corsi 2009) forecast using daily, weekly, monthly averages.

    HAR-RV model: RV_t = beta0 + beta_d*RV_{t-1} + beta_w*RV_{t-5:t-1} + beta_m*RV_{t-22:t-1} + eps
    We fit OLS and use for h-step ahead.
    """
    n = len(log_rv)
    if n < 25:
        return np.mean(log_rv)

    # Build HAR regressors
    daily = log_rv[:-1]
    weekly = np.array([np.mean(log_rv[max(0, i-5):i]) for i in range(1, n)])
    monthly = np.array([np.mean(log_rv[max(0, i-22):i]) for i in range(1, n)])
    y = log_rv[1:]

    # OLS
    X = np.column_stack([np.ones(len(daily)), daily, weekly, monthly])
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.mean(log_rv)

    # Forecast: use last available values
    x_d = log_rv[-1]
    x_w = np.mean(log_rv[-5:])
    x_m = np.mean(log_rv[-22:])
    pred = beta[0] + beta[1] * x_d + beta[2] * x_w + beta[3] * x_m
    return pred


# ---------------------------------------------------------------------------
# Forecast evaluation metrics
# ---------------------------------------------------------------------------
def ql_loss(actual, forecast):
    """Quasilikelihood loss (Patton 2011) for volatility forecasts.
    QL = actual/forecast - log(actual/forecast) - 1
    Lower is better. Insensitive to noise in RV.
    """
    ratio = actual / forecast
    ratio = np.clip(ratio, 1e-10, 1e10)
    return np.mean(ratio - np.log(ratio) - 1)


def log_score(actual, forecast, sigma=1.0):
    """Log score under Gaussian prediction: -0.5*log(2*pi*sigma^2) - 0.5*(actual-forecast)^2/sigma^2.
    Higher is better.
    """
    return np.mean(-0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * ((actual - forecast) / sigma)**2)


def rolling_forecast_evaluation(log_rv, window_size=500, h=1):
    """Rolling window out-of-sample forecast evaluation.

    For each time t from window_size to len(log_rv)-h:
    - Fit models on log_rv[t-window_size:t]
    - Forecast log_rv[t+h-1] (h steps ahead)
    - Record error

    Returns dict of {model_name: {forecasts, actuals, rmse, mae, ql, logscore}}
    """
    n = len(log_rv)
    forecasts = {
        "sf_harris": [], "ar1": [], "har": [],
        "naive": [], "rw": [],
    }
    actuals = []

    for t in range(window_size, n - h + 1):
        train = log_rv[t - window_size:t]
        actual = log_rv[t + h - 1]
        actuals.append(actual)

        # Fit ACF on training window
        acf = acf_analysis(train, max_lag=20)
        lags = np.arange(1, 21)
        acf_params = fit_double_exponential(lags, acf[1:21])

        # SF-Harris mixture forecast
        mu = np.mean(train)
        x_last = train[-1]
        w1, a1 = acf_params["w1"], acf_params["alpha1"]
        w2, a2 = acf_params["w2"], acf_params["alpha2"]
        r_h = w1 * np.exp(-a1 * h) + w2 * np.exp(-a2 * h)
        forecasts["sf_harris"].append(mu + r_h * (x_last - mu))

        # AR(1) forecast
        phi = acf[1]
        forecasts["ar1"].append(mu + phi**h * (x_last - mu))

        # HAR forecast
        har_pred = forecast_har(train, h=h)
        forecasts["har"].append(har_pred)

        # Naive
        forecasts["naive"].append(mu)

        # Random walk
        forecasts["rw"].append(x_last)

    actuals = np.array(actuals)
    results = {}

    for name, fcasts in forecasts.items():
        fcasts = np.array(fcasts)
        errors = actuals - fcasts

        rmse = np.sqrt(np.mean(errors**2))
        mae = np.mean(np.abs(errors))

        # QL loss on exp scale (volatility, not log-vol)
        rv_actual = np.exp(actuals)
        rv_forecast = np.exp(fcasts)
        ql = ql_loss(rv_actual, rv_forecast)

        # Log score (Gaussian, sigma estimated from residuals)
        sigma_resid = np.std(errors)
        ls = log_score(actuals, fcasts, sigma=sigma_resid)

        results[name] = {
            "rmse": rmse, "mae": mae, "ql": ql, "logscore": ls,
            "n_forecasts": len(fcasts),
        }

    return results


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def fit_and_evaluate(name, log_rv, window_size=500):
    """Full SF-Harris fit + forecast evaluation for one bar type."""
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    # --- In-sample fit ---
    print(f"\n  In-sample statistics:")
    print(f"    N = {len(log_rv)}")
    print(f"    Mean = {np.mean(log_rv):.4f}")
    print(f"    Std  = {np.std(log_rv):.4f}")
    print(f"    Skew = {pd.Series(log_rv).skew():.4f}")
    print(f"    Kurt = {pd.Series(log_rv).kurtosis():.4f}")

    # Fit log-normal marginal
    mu_ln = np.mean(log_rv)
    sigma_ln = np.std(log_rv)
    print(f"\n  Log-normal marginal: mu={mu_ln:.4f}, sigma={sigma_ln:.4f}")

    # Fit ACF model
    acf = acf_analysis(log_rv, max_lag=20)
    lags = np.arange(1, 21)
    acf_params = fit_double_exponential(lags, acf[1:21])

    print(f"\n  ACF (empirical):")
    for h in [1, 2, 5, 10, 20]:
        print(f"    rho({h:2d}) = {acf[h]:.4f}")

    w1, a1 = acf_params["w1"], acf_params["alpha1"]
    w2, a2 = acf_params["w2"], acf_params["alpha2"]

    print(f"\n  Mixture SF-Harris ACF: r(t) = {w1:.3f}*exp(-{a1:.3f}*t) + {w2:.3f}*exp(-{a2:.4f}*t)")
    print(f"    R^2 = {acf_params['r_squared']:.4f}")
    print(f"    Fast half-life: {np.log(2)/a1:.2f} days (weight {w1:.1%})")
    print(f"    Slow half-life: {np.log(2)/a2:.1f} days (weight {w2:.1%})")

    # 1-step ahead theoretical ACF values
    fitted_acf = [w1 * np.exp(-a1 * h) + w2 * np.exp(-a2 * h) for h in [1, 2, 5, 10, 20]]
    print(f"  Fitted ACF:  {', '.join(f'{v:.4f}' for v in fitted_acf)}")

    # --- Out-of-sample forecast evaluation ---
    print(f"\n  Rolling window forecast evaluation (window={window_size}, h=1):")
    forecast_results = rolling_forecast_evaluation(log_rv, window_size=window_size, h=1)

    print(f"\n  {'Model':<20} {'RMSE':>8} {'MAE':>8} {'QL':>10} {'LogScore':>10}")
    print(f"  {'-'*58}")
    for mname, metrics in forecast_results.items():
        print(f"  {mname:<20} {metrics['rmse']:>8.4f} {metrics['mae']:>8.4f} "
              f"{metrics['ql']:>10.4f} {metrics['logscore']:>10.4f}")

    # Also evaluate h=5 forecast
    print(f"\n  h=5 (weekly) forecast evaluation:")
    forecast_results_5 = rolling_forecast_evaluation(log_rv, window_size=window_size, h=5)
    print(f"\n  {'Model':<20} {'RMSE':>8} {'MAE':>8} {'QL':>10} {'LogScore':>10}")
    print(f"  {'-'*58}")
    for mname, metrics in forecast_results_5.items():
        print(f"  {mname:<20} {metrics['rmse']:>8.4f} {metrics['mae']:>8.4f} "
              f"{metrics['ql']:>10.4f} {metrics['logscore']:>10.4f}")

    return {
        "name": name,
        "n_obs": len(log_rv),
        "mean": np.mean(log_rv),
        "std": np.std(log_rv),
        "skew": pd.Series(log_rv).skew(),
        "kurtosis": pd.Series(log_rv).kurtosis(),
        "acf_params": acf_params,
        "forecast_h1": forecast_results,
        "forecast_h5": forecast_results_5,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("SF-Harris Forecast Comparison: Dollar Bars vs Calendar Bars")
    print("=" * 70)

    # Load data
    df = load_ibm_data()
    print(f"Loaded {len(df)} 1-minute bars")

    # Build calendar bars
    cal_returns = build_calendar_bars(df)
    cal_rv = compute_daily_rv(cal_returns)
    cal_log_rv = np.log(cal_rv.values)

    # Build dollar bars
    print("Building dollar bars...")
    dollar_bars = build_dollar_bars(df)
    print(f"Created {len(dollar_bars)} dollar bars")

    dollar_rv = compute_daily_rv(dollar_bars)
    dollar_log_rv = np.log(dollar_rv.values)

    # Fit and evaluate both
    cal_results = fit_and_evaluate("Calendar-time bars (15-min, Anzarut)", cal_log_rv)
    dollar_results = fit_and_evaluate("Dollar bars (Lopez de Prado)", dollar_log_rv)

    # --- Cross-comparison ---
    print(f"\n{'='*70}")
    print(f"  CROSS-COMPARISON")
    print(f"{'='*70}")

    # In-sample comparison
    print(f"\n  In-sample fit:")
    print(f"  {'Bar type':<35} {'Std':>7} {'Kurt':>7} {'R^2(ACF)':>10}")
    print(f"  {'-'*62}")
    for r in [cal_results, dollar_results]:
        print(f"  {r['name']:<35} {r['std']:>7.3f} {r['kurtosis']:>7.3f} "
              f"{r['acf_params']['r_squared']:>10.4f}")

    # Forecast comparison h=1
    print(f"\n  h=1 forecast comparison:")
    print(f"  {'Bar type':<35} {'Model':<15} {'RMSE':>8} {'MAE':>8} {'QL':>10}")
    print(f"  {'-'*78}")
    for r in [cal_results, dollar_results]:
        for mname in ["sf_harris", "ar1", "har", "naive", "rw"]:
            m = r["forecast_h1"][mname]
            bar_short = "Calendar" if "Calendar" in r["name"] else "Dollar"
            print(f"  {bar_short:<35} {mname:<15} {m['rmse']:>8.4f} {m['mae']:>8.4f} {m['ql']:>10.4f}")

    # Forecast comparison h=5
    print(f"\n  h=5 forecast comparison:")
    print(f"  {'Bar type':<35} {'Model':<15} {'RMSE':>8} {'MAE':>8} {'QL':>10}")
    print(f"  {'-'*78}")
    for r in [cal_results, dollar_results]:
        for mname in ["sf_harris", "ar1", "har", "naive", "rw"]:
            m = r["forecast_h5"][mname]
            bar_short = "Calendar" if "Calendar" in r["name"] else "Dollar"
            print(f"  {bar_short:<35} {mname:<15} {m['rmse']:>8.4f} {m['mae']:>8.4f} {m['ql']:>10.4f}")

    # Key takeaways
    print(f"\n{'='*70}")
    print(f"  KEY TAKEAWAYS")
    print(f"{'='*70}")

    # Compare best model per bar type
    for r in [cal_results, dollar_results]:
        best_h1 = min(r["forecast_h1"].items(), key=lambda x: x[1]["rmse"])
        best_h5 = min(r["forecast_h5"].items(), key=lambda x: x[1]["rmse"])
        bar = "Calendar" if "Calendar" in r["name"] else "Dollar"
        print(f"\n  {bar} bars:")
        print(f"    Best h=1 model: {best_h1[0]} (RMSE={best_h1[1]['rmse']:.4f})")
        print(f"    Best h=5 model: {best_h5[0]} (RMSE={best_h5[1]['rmse']:.4f})")

    # Compare SF-Harris across bar types
    sf_cal = cal_results["forecast_h1"]["sf_harris"]
    sf_dol = dollar_results["forecast_h1"]["sf_harris"]
    print(f"\n  SF-Harris model across bar types (h=1):")
    print(f"    Calendar: RMSE={sf_cal['rmse']:.4f}, QL={sf_cal['ql']:.4f}")
    print(f"    Dollar:   RMSE={sf_dol['rmse']:.4f}, QL={sf_dol['ql']:.4f}")
    rmse_ratio = sf_dol["rmse"] / sf_cal["rmse"]
    print(f"    Dollar/Calendar RMSE ratio: {rmse_ratio:.3f}")
    if rmse_ratio < 1:
        print(f"    -> Dollar bars produce {1-rmse_ratio:.1%} better SF-Harris forecasts")
    else:
        print(f"    -> Calendar bars produce {1-1/rmse_ratio:.1%} better SF-Harris forecasts")

    sf_cal5 = cal_results["forecast_h5"]["sf_harris"]
    sf_dol5 = dollar_results["forecast_h5"]["sf_harris"]
    print(f"\n  SF-Harris model across bar types (h=5):")
    print(f"    Calendar: RMSE={sf_cal5['rmse']:.4f}, QL={sf_cal5['ql']:.4f}")
    print(f"    Dollar:   RMSE={sf_dol5['rmse']:.4f}, QL={sf_dol5['ql']:.4f}")

    # Save results
    summary = {
        "calendar": {
            "std": cal_results["std"], "kurtosis": cal_results["kurtosis"],
            "acf_r2": cal_results["acf_params"]["r_squared"],
            "sf_harris_rmse_h1": cal_results["forecast_h1"]["sf_harris"]["rmse"],
            "sf_harris_ql_h1": cal_results["forecast_h1"]["sf_harris"]["ql"],
            "har_rmse_h1": cal_results["forecast_h1"]["har"]["rmse"],
            "sf_harris_rmse_h5": cal_results["forecast_h5"]["sf_harris"]["rmse"],
        },
        "dollar": {
            "std": dollar_results["std"], "kurtosis": dollar_results["kurtosis"],
            "acf_r2": dollar_results["acf_params"]["r_squared"],
            "sf_harris_rmse_h1": dollar_results["forecast_h1"]["sf_harris"]["rmse"],
            "sf_harris_ql_h1": dollar_results["forecast_h1"]["sf_harris"]["ql"],
            "har_rmse_h1": dollar_results["forecast_h1"]["har"]["rmse"],
            "sf_harris_rmse_h5": dollar_results["forecast_h5"]["sf_harris"]["rmse"],
        },
    }
    summary_df = pd.DataFrame(summary).T
    summary_df.to_csv(OUTPUT_DIR / "forecast_comparison_results.csv")
    print(f"\n  Results saved to {OUTPUT_DIR / 'forecast_comparison_results.csv'}")