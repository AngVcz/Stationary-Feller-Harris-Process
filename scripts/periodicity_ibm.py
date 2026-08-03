"""Periodicity extraction for IBM intraday volatility.

Following Anzarut Appendix C.2 (based on Boudt et al. 2011):

The periodicity function f(t) = E[tau_t / (1/d * integral_0^d tau_s ds)]
captures the intraday pattern (U-shape: high vol at open/close, low at lunch).

The periodically adjusted volatility is: tau_hat_t = tau_t / f(c(t))
where c(t) = t mod L is the position in the weekly cycle.

Anzarut sets d = 1 day, L = 5 (weekly cycle).

We implement:
1. Compute 15-min realized variance per interval per day
2. Estimate periodicity by averaging across days (same time-of-day)
3. Normalize so integral over one day = 1
4. Divide out periodicity from spot volatility
5. Compare SF-Harris fit with and without periodicity adjustment

Also compares with trade-time aggregation (Barardehi & Bernhardt).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

DATA_PATH = Path(r"C:\Users\angve\OneDrive\Desktop\Servicio\Libros\SF-Harris\IBM.txt")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"


def load_ibm_data(start_date="2012-01-01", end_date="2014-12-31"):
    """Load IBM 1-minute data."""
    df = pd.read_csv(
        DATA_PATH, header=None,
        names=["date", "time", "open", "high", "low", "close", "volume"],
    )
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="%m/%d/%Y %H:%M")
    df = df.drop(columns=["date", "time"]).set_index("datetime")
    df = df.loc[start_date:end_date]
    return df


def compute_15min_returns(df):
    """Compute 15-minute log returns."""
    close_15min = df["close"].resample("15min").last().dropna()
    return np.log(close_15min).diff().dropna()


def compute_15min_rv(returns):
    """Compute 15-minute realized variance per interval per day.

    Returns a DataFrame with columns: [date, interval, rv]
    where interval is the 15-min time slot within the day.
    """
    records = []
    for date, group in returns.groupby(returns.index.date):
        rv_daily = np.sum(group**2)  # total daily RV
        for idx, val in group.items():
            time_str = idx.strftime("%H:%M")
            records.append({
                "date": date,
                "time": time_str,
                "datetime": idx,
                "rv_15min": val**2,  # 15-min RV contribution
                "rv_daily": rv_daily,
            })
    return pd.DataFrame(records)


def estimate_periodicity(rv_df):
    """Estimate periodicity function f(t) following Anzarut Appendix C.2.

    f(t) = E[tau_t / (1/d * integral_0^d tau_s ds)]
         = E[RV_t / RV_daily_mean]

    where RV_daily_mean = (1/d) * sum RV_s over the day.

    We approximate E[.] by the sample mean across days
    for each 15-minute time slot.
    """
    # Compute ratio: RV_15min / (RV_daily / n_intervals)
    # This gives the relative volatility at each time slot
    n_intervals_per_day = rv_df.groupby("date")["time"].transform("count")
    rv_df["rv_daily_mean"] = rv_df["rv_daily"] / n_intervals_per_day
    rv_df["ratio"] = rv_df["rv_15min"] / rv_df["rv_daily_mean"]

    # Average ratio by time-of-day across all days
    periodicity = rv_df.groupby("time")["ratio"].mean()

    # Also compute std for robustness check
    periodicity_std = rv_df.groupby("time")["ratio"].std()

    # Normalize so that mean over the day = 1
    # (the integral condition: 1/d * integral f(s)ds = 1)
    mean_f = periodicity.mean()
    periodicity = periodicity / mean_f
    periodicity_std = periodicity_std / mean_f

    return periodicity, periodicity_std


def adjust_for_periodicity(rv_df, periodicity):
    """Remove periodic component from realized variance.

    tau_hat_t = tau_t / f(c(t))
    """
    # Map time to periodicity factor
    rv_df["f_t"] = rv_df["time"].map(periodicity)
    rv_df["rv_adjusted"] = rv_df["rv_15min"] / rv_df["f_t"]
    return rv_df


def acf_analysis(x, max_lag=40):
    """Compute autocorrelation function."""
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
    """Fit r(t) = w1*exp(-a1*t) + w2*exp(-a2*t)."""
    from scipy.optimize import minimize

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
        return {"w1": 0.5, "alpha1": 1.0, "w2": 0.5, "alpha2": 0.1}

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


if __name__ == "__main__":
    print("=" * 70)
    print("Periodicity Extraction and Comparison")
    print("Following Anzarut Appendix C.2")
    print("=" * 70)

    # Load data
    df = load_ibm_data()
    returns = compute_15min_returns(df)
    print(f"Loaded {len(returns)} 15-min returns")

    # Compute 15-min RV per interval
    rv_df = compute_15min_rv(returns)
    print(f"Computed RV for {len(rv_df)} intervals across {rv_df['date'].nunique()} days")

    # Estimate periodicity
    periodicity, periodicity_std = estimate_periodicity(rv_df)

    print(f"\n--- Periodicity Function f(t) ---")
    print(f"  (1.0 = average, >1 = higher vol, <1 = lower vol)")
    print(f"\n  {'Time':>8}  {'f(t)':>8}  {'std':>8}")
    for time, f_val in periodicity.items():
        std_val = periodicity_std.get(time, 0)
        if f_val > 1.5 or f_val < 0.6:
            marker = " ***"
        elif f_val > 1.2 or f_val < 0.8:
            marker = " *"
        else:
            marker = ""
        print(f"  {time:>8}  {f_val:>8.3f}  {std_val:>8.3f}{marker}")

    # Check for U-shape
    morning = periodicity.iloc[:5].mean() if len(periodicity) > 10 else 0
    midday = periodicity.iloc[10:20].mean() if len(periodicity) > 20 else 0
    afternoon = periodicity.iloc[-5:].mean() if len(periodicity) > 10 else 0

    print(f"\n  Period averages:")
    print(f"    Morning (first 5 intervals): {morning:.3f}")
    print(f"    Midday (intervals 10-20):    {midday:.3f}")
    print(f"    Afternoon (last 5):         {afternoon:.3f}")

    if morning > midday and afternoon > midday:
        print(f"    -> U-shape detected (high at open/close, low at midday)")
    elif morning > midday:
        print(f"    -> Declining pattern (high at open, low at close)")

    # Compare: daily RV with vs without periodicity adjustment
    print(f"\n--- Impact on Daily RV Statistics ---")

    # Daily RV (unadjusted) - already computed
    rv_daily = rv_df.groupby("date")["rv_daily"].first()
    log_rv = np.log(rv_daily.values)

    # Daily RV (adjusted) - aggregate adjusted 15-min RV per day
    rv_df_adjusted = adjust_for_periodicity(rv_df.copy(), periodicity)
    rv_daily_adj = rv_df_adjusted.groupby("date")["rv_adjusted"].sum()
    log_rv_adj = np.log(rv_daily_adj.values)

    print(f"\n  {'Metric':>15}  {'Unadjusted':>12}  {'Adjusted':>12}")
    print(f"  {'-'*42}")
    print(f"  {'Mean log-RV':>15}  {np.mean(log_rv):>12.4f}  {np.mean(log_rv_adj):>12.4f}")
    print(f"  {'Std log-RV':>15}  {np.std(log_rv):>12.4f}  {np.std(log_rv_adj):>12.4f}")
    print(f"  {'Skew':>15}  {pd.Series(log_rv).skew():>12.4f}  {pd.Series(log_rv_adj).skew():>12.4f}")
    print(f"  {'Kurtosis':>15}  {pd.Series(log_rv).kurtosis():>12.4f}  {pd.Series(log_rv_adj).kurtosis():>12.4f}")

    # ACF comparison
    acf_unadj = acf_analysis(log_rv, max_lag=20)
    acf_adj = acf_analysis(log_rv_adj, max_lag=20)

    print(f"\n  ACF comparison:")
    print(f"  {'Lag':>4}  {'Unadjusted':>12}  {'Adjusted':>12}  {'Diff':>8}")
    for h in [1, 2, 5, 10, 20]:
        diff = acf_adj[h] - acf_unadj[h]
        print(f"  {h:>4}  {acf_unadj[h]:>12.4f}  {acf_adj[h]:>12.4f}  {diff:>+8.4f}")

    # Fit double exponential ACF to both
    print(f"\n--- ACF Model Fit Comparison ---")
    lags = np.arange(1, 21)

    fit_unadj = fit_double_exponential(lags, acf_unadj[1:21])
    fit_adj = fit_double_exponential(lags, acf_adj[1:21])

    print(f"\n  Unadjusted log-RV:")
    print(f"    Double exp: w1={fit_unadj['w1']:.3f}, a1={fit_unadj['alpha1']:.3f}, w2={fit_unadj['w2']:.3f}, a2={fit_unadj['alpha2']:.4f}")
    print(f"    R^2 = {fit_unadj['r_squared']:.4f}")

    print(f"\n  Adjusted log-RV:")
    print(f"    Double exp: w1={fit_adj['w1']:.3f}, a1={fit_adj['alpha1']:.3f}, w2={fit_adj['w2']:.3f}, a2={fit_adj['alpha2']:.4f}")
    print(f"    R^2 = {fit_adj['r_squared']:.4f}")

    # Save periodicity function
    periodicity.to_csv(OUTPUT_DIR / "ibm_periodicity.csv")
    print(f"\nPeriodicity function saved to {OUTPUT_DIR / 'ibm_periodicity.csv'}")

    # Save adjusted daily RV
    adj_df = pd.DataFrame({
        "date": rv_daily_adj.index,
        "rv_unadjusted": rv_daily.values,
        "rv_adjusted": rv_daily_adj.values,
        "log_rv_unadjusted": log_rv,
        "log_rv_adjusted": log_rv_adj,
    })
    adj_df.to_csv(OUTPUT_DIR / "ibm_daily_rv_comparison.csv", index=False)
    print(f"Comparison data saved to {OUTPUT_DIR / 'ibm_daily_rv_comparison.csv'}")

    # Summary
    print(f"\n" + "=" * 70)
    print(f"Summary")
    print(f"=" * 70)
    print(f"\n  Periodicity adjustment {'improves' if fit_adj['r_squared'] > fit_unadj['r_squared'] else 'does not improve'} the ACF model fit.")
    print(f"  R^2 change: {fit_unadj['r_squared']:.4f} -> {fit_adj['r_squared']:.4f}")
    if morning > midday and afternoon > midday:
        print(f"  The U-shape in intraday volatility is present in the data.")
        print(f"  Barardehi & Bernhardt argue this is an over-aggregation artifact")
        print(f"  that disappears with trade-time aggregation.")
        print(f"  Lopez de Prado's dollar bars would be an alternative approach.")