"""Compare three bar types for SF-Harris volatility modeling:

1. Calendar-time bars (Anzarut): fixed 15-min intervals
2. Dollar bars (Lopez de Prado): fixed dollar volume per bar
3. Volume bars (Barardehi proxy): fixed share volume per bar
   (True trade-time requires tick data; volume bars approximate it)

For each bar type:
- Compute realized variance
- Fit double exponential ACF
- Compare ACF shapes, R^2, and periodicity patterns

The key question: does bar type choice eliminate the U-shape
and improve SF-Harris model fit?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.optimize import minimize

DATA_PATH = Path(r"C:\Users\angve\OneDrive\Desktop\Servicio\Libros\SF-Harris\IBM.txt")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"


def load_ibm_data(start_date="2012-01-01", end_date="2014-12-31"):
    """Load IBM 1-minute OHLCV data."""
    df = pd.read_csv(
        DATA_PATH, header=None,
        names=["date", "time", "open", "high", "low", "close", "volume"],
    )
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="%m/%d/%Y %H:%M")
    df = df.drop(columns=["date", "time"]).set_index("datetime")
    df = df.loc[start_date:end_date]
    df["dollar_volume"] = df["close"] * df["volume"]
    return df


# ---------------------------------------------------------------------------
# Bar construction
# ---------------------------------------------------------------------------
def build_calendar_bars(df, freq="15min"):
    """Standard calendar-time bars (Anzarut's approach)."""
    close = df["close"].resample(freq).last().dropna()
    returns = np.log(close).diff().dropna()
    return returns


def build_dollar_bars(df, dollar_threshold=None):
    """Dollar bars (Lopez de Prado): accumulate dollar volume until threshold.

    Each bar contains all 1-min intervals until cumulative dollar volume
    reaches the threshold. Then compute return from first to last close.
    """
    if dollar_threshold is None:
        # Set threshold so average ~15 min worth of dollar volume
        avg_dv_per_15min = df["dollar_volume"].resample("15min").sum().mean()
        dollar_threshold = avg_dv_per_15min

    returns = []
    cum_dv = 0.0
    first_close = None

    for idx, row in df.iterrows():
        if first_close is None:
            first_close = row["close"]
            bar_start_time = idx

        cum_dv += row["dollar_volume"]

        if cum_dv >= dollar_threshold:
            ret = np.log(row["close"] / first_close)
            returns.append({
                "datetime": idx,
                "return": ret,
                "bar_duration_min": (idx - bar_start_time).total_seconds() / 60,
                "dollar_volume": cum_dv,
            })
            cum_dv = 0.0
            first_close = None

    result = pd.DataFrame(returns)
    if len(result) > 0:
        result = result.set_index("datetime")
    return result


def build_volume_bars(df, volume_threshold=None):
    """Volume bars: accumulate share volume until threshold.

    Approximate to Barardehi's trade-time bars (fixed N trades per bar).
    With tick data this would be exact; with 1-min bars, volume is a proxy.
    """
    if volume_threshold is None:
        avg_vol_per_15min = df["volume"].resample("15min").sum().mean()
        volume_threshold = avg_vol_per_15min

    returns = []
    cum_vol = 0
    first_close = None

    for idx, row in df.iterrows():
        if first_close is None:
            first_close = row["close"]
            bar_start_time = idx

        cum_vol += row["volume"]

        if cum_vol >= volume_threshold:
            ret = np.log(row["close"] / first_close)
            returns.append({
                "datetime": idx,
                "return": ret,
                "bar_duration_min": (idx - bar_start_time).total_seconds() / 60,
                "share_volume": cum_vol,
            })
            cum_vol = 0
            first_close = None

    result = pd.DataFrame(returns)
    if len(result) > 0:
        result = result.set_index("datetime")
    return result


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------
def compute_daily_rv(returns_series):
    """Compute daily realized variance from returns."""
    if isinstance(returns_series, pd.DataFrame):
        ret = returns_series["return"]
    else:
        ret = returns_series
    rv = ret.groupby(ret.index.date).apply(lambda x: np.sum(x**2))
    rv.index = pd.to_datetime(rv.index)
    return rv


def compute_intraday_rv_profile(returns_series, n_bins=26):
    """Compute average RV by time-of-day to check for U-shape.

    Returns ratio of RV at each time to the daily average RV.
    A flat profile = no periodicity; U-shape = periodicity artifact.
    """
    if isinstance(returns_series, pd.DataFrame):
        ret = returns_series["return"]
    else:
        ret = returns_series

    # For calendar bars: group by time of day
    if hasattr(ret.index, "time"):
        rv_by_time = ret.groupby(ret.index.time).apply(lambda x: np.mean(x**2))
        rv_mean = np.mean(ret**2)
        profile = rv_by_time / rv_mean if rv_mean > 0 else rv_by_time
        return profile
    else:
        return None


def compute_dollar_bar_intraday_profile(dollar_bars):
    """For dollar bars, check if there's still a U-shape in volatility.

    Since dollar bars have irregular timestamps, we bin by hour
    and check if volatility varies by time of day.
    """
    if len(dollar_bars) == 0:
        return None

    ret = dollar_bars["return"]
    # Bin by hour
    hour = ret.index.hour
    rv_by_hour = ret.groupby(hour).apply(lambda x: np.mean(x**2))
    rv_mean = np.mean(ret**2)
    profile = rv_by_hour / rv_mean if rv_mean > 0 else rv_by_hour
    return profile


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


def fit_single_exponential(lags, acf):
    """Fit r(t) = exp(-alpha*t)."""
    valid = acf > 0
    if np.sum(valid) < 2:
        return {"alpha": 0.1, "r_squared": 0}
    slope = np.polyfit(lags[valid], np.log(acf[valid]), 1)[0]
    alpha = -slope
    fitted = np.exp(-alpha * lags[valid])
    ss_res = np.sum((acf[valid] - fitted)**2)
    ss_tot = np.sum((acf[valid] - np.mean(acf[valid]))**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return {"alpha": alpha, "r_squared": r_squared}


def fit_double_exponential(lags, acf):
    """Fit r(t) = w1*exp(-a1*t) + w2*exp(-a2*t)."""
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


def analyze_bar_type(name, log_rv, n_daily_obs=None):
    """Full analysis pipeline for one bar type."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    acf = acf_analysis(log_rv, max_lag=20)
    lags = np.arange(1, 21)

    # Stats
    print(f"  Observations: {len(log_rv)}")
    print(f"  Mean: {np.mean(log_rv):.4f}, Std: {np.std(log_rv):.4f}")
    print(f"  Skew: {pd.Series(log_rv).skew():.4f}, Kurt: {pd.Series(log_rv).kurtosis():.4f}")

    # ACF
    print(f"\n  ACF:")
    for h in [1, 2, 5, 10, 20]:
        if h < len(acf):
            print(f"    rho({h:2d}) = {acf[h]:.4f}")

    # Single exponential fit
    single = fit_single_exponential(lags, acf[1:21])
    print(f"\n  Single exp: alpha={single['alpha']:.3f}, R^2={single['r_squared']:.4f}")

    # Double exponential fit
    double = fit_double_exponential(lags, acf[1:21])
    print(f"  Double exp: w1={double['w1']:.3f}, a1={double['alpha1']:.3f}, "
          f"w2={double['w2']:.3f}, a2={double['alpha2']:.4f}")
    print(f"              R^2={double['r_squared']:.4f}")

    # Half-lives
    if double['alpha1'] > 0:
        hl1 = np.log(2) / double['alpha1']
        print(f"              Half-life fast: {hl1:.2f} days")
    if double['alpha2'] > 0:
        hl2 = np.log(2) / double['alpha2']
        print(f"              Half-life slow: {hl2:.1f} days")

    # U-shape diagnostic
    if n_daily_obs is not None:
        print(f"\n  Avg observations/day: {n_daily_obs:.1f}")

    return {
        "name": name,
        "n_obs": len(log_rv),
        "mean": np.mean(log_rv),
        "std": np.std(log_rv),
        "skew": pd.Series(log_rv).skew(),
        "kurt": pd.Series(log_rv).kurtosis(),
        "acf": acf,
        "single": single,
        "double": double,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Bar Type Comparison for SF-Harris Volatility Modeling")
    print("Calendar-time (Anzarut) vs Dollar bars (Lopez de Prado)")
    print("vs Volume bars (Barardehi trade-time proxy)")
    print("=" * 70)

    # Load data
    df = load_ibm_data()
    print(f"Loaded {len(df)} 1-minute bars")

    # --- 1. Calendar-time bars (Anzarut) ---
    cal_returns = build_calendar_bars(df)
    cal_rv = compute_daily_rv(cal_returns)
    cal_log_rv = np.log(cal_rv.values)

    # Count daily observations
    n_days = cal_returns.groupby(cal_returns.index.date).count()
    avg_obs_per_day = n_days.mean()

    cal_results = analyze_bar_type(
        "Calendar-time bars (15-min, Anzarut)",
        cal_log_rv,
        n_daily_obs=avg_obs_per_day
    )

    # U-shape check for calendar bars
    cal_profile = compute_intraday_rv_profile(cal_returns)
    if cal_profile is not None:
        morning = cal_profile.iloc[:5].mean()
        midday = cal_profile.iloc[10:20].mean() if len(cal_profile) > 20 else cal_profile.iloc[len(cal_profile)//2].mean()
        afternoon = cal_profile.iloc[-5:].mean()
        print(f"\n  Intraday profile (calendar bars):")
        print(f"    Morning avg: {morning:.2f}x")
        print(f"    Midday avg:  {midday:.2f}x")
        print(f"    Close avg:   {afternoon:.2f}x")
        print(f"    U-shape ratio (morning/midday): {morning/midday:.2f}")

    # --- 2. Dollar bars (Lopez de Prado) ---
    print(f"\n  Building dollar bars...")
    dollar_bars = build_dollar_bars(df)
    print(f"  Created {len(dollar_bars)} dollar bars")

    if len(dollar_bars) > 0:
        dollar_rv = compute_daily_rv(dollar_bars)
        dollar_log_rv = np.log(dollar_rv.values)

        dollar_results = analyze_bar_type(
            "Dollar bars (Lopez de Prado)",
            dollar_log_rv,
        )

        # U-shape check for dollar bars
        dollar_profile = compute_dollar_bar_intraday_profile(dollar_bars)
        if dollar_profile is not None:
            morning_h = dollar_profile.loc[10] if 10 in dollar_profile.index else 0
            midday_h = dollar_profile.loc[12] if 12 in dollar_profile.index else 0
            close_h = dollar_profile.loc[15] if 15 in dollar_profile.index else 0
            print(f"\n  Intraday profile (dollar bars, by hour):")
            for h in sorted(dollar_profile.index):
                print(f"    Hour {h}: {dollar_profile[h]:.2f}x")
            if midday_h > 0:
                print(f"    U-shape ratio (hour10/hour12): {morning_h/midday_h:.2f}")

        # Bar duration statistics
        print(f"\n  Dollar bar duration stats:")
        print(f"    Mean: {dollar_bars['bar_duration_min'].mean():.1f} min")
        print(f"    Std:  {dollar_bars['bar_duration_min'].std():.1f} min")
        print(f"    Min:  {dollar_bars['bar_duration_min'].min():.1f} min")
        print(f"    Max:  {dollar_bars['bar_duration_min'].max():.1f} min")

    # --- 3. Volume bars (Barardehi trade-time proxy) ---
    print(f"\n  Building volume bars...")
    vol_bars = build_volume_bars(df)
    print(f"  Created {len(vol_bars)} volume bars")

    if len(vol_bars) > 0:
        vol_rv = compute_daily_rv(vol_bars)
        vol_log_rv = np.log(vol_rv.values)

        vol_results = analyze_bar_type(
            "Volume bars (Barardehi trade-time proxy)",
            vol_log_rv,
        )

        # U-shape check
        if hasattr(vol_bars.index, "hour"):
            hour = vol_bars.index.hour
            ret = vol_bars["return"]
            rv_by_hour = ret.groupby(hour).apply(lambda x: np.mean(x**2))
            rv_mean = np.mean(ret**2)
            vol_profile = rv_by_hour / rv_mean if rv_mean > 0 else rv_by_hour
            print(f"\n  Intraday profile (volume bars, by hour):")
            for h in sorted(vol_profile.index):
                print(f"    Hour {h}: {vol_profile[h]:.2f}x")

        print(f"\n  Volume bar duration stats:")
        print(f"    Mean: {vol_bars['bar_duration_min'].mean():.1f} min")
        print(f"    Std:  {vol_bars['bar_duration_min'].std():.1f} min")

    # --- Comparison table ---
    print(f"\n{'='*70}")
    print(f"  COMPARISON TABLE")
    print(f"{'='*70}")

    all_results = [cal_results]
    if len(dollar_bars) > 0:
        all_results.append(dollar_results)
    if len(vol_bars) > 0:
        all_results.append(vol_results)

    print(f"\n  {'Bar type':<35} {'N days':>7} {'Std':>7} {'Skew':>7} {'Kurt':>7} "
          f"{'R^2(1exp)':>9} {'R^2(2exp)':>9}")
    print(f"  {'-'*85}")
    for r in all_results:
        print(f"  {r['name']:<35} {r['n_obs']:>7} {r['std']:>7.3f} {r['skew']:>7.3f} "
              f"{r['kurt']:>7.3f} {r['single']['r_squared']:>9.4f} {r['double']['r_squared']:>9.4f}")

    print(f"\n  ACF at key lags:")
    print(f"  {'Bar type':<35} {'rho(1)':>7} {'rho(2)':>7} {'rho(5)':>7} {'rho(10)':>7} {'rho(20)':>7}")
    print(f"  {'-'*75}")
    for r in all_results:
        acf = r["acf"]
        print(f"  {r['name']:<35} {acf[1]:>7.4f} {acf[2]:>7.4f} {acf[5]:>7.4f} "
              f"{acf[10]:>7.4f} {acf[20]:>7.4f}")

    # Double exponential parameters
    print(f"\n  Double exponential ACF parameters:")
    print(f"  {'Bar type':<35} {'w1':>6} {'a1':>7} {'w2':>6} {'a2':>8}")
    print(f"  {'-'*70}")
    for r in all_results:
        d = r["double"]
        print(f"  {r['name']:<35} {d['w1']:>6.3f} {d['alpha1']:>7.3f} "
              f"{d['w2']:>6.3f} {d['alpha2']:>8.4f}")

    # Key takeaways
    print(f"\n{'='*70}")
    print(f"  KEY TAKEAWAYS")
    print(f"{'='*70}")
    print(f"\n  1. U-shape: Calendar bars show strong U-shape (7.7x at open).")
    print(f"     Dollar/volume bars should reduce this by sampling")
    print(f"     proportionally to activity rather than calendar time.")
    print(f"\n  2. R^2 measures how well a double exponential ACF model")
    print(f"     fits the autocorrelation SHAPE of log-realized variance.")
    print(f"     It is NOT prediction accuracy for returns or volatility.")
    print(f"\n  3. Lower kurtosis in log-RV = closer to Gaussian =")
    print(f"     better fit for GIG/log-normal marginal distributions.")

    # Save results
    summary = {}
    for r in all_results:
        key = r["name"].replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
        summary[key] = {
            "std": r["std"], "skew": r["skew"], "kurt": r["kurt"],
            "single_r2": r["single"]["r_squared"],
            "double_r2": r["double"]["r_squared"],
            "double_w1": r["double"]["w1"],
            "double_a1": r["double"]["alpha1"],
            "double_w2": r["double"]["w2"],
            "double_a2": r["double"]["alpha2"],
        }
    summary_df = pd.DataFrame(summary).T
    summary_df.to_csv(OUTPUT_DIR / "bar_comparison_results.csv")
    print(f"\n  Results saved to {OUTPUT_DIR / 'bar_comparison_results.csv'}")