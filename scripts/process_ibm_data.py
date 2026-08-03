"""Process IBM intraday data for stochastic volatility analysis.

Steps following Anzarut Section 4:
1. Load 1-minute OHLCV data
2. Compute 15-minute returns (close-to-close)
3. Compute realized variance from 15-minute returns
4. Detect jumps using bipower variation
5. Fit SF-Harris process to the realized variance series
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.special import kv as bessel_kv

from src.sf_harris.estimation import ndnj_estimate, mle_alpha_continuous
from src.sf_harris.process import SFHarrisProcess
from src.sf_harris.distributions import GIGQ

# ---------------------------------------------------------------------------
# 1. Load and clean data
# ---------------------------------------------------------------------------
DATA_PATH = Path(r"C:\Users\angve\OneDrive\Desktop\Servicio\Libros\SF-Harris\IBM.txt")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"


def load_ibm_data(
    start_date: str = "2012-01-01",
    end_date: str = "2014-12-31",
) -> pd.DataFrame:
    """Load IBM 1-minute data and filter to the specified date range.

    Anzarut used Jan 2012 - Dec 2014 for the empirical analysis.
    """
    print(f"Loading IBM data from {DATA_PATH}...")
    df = pd.read_csv(
        DATA_PATH,
        header=None,
        names=["date", "time", "open", "high", "low", "close", "volume"],
    )
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="%m/%d/%Y %H:%M")
    df = df.drop(columns=["date", "time"]).set_index("datetime")

    # Filter to date range
    df = df.loc[start_date:end_date]
    print(f"  Loaded {len(df)} minute bars from {df.index[0]} to {df.index[-1]}")
    return df


# ---------------------------------------------------------------------------
# 2. Compute 15-minute returns
# ---------------------------------------------------------------------------
def compute_returns_15min(df: pd.DataFrame) -> pd.Series:
    """Compute 15-minute log returns from 1-minute close prices.

    Groups minutes into 15-min buckets (9:30-9:44, 9:45-9:59, etc.)
    and computes log(close_last / close_first) for each bucket.
    """
    close = df["close"]

    # Resample to 15-minute intervals, taking the last close in each bucket
    close_15min = close.resample("15min").last()

    # Drop NaN (periods with no data)
    close_15min = close_15min.dropna()

    # Log returns
    log_returns = np.log(close_15min).diff().dropna()
    return log_returns


# ---------------------------------------------------------------------------
# 3. Realized variance and bipower variation
# ---------------------------------------------------------------------------
def realized_variance(returns: pd.Series) -> float:
    """Compute realized variance: sum of squared returns."""
    return float(np.sum(returns**2))


def bipower_variation(returns: pd.Series) -> float:
    """Compute bipower variation (Barndorff-Nielsen & Shephard).

    BPV = (pi/2) * sum(|r_t| * |r_{t-1}|)

    This estimates the continuous component of variance,
    robust to jumps.
    """
    abs_returns = np.abs(returns.values)
    n = len(abs_returns)
    bpv = (np.pi / 2) * np.sum(abs_returns[1:] * abs_returns[:-1])
    return float(bpv)


def detect_jumps(
    returns: pd.Series,
    alpha: float = 0.001,
) -> pd.Series:
    """Detect jump days using the ratio statistic.

    A jump is detected when the ratio RV/BPV exceeds a threshold,
    indicating that the realized variance has a significant jump component.

    Anzarut uses the top 0.1% of bipower variation values,
    but we implement a simpler z-score based approach.

    Returns a boolean series indicating jump intervals.
    """
    rv_daily = returns.groupby(returns.index.date).apply(
        lambda x: np.sum(x**2)
    )
    bpv_daily = returns.groupby(returns.index.date).apply(
        lambda x: (np.pi / 2) * np.sum(np.abs(x.values[1:]) * np.abs(x.values[:-1]))
        if len(x) > 1 else 0.0
    )

    # Jump detection: RV >> BPV indicates a jump
    # Use the ratio statistic
    ratio = rv_daily / bpv_daily.replace(0, np.nan)
    ratio = ratio.dropna()

    # Flag extreme ratios as jumps (top 0.1%)
    threshold = ratio.quantile(1 - alpha)
    jump_days = ratio[ratio > threshold].index

    return jump_days


# ---------------------------------------------------------------------------
# 4. Prepare SF-Harris input
# ---------------------------------------------------------------------------
def prepare_variance_series(
    returns: pd.Series,
    jump_days=None,
) -> pd.Series:
    """Prepare the variance series for SF-Harris fitting.

    Computes daily realized variance, optionally removing jump days.
    Returns a series of daily variance values.
    """
    # Daily realized variance
    rv_daily = returns.groupby(returns.index.date).apply(
        lambda x: np.sum(x**2)
    )
    rv_daily.index = pd.to_datetime(rv_daily.index)

    if jump_days is not None:
        # Remove jump days
        rv_daily = rv_daily[~rv_daily.index.isin(jump_days)]

    return rv_daily


# ---------------------------------------------------------------------------
# 5. Summary statistics
# ---------------------------------------------------------------------------
def summarize_data(returns: pd.Series, rv_daily: pd.Series) -> dict:
    """Compute summary statistics for the data."""
    stats = {
        "n_returns": len(returns),
        "n_days": len(rv_daily),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "skew_return": float(pd.Series(returns).skew()),
        "kurt_return": float(pd.Series(returns).kurtosis()),
        "mean_rv": float(np.mean(rv_daily)),
        "std_rv": float(np.std(rv_daily)),
        "skew_rv": float(pd.Series(rv_daily).skew()),
        "kurt_rv": float(pd.Series(rv_daily).kurtosis()),
        "min_rv": float(np.min(rv_daily)),
        "max_rv": float(np.max(rv_daily)),
    }
    return stats


# ---------------------------------------------------------------------------
# 6. Fit SF-Harris process to realized variance
# ---------------------------------------------------------------------------
def fit_sf_harris(log_rv: np.ndarray) -> dict:
    """Fit SF-Harris process to log-realized variance series.

    For continuous data like log-RV, every transition is a change
    (since the values are continuous). We use threshold-based
    change detection to identify "stays" vs "changes".

    A "stay" means the variance didn't change significantly,
    which corresponds to the SF-Harris stay probability e^{-alpha}.
    """
    diff_log_rv = np.diff(log_rv)
    n_total = len(diff_log_rv)

    # Threshold-based: if |diff| < threshold, it's a "stay"
    threshold = np.std(diff_log_rv) * 0.5
    n_stays = int(np.sum(np.abs(diff_log_rv) <= threshold))
    n_changes = n_total - n_stays

    print(f"  Threshold for 'stay': {threshold:.6f}")
    print(f"  Total transitions: {n_total}")
    print(f"  Stays (|diff| <= threshold): {n_stays} ({n_stays/n_total*100:.1f}%)")
    print(f"  Changes (|diff| > threshold): {n_changes} ({n_changes/n_total*100:.1f}%)")

    # MLE for alpha
    if n_stays > 0:
        alpha_mle = -np.log(n_stays / n_total)
        print(f"\n  MLE alpha (threshold-based): {alpha_mle:.4f}")
        print(f"  P(stay) = e^(-alpha) = {np.exp(-alpha_mle):.4f}")
        print(f"  P(change) = 1 - e^(-alpha) = {1 - np.exp(-alpha_mle):.4f}")
    else:
        alpha_mle = 50.0
        print("\n  No stays detected — alpha is very large")

    # Also try quantile-based thresholds
    results = {"threshold": threshold, "alpha_mle": alpha_mle}

    for q in [0.1, 0.25, 0.5]:
        thr = np.quantile(np.abs(diff_log_rv), q)
        ns = int(np.sum(np.abs(diff_log_rv) <= thr))
        nc = n_total - ns
        if ns > 0:
            a = -np.log(ns / n_total)
        else:
            a = 50.0
        results[f"q{int(q*100)}_threshold"] = thr
        results[f"q{int(q*100)}_alpha"] = a
        print(f"  Quantile {q:.0%} threshold: {thr:.6f}, stays={ns}, alpha={a:.4f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("IBM Intraday Data Processing")
    print("Following Anzarut Section 4 methodology")
    print("=" * 70)

    # Load data
    df = load_ibm_data(start_date="2012-01-01", end_date="2014-12-31")
    print()

    # Compute 15-minute returns
    returns = compute_returns_15min(df)
    print(f"15-minute returns: {len(returns)} observations")
    print(f"  Mean: {np.mean(returns):.8f}")
    print(f"  Std:  {np.std(returns):.8f}")
    print(f"  Skew: {pd.Series(returns).skew():.4f}")
    print(f"  Kurt: {pd.Series(returns).kurtosis():.4f}")
    print()

    # Compute daily realized variance
    rv_daily = prepare_variance_series(returns)
    print(f"Daily realized variance: {len(rv_daily)} days")
    print(f"  Mean: {np.mean(rv_daily):.8f}")
    print(f"  Std:  {np.std(rv_daily):.8f}")
    print(f"  Skew: {pd.Series(rv_daily).skew():.4f}")
    print(f"  Kurt: {pd.Series(rv_daily).kurtosis():.4f}")
    print(f"  Min:  {np.min(rv_daily):.8f}")
    print(f"  Max:  {np.max(rv_daily):.8f}")
    print()

    # Detect jumps
    jump_days = detect_jumps(returns, alpha=0.001)
    print(f"Detected {len(jump_days)} jump days (top 0.1%)")

    # Remove jumps and recompute
    rv_clean = prepare_variance_series(returns, jump_days=jump_days)
    print(f"Clean daily variance: {len(rv_clean)} days (after removing jumps)")
    print(f"  Mean: {np.mean(rv_clean):.8f}")
    print(f"  Std:  {np.std(rv_clean):.8f}")
    print()

    # Fit SF-Harris to the clean variance series
    print("=" * 70)
    print("SF-Harris Process Fitting")
    print("=" * 70)

    # Use log-variance for the SF-Harris process
    log_rv = np.log(rv_clean.values)
    print(f"Log realized variance stats:")
    print(f"  Mean: {np.mean(log_rv):.4f}")
    print(f"  Std:  {np.std(log_rv):.4f}")
    print(f"  Skew: {pd.Series(log_rv).skew():.4f}")
    print(f"  Kurt: {pd.Series(log_rv).kurtosis():.4f}")
    print()

    # Fit SF-Harris
    fit_results = fit_sf_harris(log_rv)
    print()

    # Save processed data
    OUTPUT_DIR.mkdir(exist_ok=True)
    result_df = pd.DataFrame({
        "date": rv_clean.index,
        "rv": rv_clean.values,
        "log_rv": log_rv,
    })
    result_df.to_csv(OUTPUT_DIR / "ibm_daily_rv.csv", index=False)
    print(f"Processed data saved to {OUTPUT_DIR / 'ibm_daily_rv.csv'}")

    # Summary statistics
    stats = summarize_data(returns, rv_clean)
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    for key, val in stats.items():
        print(f"  {key}: {val:.6f}" if isinstance(val, float) else f"  {key}: {val}")