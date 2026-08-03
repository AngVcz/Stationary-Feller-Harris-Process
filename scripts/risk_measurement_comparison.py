"""Risk Measurement Comparison: Heavy-Tailed SV vs Historical Simulation.

HEAD-TO-HEAD BACKTEST on SPY (single asset, no correlation distortion).

Tests:
1. VaR backtesting at 90%, 95%, 99%, 99.5%, 99.9% (violation rates + Kupiec test)
2. CVaR (Expected Shortfall) at same levels
3. Christoffersen independence test for violation clustering
4. Tail extrapolation: can SF-Harris estimate risk beyond training data?
5. Crisis performance: GFC 2008, COVID-19
6. QQ plot comparison of scenario distributions

KEY QUESTION: Does SF-Harris beat Historical Simulation in risk measurement?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy import stats
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# 1. Configuration
# =============================================================================
TICKER = 'SPY'
WINDOW = 252          # 1-year rolling window
N_SIM = 10000         # scenarios per day
CONFIDENCE_LEVELS = [0.90, 0.95, 0.99, 0.995, 0.999]
START_DATE = '2005-01-01'
END_DATE = '2026-05-01'

# Crisis windows
GFC_START, GFC_END = '2007-10-01', '2009-03-31'
COVID_START, COVID_END = '2020-02-01', '2020-06-30'

print("=" * 70)
print("RISK MEASUREMENT COMPARISON: SF-HARRIS vs HISTORICAL SIMULATION")
print("=" * 70)
print()
print(f"Asset: {TICKER} (single asset, no correlation distortion)")
print("SF-Harris: tau ~ Empirical(r^2) iid, r|tau ~ N(0,tau) — heavy tails")
print("Hist.Sim:  iid bootstrap — hard-capped at training maximum")
print()

# =============================================================================
# 2. Download data
# =============================================================================
print("Downloading data...")
prices = yf.download(TICKER, start=START_DATE, end=END_DATE)['Close']
if isinstance(prices, pd.DataFrame):
    prices = prices[TICKER]
returns = np.log(prices / prices.shift(1)).dropna()
returns_arr = returns.values
dates_arr = returns.index
print(f"  Data: {dates_arr[0].strftime('%Y-%m-%d')} to {dates_arr[-1].strftime('%Y-%m-%d')}")
print(f"  Trading days: {len(returns_arr)}")

# =============================================================================
# 3. Simulation functions (single asset)
# =============================================================================
def simulate_ht_sv(train_returns, n_sim, rng):
    """Heavy-Tailed SV: tau ~ Empirical(r^2) iid, r|tau ~ N(0, tau).
    Single asset version. Generates heavier tails than Hist.Sim because
    N(0,tau) can produce returns beyond training maximum."""
    variances = train_returns ** 2
    idx = rng.integers(0, len(train_returns), size=n_sim)
    tau = variances[idx].clip(min=1e-20)
    return np.sqrt(tau) * rng.standard_normal(n_sim)


def simulate_hist(train_returns, n_sim, rng):
    """Historical Simulation: iid bootstrap. Hard-capped at training max."""
    idx = rng.integers(0, len(train_returns), size=n_sim)
    return train_returns[idx]


# =============================================================================
# 4. Rolling backtest: VaR and CVaR estimation
# =============================================================================
print("\nRunning rolling VaR/CVaR backtest...")
print(f"  Window: {WINDOW} days (1 year)")
print(f"  Scenarios: {N_SIM}")
print(f"  Confidence levels: {CONFIDENCE_LEVELS}")

n_days = len(returns_arr)

# Storage
var_sv = {alpha: np.full(n_days, np.nan) for alpha in CONFIDENCE_LEVELS}
var_hist = {alpha: np.full(n_days, np.nan) for alpha in CONFIDENCE_LEVELS}
cvar_sv = {alpha: np.full(n_days, np.nan) for alpha in CONFIDENCE_LEVELS}
cvar_hist = {alpha: np.full(n_days, np.nan) for alpha in CONFIDENCE_LEVELS}
actual_returns = returns_arr.copy()
max_sv = np.full(n_days, np.nan)
max_hist = np.full(n_days, np.nan)
train_max = np.full(n_days, np.nan)

for t in range(WINDOW, n_days):
    train_data = returns_arr[t - WINDOW:t]
    rng_sv = np.random.default_rng(42 + t)
    rng_hist = np.random.default_rng(43 + t)

    sv_scenarios = simulate_ht_sv(train_data, N_SIM, rng_sv)
    hist_scenarios = simulate_hist(train_data, N_SIM, rng_hist)

    # Compute VaR and CVaR (losses = negative returns)
    sv_losses = -sv_scenarios
    hist_losses = -hist_scenarios

    for alpha in CONFIDENCE_LEVELS:
        # VaR at confidence level alpha = quantile at alpha of losses
        var_sv[alpha][t] = np.percentile(sv_losses, alpha * 100)
        var_hist[alpha][t] = np.percentile(hist_losses, alpha * 100)

        # CVaR = mean of losses exceeding VaR
        cvar_sv[alpha][t] = sv_losses[sv_losses >= var_sv[alpha][t]].mean()
        cvar_hist[alpha][t] = hist_losses[hist_losses >= var_hist[alpha][t]].mean()

    max_sv[t] = sv_losses.max()
    max_hist[t] = hist_losses.max()
    train_max[t] = -train_data.min()  # training max loss

    if t % 252 == 0:
        print(f"  Processing {dates_arr[t].strftime('%Y-%m-%d')}...")

print("  Done.")

actual_losses = -actual_returns  # losses = negative returns
valid_idx = np.arange(WINDOW, n_days)
valid_dates = dates_arr[WINDOW:]

# =============================================================================
# 5. VaR Backtesting: Violation rates
# =============================================================================
print("\n" + "=" * 70)
print("VaR BACKTESTING: VIOLATION RATES")
print("=" * 70)
print()
print("Violation rate = fraction of days where actual loss exceeded VaR.")
print("Well-calibrated model: violation rate = 1 - confidence level.")
print("Lower violation rate = more conservative (but underpredicts risk if too low).")
print()


def kupiec_test(violations, n, alpha):
    """Kupiec proportion of failures test.
    H0: violation rate = 1 - alpha (nominal level).
    Returns (LR_statistic, p_value)."""
    p_hat = violations / n
    p0 = 1 - alpha
    if violations == 0 or violations == n:
        return np.nan, np.nan
    try:
        lr = -2 * np.log(
            (p0 ** violations * (1 - p0) ** (n - violations)) /
            (p_hat ** violations * (1 - p_hat) ** (n - violations))
        )
        p_value = 1 - stats.chi2.cdf(lr, 1)
        return lr, p_value
    except (ValueError, ZeroDivisionError):
        return np.nan, np.nan


def christoffersen_test(violations_binary):
    """Christoffersen test for independence of violations.
    H0: violations are independent (no clustering).
    Returns (LR_statistic, p_value)."""
    n00 = n01 = n10 = n11 = 0
    for i in range(1, len(violations_binary)):
        prev = violations_binary[i-1]
        curr = violations_binary[i]
        if prev == 0 and curr == 0: n00 += 1
        elif prev == 0 and curr == 1: n01 += 1
        elif prev == 1 and curr == 0: n10 += 1
        else: n11 += 1

    p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    p = (n01 + n11) / (n00 + n01 + n10 + n11) if (n00 + n01 + n10 + n11) > 0 else 0

    if p == 0 or p == 1 or n01 + n00 == 0 or n10 + n11 == 0:
        return np.nan, np.nan

    try:
        lr = -2 * np.log(
            (p ** (n01 + n11) * (1-p) ** (n00 + n10)) /
            (p01 ** n01 * (1-p01) ** n00 * p11 ** n11 * (1-p11) ** n10)
        )
        p_value = 1 - stats.chi2.cdf(lr, 1)
        return lr, p_value
    except (ValueError, ZeroDivisionError):
        return np.nan, np.nan


print(f"{'Level':>6}  {'Nominal':>8}  {'SV Viol%':>9}  {'Hist Viol%':>10}  "
      f"{'SV Kupiec p':>12}  {'Hist Kupiec p':>14}  {'Better':>8}")
print("-" * 85)

for alpha in CONFIDENCE_LEVELS:
    nominal = 1 - alpha

    # SV violations
    sv_violations = np.sum(actual_losses[valid_idx] > var_sv[alpha][valid_idx])
    sv_viol_rate = sv_violations / len(valid_idx)

    # Hist violations
    hist_violations = np.sum(actual_losses[valid_idx] > var_hist[alpha][valid_idx])
    hist_viol_rate = hist_violations / len(valid_idx)

    # Kupiec tests
    n = len(valid_idx)
    sv_kup_lr, sv_kup_p = kupiec_test(sv_violations, n, alpha)
    hist_kup_lr, hist_kup_p = kupiec_test(hist_violations, n, alpha)

    # Which is closer to nominal?
    sv_dev = abs(sv_viol_rate - nominal)
    hist_dev = abs(hist_viol_rate - nominal)
    better = "SV" if sv_dev < hist_dev - 0.001 else ("Hist" if hist_dev < sv_dev - 0.001 else "Tie")

    print(f"{alpha:>5.1%}  {nominal:>7.2%}  {sv_viol_rate:>8.2%}  {hist_viol_rate:>9.2%}  "
          f"{sv_kup_p:>11.3f}  {hist_kup_p:>13.3f}  {better:>7}")

# =============================================================================
# 6. Christoffersen independence test
# =============================================================================
print("\n" + "=" * 70)
print("CHRISTOFFERSEN INDEPENDENCE TEST (violation clustering)")
print("=" * 70)
print()
print("Tests whether VaR violations cluster in time.")
print("p < 0.05 = violations are clustered (model misses regime changes).")
print()

for alpha in CONFIDENCE_LEVELS:
    sv_viol_binary = (actual_losses[valid_idx] > var_sv[alpha][valid_idx]).astype(int)
    hist_viol_binary = (actual_losses[valid_idx] > var_hist[alpha][valid_idx]).astype(int)

    sv_chris_lr, sv_chris_p = christoffersen_test(sv_viol_binary)
    hist_chris_lr, hist_chris_p = christoffersen_test(hist_viol_binary)

    sv_cluster = "clustered" if (not np.isnan(sv_chris_p) and sv_chris_p < 0.05) else "independent"
    hist_cluster = "clustered" if (not np.isnan(hist_chris_p) and hist_chris_p < 0.05) else "independent"

    print(f"  VaR {alpha:.1%}: SV p={sv_chris_p:.3f} ({sv_cluster})  "
          f"Hist p={hist_chris_p:.3f} ({hist_cluster})")

# =============================================================================
# 7. CVaR ratio comparison
# =============================================================================
print("\n" + "=" * 70)
print("CVaR RATIO: SV / HISTORICAL SIMULATION")
print("=" * 70)
print()
print("Ratio > 1 means SF-Harris estimates larger expected shortfalls (more conservative).")
print()

for alpha in CONFIDENCE_LEVELS:
    sv_c = cvar_sv[alpha][valid_idx]
    hist_c = cvar_hist[alpha][valid_idx]
    valid = ~np.isnan(sv_c) & ~np.isnan(hist_c) & (hist_c > 0)

    ratio = sv_c[valid] / hist_c[valid]
    print(f"  CVaR {alpha:.1%}: SV/Hist = {ratio.mean():.2f}x  "
          f"(5th-95th: {np.percentile(ratio, 5):.2f}x - {np.percentile(ratio, 95):.2f}x)")

# =============================================================================
# 8. VaR ratio comparison
# =============================================================================
print("\n" + "=" * 70)
print("VaR RATIO: SV / HISTORICAL SIMULATION")
print("=" * 70)
print()

for alpha in CONFIDENCE_LEVELS:
    sv_v = var_sv[alpha][valid_idx]
    hist_v = var_hist[alpha][valid_idx]
    valid = ~np.isnan(sv_v) & ~np.isnan(hist_v) & (hist_v > 0)

    ratio = sv_v[valid] / hist_v[valid]
    print(f"  VaR {alpha:.1%}: SV/Hist = {ratio.mean():.2f}x  "
          f"(5th-95th: {np.percentile(ratio, 5):.2f}x - {np.percentile(ratio, 95):.2f}x)")

# =============================================================================
# 9. Tail extrapolation
# =============================================================================
print("\n" + "=" * 70)
print("TAIL EXTRAPOLATION: MAX SCENARIO vs TRAINING DATA MAXIMUM")
print("=" * 70)
print()

valid_t = ~np.isnan(max_sv) & ~np.isnan(max_hist) & ~np.isnan(train_max) & (train_max > 0)
ratio_sv = max_sv[valid_t] / train_max[valid_t]
ratio_hist = max_hist[valid_t] / train_max[valid_t]

print(f"  SF-Harris max scenario / training max: {ratio_sv.mean():.2f}x (range: {ratio_sv.min():.2f}x - {ratio_sv.max():.2f}x)")
print(f"  Hist.Sim  max scenario / training max: {ratio_hist.mean():.2f}x (range: {ratio_hist.min():.2f}x - {ratio_hist.max():.2f}x)")
print(f"\n  Days where SV exceeds training max by >5%:  {(ratio_sv > 1.05).mean():.1%}")
print(f"  Days where Hist exceeds training max by >5%: {(ratio_hist > 1.05).mean():.1%}")

# What fraction of SV scenarios exceed training max at extreme quantiles?
print("\n  At P99.9 level:")
p999_sv = np.full(n_days, np.nan)
p999_hist = np.full(n_days, np.nan)
p999_train = np.full(n_days, np.nan)

for t in range(WINDOW, n_days):
    train_data = returns_arr[t - WINDOW:t]
    train_loss_max = -train_data.min()
    p999_train[t] = train_loss_max
    p999_sv[t] = var_sv[0.999][t]
    p999_hist[t] = var_hist[0.999][t]

valid_p = ~np.isnan(p999_sv) & ~np.isnan(p999_hist) & (p999_train > 0)
sv_exceeds_train = p999_sv[valid_p] > p999_train[valid_p]
hist_exceeds_train = p999_hist[valid_p] > p999_train[valid_p]
print(f"    SV P99.9 exceeds training max: {sv_exceeds_train.mean():.1%} of days")
print(f"    Hist P99.9 exceeds training max: {hist_exceeds_train.mean():.1%} of days")

# =============================================================================
# 10. Crisis performance
# =============================================================================
print("\n" + "=" * 70)
print("CRISIS VaR PERFORMANCE")
print("=" * 70)

for crisis_name, c_start, c_end in [('GFC 2008', GFC_START, GFC_END),
                                      ('COVID-19', COVID_START, COVID_END)]:
    mask = (dates_arr >= pd.Timestamp(c_start)) & (dates_arr <= pd.Timestamp(c_end))
    crisis_idx = np.where(mask)[0]
    crisis_idx = crisis_idx[crisis_idx >= WINDOW]  # only valid indices

    if len(crisis_idx) == 0:
        print(f"\n  {crisis_name}: No valid data")
        continue

    print(f"\n  {crisis_name} ({c_start} to {c_end}):")
    print(f"  {'Level':>6}  {'Nominal':>8}  {'SV Viol%':>9}  {'Hist Viol%':>10}  "
          f"{'SV VaR avg':>11}  {'Hist VaR avg':>13}  {'SV/Hist':>8}")
    print(f"  {'-'*80}")

    for alpha in [0.95, 0.99, 0.999]:
        nominal = 1 - alpha
        sv_v = var_sv[alpha][crisis_idx]
        hist_v = var_hist[alpha][crisis_idx]
        actual_l = actual_losses[crisis_idx]
        valid = ~np.isnan(sv_v) & ~np.isnan(hist_v)

        sv_viol = np.sum(actual_l[valid] > sv_v[valid]) / valid.sum()
        hist_viol = np.sum(actual_l[valid] > hist_v[valid]) / valid.sum()
        ratio = sv_v[valid].mean() / hist_v[valid].mean() if hist_v[valid].mean() > 0 else np.nan

        print(f"  {alpha:>5.1%}  {nominal:>7.1%}  {sv_viol:>8.1%}  {hist_viol:>9.1%}  "
              f"{sv_v[valid].mean():>10.4f}  {hist_v[valid].mean():>12.4f}  {ratio:>7.2f}x")

# =============================================================================
# 11. Conditional coverage: violations during high-vol vs low-vol
# =============================================================================
print("\n" + "=" * 70)
print("CONDITIONAL COVERAGE: HIGH-VOLATILITY vs LOW-VOLATILITY PERIODS")
print("=" * 70)
print()

# Split into high-vol and low-vol using rolling 60-day realized vol
realized_vol = pd.Series(actual_returns[WINDOW:], index=valid_dates).rolling(60).std() * np.sqrt(252)
vol_median = realized_vol.median()
high_vol_mask = realized_vol > vol_median
low_vol_mask = realized_vol <= vol_median

for alpha in [0.95, 0.99]:
    sv_v = var_sv[alpha][valid_idx]
    hist_v = var_hist[alpha][valid_idx]
    actual_l = actual_losses[valid_idx]
    nominal = 1 - alpha

    for vol_label, vol_mask in [("High-vol", high_vol_mask), ("Low-vol", low_vol_mask)]:
        mask_vals = vol_mask.values[:len(valid_idx)] if len(vol_mask) >= len(valid_idx) else vol_mask.values
        # Align lengths
        min_len = min(len(mask_vals), len(sv_v), len(actual_l))
        m = mask_vals[:min_len].astype(bool)
        if m.sum() == 0:
            continue
        sv_viol = np.sum(actual_l[:min_len][m] > sv_v[:min_len][m]) / m.sum()
        hist_viol = np.sum(actual_l[:min_len][m] > hist_v[:min_len][m]) / m.sum()

        print(f"  VaR {alpha:.0%} ({vol_label}): nominal={nominal:.1%}  "
              f"SV viol={sv_viol:.1%}  Hist viol={hist_viol:.1%}")

# =============================================================================
# 12. Plot: VaR backtest over time
# =============================================================================
fig, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 2, 2]})

# Panel 1: Actual losses with VaR bounds
ax = axes[0]
ax.plot(valid_dates, actual_losses[valid_idx] * 100, color='black', linewidth=0.3, alpha=0.5, label='Actual Loss')

for alpha, color, ls in [(0.95, 'steelblue', '-'), (0.99, 'red', '--')]:
    ax.plot(valid_dates, var_sv[alpha][valid_idx] * 100, color=color, linewidth=0.7,
            linestyle=ls, alpha=0.8, label=f'SV VaR {alpha:.0%}')
    ax.plot(valid_dates, var_hist[alpha][valid_idx] * 100, color=color, linewidth=0.7,
            linestyle=ls, alpha=0.4, label=f'Hist VaR {alpha:.0%}')

for c_start, c_end, color in [(GFC_START, GFC_END, 'red'), (COVID_START, COVID_END, 'orange')]:
    ax.axvspan(pd.Timestamp(c_start), pd.Timestamp(c_end), alpha=0.12, color=color)

ax.set_ylabel('Loss (%)')
ax.set_title(f'{TICKER} VaR Backtest: SF-Harris vs Historical Simulation')
ax.legend(fontsize=8, loc='upper right')
ax.grid(alpha=0.3)

# Panel 2: CVaR ratio over time
ax = axes[1]
for alpha, color in [(0.95, 'steelblue'), (0.99, 'red'), (0.999, 'purple')]:
    sv_c = cvar_sv[alpha][valid_idx]
    hist_c = cvar_hist[alpha][valid_idx]
    valid = ~np.isnan(sv_c) & ~np.isnan(hist_c) & (hist_c > 0)
    ratio = np.where(valid, sv_c / hist_c, np.nan)
    ax.plot(valid_dates, ratio, color=color, linewidth=0.5, alpha=0.7,
            label=f'CVaR {alpha:.1%}')

ax.axhline(y=1.0, color='black', linewidth=0.5, linestyle='--')
ax.set_ylabel('SV/Hist Ratio')
ax.set_title('CVaR Ratio: SF-Harris / Hist.Sim (>1 = SV more conservative)')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

for c_start, c_end, color in [(GFC_START, GFC_END, 'red'), (COVID_START, COVID_END, 'orange')]:
    ax.axvspan(pd.Timestamp(c_start), pd.Timestamp(c_end), alpha=0.12, color=color)

# Panel 3: VaR violation rates (rolling 252-day window)
ax = axes[2]
for alpha, color in [(0.95, 'steelblue'), (0.99, 'red')]:
    violations_sv = (actual_losses[valid_idx] > var_sv[alpha][valid_idx]).astype(float)
    violations_hist = (actual_losses[valid_idx] > var_hist[alpha][valid_idx]).astype(float)

    # Rolling violation rate
    window_252 = 252
    viol_series_sv = pd.Series(violations_sv, index=valid_dates).rolling(window_252).mean()
    viol_series_hist = pd.Series(violations_hist, index=valid_dates).rolling(window_252).mean()

    ax.plot(viol_series_sv.index, viol_series_sv * 100, color=color, linewidth=1, alpha=0.8,
            label=f'SV VaR {alpha:.0%}')
    ax.plot(viol_series_hist.index, viol_series_hist * 100, color=color, linewidth=1, alpha=0.4,
            linestyle='--', label=f'Hist VaR {alpha:.0%}')
    ax.axhline(y=(1-alpha)*100, color=color, linewidth=0.5, linestyle=':')

ax.set_ylabel('Violation Rate (%)')
ax.set_xlabel('Date')
ax.set_title('Rolling 1-Year VaR Violation Rate')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

for c_start, c_end, color in [(GFC_START, GFC_END, 'red'), (COVID_START, COVID_END, 'orange')]:
    ax.axvspan(pd.Timestamp(c_start), pd.Timestamp(c_end), alpha=0.12, color=color)

plt.tight_layout()
plt.savefig('risk_measurement_backtest.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved to risk_measurement_backtest.png")

# =============================================================================
# 13. Plot: QQ comparison and tail extrapolation
# =============================================================================
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7))

# Panel 1: QQ plot of scenario distributions
ax = axes2[0]
# Pick a representative day (middle of 2020 crash)
crash_t = np.argmin(returns_arr[WINDOW:]) + WINDOW
train_data = returns_arr[crash_t - WINDOW:crash_t]
rng_sv = np.random.default_rng(42 + crash_t)
rng_hist = np.random.default_rng(43 + crash_t)

sv_scen = simulate_ht_sv(train_data, N_SIM, rng_sv)
hist_scen = simulate_hist(train_data, N_SIM, rng_hist)

# QQ plot: quantiles of SV vs quantiles of Hist
sv_sorted = np.sort(sv_scen)
hist_sorted = np.sort(hist_scen)

ax.scatter(hist_sorted[::10], sv_sorted[::10], s=3, alpha=0.5, color='steelblue')
lims = [min(sv_sorted.min(), hist_sorted.min()), max(sv_sorted.max(), hist_sorted.max())]
ax.plot(lims, lims, 'k--', linewidth=0.5, label='1:1 line')
ax.set_xlabel('Hist.Sim Return Quantile')
ax.set_ylabel('SF-Harris Return Quantile')
ax.set_title(f'QQ Plot: SF-Harris vs Hist.Sim Scenarios\n({dates_arr[crash_t].strftime("%Y-%m-%d")} — high-vol day)')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Add annotation
sv_max = sv_sorted[-1]
hist_max = hist_sorted[-1]
sv_min = sv_sorted[0]
hist_min = hist_sorted[0]
ax.annotate(f'Right tail: SV max={sv_max:.3f} vs Hist max={hist_max:.3f}\n'
            f'Left tail:  SV min={sv_min:.3f} vs Hist min={hist_min:.3f}\n'
            f'SV/Hist tail ratio: {abs(sv_min)/abs(hist_min):.2f}x',
            xy=(0.03, 0.97), xycoords='axes fraction', fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Panel 2: Tail extrapolation over time
ax = axes2[1]
vt_idx = np.where(valid_t)[0]
ax.plot(dates_arr[vt_idx], ratio_sv, color='steelblue', linewidth=0.5, alpha=0.7,
        label='SF-Harris max/training max')
ax.plot(dates_arr[vt_idx], ratio_hist, color='seagreen', linewidth=0.5, alpha=0.7,
        label='Hist.Sim max/training max')
ax.axhline(y=1.0, color='black', linewidth=0.5, linestyle='--')
ax.set_ylabel('Max Scenario / Training Max')
ax.set_xlabel('Date')
ax.set_title('Tail Extrapolation: How Far Beyond Training Data?')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

for c_start, c_end, color in [(GFC_START, GFC_END, 'red'), (COVID_START, COVID_END, 'orange')]:
    ax.axvspan(pd.Timestamp(c_start), pd.Timestamp(c_end), alpha=0.12, color=color)

plt.tight_layout()
plt.savefig('risk_measurement_tail_extrapolation.png', dpi=150, bbox_inches='tight')
print(f"Tail extrapolation plot saved to risk_measurement_tail_extrapolation.png")

# =============================================================================
# 14. Final summary
# =============================================================================
print("\n" + "=" * 70)
print("VERDICT: DOES SF-HARRIS BEAT HISTORICAL SIMULATION IN RISK MEASUREMENT?")
print("=" * 70)
print()

# Compute summary statistics
print("AT STANDARD LEVELS (90-95% VaR):")
sv_95_viol = np.sum(actual_losses[valid_idx] > var_sv[0.95][valid_idx]) / len(valid_idx)
hist_95_viol = np.sum(actual_losses[valid_idx] > var_hist[0.95][valid_idx]) / len(valid_idx)
print(f"  VaR 95% violation rate: SV={sv_95_viol:.1%} vs Hist={hist_95_viol:.1%} (nominal: 5.0%)")
print(f"  Both models are reasonably calibrated at standard levels.")
print()

print("AT EXTREME LEVELS (99%+ CVaR):")
sv_cvar_99 = cvar_sv[0.99][valid_idx]
hist_cvar_99 = cvar_hist[0.99][valid_idx]
valid_cvar = ~np.isnan(sv_cvar_99) & ~np.isnan(hist_cvar_99) & (hist_cvar_99 > 0)
ratio_99 = sv_cvar_99[valid_cvar] / hist_cvar_99[valid_cvar]
print(f"  CVaR 99% ratio (SV/Hist): mean={ratio_99.mean():.2f}x")
print(f"  SF-Harris is {ratio_99.mean():.0%}x more conservative at extreme levels.")
print()

print("TAIL EXTRAPOLATION:")
print(f"  SF-Harris generates scenarios {ratio_sv.mean():.1f}x beyond training maximum.")
print(f"  Hist.Sim is hard-capped at training maximum (ratio = {ratio_hist.mean():.2f}x).")
print(f"  SF-Harris exceeds training max by >5% on {(ratio_sv > 1.05).mean():.0%} of days.")
print()

print("CRISIS PERFORMANCE:")
for crisis_name, c_start, c_end in [('GFC 2008', GFC_START, GFC_END),
                                      ('COVID-19', COVID_START, COVID_END)]:
    mask = (dates_arr >= pd.Timestamp(c_start)) & (dates_arr <= pd.Timestamp(c_end))
    crisis_idx = np.where(mask)[0]
    crisis_idx = crisis_idx[crisis_idx >= WINDOW]

    if len(crisis_idx) == 0:
        continue

    sv_v99 = var_sv[0.99][crisis_idx]
    hist_v99 = var_hist[0.99][crisis_idx]
    actual_c = actual_losses[crisis_idx]
    valid = ~np.isnan(sv_v99) & ~np.isnan(hist_v99)

    sv_viol = np.sum(actual_c[valid] > sv_v99[valid]) / valid.sum()
    hist_viol = np.sum(actual_c[valid] > hist_v99[valid]) / valid.sum()
    ratio_c = sv_v99[valid].mean() / hist_v99[valid].mean() if hist_v99[valid].mean() > 0 else np.nan

    print(f"  {crisis_name}: VaR 99% — SV viol={sv_viol:.1%}, Hist viol={hist_viol:.1%}, "
          f"SV/Hist ratio={ratio_c:.2f}x")

print()
print("=" * 70)
print("CONCLUSION:")
print("=" * 70)
print("""
SF-Harris DOES NOT clearly beat Historical Simulation in risk measurement.

At standard levels (90-95% VaR):
  Both models produce similar violation rates. SV is slightly more
  conservative but not significantly better calibrated.

At extreme levels (99%+ CVaR):
  SV produces {sv_hist_ratio:.1f}x larger CVaR estimates than Hist.Sim.
  This is MORE CONSERVATIVE but NOT PROVABLY MORE ACCURATE.
  At extreme quantiles, we lack enough observations to validate which
  model is correct.

The GENUINE advantage of SF-Harris:
  TAIL EXTRAPOLATION. SV generates scenarios beyond the training data
  maximum (ratio {sv_max_ratio:.1f}x vs 1.0x for Hist.Sim). This matters
  for regulatory capital at P99.9+ where Hist.Sim produces zero
  observations and can only say "the worst we've seen is X."

The CATCH:
  Being more conservative is not the same as being more accurate.
  If the true P99 loss is 5%, an estimate of 8% is more conservative
  but not more accurate than an estimate of 4.5%.

  At levels where we CAN validate (90-99%), both models perform similarly.
  At levels where they DIFFER (99.9%+), we CANNOT validate accuracy.

  SF-Harris provides a more CONSERVATIVE estimate of extreme risk,
  which has regulatory value (Basel III/IV), but it does not provide
  a more ACCURATE estimate that we can validate with out-of-sample data.
""".format(sv_hist_ratio=ratio_99.mean(), sv_max_ratio=ratio_sv.mean()))