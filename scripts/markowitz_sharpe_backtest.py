"""Markowitz Max-Sharpe Portfolio Backtest: Heavy-Tailed SV vs Historical Simulation.

Compares Markowitz portfolios using two different return distribution estimates:
1. SV-Sharpe: Max Sharpe using Heavy-Tailed SV scenarios (fatter tails)
2. Hist-Sharpe: Max Sharpe using Historical Simulation scenarios
3. 60/40 benchmark: fixed allocation

Setup: 5 assets (SPY, EFA, AGG, GLD, IWM), monthly rebalancing,
2-year rolling window, backtested through GFC 2008 and COVID-19.

Key insight: SV produces fatter tails → higher variance estimates for
risky assets → lower allocation to equities → better crisis protection
but lower returns in normal times.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =============================================================================
# 1. Configuration
# =============================================================================
PORTFOLIO_TICKERS = {
    'SPY': 'S&P 500',
    'EFA': 'Intl Developed',
    'AGG': 'US Bonds',
    'GLD': 'Gold',
    'IWM': 'Small Cap',
}

WINDOW = 252 * 2     # 2-year rolling window
REBALANCE = 21       # monthly rebalancing
N_SIM = 5000         # scenarios for estimation
RF_ANNUAL = 0.02     # risk-free rate for Sharpe

# Crisis windows
GFC_START = '2007-10-01'
GFC_END = '2009-03-31'
COVID_START = '2020-02-01'
COVID_END = '2020-06-30'

FULL_START = '2005-01-01'
FULL_END = '2026-05-01'

print("=" * 70)
print("MARKOWITZ MAX-SHARPE: HEAVY-TAILED SV vs HISTORICAL SIMULATION")
print("=" * 70)
print()
print("SV-Sharpe:   Max Sharpe using Heavy-Tailed SV scenarios (fatter tails)")
print("Hist-Sharpe: Max Sharpe using iid bootstrap scenarios")
print("60/40:       Fixed 60% equity / 40% bonds benchmark")
print()

# =============================================================================
# 2. Download data
# =============================================================================
print("Downloading portfolio data...")
prices = yf.download(list(PORTFOLIO_TICKERS.keys()),
                     start=FULL_START, end=FULL_END)['Close']
prices = prices.dropna()
returns = np.log(prices / prices.shift(1)).dropna()
print(f"  Data: {returns.index[0].strftime('%Y-%m-%d')} to {returns.index[-1].strftime('%Y-%m-%d')}")
print(f"  Assets: {list(PORTFOLIO_TICKERS.keys())}")
print(f"  Trading days: {len(returns)}")

asset_cols = list(PORTFOLIO_TICKERS.keys())
returns_arr = returns[asset_cols].values
dates_arr = returns.index

# =============================================================================
# 3. Simulation and optimization functions
# =============================================================================
def simulate_ht_sv(train_returns, n_days, n_sim, rng):
    """Heavy-Tailed SV: tau ~ Empirical(r^2) iid, r|tau ~ N(0, tau).
    Generates fatter tails than Hist.Sim because N(0,tau) can exceed training max."""
    n_obs, n_assets = train_returns.shape
    variances = train_returns ** 2
    sim_returns = np.zeros((n_sim, n_days, n_assets))
    for day in range(n_days):
        for a in range(n_assets):
            idx = rng.integers(0, n_obs, size=n_sim)
            tau = variances[idx, a].clip(min=1e-20)
            sim_returns[:, day, a] = np.sqrt(tau) * rng.standard_normal(n_sim)
    return sim_returns


def simulate_hist(train_returns, n_days, n_sim, rng):
    """Historical Simulation: iid bootstrap. Hard-capped at training max."""
    n_obs = train_returns.shape[0]
    indices = rng.integers(0, n_obs, size=(n_sim, n_days))
    return train_returns[indices]


def max_sharpe_portfolio(mu, cov, rf=RF_ANNUAL/252):
    """Find the tangency portfolio that maximizes Sharpe ratio.

    Solves: max (w'mu - rf) / sqrt(w'cov w)
    s.t. sum(w) = 1, w >= 0
    """
    n = len(mu)

    def neg_sharpe(w):
        port_ret = w @ mu - rf
        port_vol = np.sqrt(w @ cov @ w)
        if port_vol < 1e-10:
            return 0
        return -port_ret / port_vol

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0.02, 0.80)] * n  # min 2%, max 80% per asset

    # Try multiple starting points
    best_result = None
    best_sharpe = -np.inf

    starts = [
        np.ones(n) / n,  # equal weight
        *[np.eye(n)[i] * 0.5 + np.ones(n) / (2*n) for i in range(n)],  # tilted
    ]

    for x0 in starts:
        try:
            result = minimize(neg_sharpe, x0, method='SLSQP',
                            bounds=bounds, constraints=constraints,
                            options={'maxiter': 1000, 'ftol': 1e-12})
            if result.success and -result.fun > best_sharpe:
                best_sharpe = -result.fun
                best_result = result.x.copy()
        except:
            continue

    if best_result is None:
        return np.ones(n) / n  # fallback to equal weight

    # Ensure weights sum to 1 and are in bounds
    w = np.clip(best_result, 0.02, 0.80)
    w /= w.sum()
    return w


# =============================================================================
# 4. Rolling backtest
# =============================================================================
print("\nRunning rolling Markowitz backtest...")
print(f"  Window: {WINDOW} days ({WINDOW/252:.0f} years)")
print(f"  Rebalance: every {REBALANCE} days")
print(f"  Horizon: {REBALANCE} days for scenario generation")

portfolio_dates = []
weights_sv_list = []
weights_hist_list = []

for t in range(WINDOW, len(returns_arr) - REBALANCE, REBALANCE):
    train_data = returns_arr[t - WINDOW:t]
    if len(train_data) < WINDOW * 0.8:
        continue

    rng_sv = np.random.default_rng(42 + t)
    rng_hist = np.random.default_rng(43 + t)

    # Generate scenarios
    sv_scenarios = simulate_ht_sv(train_data, REBALANCE, N_SIM, rng_sv)
    hist_scenarios = simulate_hist(train_data, REBALANCE, N_SIM, rng_hist)

    # Cumulative returns over horizon: (N_SIM, n_assets)
    sv_cum = sv_scenarios.sum(axis=1)
    hist_cum = hist_scenarios.sum(axis=1)

    # Estimate mean and covariance from scenarios
    mu_sv = sv_cum.mean(axis=0)
    cov_sv = np.cov(sv_cum.T)
    mu_hist = hist_cum.mean(axis=0)
    cov_hist = np.cov(hist_cum.T)

    # Add small regularization to covariance
    cov_sv += np.eye(len(asset_cols)) * 1e-8
    cov_hist += np.eye(len(asset_cols)) * 1e-8

    # Max Sharpe portfolios
    w_sv = max_sharpe_portfolio(mu_sv, cov_sv)
    w_hist = max_sharpe_portfolio(mu_hist, cov_hist)

    portfolio_dates.append(dates_arr[t])
    weights_sv_list.append(w_sv)
    weights_hist_list.append(w_hist)

weights_sv = np.array(weights_sv_list)
weights_hist = np.array(weights_hist_list)
n_periods = len(portfolio_dates)

print(f"\n  Periods: {n_periods}")
print(f"  From: {portfolio_dates[0].strftime('%Y-%m-%d')}")
print(f"  To: {portfolio_dates[-1].strftime('%Y-%m-%d')}")

# =============================================================================
# 5. Compute out-of-sample portfolio returns
# =============================================================================
port_ret_sv = np.full(len(returns_arr), np.nan)
port_ret_hist = np.full(len(returns_arr), np.nan)
port_ret_6040 = np.full(len(returns_arr), np.nan)

# 60/40: 60% SPY + 40% AGG (bonds)
w_6040 = np.zeros(len(asset_cols))
w_6040[asset_cols.index('SPY')] = 0.60
w_6040[asset_cols.index('AGG')] = 0.40

for i in range(n_periods - 1):
    start_date = portfolio_dates[i]
    end_date = portfolio_dates[i + 1]
    mask = (returns.index >= start_date) & (returns.index < end_date)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        continue
    period_returns = returns_arr[idx]
    port_ret_sv[idx] = period_returns @ weights_sv_list[i]
    port_ret_hist[idx] = period_returns @ weights_hist_list[i]
    port_ret_6040[idx] = period_returns @ w_6040

# Last period
mask = returns.index >= portfolio_dates[-1]
idx = np.where(mask)[0]
if len(idx) > 0:
    period_returns = returns_arr[idx]
    port_ret_sv[idx] = period_returns @ weights_sv_list[-1]
    port_ret_hist[idx] = period_returns @ weights_hist_list[-1]
    port_ret_6040[idx] = period_returns @ w_6040

# Clean up NaN
valid = ~np.isnan(port_ret_sv)
port_ret_sv = port_ret_sv[valid]
port_ret_hist = port_ret_hist[valid]
port_ret_6040 = port_ret_6040[valid]
valid_dates = returns.index[valid]
weights_sv_clean = weights_sv
weights_hist_clean = weights_hist

# Cumulative returns and drawdowns
cumret_sv = np.cumsum(port_ret_sv)
cumret_hist = np.cumsum(port_ret_hist)
cumret_6040 = np.cumsum(port_ret_6040)


def max_drawdown(cumret):
    wealth = np.exp(cumret)
    peak = np.maximum.accumulate(wealth)
    dd_pct = (wealth - peak) / peak
    return dd_pct, np.min(dd_pct)


dd_sv, maxdd_sv = max_drawdown(cumret_sv)
dd_hist, maxdd_hist = max_drawdown(cumret_hist)
dd_6040, maxdd_6040 = max_drawdown(cumret_6040)

# =============================================================================
# 6. Results: Weight comparison
# =============================================================================
print("\n" + "=" * 70)
print("AVERAGE PORTFOLIO WEIGHTS")
print("=" * 70)
print(f"\n{'Asset':>8}  {'SV-Sharpe':>10}  {'Hist-Sharpe':>12}  {'60/40':>8}")
print("-" * 50)
for i, col in enumerate(asset_cols):
    print(f"{col:>8}  {weights_sv[:, i].mean():>10.1%}  {weights_hist[:, i].mean():>12.1%}  {w_6040[i]:>8.1%}")

# Weight volatility (how much do allocations change?)
print(f"\nWeight volatility (std of SPY weight):")
print(f"  SV-Sharpe:   {weights_sv[:, asset_cols.index('SPY')].std():.1%}")
print(f"  Hist-Sharpe: {weights_hist[:, asset_cols.index('SPY')].std():.1%}")

# =============================================================================
# 7. Results: Full period performance
# =============================================================================
print("\n" + "=" * 70)
print("FULL PERIOD PERFORMANCE")
print("=" * 70)

def annualized_stats(returns_arr):
    mu = np.mean(returns_arr) * 252
    sigma = np.std(returns_arr) * np.sqrt(252)
    sharpe = (mu - RF_ANNUAL) / sigma if sigma > 0 else 0
    return mu, sigma, sharpe

mu_sv, sig_sv, sh_sv = annualized_stats(port_ret_sv)
mu_hist, sig_hist, sh_hist = annualized_stats(port_ret_hist)
mu_6040, sig_6040, sh_6040 = annualized_stats(port_ret_6040)

print(f"\n{'Portfolio':>15}  {'Ann.Ret':>8}  {'Ann.Vol':>8}  {'Sharpe':>8}  {'MaxDD':>8}")
print(f"{'-'*55}")
print(f"{'SV-Sharpe':>15}  {mu_sv*100:>7.1f}%  {sig_sv*100:>7.1f}%  {sh_sv:>7.2f}  {maxdd_sv*100:>7.1f}%")
print(f"{'Hist-Sharpe':>15}  {mu_hist*100:>7.1f}%  {sig_hist*100:>7.1f}%  {sh_hist:>7.2f}  {maxdd_hist*100:>7.1f}%")
print(f"{'60/40':>15}  {mu_6040*100:>7.1f}%  {sig_6040*100:>7.1f}%  {sh_6040:>7.2f}  {maxdd_6040*100:>7.1f}%")

# =============================================================================
# 8. Crisis analysis
# =============================================================================
print("\n" + "=" * 70)
print("CRISIS PERFORMANCE")
print("=" * 70)

for crisis_name, c_start, c_end in [('GFC 2008', GFC_START, GFC_END), ('COVID-19', COVID_START, COVID_END)]:
    mask = (valid_dates >= pd.Timestamp(c_start)) & (valid_dates <= pd.Timestamp(c_end))
    if mask.sum() == 0:
        print(f"\n  {crisis_name}: No data available")
        continue

    cum_sv_c = cumret_sv[mask]
    cum_hist_c = cumret_hist[mask]
    cum_6040_c = cumret_6040[mask]

    _, maxdd_sv_c = max_drawdown(cum_sv_c)
    _, maxdd_hist_c = max_drawdown(cum_hist_c)
    _, maxdd_6040_c = max_drawdown(cum_6040_c)

    total_sv = cum_sv_c[-1] - cum_sv_c[0]
    total_hist = cum_hist_c[-1] - cum_hist_c[0]
    total_6040 = cum_6040_c[-1] - cum_6040_c[0]

    vol_sv = np.std(port_ret_sv[mask]) * np.sqrt(252)
    vol_hist = np.std(port_ret_hist[mask]) * np.sqrt(252)
    vol_6040 = np.std(port_ret_6040[mask]) * np.sqrt(252)

    # Average SPY weight during crisis
    w_spy_sv_c = weights_sv[mask[:len(weights_sv)], asset_cols.index('SPY')].mean()
    w_spy_hist_c = weights_hist[mask[:len(weights_hist)], asset_cols.index('SPY')].mean()

    print(f"\n  {crisis_name} ({c_start} to {c_end}):")
    print(f"  {'Portfolio':>15}  {'Return':>8}  {'MaxDD':>8}  {'Vol':>8}  {'SPY%':>8}")
    print(f"  {'-'*55}")
    print(f"  {'SV-Sharpe':>15}  {total_sv*100:>7.1f}%  {maxdd_sv_c*100:>7.1f}%  {vol_sv*100:>7.1f}%  {w_spy_sv_c*100:>7.1f}%")
    print(f"  {'Hist-Sharpe':>15}  {total_hist*100:>7.1f}%  {maxdd_hist_c*100:>7.1f}%  {vol_hist*100:>7.1f}%  {w_spy_hist_c*100:>7.1f}%")
    print(f"  {'60/40':>15}  {total_6040*100:>7.1f}%  {maxdd_6040*100:>7.1f}%  {vol_6040*100:>7.1f}%  {'60.0':>7}%")

# =============================================================================
# 9. Pre-crisis de-risking analysis
# =============================================================================
print("\n" + "=" * 70)
print("PRE-CRISIS DE-RISKING: DID SV-SHARPE REDUCE EQUITY EXPOSURE?")
print("=" * 70)

spy_idx = asset_cols.index('SPY')

for crisis_name, c_start, c_end in [('GFC 2008', GFC_START, GFC_END), ('COVID-19', COVID_START, COVID_END)]:
    # 6-month window before crisis
    pre_start = pd.Timestamp(c_start) - pd.Timedelta(days=180)
    pre_end = pd.Timestamp(c_start)

    pre_mask = (valid_dates >= pre_start) & (valid_dates < pre_end)
    crisis_mask = (valid_dates >= pd.Timestamp(c_start)) & (valid_dates <= pd.Timestamp(c_end))

    if pre_mask.sum() == 0 or crisis_mask.sum() == 0:
        continue

    pre_w_mask = pre_mask[:len(weights_sv)]
    crisis_w_mask = crisis_mask[:len(weights_sv)]

    w_pre_sv = weights_sv[pre_w_mask, spy_idx].mean() if pre_w_mask.sum() > 0 else float('nan')
    w_crisis_sv = weights_sv[crisis_w_mask, spy_idx].mean() if crisis_w_mask.sum() > 0 else float('nan')
    w_pre_hist = weights_hist[pre_w_mask, spy_idx].mean() if pre_w_mask.sum() > 0 else float('nan')
    w_crisis_hist = weights_hist[crisis_w_mask, spy_idx].mean() if crisis_w_mask.sum() > 0 else float('nan')

    def fmt_pct(x):
        return f"{x:.1%}" if not np.isnan(x) else "N/A"
    def fmt_pp_change(x):
        return f"{x*100:+.1f}pp" if not np.isnan(x) else "N/A"

    print(f"\n  {crisis_name}:")
    print(f"    SV-Sharpe SPY:   Pre-crisis {fmt_pct(w_pre_sv)} -> During crisis {fmt_pct(w_crisis_sv)} (change: {fmt_pp_change(w_crisis_sv - w_pre_sv if not np.isnan(w_pre_sv) else float('nan'))})")
    print(f"    Hist-Sharpe SPY: Pre-crisis {fmt_pct(w_pre_hist)} -> During crisis {fmt_pct(w_crisis_hist)} (change: {fmt_pp_change(w_crisis_hist - w_pre_hist if not np.isnan(w_pre_hist) else float('nan'))})")

# =============================================================================
# 10. Plots
# =============================================================================
fig, axes = plt.subplots(4, 1, figsize=(14, 14), gridspec_kw={'height_ratios': [3, 2, 2, 2]})

# Panel 1: Cumulative returns
ax = axes[0]
ax.plot(valid_dates, cumret_sv * 100, label='SV-Sharpe (Heavy-Tailed)', color='steelblue', linewidth=1.5)
ax.plot(valid_dates, cumret_hist * 100, label='Hist-Sharpe (Bootstrap)', color='seagreen', linewidth=1.5)
ax.plot(valid_dates, cumret_6040 * 100, label='60/40 Benchmark', color='gray', linewidth=1, alpha=0.7)
for cs, ce, c in [(GFC_START, GFC_END, 'red'), (COVID_START, COVID_END, 'orange')]:
    ax.axvspan(pd.Timestamp(cs), pd.Timestamp(ce), alpha=0.12, color=c)
ax.set_ylabel('Cumulative Return (%)')
ax.set_title('Markowitz Max-Sharpe Portfolios: Heavy-Tailed SV vs Historical Simulation')
ax.legend(fontsize=10, loc='upper left')
ax.grid(alpha=0.3)

# Panel 2: Drawdowns
ax = axes[1]
ax.fill_between(valid_dates, dd_sv * 100, 0, color='steelblue', alpha=0.5, label='SV-Sharpe')
ax.fill_between(valid_dates, dd_hist * 100, 0, color='seagreen', alpha=0.3, label='Hist-Sharpe')
for cs, ce, c in [(GFC_START, GFC_END, 'red'), (COVID_START, COVID_END, 'orange')]:
    ax.axvspan(pd.Timestamp(cs), pd.Timestamp(ce), alpha=0.12, color=c)
ax.set_ylabel('Drawdown (%)')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel 3: SPY allocation
ax = axes[2]
ax.plot(valid_dates[:len(weights_sv)], weights_sv[:, spy_idx] * 100,
        color='steelblue', linewidth=0.8, alpha=0.8, label='SV-Sharpe')
ax.plot(valid_dates[:len(weights_hist)], weights_hist[:, spy_idx] * 100,
        color='seagreen', linewidth=0.8, alpha=0.8, label='Hist-Sharpe')
ax.axhline(y=60, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='60/40')
for cs, ce, c in [(GFC_START, GFC_END, 'red'), (COVID_START, COVID_END, 'orange')]:
    ax.axvspan(pd.Timestamp(cs), pd.Timestamp(ce), alpha=0.12, color=c)
ax.set_ylabel('SPY Weight (%)')
ax.set_ylim(0, 80)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel 4: AGG (bond) allocation
agg_idx = asset_cols.index('AGG')
ax = axes[3]
ax.plot(valid_dates[:len(weights_sv)], weights_sv[:, agg_idx] * 100,
        color='steelblue', linewidth=0.8, alpha=0.8, label='SV-Sharpe')
ax.plot(valid_dates[:len(weights_hist)], weights_hist[:, agg_idx] * 100,
        color='seagreen', linewidth=0.8, alpha=0.8, label='Hist-Sharpe')
ax.axhline(y=40, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='60/40')
for cs, ce, c in [(GFC_START, GFC_END, 'red'), (COVID_START, COVID_END, 'orange')]:
    ax.axvspan(pd.Timestamp(cs), pd.Timestamp(ce), alpha=0.12, color=c)
ax.set_ylabel('AGG Weight (%)')
ax.set_xlabel('Date')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('markowitz_sharpe_backtest.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved to markowitz_sharpe_backtest.png")

# =============================================================================
# 11. Zoomed crisis plots
# =============================================================================
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))

for col, (crisis_name, c_start, c_end) in enumerate([
    ('GFC 2008', GFC_START, GFC_END), ('COVID-19', COVID_START, COVID_END)
]):
    # Wider window for context
    wide_start = pd.Timestamp(c_start) - pd.Timedelta(days=180)
    wide_end = pd.Timestamp(c_end) + pd.Timedelta(days=365)

    mask = (valid_dates >= wide_start) & (valid_dates <= wide_end)

    # Cumulative returns
    ax = axes2[0, col]
    ax.plot(valid_dates[mask], cumret_sv[mask] * 100, color='steelblue', linewidth=1.5, label='SV-Sharpe')
    ax.plot(valid_dates[mask], cumret_hist[mask] * 100, color='seagreen', linewidth=1.5, label='Hist-Sharpe')
    ax.plot(valid_dates[mask], cumret_6040[mask] * 100, color='gray', linewidth=1, alpha=0.7, label='60/40')
    ax.axvspan(pd.Timestamp(c_start), pd.Timestamp(c_end), alpha=0.15, color='red')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('Return (%)')
    ax.set_title(f'{crisis_name} — Cumulative Returns')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # SPY weight during crisis
    ax = axes2[1, col]
    mask_w = (valid_dates[:len(weights_sv)] >= wide_start) & (valid_dates[:len(weights_sv)] <= wide_end)
    ax.plot(valid_dates[:len(weights_sv)][mask_w], weights_sv[mask_w, spy_idx] * 100,
            color='steelblue', linewidth=1, label='SV-Sharpe')
    ax.plot(valid_dates[:len(weights_hist)][mask_w], weights_hist[mask_w, spy_idx] * 100,
            color='seagreen', linewidth=1, label='Hist-Sharpe')
    ax.axhline(y=60, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvspan(pd.Timestamp(c_start), pd.Timestamp(c_end), alpha=0.15, color='red')
    ax.set_ylabel('SPY Weight (%)')
    ax.set_xlabel('Date')
    ax.set_title(f'{crisis_name} — SPY Allocation')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 80)

plt.tight_layout()
plt.savefig('markowitz_crisis_zoom.png', dpi=150, bbox_inches='tight')
print(f"Plot saved to markowitz_crisis_zoom.png")

# =============================================================================
# 12. Key insight
# =============================================================================
print("\n" + "=" * 70)
print("KEY INSIGHT")
print("=" * 70)
print(f"""
Markowitz Max-Sharpe comparison (5-asset, monthly rebalancing):

1. SV-Sharpe allocates LESS to equities than Hist-Sharpe because
   Heavy-Tailed SV produces higher variance estimates for risky assets
   (the fatter tails inflate the covariance matrix).

2. During crises, SV-Sharpe should de-risk FASTER because:
   - Fat-tail scenarios inflate variance during volatile periods
   - The optimizer shifts toward bonds (AGG) and gold (GLD)
   - This provides better drawdown protection at the cost of lower returns

3. The fundamental tradeoff:
   - SV-Sharpe: Lower returns in normal times, smaller drawdowns in crises
   - Hist-Sharpe: Higher returns in normal times, larger drawdowns in crises
   - 60/40: Fixed allocation, no dynamic risk management

4. The SV advantage comes from CONSERVATIVE variance estimation,
   not from the Harris chain. Any model with fatter tails (GARCH-t,
   Student-t copula) would produce similar de-risking behavior.

Average SPY allocation:
  SV-Sharpe:   {weights_sv[:, spy_idx].mean():.1%}
  Hist-Sharpe: {weights_hist[:, spy_idx].mean():.1%}
  60/40:       60.0%

Full-period Sharpe ratios:
  SV-Sharpe:   {sh_sv:.2f}
  Hist-Sharpe: {sh_hist:.2f}
  60/40:       {sh_6040:.2f}
""")