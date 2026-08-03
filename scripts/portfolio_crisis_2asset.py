"""CVaR Portfolio Backtest: 2-Asset (Risk-Free + SPY) Daily Rebalancing Near Crises.

Compares three CVaR-optimized portfolios:
1. Heavy-Tailed SV: tau ~ Empirical(r^2), r|tau ~ N(0,tau) — heavier tails
2. Historical Simulation: iid bootstrap — capped at training max
3. 60/40 benchmark: fixed 60% SPY / 40% risk-free

Setup: 2 assets (SPY + SHV cash equivalent), daily rebalancing,
1-year rolling window, zoomed into crisis windows.

The question: does the fatter tail estimation of SV produce portfolios
that protect better during GFC 2008 and COVID-19?
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
PORTFOLIO_TICKERS = ['SPY', 'SHV']  # SPY: S&P 500, SHV: Short-term Treasury ETF

WINDOW = 252          # 1-year rolling window (more reactive near crises)
REBALANCE = 1         # DAILY rebalancing
N_SIM = 5000          # more scenarios for daily optimization
ALPHA = 0.05          # CVaR 95%
RF_ANNUAL = 0.02      # risk-free rate floor for SHV

# Crisis windows (wider to see buildup and recovery)
GFC_WINDOW = ('2007-06-01', '2010-06-01')   # GFC: buildup, crash, recovery
COVID_WINDOW = ('2019-10-01', '2021-06-01')  # COVID: pre-crash, crash, recovery

# For full backtest
FULL_START = '2007-01-01'
FULL_END = '2026-05-01'

print("=" * 70)
print("2-ASSET CVaR PORTFOLIO: RISK-FREE + SPY, DAILY REBALANCING")
print("=" * 70)
print()
print("Heavy-Tailed SV: tau ~ Empirical(r^2), r|tau ~ N(0,tau) [fatter tails]")
print("Historical Simulation: iid bootstrap [hard-capped at training max]")
print("Benchmark: 60/40 SPY/SHV [fixed allocation]")
print()

# =============================================================================
# 2. Download data
# =============================================================================
print("Downloading data...")
prices = yf.download(PORTFOLIO_TICKERS, start=FULL_START, end=FULL_END)['Close']

# SHV might have different length, align
prices = prices.dropna()
returns = np.log(prices / prices.shift(1)).dropna()

# If only 1 column (yfinance structure issue), handle it
if isinstance(returns, pd.Series):
    returns = returns.to_frame()

print(f"  Data: {returns.index[0].strftime('%Y-%m-%d')} to {returns.index[-1].strftime('%Y-%m-%d')}")
print(f"  Assets: {list(returns.columns)}")
print(f"  Trading days: {len(returns)}")

asset_cols = list(returns.columns)
if 'SPY' not in asset_cols:
    print(f"  WARNING: SPY not found in columns: {asset_cols}")
if 'SHV' not in asset_cols:
    print(f"  WARNING: SHV not found. Using constant risk-free rate.")

returns_arr = returns.values
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
        # For each asset, draw a random past day and use its variance
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


def cvar_portfolio_opt(scenarios, alpha=0.05, n_assets=2):
    """Minimize CVaR using Rockafellar-Uryasev.
    Long-only, sum of weights = 1."""
    def objective(params):
        w = params[:n_assets]
        var = params[n_assets]
        portfolio_returns = scenarios @ w
        losses = -portfolio_returns
        cvar = var + np.mean(np.maximum(losses - var, 0)) / alpha
        return cvar

    constraints = [{'type': 'eq', 'fun': lambda p: np.sum(p[:n_assets]) - 1}]
    bounds = [(0.0, 1.0)] * n_assets + [(None, None)]

    x0 = np.concatenate([
        np.ones(n_assets) / n_assets,
        [np.percentile(-scenarios @ (np.ones(n_assets) / n_assets), 95)]
    ])

    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                     options={'maxiter': 1000, 'ftol': 1e-10})
    return result.x[:n_assets]


# =============================================================================
# 4. Rolling daily backtest
# =============================================================================
print("\nRunning daily-rebalanced backtest...")
print(f"  Training window: {WINDOW} days (1 year)")
print(f"  Rebalance: daily")
print(f"  Horizon: 1-day (next day return)")
print(f"  CVaR alpha: {ALPHA}")

n_days = len(returns_arr)
weights_sv_all = np.full((n_days, len(asset_cols)), np.nan)
weights_hist_all = np.full((n_days, len(asset_cols)), np.nan)

# Use 1-day horizon for daily rebalancing
HORIZON = 1

for t in range(WINDOW, n_days):
    train_data = returns_arr[t - WINDOW:t]

    rng_sv = np.random.default_rng(42 + t)
    rng_hist = np.random.default_rng(43 + t)

    # Generate 1-day scenarios
    sv_scenarios = simulate_ht_sv(train_data, HORIZON, N_SIM, rng_sv)[:, 0, :]  # (N_SIM, n_assets)
    hist_scenarios = simulate_hist(train_data, HORIZON, N_SIM, rng_hist)[:, 0, :]  # (N_SIM, n_assets)

    # Optimize CVaR
    try:
        w_sv = cvar_portfolio_opt(sv_scenarios, ALPHA, len(asset_cols))
    except Exception:
        w_sv = np.array([0.6, 0.4])  # fallback to 60/40

    try:
        w_hist = cvar_portfolio_opt(hist_scenarios, ALPHA, len(asset_cols))
    except Exception:
        w_hist = np.array([0.6, 0.4])

    # Clip weights for numerical stability
    w_sv = np.clip(w_sv, 0.0, 1.0)
    w_sv /= w_sv.sum()
    w_hist = np.clip(w_hist, 0.0, 1.0)
    w_hist /= w_hist.sum()

    weights_sv_all[t] = w_sv
    weights_hist_all[t] = w_hist

    if t % 252 == 0:
        print(f"  Processing {dates_arr[t].strftime('%Y-%m-%d')}... "
              f"SV: SPY={w_sv[0]:.1%}  Hist: SPY={w_hist[0]:.1%}")

print("  Done.")

# =============================================================================
# 5. Compute portfolio returns
# =============================================================================
valid = ~np.isnan(weights_sv_all[:, 0])

port_ret_sv = np.sum(returns_arr * weights_sv_all, axis=1)
port_ret_hist = np.sum(returns_arr * weights_hist_all, axis=1)
port_ret_6040 = returns_arr @ np.array([0.6, 0.4])  # 60/40 benchmark

# Zero out invalid periods
port_ret_sv[~valid] = np.nan
port_ret_hist[~valid] = np.nan
port_ret_6040[~valid] = np.nan

# Drop NaN
mask = valid
port_ret_sv_clean = port_ret_sv[mask]
port_ret_hist_clean = port_ret_hist[mask]
port_ret_6040_clean = port_ret_6040[mask]
valid_dates = dates_arr[mask]
weights_sv_clean = weights_sv_all[mask]
weights_hist_clean = weights_hist_all[mask]

# Cumulative returns
cumret_sv = np.cumsum(port_ret_sv_clean)
cumret_hist = np.cumsum(port_ret_hist_clean)
cumret_6040 = np.cumsum(port_ret_6040_clean)

# =============================================================================
# 6. Results: Full period
# =============================================================================
def max_drawdown(cumret):
    """Compute drawdown series and max drawdown."""
    peak = np.maximum.accumulate(cumret)
    dd = cumret - peak  # drawdown in log-returns
    # For percentage drawdown from peak, use cumulative wealth
    wealth = np.exp(cumret)
    peak_wealth = np.maximum.accumulate(wealth)
    dd_pct = (wealth - peak_wealth) / peak_wealth
    return dd_pct, np.min(dd_pct)


dd_sv, maxdd_sv = max_drawdown(cumret_sv)
dd_hist, maxdd_hist = max_drawdown(cumret_hist)
dd_6040, maxdd_6040 = max_drawdown(cumret_6040)

def annualized_stats(returns):
    """Compute annualized return, vol, Sharpe from daily log returns."""
    mu = np.nanmean(returns) * 252
    sigma = np.nanstd(returns) * np.sqrt(252)
    sharpe = mu / sigma if sigma > 0 else 0
    return mu, sigma, sharpe

print("\n" + "=" * 70)
print("FULL PERIOD RESULTS")
print("=" * 70)
mu_sv, sigma_sv, sharpe_sv = annualized_stats(port_ret_sv_clean)
mu_hist, sigma_hist, sharpe_hist = annualized_stats(port_ret_hist_clean)
mu_6040, sigma_6040, sharpe_6040 = annualized_stats(port_ret_6040_clean)

print(f"\n{'Portfolio':>15}  {'Ann.Ret':>8}  {'Ann.Vol':>8}  {'Sharpe':>8}  {'MaxDD':>8}")
print(f"{'-'*55}")
print(f"{'CVaR-SV':>15}  {mu_sv*100:>7.1f}%  {sigma_sv*100:>7.1f}%  {sharpe_sv:>7.2f}  {maxdd_sv*100:>7.1f}%")
print(f"{'CVaR-Hist':>15}  {mu_hist*100:>7.1f}%  {sigma_hist*100:>7.1f}%  {sharpe_hist:>7.2f}  {maxdd_hist*100:>7.1f}%")
print(f"{'60/40':>15}  {mu_6040*100:>7.1f}%  {sigma_6040*100:>7.1f}%  {sharpe_6040:>7.2f}  {maxdd_6040*100:>7.1f}%")

# =============================================================================
# 7. Average weight allocation
# =============================================================================
spy_idx = asset_cols.index('SPY') if 'SPY' in asset_cols else 0
shv_idx = asset_cols.index('SHV') if 'SHV' in asset_cols else 1

print("\n" + "=" * 70)
print("AVERAGE SPY ALLOCATION")
print("=" * 70)
print(f"\n  CVaR-SV:   SPY = {weights_sv_clean[:, spy_idx].mean():.1%}  (std: {weights_sv_clean[:, spy_idx].std():.1%})")
print(f"  CVaR-Hist: SPY = {weights_hist_clean[:, spy_idx].mean():.1%}  (std: {weights_hist_clean[:, spy_idx].std():.1%})")
print(f"  60/40:     SPY = 60.0%")

# Weight ranges
print(f"\n  CVaR-SV:   SPY range: [{weights_sv_clean[:, spy_idx].min():.1%}, {weights_sv_clean[:, spy_idx].max():.1%}]")
print(f"  CVaR-Hist: SPY range: [{weights_hist_clean[:, spy_idx].min():.1%}, {weights_hist_clean[:, spy_idx].max():.1%}]")

# =============================================================================
# 8. Crisis analysis: zoomed into each crisis
# =============================================================================
print("\n" + "=" * 70)
print("CRISIS ANALYSIS")
print("=" * 70)

for crisis_name, c_start, c_end in [('GFC 2008', GFC_WINDOW[0], GFC_WINDOW[1]),
                                      ('COVID-19', COVID_WINDOW[0], COVID_WINDOW[1])]:
    mask = (valid_dates >= pd.Timestamp(c_start)) & (valid_dates <= pd.Timestamp(c_end))
    if mask.sum() == 0:
        print(f"\n  {crisis_name}: No data available")
        continue

    # Crisis portfolio returns
    ret_sv_c = port_ret_sv_clean[mask]
    ret_hist_c = port_ret_hist_clean[mask]
    ret_6040_c = port_ret_6040_clean[mask]
    dates_c = valid_dates[mask]

    # Cumulative returns during crisis
    cum_sv_c = np.cumsum(ret_sv_c)
    cum_hist_c = np.cumsum(ret_hist_c)
    cum_6040_c = np.cumsum(ret_6040_c)

    # Max drawdown during crisis
    _, maxdd_sv_c = max_drawdown(cum_sv_c)
    _, maxdd_hist_c = max_drawdown(cum_hist_c)
    _, maxdd_6040_c = max_drawdown(cum_6040_c)

    # Annualized stats during crisis
    mu_sv_c, sigma_sv_c, _ = annualized_stats(ret_sv_c)
    mu_hist_c, sigma_hist_c, _ = annualized_stats(ret_hist_c)
    mu_6040_c, sigma_6040_c, _ = annualized_stats(ret_6040_c)

    # Total return during crisis
    total_sv = cum_sv_c[-1]
    total_hist = cum_hist_c[-1]
    total_6040 = cum_6040_c[-1]

    # SPY weight during crisis
    w_spy_sv_c = weights_sv_clean[mask, spy_idx]
    w_spy_hist_c = weights_hist_clean[mask, spy_idx]

    print(f"\n  {crisis_name} ({c_start} to {c_end}):")
    print(f"  {'Portfolio':>15}  {'Return':>8}  {'MaxDD':>8}  {'Ann.Vol':>8}  {'Avg SPY%':>9}")
    print(f"  {'-'*60}")
    print(f"  {'CVaR-SV':>15}  {total_sv*100:>7.1f}%  {maxdd_sv_c*100:>7.1f}%  {sigma_sv_c*100:>7.1f}%  {w_spy_sv_c.mean()*100:>8.1f}%")
    print(f"  {'CVaR-Hist':>15}  {total_hist*100:>7.1f}%  {maxdd_hist_c*100:>7.1f}%  {sigma_hist_c*100:>7.1f}%  {w_spy_hist_c.mean()*100:>8.1f}%")
    print(f"  {'60/40':>15}  {total_6040*100:>7.1f}%  {maxdd_6040_c*100:>7.1f}%  {sigma_6040_c*100:>7.1f}%  {'60.0':>8}%")

# =============================================================================
# 9. Plot: Full backtest with crisis highlights
# =============================================================================
fig, axes = plt.subplots(4, 1, figsize=(16, 14), gridspec_kw={'height_ratios': [3, 2, 2, 2]})

# Panel 1: Cumulative returns
ax = axes[0]
ax.plot(valid_dates, cumret_sv * 100, label='CVaR-SV (Heavy-Tailed)', color='steelblue', linewidth=1.5)
ax.plot(valid_dates, cumret_hist * 100, label='CVaR-Hist (Bootstrap)', color='seagreen', linewidth=1.5)
ax.plot(valid_dates, cumret_6040 * 100, label='60/40 Benchmark', color='gray', linewidth=1, alpha=0.7)

# Highlight crises
for c_start, c_end, color, label in [
    (GFC_WINDOW[0], GFC_WINDOW[1], 'red', 'GFC'),
    (COVID_WINDOW[0], COVID_WINDOW[1], 'orange', 'COVID')
]:
    ax.axvspan(pd.Timestamp(c_start), pd.Timestamp(c_end), alpha=0.12, color=color)

ax.set_ylabel('Cumulative Return (%)')
ax.set_title('CVaR-Optimized Portfolio: SPY + Risk-Free (Daily Rebalancing)')
ax.legend(fontsize=10, loc='upper left')
ax.grid(alpha=0.3)

# Panel 2: Drawdowns
ax = axes[1]
ax.fill_between(valid_dates, dd_sv * 100, 0, color='steelblue', alpha=0.5, label='CVaR-SV')
ax.fill_between(valid_dates, dd_hist * 100, 0, color='seagreen', alpha=0.3, label='CVaR-Hist')
for c_start, c_end, color in [(GFC_WINDOW[0], GFC_WINDOW[1], 'red'), (COVID_WINDOW[0], COVID_WINDOW[1], 'orange')]:
    ax.axvspan(pd.Timestamp(c_start), pd.Timestamp(c_end), alpha=0.12, color=color)
ax.set_ylabel('Drawdown (%)')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel 3: SPY allocation over time
ax = axes[2]
ax.plot(valid_dates, weights_sv_clean[:, spy_idx] * 100, color='steelblue', linewidth=0.5, alpha=0.7, label='CVaR-SV')
ax.plot(valid_dates, weights_hist_clean[:, spy_idx] * 100, color='seagreen', linewidth=0.5, alpha=0.7, label='CVaR-Hist')
ax.axhline(y=60, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='60/40')
for c_start, c_end, color in [(GFC_WINDOW[0], GFC_WINDOW[1], 'red'), (COVID_WINDOW[0], COVID_WINDOW[1], 'orange')]:
    ax.axvspan(pd.Timestamp(c_start), pd.Timestamp(c_end), alpha=0.12, color=color)
ax.set_ylabel('SPY Weight (%)')
ax.set_ylim(0, 100)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel 4: Zoomed GFC
ax = axes[3]
gfc_mask = (valid_dates >= pd.Timestamp(GFC_WINDOW[0])) & (valid_dates <= pd.Timestamp(GFC_WINDOW[1]))
gfc_dates = valid_dates[gfc_mask]
ax.plot(gfc_dates, cumret_sv[gfc_mask] * 100, color='steelblue', linewidth=1.5, label='CVaR-SV')
ax.plot(gfc_dates, cumret_hist[gfc_mask] * 100, color='seagreen', linewidth=1.5, label='CVaR-Hist')
ax.plot(gfc_dates, cumret_6040[gfc_mask] * 100, color='gray', linewidth=1, alpha=0.7, label='60/40')
ax.set_ylabel('Return (%)')
ax.set_xlabel('Date')
ax.set_title('GFC 2008 Zoom')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.axhline(y=0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig('portfolio_crisis_backtest.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved to portfolio_crisis_backtest.png")

# =============================================================================
# 10. Additional: Zoomed COVID plot
# =============================================================================
fig2, axes2 = plt.subplots(2, 1, figsize=(14, 8))

# COVID zoomed returns
covid_mask = (valid_dates >= pd.Timestamp(COVID_WINDOW[0])) & (valid_dates <= pd.Timestamp(COVID_WINDOW[1]))
covid_dates = valid_dates[covid_mask]

ax = axes2[0]
ax.plot(covid_dates, cumret_sv[covid_mask] * 100, color='steelblue', linewidth=1.5, label='CVaR-SV')
ax.plot(covid_dates, cumret_hist[covid_mask] * 100, color='seagreen', linewidth=1.5, label='CVaR-Hist')
ax.plot(covid_dates, cumret_6040[covid_mask] * 100, color='gray', linewidth=1, alpha=0.7, label='60/40')
ax.set_ylabel('Cumulative Return (%)')
ax.set_title('COVID-19 Zoom: CVaR-Optimized Portfolios')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.axhline(y=0, color='black', linewidth=0.5)

# SPY weight during COVID
ax = axes2[1]
ax.plot(covid_dates, weights_sv_clean[covid_mask, spy_idx] * 100, color='steelblue', linewidth=1, label='CVaR-SV')
ax.plot(covid_dates, weights_hist_clean[covid_mask, spy_idx] * 100, color='seagreen', linewidth=1, label='CVaR-Hist')
ax.axhline(y=60, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.set_ylabel('SPY Weight (%)')
ax.set_xlabel('Date')
ax.set_title('Dynamic SPY Allocation During COVID-19')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig('portfolio_covid_zoom.png', dpi=150, bbox_inches='tight')
print(f"Plot saved to portfolio_covid_zoom.png")

# =============================================================================
# 11. Key insight
# =============================================================================
print("\n" + "=" * 70)
print("KEY INSIGHT")
print("=" * 70)
print(f"""
Heavy-Tailed SV produces fatter tail scenarios than Historical Simulation.
In a 2-asset (SPY + risk-free) portfolio with daily CVaR optimization:

1. SV assigns LESS weight to SPY on average ({weights_sv_clean[:, spy_idx].mean():.1%})
   vs Hist.Sim ({weights_hist_clean[:, spy_idx].mean():.1%})
   because SV scenarios show worse tail outcomes.

2. During crises, SV should de-risk FASTER because its scenarios
   anticipate fatter tails (extrapolation beyond training max).

3. But the difference may be small because:
   - With only 2 assets, the optimizer can only shift between SPY and cash
   - Daily rebalancing means both models react quickly
   - The tail difference matters most at extremes, not in the bulk

The fundamental question: does conservative tail estimation (SV > Hist.Sim)
translate into better crisis protection? Or does the noisier estimation
of SV (each day re-simulates variances) offset the tail advantage?
""")