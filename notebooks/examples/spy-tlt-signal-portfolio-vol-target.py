# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Equal-Weight Portfolio of SPY/TLT Signals + BTC/QQQ Residual UPRO Signal
#
# ## Research Question
#
# If we combine the SPY/TLT calendar and relative-value signals with the
# BTC/QQQ residual long-UPRO rule into one portfolio, does a simple equal-weight
# blend improve diversification and risk-adjusted performance versus each signal
# on a standalone basis?
#
# **Portfolio decision for this notebook:** equal-weight blending across the four
# signals (each leg receives `1/4` of capital), with no optimization overlay.
# Daily weights for execution are exported with
# `scripts/export_equal_weight_portfolio_weights.py`.
#
# ## Hypothesis
#
# The four signals likely have different timing and risk profiles, so equal
# weighting can diversify idiosyncratic signal noise without adding model
# complexity.
#
# **Bitcoin UPRO prediction thesis:** when BTC strength is unusually high
# relative to a beta-adjusted QQQ move at the U.S. equity close, the residual
# can capture incremental risk-on information not fully explained by equities.
# That residual-threshold signal may improve the portfolio by contributing a
# partially distinct return stream rather than only adding more S&P 500 beta.

# %%
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

if "__file__" in globals():
    _REPO_ROOT = Path(__file__).resolve().parents[2]
else:
    _cwd = Path.cwd().resolve()
    _REPO_ROOT = _cwd if (_cwd / "research").is_dir() else _cwd.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from research.data import download_massive_daily_closes
from research.plotting import apply_default_style
from research.signal_portfolio_blend import (
    SignalPortfolioParams,
    blend_signal_exposures,
    build_signal_portfolio_bundle,
    equal_blend_weights,
    gross_exposure_shares,
)
from research.stats import annualized_turnover_one_way, mean_daily_turnover_one_way

load_dotenv(dotenv_path=_REPO_ROOT / ".env")
apply_default_style()

# %% [markdown]
# ## Assumptions
#
# - All signals use daily close data and trade on the next close-to-close return.
# - The four strategies are:
#   1. End-of-month SPY/TLT laggard rotation from trading day 15.
#   2. 5-day mean-reversion in `log(SPY/TLT)` as a long-only switch.
#   3. TLT turn-of-month long-last-5 / short-first-5 rule.
#   4. BTC/QQQ residual z-score long UPRO (flat when signal is off).
# - Portfolio construction is fixed to equal-weight (`25%` per signal) for the
#   full sample; no weight optimization or vol-target overlay is applied.
# - Transaction costs, slippage, borrow costs, and financing are excluded.
#
# ## Data Sources
#
# - Massive S3 flat files (US stock day aggregates) via `research.data`.
# - Massive S3 flat files for QQQ/UPRO daily and crypto minute BTC for the UPRO
#   residual signal via `research.upro_residual`.

# %%
START_DATE = "2004-01-01"
TRADING_DAYS_PER_YEAR = 252
EOM_TRIGGER_DAY = 15
RELATIVE_REVERSAL_LOOKBACK = 5
TURN_OF_MONTH_WINDOW = 5
BETA_LOOKBACK_DAYS = 40
ZSCORE_LOOKBACK_DAYS = 20
ENTRY_ZSCORE = 1.5
ROLLING_BETA_WINDOW = 63


def max_drawdown(return_series: pd.Series) -> float:
    equity_curve = (1 + return_series.fillna(0)).cumprod()
    drawdown = equity_curve / equity_curve.cummax() - 1
    return float(drawdown.min())


def summarize_returns(return_series: pd.Series) -> dict[str, float]:
    returns = return_series.dropna()
    if returns.empty:
        return {
            "observations": 0,
            "total_return": np.nan,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe_ratio": np.nan,
            "max_drawdown": np.nan,
            "win_rate": np.nan,
        }

    equity = (1 + returns).cumprod()
    total_return = equity.iloc[-1] - 1
    annualized_return = (
        equity.iloc[-1] ** (TRADING_DAYS_PER_YEAR / len(returns)) - 1
        if equity.iloc[-1] > 0
        else np.nan
    )
    annualized_volatility = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = (
        returns.mean() / returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        if returns.std() > 0
        else np.nan
    )
    return {
        "observations": len(returns),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown(returns),
        "win_rate": (returns > 0).mean(),
    }


def beta_to_benchmark(return_series: pd.Series, benchmark_return: pd.Series) -> tuple[float, float]:
    aligned = pd.concat(
        [return_series.rename("signal"), benchmark_return.rename("benchmark")],
        axis=1,
    ).dropna()
    if aligned.empty:
        return np.nan, np.nan
    variance = aligned["benchmark"].var()
    if variance <= 0:
        return np.nan, np.nan
    beta = aligned["signal"].cov(aligned["benchmark"]) / variance
    correlation = aligned["signal"].corr(aligned["benchmark"])
    return float(beta), float(correlation)


def summarize_beta_vs_spy(
    returns_frame: pd.DataFrame,
    benchmark_return: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for column in returns_frame.columns:
        aligned = pd.concat(
            [returns_frame[column].rename("signal"), benchmark_return.rename("benchmark")],
            axis=1,
        ).dropna()
        beta, corr = beta_to_benchmark(aligned["signal"], aligned["benchmark"])
        rows.append(
            {
                "series": column,
                "observations": len(aligned),
                "beta_to_sp500_proxy_spy": beta,
                "correlation_to_sp500_proxy_spy": corr,
            }
        )
    return pd.DataFrame(rows).sort_values("beta_to_sp500_proxy_spy")


def plot_exposure_mix(ax, shares: pd.DataFrame, title: str) -> None:
    idx = shares.index
    series = [shares[c].to_numpy() for c in shares.columns]
    ax.stackplot(idx, *series, labels=shares.columns, alpha=0.88)
    ax.set_title(title)
    ax.set_ylabel("Share of gross exposure")
    ax.set_xlabel("Date")
    ax.set_ylim(0.0, 1.0)
    ax.margins(x=0.02)
    locator = mdates.AutoDateLocator(minticks=6, maxticks=14)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.tick_params(axis="x", labelsize=9)
    plt.setp(ax.get_xticklabels(), rotation=28, ha="right")
    ax.legend(loc="upper left", fontsize=8)


_portfolio_params = SignalPortfolioParams(
    start_date=START_DATE,
    residual_start_date="2023-01-01",
    eom_trigger_day=EOM_TRIGGER_DAY,
    relative_reversal_lookback=RELATIVE_REVERSAL_LOOKBACK,
    turn_of_month_window=TURN_OF_MONTH_WINDOW,
    beta_lookback_days=BETA_LOOKBACK_DAYS,
    zscore_lookback_days=ZSCORE_LOOKBACK_DAYS,
    entry_zscore=ENTRY_ZSCORE,
)
_bundle = build_signal_portfolio_bundle(_portfolio_params, data_source="s3")
signal_returns = _bundle.signal_returns
per_signal_exposure = _bundle.per_signal_exposure
signal_returns.tail(10)

# %% [markdown]
# ## Methodology
#
# 1. Build daily return streams for the three SPY/TLT rules and the UPRO residual rule.
# 2. Inner-join on dates so all legs are defined (sample starts when UPRO/BTC data allow).
# 3. Apply fixed equal weights (`25%` per signal) to build portfolio returns.
# 4. Compare standalone signal performance with the equal-weight blend.
# 5. Measure pairwise signal correlation to check whether each added signal
#    increases diversification.
# 6. Estimate each signal's beta to the S&P 500 proxy (SPY daily returns) to
#    separate directional market exposure from idiosyncratic edge.
# 7. Export equal-weight implied ETF weights for production with
#    `scripts/export_equal_weight_portfolio_weights.py` (shared pipeline in
#    `research.signal_portfolio_blend`).

# %% [markdown]
# ## Analysis

# %%
equal_weights = equal_blend_weights(signal_returns)
weight_table = equal_weights.rename("equal_weight").to_frame()
weight_table

# %%
portfolio_returns = pd.DataFrame(
    {
        "equal_weight_return": signal_returns.mul(equal_weights, axis=1).sum(axis=1),
    }
).dropna()
portfolio_returns.tail(10)

# %%
combined_returns = signal_returns.join(portfolio_returns, how="inner")
combined_returns.tail(10)

# %%
summary_rows: list[dict[str, object]] = []
for strategy_name in combined_returns.columns:
    full_metrics = summarize_returns(combined_returns[strategy_name])
    strategy_index = combined_returns[strategy_name].dropna().index
    if strategy_name == "equal_weight_return":
        strategy_exposure = blend_signal_exposures(per_signal_exposure, equal_weights).reindex(
            strategy_index
        ).fillna(0.0)
    else:
        strategy_exposure = per_signal_exposure[strategy_name].reindex(strategy_index).fillna(0.0)
    summary_rows.append(
        {
            "strategy": strategy_name,
            "full_total_return": full_metrics["total_return"],
            "full_sharpe": full_metrics["sharpe_ratio"],
            "full_max_drawdown": full_metrics["max_drawdown"],
            "full_ann_vol": full_metrics["annualized_volatility"],
            "mean_daily_turnover_one_way": mean_daily_turnover_one_way(strategy_exposure),
            "full_annualized_turnover_one_way": annualized_turnover_one_way(
                strategy_exposure, trading_days_per_year=TRADING_DAYS_PER_YEAR
            ),
        }
    )

performance_summary = pd.DataFrame(summary_rows)
performance_summary

# %%
signal_correlation = signal_returns.corr()
signal_correlation

# %%
signal_names = signal_returns.columns.tolist()
upper_triangle_mask = np.triu(np.ones(signal_correlation.shape, dtype=bool), k=1)
pairwise_correlations = signal_correlation.where(upper_triangle_mask).stack()
spy_tlt_corr = signal_returns[[c for c in signal_names if c != "upro_residual"]].corr()
spy_tlt_pairwise = spy_tlt_corr.where(np.triu(np.ones(spy_tlt_corr.shape, dtype=bool), k=1)).stack()
diversification_summary = pd.DataFrame(
    {
        "metric": [
            "avg_pairwise_corr_spy_tlt_only",
            "avg_pairwise_corr_with_upro_residual",
            "median_pairwise_corr_with_upro_residual",
            "min_pairwise_corr_with_upro_residual",
            "max_pairwise_corr_with_upro_residual",
            "change_in_avg_pairwise_corr_after_adding_upro_residual",
        ],
        "value": [
            spy_tlt_pairwise.mean(),
            pairwise_correlations.mean(),
            pairwise_correlations.median(),
            pairwise_correlations.min(),
            pairwise_correlations.max(),
            pairwise_correlations.mean() - spy_tlt_pairwise.mean(),
        ],
    }
)
diversification_summary

# %%
benchmark_prices = download_massive_daily_closes(["SPY"], start_date=START_DATE).dropna()
if benchmark_prices.empty:
    raise ValueError("No SPY benchmark prices were downloaded for beta analysis.")
benchmark_returns = benchmark_prices["SPY"].pct_change().rename("spy_return")
beta_input_returns = combined_returns.join(benchmark_returns, how="inner").dropna()
beta_summary = summarize_beta_vs_spy(
    beta_input_returns.drop(columns=["spy_return"]),
    beta_input_returns["spy_return"],
)
beta_summary

# %% [markdown]
# ## Visualizations

# %%
equity_curves = (1 + combined_returns).cumprod()
fig, ax = plt.subplots()
equity_curves.plot(ax=ax)
ax.set_title("Signal and Equal-Weight Portfolio Equity Curves")
ax.set_xlabel("Date")
ax.set_ylabel("Growth of $1")
plt.show()

# %%
# Underlying ETF mix (per $1 of blended signal capital).
# Slices sum to 100% of gross exposure (SPY long, TLT long, TLT short, UPRO long,
# cash/flat).
net_equal = blend_signal_exposures(per_signal_exposure, equal_weights)
net_equal_plot = net_equal.reindex(combined_returns.index).fillna(0.0)
shares_equal = gross_exposure_shares(net_equal_plot)

fig, ax = plt.subplots(figsize=(14, 4))
plot_exposure_mix(ax, shares_equal, "Equal-weight signals — implied ETF mix (% of gross exposure)")
fig.subplots_adjust(bottom=0.20, left=0.07, right=0.98)
plt.show()

# %%
fig, ax = plt.subplots()
im = ax.imshow(signal_correlation.values, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(np.arange(len(signal_names)), labels=signal_names, rotation=25, ha="right")
ax.set_yticks(np.arange(len(signal_names)), labels=signal_names)
ax.set_title("Signal Return Correlation Matrix")
for i in range(len(signal_names)):
    for j in range(len(signal_names)):
        ax.text(
            j,
            i,
            f"{signal_correlation.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            color="black",
            fontsize=8,
        )
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Correlation")
plt.tight_layout()
plt.show()

# %%
rolling_betas = pd.DataFrame(index=beta_input_returns.index)
for column in combined_returns.columns:
    aligned = beta_input_returns[[column, "spy_return"]].dropna()
    rolling_cov = aligned[column].rolling(ROLLING_BETA_WINDOW).cov(aligned["spy_return"])
    rolling_var = aligned["spy_return"].rolling(ROLLING_BETA_WINDOW).var()
    rolling_betas[column] = rolling_cov / rolling_var

fig, ax = plt.subplots()
rolling_betas.plot(ax=ax)
ax.axhline(0.0, color="black", linewidth=1)
ax.set_title(f"Rolling {ROLLING_BETA_WINDOW}-Day Beta to SP500 Proxy (SPY)")
ax.set_xlabel("Date")
ax.set_ylabel("Beta")
plt.show()

# %% [markdown]
# ## Limitations
#
# - SPY is used as a liquid S&P 500 proxy; beta estimates versus index futures or
#   cash index levels may differ slightly.
# - Correlation and beta are estimated from historical daily close-to-close data
#   and can shift materially across market regimes.
# - Costs, turnover drag, and implementation constraints are not modeled.
# - The inner join with the UPRO residual leg shortens the combined sample to
#   dates where QQQ, UPRO, BTC, SPY, and TLT history all overlap.
#
# ## Conclusion
#
# This notebook backtests a combined portfolio of the three SPY/TLT signals
# plus the BTC/QQQ residual UPRO rule using an explicit **equal-weight**
# construction. The added diagnostics quantify whether the UPRO residual signal
# diversifies the signal set (correlation analysis) and how much market exposure
# each signal contributes (beta to SPY as S&P 500 proxy). Export daily weights with
# `uv run python scripts/export_equal_weight_portfolio_weights.py`.
#
# ## Next Research Ideas
#
# - Add conditional-correlation and conditional-beta analysis by volatility
#   regime to see when diversification is strongest.
# - Include turnover and trading-cost penalties in performance comparisons.
# - Add regime filters (rate volatility, trend, correlation state).
