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
# # Short-Term SPY/TLT Relative Reversal
#
# ## Research Question
#
# Does a 5-day mean-reversion rule on `log(SPY/TLT)` produce useful daily
# rotation returns between SPY and TLT?
#
# ## Hypothesis
#
# If `log(SPY/TLT)` is below its 5-day moving average, short-term stock relative
# weakness may mean-revert and favor SPY. If above, favor TLT.

# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research.data import download_massive_daily_closes
from research.plotting import apply_default_style

apply_default_style()

# %% [markdown]
# ## Assumptions
#
# - Daily close signals are known and traded at the close for next-day return.
# - One-leg long-only rotation: long SPY or long TLT each day.
# - Position is shifted by one session to avoid same-close look-ahead.
# - No transaction costs, taxes, financing, or slippage.
#
# ## Data Sources
#
# - Massive S3 flat files (US stock day aggregates) for SPY and TLT via `research.data`.

# %%
START_DATE = "2004-01-01"
LOOKBACK_DAYS = 5
TRADING_DAYS_PER_YEAR = 252


def max_drawdown(return_series: pd.Series) -> float:
    equity_curve = (1 + return_series.fillna(0)).cumprod()
    drawdown = equity_curve / equity_curve.cummax() - 1
    return float(drawdown.min())


def summarize_strategy(strategy_returns: pd.Series, position_asset: pd.Series) -> pd.DataFrame:
    returns = strategy_returns.dropna()
    position = position_asset.reindex(returns.index)
    active_returns = returns[position.isin(["SPY", "TLT"])]
    equity = (1 + returns).cumprod()
    total_return = equity.iloc[-1] - 1 if not equity.empty else np.nan
    annualized_return = (
        equity.iloc[-1] ** (TRADING_DAYS_PER_YEAR / len(returns)) - 1
        if len(returns) and equity.iloc[-1] > 0
        else np.nan
    )
    annualized_volatility = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = (
        returns.mean() / returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        if returns.std() > 0
        else np.nan
    )
    return pd.DataFrame(
        {
            "metric": [
                "observations",
                "active_days",
                "long_spy_share",
                "long_tlt_share",
                "total_return",
                "annualized_return",
                "annualized_volatility",
                "sharpe_ratio",
                "max_drawdown",
                "daily_win_rate",
            ],
            "value": [
                len(returns),
                len(active_returns),
                (position == "SPY").mean(),
                (position == "TLT").mean(),
                total_return,
                annualized_return,
                annualized_volatility,
                sharpe,
                max_drawdown(returns),
                (returns > 0).mean(),
            ],
        }
    )


def build_short_term_reversal_strategy(prices: pd.DataFrame, lookback_days: int = 5) -> pd.DataFrame:
    frame = prices.copy()
    frame["SPY_return"] = frame["SPY"].pct_change()
    frame["TLT_return"] = frame["TLT"].pct_change()
    frame["log_ratio"] = np.log(frame["SPY"] / frame["TLT"])
    frame["log_ratio_ma"] = frame["log_ratio"].rolling(lookback_days).mean()

    frame["signal_asset"] = np.where(frame["log_ratio"] < frame["log_ratio_ma"], "SPY", "TLT")
    frame["position_asset"] = frame["signal_asset"].shift(1)
    frame["strategy_return"] = np.where(
        frame["position_asset"] == "SPY",
        frame["SPY_return"],
        np.where(frame["position_asset"] == "TLT", frame["TLT_return"], np.nan),
    )
    return frame


prices = download_massive_daily_closes(["SPY", "TLT"], start_date=START_DATE).dropna()
if prices.empty:
    raise ValueError("No SPY/TLT daily prices were downloaded.")

analysis = build_short_term_reversal_strategy(prices, LOOKBACK_DAYS).dropna(
    subset=["log_ratio_ma", "position_asset", "strategy_return"]
)
analysis.tail(12)

# %% [markdown]
# ## Methodology
#
# 1. Compute `log(SPY/TLT)` each day.
# 2. Compute the 5-day moving average of that log ratio.
# 3. If the log ratio is below its MA, signal long SPY.
# 4. If above its MA, signal long TLT.
# 5. Shift signal one day forward and apply corresponding daily return.

# %% [markdown]
# ## Analysis

# %%
analysis[["SPY", "TLT", "log_ratio", "log_ratio_ma", "signal_asset", "position_asset"]].tail(15)

# %%
performance_summary = summarize_strategy(analysis["strategy_return"], analysis["position_asset"])
performance_summary

# %%
asset_benchmark = pd.DataFrame(
    {
        "strategy_total_return": [(1 + analysis["strategy_return"]).prod() - 1],
        "spy_total_return": [(1 + analysis["SPY_return"].dropna()).prod() - 1],
        "tlt_total_return": [(1 + analysis["TLT_return"].dropna()).prod() - 1],
    }
)
asset_benchmark

# %% [markdown]
# ## Visualizations

# %%
fig, ax = plt.subplots()
analysis["log_ratio"].plot(ax=ax, label="log(SPY/TLT)", alpha=0.8)
analysis["log_ratio_ma"].plot(ax=ax, label=f"{LOOKBACK_DAYS}-day MA", alpha=0.9)
ax.set_title("Relative-Value Signal")
ax.set_xlabel("Date")
ax.set_ylabel("Log ratio")
ax.legend()
plt.show()

# %%
strategy_curve = (1 + analysis["strategy_return"]).cumprod()
spy_curve = (1 + analysis["SPY_return"].fillna(0)).cumprod().reindex(strategy_curve.index)
tlt_curve = (1 + analysis["TLT_return"].fillna(0)).cumprod().reindex(strategy_curve.index)

fig, ax = plt.subplots()
strategy_curve.plot(ax=ax, label="Short-term reversal strategy")
spy_curve.plot(ax=ax, label="SPY buy and hold", alpha=0.75)
tlt_curve.plot(ax=ax, label="TLT buy and hold", alpha=0.75)
ax.set_title("Equity Curves")
ax.set_xlabel("Date")
ax.set_ylabel("Growth of $1")
ax.legend()
plt.show()

# %%
position_mix = analysis["position_asset"].value_counts(normalize=True).rename("share").to_frame()
position_mix

# %% [markdown]
# ## Limitations
#
# - One parameter (5-day MA) can overfit the chosen pair and sample.
# - Daily close-to-close implementation can differ from intraday execution.
# - Switching between assets frequently may incur non-trivial costs.
# - Rule is binary and ignores trend strength or volatility state.
#
# ## Conclusion
#
# This notebook implements the exact short-term relative reversal rule between
# SPY and TLT using `log(SPY/TLT)` versus its 5-day moving average.
#
# ## Next Research Ideas
#
# - Test additional moving-average windows and smoothing variants.
# - Add a no-trade buffer around the moving average to reduce churn.
# - Vol-target the two legs for more balanced risk contribution.
# - Compare close execution with next-open execution.
