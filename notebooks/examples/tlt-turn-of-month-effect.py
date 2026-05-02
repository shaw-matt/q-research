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
# # TLT Turn-of-Month Effect
#
# ## Research Question
#
# Does a deterministic calendar rule (long TLT for last 5 trading days of each
# month, then short TLT for first 5 trading days of next month) generate
# positive returns?
#
# ## Hypothesis
#
# End-of-month bond demand may lift prices into month-end, then partially mean
# revert at the start of the next month. A long-then-short schedule may capture
# that pattern.

# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research.data import download_massive_daily_closes
from research.plotting import apply_default_style
from research.stats import annualized_turnover_one_way, mean_daily_turnover_one_way

apply_default_style()

# %% [markdown]
# ## Assumptions
#
# - Daily close signals are executable close-to-close.
# - Go long for the final 5 sessions of each month.
# - Flip to short at month-end close and hold for first 5 sessions next month.
# - Position is shifted one day to avoid look-ahead.
# - No financing costs, borrow frictions, taxes, or slippage.
#
# ## Data Sources
#
# - Massive S3 flat files (US stock day aggregates) for TLT via `research.data`.

# %%
START_DATE = "2004-01-01"
WINDOW_DAYS = 5
TRADING_DAYS_PER_YEAR = 252


def max_drawdown(return_series: pd.Series) -> float:
    equity_curve = (1 + return_series.fillna(0)).cumprod()
    drawdown = equity_curve / equity_curve.cummax() - 1
    return float(drawdown.min())


def summarize_strategy(strategy_returns: pd.Series, position: pd.Series) -> pd.DataFrame:
    returns = strategy_returns.dropna()
    held = position.reindex(returns.index).fillna(0)
    exposure = held.astype(float)
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
                "days_long",
                "days_short",
                "days_flat",
                "mean_daily_turnover_one_way",
                "annualized_turnover_one_way",
                "total_return",
                "annualized_return",
                "annualized_volatility",
                "sharpe_ratio",
                "max_drawdown",
                "daily_win_rate",
            ],
            "value": [
                len(returns),
                (held > 0).sum(),
                (held < 0).sum(),
                (held == 0).sum(),
                mean_daily_turnover_one_way(exposure),
                annualized_turnover_one_way(
                    exposure, trading_days_per_year=TRADING_DAYS_PER_YEAR
                ),
                total_return,
                annualized_return,
                annualized_volatility,
                sharpe,
                max_drawdown(returns),
                (returns > 0).mean(),
            ],
        }
    )


def build_turn_of_month_positions(index: pd.DatetimeIndex, window_days: int = 5) -> pd.Series:
    frame = pd.DataFrame(index=index)
    frame["month"] = frame.index.to_period("M")
    frame["day_in_month"] = frame.groupby("month").cumcount() + 1
    frame["days_in_month"] = frame.groupby("month")["day_in_month"].transform("max")
    frame["days_to_month_end"] = frame["days_in_month"] - frame["day_in_month"] + 1

    month_num = frame["month"].dt.month
    next_month_num = frame["month"].shift(-1).dt.month
    is_month_end = month_num != next_month_num

    frame["position_signal"] = 0
    frame.loc[frame["days_to_month_end"] <= window_days, "position_signal"] = 1
    frame.loc[frame["day_in_month"] <= window_days, "position_signal"] = -1
    frame.loc[is_month_end, "position_signal"] = -1

    return frame["position_signal"]


tlt = download_massive_daily_closes(["TLT"], start_date=START_DATE).dropna()
if tlt.empty:
    raise ValueError("No TLT daily prices were downloaded.")

analysis = tlt.copy()
analysis["TLT_return"] = analysis["TLT"].pct_change()
analysis["position_signal"] = build_turn_of_month_positions(analysis.index, WINDOW_DAYS)
analysis["position"] = analysis["position_signal"].shift(1).fillna(0)
analysis["strategy_return"] = analysis["position"] * analysis["TLT_return"]
analysis = analysis.dropna(subset=["TLT_return"])
analysis.tail(12)

# %% [markdown]
# ## Methodology
#
# 1. Download TLT daily close prices and returns.
# 2. Label each trading day by its order within month and distance to month-end.
# 3. Signal `+1` for last 5 trading sessions of each month.
# 4. Signal `-1` for first 5 trading sessions of each month.
# 5. Explicitly keep month-end close in short state for the next session.
# 6. Shift one day and apply long/short daily TLT return.

# %% [markdown]
# ## Analysis

# %%
calendar_sample = analysis[
    [
        "TLT",
        "TLT_return",
        "position_signal",
        "position",
        "strategy_return",
    ]
].tail(20)
calendar_sample

# %%
performance_summary = summarize_strategy(analysis["strategy_return"], analysis["position"])
performance_summary

# %%
by_month = (
    analysis.assign(month=analysis.index.to_period("M"))
    .groupby("month")
    .agg(
        strategy_return=("strategy_return", lambda r: (1 + r).prod() - 1),
        tlt_return=("TLT_return", lambda r: (1 + r).prod() - 1),
    )
)
by_month.tail(12)

# %% [markdown]
# ## Visualizations

# %%
strategy_curve = (1 + analysis["strategy_return"]).cumprod()
tlt_curve = (1 + analysis["TLT_return"]).cumprod()

fig, ax = plt.subplots()
strategy_curve.plot(ax=ax, label="Turn-of-month strategy")
tlt_curve.plot(ax=ax, label="TLT buy and hold", alpha=0.75)
ax.set_title("Equity Curves")
ax.set_xlabel("Date")
ax.set_ylabel("Growth of $1")
ax.legend()
plt.show()

# %%
fig, ax = plt.subplots()
analysis["position"].plot(ax=ax, linewidth=1)
ax.set_title("Position Timeline (+1 long, -1 short, 0 flat)")
ax.set_xlabel("Date")
ax.set_ylabel("Position")
plt.show()

# %%
fig, ax = plt.subplots()
by_month["strategy_return"].plot(kind="hist", bins=35, ax=ax)
ax.axvline(by_month["strategy_return"].mean(), color="tab:red", linestyle="--")
ax.set_title("Distribution of Strategy Monthly Returns")
ax.set_xlabel("Monthly return")
plt.show()

# %% [markdown]
# ## Limitations
#
# - Purely calendar-based and not conditioned on macro regime or trend.
# - Shorting an ETF can incur borrow and financing costs not modeled here.
# - Month-end close implementation may have auction-impact assumptions.
# - Fixed 5-day windows may not be stable across decades.
#
# ## Conclusion
#
# This notebook implements the simple long-last-5 / short-first-5 turn-of-month
# rule in TLT and evaluates the resulting close-to-close return stream.
#
# ## Next Research Ideas
#
# - Test 3- to 7-day windows around month-end and month-start.
# - Compare with Treasury futures to reduce ETF-specific microstructure effects.
# - Add transaction-cost and borrow-cost stress tests.
# - Condition the short leg on volatility or trend to reduce tail risk.
