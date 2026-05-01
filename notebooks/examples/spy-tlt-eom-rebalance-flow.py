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
# # End-of-Month SPY/TLT Rebalance Flow
#
# ## Research Question
#
# Does rotating into the month-to-date underperformer (SPY or TLT) after the
# 15th trading day and holding through month-end capture end-of-month
# rebalance flow?
#
# ## Hypothesis
#
# If one leg underperforms month-to-date by the mid-month checkpoint, allocators
# may rebalance toward that laggard into month-end. Going long the laggard from
# the next session through month-end may produce positive returns.

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
# - Daily closes are tradable end-of-day marks.
# - "Trading day 15" means the 15th available market session in each month.
# - Signal formed on trading day 15 is implemented on trading day 16.
# - Position is held until the final trading session of that same month.
# - No transaction costs, borrow costs, taxes, or slippage.
#
# ## Data Sources
#
# - Massive daily close data for SPY and TLT, loaded through `research.data`.

# %%
START_DATE = "2004-01-01"
TRADING_DAYS_PER_YEAR = 252
MID_MONTH_TRADING_DAY = 15


def max_drawdown(return_series: pd.Series) -> float:
    equity_curve = (1 + return_series.fillna(0)).cumprod()
    drawdown = equity_curve / equity_curve.cummax() - 1
    return float(drawdown.min())


def summarize_strategy(strategy_returns: pd.Series, positions: pd.Series) -> pd.DataFrame:
    returns = strategy_returns.dropna()
    active_returns = returns.loc[positions.reindex(returns.index).fillna(False)]
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
                "exposure_rate",
                "total_return",
                "annualized_return",
                "annualized_volatility",
                "sharpe_ratio",
                "max_drawdown",
                "active_day_win_rate",
                "average_active_day_return",
            ],
            "value": [
                len(returns),
                len(active_returns),
                positions.reindex(returns.index).fillna(False).mean(),
                total_return,
                annualized_return,
                annualized_volatility,
                sharpe,
                max_drawdown(returns),
                (active_returns > 0).mean() if len(active_returns) else np.nan,
                active_returns.mean() if len(active_returns) else np.nan,
            ],
        }
    )


def build_eom_rebalance_strategy(prices: pd.DataFrame, trigger_day: int = 15) -> pd.DataFrame:
    frame = prices.copy()
    frame["month"] = frame.index.to_period("M")
    frame["day_in_month"] = frame.groupby("month").cumcount() + 1
    frame["SPY_mtd"] = frame.groupby("month")["SPY"].transform(lambda s: s / s.iloc[0] - 1)
    frame["TLT_mtd"] = frame.groupby("month")["TLT"].transform(lambda s: s / s.iloc[0] - 1)

    trigger = frame["day_in_month"] == trigger_day
    frame["signal_asset"] = pd.Series(index=frame.index, dtype="object")
    frame.loc[trigger & (frame["SPY_mtd"] > frame["TLT_mtd"]), "signal_asset"] = "TLT"
    frame.loc[trigger & (frame["TLT_mtd"] > frame["SPY_mtd"]), "signal_asset"] = "SPY"
    frame["signal_asset"] = frame.groupby("month")["signal_asset"].transform("first")
    frame["next_day"] = frame.groupby("month")["day_in_month"].shift(-1)
    frame["is_last_day"] = frame["day_in_month"] == frame.groupby("month")["day_in_month"].transform("max")

    frame["position_asset"] = pd.Series(index=frame.index, dtype="object")
    active_window = (frame["day_in_month"] > trigger_day) & ~frame["is_last_day"]
    frame.loc[active_window, "position_asset"] = frame.loc[active_window, "signal_asset"]
    frame["SPY_return"] = frame["SPY"].pct_change()
    frame["TLT_return"] = frame["TLT"].pct_change()
    frame["strategy_return"] = np.where(
        frame["position_asset"] == "SPY",
        frame["SPY_return"],
        np.where(frame["position_asset"] == "TLT", frame["TLT_return"], 0.0),
    )
    frame["position"] = frame["position_asset"].notna()
    return frame


prices = download_massive_daily_closes(["SPY", "TLT"], start_date=START_DATE).dropna()
if prices.empty:
    raise ValueError("No SPY/TLT daily prices were downloaded.")

analysis = build_eom_rebalance_strategy(prices, MID_MONTH_TRADING_DAY)
analysis.tail(12)

# %% [markdown]
# ## Methodology
#
# 1. Download SPY and TLT daily closes.
# 2. For each month, compute month-to-date returns through each session.
# 3. On trading day 15, compare SPY vs TLT MTD returns.
# 4. If SPY MTD > TLT MTD, set signal to long TLT from next session.
# 5. If TLT MTD > SPY MTD, set signal to long SPY from next session.
# 6. Hold that position through the final session of the month.

# %% [markdown]
# ## Analysis

# %%
signal_snapshots = analysis.loc[
    analysis["day_in_month"] == MID_MONTH_TRADING_DAY,
    ["SPY", "TLT", "SPY_mtd", "TLT_mtd", "signal_asset"],
].tail(15)
signal_snapshots

# %%
performance_summary = summarize_strategy(analysis["strategy_return"], analysis["position"])
performance_summary

# %%
monthly_returns = (
    analysis.assign(month=analysis.index.to_period("M"))
    .groupby("month")["strategy_return"]
    .apply(lambda r: (1 + r).prod() - 1)
    .to_frame(name="strategy_monthly_return")
)
monthly_returns.tail(12)

# %% [markdown]
# ## Visualizations

# %%
equity_curve = (1 + analysis["strategy_return"]).cumprod()
spy_curve = (1 + analysis["SPY_return"].fillna(0)).cumprod()
tlt_curve = (1 + analysis["TLT_return"].fillna(0)).cumprod()

fig, ax = plt.subplots()
equity_curve.plot(ax=ax, label="EOM rebalance strategy")
spy_curve.plot(ax=ax, label="SPY buy and hold", alpha=0.75)
tlt_curve.plot(ax=ax, label="TLT buy and hold", alpha=0.75)
ax.set_title("Equity Curves")
ax.set_xlabel("Date")
ax.set_ylabel("Growth of $1")
ax.legend()
plt.show()

# %%
fig, ax = plt.subplots()
monthly_returns["strategy_monthly_return"].plot(kind="hist", bins=35, ax=ax)
ax.axvline(monthly_returns["strategy_monthly_return"].mean(), color="tab:red", linestyle="--")
ax.set_title("Distribution of Strategy Monthly Returns")
ax.set_xlabel("Monthly return")
plt.show()

# %% [markdown]
# ## Limitations
#
# - Strategy uses a single arbitrary trigger day (15th trading day).
# - Daily close execution ignores bid/ask and closing-auction frictions.
# - SPY/TLT are only one stock-bond proxy pair; broader universes may differ.
# - Regime dependence can materially alter relative-flow behavior.
#
# ## Conclusion
#
# This notebook implements the requested mid-month laggard-rotation rule between
# SPY and TLT, then tracks outcomes for month-end holding periods.
#
# ## Next Research Ideas
#
# - Sweep trigger days (10-20) to test robustness.
# - Add volatility-scaling so both legs contribute similar risk.
# - Compare monthly close execution against next-open execution.
# - Extend to other stock/bond ETF pairs across regions.
