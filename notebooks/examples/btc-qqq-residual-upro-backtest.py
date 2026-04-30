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
# # BTC/QQQ Residual Z-Score UPRO Backtest
#
# ## Research Question
#
# Does a positive BTC return residual, measured after hedging recent BTC returns
# by QQQ returns aligned to the U.S. equity close, identify next-day long UPRO
# opportunities?
#
# ## Hypothesis
#
# If BTC has a statistically meaningful positive residual versus QQQ at the 4pm
# New York equity close, then leveraged Nasdaq exposure through UPRO may earn
# positive close-to-close returns while the residual z-score remains elevated.

# %%
from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import yfinance as yf
from IPython.display import Markdown, display

from research.plotting import apply_default_style

apply_default_style()

# %% [markdown]
# ## Assumptions
#
# - QQQ and UPRO daily adjusted closes represent executable 4pm New York equity
#   closes.
# - BTC trades continuously. The BTC price aligned to an equity session is the
#   latest available completed hourly BTC-USD close at or before that session's
#   4pm New York close.
# - A signal observed at today's equity close is implemented at that close and
#   earns the following close-to-close UPRO return.
# - The beta estimate uses the most recent aligned BTC and QQQ returns available
#   at the signal close. The baseline parameters mirror the prompt: 40 days for
#   beta and 20 days for residual volatility.
# - Trading costs, financing, taxes, and closing-auction slippage are excluded.
#
# ## Data Sources
#
# - Yahoo Finance via `yfinance` for QQQ and UPRO adjusted daily closes.
# - Yahoo Finance via `yfinance` for BTC-USD hourly closes.

# %%
START_DATE = "2023-01-01"
END_DATE = datetime.now(UTC).date().isoformat()

BETA_LOOKBACK_DAYS = 40
ZSCORE_LOOKBACK_DAYS = 20
ENTRY_ZSCORE = 1.5
TRADE_NOTIONAL_USD = 10_000
TRADING_DAYS_PER_YEAR = 252
NY_TZ = ZoneInfo("America/New_York")


def extract_close(download: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Normalize yfinance output into one close-price column per ticker."""
    if download.empty:
        return pd.DataFrame(columns=tickers)

    if isinstance(download.columns, pd.MultiIndex):
        if "Close" in download.columns.get_level_values(0):
            close = download["Close"]
        else:
            close = download.xs("Close", level=1, axis=1)
    else:
        close = download[["Close"]].rename(columns={"Close": tickers[0]})

    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])

    return close.rename_axis("date").sort_index()


def download_equity_closes(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download adjusted daily equity closes."""
    raw = yf.download(
        tickers,
        start=start_date,
        end=pd.Timestamp(end_date) + pd.Timedelta(days=1),
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    closes = extract_close(raw, tickers)
    closes.index = pd.to_datetime(closes.index).tz_localize(None).normalize()
    return closes.dropna(how="all")


def download_btc_hourly(start_date: str) -> pd.Series:
    """Download recent hourly BTC-USD closes for equity-close alignment."""
    raw = yf.download(
        "BTC-USD",
        period="730d",
        interval="1h",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    closes = extract_close(raw, ["BTC-USD"])["BTC-USD"].dropna()
    if closes.empty:
        return closes

    index = pd.to_datetime(closes.index)
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")

    # Yahoo labels hourly bars by interval start; move labels to interval close
    # before sampling at the equity close to avoid lookahead.
    closes.index = index + pd.Timedelta(hours=1)
    return closes.loc[closes.index >= pd.Timestamp(start_date, tz="UTC")]


def build_equity_close_times(equity_dates: pd.Index) -> pd.Series:
    """Map each equity trading date to 4pm New York time in UTC."""
    close_times = []
    for session_date in pd.to_datetime(equity_dates).date:
        close_time = datetime.combine(session_date, datetime.min.time(), NY_TZ)
        close_time = close_time.replace(hour=16)
        close_times.append(pd.Timestamp(close_time).tz_convert("UTC"))

    return pd.Series(close_times, index=pd.to_datetime(equity_dates), name="equity_close_utc")


def align_btc_to_equity_close(
    btc_hourly_close: pd.Series,
    equity_close_times: pd.Series,
) -> pd.Series:
    """Sample BTC at the latest completed hourly close at or before each equity close."""
    aligned = btc_hourly_close.reindex(
        pd.DatetimeIndex(equity_close_times),
        method="ffill",
        tolerance=pd.Timedelta(hours=3),
    )
    aligned.index = equity_close_times.index
    return aligned.rename("btc_close_at_equity_close")


def rolling_beta(y: pd.Series, x: pd.Series, lookback: int) -> pd.Series:
    """Estimate beta as rolling covariance(y, x) divided by rolling variance(x)."""
    covariance = y.rolling(lookback).cov(x)
    variance = x.rolling(lookback).var()
    return covariance / variance


def max_drawdown(return_series: pd.Series) -> float:
    """Calculate max drawdown from a daily return series."""
    equity_curve = (1 + return_series.fillna(0)).cumprod()
    drawdown = equity_curve / equity_curve.cummax() - 1
    return float(drawdown.min())


def summarize_backtest(strategy_returns: pd.Series, positions: pd.Series) -> pd.DataFrame:
    """Summarize close-to-close strategy performance."""
    returns = strategy_returns.dropna()
    active_returns = returns.loc[positions.reindex(returns.index).fillna(False)]
    equity_curve = (1 + returns).cumprod()
    total_return = equity_curve.iloc[-1] - 1 if not equity_curve.empty else np.nan
    annualized_return = (
        equity_curve.iloc[-1] ** (TRADING_DAYS_PER_YEAR / len(returns)) - 1
        if len(returns) and equity_curve.iloc[-1] > 0
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
                "total_return_on_10k_notional",
                "total_pnl_usd",
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
                total_return * TRADE_NOTIONAL_USD,
                annualized_return,
                annualized_volatility,
                sharpe,
                max_drawdown(returns),
                (active_returns > 0).mean() if len(active_returns) else np.nan,
                active_returns.mean() if len(active_returns) else np.nan,
            ],
        }
    )


def compute_trade_returns(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse consecutive held days into per-trade UPRO returns."""
    held = frame.loc[frame["position"], ["strategy_return"]].copy()
    if held.empty:
        return pd.DataFrame(columns=["entry_date", "exit_date", "holding_days", "trade_return"])

    trade_id = (~frame["position"].shift(fill_value=False) & frame["position"]).cumsum()
    held["trade_id"] = trade_id.loc[held.index]
    trades = (
        held.groupby("trade_id")
        .agg(
            entry_date=("strategy_return", lambda values: values.index.min()),
            exit_date=("strategy_return", lambda values: values.index.max()),
            holding_days=("strategy_return", "size"),
            trade_return=("strategy_return", lambda values: (1 + values).prod() - 1),
        )
        .reset_index(drop=True)
    )
    return trades


def significance_tests(frame: pd.DataFrame) -> pd.DataFrame:
    """Estimate simple significance metrics for strategy returns and signal quality."""
    strategy_returns = frame["strategy_return"].dropna()
    active_returns = frame.loc[frame["position"], "strategy_return"].dropna()
    signal_sample = frame[["zscore", "next_upro_return"]].dropna()

    if len(strategy_returns) >= 3:
        plain_t = stats.ttest_1samp(strategy_returns, popmean=0, nan_policy="omit")
        hac_model = sm.OLS(
            strategy_returns.to_numpy(),
            np.ones((len(strategy_returns), 1)),
        ).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        hac_t = float(hac_model.tvalues[0])
        hac_p = float(hac_model.pvalues[0])
    else:
        plain_t = None
        hac_t = np.nan
        hac_p = np.nan

    if len(active_returns):
        hit_test = stats.binomtest(int((active_returns > 0).sum()), len(active_returns), p=0.5)
    else:
        hit_test = None

    if len(signal_sample) >= 3:
        information_coefficient = stats.spearmanr(
            signal_sample["zscore"],
            signal_sample["next_upro_return"],
            nan_policy="omit",
        )
    else:
        information_coefficient = None

    alpha_t = np.nan
    alpha_p = np.nan
    annualized_alpha = np.nan
    regression_sample = frame[["strategy_return", "UPRO_return"]].dropna()
    if len(regression_sample) >= 20:
        model = sm.OLS(
            regression_sample["strategy_return"],
            sm.add_constant(regression_sample["UPRO_return"]),
        ).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        annualized_alpha = float(model.params["const"] * TRADING_DAYS_PER_YEAR)
        alpha_t = float(model.tvalues["const"])
        alpha_p = float(model.pvalues["const"])

    return pd.DataFrame(
        {
            "metric": [
                "mean_daily_return_t_stat",
                "mean_daily_return_p_value",
                "hac_mean_daily_return_t_stat",
                "hac_mean_daily_return_p_value",
                "active_hit_rate_binomial_p_value",
                "signal_next_return_spearman_ic",
                "signal_next_return_spearman_p_value",
                "annualized_alpha_vs_upro",
                "hac_alpha_t_stat_vs_upro",
                "hac_alpha_p_value_vs_upro",
            ],
            "value": [
                plain_t.statistic if plain_t is not None else np.nan,
                plain_t.pvalue if plain_t is not None else np.nan,
                hac_t,
                hac_p,
                hit_test.pvalue if hit_test is not None else np.nan,
                information_coefficient.statistic if information_coefficient is not None else np.nan,
                information_coefficient.pvalue if information_coefficient is not None else np.nan,
                annualized_alpha,
                alpha_t,
                alpha_p,
            ],
        }
    )


def format_dashboard_value(value: object, style: str) -> str:
    """Format dashboard values for quick end-of-day scanning."""
    if pd.isna(value):
        return "n/a"
    if style == "percent":
        return f"{value:.2%}"
    if style == "number":
        return f"{value:,.2f}"
    if style == "signed_number":
        return f"{value:+,.2f}"
    if style == "currency":
        return f"${value:,.2f}"
    if style == "integer":
        return f"{value:,.0f}"
    return str(value)


def classify_close_action(signal_at_close: bool, current_position: bool) -> str:
    """Translate today's close signal and current holding state into a trade action."""
    if signal_at_close and current_position:
        return "HOLD long UPRO"
    if signal_at_close and not current_position:
        return "ENTER long UPRO"
    if not signal_at_close and current_position:
        return "EXIT long UPRO"
    return "STAY FLAT"


def build_current_signal_dashboard(
    frame: pd.DataFrame,
    price_frame: pd.DataFrame,
    close_times: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build end-of-day decision and signal-state tables from the latest row."""
    latest_date = frame.index.max()
    latest = frame.loc[latest_date]
    latest_prices = price_frame.loc[latest_date]
    previous_zscore = frame["zscore"].shift(1).loc[latest_date]
    zscore_gap = latest["zscore"] - ENTRY_ZSCORE
    action = classify_close_action(
        bool(latest["signal_at_close"]),
        bool(latest["position"]),
    )
    target_notional = TRADE_NOTIONAL_USD if bool(latest["signal_at_close"]) else 0
    target_shares = (
        target_notional / latest_prices["UPRO"]
        if target_notional and latest_prices["UPRO"] > 0
        else 0
    )

    decision_table = pd.DataFrame(
        {
            "item": [
                "latest_equity_session",
                "equity_close_time_utc",
                "end_of_day_action",
                "signal_for_next_session",
                "position_during_latest_session",
                "target_upro_notional",
                "approx_upro_shares_at_close",
            ],
            "value": [
                latest_date.strftime("%Y-%m-%d"),
                close_times.loc[latest_date].strftime("%Y-%m-%d %H:%M %Z"),
                action,
                "ON" if bool(latest["signal_at_close"]) else "OFF",
                "LONG" if bool(latest["position"]) else "FLAT",
                format_dashboard_value(target_notional, "currency"),
                format_dashboard_value(target_shares, "number"),
            ],
        }
    )

    signal_table = pd.DataFrame(
        {
            "metric": [
                "current_zscore",
                "entry_threshold",
                "zscore_margin_to_threshold",
                "prior_session_zscore",
                "btc_return_at_equity_close",
                "qqq_return",
                "rolling_beta_btc_vs_qqq",
                "residual_return",
                "residual_volatility",
                "upro_close",
                "btc_close_at_equity_close",
            ],
            "value": [
                format_dashboard_value(latest["zscore"], "signed_number"),
                format_dashboard_value(ENTRY_ZSCORE, "number"),
                format_dashboard_value(zscore_gap, "signed_number"),
                format_dashboard_value(previous_zscore, "signed_number"),
                format_dashboard_value(latest["BTC_return"], "percent"),
                format_dashboard_value(latest["QQQ_return"], "percent"),
                format_dashboard_value(latest["beta"], "number"),
                format_dashboard_value(latest["residual"], "percent"),
                format_dashboard_value(latest["residual_volatility"], "percent"),
                format_dashboard_value(latest_prices["UPRO"], "currency"),
                format_dashboard_value(latest_prices["btc_close_at_equity_close"], "currency"),
            ],
        }
    )
    return decision_table, signal_table


def plot_current_signal_dashboard(
    frame: pd.DataFrame,
    price_frame: pd.DataFrame,
    lookback_days: int = 90,
) -> None:
    """Plot the latest signal state in the context of recent history."""
    recent = frame.tail(lookback_days)
    recent_prices = price_frame.reindex(recent.index)
    latest_date = recent.index.max()
    latest = recent.loc[latest_date]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=False)
    z_ax, residual_ax, price_ax, beta_ax = axes.ravel()

    recent["zscore"].plot(ax=z_ax, color="tab:blue", label="Residual z-score")
    z_ax.axhline(ENTRY_ZSCORE, color="tab:red", linestyle="--", label=f"entry {ENTRY_ZSCORE}")
    z_ax.axhline(0, color="black", linewidth=1)
    z_ax.scatter(
        [latest_date],
        [latest["zscore"]],
        color="tab:orange" if bool(latest["signal_at_close"]) else "tab:gray",
        zorder=5,
        label="latest close",
    )
    z_ax.set_title("Current residual z-score versus entry threshold")
    z_ax.set_ylabel("Z-score")
    z_ax.legend(loc="best")

    residual_components = pd.DataFrame(
        {
            "BTC return": recent["BTC_return"],
            "Beta-adjusted QQQ return": recent["beta"] * recent["QQQ_return"],
            "Residual": recent["residual"],
        }
    ).tail(30)
    residual_components.plot(ax=residual_ax)
    residual_ax.axhline(0, color="black", linewidth=1)
    residual_ax.set_title("Recent return inputs behind the residual")
    residual_ax.set_ylabel("Daily return")
    residual_ax.legend(loc="best")

    recent_prices["UPRO"].plot(ax=price_ax, color="tab:green", label="UPRO close")
    signal_dates = recent.index[recent["signal_at_close"]]
    for signal_date in signal_dates:
        price_ax.axvspan(signal_date, signal_date + pd.Timedelta(days=1), color="tab:green", alpha=0.08)
    price_ax.scatter([latest_date], [recent_prices.loc[latest_date, "UPRO"]], color="tab:orange", zorder=5)
    price_ax.set_title("UPRO close with signal-on sessions shaded")
    price_ax.set_ylabel("Price ($)")
    price_ax.legend(loc="best")

    recent["beta"].plot(ax=beta_ax, color="tab:purple", label="Rolling beta")
    beta_ax_twin = beta_ax.twinx()
    recent["residual_volatility"].plot(
        ax=beta_ax_twin,
        color="tab:brown",
        linestyle="--",
        label="Residual volatility",
    )
    beta_ax.set_title("Rolling beta and residual volatility")
    beta_ax.set_ylabel("Beta")
    beta_ax_twin.set_ylabel("Residual volatility")
    beta_lines, beta_labels = beta_ax.get_legend_handles_labels()
    vol_lines, vol_labels = beta_ax_twin.get_legend_handles_labels()
    beta_ax.legend(beta_lines + vol_lines, beta_labels + vol_labels, loc="best")

    fig.suptitle("End-of-Day BTC/QQQ Residual Signal Dashboard", y=1.02)
    fig.tight_layout()
    plt.show()

# %% [markdown]
# ## Methodology
#
# 1. Download QQQ and UPRO adjusted daily closes.
# 2. Download BTC-USD hourly closes and align BTC to each QQQ equity close:
#    previous 4pm New York equity close to today's 4pm New York equity close.
# 3. Calculate aligned daily returns for BTC and QQQ.
# 4. Estimate rolling beta:
#
#    `beta = covariance(BTC_returns, QQQ_returns) / variance(QQQ_returns)`
#
# 5. Calculate the residual:
#
#    `residual = BTC_return - beta * QQQ_return`
#
# 6. Standardize residuals with a rolling 20-day standard deviation:
#
#    `z = residual / stdev(residual)`
#
# 7. If `z >= +1.5`, enter or keep a long $10k UPRO position at the equity close.
#    If the next close has `z < +1.5`, close the position at that close.

# %%
equity_closes = download_equity_closes(["QQQ", "UPRO"], START_DATE, END_DATE)
btc_hourly_close = download_btc_hourly(START_DATE)

if equity_closes.empty:
    raise ValueError("No QQQ/UPRO equity closes were downloaded.")
if btc_hourly_close.empty:
    raise ValueError("No BTC-USD hourly closes were downloaded.")

equity_close_times = build_equity_close_times(equity_closes.index)
btc_close = align_btc_to_equity_close(btc_hourly_close, equity_close_times)

prices = equity_closes.join(btc_close, how="inner").dropna()
prices.tail()

# %%
returns = prices.rename(columns={"btc_close_at_equity_close": "BTC"}).pct_change()
returns = returns.rename(
    columns={
        "QQQ": "QQQ_return",
        "UPRO": "UPRO_return",
        "BTC": "BTC_return",
    }
).dropna()

analysis = returns.copy()
analysis["beta"] = rolling_beta(
    analysis["BTC_return"],
    analysis["QQQ_return"],
    BETA_LOOKBACK_DAYS,
)
analysis["residual"] = analysis["BTC_return"] - analysis["beta"] * analysis["QQQ_return"]
analysis["residual_volatility"] = analysis["residual"].rolling(ZSCORE_LOOKBACK_DAYS).std()
analysis["zscore"] = analysis["residual"] / analysis["residual_volatility"]
analysis["signal_at_close"] = analysis["zscore"] >= ENTRY_ZSCORE
analysis["position"] = analysis["signal_at_close"].shift(1).fillna(False).astype(bool)
analysis["strategy_return"] = np.where(analysis["position"], analysis["UPRO_return"], 0.0)
analysis["strategy_pnl_usd"] = TRADE_NOTIONAL_USD * analysis["strategy_return"]
analysis["next_upro_return"] = analysis["UPRO_return"].shift(-1)
analysis = analysis.dropna(subset=["beta", "residual", "zscore"]).copy()

analysis.tail(10)

# %% [markdown]
# ## Analysis
#
# The table below shows the most recent signal inputs. `position` is shifted by
# one close because the signal at today's close earns tomorrow's close-to-close
# UPRO return.

# %%
latest_signal = analysis[
    [
        "BTC_return",
        "QQQ_return",
        "beta",
        "residual",
        "residual_volatility",
        "zscore",
        "signal_at_close",
        "position",
        "UPRO_return",
        "strategy_pnl_usd",
    ]
].tail(15)

latest_signal

# %%
backtest_summary = summarize_backtest(analysis["strategy_return"], analysis["position"])
backtest_summary

# %%
trades = compute_trade_returns(analysis)
trade_summary = pd.DataFrame(
    {
        "metric": [
            "trades",
            "winning_trades",
            "trade_win_rate",
            "average_trade_return",
            "median_trade_return",
            "best_trade_return",
            "worst_trade_return",
            "average_holding_days",
        ],
        "value": [
            len(trades),
            int((trades["trade_return"] > 0).sum()) if len(trades) else 0,
            (trades["trade_return"] > 0).mean() if len(trades) else np.nan,
            trades["trade_return"].mean() if len(trades) else np.nan,
            trades["trade_return"].median() if len(trades) else np.nan,
            trades["trade_return"].max() if len(trades) else np.nan,
            trades["trade_return"].min() if len(trades) else np.nan,
            trades["holding_days"].mean() if len(trades) else np.nan,
        ],
    }
)
trade_summary

# %%
trades.tail(10)

# %%
significance = significance_tests(analysis)
significance

# %% [markdown]
# Statistical metrics:
#
# - `mean_daily_return_t_stat` and p-value test whether the strategy's average
#   daily return differs from zero.
# - HAC metrics repeat the mean-return test with Newey-West style robust standard
#   errors to reduce sensitivity to short-term autocorrelation.
# - The binomial p-value tests whether active held days win more often than a
#   50/50 process.
# - The Spearman information coefficient tests whether larger signal z-scores
#   are monotonically associated with higher next-day UPRO returns.
# - Alpha metrics regress strategy returns against UPRO daily returns and report
#   robust intercept significance.

# %% [markdown]
# ## Visualizations

# %%
fig, ax = plt.subplots()
prices[["QQQ", "UPRO", "btc_close_at_equity_close"]].div(
    prices[["QQQ", "UPRO", "btc_close_at_equity_close"]].iloc[0]
).plot(ax=ax)
ax.set_title("Normalized QQQ, UPRO, and BTC-at-Equity-Close Prices")
ax.set_xlabel("Date")
ax.set_ylabel("Growth of $1")
plt.show()

# %%
fig, ax = plt.subplots()
analysis["zscore"].plot(ax=ax, label="BTC residual z-score")
ax.axhline(ENTRY_ZSCORE, color="tab:red", linestyle="--", label=f"entry threshold ({ENTRY_ZSCORE})")
ax.axhline(0, color="black", linewidth=1)
ax.set_title("BTC Residual Z-Score vs QQQ")
ax.set_xlabel("Date")
ax.set_ylabel("Z-score")
ax.legend()
plt.show()

# %%
strategy_equity = (1 + analysis["strategy_return"]).cumprod()
upro_equity = (1 + analysis["UPRO_return"]).cumprod()

fig, ax = plt.subplots()
strategy_equity.plot(ax=ax, label="Residual signal strategy")
upro_equity.plot(ax=ax, label="Buy and hold UPRO")
ax.set_title("Backtest Equity Curve")
ax.set_xlabel("Date")
ax.set_ylabel("Growth of $1 notional")
ax.legend()
plt.show()

# %%
fig, ax = plt.subplots()
(TRADE_NOTIONAL_USD * analysis["strategy_return"].cumsum()).plot(ax=ax)
ax.set_title("Cumulative Strategy P&L on $10k UPRO Notional")
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative P&L ($)")
plt.show()

# %% [markdown]
# ## Limitations
#
# - Yahoo's hourly BTC bars are an approximation of exact 4pm New York BTC marks;
#   higher-quality exchange tick or minute data would improve alignment.
# - The backtest assumes closing-price execution for UPRO, no spread/slippage, no
#   borrow or financing effects, and no tax impact.
# - UPRO is a 3x S&P 500 ETF, not a Nasdaq ETF. If the intended instrument is
#   TQQQ, the same process can be rerun by replacing `UPRO` with `TQQQ`.
# - Rolling beta and z-score parameters are fixed at 40 and 20 days. Results may
#   be sensitive to these choices and to the sampled market regime.
# - Multiple hypothesis testing is not adjusted for; statistical significance
#   should be treated as exploratory rather than conclusive.
#
# ## Conclusion
#
# This notebook implements the requested BTC/QQQ aligned-return residual process
# and evaluates the rule that buys $10k of UPRO at the equity close when the
# residual z-score is at least +1.5. The backtest tables report performance,
# trade-level outcomes, and several significance diagnostics so the signal can be
# assessed beyond the raw equity curve.
#
# ## Next Research Ideas
#
# - Replace Yahoo hourly BTC data with exchange minute bars sampled exactly at
#   4pm New York.
# - Compare UPRO with TQQQ and QQQ to separate leverage effects from index choice.
# - Walk forward the beta and z-score lookbacks to test parameter stability.
# - Add realistic closing-auction slippage and transaction-cost assumptions.

# %% [markdown]
# ## Current Signal Dashboard
#
# Use this final section after the market close to decide whether the strategy
# calls for a UPRO trade for the next close-to-close holding interval. The action
# compares today's close signal with the position that was active during the most
# recent completed session.

# %%
current_decision, current_signal_state = build_current_signal_dashboard(
    analysis,
    prices,
    equity_close_times,
)
latest_action = current_decision.loc[
    current_decision["item"] == "end_of_day_action",
    "value",
].iloc[0]
latest_signal_state = current_decision.loc[
    current_decision["item"] == "signal_for_next_session",
    "value",
].iloc[0]
latest_session = current_decision.loc[
    current_decision["item"] == "latest_equity_session",
    "value",
].iloc[0]

display(
    Markdown(
        f"### End-of-day decision for {latest_session}: **{latest_action}**\n\n"
        f"Signal for the next close-to-close interval: **{latest_signal_state}**."
    )
)

display(current_decision.style.hide(axis="index"))
display(current_signal_state.style.hide(axis="index"))

# %%
plot_current_signal_dashboard(analysis, prices)
