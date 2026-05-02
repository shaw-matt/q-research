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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from dotenv import load_dotenv
from IPython.display import Markdown, display

from research.massive_flatfiles import (
    align_btc_to_equity_close,
    build_equity_close_times,
    download_flatfile_btc_hourly_closes as download_btc_hourly,
    download_flatfile_stock_day_closes as download_equity_closes,
)
from research.plotting import apply_default_style

load_dotenv(dotenv_path=".env")
apply_default_style()

# %% [markdown]
# ## Assumptions
#
# - QQQ and UPRO daily closes from Massive US stock day-aggregate flat files
#   represent executable 4pm New York equity closes.
# - BTC trades continuously. The BTC price aligned to an equity session is the
#   latest available completed hourly X:BTC-USD close at or before that
#   session's 4pm New York close. Hourly series is built from Massive crypto
#   minute flat files, then aligned to equity closes the same way as before.
# - A signal observed at today's equity close is implemented at that close and
#   earns the following close-to-close UPRO return.
# - The beta estimate uses the most recent aligned BTC and QQQ returns available
#   at the signal close. The baseline parameters mirror the prompt: 40 days for
#   beta and 20 days for residual volatility.
# - Trading costs, financing, taxes, and closing-auction slippage are excluded.
#
# ## Data Sources
#
# - Massive S3 flat files: `us_stocks_sip/day_aggs_v1` for QQQ and UPRO daily OHLC.
# - Massive S3 flat files: global crypto `minute_aggs_v1` for X:BTC-USD, resampled to hourly.

# %%
START_DATE = "2023-01-01"
END_DATE = datetime.now(UTC).date().isoformat()

BETA_LOOKBACK_DAYS = 40
ZSCORE_LOOKBACK_DAYS = 20
ENTRY_ZSCORE = 1.5
ENTRY_ZSCORE_COMPARISON_GRID = [0.5, 1.0, 1.5, 2.0]
TRADE_NOTIONAL_USD = 10_000
TRADING_DAYS_PER_YEAR = 252
WALK_FORWARD_BETA_LOOKBACK_GRID = [20, 40, 60, 80]
WALK_FORWARD_ZSCORE_LOOKBACK_GRID = [10, 20, 30, 40]
WALK_FORWARD_ENTRY_ZSCORE_GRID = [1.0, 1.5, 2.0]
WALK_FORWARD_TRAIN_DAYS = 252
WALK_FORWARD_TEST_DAYS = 63
WALK_FORWARD_MIN_TRAIN_ACTIVE_DAYS = 5


def rolling_beta(y: pd.Series, x: pd.Series, lookback: int) -> pd.Series:
    """Estimate beta as rolling covariance(y, x) divided by rolling variance(x)."""
    covariance = y.rolling(lookback).cov(x)
    variance = x.rolling(lookback).var()
    return covariance / variance


def build_strategy_analysis(
    returns: pd.DataFrame,
    *,
    beta_lookback: int,
    zscore_lookback: int,
    entry_zscore: float,
) -> pd.DataFrame:
    """Build the residual z-score signal and shifted UPRO strategy returns."""
    frame = returns.copy()
    frame["beta"] = rolling_beta(
        frame["BTC_return"],
        frame["QQQ_return"],
        beta_lookback,
    )
    frame["residual"] = frame["BTC_return"] - frame["beta"] * frame["QQQ_return"]
    frame["residual_volatility"] = frame["residual"].rolling(zscore_lookback).std()
    frame["zscore"] = frame["residual"] / frame["residual_volatility"]
    frame["signal_at_close"] = frame["zscore"] >= entry_zscore
    frame["position"] = frame["signal_at_close"].shift(1).fillna(False).astype(bool)
    frame["strategy_return"] = np.where(frame["position"], frame["UPRO_return"], 0.0)
    frame["strategy_pnl_usd"] = TRADE_NOTIONAL_USD * frame["strategy_return"]
    frame["next_upro_return"] = frame["UPRO_return"].shift(-1)
    return frame.dropna(subset=["beta", "residual", "zscore"]).copy()


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


def summarize_strategy_window(frame: pd.DataFrame) -> dict[str, float]:
    """Return scalar performance metrics for a strategy window."""
    if frame.empty:
        return {
            "observations": 0,
            "active_days": 0,
            "exposure_rate": np.nan,
            "total_return": np.nan,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe_ratio": np.nan,
            "max_drawdown": np.nan,
            "active_day_win_rate": np.nan,
            "average_active_day_return": np.nan,
        }

    returns = frame["strategy_return"].dropna()
    positions = frame["position"].reindex(returns.index).fillna(False).astype(bool)
    active_returns = returns.loc[positions]
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

    return {
        "observations": len(returns),
        "active_days": len(active_returns),
        "exposure_rate": positions.mean() if len(positions) else np.nan,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown(returns) if len(returns) else np.nan,
        "active_day_win_rate": (active_returns > 0).mean() if len(active_returns) else np.nan,
        "average_active_day_return": active_returns.mean() if len(active_returns) else np.nan,
    }


def compare_entry_zscores(
    returns: pd.DataFrame,
    entry_zscores: list[float],
    *,
    beta_lookback: int,
    zscore_lookback: int,
) -> pd.DataFrame:
    """Compare fixed-lookback strategy performance across entry z-score thresholds."""
    rows = []
    for entry_zscore in entry_zscores:
        frame = build_strategy_analysis(
            returns,
            beta_lookback=beta_lookback,
            zscore_lookback=zscore_lookback,
            entry_zscore=entry_zscore,
        )
        metrics = summarize_strategy_window(frame)
        rows.append(
            {
                "entry_zscore": entry_zscore,
                "active_days": metrics["active_days"],
                "exposure_rate": metrics["exposure_rate"],
                "total_return_on_10k_notional": metrics["total_return"],
                "total_pnl_usd": metrics["total_return"] * TRADE_NOTIONAL_USD,
                "annualized_return": metrics["annualized_return"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "max_drawdown": metrics["max_drawdown"],
                "active_day_win_rate": metrics["active_day_win_rate"],
                "average_active_day_return": metrics["average_active_day_return"],
            }
        )

    return pd.DataFrame(rows)


def run_walk_forward_parameter_test(
    returns: pd.DataFrame,
    *,
    beta_lookbacks: list[int],
    zscore_lookbacks: list[int],
    entry_zscores: list[float],
    train_days: int,
    test_days: int,
    min_train_active_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Optimize parameters on rolling train windows and evaluate the next window."""
    strategy_frames = {}
    for beta_lookback in beta_lookbacks:
        for zscore_lookback in zscore_lookbacks:
            for entry_zscore in entry_zscores:
                key = (beta_lookback, zscore_lookback, entry_zscore)
                strategy_frames[key] = build_strategy_analysis(
                    returns,
                    beta_lookback=beta_lookback,
                    zscore_lookback=zscore_lookback,
                    entry_zscore=entry_zscore,
                )

    window_rows = []
    training_rank_rows = []
    oos_segments = []
    dates = returns.index
    window_id = 1
    test_start_position = train_days
    while test_start_position + test_days <= len(dates):
        train_index = dates[test_start_position - train_days : test_start_position]
        test_index = dates[test_start_position : test_start_position + test_days]
        train_start = train_index.min()
        train_end = train_index.max()
        test_start = test_index.min()
        test_end = test_index.max()

        candidate_rows = []
        for key, frame in strategy_frames.items():
            beta_lookback, zscore_lookback, entry_zscore = key
            train_frame = frame.reindex(train_index).dropna(subset=["strategy_return"])
            train_metrics = summarize_strategy_window(train_frame)
            score = (
                train_metrics["sharpe_ratio"]
                if train_metrics["active_days"] >= min_train_active_days
                else np.nan
            )
            candidate_rows.append(
                {
                    "window_id": window_id,
                    "beta_lookback": beta_lookback,
                    "zscore_lookback": zscore_lookback,
                    "entry_zscore": entry_zscore,
                    "train_start": train_start,
                    "train_end": train_end,
                    "train_score": score,
                    **{f"train_{key}": value for key, value in train_metrics.items()},
                }
            )

        train_rankings = pd.DataFrame(candidate_rows)
        train_rankings = train_rankings.sort_values(
            ["train_score", "train_total_return", "train_active_days"],
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)
        train_rankings["train_rank"] = np.arange(1, len(train_rankings) + 1)
        training_rank_rows.append(train_rankings)

        eligible = train_rankings.dropna(subset=["train_score"])
        if eligible.empty:
            selected = train_rankings.iloc[0]
        else:
            selected = eligible.iloc[0]

        selected_key = (
            int(selected["beta_lookback"]),
            int(selected["zscore_lookback"]),
            float(selected["entry_zscore"]),
        )
        test_frame = strategy_frames[selected_key].reindex(test_index).dropna(
            subset=["strategy_return"]
        )
        test_metrics = summarize_strategy_window(test_frame)
        if not test_frame.empty:
            oos_segments.append(
                test_frame[["strategy_return", "position"]].assign(window_id=window_id)
            )

        window_rows.append(
            {
                "window_id": window_id,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "beta_lookback": selected_key[0],
                "zscore_lookback": selected_key[1],
                "entry_zscore": selected_key[2],
                "train_score": selected["train_score"],
                "train_ranked_candidates": len(train_rankings),
                **{f"test_{key}": value for key, value in test_metrics.items()},
            }
        )

        window_id += 1
        test_start_position += test_days

    walk_forward_results = pd.DataFrame(window_rows)
    candidate_rankings = (
        pd.concat(training_rank_rows, ignore_index=True)
        if training_rank_rows
        else pd.DataFrame()
    )
    walk_forward_oos = (
        pd.concat(oos_segments).sort_index()
        if oos_segments
        else pd.DataFrame(columns=["strategy_return", "position", "window_id"])
    )
    return walk_forward_results, candidate_rankings, walk_forward_oos


def summarize_walk_forward_stability(
    walk_forward_results: pd.DataFrame,
    walk_forward_oos: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize selected-parameter dispersion and combined OOS performance."""
    parameter_columns = ["beta_lookback", "zscore_lookback", "entry_zscore"]
    if walk_forward_results.empty:
        empty_summary = pd.DataFrame(columns=["metric", "value"])
        empty_counts = pd.DataFrame(columns=parameter_columns + ["windows", "share"])
        return empty_summary, empty_counts

    parameter_counts = (
        walk_forward_results.value_counts(parameter_columns)
        .rename("windows")
        .reset_index()
        .sort_values(["windows", *parameter_columns], ascending=[False, True, True, True])
    )
    parameter_counts["share"] = parameter_counts["windows"] / len(walk_forward_results)
    most_common = parameter_counts.iloc[0]
    selected = walk_forward_results[parameter_columns]
    parameter_change_rate = (
        selected.ne(selected.shift()).any(axis=1).iloc[1:].mean()
        if len(selected) > 1
        else 0.0
    )
    oos_metrics = summarize_strategy_window(walk_forward_oos)

    stability_summary = pd.DataFrame(
        {
            "metric": [
                "walk_forward_windows",
                "unique_parameter_sets",
                "most_common_parameter_set",
                "most_common_parameter_share",
                "parameter_change_rate",
                "beta_lookback_min",
                "beta_lookback_median",
                "beta_lookback_max",
                "zscore_lookback_min",
                "zscore_lookback_median",
                "zscore_lookback_max",
                "entry_zscore_min",
                "entry_zscore_median",
                "entry_zscore_max",
                "oos_observations",
                "oos_active_days",
                "oos_exposure_rate",
                "oos_total_return_on_10k_notional",
                "oos_total_pnl_usd",
                "oos_annualized_return",
                "oos_annualized_volatility",
                "oos_sharpe_ratio",
                "oos_max_drawdown",
                "oos_active_day_win_rate",
            ],
            "value": [
                len(walk_forward_results),
                len(parameter_counts),
                (
                    f"beta={most_common['beta_lookback']}, "
                    f"zlookback={most_common['zscore_lookback']}, "
                    f"entry_z={most_common['entry_zscore']}"
                ),
                most_common["share"],
                parameter_change_rate,
                selected["beta_lookback"].min(),
                selected["beta_lookback"].median(),
                selected["beta_lookback"].max(),
                selected["zscore_lookback"].min(),
                selected["zscore_lookback"].median(),
                selected["zscore_lookback"].max(),
                selected["entry_zscore"].min(),
                selected["entry_zscore"].median(),
                selected["entry_zscore"].max(),
                oos_metrics["observations"],
                oos_metrics["active_days"],
                oos_metrics["exposure_rate"],
                oos_metrics["total_return"],
                oos_metrics["total_return"] * TRADE_NOTIONAL_USD,
                oos_metrics["annualized_return"],
                oos_metrics["annualized_volatility"],
                oos_metrics["sharpe_ratio"],
                oos_metrics["max_drawdown"],
                oos_metrics["active_day_win_rate"],
            ],
        }
    )
    return stability_summary, parameter_counts


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
# 8. Walk forward a grid of beta lookbacks, residual-volatility lookbacks, and
#    entry thresholds by choosing the best in-sample Sharpe over a one-year
#    window, then evaluating that choice over the next quarter.

# %%
equity_closes = download_equity_closes(["QQQ", "UPRO"], START_DATE, END_DATE)
btc_hourly_close = download_btc_hourly(START_DATE, END_DATE)

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

analysis = build_strategy_analysis(
    returns,
    beta_lookback=BETA_LOOKBACK_DAYS,
    zscore_lookback=ZSCORE_LOOKBACK_DAYS,
    entry_zscore=ENTRY_ZSCORE,
)

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

# %% [markdown]
# ### Entry Z-Score Comparison
#
# The table below keeps the baseline 40-day beta lookback and 20-day residual
# volatility lookback fixed, then compares strategy returns and Sharpe ratio at
# entry thresholds of 0.5, 1.0, 1.5, and 2.0.

# %%
entry_zscore_comparison = compare_entry_zscores(
    returns,
    ENTRY_ZSCORE_COMPARISON_GRID,
    beta_lookback=BETA_LOOKBACK_DAYS,
    zscore_lookback=ZSCORE_LOOKBACK_DAYS,
)
entry_zscore_comparison

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
# ## Walk-Forward Parameter Stability Test
#
# This test checks whether the baseline parameter choices are stable rather than
# treating one fixed lookback and threshold as settled. Each walk-forward window:
#
# 1. Builds the strategy for every parameter combination in the grid below.
# 2. Ranks combinations by in-sample Sharpe over the trailing one-year training
#    window, requiring at least five active training days to avoid selecting a
#    nearly dormant strategy.
# 3. Applies the selected parameters unchanged to the next quarter of returns.
#
# Stable parameters should be selected repeatedly across windows, show a low
# parameter-change rate, and retain reasonable out-of-sample performance.

# %%
walk_forward_results, walk_forward_candidate_rankings, walk_forward_oos = (
    run_walk_forward_parameter_test(
        returns,
        beta_lookbacks=WALK_FORWARD_BETA_LOOKBACK_GRID,
        zscore_lookbacks=WALK_FORWARD_ZSCORE_LOOKBACK_GRID,
        entry_zscores=WALK_FORWARD_ENTRY_ZSCORE_GRID,
        train_days=WALK_FORWARD_TRAIN_DAYS,
        test_days=WALK_FORWARD_TEST_DAYS,
        min_train_active_days=WALK_FORWARD_MIN_TRAIN_ACTIVE_DAYS,
    )
)
walk_forward_stability, walk_forward_parameter_counts = summarize_walk_forward_stability(
    walk_forward_results,
    walk_forward_oos,
)

walk_forward_stability

# %%
walk_forward_parameter_counts

# %%
walk_forward_window_summary_columns = [
    "window_id",
    "train_start",
    "train_end",
    "test_start",
    "test_end",
    "beta_lookback",
    "zscore_lookback",
    "entry_zscore",
    "train_score",
    "test_active_days",
    "test_total_return",
    "test_sharpe_ratio",
    "test_max_drawdown",
]
walk_forward_results.reindex(columns=walk_forward_window_summary_columns)

# %%
if not walk_forward_results.empty:
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    plot_frame = walk_forward_results.set_index("test_start")
    plot_frame["beta_lookback"].plot(ax=axes[0], marker="o")
    axes[0].set_title("Selected beta lookback by walk-forward test window")
    axes[0].set_ylabel("Days")
    plot_frame["zscore_lookback"].plot(ax=axes[1], marker="o", color="tab:orange")
    axes[1].set_title("Selected residual-volatility lookback")
    axes[1].set_ylabel("Days")
    plot_frame["entry_zscore"].plot(ax=axes[2], marker="o", color="tab:green")
    axes[2].set_title("Selected entry z-score")
    axes[2].set_ylabel("Z-score")
    axes[2].set_xlabel("Out-of-sample test start")
    fig.tight_layout()
    plt.show()

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
# - BTC hourly marks are derived from Massive crypto minute flat files; raw
#   tick or sub-minute data would still capture more precise marks.
# - The backtest assumes closing-price execution for UPRO, no spread/slippage, no
#   borrow or financing effects, and no tax impact.
# - UPRO is a 3x S&P 500 ETF, not a Nasdaq ETF. If the intended instrument is
#   TQQQ, the same process can be rerun by replacing `UPRO` with `TQQQ`.
# - The walk-forward test only covers the parameter grid, one-year training
#   windows, and quarterly test windows shown above. Different grids or scoring
#   metrics may select different parameters.
# - Multiple hypothesis testing is not adjusted for; statistical significance
#   should be treated as exploratory rather than conclusive.
#
# ## Conclusion
#
# This notebook implements the requested BTC/QQQ aligned-return residual process
# and evaluates the rule that buys $10k of UPRO at the equity close when the
# residual z-score is at least +1.5. The backtest tables report performance,
# trade-level outcomes, several significance diagnostics, and a walk-forward
# parameter-stability check so the signal can be assessed beyond the raw equity
# curve.
#
# ## Next Research Ideas
#
# - Compare resampled hourly BTC with exchange tick or sub-minute marks
#   sampled exactly at 4pm New York.
# - Compare UPRO with TQQQ and QQQ to separate leverage effects from index choice.
# - Try alternative walk-forward objective functions such as Sortino ratio,
#   average active-day return, or alpha versus UPRO.
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

display(current_decision)
display(current_signal_state)

# %%
plot_current_signal_dashboard(analysis, prices)
