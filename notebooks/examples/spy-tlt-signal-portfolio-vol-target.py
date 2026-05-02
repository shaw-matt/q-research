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
# # Vol-Targeted Optimized Portfolio of SPY/TLT Signals
#
# ## Research Question
#
# If we combine the SPY/TLT calendar and relative-value signals with the
# BTC/QQQ residual long-UPRO rule into one portfolio, can optimized weights
# plus a volatility target improve risk-adjusted performance?
#
# ## Hypothesis
#
# The four signals likely have different timing and risk profiles. A
# constrained optimized blend should diversify signal noise, and a volatility
# target should stabilize drawdowns and annualized risk.

# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.optimize import minimize

from research.data import download_massive_daily_closes
from research.plotting import apply_default_style
from research.stats import annualized_turnover_one_way, mean_daily_turnover_one_way
from research.upro_residual import build_upro_residual_strategy_frame

load_dotenv(dotenv_path=".env")
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
# - Weight optimization is done in-sample only, then evaluated out-of-sample.
# - Vol targeting uses rolling realized volatility and next-day scaling.
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
TRAIN_FRACTION = 0.60
TARGET_ANNUAL_VOL = 0.10
VOL_LOOKBACK_DAYS = 20
MAX_LEVERAGE = 2.5
EOM_TRIGGER_DAY = 15
RELATIVE_REVERSAL_LOOKBACK = 5
TURN_OF_MONTH_WINDOW = 5
BETA_LOOKBACK_DAYS = 40
ZSCORE_LOOKBACK_DAYS = 20
ENTRY_ZSCORE = 1.5


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


def optimize_sharpe_weights(returns_frame: pd.DataFrame) -> pd.Series:
    clean = returns_frame.dropna()
    if clean.empty:
        raise ValueError("Cannot optimize weights with an empty returns frame.")

    assets = clean.columns.tolist()
    n_assets = len(assets)
    initial = np.repeat(1 / n_assets, n_assets)

    def objective(weights: np.ndarray) -> float:
        portfolio = clean.to_numpy() @ weights
        vol = portfolio.std()
        if vol <= 0:
            return 1e6
        sharpe = portfolio.mean() / vol * np.sqrt(TRADING_DAYS_PER_YEAR)
        return -sharpe

    bounds = [(0.0, 1.0)] * n_assets
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    result = minimize(objective, x0=initial, bounds=bounds, constraints=constraints, method="SLSQP")
    if not result.success:
        raise RuntimeError(f"Weight optimization failed: {result.message}")
    return pd.Series(result.x, index=assets, name="weight")


def build_eom_rebalance_returns(
    prices: pd.DataFrame, trigger_day: int = 15
) -> tuple[pd.Series, pd.DataFrame]:
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
    exposure = pd.DataFrame(
        {
            "SPY": (frame["position_asset"] == "SPY").astype(float),
            "TLT": (frame["position_asset"] == "TLT").astype(float),
            "UPRO": 0.0,
        },
        index=frame.index,
    )
    return frame["strategy_return"].rename("eom_rebalance"), exposure


def build_relative_reversal_returns(
    prices: pd.DataFrame, lookback_days: int = 5
) -> tuple[pd.Series, pd.DataFrame]:
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
    exposure = pd.DataFrame(
        {
            "SPY": (frame["position_asset"] == "SPY").astype(float),
            "TLT": (frame["position_asset"] == "TLT").astype(float),
            "UPRO": 0.0,
        },
        index=frame.index,
    )
    return frame["strategy_return"].rename("relative_reversal"), exposure


def build_turn_of_month_tlt_returns(
    prices: pd.DataFrame, window_days: int = 5
) -> tuple[pd.Series, pd.DataFrame]:
    frame = prices.copy()
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

    frame["TLT_return"] = frame["TLT"].pct_change()
    frame["position"] = frame["position_signal"].shift(1).fillna(0)
    frame["strategy_return"] = frame["position"] * frame["TLT_return"]
    exposure = pd.DataFrame(
        {
            "SPY": 0.0,
            "TLT": frame["position"].astype(float),
            "UPRO": 0.0,
        },
        index=frame.index,
    )
    return frame["strategy_return"].rename("tlt_turn_of_month"), exposure


def blend_signal_exposures(
    per_signal: pd.DataFrame,
    blend_weights: pd.Series,
) -> pd.DataFrame:
    """Net SPY / TLT / UPRO weights per $1 of blended signal capital (pre vol overlay)."""
    assets = ["SPY", "TLT", "UPRO"]
    out = pd.DataFrame(0.0, index=per_signal.index, columns=assets)
    for sig in blend_weights.index:
        w = float(blend_weights[sig])
        for a in assets:
            out[a] = out[a] + w * per_signal[(sig, a)]
    return out


def gross_exposure_shares(net: pd.DataFrame) -> pd.DataFrame:
    """100% stacked shares: SPY long, TLT long, TLT short, UPRO long, cash/flat."""
    spy = net["SPY"]
    tlt = net["TLT"]
    upro = net["UPRO"]
    c_spy = spy.clip(lower=0.0)
    c_tlt_long = tlt.clip(lower=0.0)
    c_tlt_short = (-tlt).clip(lower=0.0)
    c_upro = upro.clip(lower=0.0)
    gross = c_spy + c_tlt_long + c_tlt_short + c_upro
    denom = gross.replace(0.0, np.nan)
    parts = pd.DataFrame(
        {
            "SPY": (c_spy / denom).fillna(0.0),
            "TLT long": (c_tlt_long / denom).fillna(0.0),
            "TLT short": (c_tlt_short / denom).fillna(0.0),
            "UPRO": (c_upro / denom).fillna(0.0),
        },
        index=net.index,
    )
    cash = (gross <= 0.0).astype(float)
    parts = parts.mul((gross > 0.0).astype(float), axis=0)
    parts["Cash / flat"] = cash
    return parts


def plot_exposure_mix(ax, shares: pd.DataFrame, title: str) -> None:
    idx = shares.index
    series = [shares[c].to_numpy() for c in shares.columns]
    ax.stackplot(idx, *series, labels=shares.columns, alpha=0.88)
    ax.set_title(title)
    ax.set_ylabel("Share of gross exposure")
    ax.set_xlabel("Date")
    ax.set_ylim(0.0, 1.0)
    ax.margins(x=0)
    ax.legend(loc="upper left", fontsize=8)


def apply_vol_target(
    return_series: pd.Series,
    *,
    target_annual_vol: float,
    lookback_days: int,
    max_leverage: float,
) -> pd.DataFrame:
    frame = pd.DataFrame({"base_return": return_series})
    realized_vol = frame["base_return"].rolling(lookback_days).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    raw_leverage = target_annual_vol / realized_vol
    frame["leverage"] = raw_leverage.replace([np.inf, -np.inf], np.nan).clip(upper=max_leverage)
    frame["leverage"] = frame["leverage"].fillna(1.0).shift(1).fillna(1.0)
    frame["vol_targeted_return"] = frame["base_return"] * frame["leverage"]
    frame["realized_annual_vol"] = realized_vol
    return frame


prices = download_massive_daily_closes(["SPY", "TLT"], start_date=START_DATE).dropna()
if prices.empty:
    raise ValueError("No SPY/TLT daily prices were downloaded.")

eom_ret, eom_exp = build_eom_rebalance_returns(prices, trigger_day=EOM_TRIGGER_DAY)
rel_ret, rel_exp = build_relative_reversal_returns(prices, lookback_days=RELATIVE_REVERSAL_LOOKBACK)
tom_ret, tom_exp = build_turn_of_month_tlt_returns(prices, window_days=TURN_OF_MONTH_WINDOW)
spy_tlt_returns = pd.concat([eom_ret, rel_ret, tom_ret], axis=1)

upro_frame = build_upro_residual_strategy_frame(
    start_date=START_DATE,
    beta_lookback=BETA_LOOKBACK_DAYS,
    zscore_lookback=ZSCORE_LOOKBACK_DAYS,
    entry_zscore=ENTRY_ZSCORE,
)
upro_returns = upro_frame["strategy_return"].rename("upro_residual")

signal_returns = spy_tlt_returns.join(upro_returns, how="inner").dropna()
if signal_returns.empty:
    raise ValueError("No overlapping history after joining SPY/TLT signals with UPRO residual.")

_signal_cols = list(signal_returns.columns)
per_signal_exposure = pd.concat(
    [
        eom_exp.reindex(signal_returns.index).fillna(0.0),
        rel_exp.reindex(signal_returns.index).fillna(0.0),
        tom_exp.reindex(signal_returns.index).fillna(0.0),
        pd.DataFrame(
            {
                "SPY": 0.0,
                "TLT": 0.0,
                "UPRO": upro_frame["upro_exposure"]
                .reindex(signal_returns.index)
                .fillna(0.0),
            },
            index=signal_returns.index,
        ),
    ],
    axis=1,
    keys=_signal_cols,
)
signal_returns.tail(10)

# %% [markdown]
# ## Methodology
#
# 1. Build daily return streams for the three SPY/TLT rules and the UPRO residual rule.
# 2. Inner-join on dates so all legs are defined (sample starts when UPRO/BTC data allow).
# 3. Split the sample into train (first 60%) and test (last 40%).
# 4. Optimize non-negative weights that sum to one to maximize train Sharpe.
# 5. Apply optimized static weights to create blended portfolio returns.
# 6. Vol-target the blended portfolio using rolling 20-day realized volatility.
# 7. Compare equal-weight, optimized, and optimized-plus-vol-targeted variants.

# %% [markdown]
# ## Analysis

# %%
split_idx = int(len(signal_returns) * TRAIN_FRACTION)
train_returns = signal_returns.iloc[:split_idx]
test_returns = signal_returns.iloc[split_idx:]

optimized_weights = optimize_sharpe_weights(train_returns)
equal_weights = pd.Series(
    np.repeat(1 / signal_returns.shape[1], signal_returns.shape[1]),
    index=signal_returns.columns,
    name="weight",
)

weight_table = pd.concat([equal_weights.rename("equal_weight"), optimized_weights.rename("optimized_weight")], axis=1)
weight_table

# %%
portfolio_returns = pd.DataFrame(index=signal_returns.index)
portfolio_returns["equal_weight_return"] = signal_returns.mul(equal_weights, axis=1).sum(axis=1)
portfolio_returns["optimized_return"] = signal_returns.mul(optimized_weights, axis=1).sum(axis=1)

vol_target_frame = apply_vol_target(
    portfolio_returns["optimized_return"],
    target_annual_vol=TARGET_ANNUAL_VOL,
    lookback_days=VOL_LOOKBACK_DAYS,
    max_leverage=MAX_LEVERAGE,
)
portfolio_returns["optimized_vol_target_return"] = vol_target_frame["vol_targeted_return"]
portfolio_returns = portfolio_returns.dropna()
portfolio_returns.tail(10)

# %%
summary_rows: list[dict[str, object]] = []
for strategy_name in [
    "equal_weight_return",
    "optimized_return",
    "optimized_vol_target_return",
]:
    full_metrics = summarize_returns(portfolio_returns[strategy_name])
    train_series = portfolio_returns.loc[train_returns.index, strategy_name].dropna()
    test_series = portfolio_returns.loc[test_returns.index, strategy_name].dropna()
    train_metrics = summarize_returns(train_series)
    test_metrics = summarize_returns(test_series)
    if strategy_name == "optimized_vol_target_return":
        full_exposure = vol_target_frame["leverage"].reindex(
            portfolio_returns[strategy_name].dropna().index
        ).fillna(1.0)
        train_exposure = vol_target_frame["leverage"].reindex(train_series.index).fillna(1.0)
        test_exposure = vol_target_frame["leverage"].reindex(test_series.index).fillna(1.0)
    else:
        full_exposure = pd.Series(
            1.0, index=portfolio_returns[strategy_name].dropna().index
        )
        train_exposure = pd.Series(1.0, index=train_series.index)
        test_exposure = pd.Series(1.0, index=test_series.index)
    summary_rows.append(
        {
            "strategy": strategy_name,
            "full_total_return": full_metrics["total_return"],
            "full_sharpe": full_metrics["sharpe_ratio"],
            "full_max_drawdown": full_metrics["max_drawdown"],
            "full_ann_vol": full_metrics["annualized_volatility"],
            "full_mean_daily_turnover_one_way": mean_daily_turnover_one_way(full_exposure),
            "full_annualized_turnover_one_way": annualized_turnover_one_way(
                full_exposure, trading_days_per_year=TRADING_DAYS_PER_YEAR
            ),
            "train_sharpe": train_metrics["sharpe_ratio"],
            "train_mean_daily_turnover_one_way": mean_daily_turnover_one_way(train_exposure),
            "train_annualized_turnover_one_way": annualized_turnover_one_way(
                train_exposure, trading_days_per_year=TRADING_DAYS_PER_YEAR
            ),
            "test_sharpe": test_metrics["sharpe_ratio"],
            "test_total_return": test_metrics["total_return"],
            "test_max_drawdown": test_metrics["max_drawdown"],
            "test_mean_daily_turnover_one_way": mean_daily_turnover_one_way(test_exposure),
            "test_annualized_turnover_one_way": annualized_turnover_one_way(
                test_exposure, trading_days_per_year=TRADING_DAYS_PER_YEAR
            ),
        }
    )

performance_summary = pd.DataFrame(summary_rows)
performance_summary

# %%
vol_target_diagnostics = pd.DataFrame(
    {
        "metric": [
            "target_annual_vol",
            "average_applied_leverage",
            "median_applied_leverage",
            "max_applied_leverage",
            "realized_ann_vol_of_vol_target_portfolio",
            "realized_ann_vol_of_unscaled_optimized_portfolio",
        ],
        "value": [
            TARGET_ANNUAL_VOL,
            vol_target_frame["leverage"].mean(),
            vol_target_frame["leverage"].median(),
            vol_target_frame["leverage"].max(),
            summarize_returns(portfolio_returns["optimized_vol_target_return"])["annualized_volatility"],
            summarize_returns(portfolio_returns["optimized_return"])["annualized_volatility"],
        ],
    }
)
vol_target_diagnostics

# %% [markdown]
# ## Visualizations

# %%
equity_curves = (1 + portfolio_returns).cumprod()
fig, ax = plt.subplots()
equity_curves.plot(ax=ax)
ax.set_title("Signal Portfolio Equity Curves")
ax.set_xlabel("Date")
ax.set_ylabel("Growth of $1")
ax.legend(
    [
        "Equal-weight signals",
        "Optimized signals",
        "Optimized + vol target",
    ]
)
plt.show()

# %%
# Underlying ETF mix (per $1 of blended signal capital, before the vol overlay).
# Slices sum to 100% of gross exposure (SPY long, TLT long, TLT short, UPRO long,
# cash/flat). Vol targeting applies leverage to optimized returns and does not
# change these percentages versus the optimized panel below.
net_equal = blend_signal_exposures(per_signal_exposure, equal_weights)
net_optimized = blend_signal_exposures(per_signal_exposure, optimized_weights)
net_equal_plot = net_equal.reindex(portfolio_returns.index).fillna(0.0)
net_optimized_plot = net_optimized.reindex(portfolio_returns.index).fillna(0.0)
shares_equal = gross_exposure_shares(net_equal_plot)
shares_optimized = gross_exposure_shares(net_optimized_plot)

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
plot_exposure_mix(axes[0], shares_equal, "Equal-weight signals — implied ETF mix (% of gross exposure)")
plot_exposure_mix(
    axes[1],
    shares_optimized,
    "Optimized signals — implied ETF mix (vol-target variant uses this same mix)",
)
plt.tight_layout()
plt.show()

# %%
fig, ax = plt.subplots()
vol_target_frame["leverage"].reindex(portfolio_returns.index).plot(ax=ax, color="tab:purple")
ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
ax.axhline(MAX_LEVERAGE, color="tab:red", linewidth=1, linestyle=":")
ax.set_title("Applied Vol-Target Leverage")
ax.set_xlabel("Date")
ax.set_ylabel("Leverage multiple")
plt.show()

# %%
rolling_window = 63
rolling_sharpes = pd.DataFrame(index=portfolio_returns.index)
for column in portfolio_returns.columns:
    rolling_mean = portfolio_returns[column].rolling(rolling_window).mean()
    rolling_std = portfolio_returns[column].rolling(rolling_window).std()
    rolling_sharpes[column] = rolling_mean / rolling_std * np.sqrt(TRADING_DAYS_PER_YEAR)

fig, ax = plt.subplots()
rolling_sharpes.plot(ax=ax)
ax.axhline(0, color="black", linewidth=1)
ax.set_title(f"Rolling {rolling_window}-Day Sharpe")
ax.set_xlabel("Date")
ax.set_ylabel("Sharpe")
ax.legend(
    [
        "Equal-weight signals",
        "Optimized signals",
        "Optimized + vol target",
    ]
)
plt.show()

# %% [markdown]
# ## Limitations
#
# - Weight optimization can still overfit the in-sample window.
# - This notebook uses static optimized weights, not dynamic re-optimization.
# - Vol-target scaling uses historical realized volatility and may lag shocks.
# - Costs, turnover drag, and implementation constraints are not modeled.
# - The inner join with the UPRO residual leg shortens the combined sample to
#   dates where QQQ, UPRO, BTC, SPY, and TLT history all overlap.
#
# ## Conclusion
#
# This notebook backtests a combined portfolio of the three SPY/TLT signals
# plus the BTC/QQQ residual UPRO rule, compares equal-weight versus optimized
# blending, and evaluates a volatility targeting overlay on the optimized mix.
#
# ## Next Research Ideas
#
# - Add walk-forward re-optimization of signal weights.
# - Include turnover and trading-cost penalties in the objective.
# - Compare max-Sharpe optimization with risk parity and minimum-variance blends.
# - Add regime filters (rate volatility, trend, correlation state).
