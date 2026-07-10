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
# # Volatility-Targeted Portfolio of SPY/TLT, UPRO Residual, and Dirty VIX Signals
#
# ## Research Question
#
# If we build the same signal set used by the equal-weight portfolio notebook,
# including the SPY/TLT rules, BTC/QQQ residual UPRO leg, and dirty VIX leg, then
# scale the combined portfolio to a fixed volatility target, does the volatility
# overlay improve realized risk control without materially diluting risk-adjusted
# returns?
#
# **Portfolio decision for this notebook:** use the equal-weight notebook's
# signal construction, keep the cross-signal blend at fixed `1/N` weights, and
# apply a lagged realized-volatility multiplier to the blended portfolio.
#
# ## Hypothesis
#
# The equal-weight signal blend has time-varying realized volatility because the
# active signal mix changes across SPY, TLT, and UPRO exposure. A
# volatility target should reduce exposure after high-volatility periods and
# increase exposure after low-volatility periods, producing a return stream with
# more stable risk than the unscaled equal-weight blend.

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

load_dotenv(dotenv_path=_REPO_ROOT / ".env")

from research.btc_derivative_signals import (
    DEFAULT_BTC_OPTION_PROXY_UNDERLYINGS,
    build_btc_derivative_features,
    build_btc_derivative_signal_legs,
    build_massive_etf_option_features,
    default_derivatives_path_candidates,
    load_derivatives_daily,
)
from research.data import download_massive_daily_closes
from research.plotting import apply_default_style
from research.signal_portfolio_blend import (
    SignalPortfolioParams,
    blend_signal_exposures,
    build_signal_portfolio_bundle,
    equal_blend_weights,
)
from research.stats import annualized_turnover_one_way, mean_daily_turnover_one_way

apply_default_style()

# %% [markdown]
# ## Assumptions
#
# - Signal returns are built exactly like the equal-weight portfolio notebook:
#   1. End-of-month SPY/TLT laggard rotation from trading day 15.
#   2. 5-day mean-reversion in `log(SPY/TLT)` as a long-only switch.
#   3. TLT turn-of-month long-last-5 / short-first-5 rule.
#   4. BTC/QQQ residual z-score long UPRO (flat when signal is off).
#   5. Optional BTC-derivatives UPRO legs when supplemental derivatives data or
#      Massive OPRA option access is available.
# - Signals use daily close data and earn the next close-to-close return.
# - The volatility target is estimated from trailing daily returns of the
#   equal-weight blend and shifted by one session before use.
# - Volatility targeting scales the whole blended portfolio; it does not change
#   relative signal weights.
# - Transaction costs, slippage, borrow costs, margin requirements, and financing
#   costs are excluded.
#
# ## Data Sources
#
# - Massive S3 flat files (US stock day aggregates) via `research.data`.
# - Massive S3 flat files for QQQ/UPRO daily and crypto minute BTC for the UPRO
#   residual signal via `research.upro_residual`.
# - Optional Massive REST OPRA option contracts/day-aggregates for IBIT/BITO
#   proxy derivatives features, plus optional local BTC derivatives fields.
# - Cboe public VIX3M history plus Yahoo `VX=F` (with Cboe VIX fallback) for
#   the dirty VIX long-volatility proxy.

# %%
START_DATE = "2004-01-01"
RESIDUAL_START_DATE = "2023-01-01"
TRADING_DAYS_PER_YEAR = 252

EOM_TRIGGER_DAY = 15
RELATIVE_REVERSAL_LOOKBACK = 5
TURN_OF_MONTH_WINDOW = 5
BETA_LOOKBACK_DAYS = 40
ZSCORE_LOOKBACK_DAYS = 20
ENTRY_ZSCORE = 1.5

DERIVATIVES_FEATURE_LAG_SESSIONS = 1
BTC_OPTION_PROXY_UNDERLYINGS = DEFAULT_BTC_OPTION_PROXY_UNDERLYINGS
DERIVATIVES_PATH_CANDIDATES = default_derivatives_path_candidates(_REPO_ROOT)

VOL_TARGET = 0.10
REALIZED_VOL_LOOKBACK = 63
VOL_SCALE_LAG_SESSIONS = 1
MAX_LEVERAGE = 2.0


def max_drawdown(return_series: pd.Series) -> float:
    equity_curve = (1 + return_series.fillna(0)).cumprod()
    drawdown = equity_curve / equity_curve.cummax() - 1
    return float(drawdown.min())


def annualized_volatility(return_series: pd.Series) -> float:
    returns = return_series.dropna()
    if returns.empty:
        return float("nan")
    return float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


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
    annualized_vol = annualized_volatility(returns)
    sharpe = (
        returns.mean() / returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        if returns.std() > 0
        else np.nan
    )
    return {
        "observations": len(returns),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown(returns),
        "win_rate": (returns > 0).mean(),
    }


def rolling_realized_volatility(return_series: pd.Series, lookback_days: int) -> pd.Series:
    return return_series.rolling(lookback_days).std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def build_volatility_target_scale(
    return_series: pd.Series,
    *,
    target_volatility: float,
    lookback_days: int,
    lag_sessions: int,
    max_leverage: float,
) -> pd.Series:
    realized_vol = rolling_realized_volatility(return_series, lookback_days)
    raw_scale = target_volatility / realized_vol.replace(0.0, np.nan)
    scale = raw_scale.replace([np.inf, -np.inf], np.nan).clip(lower=0.0, upper=max_leverage)
    return scale.shift(lag_sessions).rename("vol_target_scale")


def plot_date_axis(ax) -> None:
    locator = mdates.AutoDateLocator(minticks=6, maxticks=14)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.tick_params(axis="x", labelsize=9)
    plt.setp(ax.get_xticklabels(), rotation=28, ha="right")


def build_equal_weight_notebook_signal_set(
    params: SignalPortfolioParams,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame]:
    bundle = build_signal_portfolio_bundle(params, data_source="s3")
    signal_returns = bundle.signal_returns
    core_signal_names = signal_returns.columns.tolist()
    per_signal_exposure = bundle.per_signal_exposure

    upro_prices = download_massive_daily_closes(
        sorted({"UPRO", *BTC_OPTION_PROXY_UNDERLYINGS}),
        start_date=START_DATE,
    ).dropna(how="all")
    if "UPRO" not in upro_prices or upro_prices["UPRO"].dropna().empty:
        raise ValueError("No UPRO prices were downloaded.")
    upro_return = upro_prices["UPRO"].pct_change()

    derivatives_raw = load_derivatives_daily(DERIVATIVES_PATH_CANDIDATES)
    derivative_features = build_btc_derivative_features(derivatives_raw).reindex(upro_prices.index)
    etf_option_features = build_massive_etf_option_features(
        upro_prices,
        start_date=START_DATE,
        underlying_tickers=BTC_OPTION_PROXY_UNDERLYINGS,
    )
    derivative_features = derivative_features.combine_first(etf_option_features)
    if not derivative_features.empty:
        derivative_features = derivative_features.shift(DERIVATIVES_FEATURE_LAG_SESSIONS)

    btc_derivative_returns, btc_derivative_exposure = build_btc_derivative_signal_legs(
        derivative_features,
        upro_return,
    )
    if not btc_derivative_returns.empty:
        signal_returns = signal_returns.join(btc_derivative_returns, how="inner").dropna()
        per_signal_exposure = per_signal_exposure.reindex(signal_returns.index).fillna(0.0)
        derivative_exposure_panel = pd.concat(
            [
                pd.DataFrame(
                    {
                        "SPY": 0.0,
                        "TLT": 0.0,
                        "UPRO": btc_derivative_exposure[column]
                        .reindex(signal_returns.index)
                        .fillna(0.0),
                    },
                    index=signal_returns.index,
                )
                for column in btc_derivative_exposure.columns
            ],
            axis=1,
            keys=list(btc_derivative_exposure.columns),
        )
        per_signal_exposure = pd.concat([per_signal_exposure, derivative_exposure_panel], axis=1)

    return signal_returns, per_signal_exposure, core_signal_names, derivative_features


_portfolio_params = SignalPortfolioParams(
    start_date=START_DATE,
    residual_start_date=RESIDUAL_START_DATE,
    eom_trigger_day=EOM_TRIGGER_DAY,
    relative_reversal_lookback=RELATIVE_REVERSAL_LOOKBACK,
    turn_of_month_window=TURN_OF_MONTH_WINDOW,
    beta_lookback_days=BETA_LOOKBACK_DAYS,
    zscore_lookback_days=ZSCORE_LOOKBACK_DAYS,
    entry_zscore=ENTRY_ZSCORE,
)
signal_returns, per_signal_exposure, core_signal_names, derivative_features = (
    build_equal_weight_notebook_signal_set(_portfolio_params)
)
signal_returns.tail(10)

# %% [markdown]
# ## Methodology
#
# 1. Build the same daily signal return streams used by the equal-weight
#    portfolio notebook, including optional BTC-derivatives UPRO legs when data
#    is available.
# 2. Inner-join on dates so all active signal legs are defined.
# 3. Assign each signal an equal fixed blend weight of `1/N`.
# 4. Compute the unscaled equal-weight return stream.
# 5. Estimate trailing realized volatility of that equal-weight return stream.
# 6. Lag the volatility multiplier by one session and cap leverage.
# 7. Apply the multiplier to equal-weight returns and implied ETF exposures.
# 8. Compare unscaled and volatility-targeted performance, turnover, leverage,
#    drawdowns, and realized volatility.

# %% [markdown]
# ## Analysis

# %%
signal_inventory = pd.DataFrame(
    {
        "signal": signal_returns.columns,
        "source": [
            "core equal-weight notebook signal"
            if signal in core_signal_names
            else "optional BTC derivatives UPRO signal"
            for signal in signal_returns.columns
        ],
    }
)
signal_inventory

# %%
equal_weights = equal_blend_weights(signal_returns)
equal_weights.rename("equal_weight").to_frame()

# %%
equal_weight_return = signal_returns.mul(equal_weights, axis=1).sum(axis=1).rename(
    "equal_weight_return"
)
vol_target_scale = build_volatility_target_scale(
    equal_weight_return,
    target_volatility=VOL_TARGET,
    lookback_days=REALIZED_VOL_LOOKBACK,
    lag_sessions=VOL_SCALE_LAG_SESSIONS,
    max_leverage=MAX_LEVERAGE,
)
vol_target_return = (equal_weight_return * vol_target_scale).rename("vol_target_return")
portfolio_returns = pd.concat([equal_weight_return, vol_target_return], axis=1).dropna()
portfolio_returns.tail(10)

# %%
net_equal_exposure = blend_signal_exposures(per_signal_exposure, equal_weights)
net_vol_target_exposure = net_equal_exposure.mul(vol_target_scale, axis=0)
net_exposure_summary = pd.DataFrame(
    {
        "equal_weight_gross_exposure": net_equal_exposure.abs().sum(axis=1),
        "vol_target_gross_exposure": net_vol_target_exposure.abs().sum(axis=1),
        "vol_target_scale": vol_target_scale,
    }
).reindex(portfolio_returns.index)
net_exposure_summary.tail(10)

# %%
summary_rows: list[dict[str, object]] = []
standalone_common = signal_returns.reindex(portfolio_returns.index)
combined_returns = standalone_common.join(portfolio_returns, how="inner")
for strategy_name in combined_returns.columns:
    metrics = summarize_returns(combined_returns[strategy_name])
    if strategy_name == "equal_weight_return":
        strategy_exposure = net_equal_exposure.reindex(combined_returns.index).fillna(0.0)
    elif strategy_name == "vol_target_return":
        strategy_exposure = net_vol_target_exposure.reindex(combined_returns.index).fillna(0.0)
    else:
        strategy_exposure = (
            per_signal_exposure[strategy_name].reindex(combined_returns.index).fillna(0.0)
        )
    summary_rows.append(
        {
            "strategy": strategy_name,
            "total_return": metrics["total_return"],
            "annualized_return": metrics["annualized_return"],
            "annualized_volatility": metrics["annualized_volatility"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "max_drawdown": metrics["max_drawdown"],
            "win_rate": metrics["win_rate"],
            "mean_daily_turnover_one_way": mean_daily_turnover_one_way(strategy_exposure),
            "annualized_turnover_one_way": annualized_turnover_one_way(
                strategy_exposure,
                trading_days_per_year=TRADING_DAYS_PER_YEAR,
            ),
        }
    )

performance_summary = pd.DataFrame(summary_rows)
performance_summary

# %%
valid_scale = vol_target_scale.reindex(portfolio_returns.index).dropna()
rolling_vol = pd.DataFrame(
    {
        "equal_weight_realized_vol": rolling_realized_volatility(
            portfolio_returns["equal_weight_return"],
            REALIZED_VOL_LOOKBACK,
        ),
        "vol_target_realized_vol": rolling_realized_volatility(
            portfolio_returns["vol_target_return"],
            REALIZED_VOL_LOOKBACK,
        ),
    }
)
volatility_diagnostics = pd.DataFrame(
    {
        "metric": [
            "target_annualized_volatility",
            "lookback_days",
            "lag_sessions",
            "max_leverage",
            "equal_weight_realized_vol_common_sample",
            "vol_target_realized_vol_common_sample",
            "mean_vol_target_scale",
            "median_vol_target_scale",
            "pct_days_at_max_leverage",
            "mean_equal_weight_gross_exposure",
            "mean_vol_target_gross_exposure",
        ],
        "value": [
            VOL_TARGET,
            REALIZED_VOL_LOOKBACK,
            VOL_SCALE_LAG_SESSIONS,
            MAX_LEVERAGE,
            annualized_volatility(portfolio_returns["equal_weight_return"]),
            annualized_volatility(portfolio_returns["vol_target_return"]),
            valid_scale.mean(),
            valid_scale.median(),
            (valid_scale >= MAX_LEVERAGE).mean(),
            net_exposure_summary["equal_weight_gross_exposure"].mean(),
            net_exposure_summary["vol_target_gross_exposure"].mean(),
        ],
    }
)
volatility_diagnostics

# %%
signal_correlation = signal_returns.corr()
signal_correlation

# %% [markdown]
# ## Visualizations

# %%
equity_curves = (1 + portfolio_returns).cumprod()
fig, ax = plt.subplots()
equity_curves.plot(ax=ax)
ax.set_title("Equal-Weight vs Volatility-Targeted Portfolio Equity Curves")
ax.set_xlabel("Date")
ax.set_ylabel("Growth of $1")
plot_date_axis(ax)
plt.show()

# %%
fig, ax = plt.subplots()
vol_target_scale.reindex(portfolio_returns.index).plot(ax=ax)
ax.axhline(1.0, color="black", linewidth=1, linestyle="--", label="1.0x")
ax.axhline(MAX_LEVERAGE, color="tab:red", linewidth=1, linestyle=":", label="Max leverage")
ax.set_title(
    f"Lagged Volatility Target Scale ({REALIZED_VOL_LOOKBACK}-Day Lookback, "
    f"{VOL_TARGET:.0%} Target)"
)
ax.set_xlabel("Date")
ax.set_ylabel("Portfolio multiplier")
plot_date_axis(ax)
ax.legend()
plt.show()

# %%
fig, ax = plt.subplots()
rolling_vol.plot(ax=ax)
ax.axhline(VOL_TARGET, color="black", linewidth=1, linestyle="--", label="Target")
ax.set_title(f"Rolling {REALIZED_VOL_LOOKBACK}-Day Realized Volatility")
ax.set_xlabel("Date")
ax.set_ylabel("Annualized volatility")
plot_date_axis(ax)
ax.legend()
plt.show()

# %%
fig, ax = plt.subplots(figsize=(14, 4))
net_vol_target_exposure.reindex(portfolio_returns.index).plot(ax=ax)
ax.axhline(0.0, color="black", linewidth=1)
ax.set_title("Volatility-Targeted Portfolio - Implied ETF Weights")
ax.set_xlabel("Date")
ax.set_ylabel("Weight per $1 capital")
plot_date_axis(ax)
fig.subplots_adjust(bottom=0.20, left=0.07, right=0.98)
plt.show()

# %% [markdown]
# ## Limitations
#
# - The volatility target uses realized volatility, so it reacts after volatility
#   changes rather than forecasting them.
# - The leverage cap limits the ability to reach the target during very low-vol
#   regimes; financing and margin costs are not modeled.
# - Scaling UPRO exposures can create large effective equity beta even when the
#   portfolio's ex-ante volatility estimate is near target.
# - Optional BTC-derivatives legs depend on data availability, ETF option
#   liquidity, DTE/strike filters, and Massive REST API access.
# - The inner join across active signal legs can shorten the combined sample.
#
# ## Conclusion
#
# This notebook uses the same signal construction as the equal-weight portfolio
# study, then applies a lagged realized-volatility overlay to the equal-weight
# blend. The diagnostics compare whether the overlay delivers realized volatility
# closer to the target, how often it uses the leverage cap, and how the scaled ETF
# exposures change turnover and drawdown versus the unscaled equal-weight blend.
#
# ## Next Research Ideas
#
# - Compare realized-volatility targeting with EWMA or GARCH-style volatility
#   estimates.
# - Add financing, borrow, and transaction-cost assumptions to the scaled weights.
# - Test signal-level volatility targeting before the equal-weight blend.
# - Evaluate alternate target levels and leverage caps across market regimes.
