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
# # Volatility-Targeted Futures Portfolio of SPY/TLT, UPRO Residual, and Dirty VIX Signals
#
# ## Research Question
#
# If we keep the volatility-targeted signal workflow from the ETF/proxy portfolio
# notebook but implement the risk using futures-equivalent notionals, how do the
# returns, notional exposure, turnover, and realized volatility change when the
# volatility-target overlay can lever the portfolio up to 5x?
#
# **Portfolio decision for this notebook:** use the same signal set and fixed
# `1/N` signal blend as the equal-weight and volatility-targeted notebooks, map
# each proxy exposure to a futures-equivalent notional exposure, and apply a
# lagged realized-volatility multiplier capped at `5.0x`.
#
# ## Hypothesis
#
# Futures implementation should express the same directional signal decisions
# with more direct notional control than ETF shares. A 5x cap gives the
# volatility-target overlay more room to reach the target in quiet regimes, but
# it should also make leverage, turnover, and financing assumptions more
# important than in the 2x-capped ETF/proxy notebook.

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
    SignalPortfolioBundle,
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
# - Signal timing and feature construction match the volatility-targeted
#   ETF/proxy notebook:
#   1. End-of-month SPY/TLT laggard rotation from trading day 15.
#   2. 5-day mean-reversion in `log(SPY/TLT)` as a long-only switch.
#   3. TLT turn-of-month long-last-5 / short-first-5 rule.
#   4. BTC/QQQ residual z-score long UPRO (flat when signal is off).
#   5. Dirty VIX cheapness long VX30 proxy when `zscore(log(VX30 / VIX3M))`
#      is below the entry threshold.
#   6. Optional BTC-derivatives UPRO legs when supplemental derivatives data or
#      Massive OPRA option access is available.
# - Futures-equivalent returns use close-to-close proxy returns:
#   - `SPY` exposure maps to S&P 500 futures notional (`ES` proxy return = SPY).
#   - `UPRO` exposure maps to 3x S&P 500 futures notional (`ES` proxy return = SPY).
#   - `TLT` exposure maps to long-bond futures notional (`ZB` proxy return = TLT).
#   - `VX30` exposure maps to VIX futures notional (`VX` proxy return = VX30).
# - The futures rows are not contract-level backtests. They are notional weights
#   per $1 of strategy capital; convert to contracts with broker futures prices,
#   multipliers, and account equity.
# - The volatility target is estimated from trailing daily returns of the
#   equal-weight futures-equivalent blend and shifted by one session before use.
# - Transaction costs, slippage, bid/ask spread, margin interest, collateral
#   yield, financing, and futures roll costs are excluded.
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
MAX_LEVERAGE = 5.0
ILLUSTRATIVE_ACCOUNT_EQUITY = 1_000_000.0

FUTURES_EXPOSURE_MAP = pd.DataFrame(
    [
        {
            "proxy_asset": "SPY",
            "futures_symbol": "ES",
            "notional_multiplier": 1.0,
            "return_proxy": "SPY",
            "contract_multiplier": 50.0,
            "implementation_note": "E-mini S&P 500 futures notional; MES can be used for smaller accounts.",
        },
        {
            "proxy_asset": "UPRO",
            "futures_symbol": "ES",
            "notional_multiplier": 3.0,
            "return_proxy": "SPY",
            "contract_multiplier": 50.0,
            "implementation_note": "Approximate UPRO with 3x S&P 500 futures notional.",
        },
        {
            "proxy_asset": "TLT",
            "futures_symbol": "ZB",
            "notional_multiplier": 1.0,
            "return_proxy": "TLT",
            "contract_multiplier": 1_000.0,
            "implementation_note": "Approximate TLT duration with CBOT 30-year Treasury bond futures.",
        },
        {
            "proxy_asset": "VX30",
            "futures_symbol": "VX",
            "notional_multiplier": 1.0,
            "return_proxy": "VX30",
            "contract_multiplier": 1_000.0,
            "implementation_note": "Dirty VX30 proxy mapped to VIX futures notional.",
        },
    ]
).set_index("proxy_asset")
FUTURES_SYMBOLS = list(dict.fromkeys(FUTURES_EXPOSURE_MAP["futures_symbol"]))


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


def map_proxy_exposures_to_futures_notional(proxy_exposure: pd.DataFrame) -> pd.DataFrame:
    """Map ETF/proxy weights into futures notional weights per $1 capital."""
    aligned = proxy_exposure.reindex(columns=FUTURES_EXPOSURE_MAP.index, fill_value=0.0).fillna(0.0)
    futures_notional = pd.DataFrame(0.0, index=aligned.index, columns=FUTURES_SYMBOLS)
    for proxy_asset, spec in FUTURES_EXPOSURE_MAP.iterrows():
        futures_notional[spec["futures_symbol"]] += (
            aligned[proxy_asset] * float(spec["notional_multiplier"])
        )
    return futures_notional


def build_futures_return_proxies(
    bundle: SignalPortfolioBundle,
    signal_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    market_prices = download_massive_daily_closes(["SPY", "TLT"], start_date=START_DATE).dropna(
        how="all"
    )
    missing = {"SPY", "TLT"} - set(market_prices.columns)
    if missing:
        raise ValueError(f"Missing futures return proxy price data: {sorted(missing)}")

    returns = pd.DataFrame(index=signal_index)
    returns["ES"] = market_prices["SPY"].pct_change().reindex(signal_index)
    returns["ZB"] = market_prices["TLT"].pct_change().reindex(signal_index)
    returns["VX"] = bundle.dirty_vix_frame["VX30"].pct_change().reindex(signal_index)
    return returns


def futures_notional_return(
    futures_notional: pd.DataFrame,
    futures_return_proxies: pd.DataFrame,
) -> pd.Series:
    aligned_returns = futures_return_proxies.reindex(futures_notional.index)
    missing_active_return = futures_notional.ne(0.0) & aligned_returns.isna()
    weighted_returns = futures_notional * aligned_returns
    out = weighted_returns.fillna(0.0).sum(axis=1)
    out[missing_active_return.any(axis=1)] = np.nan
    return out


def build_futures_signal_returns(
    per_signal_exposure: pd.DataFrame,
    signal_names: list[str],
    futures_return_proxies: pd.DataFrame,
) -> pd.DataFrame:
    returns: dict[str, pd.Series] = {}
    for signal_name in signal_names:
        proxy_exposure = per_signal_exposure[signal_name].reindex(futures_return_proxies.index)
        futures_notional = map_proxy_exposures_to_futures_notional(proxy_exposure)
        returns[signal_name] = futures_notional_return(
            futures_notional,
            futures_return_proxies,
        ).rename(signal_name)
    return pd.DataFrame(returns, index=futures_return_proxies.index).dropna()


def build_equal_weight_notebook_signal_set(
    params: SignalPortfolioParams,
) -> tuple[SignalPortfolioBundle, pd.DataFrame, list[str], pd.DataFrame]:
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
                        "VX30": 0.0,
                    },
                    index=signal_returns.index,
                )
                for column in btc_derivative_exposure.columns
            ],
            axis=1,
            keys=list(btc_derivative_exposure.columns),
        )
        per_signal_exposure = pd.concat([per_signal_exposure, derivative_exposure_panel], axis=1)
        bundle = SignalPortfolioBundle(
            signal_returns=signal_returns,
            per_signal_exposure=per_signal_exposure,
            upro_frame=bundle.upro_frame,
            dirty_vix_frame=bundle.dirty_vix_frame,
        )

    return bundle, per_signal_exposure, core_signal_names, derivative_features


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
bundle, per_signal_exposure, core_signal_names, derivative_features = (
    build_equal_weight_notebook_signal_set(_portfolio_params)
)
signal_returns = bundle.signal_returns
signal_returns.tail(10)

# %% [markdown]
# ## Methodology
#
# 1. Build the same daily signal exposure streams used by the equal-weight and
#    volatility-targeted notebooks, including optional BTC-derivatives UPRO legs
#    when data is available.
# 2. Map each proxy exposure to futures-equivalent notional:
#    `SPY -> ES`, `UPRO -> 3x ES`, `TLT -> ZB`, and `VX30 -> VX`.
# 3. Recompute signal returns from futures notional weights and proxy futures
#    return streams.
# 4. Assign each signal an equal fixed blend weight of `1/N`.
# 5. Compute the unscaled equal-weight futures-equivalent return stream.
# 6. Estimate trailing realized volatility of that stream, lag the multiplier by
#    one session, and cap leverage at `5.0x`.
# 7. Apply the multiplier to futures-equivalent returns and futures notional
#    weights.
# 8. Compare unscaled and volatility-targeted performance, turnover, leverage,
#    drawdowns, realized volatility, and futures notional usage.

# %% [markdown]
# ## Analysis

# %%
futures_mapping_table = FUTURES_EXPOSURE_MAP.reset_index()
futures_mapping_table

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
futures_return_proxies = build_futures_return_proxies(bundle, signal_returns.index)
futures_signal_returns = build_futures_signal_returns(
    per_signal_exposure,
    signal_returns.columns.tolist(),
    futures_return_proxies,
)
per_signal_exposure = per_signal_exposure.reindex(futures_signal_returns.index).fillna(0.0)
futures_signal_returns.tail(10)

# %%
equal_weights = equal_blend_weights(futures_signal_returns)
equal_weights.rename("equal_weight").to_frame()

# %%
equal_weight_return = futures_signal_returns.mul(equal_weights, axis=1).sum(axis=1).rename(
    "equal_weight_futures_return"
)
vol_target_scale = build_volatility_target_scale(
    equal_weight_return,
    target_volatility=VOL_TARGET,
    lookback_days=REALIZED_VOL_LOOKBACK,
    lag_sessions=VOL_SCALE_LAG_SESSIONS,
    max_leverage=MAX_LEVERAGE,
)
vol_target_return = (equal_weight_return * vol_target_scale).rename(
    "vol_target_futures_return"
)
portfolio_returns = pd.concat([equal_weight_return, vol_target_return], axis=1).dropna()
portfolio_returns.tail(10)

# %%
net_equal_proxy_exposure = blend_signal_exposures(per_signal_exposure, equal_weights)
net_equal_futures_notional = map_proxy_exposures_to_futures_notional(net_equal_proxy_exposure)
net_vol_target_futures_notional = net_equal_futures_notional.mul(vol_target_scale, axis=0)
net_futures_notional_summary = pd.DataFrame(
    {
        "equal_weight_gross_futures_notional": net_equal_futures_notional.abs().sum(axis=1),
        "vol_target_gross_futures_notional": net_vol_target_futures_notional.abs().sum(axis=1),
        "vol_target_scale": vol_target_scale,
    }
).reindex(portfolio_returns.index)
net_futures_notional_summary.tail(10)

# %%
summary_rows: list[dict[str, object]] = []
standalone_common = futures_signal_returns.reindex(portfolio_returns.index)
combined_returns = standalone_common.join(portfolio_returns, how="inner")
for strategy_name in combined_returns.columns:
    metrics = summarize_returns(combined_returns[strategy_name])
    if strategy_name == "equal_weight_futures_return":
        strategy_notional = net_equal_futures_notional.reindex(combined_returns.index).fillna(0.0)
    elif strategy_name == "vol_target_futures_return":
        strategy_notional = net_vol_target_futures_notional.reindex(
            combined_returns.index
        ).fillna(0.0)
    else:
        proxy_exposure = per_signal_exposure[strategy_name].reindex(combined_returns.index)
        strategy_notional = map_proxy_exposures_to_futures_notional(proxy_exposure)
    summary_rows.append(
        {
            "strategy": strategy_name,
            "total_return": metrics["total_return"],
            "annualized_return": metrics["annualized_return"],
            "annualized_volatility": metrics["annualized_volatility"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "max_drawdown": metrics["max_drawdown"],
            "win_rate": metrics["win_rate"],
            "mean_daily_notional_turnover_one_way": mean_daily_turnover_one_way(
                strategy_notional
            ),
            "annualized_notional_turnover_one_way": annualized_turnover_one_way(
                strategy_notional,
                trading_days_per_year=TRADING_DAYS_PER_YEAR,
            ),
            "mean_gross_futures_notional": strategy_notional.abs().sum(axis=1).mean(),
            "max_gross_futures_notional": strategy_notional.abs().sum(axis=1).max(),
        }
    )

performance_summary = pd.DataFrame(summary_rows)
performance_summary

# %%
valid_scale = vol_target_scale.reindex(portfolio_returns.index).dropna()
rolling_vol = pd.DataFrame(
    {
        "equal_weight_futures_realized_vol": rolling_realized_volatility(
            portfolio_returns["equal_weight_futures_return"],
            REALIZED_VOL_LOOKBACK,
        ),
        "vol_target_futures_realized_vol": rolling_realized_volatility(
            portfolio_returns["vol_target_futures_return"],
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
            "mean_equal_weight_gross_futures_notional",
            "mean_vol_target_gross_futures_notional",
            "max_vol_target_gross_futures_notional",
        ],
        "value": [
            VOL_TARGET,
            REALIZED_VOL_LOOKBACK,
            VOL_SCALE_LAG_SESSIONS,
            MAX_LEVERAGE,
            annualized_volatility(portfolio_returns["equal_weight_futures_return"]),
            annualized_volatility(portfolio_returns["vol_target_futures_return"]),
            valid_scale.mean(),
            valid_scale.median(),
            (valid_scale >= MAX_LEVERAGE).mean(),
            net_futures_notional_summary["equal_weight_gross_futures_notional"].mean(),
            net_futures_notional_summary["vol_target_gross_futures_notional"].mean(),
            net_futures_notional_summary["vol_target_gross_futures_notional"].max(),
        ],
    }
)
volatility_diagnostics

# %%
latest_futures_notional = (
    net_vol_target_futures_notional.reindex(portfolio_returns.index)
    .tail(1)
    .T.rename(columns=lambda date: "notional_weight_per_1_capital")
)
latest_futures_notional["dollar_notional_per_$1mm_equity"] = (
    latest_futures_notional["notional_weight_per_1_capital"] * ILLUSTRATIVE_ACCOUNT_EQUITY
)
latest_futures_notional = latest_futures_notional.join(
    FUTURES_EXPOSURE_MAP.reset_index()
    .drop_duplicates("futures_symbol")
    .set_index("futures_symbol")[["contract_multiplier", "implementation_note"]]
)
latest_futures_notional

# %%
signal_correlation = futures_signal_returns.corr()
signal_correlation

# %% [markdown]
# ## Visualizations

# %%
equity_curves = (1 + portfolio_returns).cumprod()
fig, ax = plt.subplots()
equity_curves.plot(ax=ax)
ax.set_title("Equal-Weight vs 5x-Capped Volatility-Targeted Futures Equity Curves")
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
    f"Lagged Futures Volatility Target Scale ({REALIZED_VOL_LOOKBACK}-Day Lookback, "
    f"{VOL_TARGET:.0%} Target, {MAX_LEVERAGE:.0f}x Cap)"
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
ax.set_title(f"Rolling {REALIZED_VOL_LOOKBACK}-Day Futures Realized Volatility")
ax.set_xlabel("Date")
ax.set_ylabel("Annualized volatility")
plot_date_axis(ax)
ax.legend()
plt.show()

# %%
fig, ax = plt.subplots(figsize=(14, 4))
net_vol_target_futures_notional.reindex(portfolio_returns.index).plot(ax=ax)
ax.axhline(0.0, color="black", linewidth=1)
ax.set_title("5x-Capped Volatility-Targeted Portfolio - Futures Notional Weights")
ax.set_xlabel("Date")
ax.set_ylabel("Futures notional per $1 capital")
plot_date_axis(ax)
fig.subplots_adjust(bottom=0.20, left=0.07, right=0.98)
plt.show()

# %%
fig, ax = plt.subplots()
net_futures_notional_summary[
    [
        "equal_weight_gross_futures_notional",
        "vol_target_gross_futures_notional",
    ]
].plot(ax=ax)
ax.axhline(MAX_LEVERAGE, color="tab:red", linewidth=1, linestyle=":", label="5x scale cap")
ax.set_title("Gross Futures Notional Usage")
ax.set_xlabel("Date")
ax.set_ylabel("Gross notional per $1 capital")
plot_date_axis(ax)
ax.legend()
plt.show()

# %% [markdown]
# ## Limitations
#
# - The futures mapping is a notional proxy, not a contract-level simulation with
#   real futures settlements, point values, roll calendars, tick sizes, or margin.
# - `TLT -> ZB` is a duration approximation. Contract choice and hedge ratio
#   should be revisited with actual Treasury futures data and duration matching.
# - `UPRO -> 3x ES` avoids UPRO ETF mechanics, fees, and daily compounding, so the
#   futures-equivalent UPRO legs can differ materially from ETF realized returns.
# - The VX30 leg uses a public-data VIX futures proxy and does not model futures
#   rolls or term-structure carry beyond that proxy return stream.
# - The 5x leverage cap only limits the volatility multiplier; gross futures
#   notional can be higher when underlying signal exposures are already levered
#   through the `UPRO -> 3x ES` mapping.
# - Financing, collateral yield, exchange margin, slippage, commissions, and
#   market impact are excluded.
# - Optional BTC-derivatives legs depend on data availability, ETF option
#   liquidity, DTE/strike filters, and Massive REST API access.
#
# ## Conclusion
#
# This notebook replicates the volatility-targeted signal portfolio with a
# futures-equivalent implementation layer. It rebuilds signal returns from
# mapped futures notionals, applies a lagged realized-volatility overlay capped
# at 5x, and reports the resulting return, volatility, turnover, leverage, and
# futures notional diagnostics.
#
# ## Next Research Ideas
#
# - Replace ETF/proxy returns with actual continuous futures histories and roll
#   rules for ES, Treasury futures, and VX.
# - Add contract-level sizing with current futures prices, multipliers, tick
#   values, and account equity.
# - Estimate financing, collateral yield, margin utilization, commissions,
#   slippage, and roll costs.
# - Duration-match the Treasury futures leg against TLT with rolling DV01 hedge
#   ratios.
