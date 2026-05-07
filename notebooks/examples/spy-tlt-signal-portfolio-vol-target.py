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
# **Portfolio decision for this notebook:** equal-weight blending across all
# active signal legs (each leg receives `1/N` of capital), with no optimization
# overlay.
# Daily weights for execution are exported with
# `scripts/export_equal_weight_portfolio_weights.py`.
#
# ## Hypothesis
#
# These signals likely have different timing and risk profiles, so equal
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

import os
import sys
from hashlib import sha256
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
from research.massive_rest import (
    download_rest_option_contracts,
    download_rest_option_day_aggs,
    has_api_key as has_massive_rest_api_key,
)
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
# - Baseline strategies:
#   1. End-of-month SPY/TLT laggard rotation from trading day 15.
#   2. 5-day mean-reversion in `log(SPY/TLT)` as a long-only switch.
#   3. TLT turn-of-month long-last-5 / short-first-5 rule.
#   4. BTC/QQQ residual z-score long UPRO (flat when signal is off).
# - BTC-derivatives long/flat UPRO overlays are built from Massive OPRA ETF
#   option features and optional supplemental local derivatives fields.
# - Portfolio construction is fixed to equal-weight (`1/N` per signal) for the
#   full sample; no weight optimization or vol-target overlay is applied.
# - Transaction costs, slippage, borrow costs, and financing are excluded.
#
# ## Data Sources
#
# - Massive S3 flat files (US stock day aggregates) via `research.data`.
# - Massive S3 flat files for QQQ/UPRO daily and crypto minute BTC for the UPRO
#   residual signal via `research.upro_residual`.
# - Massive REST OPRA option contracts/day-aggregates for IBIT/BITO proxy
#   derivatives features (cached on disk).

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
FORWARD_RETURN_HORIZONS = [1, 5, 10, 21]
DERIVATIVES_FEATURE_LAG_SESSIONS = 1
DERIVATIVES_PATH_CANDIDATES = [
    Path(os.environ["BTC_DERIVATIVES_DAILY_PATH"])
    for _ in [0]
    if os.getenv("BTC_DERIVATIVES_DAILY_PATH")
] + [
    _REPO_ROOT / "data/btc_derivatives_daily.parquet",
    _REPO_ROOT / "data/btc_derivatives_daily.csv",
]
BTC_OPTION_PROXY_UNDERLYINGS = [
    ticker.strip().upper()
    for ticker in os.getenv("BTC_OPTION_PROXY_UNDERLYINGS", "IBIT,BITO").split(",")
    if ticker.strip()
]
OPTION_TARGET_DTE_DAYS = 30
OPTION_MIN_DTE_DAYS = 14
OPTION_MAX_DTE_DAYS = 60
OPTION_CONTRACT_EXPIRATION_BUFFER_DAYS = 90
OPTION_MAX_UNIQUE_CONTRACTS = int(os.getenv("BTC_OPTION_PROXY_MAX_CONTRACTS", "250"))
REST_OPTION_CACHE_DIR = Path(
    os.getenv("Q_RESEARCH_REST_OPTION_CACHE_DIR", ".cache/q-research/massive-rest-options")
)
DERIVATIVE_SIGNAL_COLUMNS = [
    "btc_open_interest_5d_change",
    "btc_option_volume_zscore",
    "btc_call_put_volume_ratio",
    "btc_atm_iv_30d_change",
    "btc_dvol_change",
    "btc_25delta_risk_reversal_30d",
]


def max_drawdown(return_series: pd.Series) -> float:
    equity_curve = (1 + return_series.fillna(0)).cumprod()
    drawdown = equity_curve / equity_curve.cummax() - 1
    return float(drawdown.min())


def canonicalize_column_name(column: object) -> str:
    cleaned = str(column).strip().lower()
    cleaned = cleaned.replace("%", "pct").replace("/", "_").replace("-", "_")
    cleaned = cleaned.replace(" ", "_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def read_derivatives_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        raw = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        raw = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported derivatives file extension: {path.suffix}")
    raw = raw.rename(columns={column: canonicalize_column_name(column) for column in raw.columns})
    if "date" in raw.columns:
        index = pd.to_datetime(raw.pop("date"), errors="coerce")
    else:
        index = pd.to_datetime(raw.index, errors="coerce")
    raw = raw.loc[index.notna()].copy()
    raw.index = pd.DatetimeIndex(index[index.notna()]).normalize()
    raw = raw.sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]
    for column in raw.columns:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    return raw.select_dtypes(include=[np.number]).copy()


def load_derivatives_daily(paths: list[Path]) -> pd.DataFrame:
    for path in paths:
        if path.exists():
            return read_derivatives_file(path)
    return pd.DataFrame()


def rolling_zscore(series: pd.Series, lookback: int = 63) -> pd.Series:
    mean = series.rolling(lookback).mean()
    volatility = series.rolling(lookback).std()
    return (series - mean) / volatility


def option_cache_path(kind: str, parts: list[str]) -> Path:
    digest = sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return REST_OPTION_CACHE_DIR / f"{kind}_{digest}.parquet"


def load_or_download_option_contracts(underlying: str, start_date: str, end_date: str) -> pd.DataFrame:
    cache_path = option_cache_path("contracts", [underlying, start_date, end_date])
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    contracts = download_rest_option_contracts(underlying, start_date, end_date)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    contracts.to_parquet(cache_path)
    return contracts


def load_or_download_option_day_aggs(
    option_tickers: list[str], start_date: str, end_date: str
) -> pd.DataFrame:
    tickers = sorted({ticker for ticker in option_tickers if ticker})
    cache_path = option_cache_path("day_aggs", [start_date, end_date, *tickers])
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    option_aggs = download_rest_option_day_aggs(tickers, start_date, end_date)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    option_aggs.to_parquet(cache_path)
    return option_aggs


def build_contract_pairs(contracts: pd.DataFrame) -> pd.DataFrame:
    if contracts.empty:
        return pd.DataFrame()
    calls = contracts.loc[contracts["contract_type"] == "call"].copy()
    puts = contracts.loc[contracts["contract_type"] == "put"].copy()
    merge_keys = ["underlying_ticker", "expiration_date", "strike_price"]
    pairs = calls.merge(puts, on=merge_keys, suffixes=("_call", "_put"))
    return pairs.rename(columns={"ticker_call": "call_ticker", "ticker_put": "put_ticker"})


def select_near_atm_option_pairs(contracts: pd.DataFrame, underlying_prices: pd.Series) -> pd.DataFrame:
    pairs = build_contract_pairs(contracts)
    if pairs.empty or underlying_prices.dropna().empty:
        return pd.DataFrame()
    selections = []
    for session_date, underlying_close in underlying_prices.dropna().items():
        dte = (pairs["expiration_date"] - session_date).dt.days
        candidates = pairs.loc[dte.between(OPTION_MIN_DTE_DAYS, OPTION_MAX_DTE_DAYS)].copy()
        if candidates.empty:
            continue
        candidates["days_to_expiration"] = dte.loc[candidates.index]
        candidates["dte_distance"] = (candidates["days_to_expiration"] - OPTION_TARGET_DTE_DAYS).abs()
        candidates["strike_distance"] = (candidates["strike_price"] - underlying_close).abs()
        selected = candidates.sort_values(
            ["dte_distance", "strike_distance", "expiration_date", "strike_price"]
        ).iloc[0]
        selections.append(
            {
                "date": session_date,
                "underlying_ticker": selected["underlying_ticker"],
                "underlying_close": float(underlying_close),
                "expiration_date": selected["expiration_date"],
                "days_to_expiration": int(selected["days_to_expiration"]),
                "strike_price": float(selected["strike_price"]),
                "call_ticker": selected["call_ticker"],
                "put_ticker": selected["put_ticker"],
            }
        )
    if not selections:
        return pd.DataFrame()
    return pd.DataFrame(selections).set_index("date").sort_index()


def lookup_option_field(option_aggs: pd.DataFrame, tickers: pd.Series, field: str) -> pd.Series:
    if option_aggs.empty:
        return pd.Series(np.nan, index=tickers.index, name=field)
    long_field = option_aggs.reset_index().set_index(["date", "ticker"])[field].sort_index()
    lookup_index = pd.MultiIndex.from_arrays([tickers.index, tickers], names=["date", "ticker"])
    return pd.Series(long_field.reindex(lookup_index).to_numpy(), index=tickers.index, name=field)


def build_etf_option_features_from_selection(
    selections: pd.DataFrame, option_aggs: pd.DataFrame
) -> pd.DataFrame:
    if selections.empty or option_aggs.empty:
        return pd.DataFrame(index=selections.index)
    frame = selections.copy()
    frame["call_close"] = lookup_option_field(option_aggs, frame["call_ticker"], "close")
    frame["put_close"] = lookup_option_field(option_aggs, frame["put_ticker"], "close")
    frame["call_volume"] = lookup_option_field(option_aggs, frame["call_ticker"], "volume")
    frame["put_volume"] = lookup_option_field(option_aggs, frame["put_ticker"], "volume")
    total_volume = frame["call_volume"].fillna(0) + frame["put_volume"].fillna(0)
    features = pd.DataFrame(index=frame.index)
    features["btc_option_volume_zscore"] = rolling_zscore(np.log1p(total_volume))
    features["btc_call_put_volume_ratio"] = np.log(
        (frame["call_volume"].fillna(0) + 1) / (frame["put_volume"].fillna(0) + 1)
    )
    features["btc_atm_iv_30d_change"] = (
        ((frame["call_close"] + frame["put_close"]) / frame["underlying_close"]).diff()
    )
    return features.replace([np.inf, -np.inf], np.nan)


def build_massive_etf_option_features(prices: pd.DataFrame) -> pd.DataFrame:
    if not has_massive_rest_api_key():
        return pd.DataFrame(index=prices.index)
    contract_end = (
        pd.Timestamp.today().normalize() + pd.Timedelta(days=OPTION_CONTRACT_EXPIRATION_BUFFER_DAYS)
    ).date().isoformat()
    feature_frames: list[pd.DataFrame] = []
    for underlying in BTC_OPTION_PROXY_UNDERLYINGS:
        if underlying not in prices:
            continue
        contracts = load_or_download_option_contracts(underlying, START_DATE, contract_end)
        selections = select_near_atm_option_pairs(contracts, prices[underlying])
        if selections.empty:
            continue
        selected_tickers = sorted(
            set(selections["call_ticker"].dropna()) | set(selections["put_ticker"].dropna())
        )
        if len(selected_tickers) > OPTION_MAX_UNIQUE_CONTRACTS:
            continue
        option_aggs = load_or_download_option_day_aggs(
            selected_tickers,
            selections.index.min().date().isoformat(),
            selections.index.max().date().isoformat(),
        )
        features = build_etf_option_features_from_selection(selections, option_aggs)
        if not features.empty:
            feature_frames.append(features.reindex(prices.index))
    if not feature_frames:
        return pd.DataFrame(index=prices.index)
    combined = pd.DataFrame(index=prices.index)
    for features in feature_frames:
        combined = combined.combine_first(features)
    return combined


def build_derivative_features(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(index=raw.index)
    frame = raw.copy()
    features = pd.DataFrame(index=frame.index)
    if "btc_open_interest_usd" in frame:
        oi = frame["btc_open_interest_usd"].replace(0, np.nan).astype(float)
        oi_log = np.log(oi)
        features["btc_open_interest_5d_change"] = oi_log.diff(5)
    if "btc_option_volume_usd" in frame:
        features["btc_option_volume_zscore"] = rolling_zscore(
            np.log(frame["btc_option_volume_usd"].replace(0, np.nan).astype(float))
        )
    if {"btc_call_volume_usd", "btc_put_volume_usd"}.issubset(frame.columns):
        call_volume = frame["btc_call_volume_usd"].replace(0, np.nan).astype(float)
        put_volume = frame["btc_put_volume_usd"].replace(0, np.nan).astype(float)
        features["btc_call_put_volume_ratio"] = np.log(call_volume / put_volume)
    if "btc_atm_iv_30d" in frame:
        features["btc_atm_iv_30d_change"] = frame["btc_atm_iv_30d"].astype(float).diff()
    if "btc_dvol" in frame:
        features["btc_dvol_change"] = frame["btc_dvol"].astype(float).diff()
    if "btc_25delta_risk_reversal_30d" in frame:
        features["btc_25delta_risk_reversal_30d"] = frame["btc_25delta_risk_reversal_30d"].astype(float)
    return features.replace([np.inf, -np.inf], np.nan)


def add_forward_returns(upro_close: pd.Series, horizons: list[int]) -> pd.DataFrame:
    out = pd.DataFrame(index=upro_close.index)
    for horizon in horizons:
        out[f"UPRO_fwd_{horizon}d_return"] = upro_close.shift(-horizon) / upro_close - 1
    return out


def build_btc_derivative_signal_legs(
    features: pd.DataFrame,
    upro_return: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if features.empty:
        return pd.DataFrame(index=upro_return.index), pd.DataFrame()
    returns: dict[str, pd.Series] = {}
    exposures: dict[str, pd.Series] = {}
    aligned = features.reindex(upro_return.index)
    for column in DERIVATIVE_SIGNAL_COLUMNS:
        if column not in aligned or aligned[column].notna().sum() < 120:
            continue
        high = aligned[column].rolling(252, min_periods=63).quantile(0.8)
        signal_at_close = aligned[column] >= high
        position = signal_at_close.shift(1).fillna(False).astype(float)
        leg_name = f"btc_deriv_{column}"
        returns[leg_name] = (position * upro_return).rename(leg_name)
        exposures[leg_name] = position.rename(leg_name)
    if not returns:
        return pd.DataFrame(index=upro_return.index), pd.DataFrame()
    return_frame = pd.DataFrame(returns, index=upro_return.index)
    exposure_frame = pd.DataFrame(exposures, index=upro_return.index)
    return return_frame, exposure_frame


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
core_signal_names = signal_returns.columns.tolist()
per_signal_exposure = _bundle.per_signal_exposure

upro_prices = download_massive_daily_closes(
    sorted({"UPRO", *BTC_OPTION_PROXY_UNDERLYINGS}),
    start_date=START_DATE,
).dropna(how="all")
if upro_prices.empty:
    raise ValueError("No UPRO prices were downloaded.")
upro_return = upro_prices["UPRO"].pct_change()
upro_forward_returns = add_forward_returns(upro_prices["UPRO"], FORWARD_RETURN_HORIZONS)

derivatives_raw = load_derivatives_daily(DERIVATIVES_PATH_CANDIDATES)
derivative_features = build_derivative_features(derivatives_raw).reindex(upro_prices.index)
etf_option_features = build_massive_etf_option_features(upro_prices)
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
                {"SPY": 0.0, "TLT": 0.0, "UPRO": btc_derivative_exposure[column].reindex(signal_returns.index).fillna(0.0)},
                index=signal_returns.index,
            )
            for column in btc_derivative_exposure.columns
        ],
        axis=1,
        keys=list(btc_derivative_exposure.columns),
    )
    per_signal_exposure = pd.concat([per_signal_exposure, derivative_exposure_panel], axis=1)
signal_returns.tail(10)

# %%
upro_forward_returns.reindex(signal_returns.index).tail(10)

# %% [markdown]
# ## Methodology
#
# 1. Build daily return streams for the three SPY/TLT rules and the UPRO residual rule.
# 2. Build BTC derivatives features from Massive OPRA ETF option data (plus any
#    supplemental local derivatives fields), then lag features by one session.
# 3. Compute UPRO forward returns (1/5/10/21-day) for cross-horizon diagnostics.
# 4. Inner-join on dates so all legs are defined.
# 5. Apply fixed equal weights across all active signals to build portfolio returns.
# 6. Compare standalone signal performance with the equal-weight blend.
# 7. Measure pairwise signal correlation to check whether each added signal
#    increases diversification.
# 8. Estimate each signal's beta to the S&P 500 proxy (SPY daily returns) to
#    separate directional market exposure from idiosyncratic edge.
# 9. Export equal-weight implied ETF weights for production with
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
base_corr = signal_returns[core_signal_names].corr()
base_pairwise = base_corr.where(np.triu(np.ones(base_corr.shape, dtype=bool), k=1)).stack()
diversification_summary = pd.DataFrame(
    {
        "metric": [
            "avg_pairwise_corr_base_signals",
            "avg_pairwise_corr_all_signals",
            "median_pairwise_corr_all_signals",
            "min_pairwise_corr_all_signals",
            "max_pairwise_corr_all_signals",
            "change_in_avg_pairwise_corr_after_adding_derivative_signals",
        ],
        "value": [
            base_pairwise.mean(),
            pairwise_correlations.mean(),
            pairwise_correlations.median(),
            pairwise_correlations.min(),
            pairwise_correlations.max(),
            pairwise_correlations.mean() - base_pairwise.mean(),
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
# - OPRA feature coverage depends on ETF option liquidity, selected DTE/strike
#   filters, and Massive REST API access.
# - The inner join across active signal legs can shorten the combined sample.
#
# ## Conclusion
#
# This notebook backtests an equal-weight combined-signal portfolio using the
# three SPY/TLT legs, the BTC/QQQ residual UPRO leg, and optional BTC-derivatives
# UPRO signal legs. It also reports UPRO forward returns across multiple horizons
# for diagnostic checks while comparing diversification (correlation) and market
# exposure (beta to SPY proxy) across all active signals. Export daily weights with
# `uv run python scripts/export_equal_weight_portfolio_weights.py`.
#
# ## Next Research Ideas
#
# - Add conditional-correlation and conditional-beta analysis by volatility
#   regime to see when diversification is strongest.
# - Include turnover and trading-cost penalties in performance comparisons.
# - Add regime filters (rate volatility, trend, correlation state).
