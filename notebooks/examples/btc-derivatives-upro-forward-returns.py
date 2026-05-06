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
# # BTC Derivatives Signals and UPRO Forward Returns
#
# ## Research Question
#
# Can Bitcoin derivatives positioning and option-price information predict UPRO
# forward returns?
#
# ## Hypothesis
#
# If Bitcoin open interest is expanding and Bitcoin option prices show improving
# risk appetite, then the same liquidity and risk-on impulse may lead U.S. equity
# beta and produce positive forward returns for UPRO.

# %%
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

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
from research.massive_rest import (
    download_rest_option_contracts,
    download_rest_option_day_aggs,
    has_api_key as has_massive_rest_api_key,
)
from research.plotting import apply_default_style

load_dotenv(dotenv_path=".env")
apply_default_style()

# %% [markdown]
# ## Assumptions
#
# - UPRO daily closes come from Massive US stock day-aggregate flat files and
#   represent executable 4pm New York equity closes.
# - BTC spot prices come from Massive crypto minute flat files resampled to hourly
#   closes, then aligned to the latest completed hourly close at or before each
#   equity close.
# - BTC option-price features use U.S.-listed OPRA options on Bitcoin-linked ETFs
#   as a Massive-native proxy for native BTC options. The default proxy universe
#   is IBIT, then BITO as a fallback.
# - ETF option features use near-ATM call/put daily closes around a target 30-day
#   expiration. They are lagged by one equity session by default to avoid using
#   option marks that may not have been known at the U.S. equity close.
# - Historical native BTC open interest is not available from the Massive helpers
#   in this repository, so open-interest fields can still be supplied through an
#   optional local daily derivatives file.
# - The tests are predictive diagnostics, not an executable trading model. They
#   ignore transaction costs, slippage, taxes, financing, and UPRO path-dependent
#   leverage effects.
#
# ## Data Sources
#
# - Massive S3 flat files: `us_stocks_sip/day_aggs_v1` for UPRO and Bitcoin ETF
#   daily closes.
# - Massive S3 flat files: global crypto `minute_aggs_v1` for X:BTC-USD, resampled
#   to hourly and aligned to U.S. equity closes.
# - Massive REST OPRA endpoints: option contract reference data and daily option
#   OHLC aggregates for Bitcoin-linked ETF options.
# - Optional local derivatives file:
#   `data/btc_derivatives_daily.csv`, `data/btc_derivatives_daily.parquet`, or the
#   path in `BTC_DERIVATIVES_DAILY_PATH`.
#
# Expected optional derivatives columns include a `date` column plus any of:
#
# - `btc_open_interest_usd`
# - `btc_option_open_interest_usd`
# - `btc_option_volume_usd`
# - `btc_call_volume_usd`
# - `btc_put_volume_usd`
# - `btc_atm_iv_30d`
# - `btc_dvol`
# - `btc_25delta_risk_reversal_30d`
# - `btc_25delta_butterfly_30d`

# %%
START_DATE = "2020-01-01"
END_DATE = datetime.now(UTC).date().isoformat()

FORWARD_RETURN_HORIZONS = [1, 5, 10, 21]
DERIVATIVES_FEATURE_LAG_SESSIONS = 1
ROLLING_ZSCORE_DAYS = 63
MIN_TEST_OBSERVATIONS = 60
TRADING_DAYS_PER_YEAR = 252

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

DERIVATIVES_PATH_CANDIDATES = [
    Path(os.environ["BTC_DERIVATIVES_DAILY_PATH"])
    for _ in [0]
    if os.getenv("BTC_DERIVATIVES_DAILY_PATH")
] + [
    Path("data/btc_derivatives_daily.parquet"),
    Path("data/btc_derivatives_daily.csv"),
]

BASELINE_FEATURES = [
    "BTC_return",
    "BTC_5d_return",
    "BTC_21d_return",
    "BTC_realized_vol_21d",
]


@dataclass(frozen=True)
class FeatureDefinition:
    column: str
    label: str
    family: str


FEATURE_DEFINITIONS = [
    FeatureDefinition("BTC_return", "BTC 1-day return", "spot control"),
    FeatureDefinition("BTC_5d_return", "BTC 5-day return", "spot control"),
    FeatureDefinition("BTC_21d_return", "BTC 21-day return", "spot control"),
    FeatureDefinition("BTC_realized_vol_21d", "BTC 21-day realized volatility", "spot control"),
    FeatureDefinition("btc_open_interest_log", "BTC futures/perp open interest level", "open interest"),
    FeatureDefinition("btc_open_interest_1d_change", "BTC open interest 1-day change", "open interest"),
    FeatureDefinition("btc_open_interest_5d_change", "BTC open interest 5-day change", "open interest"),
    FeatureDefinition("btc_open_interest_zscore", "BTC open interest z-score", "open interest"),
    FeatureDefinition("btc_open_interest_change_zscore", "BTC open interest change z-score", "open interest"),
    FeatureDefinition("btc_option_open_interest_log", "BTC option open interest level", "options"),
    FeatureDefinition("btc_option_volume_zscore", "BTC option volume z-score", "options"),
    FeatureDefinition("btc_call_put_volume_ratio", "BTC call/put volume ratio", "options"),
    FeatureDefinition("btc_atm_iv_30d", "BTC 30-day ATM IV", "options"),
    FeatureDefinition("btc_atm_iv_30d_change", "BTC 30-day ATM IV change", "options"),
    FeatureDefinition("btc_dvol", "BTC implied volatility index", "options"),
    FeatureDefinition("btc_dvol_change", "BTC implied volatility index change", "options"),
    FeatureDefinition("btc_25delta_risk_reversal_30d", "BTC 25-delta risk reversal", "options"),
    FeatureDefinition("btc_25delta_butterfly_30d", "BTC 25-delta butterfly", "options"),
    FeatureDefinition("btc_etf_option_call_yield", "BTC ETF ATM call price / ETF", "etf options"),
    FeatureDefinition("btc_etf_option_put_yield", "BTC ETF ATM put price / ETF", "etf options"),
    FeatureDefinition("btc_etf_option_straddle_yield", "BTC ETF ATM straddle price / ETF", "etf options"),
    FeatureDefinition("btc_etf_option_straddle_yield_change", "BTC ETF straddle yield change", "etf options"),
    FeatureDefinition("btc_etf_option_volume_zscore", "BTC ETF option volume z-score", "etf options"),
    FeatureDefinition("btc_etf_option_call_put_volume_ratio", "BTC ETF call/put volume ratio", "etf options"),
]

FEATURE_LABELS = {definition.column: definition.label for definition in FEATURE_DEFINITIONS}
FEATURE_FAMILIES = {definition.column: definition.family for definition in FEATURE_DEFINITIONS}


def canonicalize_column_name(column: object) -> str:
    """Normalize file columns to lower snake-case names."""
    cleaned = str(column).strip().lower()
    cleaned = cleaned.replace("%", "pct").replace("/", "_").replace("-", "_")
    cleaned = cleaned.replace(" ", "_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def read_derivatives_file(path: Path) -> pd.DataFrame:
    """Read a local daily derivatives file from CSV or Parquet."""
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
    numeric = raw.select_dtypes(include=[np.number]).copy()
    return numeric


def load_derivatives_daily(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the first available derivatives dataset and return a status table."""
    status_rows = []
    for path in paths:
        exists = path.exists()
        status_rows.append(
            {
                "candidate_path": path.as_posix(),
                "exists": exists,
                "selected": False,
                "message": "available" if exists else "not found",
            }
        )
        if not exists:
            continue

        frame = read_derivatives_file(path)
        status_rows[-1]["selected"] = True
        status_rows[-1]["message"] = f"loaded {len(frame):,} rows and {len(frame.columns):,} numeric columns"
        return frame, pd.DataFrame(status_rows)

    return pd.DataFrame(), pd.DataFrame(status_rows)


def rolling_zscore(series: pd.Series, lookback: int = ROLLING_ZSCORE_DAYS) -> pd.Series:
    """Standardize a series with rolling mean and volatility."""
    mean = series.rolling(lookback).mean()
    volatility = series.rolling(lookback).std()
    return (series - mean) / volatility


def option_cache_path(kind: str, parts: list[str]) -> Path:
    """Build a stable cache path for Massive REST option downloads."""
    digest = sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return REST_OPTION_CACHE_DIR / f"{kind}_{digest}.parquet"


def load_or_download_option_contracts(
    underlying: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Load cached ETF option contracts or download them from Massive REST."""
    cache_path = option_cache_path("contracts", [underlying, start_date, end_date])
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    contracts = download_rest_option_contracts(underlying, start_date, end_date)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    contracts.to_parquet(cache_path)
    return contracts


def load_or_download_option_day_aggs(
    option_tickers: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Load cached option daily aggregates or download them from Massive REST."""
    tickers = sorted({ticker for ticker in option_tickers if ticker})
    cache_path = option_cache_path("day_aggs", [start_date, end_date, *tickers])
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    option_aggs = download_rest_option_day_aggs(tickers, start_date, end_date)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    option_aggs.to_parquet(cache_path)
    return option_aggs


def build_contract_pairs(contracts: pd.DataFrame) -> pd.DataFrame:
    """Pair calls and puts with the same expiration and strike."""
    if contracts.empty:
        return pd.DataFrame()

    calls = contracts.loc[contracts["contract_type"] == "call"].copy()
    puts = contracts.loc[contracts["contract_type"] == "put"].copy()
    merge_keys = ["underlying_ticker", "expiration_date", "strike_price"]
    pairs = calls.merge(
        puts,
        on=merge_keys,
        suffixes=("_call", "_put"),
    )
    return pairs.rename(
        columns={
            "ticker_call": "call_ticker",
            "ticker_put": "put_ticker",
        }
    )


def select_near_atm_option_pairs(
    contracts: pd.DataFrame,
    underlying_prices: pd.Series,
    *,
    target_dte: int = OPTION_TARGET_DTE_DAYS,
    min_dte: int = OPTION_MIN_DTE_DAYS,
    max_dte: int = OPTION_MAX_DTE_DAYS,
) -> pd.DataFrame:
    """Select one near-ATM call/put pair around the target maturity per date."""
    pairs = build_contract_pairs(contracts)
    if pairs.empty or underlying_prices.dropna().empty:
        return pd.DataFrame()

    selections = []
    for session_date, underlying_close in underlying_prices.dropna().items():
        dte = (pairs["expiration_date"] - session_date).dt.days
        candidates = pairs.loc[dte.between(min_dte, max_dte)].copy()
        if candidates.empty:
            continue

        candidates["days_to_expiration"] = dte.loc[candidates.index]
        candidates["dte_distance"] = (candidates["days_to_expiration"] - target_dte).abs()
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
    """Align option aggregate values to a date-indexed selected-ticker series."""
    if option_aggs.empty:
        return pd.Series(np.nan, index=tickers.index, name=field)

    long_field = option_aggs.reset_index().set_index(["date", "ticker"])[field].sort_index()
    lookup_index = pd.MultiIndex.from_arrays([tickers.index, tickers], names=["date", "ticker"])
    return pd.Series(long_field.reindex(lookup_index).to_numpy(), index=tickers.index, name=field)


def build_etf_option_features_from_selection(
    selections: pd.DataFrame,
    option_aggs: pd.DataFrame,
) -> pd.DataFrame:
    """Convert selected ETF option pair prices into BTC option-price proxy features."""
    if selections.empty or option_aggs.empty:
        return pd.DataFrame(index=selections.index)

    frame = selections.copy()
    frame["call_close"] = lookup_option_field(option_aggs, frame["call_ticker"], "close")
    frame["put_close"] = lookup_option_field(option_aggs, frame["put_ticker"], "close")
    frame["call_volume"] = lookup_option_field(option_aggs, frame["call_ticker"], "volume")
    frame["put_volume"] = lookup_option_field(option_aggs, frame["put_ticker"], "volume")

    total_volume = frame["call_volume"].fillna(0) + frame["put_volume"].fillna(0)
    features = pd.DataFrame(index=frame.index)
    features["btc_etf_option_call_yield"] = frame["call_close"] / frame["underlying_close"]
    features["btc_etf_option_put_yield"] = frame["put_close"] / frame["underlying_close"]
    features["btc_etf_option_straddle_yield"] = (
        frame["call_close"] + frame["put_close"]
    ) / frame["underlying_close"]
    features["btc_etf_option_straddle_yield_change"] = features[
        "btc_etf_option_straddle_yield"
    ].diff()
    features["btc_etf_option_volume_zscore"] = rolling_zscore(np.log1p(total_volume))
    features["btc_etf_option_call_put_volume_ratio"] = np.log(
        (frame["call_volume"].fillna(0) + 1) / (frame["put_volume"].fillna(0) + 1)
    )
    features["btc_etf_option_days_to_expiration"] = frame["days_to_expiration"]
    features["btc_etf_option_moneyness"] = frame["strike_price"] / frame["underlying_close"] - 1
    features["btc_etf_option_underlying"] = frame["underlying_ticker"]
    return features.replace([np.inf, -np.inf], np.nan)


def build_massive_etf_option_features(
    prices: pd.DataFrame,
    underlyings: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build Massive OPRA option-price features from Bitcoin-linked ETF options."""
    status_rows = []
    selection_frames = []
    feature_frames = []

    if not has_massive_rest_api_key():
        return (
            pd.DataFrame(index=prices.index),
            pd.DataFrame(
                [
                    {
                        "underlying": ",".join(underlyings),
                        "status": "skipped",
                        "message": (
                            "Massive REST API key not found; set MASSIVE_API_KEY or "
                            "POLYGON_API_KEY to download ETF option prices."
                        ),
                    }
                ]
            ),
            pd.DataFrame(),
        )

    contract_end = (
        pd.Timestamp(END_DATE) + pd.Timedelta(days=OPTION_CONTRACT_EXPIRATION_BUFFER_DAYS)
    ).date().isoformat()

    for underlying in underlyings:
        if underlying not in prices:
            status_rows.append(
                {
                    "underlying": underlying,
                    "status": "skipped",
                    "message": "underlying ETF close was not downloaded from Massive flat files",
                }
            )
            continue

        contracts = load_or_download_option_contracts(underlying, START_DATE, contract_end)
        selections = select_near_atm_option_pairs(contracts, prices[underlying])
        if selections.empty:
            status_rows.append(
                {
                    "underlying": underlying,
                    "status": "skipped",
                    "message": f"no near-ATM option pairs found in {OPTION_MIN_DTE_DAYS}-{OPTION_MAX_DTE_DAYS} DTE window",
                }
            )
            continue

        selected_tickers = sorted(
            set(selections["call_ticker"].dropna()) | set(selections["put_ticker"].dropna())
        )
        if len(selected_tickers) > OPTION_MAX_UNIQUE_CONTRACTS:
            status_rows.append(
                {
                    "underlying": underlying,
                    "status": "skipped",
                    "message": (
                        f"selected {len(selected_tickers):,} unique contracts; raise "
                        "BTC_OPTION_PROXY_MAX_CONTRACTS to download them"
                    ),
                }
            )
            continue

        option_aggs = load_or_download_option_day_aggs(
            selected_tickers,
            selections.index.min().date().isoformat(),
            selections.index.max().date().isoformat(),
        )
        features = build_etf_option_features_from_selection(selections, option_aggs)
        feature_rows = features.dropna(subset=["btc_etf_option_straddle_yield"])
        if feature_rows.empty:
            status_rows.append(
                {
                    "underlying": underlying,
                    "status": "skipped",
                    "message": "downloaded contracts but found no matched daily option closes",
                }
            )
            continue

        status_rows.append(
            {
                "underlying": underlying,
                "status": "loaded",
                "message": (
                    f"{len(feature_rows):,} feature rows from {len(selected_tickers):,} "
                    "near-ATM call/put contracts"
                ),
            }
        )
        feature_frames.append(features)
        selection_frames.append(selections.assign(proxy_underlying=underlying))

    if not feature_frames:
        return pd.DataFrame(index=prices.index), pd.DataFrame(status_rows), pd.DataFrame()

    combined = pd.DataFrame(index=prices.index)
    for features in feature_frames:
        combined = combined.combine_first(features.reindex(prices.index))
    selections = pd.concat(selection_frames).sort_index() if selection_frames else pd.DataFrame()
    return combined, pd.DataFrame(status_rows), selections


def add_if_present(frame: pd.DataFrame, output: pd.DataFrame, column: str) -> None:
    if column in frame:
        output[column] = frame[column].astype(float)


def build_derivative_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert raw open-interest and option-price fields into daily predictors."""
    if raw.empty:
        return pd.DataFrame(index=raw.index)

    frame = raw.copy()
    features = pd.DataFrame(index=frame.index)

    if "btc_open_interest_usd" in frame:
        oi = frame["btc_open_interest_usd"].replace(0, np.nan).astype(float)
        features["btc_open_interest_log"] = np.log(oi)
        features["btc_open_interest_1d_change"] = features["btc_open_interest_log"].diff()
        features["btc_open_interest_5d_change"] = features["btc_open_interest_log"].diff(5)
        features["btc_open_interest_zscore"] = rolling_zscore(features["btc_open_interest_log"])
        features["btc_open_interest_change_zscore"] = rolling_zscore(
            features["btc_open_interest_5d_change"]
        )

    if "btc_option_open_interest_usd" in frame:
        option_oi = frame["btc_option_open_interest_usd"].replace(0, np.nan).astype(float)
        features["btc_option_open_interest_log"] = np.log(option_oi)

    if "btc_option_volume_usd" in frame:
        features["btc_option_volume_zscore"] = rolling_zscore(
            np.log(frame["btc_option_volume_usd"].replace(0, np.nan).astype(float))
        )

    if {"btc_call_volume_usd", "btc_put_volume_usd"}.issubset(frame.columns):
        call_volume = frame["btc_call_volume_usd"].replace(0, np.nan).astype(float)
        put_volume = frame["btc_put_volume_usd"].replace(0, np.nan).astype(float)
        features["btc_call_put_volume_ratio"] = np.log(call_volume / put_volume)

    for column in [
        "btc_atm_iv_30d",
        "btc_dvol",
        "btc_25delta_risk_reversal_30d",
        "btc_25delta_butterfly_30d",
    ]:
        add_if_present(frame, features, column)

    for column in ["btc_atm_iv_30d", "btc_dvol"]:
        if column in features:
            features[f"{column}_change"] = features[column].diff()

    return features.replace([np.inf, -np.inf], np.nan)


def max_drawdown(return_series: pd.Series) -> float:
    equity_curve = (1 + return_series.fillna(0)).cumprod()
    drawdown = equity_curve / equity_curve.cummax() - 1
    return float(drawdown.min())


def add_forward_returns(frame: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    output = frame.copy()
    for horizon in horizons:
        output[f"UPRO_fwd_{horizon}d_return"] = output["UPRO"].shift(-horizon) / output["UPRO"] - 1
    return output


def summarize_data_coverage(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        series = frame[column].dropna() if column in frame else pd.Series(dtype=float)
        rows.append(
            {
                "column": column,
                "label": FEATURE_LABELS.get(column, column),
                "family": FEATURE_FAMILIES.get(column, "unknown"),
                "observations": int(series.count()),
                "first_date": series.index.min() if not series.empty else pd.NaT,
                "last_date": series.index.max() if not series.empty else pd.NaT,
                "coverage_rate": series.count() / len(frame) if len(frame) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def hac_univariate_regression(sample: pd.DataFrame, feature: str, target: str, horizon: int) -> dict[str, float]:
    """Estimate a one-feature predictive regression with HAC robust errors."""
    y = sample[target].astype(float)
    x = sample[feature].astype(float)
    x_std = (x - x.mean()) / x.std()
    model = sm.OLS(y, sm.add_constant(x_std)).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": max(1, min(5, horizon))},
    )
    return {
        "ols_beta_per_1sd": float(model.params[feature]),
        "ols_t_stat": float(model.tvalues[feature]),
        "ols_p_value": float(model.pvalues[feature]),
        "r_squared": float(model.rsquared),
    }


def quantile_spread(sample: pd.DataFrame, feature: str, target: str) -> dict[str, float]:
    """Compare forward returns in high-feature and low-feature quintiles."""
    ranked = sample[[feature, target]].dropna().copy()
    try:
        ranked["bucket"] = pd.qcut(ranked[feature], 5, labels=False, duplicates="drop")
    except ValueError:
        return {
            "bottom_quintile_mean": np.nan,
            "top_quintile_mean": np.nan,
            "top_minus_bottom": np.nan,
        }
    if ranked["bucket"].nunique() < 2:
        return {
            "bottom_quintile_mean": np.nan,
            "top_quintile_mean": np.nan,
            "top_minus_bottom": np.nan,
        }

    bottom = ranked.loc[ranked["bucket"] == ranked["bucket"].min(), target]
    top = ranked.loc[ranked["bucket"] == ranked["bucket"].max(), target]
    return {
        "bottom_quintile_mean": float(bottom.mean()),
        "top_quintile_mean": float(top.mean()),
        "top_minus_bottom": float(top.mean() - bottom.mean()),
    }


def run_predictive_tests(
    frame: pd.DataFrame,
    features: list[str],
    horizons: list[int],
    *,
    min_observations: int = MIN_TEST_OBSERVATIONS,
) -> pd.DataFrame:
    """Run univariate rank-correlation, quantile, and robust regression tests."""
    rows = []
    for feature in features:
        if feature not in frame:
            continue
        for horizon in horizons:
            target = f"UPRO_fwd_{horizon}d_return"
            sample = frame[[feature, target]].dropna()
            if len(sample) < min_observations or sample[feature].std() == 0:
                continue

            spearman = stats.spearmanr(sample[feature], sample[target], nan_policy="omit")
            pearson = stats.pearsonr(sample[feature], sample[target])
            spread = quantile_spread(sample, feature, target)
            regression = hac_univariate_regression(sample, feature, target, horizon)
            rows.append(
                {
                    "feature": feature,
                    "label": FEATURE_LABELS.get(feature, feature),
                    "family": FEATURE_FAMILIES.get(feature, "unknown"),
                    "horizon_days": horizon,
                    "observations": len(sample),
                    "spearman_ic": float(spearman.statistic),
                    "spearman_p_value": float(spearman.pvalue),
                    "pearson_ic": float(pearson.statistic),
                    "pearson_p_value": float(pearson.pvalue),
                    **spread,
                    **regression,
                }
            )
    return pd.DataFrame(rows)


def build_directional_signal_backtest(
    frame: pd.DataFrame,
    feature: str,
    *,
    horizon: int = 1,
    high_quantile: float = 0.80,
    low_quantile: float = 0.20,
) -> pd.DataFrame:
    """Build a simple long/flat daily signal from rolling feature quantiles."""
    target = f"UPRO_fwd_{horizon}d_return"
    signal_frame = frame[[feature, "UPRO_return", target]].dropna().copy()
    rolling_high = signal_frame[feature].rolling(252, min_periods=63).quantile(high_quantile)
    rolling_low = signal_frame[feature].rolling(252, min_periods=63).quantile(low_quantile)

    same_direction_sample = signal_frame[[feature, target]].dropna()
    ic = (
        stats.spearmanr(same_direction_sample[feature], same_direction_sample[target]).statistic
        if len(same_direction_sample) >= MIN_TEST_OBSERVATIONS
        else np.nan
    )
    if pd.isna(ic) or ic >= 0:
        signal_at_close = signal_frame[feature] >= rolling_high
        direction = "high feature"
    else:
        signal_at_close = signal_frame[feature] <= rolling_low
        direction = "low feature"

    signal_frame["signal_direction"] = direction
    signal_frame["signal_at_close"] = signal_at_close.fillna(False)
    signal_frame["position"] = signal_frame["signal_at_close"].shift(1).fillna(False)
    signal_frame["strategy_return"] = np.where(
        signal_frame["position"],
        signal_frame["UPRO_return"],
        0.0,
    )
    return signal_frame


def summarize_signal_backtest(signal_frame: pd.DataFrame) -> pd.DataFrame:
    returns = signal_frame["strategy_return"].dropna()
    positions = signal_frame["position"].reindex(returns.index).fillna(False)
    active_returns = returns.loc[positions]
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
                int(positions.sum()),
                positions.mean(),
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

# %% [markdown]
# ## Methodology
#
# 1. Download UPRO and Bitcoin-linked ETF daily closes, then align BTC spot to
#    each U.S. equity close.
# 2. Use Massive OPRA REST endpoints to discover near-ATM IBIT/BITO option pairs
#    and download their daily call/put option prices when a REST key is present.
# 3. Convert ETF option prices into BTC option-price proxy features such as
#    call/put premium yield, straddle yield, option-volume z-score, and call/put
#    volume ratio.
# 4. Load a local daily BTC derivatives feature file if one is present, primarily
#    for native BTC open-interest or option-surface fields not covered by OPRA
#    ETF option aggregates.
# 5. Lag derivatives features by one equity session by default.
# 6. Measure predictive power against 1-, 5-, 10-, and 21-session forward UPRO
#    returns with:
#    - Spearman and Pearson information coefficients.
#    - Top-minus-bottom quintile forward-return spreads.
#    - HAC-robust univariate predictive regressions.
# 7. Build a simple daily long/flat UPRO signal from the strongest available
#    one-day feature as an implementation sanity check.

# %% [markdown]
# ## Data

# %%
equity_tickers = sorted({"UPRO", *BTC_OPTION_PROXY_UNDERLYINGS})
equity_closes = download_equity_closes(equity_tickers, START_DATE, END_DATE)
btc_hourly_close = download_btc_hourly(START_DATE, END_DATE)

if equity_closes.empty or "UPRO" not in equity_closes:
    raise ValueError("No UPRO equity closes were downloaded.")
if btc_hourly_close.empty:
    raise ValueError("No BTC-USD hourly closes were downloaded.")

equity_close_times = build_equity_close_times(equity_closes.index)
btc_close = align_btc_to_equity_close(btc_hourly_close, equity_close_times)

prices = equity_closes.join(btc_close, how="inner")
prices = prices.dropna(subset=["UPRO", "btc_close_at_equity_close"])
prices = prices.rename(columns={"btc_close_at_equity_close": "BTC"})
prices.tail()

# %%
etf_option_features_raw, etf_option_status, etf_option_selections = build_massive_etf_option_features(
    prices,
    BTC_OPTION_PROXY_UNDERLYINGS,
)
etf_option_status

# %%
if etf_option_features_raw.empty:
    display(
        Markdown(
            "### Massive ETF option proxy status\n\n"
            "No Bitcoin-linked ETF option-price proxy features were loaded. Check the "
            "`etf_option_status` table above. A Massive REST key (`MASSIVE_API_KEY` or "
            "`POLYGON_API_KEY`) is required for OPRA contract discovery and option "
            "daily aggregates."
        )
    )
else:
    display(
        Markdown(
            "### Massive ETF option proxy status\n\n"
            f"Loaded **{etf_option_features_raw['btc_etf_option_straddle_yield'].count():,}** "
            "aligned near-ATM option-price observations."
        )
    )
    display(etf_option_features_raw.dropna(how="all").tail())

# %%
derivatives_raw, derivatives_load_status = load_derivatives_daily(DERIVATIVES_PATH_CANDIDATES)
derivatives_load_status

# %%
if derivatives_raw.empty:
    display(
        Markdown(
            "### Derivatives data status\n\n"
            "No local BTC derivatives file was found. Native BTC open-interest fields "
            "will be absent unless `data/btc_derivatives_daily.csv` is added or "
            "`BTC_DERIVATIVES_DAILY_PATH` is set. Massive ETF option-price proxy "
            "features are handled separately above."
        )
    )
else:
    display(
        Markdown(
            f"### Derivatives data status\n\n"
            f"Loaded **{len(derivatives_raw):,}** daily rows with "
            f"**{len(derivatives_raw.columns):,}** numeric fields."
        )
    )
    display(derivatives_raw.tail())

# %%
derivative_features = build_derivative_features(derivatives_raw)
if not derivative_features.empty:
    derivative_features = derivative_features.shift(DERIVATIVES_FEATURE_LAG_SESSIONS)
etf_option_features = etf_option_features_raw.shift(DERIVATIVES_FEATURE_LAG_SESSIONS)

spot_features = prices.copy()
spot_features["UPRO_return"] = spot_features["UPRO"].pct_change()
spot_features["BTC_return"] = spot_features["BTC"].pct_change()
spot_features["BTC_5d_return"] = spot_features["BTC"].pct_change(5)
spot_features["BTC_21d_return"] = spot_features["BTC"].pct_change(21)
spot_features["BTC_realized_vol_21d"] = (
    spot_features["BTC_return"].rolling(21).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
)

analysis = spot_features.join([etf_option_features, derivative_features], how="left")
analysis = add_forward_returns(analysis, FORWARD_RETURN_HORIZONS)
analysis.tail()

# %%
available_feature_columns = [
    definition.column
    for definition in FEATURE_DEFINITIONS
    if definition.column in analysis and analysis[definition.column].notna().sum() >= MIN_TEST_OBSERVATIONS
]

coverage = summarize_data_coverage(analysis, [definition.column for definition in FEATURE_DEFINITIONS])
coverage.loc[coverage["observations"] > 0].sort_values(
    ["family", "observations"],
    ascending=[True, False],
)

# %% [markdown]
# ## Analysis
#
# The first table reports feature coverage. ETF option-price features populate
# from Massive OPRA when a REST key is present; native BTC open-interest fields
# populate from the optional local derivatives file. The tests below
# automatically include whichever feature families have enough observations.

# %%
coverage

# %%
predictive_tests = run_predictive_tests(
    analysis,
    available_feature_columns,
    FORWARD_RETURN_HORIZONS,
)

if predictive_tests.empty:
    display(
        Markdown(
            "No feature has enough observations for the predictive tests. "
            f"The current minimum is {MIN_TEST_OBSERVATIONS} aligned sessions."
        )
    )
else:
    predictive_tests.sort_values(
        ["horizon_days", "spearman_ic"],
        ascending=[True, False],
    )

# %%
thesis_feature_tests = (
    predictive_tests.loc[
        predictive_tests["family"].isin(["open interest", "options", "etf options"])
    ]
    if not predictive_tests.empty
    else pd.DataFrame()
)

if thesis_feature_tests.empty:
    display(
        Markdown(
            "No open-interest or option-price feature has enough observations yet. "
            "The thesis-specific evidence table will populate after Massive ETF "
            "option data or the local derivatives file is available."
        )
    )
else:
    thesis_feature_tests.sort_values(
        ["horizon_days", "spearman_ic"],
        ascending=[True, False],
    )

# %%
if not predictive_tests.empty:
    family_priority = {"open interest": 0, "etf options": 1, "options": 2, "spot control": 3}
    best_one_day = (
        predictive_tests.loc[predictive_tests["horizon_days"] == 1]
        .assign(abs_spearman_ic=lambda frame: frame["spearman_ic"].abs())
        .assign(family_priority=lambda frame: frame["family"].map(family_priority).fillna(99))
        .sort_values(["family_priority", "abs_spearman_ic"], ascending=[True, False])
    )
    best_one_day
else:
    best_one_day = pd.DataFrame()
    best_one_day

# %%
if not best_one_day.empty:
    selected_feature = best_one_day.iloc[0]["feature"]
    selected_label = best_one_day.iloc[0]["label"]
    signal_backtest = build_directional_signal_backtest(analysis, selected_feature, horizon=1)
    display(Markdown(f"### Simple Signal Sanity Check\n\nSelected feature: **{selected_label}**."))
    display(summarize_signal_backtest(signal_backtest))
else:
    selected_feature = None
    signal_backtest = pd.DataFrame()

# %% [markdown]
# ## Visualizations

# %%
fig, ax = plt.subplots()
prices[["UPRO", "BTC"]].div(prices[["UPRO", "BTC"]].iloc[0]).plot(ax=ax)
ax.set_title("Normalized UPRO and BTC Spot Prices")
ax.set_xlabel("Date")
ax.set_ylabel("Growth of $1")
plt.show()

# %%
if not etf_option_features.empty:
    plot_columns = [
        column
        for column in [
            "btc_etf_option_straddle_yield",
            "btc_etf_option_straddle_yield_change",
            "btc_etf_option_volume_zscore",
            "btc_etf_option_call_put_volume_ratio",
        ]
        if column in etf_option_features
    ]
    if plot_columns:
        fig, ax = plt.subplots()
        etf_option_features[plot_columns].dropna(how="all").plot(ax=ax)
        ax.set_title("Massive BTC ETF Option-Price Proxy Features")
        ax.set_xlabel("Date")
        ax.legend()
        plt.show()

# %%
if not derivative_features.empty:
    plot_columns = [
        column
        for column in [
            "btc_open_interest_log",
            "btc_open_interest_zscore",
            "btc_atm_iv_30d",
            "btc_dvol",
            "btc_25delta_risk_reversal_30d",
        ]
        if column in derivative_features
    ]
    if plot_columns:
        fig, ax = plt.subplots()
        derivative_features[plot_columns].dropna(how="all").plot(ax=ax)
        ax.set_title("BTC Derivatives Feature History")
        ax.set_xlabel("Date")
        ax.legend()
        plt.show()

# %%
if not predictive_tests.empty:
    fig, ax = plt.subplots(figsize=(12, 6))
    ic_plot = predictive_tests.pivot_table(
        index="label",
        columns="horizon_days",
        values="spearman_ic",
        aggfunc="first",
    )
    sort_horizon = 1 if 1 in ic_plot.columns else ic_plot.columns.min()
    ic_plot = ic_plot.reindex(ic_plot[sort_horizon].abs().sort_values(ascending=False).index)
    ic_plot.plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Spearman Information Coefficient by Feature and Horizon")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Spearman IC")
    ax.legend(title="Forward sessions")
    fig.tight_layout()
    plt.show()

# %%
if not predictive_tests.empty:
    spread_plot = predictive_tests.loc[predictive_tests["horizon_days"] == 5].copy()
    if not spread_plot.empty:
        spread_plot = spread_plot.sort_values("top_minus_bottom")
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.barh(spread_plot["label"], spread_plot["top_minus_bottom"])
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title("5-Day Forward UPRO Return: Top Minus Bottom Feature Quintile")
        ax.set_xlabel("Average return spread")
        fig.tight_layout()
        plt.show()

# %%
if not signal_backtest.empty:
    strategy_curve = (1 + signal_backtest["strategy_return"]).cumprod()
    upro_curve = (1 + signal_backtest["UPRO_return"].fillna(0)).cumprod().reindex(strategy_curve.index)
    fig, ax = plt.subplots()
    strategy_curve.plot(ax=ax, label="Signal long/flat UPRO")
    upro_curve.plot(ax=ax, label="Buy and hold UPRO", alpha=0.75)
    ax.set_title("Simple Signal Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    plt.show()

# %% [markdown]
# ## Limitations
#
# - U.S.-listed Bitcoin ETF options are a proxy for BTC options, not native BTC
#   options. ETF-specific flows, creation/redemption mechanics, and equity-market
#   trading hours may affect the signal.
# - Massive OPRA daily aggregates provide option trade OHLC and volume, but not a
#   historical implied-volatility surface in this workflow. Straddle yield is a
#   model-free option-price proxy rather than an IV estimate.
# - Historical native BTC open interest still requires the optional local
#   derivatives file. That file may combine venues with different reporting
#   times; the default one-session lag is conservative but may be too conservative
#   for intraday marks known before the U.S. equity close.
# - Open interest can rise because of long or short positioning; direction is not
#   identifiable without additional long/short or dealer-positioning data.
# - Option implied volatility and skew can indicate either risk appetite or crash
#   demand. The univariate tests here should be followed with controls for BTC
#   spot momentum, BTC volatility, SPY/QQQ returns, and macro-volatility regimes.
# - UPRO is a 3x S&P 500 ETF. A Bitcoin-linked risk-on signal may map differently
#   to QQQ, TQQQ, SPY, or unlevered equity exposure.
# - Multiple comparisons are not adjusted for; p-values are exploratory.
#
# ## Conclusion
#
# This notebook creates the investigation framework for the thesis that Bitcoin
# open interest and option prices can predict UPRO forward returns. It aligns
# UPRO and BTC spot data to the equity close, builds Massive-native IBIT/BITO
# near-ATM option-price proxy features when REST credentials are available,
# accepts optional native BTC derivatives fields, checks feature coverage, and
# runs rank-correlation, quintile-spread, and HAC regression diagnostics for each
# available feature and horizon.
#
# ## Next Research Ideas
#
# - Add venue-level BTC futures and perpetual open interest to separate CME,
#   Binance, OKX, and Deribit positioning.
# - Expand the ETF option proxy from one near-ATM 30-day straddle to multiple
#   tenors, deltas, term-structure slopes, and skew approximations.
# - Add native option-surface features by tenor and delta: ATM IV, 25-delta risk
#   reversal, butterflies, and call/put open-interest imbalance.
# - Compare UPRO results with SPY, QQQ, TQQQ, and BTC itself to separate broad
#   risk-on effects from UPRO-specific leverage.
# - Use expanding or walk-forward models that choose features on an in-sample
#   window and evaluate them out of sample.
# - Add transaction costs and slippage if the diagnostics graduate into a live
#   UPRO strategy.
