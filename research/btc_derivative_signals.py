"""BTC derivatives feature helpers for optional UPRO signal legs."""

from __future__ import annotations

import os
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from research.massive_rest import (
    download_rest_option_contracts,
    download_rest_option_day_aggs,
    has_api_key as has_massive_rest_api_key,
)

DEFAULT_DERIVATIVE_SIGNAL_COLUMNS = [
    "btc_open_interest_5d_change",
    "btc_option_volume_zscore",
    "btc_call_put_volume_ratio",
    "btc_atm_iv_30d_change",
    "btc_dvol_change",
    "btc_25delta_risk_reversal_30d",
]
DEFAULT_BTC_OPTION_PROXY_UNDERLYINGS = [
    ticker.strip().upper()
    for ticker in os.getenv("BTC_OPTION_PROXY_UNDERLYINGS", "IBIT,BITO").split(",")
    if ticker.strip()
]
DEFAULT_OPTION_TARGET_DTE_DAYS = 30
DEFAULT_OPTION_MIN_DTE_DAYS = 14
DEFAULT_OPTION_MAX_DTE_DAYS = 60
DEFAULT_OPTION_CONTRACT_EXPIRATION_BUFFER_DAYS = 90
DEFAULT_OPTION_MAX_UNIQUE_CONTRACTS = int(os.getenv("BTC_OPTION_PROXY_MAX_CONTRACTS", "250"))
DEFAULT_REST_OPTION_CACHE_DIR = Path(
    os.getenv("Q_RESEARCH_REST_OPTION_CACHE_DIR", ".cache/q-research/massive-rest-options")
)


def default_derivatives_path_candidates(repo_root: Path) -> list[Path]:
    """Candidate supplemental BTC derivatives files used by the portfolio notebooks."""
    paths = [
        Path(os.environ["BTC_DERIVATIVES_DAILY_PATH"])
        for _ in [0]
        if os.getenv("BTC_DERIVATIVES_DAILY_PATH")
    ]
    paths.extend(
        [
            repo_root / "data/btc_derivatives_daily.parquet",
            repo_root / "data/btc_derivatives_daily.csv",
        ]
    )
    return paths


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


def option_cache_path(kind: str, parts: list[str], cache_dir: Path) -> Path:
    digest = sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{kind}_{digest}.parquet"


def load_or_download_option_contracts(
    underlying: str,
    start_date: str,
    end_date: str,
    *,
    cache_dir: Path = DEFAULT_REST_OPTION_CACHE_DIR,
) -> pd.DataFrame:
    cache_path = option_cache_path("contracts", [underlying, start_date, end_date], cache_dir)
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
    *,
    cache_dir: Path = DEFAULT_REST_OPTION_CACHE_DIR,
) -> pd.DataFrame:
    tickers = sorted({ticker for ticker in option_tickers if ticker})
    cache_path = option_cache_path("day_aggs", [start_date, end_date, *tickers], cache_dir)
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


def select_near_atm_option_pairs(
    contracts: pd.DataFrame,
    underlying_prices: pd.Series,
    *,
    target_dte_days: int = DEFAULT_OPTION_TARGET_DTE_DAYS,
    min_dte_days: int = DEFAULT_OPTION_MIN_DTE_DAYS,
    max_dte_days: int = DEFAULT_OPTION_MAX_DTE_DAYS,
) -> pd.DataFrame:
    pairs = build_contract_pairs(contracts)
    if pairs.empty or underlying_prices.dropna().empty:
        return pd.DataFrame()
    selections = []
    for session_date, underlying_close in underlying_prices.dropna().items():
        dte = (pairs["expiration_date"] - session_date).dt.days
        candidates = pairs.loc[dte.between(min_dte_days, max_dte_days)].copy()
        if candidates.empty:
            continue
        candidates["days_to_expiration"] = dte.loc[candidates.index]
        candidates["dte_distance"] = (candidates["days_to_expiration"] - target_dte_days).abs()
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


def build_massive_etf_option_features(
    prices: pd.DataFrame,
    *,
    start_date: str,
    underlying_tickers: list[str] | None = None,
    cache_dir: Path = DEFAULT_REST_OPTION_CACHE_DIR,
    contract_expiration_buffer_days: int = DEFAULT_OPTION_CONTRACT_EXPIRATION_BUFFER_DAYS,
    max_unique_contracts: int = DEFAULT_OPTION_MAX_UNIQUE_CONTRACTS,
) -> pd.DataFrame:
    if not has_massive_rest_api_key():
        return pd.DataFrame(index=prices.index)
    contract_end = (
        pd.Timestamp.today().normalize() + pd.Timedelta(days=contract_expiration_buffer_days)
    ).date().isoformat()
    feature_frames: list[pd.DataFrame] = []
    for underlying in underlying_tickers or DEFAULT_BTC_OPTION_PROXY_UNDERLYINGS:
        if underlying not in prices:
            continue
        contracts = load_or_download_option_contracts(
            underlying,
            start_date,
            contract_end,
            cache_dir=cache_dir,
        )
        selections = select_near_atm_option_pairs(contracts, prices[underlying])
        if selections.empty:
            continue
        selected_tickers = sorted(
            set(selections["call_ticker"].dropna()) | set(selections["put_ticker"].dropna())
        )
        if len(selected_tickers) > max_unique_contracts:
            continue
        option_aggs = load_or_download_option_day_aggs(
            selected_tickers,
            selections.index.min().date().isoformat(),
            selections.index.max().date().isoformat(),
            cache_dir=cache_dir,
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


def build_btc_derivative_features(raw: pd.DataFrame) -> pd.DataFrame:
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
        features["btc_25delta_risk_reversal_30d"] = frame[
            "btc_25delta_risk_reversal_30d"
        ].astype(float)
    return features.replace([np.inf, -np.inf], np.nan)


def build_btc_derivative_signal_legs(
    features: pd.DataFrame,
    upro_return: pd.Series,
    *,
    signal_columns: list[str] | None = None,
    quantile_lookback_days: int = 252,
    min_observations: int = 120,
    high_quantile: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if features.empty:
        return pd.DataFrame(index=upro_return.index), pd.DataFrame()
    returns: dict[str, pd.Series] = {}
    exposures: dict[str, pd.Series] = {}
    aligned = features.reindex(upro_return.index)
    for column in signal_columns or DEFAULT_DERIVATIVE_SIGNAL_COLUMNS:
        if column not in aligned or aligned[column].notna().sum() < min_observations:
            continue
        high = aligned[column].rolling(quantile_lookback_days, min_periods=63).quantile(high_quantile)
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
