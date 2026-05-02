"""BTC/QQQ residual z-score long UPRO strategy (Massive daily + hourly data)."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from research.data import request_massive_aggregates

NY_TZ = ZoneInfo("America/New_York")


def download_massive_equity_closes(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download Massive adjusted daily equity closes (calendar dates, America/New_York)."""
    close_series: dict[str, pd.Series] = {}
    for ticker in tickers:
        bars = request_massive_aggregates(
            ticker,
            multiplier=1,
            timespan="day",
            start_date=start_date,
            end_date=end_date,
            adjusted=True,
        )
        closes = bars["close"].dropna()
        closes.index = closes.index.tz_convert(NY_TZ).tz_localize(None).normalize()
        close_series[ticker] = closes.rename(ticker)

    return pd.concat(close_series.values(), axis=1).sort_index().dropna(how="all")


def download_btc_hourly_closes(start_date: str, end_date: str) -> pd.Series:
    """Download hourly BTC-USD closes; relabel bar timestamps to interval close."""
    bars = request_massive_aggregates(
        "X:BTC-USD",
        multiplier=1,
        timespan="hour",
        start_date=start_date,
        end_date=end_date,
        adjusted=True,
    )
    closes = bars["close"].dropna()
    if closes.empty:
        return closes

    closes.index = closes.index + pd.Timedelta(hours=1)
    return closes.loc[closes.index >= pd.Timestamp(start_date, tz="UTC")]


def build_equity_close_times(equity_dates: pd.Index) -> pd.Series:
    """Map each equity session date to 4pm New York in UTC."""
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
        tolerance=pd.Timedelta(minutes=1),
    )
    aligned.index = equity_close_times.index
    return aligned.rename("btc_close_at_equity_close")


def rolling_beta(y: pd.Series, x: pd.Series, lookback: int) -> pd.Series:
    """Rolling covariance(y, x) / variance(x)."""
    covariance = y.rolling(lookback).cov(x)
    variance = x.rolling(lookback).var()
    return covariance / variance


def build_upro_residual_strategy_returns(
    *,
    start_date: str,
    end_date: str | None = None,
    beta_lookback: int = 40,
    zscore_lookback: int = 20,
    entry_zscore: float = 1.5,
) -> pd.Series:
    """
    Daily strategy returns: long UPRO when prior close had residual z-score >= entry.

    Matches the baseline rule in ``notebooks/examples/btc-qqq-residual-upro-backtest.py``:
    signal at close t, position earns UPRO return from t to t+1.
    """
    end = pd.Timestamp.today(tz="UTC").date().isoformat() if end_date is None else end_date

    equity_closes = download_massive_equity_closes(["QQQ", "UPRO"], start_date, end)
    btc_hourly = download_btc_hourly_closes(start_date, end)
    if equity_closes.empty or btc_hourly.empty:
        raise ValueError("Missing QQQ/UPRO or BTC-USD data for UPRO residual signal.")

    equity_close_times = build_equity_close_times(equity_closes.index)
    btc_close = align_btc_to_equity_close(btc_hourly, equity_close_times)

    prices = equity_closes.join(btc_close, how="inner").dropna()
    if prices.empty:
        raise ValueError("Could not align BTC to equity closes for UPRO residual signal.")

    returns = prices.rename(columns={"btc_close_at_equity_close": "BTC"}).pct_change()
    returns = returns.rename(
        columns={
            "QQQ": "QQQ_return",
            "UPRO": "UPRO_return",
            "BTC": "BTC_return",
        }
    ).dropna()

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

    out = frame.dropna(subset=["beta", "residual", "zscore"])["strategy_return"]
    return out.rename("upro_residual")
