"""BTC/QQQ residual z-score long UPRO strategy (Massive flat-file data)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.massive_flatfiles import (
    align_btc_to_equity_close,
    build_equity_close_times,
    download_flatfile_btc_hourly_closes,
    download_flatfile_stock_day_closes,
)


def rolling_beta(y: pd.Series, x: pd.Series, lookback: int) -> pd.Series:
    """Rolling covariance(y, x) / variance(x)."""
    covariance = y.rolling(lookback).cov(x)
    variance = x.rolling(lookback).var()
    return covariance / variance


def build_upro_residual_strategy_frame(
    *,
    start_date: str,
    end_date: str | None = None,
    beta_lookback: int = 40,
    zscore_lookback: int = 20,
    entry_zscore: float = 1.5,
) -> pd.DataFrame:
    """
    Strategy returns plus 0/1 UPRO exposure (same index as ``strategy_return``).

    Long UPRO when the prior close had residual z-score >= ``entry_zscore``.
    """
    end = pd.Timestamp.today(tz="UTC").date().isoformat() if end_date is None else end_date

    equity_closes = download_flatfile_stock_day_closes(["QQQ", "UPRO"], start_date, end)
    btc_hourly = download_flatfile_btc_hourly_closes(start_date, end)
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

    out = frame.dropna(subset=["beta", "residual", "zscore"])
    return out[["strategy_return", "position"]].assign(
        upro_exposure=lambda df: df["position"].astype(float)
    ).drop(columns=["position"])


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
    frame = build_upro_residual_strategy_frame(
        start_date=start_date,
        end_date=end_date,
        beta_lookback=beta_lookback,
        zscore_lookback=zscore_lookback,
        entry_zscore=entry_zscore,
    )
    return frame["strategy_return"].rename("upro_residual")
