"""Data helpers for research notebooks."""

from __future__ import annotations

from io import StringIO

import numpy as np
import pandas as pd
import requests


def make_synthetic_liquidation_data(
    periods: int = 14 * 24 * 4,
    frequency: str = "15min",
    seed: int = 42,
) -> pd.DataFrame:
    """Create reproducible synthetic price and liquidation observations."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2025-01-01", periods=periods, freq=frequency)

    baseline_returns = rng.normal(loc=0.00015, scale=0.004, size=periods)
    price = 100_000 * np.exp(np.cumsum(baseline_returns))

    long_liquidations = rng.gamma(shape=1.8, scale=45_000, size=periods)
    short_liquidations = rng.gamma(shape=1.8, scale=42_000, size=periods)

    event_locations = rng.choice(np.arange(24, periods - 24), size=10, replace=False)
    for location in event_locations:
        direction = rng.choice([-1, 1])
        shock = direction * rng.uniform(0.012, 0.03)
        baseline_returns[location] += shock

        if direction < 0:
            long_liquidations[location : location + 3] += rng.uniform(900_000, 1_800_000)
            baseline_returns[location + 1 : location + 5] += rng.normal(0.0025, 0.003, size=4)
        else:
            short_liquidations[location : location + 3] += rng.uniform(800_000, 1_600_000)
            baseline_returns[location + 1 : location + 5] += rng.normal(-0.0022, 0.003, size=4)

    price = 100_000 * np.exp(np.cumsum(baseline_returns))
    return pd.DataFrame(
        {
            "timestamp": index,
            "price": price,
            "return": baseline_returns,
            "long_liquidations_usd": long_liquidations,
            "short_liquidations_usd": short_liquidations,
        }
    )


def download_stooq_daily_closes(
    tickers: list[str],
    *,
    start_date: str = "2004-01-01",
    end_date: str | None = None,
) -> pd.DataFrame:
    """Download daily close data from Stooq for the provided U.S. tickers."""
    close_series: list[pd.Series] = []
    end_timestamp = pd.Timestamp.today().normalize() if end_date is None else pd.Timestamp(end_date)
    start_timestamp = pd.Timestamp(start_date)

    for ticker in tickers:
        symbol = f"{ticker.lower()}.us"
        url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        frame = pd.read_csv(StringIO(response.text))
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame = frame.dropna(subset=["Date", "Close"]).sort_values("Date")
        frame = frame.loc[
            (frame["Date"] >= start_timestamp) & (frame["Date"] <= end_timestamp),
            ["Date", "Close"],
        ]
        series = frame.set_index("Date")["Close"].rename(ticker.upper())
        close_series.append(series)

    if not close_series:
        return pd.DataFrame()

    return pd.concat(close_series, axis=1).sort_index().dropna(how="all")
