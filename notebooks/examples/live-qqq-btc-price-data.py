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
# # Live QQQ and BTC Price Data Smoke Test
#
# ## Research Question
#
# Can the research notebook pipeline pull live daily price bars for QQQ and BTC
# from Massive using the `MASSIVE_API_KEY` Actions secret?
#
# ## Hypothesis
#
# If the secret is available to the render environment, Massive's aggregate bars
# endpoint should return recent, non-empty OHLCV data for both `QQQ` and
# `X:BTCUSD`.

# %%
from __future__ import annotations

import os
from datetime import UTC, date, datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import requests
from IPython.display import Markdown, display

# %% [markdown]
# ## Assumptions
#
# - The Massive API key is supplied through `MASSIVE_API_KEY`; the notebook never
#   stores or prints the secret value.
# - The smoke test uses daily aggregate bars because the endpoint works for both
#   equities and crypto with the same request shape.
# - `QQQ` may omit weekends and market holidays, while BTC should trade every
#   calendar day.
# - Local renders without the secret skip the live request and clearly report
#   that no live test was executed. Renders with the secret fail if either symbol
#   returns no bars.
#
# ## Data Sources
#
# - Massive stocks aggregates endpoint for `QQQ`.
# - Massive crypto aggregates endpoint for `X:BTCUSD`.

# %%
BASE_URL = "https://api.massive.com"
SYMBOLS = {
    "QQQ": "Invesco QQQ Trust ETF",
    "X:BTCUSD": "Bitcoin / U.S. Dollar",
}
LOOKBACK_DAYS = 30
REQUEST_TIMEOUT_SECONDS = 30


def date_window(today: date | None = None, lookback_days: int = LOOKBACK_DAYS) -> tuple[str, str]:
    """Return an inclusive calendar window for recent aggregate bars."""
    end_date = today or datetime.now(UTC).date()
    start_date = end_date - timedelta(days=lookback_days)
    return start_date.isoformat(), end_date.isoformat()


def fetch_massive_aggregates(
    ticker: str,
    api_key: str,
    start_date: str,
    end_date: str,
    *,
    multiplier: int = 1,
    timespan: str = "day",
) -> pd.DataFrame:
    """Fetch Massive aggregate bars and normalize the response into a DataFrame."""
    endpoint = (
        f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/"
        f"{multiplier}/{timespan}/{start_date}/{end_date}"
    )
    response = requests.get(
        endpoint,
        params={
            "adjusted": "true",
            "sort": "asc",
            "limit": 50_000,
            "apiKey": api_key,
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("results", [])
    if not rows:
        return pd.DataFrame()

    data = pd.DataFrame(rows).rename(
        columns={
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "vw": "volume_weighted_average_price",
            "n": "transactions",
        }
    )
    data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms", utc=True)
    data["ticker"] = payload.get("ticker", ticker)
    data["description"] = SYMBOLS[ticker]
    return data[
        [
            "ticker",
            "description",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "volume_weighted_average_price",
            "transactions",
        ]
    ]


start_date, end_date = date_window()
api_key = os.environ.get("MASSIVE_API_KEY")
has_api_key = bool(api_key)

# %% [markdown]
# ## Methodology
#
# 1. Build a recent daily-bar request window.
# 2. Pull `QQQ` and `X:BTCUSD` aggregate bars from Massive when the secret is
#    present.
# 3. Assert that both symbols return at least one bar during secret-backed
#    renders.
# 4. Summarize latest close, freshness, and recent return behavior.

# %%
if not has_api_key:
    display(
        Markdown(
            "**Live Massive request skipped:** set `MASSIVE_API_KEY` in the render "
            "environment to execute the QQQ and BTC smoke test."
        )
    )
    prices = pd.DataFrame()
else:
    frames = [
        fetch_massive_aggregates(ticker, api_key, start_date, end_date)
        for ticker in SYMBOLS
    ]
    prices = pd.concat(frames, ignore_index=True)

prices.head()

# %%
if has_api_key:
    returned_tickers = set(prices.get("ticker", []))
    missing_tickers = set(SYMBOLS) - returned_tickers
    assert not missing_tickers, f"Massive returned no bars for: {sorted(missing_tickers)}"
    assert prices["close"].notna().all(), "Close prices should be populated for all bars."

test_status = pd.DataFrame(
    {
        "check": [
            "MASSIVE_API_KEY present",
            "QQQ bars returned",
            "BTC bars returned",
            "close prices populated",
        ],
        "passed": [
            has_api_key,
            has_api_key and "QQQ" in set(prices.get("ticker", [])),
            has_api_key and "X:BTCUSD" in set(prices.get("ticker", [])),
            has_api_key and not prices.empty and prices["close"].notna().all(),
        ],
    }
)
test_status

# %% [markdown]
# ## Analysis
#
# The table below is populated only when the notebook renders with
# `MASSIVE_API_KEY`. In GitHub Actions, this verifies that the newly added secret
# can authenticate and return live price data for both requested markets.

# %%
if prices.empty:
    latest_summary = pd.DataFrame(
        columns=[
            "ticker",
            "description",
            "latest_timestamp",
            "latest_close",
            "bars_returned",
            "first_timestamp",
        ]
    )
else:
    latest_summary = (
        prices.sort_values("timestamp")
        .groupby(["ticker", "description"], as_index=False)
        .agg(
            latest_timestamp=("timestamp", "last"),
            latest_close=("close", "last"),
            bars_returned=("close", "size"),
            first_timestamp=("timestamp", "first"),
        )
    )

latest_summary

# %%
if prices.empty:
    returns = pd.DataFrame()
else:
    returns = prices.sort_values(["ticker", "timestamp"]).copy()
    returns["daily_return"] = returns.groupby("ticker")["close"].pct_change()
    returns["cumulative_return"] = returns.groupby("ticker")["daily_return"].transform(
        lambda series: (1 + series.fillna(0)).cumprod() - 1
    )

returns.tail(10)

# %% [markdown]
# ## Visualizations
#
# These plots render when live data is available. They are intentionally simple:
# the goal is to confirm data access and shape rather than to draw a trading
# conclusion.

# %%
if prices.empty:
    display(Markdown("No live price rows available for plotting in this render."))
else:
    fig, ax = plt.subplots()
    for ticker, group in returns.groupby("ticker"):
        ax.plot(group["timestamp"], group["close"], marker="o", label=ticker)

    ax.set_title("Recent Massive Daily Closes")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close")
    ax.legend()
    fig.autofmt_xdate()
    plt.show()

# %%
if not returns.empty:
    fig, ax = plt.subplots()
    for ticker, group in returns.groupby("ticker"):
        ax.plot(group["timestamp"], group["cumulative_return"], marker="o", label=ticker)

    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Recent Cumulative Return by Symbol")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return")
    ax.legend()
    fig.autofmt_xdate()
    plt.show()

# %% [markdown]
# ## Limitations
#
# - This is a connectivity and schema smoke test, not a strategy notebook.
# - Daily bars hide intraday gaps and latency characteristics.
# - QQQ trades on the U.S. market calendar, while BTC trades continuously, so row
#   counts differ by design.
# - The notebook does not persist raw responses or intermediate outputs.
#
# ## Conclusion
#
# When `MASSIVE_API_KEY` is available, this notebook pulls live Massive price bars
# for both QQQ and BTC, asserts that each symbol returns close prices, and renders
# summary tables and plots for a quick pipeline check. Local renders without the
# secret remain publishable but mark the live test as skipped.
#
# ## Next Research Ideas
#
# - Add intraday bars after confirming the account's plan supports the desired
#   history depth.
# - Compare Massive close prices against a second vendor for reconciliation.
# - Parameterize the notebook to test additional ETF and crypto pairs.
