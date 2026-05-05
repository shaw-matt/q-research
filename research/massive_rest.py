"""Massive REST API — aggregates for live scripts (notebooks / backtests use S3 flat files)."""

from __future__ import annotations

import os
import time
from datetime import timedelta
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests

DEFAULT_REST_BASE = "https://api.massive.com"


def _rest_base() -> str:
    return (os.getenv("MASSIVE_REST_URL") or DEFAULT_REST_BASE).rstrip("/")


def _api_key() -> str:
    return (
        os.getenv("MASSIVE_API_KEY")
        or os.getenv("POLYGON_API_KEY")
        or os.getenv("POLYGON_API_KEY_ID")
        or ""
    )


def _require_api_key() -> str:
    key = _api_key()
    if not key:
        raise ValueError(
            "Massive REST requires an API key: set MASSIVE_API_KEY or POLYGON_API_KEY "
            "(see https://massive.com/docs/rest/quickstart)."
        )
    return key


def _session_bar_to_ny_session_date(bar: dict[str, Any]) -> pd.Timestamp:
    ts = pd.Timestamp(int(bar["t"]), unit="ms", tz="UTC")
    return ts.tz_convert("America/New_York").normalize().tz_localize(None)


def _fetch_aggs_paginated(
    session: requests.Session,
    initial_url: str,
    params: dict[str, Any],
    *,
    api_key: str,
) -> list[dict[str, Any]]:
    """Follow next_url until exhausted.

    Massive/Polygon ``next_url`` values usually omit ``apiKey``; the first page
    succeeds but the second request would otherwise return **401**. Always
    re-attach the key when following ``next_url`` unless it is already present.
    """
    rows: list[dict[str, Any]] = []
    next_url: str | None = initial_url
    first = True
    while next_url:
        if first:
            response = session.get(next_url, params=params, timeout=120)
            first = False
        else:
            follow_params = None
            if "apiKey=" not in next_url and "apikey=" not in next_url.lower():
                follow_params = {"apiKey": api_key}
            response = session.get(next_url, params=follow_params, timeout=120)
        if response.status_code == 401:
            raise ValueError(
                "Massive REST returned 401 Unauthorized. "
                "Confirm MASSIVE_API_KEY or POLYGON_API_KEY, and that your subscription "
                "includes crypto aggregates. If only BTC failed after stocks worked, "
                "ensure pagination sends apiKey on next_url (this client re-attaches it)."
            )
        response.raise_for_status()
        payload = response.json()
        status = payload.get("status")
        if status not in ("OK", "DELAYED"):
            raise ValueError(f"Massive REST unexpected status {status!r}: {payload}")
        rows.extend(payload.get("results") or [])
        next_url = payload.get("next_url")
        if next_url:
            time.sleep(0.05)
    return rows


def download_rest_stock_day_closes(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Daily adjusted closes from ``GET /v2/aggs/ticker/{ticker}/range/1/day/{from}/{to}``.

    Index: NY session calendar dates (aligned with flat-file convention in ``massive_flatfiles``).
    """
    tickers_u = [t.upper() for t in tickers]
    key = _require_api_key()
    base = _rest_base()
    session = requests.Session()

    series_list: list[pd.Series] = []
    for ticker in tickers_u:
        enc = quote(ticker, safe="")
        url = f"{base}/v2/aggs/ticker/{enc}/range/1/day/{start_date}/{end_date}"
        params: dict[str, Any] = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": key,
        }
        bars = _fetch_aggs_paginated(session, url, params, api_key=key)
        if not bars:
            raise ValueError(f"No REST daily bars returned for {ticker} between {start_date} and {end_date}.")
        by_date: dict[pd.Timestamp, float] = {}
        for bar in bars:
            d = _session_bar_to_ny_session_date(bar)
            by_date[d] = float(bar["c"])
        s = pd.Series(by_date, name=ticker).sort_index()
        series_list.append(s)
        time.sleep(0.05)

    frame = pd.concat(series_list, axis=1).sort_index()
    return frame


def download_rest_crypto_hourly_closes(
    ticker: str,
    start_date: str,
    end_date: str,
    *,
    chunk_days: int = 120,
) -> pd.Series:
    """
    Hourly closes from ``GET /v2/aggs/ticker/{ticker}/range/1/hour/{from}/{to}``.

    Long histories are requested in chunks to stay under aggregate ``limit``.
    Index: UTC timestamps at bar start (milliseconds from API).
    """
    key = _require_api_key()
    base = _rest_base()
    session = requests.Session()
    enc = quote(ticker, safe="")

    start_d = pd.Timestamp(start_date).date()
    end_d = pd.Timestamp(end_date).date()
    all_bars: list[dict[str, Any]] = []
    cursor = start_d
    while cursor <= end_d:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), end_d)
        url = (
            f"{base}/v2/aggs/ticker/{enc}/range/1/hour/"
            f"{cursor.isoformat()}/{chunk_end.isoformat()}"
        )
        params: dict[str, Any] = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": key,
        }
        chunk = _fetch_aggs_paginated(session, url, params, api_key=key)
        all_bars.extend(chunk)
        cursor = chunk_end + timedelta(days=1)
        time.sleep(0.05)

    if not all_bars:
        raise ValueError(f"No REST hourly bars returned for {ticker} between {start_date} and {end_date}.")

    idx = [pd.Timestamp(int(b["t"]), unit="ms", tz="UTC") for b in all_bars]
    vals = [float(b["c"]) for b in all_bars]
    series = pd.Series(vals, index=pd.DatetimeIndex(idx, name="timestamp")).sort_index()
    series = series[~series.index.duplicated(keep="last")]
    return series.loc[series.index >= pd.Timestamp(start_date, tz="UTC")]
