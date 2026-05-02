"""Massive REST helpers for market data when S3 flat files are unavailable."""

from __future__ import annotations

import os
from datetime import date

import pandas as pd
import requests

MASSIVE_API_BASE = "https://api.massive.com"


def get_massive_api_key() -> str | None:
    return os.getenv("MASSIVE_API_KEY")


def download_grouped_daily_stock_closes(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    One REST request per calendar day; each response includes all US stocks for that session.

    Used as a fallback when flat-file S3 access returns 403 or is not configured.
    """
    api_key = get_massive_api_key()
    if not api_key:
        raise ValueError(
            "MASSIVE_API_KEY is not set. Set it for REST fallback, or fix S3 flat-file credentials."
        )

    want = {t.upper() for t in tickers}
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    records: list[dict[str, object]] = []

    for ts in pd.date_range(start, end, freq="D"):
        day: date = ts.date()
        url = f"{MASSIVE_API_BASE}/v2/aggs/grouped/locale/us/market/stocks/{day.isoformat()}"
        response = requests.get(
            url,
            params={"adjusted": "true", "apiKey": api_key},
            timeout=60,
        )
        if response.status_code != 200:
            continue
        payload = response.json()
        for row in payload.get("results", []):
            sym = row.get("T")
            if sym not in want:
                continue
            close = row.get("c")
            bar_ms = row.get("t")
            if close is None or bar_ms is None:
                continue
            bar_start = pd.Timestamp(int(bar_ms), unit="ms", tz="UTC")
            session = bar_start.tz_convert("America/New_York").normalize().tz_localize(None)
            records.append({"date": session, "ticker": sym, "close": float(close)})

    if not records:
        return pd.DataFrame()

    frame = pd.DataFrame.from_records(records)
    return frame.pivot_table(index="date", columns="ticker", values="close", aggfunc="last").sort_index()


def download_ticker_hourly_crypto_closes(
    crypto_ticker: str,
    start_date: str,
    end_date: str,
) -> pd.Series:
    """Hourly crypto closes via REST aggregates (pagination)."""
    api_key = get_massive_api_key()
    if not api_key:
        raise ValueError("MASSIVE_API_KEY is not set.")

    url = (
        f"{MASSIVE_API_BASE}/v2/aggs/ticker/{crypto_ticker}/range/"
        f"1/hour/{start_date}/{end_date}"
    )
    params: dict[str, str | int] = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50_000,
        "apiKey": api_key,
    }
    rows: list[dict] = []
    while url:
        response = requests.get(url, params=params, timeout=60)
        if response.status_code != 200:
            raise RuntimeError(
                f"Massive crypto aggs failed HTTP {response.status_code}: {response.text[:300]}"
            )
        payload = response.json()
        rows.extend(payload.get("results", []))
        next_url = payload.get("next_url")
        url = next_url or ""
        params = {"apiKey": api_key} if next_url else {}

    if not rows:
        return pd.Series(dtype=float)

    frame = pd.DataFrame.from_records(rows)
    frame["timestamp"] = pd.to_datetime(frame["t"], unit="ms", utc=True)
    closes = frame.set_index("timestamp")["c"].dropna().sort_index()
    closes.index = closes.index + pd.Timedelta(hours=1)
    return closes.loc[closes.index >= pd.Timestamp(start_date, tz="UTC")]
