"""Massive (Polygon) S3 flat files — day stock aggregates and minute crypto aggregates."""

from __future__ import annotations

import gzip
import io
import os
from datetime import date, datetime

import boto3
import pandas as pd
from botocore.client import Config
from botocore.exceptions import ClientError
from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")
FLATFILES_BUCKET = "flatfiles"
DEFAULT_FILES_ENDPOINT = "https://files.massive.com"


def get_massive_flatfile_s3_client():
    """S3-compatible client for Massive flat files (separate S3 keys from REST API key)."""
    access = os.getenv("MASSIVE_S3_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID")
    secret = os.getenv("MASSIVE_S3_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY")
    if not access or not secret:
        raise ValueError(
            "Set MASSIVE_S3_ACCESS_KEY_ID and MASSIVE_S3_SECRET_ACCESS_KEY with your Massive "
            "flat-file S3 credentials (from the Massive dashboard), or set AWS_ACCESS_KEY_ID "
            "and AWS_SECRET_ACCESS_KEY to the same values."
        )
    endpoint = os.getenv("MASSIVE_FILES_ENDPOINT", DEFAULT_FILES_ENDPOINT).rstrip("/")
    return boto3.client(
        "s3",
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        endpoint_url=endpoint,
        config=Config(signature_version="s3v4"),
    )


def _read_s3_gzip_csv(client, key: str) -> pd.DataFrame | None:
    try:
        response = client.get_object(Bucket=FLATFILES_BUCKET, Key=key)
        raw = response["Body"].read()
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            return None
        raise
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
        return pd.read_csv(gz)


def _stock_day_agg_key(day: date) -> str:
    return f"us_stocks_sip/day_aggs_v1/{day.year}/{day.month:02d}/{day.isoformat()}.csv.gz"


def _crypto_minute_agg_keys(day: date) -> list[str]:
    y, m, ds = day.year, day.month, day.isoformat()
    return [
        f"global_crypto/minute_aggs_v1/{y}/{m:02d}/{ds}.csv.gz",
        f"crypto/minute_aggs_v1/{y}/{m:02d}/{ds}.csv.gz",
    ]


def download_flatfile_stock_day_closes(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Load adjusted daily equity closes from per-day US stock day-aggregate flat files.

    One S3 GET per calendar day in range (missing weekends/holidays return no object).
    """
    tickers_u = [t.upper() for t in tickers]
    want = set(tickers_u)
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    client = get_massive_flatfile_s3_client()
    records: list[dict[str, object]] = []

    for ts in pd.date_range(start, end, freq="D"):
        day = ts.date()
        df = _read_s3_gzip_csv(client, _stock_day_agg_key(day))
        if df is None or df.empty:
            continue
        sub = df.loc[df["ticker"].isin(want), ["ticker", "window_start", "close"]]
        if sub.empty:
            continue
        for ticker in tickers_u:
            rows = sub.loc[sub["ticker"] == ticker]
            if rows.empty:
                continue
            row = rows.iloc[-1]
            bar_start = pd.Timestamp(int(row["window_start"]), unit="ns", tz="UTC")
            session = bar_start.tz_convert("America/New_York").normalize().tz_localize(None)
            records.append({"date": session, "ticker": ticker, "close": float(row["close"])})

    if not records:
        return pd.DataFrame()

    frame = pd.DataFrame.from_records(records)
    wide = frame.pivot_table(index="date", columns="ticker", values="close", aggfunc="last").sort_index()
    return wide


def download_flatfile_btc_hourly_closes(start_date: str, end_date: str) -> pd.Series:
    """
    Build hourly BTC-USD closes from global crypto minute-aggregate flat files.

    Matches REST behavior: bar timestamps mark interval start; relabel to interval end,
    then aggregate minutes to hourly closes (UTC).
    """
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    client = get_massive_flatfile_s3_client()
    chunks: list[pd.DataFrame] = []

    for ts in pd.date_range(start, end, freq="D"):
        day = ts.date()
        df = None
        for key in _crypto_minute_agg_keys(day):
            df = _read_s3_gzip_csv(client, key)
            if df is not None and not df.empty:
                break
        if df is None or df.empty:
            continue
        mask = df["ticker"].isin(["X:BTC-USD", "X:BTCUSD"])
        sub = df.loc[mask, ["window_start", "close"]]
        if sub.empty:
            continue
        chunks.append(sub)

    if not chunks:
        return pd.Series(dtype=float)

    all_minutes = pd.concat(chunks, ignore_index=True).sort_values("window_start")
    idx = pd.to_datetime(all_minutes["window_start"], unit="ns", utc=True) + pd.Timedelta(minutes=1)
    minute_close = pd.Series(all_minutes["close"].to_numpy(), index=idx).sort_index()
    minute_close = minute_close[~minute_close.index.duplicated(keep="last")]
    hourly = minute_close.resample("1h", label="right", closed="right").last().dropna()
    return hourly.loc[hourly.index >= pd.Timestamp(start_date, tz="UTC")]


def build_equity_close_times(equity_dates: pd.Index) -> pd.Series:
    """Map each equity session date to 4pm New York time in UTC."""
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
