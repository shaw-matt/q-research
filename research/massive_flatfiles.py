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

from research import flatfile_cache

NY_TZ = ZoneInfo("America/New_York")
DEFAULT_FILES_ENDPOINT = "https://files.massive.com"


def _flatfiles_bucket() -> str:
    return os.getenv("MASSIVE_FLATFILE_BUCKET", "flatfiles")


def get_massive_flatfile_s3_client():
    """
    S3-compatible client for Massive flat files.

    Returns None if access keys are not set.
    Also accepts Massive gist env names: MASSIVE_ACCESS_KEY / MASSIVE_SECRET_KEY.
    """
    access = (
        os.getenv("MASSIVE_S3_ACCESS_KEY_ID")
        or os.getenv("MASSIVE_ACCESS_KEY")
        or os.getenv("AWS_ACCESS_KEY_ID")
    )
    secret = (
        os.getenv("MASSIVE_S3_SECRET_ACCESS_KEY")
        or os.getenv("MASSIVE_SECRET_KEY")
        or os.getenv("AWS_SECRET_ACCESS_KEY")
    )
    if not access or not secret:
        return None

    endpoint = os.getenv("MASSIVE_FILES_ENDPOINT", DEFAULT_FILES_ENDPOINT).rstrip("/")
    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
    addressing = os.getenv("MASSIVE_S3_ADDRESSING_STYLE", "path")

    return boto3.client(
        "s3",
        region_name=region,
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        endpoint_url=endpoint,
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": addressing},
            connect_timeout=15,
            read_timeout=120,
        ),
    )


def _require_s3_client():
    client = get_massive_flatfile_s3_client()
    if client is None:
        raise ValueError(
            "Massive flat files require S3 credentials: set MASSIVE_S3_ACCESS_KEY_ID and "
            "MASSIVE_S3_SECRET_ACCESS_KEY (or MASSIVE_ACCESS_KEY / MASSIVE_SECRET_KEY, or "
            "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)."
        )
    return client


def _read_s3_gzip_csv(client, bucket: str, key: str) -> pd.DataFrame | None:
    try:
        response = client.get_object(Bucket=bucket, Key=key)
        raw = response["Body"].read()
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        # Wrong path often returns 403 (no ListBucket) instead of 404.
        if code in ("404", "NoSuchKey", "NotFound", "403", "AccessDenied"):
            return None
        raise
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
        return pd.read_csv(gz)


def _stock_day_agg_key_candidates(day: date) -> list[str]:
    y, m, ds = day.year, day.month, day.isoformat()
    return [
        f"us_stocks_sip/day_aggs_v1/{y}/{m:02d}/{ds}.csv.gz",
        f"us_stocks_sip/day_aggs_v1/{y}/{ds}.csv.gz",
    ]


def _crypto_minute_agg_keys(day: date) -> list[str]:
    y, m, ds = day.year, day.month, day.isoformat()
    return [
        f"global_crypto/minute_aggs_v1/{y}/{m:02d}/{ds}.csv.gz",
        f"crypto/minute_aggs_v1/{y}/{m:02d}/{ds}.csv.gz",
    ]


def _load_stock_day_from_s3(client, bucket: str, day: date) -> pd.DataFrame | None:
    for key in _stock_day_agg_key_candidates(day):
        df = _read_s3_gzip_csv(client, bucket, key)
        if df is not None and not df.empty:
            return df
    return None


def _download_stock_days_s3(
    client,
    bucket: str,
    tickers_u: list[str],
    start: date,
    end: date,
) -> pd.DataFrame:
    want = set(tickers_u)
    records: list[dict[str, object]] = []

    for ts in pd.date_range(start, end, freq="D"):
        day = ts.date()
        df = _load_stock_day_from_s3(client, bucket, day)
        if df is None or df.empty:
            continue
        if "ticker" not in df.columns or "close" not in df.columns:
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
    return frame.pivot_table(index="date", columns="ticker", values="close", aggfunc="last").sort_index()


def download_flatfile_stock_day_closes(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Load daily equity closes from Massive S3 flat files only (US stock day aggregates)."""
    tickers_u = [t.upper() for t in tickers]
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()

    client = _require_s3_client()
    bucket = _flatfiles_bucket()
    cache_path = flatfile_cache.stock_cache_path(tickers_u, start)

    cached: pd.DataFrame | None = None
    if flatfile_cache.enabled():
        cached = flatfile_cache.try_load_stock_frame(cache_path, tickers_u)

    ranges = flatfile_cache.stock_ranges_to_download(cached, start, end)
    pieces: list[pd.DataFrame] = []
    for range_start, range_end in ranges:
        part = _download_stock_days_s3(client, bucket, tickers_u, range_start, range_end)
        if not part.empty:
            pieces.append(part)
    if cached is not None and not cached.empty:
        pieces.append(cached)

    wide_full = flatfile_cache.merge_stock_frames(pieces)
    if wide_full.empty:
        raise ValueError(
            f"No US stock day-aggregate rows found for {tickers_u} between {start_date} and {end_date}. "
            "Confirm flat-file subscription, S3 credentials, MASSIVE_FILES_ENDPOINT, "
            "MASSIVE_FLATFILE_BUCKET, and MASSIVE_S3_ADDRESSING_STYLE (path vs virtual)."
        )
    wide_full = wide_full.sort_index()
    wide_full.index = pd.to_datetime(wide_full.index).normalize()
    wide_full = wide_full[[t for t in tickers_u if t in wide_full.columns]]

    out = wide_full.loc[
        (wide_full.index >= pd.Timestamp(start)) & (wide_full.index <= pd.Timestamp(end)),
        :,
    ]
    if flatfile_cache.enabled():
        flatfile_cache.save_stock_frame(cache_path, wide_full, tickers_u)

    return out


def _download_btc_hourly_s3(client, bucket: str, start: date, end: date, start_date: str) -> pd.Series:
    chunks: list[pd.DataFrame] = []

    for ts in pd.date_range(start, end, freq="D"):
        day = ts.date()
        df = None
        for key in _crypto_minute_agg_keys(day):
            df = _read_s3_gzip_csv(client, bucket, key)
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


def download_flatfile_btc_hourly_closes(start_date: str, end_date: str) -> pd.Series:
    """Build hourly BTC-USD closes from global crypto minute flat files only."""
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()

    client = _require_s3_client()
    bucket = _flatfiles_bucket()
    cache_path = flatfile_cache.btc_cache_path(start_date)

    cached: pd.Series | None = None
    if flatfile_cache.enabled():
        cached = flatfile_cache.try_load_btc_series(cache_path)

    ranges = flatfile_cache.btc_ranges_to_download(cached, start, end)
    parts: list[pd.Series] = []
    for range_start, range_end in ranges:
        part = _download_btc_hourly_s3(client, bucket, range_start, range_end, start_date)
        if not part.empty:
            parts.append(part)
    if cached is not None and not cached.empty:
        parts.append(cached)

    merged = flatfile_cache.merge_btc_series(parts)
    if merged.empty:
        raise ValueError(
            f"No crypto minute flat-file bars for X:BTC-USD between {start_date} and {end_date}. "
            "Confirm flat-file subscription, S3 credentials, and key prefix "
            "(global_crypto/minute_aggs_v1 or crypto/minute_aggs_v1)."
        )
    merged = merged.loc[merged.index >= pd.Timestamp(start_date, tz="UTC")]
    if flatfile_cache.enabled():
        flatfile_cache.save_btc_series(cache_path, merged)
    return merged


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


def probe_stock_flatfile_keys(day: date) -> list[str]:
    """Return S3 keys that successfully download for ``day`` (for debugging)."""
    client = get_massive_flatfile_s3_client()
    if client is None:
        return []
    bucket = _flatfiles_bucket()
    ok: list[str] = []
    for key in _stock_day_agg_key_candidates(day):
        df = _read_s3_gzip_csv(client, bucket, key)
        if df is not None and not df.empty:
            ok.append(key)
    return ok
