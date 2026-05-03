"""On-disk cache for Massive flat-file downloads (stocks and BTC hourly)."""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd


def enabled() -> bool:
    """Disable with Q_RESEARCH_FLATFILE_CACHE=0|false|no."""
    raw = os.getenv("Q_RESEARCH_FLATFILE_CACHE")
    if raw is None:
        return True
    v = raw.strip().lower()
    if not v:
        return True
    return v not in ("0", "false", "no")


def cache_root() -> Path:
    return Path(os.getenv("Q_RESEARCH_FLATFILE_CACHE_DIR", ".cache/q-research/flatfiles"))


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def stock_cache_path(tickers_u: list[str], start: date) -> Path:
    slug = "+".join(sorted(tickers_u))
    return cache_root() / f"stock_day_{slug}_{start.isoformat()}.parquet"


def try_load_stock_frame(path: Path, tickers_u: list[str]) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError):
        return None
    if frame.empty:
        return None
    frame = frame.sort_index()
    frame.index = pd.to_datetime(frame.index).normalize()
    missing = [t for t in tickers_u if t not in frame.columns]
    if missing:
        return None
    return frame


def save_stock_frame(path: Path, frame: pd.DataFrame, tickers_u: list[str]) -> None:
    _ensure_dir(path)
    out = frame.sort_index()
    out.index = pd.to_datetime(out.index).normalize()
    out = out[[c for c in tickers_u if c in out.columns]]
    out.to_parquet(path)


def btc_cache_path(start_date: str) -> Path:
    return cache_root() / f"btc_hourly_{start_date}.parquet"


def try_load_btc_series(path: Path) -> pd.Series | None:
    if not path.exists():
        return None
    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError):
        return None
    if frame.empty or "close" not in frame.columns:
        return None
    series = frame["close"].sort_index()
    if series.index.tz is None:
        series.index = series.index.tz_localize("UTC")
    else:
        series.index = series.index.tz_convert("UTC")
    return series


def save_btc_series(path: Path, series: pd.Series) -> None:
    _ensure_dir(path)
    s = series.sort_index()
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    else:
        s.index = s.index.tz_convert("UTC")
    s.to_frame(name="close").to_parquet(path)


def merge_stock_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=0).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def merge_btc_series(parts: list[pd.Series]) -> pd.Series:
    if not parts:
        return pd.Series(dtype=float)
    out = pd.concat(parts, axis=0).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def stock_ranges_to_download(
    cached: pd.DataFrame | None, start: date, end: date
) -> list[tuple[date, date]]:
    """Ordered list of disjoint inclusive date ranges to fetch from S3."""
    if cached is None or cached.empty:
        return [(start, end)]
    c_min = cached.index.min().date()
    c_max = cached.index.max().date()
    ranges: list[tuple[date, date]] = []
    if c_min > start:
        ranges.append((start, c_min - timedelta(days=1)))
    if c_max < end:
        ranges.append((c_max + timedelta(days=1), end))
    if not ranges:
        return []
    return ranges


def btc_ranges_to_download(
    cached: pd.Series | None, start: date, end: date
) -> list[tuple[date, date]]:
    if cached is None or cached.empty:
        return [(start, end)]
    c_min = cached.index.min().date()
    c_max = cached.index.max().date()
    ranges: list[tuple[date, date]] = []
    if c_min > start:
        ranges.append((start, c_min - timedelta(days=1)))
    if c_max < end:
        ranges.append((c_max + timedelta(days=1), end))
    if not ranges:
        return []
    return ranges
