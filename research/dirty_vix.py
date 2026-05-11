"""Dirty VIX futures cheapness signal built from public volatility data."""

from __future__ import annotations

import os
import re
from datetime import UTC
from io import StringIO
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

DEFAULT_PUBLIC_VOL_CACHE_DIR = Path(".cache/q-research/public-vol-data")


def _cache_dir(cache_dir: Path | str | None = None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir)
    return Path(os.getenv("Q_RESEARCH_PUBLIC_VOL_CACHE_DIR", DEFAULT_PUBLIC_VOL_CACHE_DIR))


def cache_path(*parts: str, cache_dir: Path | str | None = None) -> Path:
    safe_parts = [part.replace("/", "_").replace(":", "_") for part in parts]
    return _cache_dir(cache_dir).joinpath(*safe_parts)


def read_cached_frame(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def write_cached_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)


def download_cboe_index_history(
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    cache_dir: Path | str | None = None,
) -> pd.Series:
    """Download a public Cboe daily index history CSV."""
    symbol_u = symbol.upper().lstrip("^")
    path = cache_path(
        "cboe-index",
        f"{symbol_u}_{start_date}_{end_date}.parquet",
        cache_dir=cache_dir,
    )
    cached = read_cached_frame(path)
    if cached is not None:
        return cached["close"].rename(symbol_u)

    url = f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{symbol_u}_History.csv"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    raw = pd.read_csv(StringIO(response.text))
    raw.columns = [str(column).strip().lower() for column in raw.columns]
    if "date" not in raw.columns or "close" not in raw.columns:
        raise ValueError(f"Cboe {symbol_u} CSV did not contain date/close columns.")
    frame = raw[["date", "close"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["date", "close"]).drop_duplicates("date", keep="last")
    frame = frame.sort_values("date").set_index("date")
    frame = frame.loc[
        (frame.index >= pd.Timestamp(start_date)) & (frame.index <= pd.Timestamp(end_date))
    ]
    if frame.empty:
        raise ValueError(f"No Cboe {symbol_u} rows between {start_date} and {end_date}.")
    write_cached_frame(path, frame)
    return frame["close"].rename(symbol_u)


def download_yahoo_daily_close(
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    cache_dir: Path | str | None = None,
) -> pd.Series:
    """Download daily closes from Yahoo's public chart endpoint."""
    path = cache_path(
        "yahoo-chart",
        f"{symbol}_{start_date}_{end_date}.parquet",
        cache_dir=cache_dir,
    )
    cached = read_cached_frame(path)
    if cached is not None:
        return cached["close"].rename(symbol)

    period1 = int(pd.Timestamp(start_date, tz="UTC").timestamp())
    # Yahoo's period2 is exclusive. Add one day so the requested end date can appear.
    period2 = int((pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)).timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol, safe='')}"
    response = requests.get(
        url,
        params={
            "period1": period1,
            "period2": period2,
            "interval": "1d",
            "events": "history",
            "includeAdjustedClose": "true",
        },
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    chart = payload.get("chart") or {}
    if chart.get("error"):
        raise ValueError(f"Yahoo returned an error for {symbol}: {chart['error']}")
    result = (chart.get("result") or [None])[0]
    if not result or not result.get("timestamp"):
        raise ValueError(f"Yahoo returned no daily rows for {symbol}.")

    quote_data = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    timestamps = result["timestamp"]
    closes = quote_data.get("close") or []
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(timestamps, unit="s", utc=True)
            .tz_convert("America/New_York")
            .normalize()
            .tz_localize(None),
            "close": closes,
        }
    )
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["date", "close"]).drop_duplicates("date", keep="last")
    frame = frame.sort_values("date").set_index("date")
    frame = frame.loc[
        (frame.index >= pd.Timestamp(start_date)) & (frame.index <= pd.Timestamp(end_date))
    ]
    if frame.empty:
        raise ValueError(f"No Yahoo {symbol} rows between {start_date} and {end_date}.")
    write_cached_frame(path, frame)
    return frame["close"].rename(symbol)


def safe_public_data_exception_message(exc: BaseException) -> str:
    """Keep API keys out of notebook output and logs."""
    message = str(exc)
    key = (
        os.getenv("MASSIVE_API_KEY")
        or os.getenv("POLYGON_API_KEY")
        or os.getenv("POLYGON_API_KEY_ID")
        or ""
    )
    if key:
        message = message.replace(key, "***")
    return re.sub(r"([?&]apiKey=)[^&\s]+", r"\1***", message)


def load_public_vol_proxy_data(
    start_date: str,
    end_date: str,
    *,
    yahoo_vx_ticker: str = "VX=F",
    cache_dir: Path | str | None = None,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Load VIX3M and a public volatility trade-leg proxy without paid data."""
    status_rows: list[dict[str, str]] = []
    vix3m_series = download_cboe_index_history(
        "VIX3M",
        start_date,
        end_date,
        cache_dir=cache_dir,
    )
    status_rows.append(
        {
            "series": "VIX3M",
            "source": "Cboe public CSV",
            "status": "loaded",
            "message": "Cboe VIX3M_History.csv",
        }
    )

    try:
        proxy = download_yahoo_daily_close(
            yahoo_vx_ticker,
            start_date,
            end_date,
            cache_dir=cache_dir,
        )
        status_rows.append(
            {
                "series": "VX30 proxy",
                "source": "Yahoo Finance",
                "status": "loaded",
                "message": f"{yahoo_vx_ticker} daily close",
            }
        )
    except Exception as exc:
        proxy = download_cboe_index_history("VIX", start_date, end_date, cache_dir=cache_dir)
        status_rows.append(
            {
                "series": "VX30 proxy",
                "source": "Cboe public CSV fallback",
                "status": "loaded",
                "message": (
                    f"Yahoo {yahoo_vx_ticker} unavailable "
                    f"({safe_public_data_exception_message(exc)}); using VIX_History.csv"
                ),
            }
        )

    return vix3m_series, proxy.rename("VX30"), pd.DataFrame(status_rows)


def build_dirty_vix_signal_frame(
    *,
    start_date: str,
    end_date: str | None = None,
    rolling_zscore_days: int = 252,
    min_zscore_obs: int | None = None,
    entry_zscore: float = -1.5,
    execution_lag_sessions: int = 1,
    yahoo_vx_ticker: str = "VX=F",
    cache_dir: Path | str | None = None,
) -> pd.DataFrame:
    """
    Build daily dirty-VIX signal returns and VX30 exposure.

    The rule goes long the VX30 proxy when the prior close's
    ``zscore(log(VX30 / VIX3M))`` is below ``entry_zscore``.
    """
    end = pd.Timestamp.now(tz=UTC).date().isoformat() if end_date is None else end_date
    min_obs = rolling_zscore_days if min_zscore_obs is None else min_zscore_obs
    vix3m, vx30, source_status = load_public_vol_proxy_data(
        start_date,
        end,
        yahoo_vx_ticker=yahoo_vx_ticker,
        cache_dir=cache_dir,
    )

    frame = pd.concat([vx30.rename("VX30"), vix3m.rename("VIX3M")], axis=1).dropna()
    frame = frame.loc[(frame["VX30"] > 0) & (frame["VIX3M"] > 0)].copy()
    if frame.empty:
        raise ValueError("No overlapping positive VX30/VIX3M observations were available.")

    frame["vx30_return"] = frame["VX30"].pct_change()
    frame["cheapness_log_ratio"] = np.log(frame["VX30"] / frame["VIX3M"])
    rolling = frame["cheapness_log_ratio"].rolling(rolling_zscore_days, min_periods=min_obs)
    frame["cheapness_zscore"] = (
        frame["cheapness_log_ratio"] - rolling.mean()
    ) / rolling.std(ddof=0)
    frame["signal_at_close"] = frame["cheapness_zscore"] < entry_zscore
    frame["vx30_exposure"] = (
        frame["signal_at_close"].shift(execution_lag_sessions).fillna(False).astype(float)
    )
    frame["strategy_return"] = (frame["vx30_exposure"] * frame["vx30_return"]).fillna(0.0)

    out = frame.dropna(subset=["cheapness_zscore"])[
        [
            "strategy_return",
            "vx30_exposure",
            "cheapness_zscore",
            "cheapness_log_ratio",
            "VX30",
            "VIX3M",
        ]
    ]
    out.attrs["source_status"] = source_status
    return out
