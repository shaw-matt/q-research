"""Preload Massive flat-file datasets used by rendered notebooks."""

from __future__ import annotations

import sys
from datetime import UTC
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from research.data import download_massive_daily_closes
from research.massive_flatfiles import (
    download_flatfile_btc_hourly_closes,
    download_flatfile_stock_day_closes,
)


SPY_TLT_START_DATE = "2004-01-01"
UPRO_RESIDUAL_START_DATE = "2023-01-01"


def main() -> None:
    end = pd.Timestamp.now(tz=UTC).date().isoformat()

    datasets = [
        (
            "SPY/TLT daily closes",
            lambda: download_massive_daily_closes(
                ["SPY", "TLT"],
                start_date=SPY_TLT_START_DATE,
                end_date=end,
            ),
        ),
        (
            "TLT daily closes",
            lambda: download_massive_daily_closes(
                ["TLT"],
                start_date=SPY_TLT_START_DATE,
                end_date=end,
            ),
        ),
        (
            "QQQ/UPRO daily closes",
            lambda: download_flatfile_stock_day_closes(
                ["QQQ", "UPRO"],
                UPRO_RESIDUAL_START_DATE,
                end,
            ),
        ),
        (
            "BTC hourly closes",
            lambda: download_flatfile_btc_hourly_closes(
                UPRO_RESIDUAL_START_DATE,
                end,
            ),
        ),
    ]

    for label, load in datasets:
        print(f"Warming {label}...", flush=True)
        data = load()
        print(f"Warmed {label}: {len(data):,} row(s).", flush=True)


if __name__ == "__main__":
    main()
