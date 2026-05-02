#!/usr/bin/env python3
"""Verify Massive S3 flat-file access for US stock day aggregates."""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    from research.massive_flatfiles import (
        download_flatfile_stock_day_closes,
        get_massive_flatfile_s3_client,
        probe_stock_flatfile_keys,
    )

    probe_day = date.today() - timedelta(days=9)
    while probe_day.weekday() >= 5:
        probe_day -= timedelta(days=1)

    client = get_massive_flatfile_s3_client()
    if not client:
        print(
            "FAIL: S3 credentials not set. Set MASSIVE_S3_ACCESS_KEY_ID and "
            "MASSIVE_S3_SECRET_ACCESS_KEY (or MASSIVE_ACCESS_KEY / MASSIVE_SECRET_KEY / AWS_*)."
        )
        return 1

    keys = probe_stock_flatfile_keys(probe_day)
    if keys:
        print("S3 flat files: OK for", probe_day.isoformat(), "via", keys[0])
    else:
        print(
            "S3: no readable day-aggregate object for",
            probe_day.isoformat(),
            "(check bucket, MASSIVE_FILES_ENDPOINT, MASSIVE_S3_ADDRESSING_STYLE, subscription)",
        )

    try:
        df = download_flatfile_stock_day_closes(
            ["QQQ", "SPY"], probe_day.isoformat(), probe_day.isoformat()
        )
    except ValueError as exc:
        print("FAIL:", exc)
        return 1
    if df.empty:
        print("FAIL: download_flatfile_stock_day_closes returned empty.")
        return 1
    print("download_flatfile_stock_day_closes: OK", df.shape, list(df.columns))
    return 0


if __name__ == "__main__":
    sys.exit(main())
