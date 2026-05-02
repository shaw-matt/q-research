#!/usr/bin/env python3
"""Verify Massive data access: S3 flat files first, then REST grouped daily."""

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
    if client:
        keys = probe_stock_flatfile_keys(probe_day)
        if keys:
            print("S3 flat files: OK for", probe_day.isoformat(), "via", keys[0])
        else:
            print(
                "S3 flat files: no readable object for",
                probe_day.isoformat(),
                "(check keys, bucket, MASSIVE_FILES_ENDPOINT, MASSIVE_S3_ADDRESSING_STYLE)",
            )
    else:
        print("S3: credentials not set (MASSIVE_S3_* / MASSIVE_ACCESS_KEY / AWS_*)")

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
    print("Combined download: OK", df.shape, list(df.columns))
    return 0


if __name__ == "__main__":
    sys.exit(main())
