#!/usr/bin/env python3
"""Export daily equal-weight four-signal portfolio ETF weights (vol-target notebook rules).

Writes a CSV suitable for dashboards or manual trading: net SPY/TLT/UPRO weights per $1
of strategy capital and the model equal-weight portfolio return by session date.

**Default data path is Massive REST** (``MASSIVE_API_KEY`` / ``POLYGON_API_KEY``). Notebooks and
Quarto backtests should keep using **S3 flat files** (``MASSIVE_S3_*``) via ``--data-source s3``
here only when you want parity with research.

**Start date:** If ``--start-date`` is omitted, REST mode defaults to **2017-01-01** so BTC hourly
requests skip empty years (global crypto aggregates on REST are typically sparse before ~2017).
S3 mode still defaults to **2004-01-01** to match the research notebooks. Override anytime.

Requires credentials in a repo-root ``.env`` (REST API key for default mode; S3 keys for ``--data-source s3``).

Usage:
  uv run python scripts/export_equal_weight_portfolio_weights.py
  uv run python scripts/export_equal_weight_portfolio_weights.py --latest-only --out outputs/ew_weights.csv
  uv run python scripts/export_equal_weight_portfolio_weights.py --data-source s3
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
from dotenv import load_dotenv

from research.signal_portfolio_blend import (
    SignalPortfolioParams,
    build_signal_portfolio_bundle,
    blend_signal_exposures,
    equal_blend_weights,
    equal_weight_portfolio_returns,
)

load_dotenv(dotenv_path=_REPO_ROOT / ".env")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "outputs/signal_portfolio/equal_weight_daily_weights.csv",
        help="Output CSV path (default: under repo outputs/)",
    )
    parser.add_argument(
        "--meta-out",
        type=Path,
        default=None,
        help="Optional JSON sidecar with parameters (default: <out>.meta.json)",
    )
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="Write only the last row (most recent session in the joined sample)",
    )
    parser.add_argument("--as-of", type=str, default=None, help="Only include rows through YYYY-MM-DD")
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="First calendar date for data (default: 2017-01-01 for --data-source rest, "
        "2004-01-01 for s3 — see script docstring)",
    )
    parser.add_argument("--eom-trigger-day", type=int, default=15)
    parser.add_argument("--relative-reversal-lookback", type=int, default=5)
    parser.add_argument("--turn-of-month-window", type=int, default=5)
    parser.add_argument("--beta-lookback-days", type=int, default=40)
    parser.add_argument("--zscore-lookback-days", type=int, default=20)
    parser.add_argument("--entry-zscore", type=float, default=1.5)
    parser.add_argument(
        "--data-source",
        choices=("rest", "s3"),
        default="rest",
        help="rest = Massive REST API (default for this script); s3 = flat files (backtest parity)",
    )
    args = parser.parse_args()

    effective_start = args.start_date or (
        "2017-01-01" if args.data_source == "rest" else "2004-01-01"
    )

    params = SignalPortfolioParams(
        start_date=effective_start,
        eom_trigger_day=args.eom_trigger_day,
        relative_reversal_lookback=args.relative_reversal_lookback,
        turn_of_month_window=args.turn_of_month_window,
        beta_lookback_days=args.beta_lookback_days,
        zscore_lookback_days=args.zscore_lookback_days,
        entry_zscore=args.entry_zscore,
    )
    bundle = build_signal_portfolio_bundle(params, data_source=args.data_source)
    signal_returns = bundle.signal_returns
    w = equal_blend_weights(signal_returns)
    net = blend_signal_exposures(bundle.per_signal_exposure, w)
    model_ret = equal_weight_portfolio_returns(signal_returns)

    table = pd.DataFrame(
        {
            "weight_SPY": net["SPY"],
            "weight_TLT": net["TLT"],
            "weight_UPRO": net["UPRO"],
            "model_equal_weight_return": model_ret.reindex(net.index),
        },
        index=net.index.rename("session_date"),
    )
    if args.as_of is not None:
        table = table.loc[: pd.Timestamp(args.as_of)]
    if args.latest_only:
        table = table.tail(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, date_format="%Y-%m-%d")
    meta_path = args.meta_out if args.meta_out is not None else args.out.with_suffix(".meta.json")
    meta = {
        "exported_at_utc": datetime.now(tz=UTC).isoformat(),
        "params": {
            "start_date": params.start_date,
            "start_date_cli": args.start_date,
            "eom_trigger_day": params.eom_trigger_day,
            "relative_reversal_lookback": params.relative_reversal_lookback,
            "turn_of_month_window": params.turn_of_month_window,
            "beta_lookback_days": params.beta_lookback_days,
            "zscore_lookback_days": params.zscore_lookback_days,
            "entry_zscore": params.entry_zscore,
        },
        "blend": "equal_weight_four_signal",
        "data_source": args.data_source,
        "equal_weight_per_signal": float(w.iloc[0]),
        "rows": int(len(table)),
        "session_date_min": str(table.index.min().date()) if len(table) else None,
        "session_date_max": str(table.index.max().date()) if len(table) else None,
        "note": (
            "weight_* are net exposures per $1 of blended capital (each signal gets 1/n weight). "
            "Map to shares with account_equity * weight / price. Convention matches the "
            "spy-tlt-signal-portfolio-vol-target notebook (close-to-close)."
        ),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} ({len(table)} rows)")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
