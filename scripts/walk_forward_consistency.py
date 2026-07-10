#!/usr/bin/env python3
"""Walk-forward consistency check — fixed-size sliding windows.

Runs the portfolio bundle over N fixed-length windows that slide forward by
--step-months.  For every date that appears in more than one window the signal
positions and portfolio returns must agree; any disagreement is a look-ahead or
data-leakage bug.

Window layout (--window-months W, --step-months S, N windows):
  window 0: [T0,          T0 + W]
  window 1: [T0 + S,      T0 + S + W]
  window 2: [T0 + 2S,     T0 + 2S + W]
  ...
  window N-1: [--as-of - W, --as-of]

Overlapping dates between window i and window i+1 span [T0+(i+1)*S, T0+i*W].

Usage:
  uv run python scripts/walk_forward_consistency.py
  uv run python scripts/walk_forward_consistency.py --as-of 2025-06-01 --num-windows 6
  uv run python scripts/walk_forward_consistency.py --window-months 12 --step-months 3 --num-windows 8
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

from research.signal_portfolio_blend import (
    SignalPortfolioParams,
    blend_signal_exposures,
    build_signal_portfolio_bundle,
    equal_blend_weights,
)

load_dotenv(dotenv_path=_REPO_ROOT / ".env")

TRADING_DAYS_PER_YEAR = 252


def _sharpe(returns: pd.Series) -> float:
    r = returns.dropna()
    return float(r.mean() / r.std() * math.sqrt(TRADING_DAYS_PER_YEAR)) if r.std() > 0 else math.nan


def _run_window(start: str, end: str, residual_start: str, data_source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (signal_returns, net_exposure) for the window [start, end]."""
    params = SignalPortfolioParams(
        start_date=start,
        residual_start_date=residual_start,
    )
    bundle = build_signal_portfolio_bundle(params, data_source=data_source)
    sr = bundle.signal_returns.loc[:end]
    w = equal_blend_weights(sr)
    net = blend_signal_exposures(bundle.per_signal_exposure.loc[:end], w)
    portfolio_ret = sr.mul(w, axis=1).sum(axis=1).rename("portfolio")
    # Attach portfolio return as an extra column for easy comparison
    net = net.copy()
    net["portfolio_return"] = portfolio_ret
    return sr, net


def _generate_windows(as_of: str, num_windows: int, window_months: int, step_months: int) -> list[tuple[str, str]]:
    end_anchor = pd.Timestamp(as_of)
    windows = []
    for i in range(num_windows - 1, -1, -1):
        end = end_anchor - pd.DateOffset(months=i * step_months)
        start = end - pd.DateOffset(months=window_months)
        windows.append((start.date().isoformat(), end.date().isoformat()))
    return windows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", type=str, default=None, help="Last window end date (default: today)")
    parser.add_argument("--num-windows", type=int, default=6, help="Number of windows (default: 6)")
    parser.add_argument("--window-months", type=int, default=18, help="Length of each window in months (default: 18)")
    parser.add_argument("--step-months", type=int, default=3, help="Slide between windows in months (default: 3)")
    parser.add_argument("--residual-start-offset-months", type=int, default=None,
                        help="Months before window end to start the residual leg (default: same as window start)")
    parser.add_argument("--data-source", choices=("s3", "rest"), default="rest",
                        help="Data source: rest (default, uses MASSIVE_API_KEY) or s3 (flat files)")
    parser.add_argument("--out", type=Path,
                        default=_REPO_ROOT / "outputs/signal_portfolio/walk_forward_consistency.csv")
    args = parser.parse_args()

    as_of = args.as_of or datetime.now(tz=UTC).date().isoformat()
    windows = _generate_windows(as_of, args.num_windows, args.window_months, args.step_months)

    print(f"Windows ({args.window_months}-month, step {args.step_months}-month):")
    for i, (s, e) in enumerate(windows):
        print(f"  [{i}] {s} → {e}")

    # ── Run each window ──────────────────────────────────────────────────────
    results: list[tuple[str, str, pd.DataFrame, pd.DataFrame]] = []
    for start, end in windows:
        residual_start = start
        if args.residual_start_offset_months is not None:
            residual_start = (
                pd.Timestamp(end) - pd.DateOffset(months=args.residual_start_offset_months)
            ).date().isoformat()
        print(f"\nRunning {start} → {end} …", flush=True)
        sr, net = _run_window(start, end, residual_start, args.data_source)
        results.append((start, end, sr, net))
        port = net["portfolio_return"]
        print(f"  sessions={len(sr)}  sharpe={_sharpe(port):.3f}")

    # ── Find overlapping dates and compare ───────────────────────────────────
    print("\n── Overlap consistency check ──")
    all_discrepancies: list[dict] = []

    for i in range(len(results) - 1):
        for j in range(i + 1, len(results)):
            s_i, e_i, sr_i, net_i = results[i]
            s_j, e_j, sr_j, net_j = results[j]

            overlap_dates = net_i.index.intersection(net_j.index)
            if overlap_dates.empty:
                print(f"  windows [{i}] and [{j}]: no overlap")
                continue

            # Compare net exposures + portfolio_return on overlapping dates
            a = net_i.loc[overlap_dates]
            b = net_j.loc[overlap_dates]
            diff = (a - b).abs()
            max_diff = diff.max().max()
            n_mismatches = (diff > 1e-8).any(axis=1).sum()

            print(
                f"  windows [{i}] ∩ [{j}]: {len(overlap_dates)} dates  "
                f"max_diff={max_diff:.2e}  mismatches={n_mismatches}"
            )

            if n_mismatches > 0:
                bad_dates = overlap_dates[(diff > 1e-8).any(axis=1)]
                for col in diff.columns:
                    bad_col = diff.loc[bad_dates, col]
                    bad_col = bad_col[bad_col > 1e-8]
                    for date, val in bad_col.items():
                        all_discrepancies.append({
                            "window_i": f"{s_i}:{e_i}",
                            "window_j": f"{s_j}:{e_j}",
                            "date": str(date.date()),
                            "column": col,
                            "value_i": float(a.loc[date, col]),
                            "value_j": float(b.loc[date, col]),
                            "abs_diff": float(val),
                        })

    # ── Output ───────────────────────────────────────────────────────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if all_discrepancies:
        disc_df = pd.DataFrame(all_discrepancies)
        print(f"\n⚠  {len(disc_df)} discrepancies found:")
        print(disc_df.to_string(index=False))
        disc_path = args.out.with_name(args.out.stem + "_discrepancies.csv")
        disc_df.to_csv(disc_path, index=False)
        print(f"\nWrote discrepancies to {disc_path}")
    else:
        print("\n✓ All overlapping dates are consistent across windows.")

    # Save per-window summary
    summary_rows = []
    for start, end, sr, net in results:
        port = net["portfolio_return"]
        summary_rows.append({
            "window_start": start,
            "window_end": end,
            "sessions": len(sr),
            "sharpe": round(_sharpe(port), 3),
            **{f"last_{col}": round(float(net[col].iloc[-1]), 4) for col in net.columns},
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out, index=False)
    print(f"Wrote summary to {args.out}")


if __name__ == "__main__":
    main()
