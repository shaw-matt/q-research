#!/usr/bin/env python3
"""Export daily equal-weight signal portfolio weights (portfolio notebook rules).

Writes a CSV suitable for dashboards or manual trading: net SPY/TLT/UPRO/VX30-proxy
weights per $1 of strategy capital and the model equal-weight portfolio return by
session date.

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
import math
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

TRADING_DAYS_PER_YEAR = 252.0


def finite_or_none(value: float) -> float | None:
    """Return JSON-friendly numbers for metrics that may be undefined."""
    return value if math.isfinite(value) else None


def format_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def format_decimal(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def summarize_return_stats(
    return_series: pd.Series,
    *,
    trading_days_per_year: float = TRADING_DAYS_PER_YEAR,
) -> dict[str, float | int | str | None]:
    """Compute aggregate return statistics for a daily return series."""
    returns = pd.to_numeric(return_series, errors="coerce").dropna()
    if returns.empty:
        return {
            "observations": 0,
            "start_date": None,
            "end_date": None,
            "total_return": None,
            "annualized_return": None,
            "annualized_volatility": None,
            "sharpe_ratio": None,
        }

    equity = (1.0 + returns).cumprod()
    ending_equity = float(equity.iloc[-1])
    daily_std = float(returns.std())
    annualized_volatility = daily_std * math.sqrt(trading_days_per_year)
    sharpe_ratio = (
        float(returns.mean()) / daily_std * math.sqrt(trading_days_per_year)
        if daily_std > 0.0
        else math.nan
    )
    annualized_return = (
        ending_equity ** (trading_days_per_year / len(returns)) - 1.0
        if ending_equity > 0.0
        else math.nan
    )

    return {
        "observations": int(len(returns)),
        "start_date": str(returns.index.min().date()),
        "end_date": str(returns.index.max().date()),
        "total_return": finite_or_none(ending_equity - 1.0),
        "annualized_return": finite_or_none(annualized_return),
        "annualized_volatility": finite_or_none(annualized_volatility),
        "sharpe_ratio": finite_or_none(sharpe_ratio),
    }


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
    parser.add_argument(
        "--residual-start-date",
        type=str,
        default=None,
        help="Optional first date for the BTC/QQQ residual leg. Defaults to --start-date.",
    )
    parser.add_argument("--eom-trigger-day", type=int, default=15)
    parser.add_argument("--relative-reversal-lookback", type=int, default=5)
    parser.add_argument("--turn-of-month-window", type=int, default=5)
    parser.add_argument("--beta-lookback-days", type=int, default=40)
    parser.add_argument("--zscore-lookback-days", type=int, default=20)
    parser.add_argument("--entry-zscore", type=float, default=1.5)
    parser.add_argument("--dirty-vix-start-date", type=str, default=None)
    parser.add_argument("--dirty-vix-zscore-days", type=int, default=252)
    parser.add_argument("--dirty-vix-min-zscore-obs", type=int, default=None)
    parser.add_argument("--dirty-vix-entry-zscore", type=float, default=-1.5)
    parser.add_argument("--dirty-vix-execution-lag-sessions", type=int, default=1)
    parser.add_argument("--dirty-vix-yahoo-ticker", type=str, default="VX=F")
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
        residual_start_date=args.residual_start_date,
        eom_trigger_day=args.eom_trigger_day,
        relative_reversal_lookback=args.relative_reversal_lookback,
        turn_of_month_window=args.turn_of_month_window,
        beta_lookback_days=args.beta_lookback_days,
        zscore_lookback_days=args.zscore_lookback_days,
        entry_zscore=args.entry_zscore,
        dirty_vix_start_date=args.dirty_vix_start_date,
        dirty_vix_rolling_zscore_days=args.dirty_vix_zscore_days,
        dirty_vix_min_zscore_obs=args.dirty_vix_min_zscore_obs,
        dirty_vix_entry_zscore=args.dirty_vix_entry_zscore,
        dirty_vix_execution_lag_sessions=args.dirty_vix_execution_lag_sessions,
        dirty_vix_yahoo_ticker=args.dirty_vix_yahoo_ticker,
    )
    bundle = build_signal_portfolio_bundle(params, data_source=args.data_source)
    signal_returns = bundle.signal_returns
    w = equal_blend_weights(signal_returns)
    net = blend_signal_exposures(bundle.per_signal_exposure, w)
    model_ret = equal_weight_portfolio_returns(signal_returns)

    table = net.add_prefix("weight_")
    table["model_equal_weight_return"] = model_ret.reindex(net.index)
    table.index = table.index.rename("session_date")
    if args.as_of is not None:
        table = table.loc[: pd.Timestamp(args.as_of)]
    return_stats = summarize_return_stats(table["model_equal_weight_return"])
    if args.latest_only:
        table = table.tail(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, date_format="%Y-%m-%d")
    meta_path = args.meta_out if args.meta_out is not None else args.out.with_suffix(".meta.json")
    meta = {
        "exported_at_utc": datetime.now(tz=UTC).isoformat(),
        "params": {
            "start_date": params.start_date,
            "residual_start_date": params.residual_start_date,
            "start_date_cli": args.start_date,
            "eom_trigger_day": params.eom_trigger_day,
            "relative_reversal_lookback": params.relative_reversal_lookback,
            "turn_of_month_window": params.turn_of_month_window,
            "beta_lookback_days": params.beta_lookback_days,
            "zscore_lookback_days": params.zscore_lookback_days,
            "entry_zscore": params.entry_zscore,
            "dirty_vix_start_date": params.dirty_vix_start_date,
            "dirty_vix_rolling_zscore_days": params.dirty_vix_rolling_zscore_days,
            "dirty_vix_min_zscore_obs": params.dirty_vix_min_zscore_obs,
            "dirty_vix_entry_zscore": params.dirty_vix_entry_zscore,
            "dirty_vix_execution_lag_sessions": params.dirty_vix_execution_lag_sessions,
            "dirty_vix_yahoo_ticker": params.dirty_vix_yahoo_ticker,
        },
        "blend": "equal_weight_signal_portfolio",
        "data_source": args.data_source,
        "signal_count": int(len(w)),
        "signal_names": list(w.index),
        "equal_weight_per_signal": float(w.iloc[0]),
        "return_stats": return_stats,
        "rows": int(len(table)),
        "session_date_min": str(table.index.min().date()) if len(table) else None,
        "session_date_max": str(table.index.max().date()) if len(table) else None,
        "note": (
            "weight_* are net exposures per $1 of blended capital (each signal gets 1/n weight). "
            "Map ETF/proxy exposures to shares or futures-equivalent notional with "
            "account_equity * weight / price. Convention matches the "
            "spy-tlt-signal-portfolio-vol-target notebook (close-to-close). VX30 is the "
            "dirty public-data VIX futures/volatility proxy, not an ETF share class."
        ),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} ({len(table)} rows)")
    print(f"Wrote {meta_path}")
    print(
        "Return stats: "
        f"total_return={format_percent(return_stats['total_return'])}, "
        f"annualized_return={format_percent(return_stats['annualized_return'])}, "
        f"annualized_volatility={format_percent(return_stats['annualized_volatility'])}, "
        f"sharpe_ratio={format_decimal(return_stats['sharpe_ratio'])}"
    )


if __name__ == "__main__":
    main()
