#!/usr/bin/env python3
"""Export daily volatility-targeted futures-equivalent portfolio weights.

Same signal blend as export_vol_target_portfolio_weights.py, but maps proxy
exposures to futures-equivalent notional before applying the vol-target scalar:

  SPY  -> ES  (1x notional)
  UPRO -> ES  (3x notional)
  TLT  -> ZB  (1x notional)

Output columns:
  notional_ES, notional_ZB          – futures notional per $1 of strategy capital
  vol_target_scale                  – leverage scalar applied that day
  model_vol_target_futures_return   – daily return of the scaled futures portfolio

The last row is today's date with notionals/scale forward-filled from the most
recent close signal; model_vol_target_futures_return is blank for today.

Usage:
  uv run python scripts/export_vol_target_futures_weights.py
  uv run python scripts/export_vol_target_futures_weights.py --target-vol 0.40 --max-leverage 5.0
  uv run python scripts/export_vol_target_futures_weights.py --data-source s3
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

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from research.signal_portfolio_blend import (
    SignalPortfolioParams,
    build_signal_portfolio_bundle,
    blend_signal_exposures,
    equal_blend_weights,
)

load_dotenv(dotenv_path=_REPO_ROOT / ".env")

TRADING_DAYS_PER_YEAR = 252.0

# proxy_asset -> (futures_symbol, notional_multiplier)
FUTURES_MAP = {
    "SPY":  ("ES", 1.0),
    "UPRO": ("ES", 3.0),
    "TLT":  ("ZB", 1.0),
}
FUTURES_SYMBOLS = ["ES", "ZB"]


def map_to_futures_notional(proxy_exposure: pd.DataFrame) -> pd.DataFrame:
    """Convert proxy ETF weights to futures notional weights per $1 capital."""
    out = pd.DataFrame(0.0, index=proxy_exposure.index, columns=FUTURES_SYMBOLS)
    for proxy, (symbol, mult) in FUTURES_MAP.items():
        if proxy in proxy_exposure.columns:
            out[symbol] += proxy_exposure[proxy] * mult
    return out


def rolling_realized_volatility(return_series: pd.Series, lookback_days: int) -> pd.Series:
    return return_series.rolling(lookback_days).std() * math.sqrt(TRADING_DAYS_PER_YEAR)


def build_volatility_target_scale(
    return_series: pd.Series,
    *,
    target_volatility: float,
    lookback_days: int,
    lag_sessions: int,
    max_leverage: float,
) -> pd.Series:
    realized_vol = rolling_realized_volatility(return_series, lookback_days)
    raw_scale = target_volatility / realized_vol.replace(0.0, np.nan)
    scale = raw_scale.replace([np.inf, -np.inf], np.nan).clip(lower=0.0, upper=max_leverage)
    return scale.shift(lag_sessions).rename("vol_target_scale")


def finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def format_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def format_decimal(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def summarize_return_stats(return_series: pd.Series) -> dict:
    returns = pd.to_numeric(return_series, errors="coerce").dropna()
    if returns.empty:
        return {"observations": 0, "start_date": None, "end_date": None,
                "total_return": None, "annualized_return": None,
                "annualized_volatility": None, "sharpe_ratio": None}
    equity = (1.0 + returns).cumprod()
    ending_equity = float(equity.iloc[-1])
    daily_std = float(returns.std())
    ann_vol = daily_std * math.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = float(returns.mean()) / daily_std * math.sqrt(TRADING_DAYS_PER_YEAR) if daily_std > 0 else math.nan
    ann_ret = ending_equity ** (TRADING_DAYS_PER_YEAR / len(returns)) - 1.0 if ending_equity > 0 else math.nan
    return {
        "observations": int(len(returns)),
        "start_date": str(returns.index.min().date()),
        "end_date": str(returns.index.max().date()),
        "total_return": finite_or_none(ending_equity - 1.0),
        "annualized_return": finite_or_none(ann_ret),
        "annualized_volatility": finite_or_none(ann_vol),
        "sharpe_ratio": finite_or_none(sharpe),
    }


def build_futures_return_proxies(bundle, signal_index: pd.DatetimeIndex, data_source: str) -> pd.DataFrame:
    """SPY and TLT pct_change as ES/ZB return proxies."""
    if data_source == "rest":
        from research.massive_rest import download_rest_stock_day_closes
        prices = download_rest_stock_day_closes(["SPY", "TLT"], signal_index.min().date().isoformat(),
                                                signal_index.max().date().isoformat())
    else:
        from research.data import download_massive_daily_closes
        prices = download_massive_daily_closes(["SPY", "TLT"],
                                               start_date=signal_index.min().date().isoformat(),
                                               end_date=signal_index.max().date().isoformat())
    return pd.DataFrame({
        "ES": prices["SPY"].pct_change().reindex(signal_index),
        "ZB": prices["TLT"].pct_change().reindex(signal_index),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=_REPO_ROOT / "outputs/signal_portfolio/vol_target_futures_weights.csv")
    parser.add_argument("--meta-out", type=Path, default=None)
    parser.add_argument("--latest-only", action="store_true")
    parser.add_argument("--as-of", type=str, default=None)
    parser.add_argument("--trim-start", type=str, default=None,
                        help="Drop rows before YYYY-MM-DD (data still fetched from --start-date for warmup)")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--residual-start-date", type=str, default=None)
    parser.add_argument("--target-vol", type=float, default=0.10)
    parser.add_argument("--vol-lookback", type=int, default=63)
    parser.add_argument("--vol-lag", type=int, default=1)
    parser.add_argument("--max-leverage", type=float, default=2.0)
    parser.add_argument("--eom-trigger-day", type=int, default=15)
    parser.add_argument("--relative-reversal-lookback", type=int, default=5)
    parser.add_argument("--turn-of-month-window", type=int, default=5)
    parser.add_argument("--beta-lookback-days", type=int, default=40)
    parser.add_argument("--zscore-lookback-days", type=int, default=20)
    parser.add_argument("--entry-zscore", type=float, default=1.5)
    parser.add_argument("--data-source", choices=("rest", "s3"), default="rest")
    args = parser.parse_args()

    effective_start = args.start_date or ("2017-01-01" if args.data_source == "rest" else "2004-01-01")

    params = SignalPortfolioParams(
        start_date=effective_start,
        residual_start_date=args.residual_start_date,
        eom_trigger_day=args.eom_trigger_day,
        relative_reversal_lookback=args.relative_reversal_lookback,
        turn_of_month_window=args.turn_of_month_window,
        beta_lookback_days=args.beta_lookback_days,
        zscore_lookback_days=args.zscore_lookback_days,
        entry_zscore=args.entry_zscore,
    )
    bundle = build_signal_portfolio_bundle(params, data_source=args.data_source)
    signal_returns = bundle.signal_returns
    per_signal_exposure = bundle.per_signal_exposure

    w = equal_blend_weights(signal_returns)
    net_proxy = blend_signal_exposures(per_signal_exposure, w)
    current_net_proxy = blend_signal_exposures(bundle.current_signal_exposure, w)

    # Map each signal's proxy exposure to futures notional, recompute signal returns
    futures_proxies = build_futures_return_proxies(bundle, signal_returns.index, args.data_source)
    futures_signal_returns: dict[str, pd.Series] = {}
    for sig in signal_returns.columns:
        sig_proxy = per_signal_exposure[sig].reindex(futures_proxies.index)
        sig_futures = map_to_futures_notional(sig_proxy)
        futures_signal_returns[sig] = (sig_futures * futures_proxies).sum(axis=1).rename(sig)
    futures_signal_df = pd.DataFrame(futures_signal_returns).dropna()

    net_futures = map_to_futures_notional(net_proxy.reindex(futures_signal_df.index))
    current_net_futures = map_to_futures_notional(current_net_proxy.reindex(futures_signal_df.index))
    equal_futures_ret = futures_signal_df.mul(w, axis=1).sum(axis=1).rename("equal_weight_futures_return")

    vol_scale = build_volatility_target_scale(
        equal_futures_ret,
        target_volatility=args.target_vol,
        lookback_days=args.vol_lookback,
        lag_sessions=args.vol_lag,
        max_leverage=args.max_leverage,
    )

    net_scaled = net_futures.mul(vol_scale, axis=0)
    vol_target_ret = (equal_futures_ret * vol_scale).rename("model_vol_target_futures_return")

    table = net_scaled.add_prefix("notional_")
    table["vol_target_scale"] = vol_scale
    table["model_vol_target_futures_return"] = vol_target_ret.reindex(net_scaled.index)
    table.index = table.index.rename("session_date")

    if args.as_of is not None:
        table = table.loc[: pd.Timestamp(args.as_of)]
    if args.trim_start is not None:
        table = table.loc[pd.Timestamp(args.trim_start):]

    today = pd.Timestamp.now(tz=UTC).normalize().tz_localize(None)
    last_date = table.index[-1] if len(table) > 0 else None
    if last_date is not None:
        next_session = last_date + pd.offsets.BDay(1)
        target_date = today if today > last_date else next_session
        latest_scale = float(vol_scale.dropna().iloc[-1]) if vol_scale.dropna().size > 0 else float("nan")
        today_weights = (current_net_futures.iloc[[-1]] * latest_scale).add_prefix("notional_").copy()
        today_weights.index = pd.DatetimeIndex([target_date])
        today_weights["vol_target_scale"] = latest_scale
        today_weights["model_vol_target_futures_return"] = float("nan")
        table = pd.concat([table, today_weights])

    return_stats = summarize_return_stats(table["model_vol_target_futures_return"])
    valid_scale = vol_scale.dropna()
    max_leverage_hit_pct = finite_or_none(float((valid_scale >= args.max_leverage).mean())) if len(valid_scale) else None

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
            "eom_trigger_day": params.eom_trigger_day,
            "relative_reversal_lookback": params.relative_reversal_lookback,
            "turn_of_month_window": params.turn_of_month_window,
            "beta_lookback_days": params.beta_lookback_days,
            "zscore_lookback_days": params.zscore_lookback_days,
            "entry_zscore": params.entry_zscore,
        },
        "blend": "volatility_weighted_futures_signal_portfolio",
        "data_source": args.data_source,
        "futures_map": {k: {"symbol": v[0], "notional_multiplier": v[1]} for k, v in FUTURES_MAP.items()},
        "vol_target": args.target_vol,
        "vol_lookback_days": args.vol_lookback,
        "vol_lag_sessions": args.vol_lag,
        "max_leverage": args.max_leverage,
        "signal_count": int(len(w)),
        "signal_names": list(w.index),
        "max_leverage_hit_pct": max_leverage_hit_pct,
        "return_stats": return_stats,
        "rows": int(len(table)),
        "session_date_min": str(table.index.min().date()) if len(table) else None,
        "session_date_max": str(table.index.max().date()) if len(table) else None,
        "note": (
            "notional_* are futures notional weights per $1 of strategy capital after the "
            "vol-target scalar. SPY->ES(1x), UPRO->ES(3x), TLT->ZB(1x). "
            "vol_target_scale is the leverage multiplier (capped at max_leverage). "
            "The last row is today's date forward-filled from the previous close; "
            "model_vol_target_futures_return is blank for that row."
        ),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} ({len(table)} rows)")
    print(f"Wrote {meta_path}")
    print(
        f"Vol target: {args.target_vol:.0%}  |  Max leverage: {args.max_leverage:.1f}x  |  "
        f"Days at max leverage: {format_percent(max_leverage_hit_pct)}"
    )
    print(
        "Return stats: "
        f"total_return={format_percent(return_stats['total_return'])}, "
        f"annualized_return={format_percent(return_stats['annualized_return'])}, "
        f"annualized_volatility={format_percent(return_stats['annualized_volatility'])}, "
        f"sharpe_ratio={format_decimal(return_stats['sharpe_ratio'])}"
    )


if __name__ == "__main__":
    main()
