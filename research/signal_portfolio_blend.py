"""SPY/TLT + UPRO residual blend (same rules as the portfolio notebook)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC
from typing import Literal

import numpy as np
import pandas as pd

from research.data import download_massive_daily_closes
from research.upro_residual import build_upro_residual_strategy_frame


@dataclass(frozen=True)
class SignalPortfolioParams:
    start_date: str = "2004-01-01"
    residual_start_date: str | None = None
    eom_trigger_day: int = 15
    relative_reversal_lookback: int = 5
    turn_of_month_window: int = 5
    beta_lookback_days: int = 40
    zscore_lookback_days: int = 20
    entry_zscore: float = 1.5



@dataclass(frozen=True)
class SignalPortfolioBundle:
    signal_returns: pd.DataFrame
    per_signal_exposure: pd.DataFrame
    upro_frame: pd.DataFrame


def build_eom_rebalance_returns(
    prices: pd.DataFrame, trigger_day: int = 15
) -> tuple[pd.Series, pd.DataFrame]:
    frame = prices.copy()
    frame["month"] = frame.index.to_period("M")
    frame["day_in_month"] = frame.groupby("month").cumcount() + 1
    frame["SPY_mtd"] = frame.groupby("month")["SPY"].transform(lambda s: s / s.iloc[0] - 1)
    frame["TLT_mtd"] = frame.groupby("month")["TLT"].transform(lambda s: s / s.iloc[0] - 1)

    trigger = frame["day_in_month"] == trigger_day
    frame["signal_asset"] = pd.Series(index=frame.index, dtype="object")
    frame.loc[trigger & (frame["SPY_mtd"] > frame["TLT_mtd"]), "signal_asset"] = "TLT"
    frame.loc[trigger & (frame["TLT_mtd"] > frame["SPY_mtd"]), "signal_asset"] = "SPY"
    frame["signal_asset"] = frame.groupby("month")["signal_asset"].transform("first")
    frame["is_last_day"] = frame["day_in_month"] == frame.groupby("month")["day_in_month"].transform("max")

    frame["position_asset"] = pd.Series(index=frame.index, dtype="object")
    active_window = (frame["day_in_month"] > trigger_day) & ~frame["is_last_day"]
    frame.loc[active_window, "position_asset"] = frame.loc[active_window, "signal_asset"]

    frame["SPY_return"] = frame["SPY"].pct_change()
    frame["TLT_return"] = frame["TLT"].pct_change()
    frame["strategy_return"] = np.where(
        frame["position_asset"] == "SPY",
        frame["SPY_return"],
        np.where(frame["position_asset"] == "TLT", frame["TLT_return"], 0.0),
    )
    exposure = pd.DataFrame(
        {
            "SPY": (frame["position_asset"] == "SPY").astype(float),
            "TLT": (frame["position_asset"] == "TLT").astype(float),
            "UPRO": 0.0,
        },
        index=frame.index,
    )
    return frame["strategy_return"].rename("eom_rebalance"), exposure


def build_relative_reversal_returns(
    prices: pd.DataFrame, lookback_days: int = 5
) -> tuple[pd.Series, pd.DataFrame]:
    frame = prices.copy()
    frame["SPY_return"] = frame["SPY"].pct_change()
    frame["TLT_return"] = frame["TLT"].pct_change()
    frame["log_ratio"] = np.log(frame["SPY"] / frame["TLT"])
    frame["log_ratio_ma"] = frame["log_ratio"].rolling(lookback_days).mean()
    frame["signal_asset"] = np.where(frame["log_ratio"] < frame["log_ratio_ma"], "SPY", "TLT")
    frame["position_asset"] = frame["signal_asset"].shift(1)
    frame["strategy_return"] = np.where(
        frame["position_asset"] == "SPY",
        frame["SPY_return"],
        np.where(frame["position_asset"] == "TLT", frame["TLT_return"], np.nan),
    )
    exposure = pd.DataFrame(
        {
            "SPY": (frame["position_asset"] == "SPY").astype(float),
            "TLT": (frame["position_asset"] == "TLT").astype(float),
            "UPRO": 0.0,
        },
        index=frame.index,
    )
    return frame["strategy_return"].rename("relative_reversal"), exposure


def build_turn_of_month_tlt_returns(
    prices: pd.DataFrame, window_days: int = 5
) -> tuple[pd.Series, pd.DataFrame]:
    frame = prices.copy()
    frame["month"] = frame.index.to_period("M")
    frame["day_in_month"] = frame.groupby("month").cumcount() + 1
    frame["days_in_month"] = frame.groupby("month")["day_in_month"].transform("max")
    frame["days_to_month_end"] = frame["days_in_month"] - frame["day_in_month"] + 1

    month_num = frame["month"].dt.month
    next_month_num = frame["month"].shift(-1).dt.month
    is_month_end = month_num != next_month_num

    frame["position_signal"] = 0
    frame.loc[frame["days_to_month_end"] <= window_days, "position_signal"] = 1
    frame.loc[frame["day_in_month"] <= window_days, "position_signal"] = -1
    frame.loc[is_month_end, "position_signal"] = -1

    frame["TLT_return"] = frame["TLT"].pct_change()
    frame["position"] = frame["position_signal"].shift(1).fillna(0)
    frame["strategy_return"] = frame["position"] * frame["TLT_return"]
    exposure = pd.DataFrame(
        {
            "SPY": 0.0,
            "TLT": frame["position"].astype(float),
            "UPRO": 0.0,
        },
        index=frame.index,
    )
    return frame["strategy_return"].rename("tlt_turn_of_month"), exposure


def blend_signal_exposures(
    per_signal: pd.DataFrame,
    blend_weights: pd.Series,
) -> pd.DataFrame:
    """Net asset weights per $1 of blended signal capital."""
    if not isinstance(per_signal.columns, pd.MultiIndex):
        raise ValueError("per_signal must have MultiIndex columns of (signal, asset).")
    assets = list(dict.fromkeys(per_signal.columns.get_level_values(1)))
    out = pd.DataFrame(0.0, index=per_signal.index, columns=assets)
    for sig in blend_weights.index:
        w = float(blend_weights[sig])
        if sig not in per_signal.columns.get_level_values(0):
            continue
        signal_exposure = per_signal[sig].reindex(columns=assets, fill_value=0.0).fillna(0.0)
        for a in assets:
            out[a] = out[a] + w * signal_exposure[a]
    return out


def gross_exposure_shares(net: pd.DataFrame) -> pd.DataFrame:
    """100% stacked shares of gross long/short exposure plus cash/flat."""
    if net.empty:
        return pd.DataFrame(index=net.index)
    cleaned = net.fillna(0.0)
    gross = cleaned.abs().sum(axis=1)
    denom = gross.replace(0.0, np.nan)
    parts = pd.DataFrame(index=cleaned.index)
    for asset in cleaned.columns:
        long_exposure = cleaned[asset].clip(lower=0.0)
        short_exposure = (-cleaned[asset]).clip(lower=0.0)
        if (short_exposure > 0.0).any():
            parts[f"{asset} long"] = (long_exposure / denom).fillna(0.0)
            parts[f"{asset} short"] = (short_exposure / denom).fillna(0.0)
        else:
            parts[str(asset)] = (long_exposure / denom).fillna(0.0)
    cash = (gross <= 0.0).astype(float)
    parts = parts.mul((gross > 0.0).astype(float), axis=0)
    parts["Cash / flat"] = cash
    return parts


def equal_blend_weights(signal_returns: pd.DataFrame) -> pd.Series:
    n = signal_returns.shape[1]
    return pd.Series(np.repeat(1.0 / n, n), index=signal_returns.columns, name="weight")


def build_signal_portfolio_bundle(
    params: SignalPortfolioParams | None = None,
    *,
    data_source: Literal["s3", "rest"] = "s3",
) -> SignalPortfolioBundle:
    p = params or SignalPortfolioParams()
    end = pd.Timestamp.now(tz=UTC).date().isoformat()
    residual_start = p.residual_start_date or p.start_date
    if data_source == "rest":
        from research.massive_rest import download_rest_stock_day_closes

        prices = download_rest_stock_day_closes(["SPY", "TLT"], p.start_date, end).dropna()
    else:
        prices = download_massive_daily_closes(
            ["SPY", "TLT"], start_date=p.start_date, end_date=end
        ).dropna()
    if prices.empty:
        raise ValueError("No SPY/TLT daily prices were downloaded.")

    eom_ret, eom_exp = build_eom_rebalance_returns(prices, trigger_day=p.eom_trigger_day)
    rel_ret, rel_exp = build_relative_reversal_returns(
        prices, lookback_days=p.relative_reversal_lookback
    )
    tom_ret, tom_exp = build_turn_of_month_tlt_returns(prices, window_days=p.turn_of_month_window)
    spy_tlt_returns = pd.concat([eom_ret, rel_ret, tom_ret], axis=1)

    upro_frame = build_upro_residual_strategy_frame(
        start_date=residual_start,
        end_date=end,
        beta_lookback=p.beta_lookback_days,
        zscore_lookback=p.zscore_lookback_days,
        entry_zscore=p.entry_zscore,
        data_source=data_source,
    )
    upro_returns = upro_frame["strategy_return"].rename("upro_residual")

    signal_returns = pd.concat(
        [spy_tlt_returns, upro_returns],
        axis=1,
        join="inner",
    ).dropna()
    if signal_returns.empty:
        raise ValueError(
            "No overlapping history after joining SPY/TLT and UPRO residual signals."
        )

    cols = list(signal_returns.columns)
    per_signal_exposure = pd.concat(
        [
            eom_exp.reindex(signal_returns.index).fillna(0.0),
            rel_exp.reindex(signal_returns.index).fillna(0.0),
            tom_exp.reindex(signal_returns.index).fillna(0.0),
            pd.DataFrame(
                {
                    "SPY": 0.0,
                    "TLT": 0.0,
                    "UPRO": upro_frame["upro_exposure"].reindex(signal_returns.index).fillna(0.0),
                },
                index=signal_returns.index,
            ),
        ],
        axis=1,
        keys=cols,
    )

    return SignalPortfolioBundle(
        signal_returns=signal_returns,
        per_signal_exposure=per_signal_exposure,
        upro_frame=upro_frame,
    )


def equal_weight_portfolio_returns(signal_returns: pd.DataFrame) -> pd.Series:
    w = equal_blend_weights(signal_returns)
    return signal_returns.mul(w, axis=1).sum(axis=1).rename("equal_weight_return")
