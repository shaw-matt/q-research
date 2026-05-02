"""Statistical helpers for research notebooks."""

from __future__ import annotations

import pandas as pd


def mean_daily_turnover_one_way(weights: pd.Series | pd.DataFrame) -> float:
    """Average one-way portfolio turnover from day-to-day weight changes.

    For a weight vector :math:`w_t`, one-way turnover on day :math:`t` is
    :math:`\\frac{1}{2} \\sum_i |w_{i,t} - w_{i,t-1}|`. For a single exposure
    series this reduces to :math:`\\frac{1}{2} |w_t - w_{t-1}|`.
    """
    if isinstance(weights, pd.DataFrame):
        deltas = weights.astype(float).diff().abs().sum(axis=1)
    else:
        deltas = weights.astype(float).diff().abs()
    deltas = deltas.dropna()
    if deltas.empty:
        return float("nan")
    return float(0.5 * deltas.mean())


def annualized_turnover_one_way(
    weights: pd.Series | pd.DataFrame,
    *,
    trading_days_per_year: float = 252.0,
) -> float:
    """Scale average daily one-way turnover to an annual figure."""
    daily = mean_daily_turnover_one_way(weights)
    if daily != daily:
        return float("nan")
    return float(daily * trading_days_per_year)


def spy_tlt_long_only_weights(position_asset: pd.Series) -> pd.DataFrame:
    """One-hot SPY vs TLT weights (NaN/others map to cash in both legs)."""
    w = pd.DataFrame(0.0, index=position_asset.index, columns=["SPY", "TLT"])
    mask_spy = position_asset == "SPY"
    mask_tlt = position_asset == "TLT"
    w.loc[mask_spy, "SPY"] = 1.0
    w.loc[mask_tlt, "TLT"] = 1.0
    return w


def summarize_event_returns(
    events: pd.DataFrame,
    return_columns: list[str],
) -> pd.DataFrame:
    """Summarize event-study returns for a set of return columns."""
    if events.empty:
        return pd.DataFrame(
            columns=["window", "count", "mean", "median", "std", "positive_rate"]
        )

    summary = []
    for column in return_columns:
        series = events[column].dropna()
        summary.append(
            {
                "window": column,
                "count": int(series.count()),
                "mean": series.mean(),
                "median": series.median(),
                "std": series.std(),
                "positive_rate": (series > 0).mean(),
            }
        )

    return pd.DataFrame(summary)
