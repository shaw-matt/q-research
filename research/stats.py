"""Statistical helpers for research notebooks."""

from __future__ import annotations

import pandas as pd


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
