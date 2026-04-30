"""Data helpers for research notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_liquidation_data(
    periods: int = 14 * 24 * 4,
    frequency: str = "15min",
    seed: int = 42,
) -> pd.DataFrame:
    """Create reproducible synthetic price and liquidation observations."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2025-01-01", periods=periods, freq=frequency)

    baseline_returns = rng.normal(loc=0.00015, scale=0.004, size=periods)
    price = 100_000 * np.exp(np.cumsum(baseline_returns))

    long_liquidations = rng.gamma(shape=1.8, scale=45_000, size=periods)
    short_liquidations = rng.gamma(shape=1.8, scale=42_000, size=periods)

    event_locations = rng.choice(np.arange(24, periods - 24), size=10, replace=False)
    for location in event_locations:
        direction = rng.choice([-1, 1])
        shock = direction * rng.uniform(0.012, 0.03)
        baseline_returns[location] += shock

        if direction < 0:
            long_liquidations[location : location + 3] += rng.uniform(900_000, 1_800_000)
            baseline_returns[location + 1 : location + 5] += rng.normal(0.0025, 0.003, size=4)
        else:
            short_liquidations[location : location + 3] += rng.uniform(800_000, 1_600_000)
            baseline_returns[location + 1 : location + 5] += rng.normal(-0.0022, 0.003, size=4)

    price = 100_000 * np.exp(np.cumsum(baseline_returns))
    return pd.DataFrame(
        {
            "timestamp": index,
            "price": price,
            "return": baseline_returns,
            "long_liquidations_usd": long_liquidations,
            "short_liquidations_usd": short_liquidations,
        }
    )
