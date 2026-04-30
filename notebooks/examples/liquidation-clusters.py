# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Liquidation Cluster Study
#
# ## Research Question
#
# Can liquidation clusters predict short-term reversal or continuation in crypto
# perpetual futures markets?
#
# ## Hypothesis
#
# Large forced-liquidation events may create short-term price dislocations. If a
# liquidation cluster represents capitulation into a temporary liquidity vacuum,
# the next few bars should show reversal. If it represents information-driven
# deleveraging, the next few bars should show continuation.

# %%
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

repo_root = next(
    parent for parent in [Path.cwd(), *Path.cwd().parents] if (parent / "pyproject.toml").exists()
)
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from research.data import make_synthetic_liquidation_data
from research.plotting import apply_default_style
from research.stats import summarize_event_returns

apply_default_style()

# %% [markdown]
# ## Assumptions
#
# - The first version must render without API keys, so it uses synthetic data.
# - Each row represents a 15-minute bar for a liquid crypto perpetual futures
#   market.
# - Long liquidation clusters are treated as downside forced-selling events.
# - Short liquidation clusters are treated as upside forced-buying events.
# - Event returns are measured from the event close to future closes.
#
# ## Data Sources
#
# Production research should replace this synthetic dataset with exchange or
# vendor liquidation feeds such as Hyperliquid, Binance futures, Bybit, Coinglass,
# or another licensed data source. The same event-study structure can be reused
# after ingestion normalizes timestamps, mark prices, open interest, and
# liquidation notional.

# %% [markdown]
# ## Synthetic Data
#
# The helper below creates reproducible prices, baseline liquidation noise, and a
# handful of injected liquidation shocks. The synthetic process is intentionally
# simple: it is suitable for validating the publishing pipeline and analysis
# shape, not for drawing market conclusions.

# %%
df = make_synthetic_liquidation_data()
df.head()

# %%
df.tail()

# %% [markdown]
# ## Methodology
#
# 1. Compute total liquidation notional per bar.
# 2. Flag clusters where total liquidation notional is above a rolling quantile.
# 3. Classify cluster pressure as `long_liquidation` or `short_liquidation`.
# 4. Calculate future returns over short horizons after each cluster.
# 5. Compare event returns by pressure side and discuss reversal versus
#    continuation.

# %%
analysis = df.copy()
analysis["total_liquidations_usd"] = (
    analysis["long_liquidations_usd"] + analysis["short_liquidations_usd"]
)
analysis["liquidation_threshold_usd"] = (
    analysis["total_liquidations_usd"]
    .rolling(window=96, min_periods=48)
    .quantile(0.98)
)
analysis["is_cluster"] = (
    analysis["total_liquidations_usd"] >= analysis["liquidation_threshold_usd"]
)
analysis["pressure_side"] = np.where(
    analysis["long_liquidations_usd"] >= analysis["short_liquidations_usd"],
    "long_liquidation",
    "short_liquidation",
)

cluster_cols = [
    "timestamp",
    "price",
    "return",
    "long_liquidations_usd",
    "short_liquidations_usd",
    "total_liquidations_usd",
    "liquidation_threshold_usd",
    "pressure_side",
]
clusters = analysis.loc[analysis["is_cluster"], cluster_cols].reset_index(drop=True)
clusters.head(10)

# %% [markdown]
# ## Event Return Calculation
#
# For each event, future returns are measured over 1, 4, and 12 bars. With
# 15-minute data, these correspond to roughly 15 minutes, 1 hour, and 3 hours.
# A negative forward return after long liquidations is continuation; a positive
# forward return is reversal. For short liquidations, the interpretation is
# inverted.

# %%
horizons = {
    "forward_return_1_bar": 1,
    "forward_return_4_bars": 4,
    "forward_return_12_bars": 12,
}

for column, horizon in horizons.items():
    analysis[column] = analysis["price"].shift(-horizon) / analysis["price"] - 1

events = analysis.loc[analysis["is_cluster"], cluster_cols + list(horizons)].copy()
events["signed_reversal_4_bars"] = np.where(
    events["pressure_side"] == "long_liquidation",
    events["forward_return_4_bars"],
    -events["forward_return_4_bars"],
)
events.head(10)

# %%
summary = summarize_event_returns(events, list(horizons))
summary

# %%
by_side = (
    events.groupby("pressure_side")[list(horizons) + ["signed_reversal_4_bars"]]
    .agg(["count", "mean", "median"])
    .round(4)
)
by_side

# %% [markdown]
# ## Visualizations
#
# The first plot shows price and detected clusters. The second plot summarizes
# 4-bar event returns by liquidation pressure side.

# %%
fig, ax_price = plt.subplots()
ax_liq = ax_price.twinx()

ax_price.plot(analysis["timestamp"], analysis["price"], label="Synthetic price")
ax_liq.bar(
    analysis["timestamp"],
    analysis["total_liquidations_usd"],
    width=0.01,
    alpha=0.25,
    color="tab:orange",
    label="Total liquidations",
)

cluster_points = analysis.loc[analysis["is_cluster"]]
ax_price.scatter(
    cluster_points["timestamp"],
    cluster_points["price"],
    color="tab:red",
    label="Detected cluster",
    zorder=3,
)

ax_price.set_title("Synthetic Price with Liquidation Clusters")
ax_price.set_ylabel("Price")
ax_liq.set_ylabel("Liquidations, USD")
ax_price.legend(loc="upper left")
ax_liq.legend(loc="upper right")
fig.autofmt_xdate()
plt.show()

# %%
plot_data = events.dropna(subset=["forward_return_4_bars"]).copy()

fig, ax = plt.subplots()
plot_data.boxplot(column="forward_return_4_bars", by="pressure_side", ax=ax)
ax.axhline(0, color="black", linewidth=1)
ax.set_title("4-Bar Forward Returns After Liquidation Clusters")
ax.set_xlabel("Pressure side")
ax.set_ylabel("Forward return")
fig.suptitle("")
plt.show()

# %% [markdown]
# ## Analysis
#
# The synthetic sample produces a small set of clusters and event returns. The
# table below interprets the 4-bar signed reversal metric:
#
# - Positive values indicate reversal after forced liquidation pressure.
# - Negative values indicate continuation after forced liquidation pressure.
#
# Because the data-generating process injected some reversal after large shocks,
# a positive average signed reversal is expected here. Real market data may show
# regime dependence by volatility, funding, open interest, session, and the
# exchange where the liquidation occurred.

# %%
interpretation = pd.DataFrame(
    {
        "metric": [
            "cluster_count",
            "mean_signed_reversal_4_bars",
            "median_signed_reversal_4_bars",
            "positive_signed_reversal_rate",
        ],
        "value": [
            len(events),
            events["signed_reversal_4_bars"].mean(),
            events["signed_reversal_4_bars"].median(),
            (events["signed_reversal_4_bars"] > 0).mean(),
        ],
    }
)
interpretation

# %% [markdown]
# ## Confounders
#
# Liquidation clusters do not happen in isolation. Important confounders include:
#
# - Funding rates and crowded positioning before the event.
# - Market-wide volatility and correlation across BTC, ETH, and alt perpetuals.
# - Venue-specific liquidation engine behavior and reporting latency.
# - Spot market liquidity, ETF flows, and macro news around the event.
# - Open interest changes that separate position closure from new risk taking.
# - Survivorship bias if only highly liquid symbols are studied.
#
# ## Limitations
#
# - Synthetic data cannot validate a profitable or statistically reliable signal.
# - Cluster thresholds are heuristic and should be stress-tested.
# - Event windows overlap when clusters occur close together.
# - Fees, slippage, latency, and borrow/funding costs are ignored.
# - The study does not distinguish isolated liquidation prints from cascading
#   multi-bar events.
#
# ## Conclusion
#
# This notebook establishes a reproducible event-study template for liquidation
# clusters. In the synthetic example, clusters can be detected, forward returns
# can be measured, and reversal versus continuation can be summarized. The next
# research step is to replace the synthetic generator with real exchange data and
# evaluate whether the signal survives realistic execution assumptions.
#
# ## Next Research Ideas
#
# - Ingest Hyperliquid liquidation and trade data without requiring notebook-time
#   secrets.
# - Compare thresholds based on rolling quantiles, z-scores, and dollar notional
#   normalized by open interest.
# - Add parameterized runs for symbol, timeframe, date range, lookback window, and
#   event threshold.
# - Separate single-bar spikes from liquidation cascades.
# - Model conditional effects by volatility regime and funding-rate extremes.
# - Store normalized event tables as Parquet files for dashboards and scheduled
#   refreshes.
