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
# # Dirty VIX Futures Cheapness Signal vs VIX3M
#
# ## Research Question
#
# Does a very simple measure of VIX futures cheapness,
# `zscore(log(VX30 / VIX3M), 252 trading days)`, predict next-day returns for a
# 30-day constant-maturity VIX futures proxy?
#
# ## Hypothesis
#
# When 30-day VIX futures are unusually cheap versus the VIX3M index, VIX futures
# may be more likely to mean-revert upward. A long-only VX30 rule that enters
# below a negative z-score threshold should therefore have positive convexity-like
# bursts and better average returns than always holding VX30.

# %%
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from IPython.display import Markdown, display

if "__file__" in globals():
    _REPO_ROOT = Path(__file__).resolve().parents[2]
else:
    _cwd = Path.cwd().resolve()
    _REPO_ROOT = _cwd if (_cwd / "research").is_dir() else _cwd.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from research.plotting import apply_default_style

load_dotenv(dotenv_path=_REPO_ROOT / ".env")
apply_default_style()

# %% [markdown]
# ## Assumptions
#
# - `VIX3M` is treated as the market price of SPX variance over roughly the next
#   90 calendar days.
# - `VX30` is approximated from listed VIX futures by linearly interpolating
#   daily futures settlement/close prices to a 30-calendar-day constant maturity.
# - If bracketing contracts are unavailable for a date, the nearest active VX
#   contract within the maximum DTE filter is used as a deliberately dirty
#   fallback.
# - Signals are computed from daily end-of-session marks. The lagged simulation
#   waits one full trading session before acting on the signal.
# - Returns are log returns on the VX30 price proxy. Transaction costs, slippage,
#   margin, collateral yield, taxes, exchange holidays, and futures roll execution
#   costs are ignored.
#
# ## Data Sources
#
# - Massive REST indices/stocks aggregate endpoint:
#   `GET /v2/aggs/ticker/I:VIX3M/range/1/day/{from}/{to}`.
# - Massive REST futures contracts endpoint:
#   `GET /futures/v1/contracts?product_code=VX`.
# - Massive REST futures aggregate endpoint:
#   `GET /futures/v1/aggs/{ticker}?resolution=1session`.
#
# Set `MASSIVE_API_KEY` or `POLYGON_API_KEY` in the environment or `.env` file.

# %%
START_DATE = os.getenv("VIX_CHEAPNESS_START_DATE", "2020-01-01")
END_DATE = os.getenv("VIX_CHEAPNESS_END_DATE", pd.Timestamp.today(tz="UTC").date().isoformat())

MASSIVE_REST_BASE = os.getenv("MASSIVE_REST_URL", "https://api.massive.com").rstrip("/")
VIX3M_TICKER = os.getenv("VIX3M_TICKER", "I:VIX3M")
VX_PRODUCT_CODE = os.getenv("VX_PRODUCT_CODE", "VX")

TARGET_DTE_DAYS = int(os.getenv("VX_TARGET_DTE_DAYS", "30"))
MAX_CONTRACT_DTE_DAYS = int(os.getenv("VX_MAX_CONTRACT_DTE_DAYS", "120"))
ROLLING_ZSCORE_DAYS = int(os.getenv("VIX_CHEAPNESS_ZSCORE_DAYS", "252"))
MIN_ZSCORE_OBS = int(os.getenv("VIX_CHEAPNESS_MIN_ZSCORE_OBS", str(ROLLING_ZSCORE_DAYS)))
ENTRY_ZSCORE = float(os.getenv("VIX_CHEAPNESS_ENTRY_ZSCORE", "-1.5"))
TRADING_DAYS_PER_YEAR = 252

FORWARD_RETURN_HORIZONS = [1, 5, 10, 21]
THRESHOLD_GRID = np.round(np.arange(-2.50, -0.45, 0.25), 2)

REST_CACHE_DIR = Path(
    os.getenv("Q_RESEARCH_MASSIVE_REST_CACHE_DIR", ".cache/q-research/massive-rest-vix")
)


def massive_api_key() -> str:
    key = (
        os.getenv("MASSIVE_API_KEY")
        or os.getenv("POLYGON_API_KEY")
        or os.getenv("POLYGON_API_KEY_ID")
        or ""
    )
    if not key:
        raise ValueError(
            "Massive REST requires an API key. Set MASSIVE_API_KEY or POLYGON_API_KEY."
        )
    return key


def request_json(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    *,
    api_key: str,
) -> dict[str, Any]:
    response = session.get(url, params=params, timeout=120)
    if response.status_code == 401:
        raise ValueError(
            "Massive REST returned 401 Unauthorized. Confirm the API key and that "
            "the subscription includes indices and futures data."
        )
    response.raise_for_status()
    payload = response.json()
    status = payload.get("status")
    if status not in ("OK", "DELAYED"):
        raise ValueError(f"Massive REST unexpected status {status!r}: {payload}")
    return payload


def fetch_paginated(
    session: requests.Session,
    initial_url: str,
    params: dict[str, Any],
    *,
    api_key: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    next_url: str | None = initial_url
    first = True
    while next_url:
        page_params = params if first else {}
        if not first and "apikey=" not in next_url.lower():
            page_params = {"apiKey": api_key}
        payload = request_json(session, next_url, page_params, api_key=api_key)
        rows.extend(payload.get("results") or [])
        next_url = payload.get("next_url")
        first = False
        if next_url:
            time.sleep(0.05)
    return rows


def cache_path(*parts: str) -> Path:
    safe_parts = [part.replace("/", "_").replace(":", "_") for part in parts]
    return REST_CACHE_DIR.joinpath(*safe_parts)


def read_cached_frame(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def write_cached_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)


def download_massive_daily_closes(
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.Series:
    path = cache_path("daily-closes", f"{ticker}_{start_date}_{end_date}.parquet")
    cached = read_cached_frame(path)
    if cached is not None:
        return cached["close"].rename(ticker)

    api_key = massive_api_key()
    session = requests.Session()
    enc = quote(ticker, safe="")
    url = f"{MASSIVE_REST_BASE}/v2/aggs/ticker/{enc}/range/1/day/{start_date}/{end_date}"
    params: dict[str, Any] = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key,
    }
    rows = fetch_paginated(session, url, params, api_key=api_key)
    if not rows:
        raise ValueError(f"No Massive aggregate rows returned for {ticker}.")

    records = []
    for row in rows:
        date = pd.Timestamp(int(row["t"]), unit="ms", tz="UTC")
        session_date = date.tz_convert("America/New_York").normalize().tz_localize(None)
        records.append({"date": session_date, "close": float(row["c"])})
    frame = pd.DataFrame.from_records(records).drop_duplicates("date", keep="last")
    frame = frame.sort_values("date").set_index("date")
    frame.index = pd.to_datetime(frame.index).normalize()
    write_cached_frame(path, frame)
    return frame["close"].rename(ticker)


@dataclass(frozen=True)
class FuturesContract:
    ticker: str
    first_trade_date: pd.Timestamp
    last_trade_date: pd.Timestamp
    settlement_date: pd.Timestamp


def download_vx_contracts(start_date: str, end_date: str) -> pd.DataFrame:
    path = cache_path("contracts", f"{VX_PRODUCT_CODE}_{start_date}_{end_date}.parquet")
    cached = read_cached_frame(path)
    if cached is not None:
        return cached

    api_key = massive_api_key()
    session = requests.Session()
    url = f"{MASSIVE_REST_BASE}/futures/v1/contracts"
    params: dict[str, Any] = {
        "product_code": VX_PRODUCT_CODE,
        "first_trade_date.lte": end_date,
        "last_trade_date.gte": start_date,
        "limit": 1000,
        "sort": "last_trade_date.asc",
        "apiKey": api_key,
    }
    rows = fetch_paginated(session, url, params, api_key=api_key)
    if not rows:
        raise ValueError(
            f"No Massive futures contracts found for product_code={VX_PRODUCT_CODE!r}."
        )

    frame = pd.DataFrame(rows)
    if "type" in frame.columns:
        frame = frame.loc[frame["type"].isna() | (frame["type"].astype(str).str.lower() != "combo")]
    keep = ["ticker", "first_trade_date", "last_trade_date", "settlement_date", "name", "type"]
    frame = frame[[column for column in keep if column in frame.columns]].copy()
    for column in ["first_trade_date", "last_trade_date", "settlement_date"]:
        frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["ticker", "first_trade_date", "last_trade_date", "settlement_date"])
    if frame.empty:
        raise ValueError(
            f"Massive returned contracts for product_code={VX_PRODUCT_CODE!r}, but none "
            "survived the single-contract/date filters."
        )
    frame = frame.drop_duplicates("ticker").sort_values(["settlement_date", "ticker"])
    write_cached_frame(path, frame)
    return frame


def download_futures_session_aggs(
    contract: FuturesContract,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    begin = max(pd.Timestamp(start_date), contract.first_trade_date).date().isoformat()
    end = min(pd.Timestamp(end_date), contract.last_trade_date).date().isoformat()
    if begin > end:
        return pd.DataFrame(columns=["date", "ticker", "price", "close", "settlement_price", "volume"])

    path = cache_path("futures-aggs", f"{contract.ticker}_{begin}_{end}.parquet")
    cached = read_cached_frame(path)
    if cached is not None:
        return cached

    api_key = massive_api_key()
    session = requests.Session()
    enc = quote(contract.ticker, safe="")
    url = f"{MASSIVE_REST_BASE}/futures/v1/aggs/{enc}"
    params: dict[str, Any] = {
        "resolution": "1session",
        "window_start.gte": begin,
        "window_start.lte": end,
        "limit": 50000,
        "sort": "window_start.asc",
        "apiKey": api_key,
    }
    rows = fetch_paginated(session, url, params, api_key=api_key)
    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "price", "close", "settlement_price", "volume"])

    records = []
    for row in rows:
        close = row.get("close")
        settlement = row.get("settlement_price")
        price = settlement if settlement is not None else close
        if price is None:
            continue
        session_end = row.get("session_end_date")
        date = (
            pd.Timestamp(session_end).normalize()
            if session_end
            else pd.Timestamp(int(row["window_start"]), unit="ns", tz="UTC")
            .tz_convert("America/Chicago")
            .normalize()
            .tz_localize(None)
        )
        records.append(
            {
                "date": date,
                "ticker": contract.ticker,
                "price": float(price),
                "close": float(close) if close is not None else np.nan,
                "settlement_price": float(settlement) if settlement is not None else np.nan,
                "volume": float(row.get("volume", np.nan)),
            }
        )

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.drop_duplicates(["date", "ticker"], keep="last").sort_values(["date", "ticker"])
    write_cached_frame(path, frame)
    return frame


def load_vx_futures_panel(start_date: str, end_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    contracts = download_vx_contracts(start_date, end_date)
    contract_objects = [
        FuturesContract(
            ticker=str(row.ticker),
            first_trade_date=row.first_trade_date,
            last_trade_date=row.last_trade_date,
            settlement_date=row.settlement_date,
        )
        for row in contracts.itertuples(index=False)
    ]

    pieces = [
        download_futures_session_aggs(contract, start_date, end_date)
        for contract in contract_objects
    ]
    nonempty_pieces = [piece for piece in pieces if not piece.empty]
    if not nonempty_pieces:
        raise ValueError("No Massive futures aggregate rows were returned for VX contracts.")
    aggs = pd.concat(nonempty_pieces, ignore_index=True)
    aggs = aggs.merge(
        contracts[["ticker", "settlement_date"]],
        on="ticker",
        how="left",
        validate="many_to_one",
    )
    aggs["dte"] = (aggs["settlement_date"] - aggs["date"]).dt.days
    aggs = aggs.loc[
        (aggs["dte"] > 0)
        & (aggs["dte"] <= MAX_CONTRACT_DTE_DAYS)
        & np.isfinite(aggs["price"])
        & (aggs["price"] > 0)
    ].copy()
    return contracts, aggs.sort_values(["date", "dte", "ticker"])


def interpolate_vx30(day_rows: pd.DataFrame) -> pd.Series:
    rows = day_rows.sort_values("dte")
    target = TARGET_DTE_DAYS

    exact = rows.loc[rows["dte"] == target]
    if not exact.empty:
        row = exact.iloc[0]
        return pd.Series(
            {
                "VX30": row["price"],
                "front_ticker": row["ticker"],
                "back_ticker": row["ticker"],
                "front_dte": row["dte"],
                "back_dte": row["dte"],
                "interp_weight_back": 0.0,
                "construction": "exact",
            }
        )

    front = rows.loc[rows["dte"] < target].tail(1)
    back = rows.loc[rows["dte"] > target].head(1)
    if not front.empty and not back.empty:
        front_row = front.iloc[0]
        back_row = back.iloc[0]
        weight_back = (target - front_row["dte"]) / (back_row["dte"] - front_row["dte"])
        vx30 = (1.0 - weight_back) * front_row["price"] + weight_back * back_row["price"]
        return pd.Series(
            {
                "VX30": vx30,
                "front_ticker": front_row["ticker"],
                "back_ticker": back_row["ticker"],
                "front_dte": front_row["dte"],
                "back_dte": back_row["dte"],
                "interp_weight_back": weight_back,
                "construction": "interpolated",
            }
        )

    nearest_row = rows.iloc[(rows["dte"] - target).abs().to_numpy().argmin()]
    return pd.Series(
        {
            "VX30": nearest_row["price"],
            "front_ticker": nearest_row["ticker"],
            "back_ticker": nearest_row["ticker"],
            "front_dte": nearest_row["dte"],
            "back_dte": nearest_row["dte"],
            "interp_weight_back": 0.0,
            "construction": "nearest",
        }
    )


def build_vx30_series(futures_aggs: pd.DataFrame) -> pd.DataFrame:
    vx30 = futures_aggs.groupby("date", group_keys=False).apply(interpolate_vx30)
    vx30.index = pd.to_datetime(vx30.index).normalize()
    vx30["VX30"] = pd.to_numeric(vx30["VX30"], errors="coerce")
    for column in ["front_dte", "back_dte", "interp_weight_back"]:
        vx30[column] = pd.to_numeric(vx30[column], errors="coerce")
    return vx30.sort_index()


def max_drawdown(log_returns: pd.Series) -> float:
    equity = np.exp(log_returns.fillna(0.0).cumsum())
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def summarize_log_returns(log_returns: pd.Series) -> pd.Series:
    r = log_returns.dropna()
    if r.empty:
        return pd.Series(dtype=float)
    ann_return = float(np.expm1(r.mean() * TRADING_DAYS_PER_YEAR))
    ann_vol = float(r.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    return pd.Series(
        {
            "observations": int(r.count()),
            "total_return": float(np.expm1(r.sum())),
            "ann_return": ann_return,
            "ann_vol": ann_vol,
            "sharpe": ann_return / ann_vol if ann_vol else np.nan,
            "max_drawdown": max_drawdown(r),
            "positive_days": float((r > 0).mean()),
        }
    )


def add_forward_log_returns(frame: pd.DataFrame, price_col: str, horizons: list[int]) -> pd.DataFrame:
    out = frame.copy()
    for horizon in horizons:
        out[f"vx30_fwd_{horizon}d_log_return"] = np.log(out[price_col].shift(-horizon) / out[price_col])
    return out


def assign_deciles(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    deciles = pd.Series(index=series.index, dtype="float64")
    if valid.empty:
        return deciles
    labels = range(1, 11)
    ranked = valid.rank(method="first")
    deciles.loc[valid.index] = pd.qcut(ranked, q=10, labels=labels).astype(float)
    return deciles


def strategy_returns(
    zscore: pd.Series,
    next_log_return: pd.Series,
    *,
    threshold: float,
    execution_lag_sessions: int,
) -> pd.DataFrame:
    raw_signal = zscore < threshold
    position = raw_signal.astype(float).shift(execution_lag_sessions).fillna(0.0)
    out = pd.DataFrame(
        {
            "zscore": zscore,
            "raw_signal": raw_signal.astype(float),
            "position": position,
            "vx30_next_log_return": next_log_return,
        }
    )
    out["strategy_log_return"] = out["position"] * out["vx30_next_log_return"]
    out["trade"] = out["position"].diff().abs().fillna(out["position"])
    return out


# %% [markdown]
# ## Load Massive Data
#
# The futures section intentionally discovers contracts from Massive rather than
# assuming a continuous-contract symbol. This keeps the notebook explicit about
# how `VX30` is built and makes the interpolation auditable.

# %%
vix3m = download_massive_daily_closes(VIX3M_TICKER, START_DATE, END_DATE)
contracts, futures_aggs = load_vx_futures_panel(START_DATE, END_DATE)
vx30 = build_vx30_series(futures_aggs)

data = pd.concat([vx30["VX30"], vix3m.rename("VIX3M")], axis=1).dropna()
data = data.loc[(data["VX30"] > 0) & (data["VIX3M"] > 0)].copy()
data = add_forward_log_returns(data, "VX30", FORWARD_RETURN_HORIZONS)
data["vx30_log_return"] = np.log(data["VX30"] / data["VX30"].shift(1))
data["cheapness_log_ratio"] = np.log(data["VX30"] / data["VIX3M"])
rolling = data["cheapness_log_ratio"].rolling(ROLLING_ZSCORE_DAYS, min_periods=MIN_ZSCORE_OBS)
data["cheapness_zscore"] = (
    data["cheapness_log_ratio"] - rolling.mean()
) / rolling.std(ddof=0)
data["cheapness_zscore_lag1"] = data["cheapness_zscore"].shift(1)
data["decile"] = assign_deciles(data["cheapness_zscore"])
data["decile_lag1"] = assign_deciles(data["cheapness_zscore_lag1"])

display(
    Markdown(
        f"""
Loaded **{len(data):,}** aligned VX30/VIX3M observations from **{data.index.min().date()}**
through **{data.index.max().date()}**.

- VX contracts discovered: **{contracts['ticker'].nunique():,}**
- VX futures session rows after DTE filters: **{len(futures_aggs):,}**
- VX30 construction mix:
"""
    )
)
display(vx30["construction"].value_counts().to_frame("sessions"))
display(data[["VX30", "VIX3M", "cheapness_log_ratio", "cheapness_zscore"]].tail())

# %% [markdown]
# ## Visual Check
#
# First, inspect the raw series and the standardized cheapness signal. Lower
# z-scores mean VX30 is cheap versus its own recent history against VIX3M.

# %%
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
data[["VX30", "VIX3M"]].plot(ax=axes[0], lw=1.2)
axes[0].set_title("VX30 proxy vs VIX3M")
axes[0].set_ylabel("Index / futures price")

data["cheapness_zscore"].plot(ax=axes[1], color="tab:purple", lw=1.0)
axes[1].axhline(ENTRY_ZSCORE, color="tab:red", ls="--", label=f"entry z = {ENTRY_ZSCORE:g}")
axes[1].axhline(0, color="black", lw=0.8)
axes[1].set_title("252-day z-score of log(VX30 / VIX3M)")
axes[1].set_ylabel("z-score")
axes[1].legend()
plt.tight_layout()

# %% [markdown]
# ## Dirty Decile Sort
#
# Sort every daily observation by the cheapness z-score, then average subsequent
# VX30 log returns. Decile 1 is the cheapest 10% of observations; decile 10 is the
# richest 10%.

# %%
decile_summary = (
    data.dropna(subset=["decile", "vx30_fwd_1d_log_return"])
    .groupby("decile")
    .agg(
        observations=("vx30_fwd_1d_log_return", "size"),
        mean_next_day_log_return=("vx30_fwd_1d_log_return", "mean"),
        median_next_day_log_return=("vx30_fwd_1d_log_return", "median"),
        positive_rate=("vx30_fwd_1d_log_return", lambda x: float((x > 0).mean())),
        mean_zscore=("cheapness_zscore", "mean"),
    )
)

decile_summary_lag1 = (
    data.dropna(subset=["decile_lag1", "vx30_fwd_1d_log_return"])
    .groupby("decile_lag1")
    .agg(
        observations=("vx30_fwd_1d_log_return", "size"),
        mean_next_day_log_return=("vx30_fwd_1d_log_return", "mean"),
        median_next_day_log_return=("vx30_fwd_1d_log_return", "median"),
        positive_rate=("vx30_fwd_1d_log_return", lambda x: float((x > 0).mean())),
        mean_lagged_zscore=("cheapness_zscore_lag1", "mean"),
    )
)

display(
    decile_summary.style.format(
        {
            "observations": "{:.0f}",
            "mean_next_day_log_return": "{:.3%}",
            "median_next_day_log_return": "{:.3%}",
            "positive_rate": "{:.1%}",
            "mean_zscore": "{:.2f}",
        }
    )
)

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
(decile_summary["mean_next_day_log_return"] * 100).plot.bar(ax=axes[0], color="tab:blue")
axes[0].axhline(0, color="black", lw=0.8)
axes[0].set_title("Same-day signal sort")
axes[0].set_xlabel("Cheapness z-score decile")
axes[0].set_ylabel("Mean next-day VX30 log return (%)")

(decile_summary_lag1["mean_next_day_log_return"] * 100).plot.bar(ax=axes[1], color="tab:orange")
axes[1].axhline(0, color="black", lw=0.8)
axes[1].set_title("Signal lagged by one session")
axes[1].set_xlabel("Lagged cheapness z-score decile")
plt.tight_layout()

# %% [markdown]
# ## Forward Return Diagnostics
#
# Check whether the cheapness signal is monotonic across several VX30 forward
# return horizons. These are diagnostics, not a trade implementation.

# %%
forward_summary = []
for horizon in FORWARD_RETURN_HORIZONS:
    column = f"vx30_fwd_{horizon}d_log_return"
    grouped = data.dropna(subset=["decile", column]).groupby("decile")[column].mean()
    forward_summary.append(grouped.rename(f"{horizon}d"))
forward_summary = pd.concat(forward_summary, axis=1)
display(forward_summary.style.format("{:.3%}"))

fig, ax = plt.subplots(figsize=(11, 5))
(forward_summary * 100).plot(marker="o", ax=ax)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Mean VX30 forward log returns by cheapness decile")
ax.set_xlabel("Cheapness z-score decile")
ax.set_ylabel("Mean forward log return (%)")
plt.tight_layout()

# %% [markdown]
# ## Threshold Strategy
#
# The trading rule is intentionally simple:
#
# 1. Compute `zscore(log(VX30 / VIX3M), 252 trading days)`.
# 2. Get long VX30 when the z-score is below the threshold.
# 3. Close the long when the z-score is no longer below the threshold.
#
# The primary simulation below waits one full session before acting on the
# signal, matching the stricter "lag it by a day" check.

# %%
same_day_strategy = strategy_returns(
    data["cheapness_zscore"],
    data["vx30_fwd_1d_log_return"],
    threshold=ENTRY_ZSCORE,
    execution_lag_sessions=0,
)
lagged_strategy = strategy_returns(
    data["cheapness_zscore"],
    data["vx30_fwd_1d_log_return"],
    threshold=ENTRY_ZSCORE,
    execution_lag_sessions=1,
)

strategy_summary = pd.concat(
    {
        "Always long VX30": summarize_log_returns(data["vx30_fwd_1d_log_return"]),
        f"Long VX30 z < {ENTRY_ZSCORE:g}": summarize_log_returns(
            same_day_strategy["strategy_log_return"]
        ),
        f"Long VX30 z < {ENTRY_ZSCORE:g}, lag 1": summarize_log_returns(
            lagged_strategy["strategy_log_return"]
        ),
    },
    axis=1,
).T
strategy_summary["exposure"] = [
    1.0,
    same_day_strategy["position"].mean(),
    lagged_strategy["position"].mean(),
]
strategy_summary["trades_per_year"] = [
    np.nan,
    same_day_strategy["trade"].sum() / len(same_day_strategy) * TRADING_DAYS_PER_YEAR,
    lagged_strategy["trade"].sum() / len(lagged_strategy) * TRADING_DAYS_PER_YEAR,
]
display(
    strategy_summary.style.format(
        {
            "observations": "{:.0f}",
            "total_return": "{:.1%}",
            "ann_return": "{:.1%}",
            "ann_vol": "{:.1%}",
            "sharpe": "{:.2f}",
            "max_drawdown": "{:.1%}",
            "positive_days": "{:.1%}",
            "exposure": "{:.1%}",
            "trades_per_year": "{:.1f}",
        }
    )
)

# %%
equity = pd.DataFrame(
    {
        "Always long VX30": np.exp(data["vx30_fwd_1d_log_return"].fillna(0).cumsum()),
        f"z < {ENTRY_ZSCORE:g}": np.exp(
            same_day_strategy["strategy_log_return"].fillna(0).cumsum()
        ),
        f"z < {ENTRY_ZSCORE:g}, lag 1": np.exp(
            lagged_strategy["strategy_log_return"].fillna(0).cumsum()
        ),
    }
)

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
equity.plot(ax=axes[0], lw=1.3)
axes[0].set_title("Dirty VX30 cheapness strategy equity curves")
axes[0].set_ylabel("Growth of $1")

lagged_strategy["position"].plot(ax=axes[1], color="tab:green", lw=0.9)
axes[1].set_title("Lagged strategy position")
axes[1].set_ylabel("Long VX30 exposure")
axes[1].set_ylim(-0.05, 1.05)
plt.tight_layout()

# %% [markdown]
# ## Threshold Sensitivity
#
# Sweep several entry thresholds. The lagged version is the more conservative
# interpretation because it requires waiting one full session after the signal.

# %%
threshold_rows = []
for threshold in THRESHOLD_GRID:
    for lag in [0, 1]:
        sim = strategy_returns(
            data["cheapness_zscore"],
            data["vx30_fwd_1d_log_return"],
            threshold=threshold,
            execution_lag_sessions=lag,
        )
        summary = summarize_log_returns(sim["strategy_log_return"])
        threshold_rows.append(
            {
                "threshold": threshold,
                "lag_sessions": lag,
                "ann_return": summary.get("ann_return", np.nan),
                "ann_vol": summary.get("ann_vol", np.nan),
                "sharpe": summary.get("sharpe", np.nan),
                "max_drawdown": summary.get("max_drawdown", np.nan),
                "exposure": sim["position"].mean(),
                "trades_per_year": sim["trade"].sum() / len(sim) * TRADING_DAYS_PER_YEAR,
            }
        )
threshold_sensitivity = pd.DataFrame(threshold_rows)
display(
    threshold_sensitivity.style.format(
        {
            "threshold": "{:.2f}",
            "ann_return": "{:.1%}",
            "ann_vol": "{:.1%}",
            "sharpe": "{:.2f}",
            "max_drawdown": "{:.1%}",
            "exposure": "{:.1%}",
            "trades_per_year": "{:.1f}",
        }
    )
)

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharex=True)
for lag, group in threshold_sensitivity.groupby("lag_sessions"):
    label = "same day" if lag == 0 else "lag 1 session"
    axes[0].plot(group["threshold"], group["sharpe"], marker="o", label=label)
    axes[1].plot(group["threshold"], group["ann_return"] * 100, marker="o", label=label)
axes[0].axhline(0, color="black", lw=0.8)
axes[0].set_title("Sharpe by entry threshold")
axes[0].set_xlabel("Entry z-score")
axes[0].set_ylabel("Sharpe")
axes[0].legend()

axes[1].axhline(0, color="black", lw=0.8)
axes[1].set_title("Annualized log-return by entry threshold")
axes[1].set_xlabel("Entry z-score")
axes[1].set_ylabel("Annualized return (%)")
axes[1].legend()
plt.tight_layout()

# %% [markdown]
# ## Limitations
#
# - The comparison is intentionally biased: VIX3M is an SPX implied-volatility
#   index, while VX30 is a futures price built from listed VIX futures.
# - The VX30 construction is a price-level interpolation. It does not replicate
#   VIX futures from SPX/VIX options, does not model convexity, and does not
#   match an official constant-maturity index.
# - Futures settlement and index timestamps are assumed to be comparable at a
#   daily frequency.
# - The backtest ignores costs and assumes exposure can be obtained at the daily
#   VX30 proxy mark.
# - Long volatility returns are episodic. Threshold performance can be dominated
#   by crisis windows and the sample covered by Massive futures data.
#
# ## Conclusion
#
# This notebook implements the deliberately dirty idea: use VIX3M as the SPX
# implied-volatility anchor, use a Massive-derived 30-day VIX futures proxy as
# the tradeable leg, standardize `log(VX30 / VIX3M)` against its own one-year
# history, and get long VX30 only when that ratio is extremely low.
#
# The important checks are whether decile 1 has meaningfully positive forward
# VX30 returns and whether the threshold rule still looks useful after a full
# one-session signal lag. If those survive, the result is evidence for a rough
# long-volatility dislocation signal rather than a precise pricing model.

