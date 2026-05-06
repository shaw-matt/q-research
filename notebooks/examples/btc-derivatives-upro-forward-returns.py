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
# # BTC Derivatives Signals and UPRO Forward Returns
#
# ## Research Question
#
# Can Bitcoin derivatives positioning and option-price information predict UPRO
# forward returns?
#
# ## Hypothesis
#
# If Bitcoin open interest is expanding and Bitcoin option prices show improving
# risk appetite, then the same liquidity and risk-on impulse may lead U.S. equity
# beta and produce positive forward returns for UPRO.

# %%
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from dotenv import load_dotenv
from IPython.display import Markdown, display

from research.massive_flatfiles import (
    align_btc_to_equity_close,
    build_equity_close_times,
    download_flatfile_btc_hourly_closes as download_btc_hourly,
    download_flatfile_stock_day_closes as download_equity_closes,
)
from research.plotting import apply_default_style

load_dotenv(dotenv_path=".env")
apply_default_style()

# %% [markdown]
# ## Assumptions
#
# - UPRO daily closes come from Massive US stock day-aggregate flat files and
#   represent executable 4pm New York equity closes.
# - BTC spot prices come from Massive crypto minute flat files resampled to hourly
#   closes, then aligned to the latest completed hourly close at or before each
#   equity close.
# - Historical Bitcoin derivatives open-interest and option-surface fields are
#   not present in the Massive flat-file helpers used by the existing notebooks.
#   This notebook therefore looks for a daily user-supplied derivatives file and
#   makes the coverage explicit before testing the thesis.
# - Derivatives features are lagged by one equity session by default. This avoids
#   using end-of-day open-interest or option marks that may not have been known at
#   the U.S. equity close.
# - The tests are predictive diagnostics, not an executable trading model. They
#   ignore transaction costs, slippage, taxes, financing, and UPRO path-dependent
#   leverage effects.
#
# ## Data Sources
#
# - Massive S3 flat files: `us_stocks_sip/day_aggs_v1` for UPRO daily closes.
# - Massive S3 flat files: global crypto `minute_aggs_v1` for X:BTC-USD, resampled
#   to hourly and aligned to U.S. equity closes.
# - Optional local derivatives file:
#   `data/btc_derivatives_daily.csv`, `data/btc_derivatives_daily.parquet`, or the
#   path in `BTC_DERIVATIVES_DAILY_PATH`.
#
# Expected optional derivatives columns include a `date` column plus any of:
#
# - `btc_open_interest_usd`
# - `btc_option_open_interest_usd`
# - `btc_option_volume_usd`
# - `btc_call_volume_usd`
# - `btc_put_volume_usd`
# - `btc_atm_iv_30d`
# - `btc_dvol`
# - `btc_25delta_risk_reversal_30d`
# - `btc_25delta_butterfly_30d`

# %%
START_DATE = "2020-01-01"
END_DATE = datetime.now(UTC).date().isoformat()

FORWARD_RETURN_HORIZONS = [1, 5, 10, 21]
DERIVATIVES_FEATURE_LAG_SESSIONS = 1
ROLLING_ZSCORE_DAYS = 63
MIN_TEST_OBSERVATIONS = 60
TRADING_DAYS_PER_YEAR = 252

DERIVATIVES_PATH_CANDIDATES = [
    Path(os.environ["BTC_DERIVATIVES_DAILY_PATH"])
    for _ in [0]
    if os.getenv("BTC_DERIVATIVES_DAILY_PATH")
] + [
    Path("data/btc_derivatives_daily.parquet"),
    Path("data/btc_derivatives_daily.csv"),
]

BASELINE_FEATURES = [
    "BTC_return",
    "BTC_5d_return",
    "BTC_21d_return",
    "BTC_realized_vol_21d",
]


@dataclass(frozen=True)
class FeatureDefinition:
    column: str
    label: str
    family: str


FEATURE_DEFINITIONS = [
    FeatureDefinition("BTC_return", "BTC 1-day return", "spot control"),
    FeatureDefinition("BTC_5d_return", "BTC 5-day return", "spot control"),
    FeatureDefinition("BTC_21d_return", "BTC 21-day return", "spot control"),
    FeatureDefinition("BTC_realized_vol_21d", "BTC 21-day realized volatility", "spot control"),
    FeatureDefinition("btc_open_interest_log", "BTC futures/perp open interest level", "open interest"),
    FeatureDefinition("btc_open_interest_1d_change", "BTC open interest 1-day change", "open interest"),
    FeatureDefinition("btc_open_interest_5d_change", "BTC open interest 5-day change", "open interest"),
    FeatureDefinition("btc_open_interest_zscore", "BTC open interest z-score", "open interest"),
    FeatureDefinition("btc_open_interest_change_zscore", "BTC open interest change z-score", "open interest"),
    FeatureDefinition("btc_option_open_interest_log", "BTC option open interest level", "options"),
    FeatureDefinition("btc_option_volume_zscore", "BTC option volume z-score", "options"),
    FeatureDefinition("btc_call_put_volume_ratio", "BTC call/put volume ratio", "options"),
    FeatureDefinition("btc_atm_iv_30d", "BTC 30-day ATM IV", "options"),
    FeatureDefinition("btc_atm_iv_30d_change", "BTC 30-day ATM IV change", "options"),
    FeatureDefinition("btc_dvol", "BTC implied volatility index", "options"),
    FeatureDefinition("btc_dvol_change", "BTC implied volatility index change", "options"),
    FeatureDefinition("btc_25delta_risk_reversal_30d", "BTC 25-delta risk reversal", "options"),
    FeatureDefinition("btc_25delta_butterfly_30d", "BTC 25-delta butterfly", "options"),
]

FEATURE_LABELS = {definition.column: definition.label for definition in FEATURE_DEFINITIONS}
FEATURE_FAMILIES = {definition.column: definition.family for definition in FEATURE_DEFINITIONS}


def canonicalize_column_name(column: object) -> str:
    """Normalize file columns to lower snake-case names."""
    cleaned = str(column).strip().lower()
    cleaned = cleaned.replace("%", "pct").replace("/", "_").replace("-", "_")
    cleaned = cleaned.replace(" ", "_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def read_derivatives_file(path: Path) -> pd.DataFrame:
    """Read a local daily derivatives file from CSV or Parquet."""
    if path.suffix.lower() == ".parquet":
        raw = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        raw = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported derivatives file extension: {path.suffix}")

    raw = raw.rename(columns={column: canonicalize_column_name(column) for column in raw.columns})
    if "date" in raw.columns:
        index = pd.to_datetime(raw.pop("date"), errors="coerce")
    else:
        index = pd.to_datetime(raw.index, errors="coerce")
    raw = raw.loc[index.notna()].copy()
    raw.index = pd.DatetimeIndex(index[index.notna()]).normalize()
    raw = raw.sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]

    for column in raw.columns:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    numeric = raw.select_dtypes(include=[np.number]).copy()
    return numeric


def load_derivatives_daily(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the first available derivatives dataset and return a status table."""
    status_rows = []
    for path in paths:
        exists = path.exists()
        status_rows.append(
            {
                "candidate_path": path.as_posix(),
                "exists": exists,
                "selected": False,
                "message": "available" if exists else "not found",
            }
        )
        if not exists:
            continue

        frame = read_derivatives_file(path)
        status_rows[-1]["selected"] = True
        status_rows[-1]["message"] = f"loaded {len(frame):,} rows and {len(frame.columns):,} numeric columns"
        return frame, pd.DataFrame(status_rows)

    return pd.DataFrame(), pd.DataFrame(status_rows)


def rolling_zscore(series: pd.Series, lookback: int = ROLLING_ZSCORE_DAYS) -> pd.Series:
    """Standardize a series with rolling mean and volatility."""
    mean = series.rolling(lookback).mean()
    volatility = series.rolling(lookback).std()
    return (series - mean) / volatility


def add_if_present(frame: pd.DataFrame, output: pd.DataFrame, column: str) -> None:
    if column in frame:
        output[column] = frame[column].astype(float)


def build_derivative_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert raw open-interest and option-price fields into daily predictors."""
    if raw.empty:
        return pd.DataFrame(index=raw.index)

    frame = raw.copy()
    features = pd.DataFrame(index=frame.index)

    if "btc_open_interest_usd" in frame:
        oi = frame["btc_open_interest_usd"].replace(0, np.nan).astype(float)
        features["btc_open_interest_log"] = np.log(oi)
        features["btc_open_interest_1d_change"] = features["btc_open_interest_log"].diff()
        features["btc_open_interest_5d_change"] = features["btc_open_interest_log"].diff(5)
        features["btc_open_interest_zscore"] = rolling_zscore(features["btc_open_interest_log"])
        features["btc_open_interest_change_zscore"] = rolling_zscore(
            features["btc_open_interest_5d_change"]
        )

    if "btc_option_open_interest_usd" in frame:
        option_oi = frame["btc_option_open_interest_usd"].replace(0, np.nan).astype(float)
        features["btc_option_open_interest_log"] = np.log(option_oi)

    if "btc_option_volume_usd" in frame:
        features["btc_option_volume_zscore"] = rolling_zscore(
            np.log(frame["btc_option_volume_usd"].replace(0, np.nan).astype(float))
        )

    if {"btc_call_volume_usd", "btc_put_volume_usd"}.issubset(frame.columns):
        call_volume = frame["btc_call_volume_usd"].replace(0, np.nan).astype(float)
        put_volume = frame["btc_put_volume_usd"].replace(0, np.nan).astype(float)
        features["btc_call_put_volume_ratio"] = np.log(call_volume / put_volume)

    for column in [
        "btc_atm_iv_30d",
        "btc_dvol",
        "btc_25delta_risk_reversal_30d",
        "btc_25delta_butterfly_30d",
    ]:
        add_if_present(frame, features, column)

    for column in ["btc_atm_iv_30d", "btc_dvol"]:
        if column in features:
            features[f"{column}_change"] = features[column].diff()

    return features.replace([np.inf, -np.inf], np.nan)


def max_drawdown(return_series: pd.Series) -> float:
    equity_curve = (1 + return_series.fillna(0)).cumprod()
    drawdown = equity_curve / equity_curve.cummax() - 1
    return float(drawdown.min())


def add_forward_returns(frame: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    output = frame.copy()
    for horizon in horizons:
        output[f"UPRO_fwd_{horizon}d_return"] = output["UPRO"].shift(-horizon) / output["UPRO"] - 1
    return output


def summarize_data_coverage(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        series = frame[column].dropna() if column in frame else pd.Series(dtype=float)
        rows.append(
            {
                "column": column,
                "label": FEATURE_LABELS.get(column, column),
                "family": FEATURE_FAMILIES.get(column, "unknown"),
                "observations": int(series.count()),
                "first_date": series.index.min() if not series.empty else pd.NaT,
                "last_date": series.index.max() if not series.empty else pd.NaT,
                "coverage_rate": series.count() / len(frame) if len(frame) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def hac_univariate_regression(sample: pd.DataFrame, feature: str, target: str, horizon: int) -> dict[str, float]:
    """Estimate a one-feature predictive regression with HAC robust errors."""
    y = sample[target].astype(float)
    x = sample[feature].astype(float)
    x_std = (x - x.mean()) / x.std()
    model = sm.OLS(y, sm.add_constant(x_std)).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": max(1, min(5, horizon))},
    )
    return {
        "ols_beta_per_1sd": float(model.params[feature]),
        "ols_t_stat": float(model.tvalues[feature]),
        "ols_p_value": float(model.pvalues[feature]),
        "r_squared": float(model.rsquared),
    }


def quantile_spread(sample: pd.DataFrame, feature: str, target: str) -> dict[str, float]:
    """Compare forward returns in high-feature and low-feature quintiles."""
    ranked = sample[[feature, target]].dropna().copy()
    try:
        ranked["bucket"] = pd.qcut(ranked[feature], 5, labels=False, duplicates="drop")
    except ValueError:
        return {
            "bottom_quintile_mean": np.nan,
            "top_quintile_mean": np.nan,
            "top_minus_bottom": np.nan,
        }
    if ranked["bucket"].nunique() < 2:
        return {
            "bottom_quintile_mean": np.nan,
            "top_quintile_mean": np.nan,
            "top_minus_bottom": np.nan,
        }

    bottom = ranked.loc[ranked["bucket"] == ranked["bucket"].min(), target]
    top = ranked.loc[ranked["bucket"] == ranked["bucket"].max(), target]
    return {
        "bottom_quintile_mean": float(bottom.mean()),
        "top_quintile_mean": float(top.mean()),
        "top_minus_bottom": float(top.mean() - bottom.mean()),
    }


def run_predictive_tests(
    frame: pd.DataFrame,
    features: list[str],
    horizons: list[int],
    *,
    min_observations: int = MIN_TEST_OBSERVATIONS,
) -> pd.DataFrame:
    """Run univariate rank-correlation, quantile, and robust regression tests."""
    rows = []
    for feature in features:
        if feature not in frame:
            continue
        for horizon in horizons:
            target = f"UPRO_fwd_{horizon}d_return"
            sample = frame[[feature, target]].dropna()
            if len(sample) < min_observations or sample[feature].std() == 0:
                continue

            spearman = stats.spearmanr(sample[feature], sample[target], nan_policy="omit")
            pearson = stats.pearsonr(sample[feature], sample[target])
            spread = quantile_spread(sample, feature, target)
            regression = hac_univariate_regression(sample, feature, target, horizon)
            rows.append(
                {
                    "feature": feature,
                    "label": FEATURE_LABELS.get(feature, feature),
                    "family": FEATURE_FAMILIES.get(feature, "unknown"),
                    "horizon_days": horizon,
                    "observations": len(sample),
                    "spearman_ic": float(spearman.statistic),
                    "spearman_p_value": float(spearman.pvalue),
                    "pearson_ic": float(pearson.statistic),
                    "pearson_p_value": float(pearson.pvalue),
                    **spread,
                    **regression,
                }
            )
    return pd.DataFrame(rows)


def build_directional_signal_backtest(
    frame: pd.DataFrame,
    feature: str,
    *,
    horizon: int = 1,
    high_quantile: float = 0.80,
    low_quantile: float = 0.20,
) -> pd.DataFrame:
    """Build a simple long/flat daily signal from rolling feature quantiles."""
    target = f"UPRO_fwd_{horizon}d_return"
    signal_frame = frame[[feature, "UPRO_return", target]].dropna().copy()
    rolling_high = signal_frame[feature].rolling(252, min_periods=63).quantile(high_quantile)
    rolling_low = signal_frame[feature].rolling(252, min_periods=63).quantile(low_quantile)

    same_direction_sample = signal_frame[[feature, target]].dropna()
    ic = (
        stats.spearmanr(same_direction_sample[feature], same_direction_sample[target]).statistic
        if len(same_direction_sample) >= MIN_TEST_OBSERVATIONS
        else np.nan
    )
    if pd.isna(ic) or ic >= 0:
        signal_at_close = signal_frame[feature] >= rolling_high
        direction = "high feature"
    else:
        signal_at_close = signal_frame[feature] <= rolling_low
        direction = "low feature"

    signal_frame["signal_direction"] = direction
    signal_frame["signal_at_close"] = signal_at_close.fillna(False)
    signal_frame["position"] = signal_frame["signal_at_close"].shift(1).fillna(False)
    signal_frame["strategy_return"] = np.where(
        signal_frame["position"],
        signal_frame["UPRO_return"],
        0.0,
    )
    return signal_frame


def summarize_signal_backtest(signal_frame: pd.DataFrame) -> pd.DataFrame:
    returns = signal_frame["strategy_return"].dropna()
    positions = signal_frame["position"].reindex(returns.index).fillna(False)
    active_returns = returns.loc[positions]
    equity = (1 + returns).cumprod()
    total_return = equity.iloc[-1] - 1 if not equity.empty else np.nan
    annualized_return = (
        equity.iloc[-1] ** (TRADING_DAYS_PER_YEAR / len(returns)) - 1
        if len(returns) and equity.iloc[-1] > 0
        else np.nan
    )
    annualized_volatility = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = (
        returns.mean() / returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        if returns.std() > 0
        else np.nan
    )
    return pd.DataFrame(
        {
            "metric": [
                "observations",
                "active_days",
                "exposure_rate",
                "total_return",
                "annualized_return",
                "annualized_volatility",
                "sharpe_ratio",
                "max_drawdown",
                "active_day_win_rate",
                "average_active_day_return",
            ],
            "value": [
                len(returns),
                int(positions.sum()),
                positions.mean(),
                total_return,
                annualized_return,
                annualized_volatility,
                sharpe,
                max_drawdown(returns),
                (active_returns > 0).mean() if len(active_returns) else np.nan,
                active_returns.mean() if len(active_returns) else np.nan,
            ],
        }
    )

# %% [markdown]
# ## Methodology
#
# 1. Download UPRO daily closes and BTC spot closes aligned to each U.S. equity
#    close.
# 2. Load a daily BTC derivatives feature file if one is present.
# 3. Transform raw derivatives fields into stationary predictors: log open
#    interest, open-interest changes, rolling z-scores, option implied-volatility
#    changes, risk-reversal/skew fields, and option-activity z-scores.
# 4. Lag derivatives features by one equity session by default.
# 5. Measure predictive power against 1-, 5-, 10-, and 21-session forward UPRO
#    returns with:
#    - Spearman and Pearson information coefficients.
#    - Top-minus-bottom quintile forward-return spreads.
#    - HAC-robust univariate predictive regressions.
# 6. Build a simple daily long/flat UPRO signal from the strongest available
#    one-day feature as an implementation sanity check.

# %% [markdown]
# ## Data

# %%
equity_closes = download_equity_closes(["UPRO"], START_DATE, END_DATE)
btc_hourly_close = download_btc_hourly(START_DATE, END_DATE)

if equity_closes.empty:
    raise ValueError("No UPRO equity closes were downloaded.")
if btc_hourly_close.empty:
    raise ValueError("No BTC-USD hourly closes were downloaded.")

equity_close_times = build_equity_close_times(equity_closes.index)
btc_close = align_btc_to_equity_close(btc_hourly_close, equity_close_times)

prices = equity_closes.join(btc_close, how="inner").dropna()
prices = prices.rename(columns={"btc_close_at_equity_close": "BTC"})
prices.tail()

# %%
derivatives_raw, derivatives_load_status = load_derivatives_daily(DERIVATIVES_PATH_CANDIDATES)
derivatives_load_status

# %%
if derivatives_raw.empty:
    display(
        Markdown(
            "### Derivatives data status\n\n"
            "No local BTC derivatives file was found, so the notebook renders the Massive "
            "UPRO/BTC spot-control baseline and leaves the open-interest/option-price "
            "tests inactive. Add `data/btc_derivatives_daily.csv` or set "
            "`BTC_DERIVATIVES_DAILY_PATH` to activate the thesis-specific tests."
        )
    )
else:
    display(
        Markdown(
            f"### Derivatives data status\n\n"
            f"Loaded **{len(derivatives_raw):,}** daily rows with "
            f"**{len(derivatives_raw.columns):,}** numeric fields."
        )
    )
    display(derivatives_raw.tail())

# %%
derivative_features = build_derivative_features(derivatives_raw)
if not derivative_features.empty:
    derivative_features = derivative_features.shift(DERIVATIVES_FEATURE_LAG_SESSIONS)

spot_features = prices.copy()
spot_features["UPRO_return"] = spot_features["UPRO"].pct_change()
spot_features["BTC_return"] = spot_features["BTC"].pct_change()
spot_features["BTC_5d_return"] = spot_features["BTC"].pct_change(5)
spot_features["BTC_21d_return"] = spot_features["BTC"].pct_change(21)
spot_features["BTC_realized_vol_21d"] = (
    spot_features["BTC_return"].rolling(21).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
)

analysis = spot_features.join(derivative_features, how="left")
analysis = add_forward_returns(analysis, FORWARD_RETURN_HORIZONS)
analysis.tail()

# %%
available_feature_columns = [
    definition.column
    for definition in FEATURE_DEFINITIONS
    if definition.column in analysis and analysis[definition.column].notna().sum() >= MIN_TEST_OBSERVATIONS
]

coverage = summarize_data_coverage(analysis, [definition.column for definition in FEATURE_DEFINITIONS])
coverage.loc[coverage["observations"] > 0].sort_values(
    ["family", "observations"],
    ascending=[True, False],
)

# %% [markdown]
# ## Analysis
#
# The first table reports feature coverage. If the derivatives file is absent,
# only the BTC spot-control rows will be available. Once open-interest or option
# columns are supplied, the same tests below automatically include them.

# %%
coverage

# %%
predictive_tests = run_predictive_tests(
    analysis,
    available_feature_columns,
    FORWARD_RETURN_HORIZONS,
)

if predictive_tests.empty:
    display(
        Markdown(
            "No feature has enough observations for the predictive tests. "
            f"The current minimum is {MIN_TEST_OBSERVATIONS} aligned sessions."
        )
    )
else:
    predictive_tests.sort_values(
        ["horizon_days", "spearman_ic"],
        ascending=[True, False],
    )

# %%
thesis_feature_tests = (
    predictive_tests.loc[predictive_tests["family"].isin(["open interest", "options"])]
    if not predictive_tests.empty
    else pd.DataFrame()
)

if thesis_feature_tests.empty:
    display(
        Markdown(
            "No open-interest or option-price feature has enough observations yet. "
            "The thesis-specific evidence table will populate after the derivatives "
            "file is added."
        )
    )
else:
    thesis_feature_tests.sort_values(
        ["horizon_days", "spearman_ic"],
        ascending=[True, False],
    )

# %%
if not predictive_tests.empty:
    family_priority = {"open interest": 0, "options": 1, "spot control": 2}
    best_one_day = (
        predictive_tests.loc[predictive_tests["horizon_days"] == 1]
        .assign(abs_spearman_ic=lambda frame: frame["spearman_ic"].abs())
        .assign(family_priority=lambda frame: frame["family"].map(family_priority).fillna(99))
        .sort_values(["family_priority", "abs_spearman_ic"], ascending=[True, False])
    )
    best_one_day
else:
    best_one_day = pd.DataFrame()
    best_one_day

# %%
if not best_one_day.empty:
    selected_feature = best_one_day.iloc[0]["feature"]
    selected_label = best_one_day.iloc[0]["label"]
    signal_backtest = build_directional_signal_backtest(analysis, selected_feature, horizon=1)
    display(Markdown(f"### Simple Signal Sanity Check\n\nSelected feature: **{selected_label}**."))
    display(summarize_signal_backtest(signal_backtest))
else:
    selected_feature = None
    signal_backtest = pd.DataFrame()

# %% [markdown]
# ## Visualizations

# %%
fig, ax = plt.subplots()
prices[["UPRO", "BTC"]].div(prices[["UPRO", "BTC"]].iloc[0]).plot(ax=ax)
ax.set_title("Normalized UPRO and BTC Spot Prices")
ax.set_xlabel("Date")
ax.set_ylabel("Growth of $1")
plt.show()

# %%
if not derivative_features.empty:
    plot_columns = [
        column
        for column in [
            "btc_open_interest_log",
            "btc_open_interest_zscore",
            "btc_atm_iv_30d",
            "btc_dvol",
            "btc_25delta_risk_reversal_30d",
        ]
        if column in derivative_features
    ]
    if plot_columns:
        fig, ax = plt.subplots()
        derivative_features[plot_columns].dropna(how="all").plot(ax=ax)
        ax.set_title("BTC Derivatives Feature History")
        ax.set_xlabel("Date")
        ax.legend()
        plt.show()

# %%
if not predictive_tests.empty:
    fig, ax = plt.subplots(figsize=(12, 6))
    ic_plot = predictive_tests.pivot_table(
        index="label",
        columns="horizon_days",
        values="spearman_ic",
        aggfunc="first",
    )
    sort_horizon = 1 if 1 in ic_plot.columns else ic_plot.columns.min()
    ic_plot = ic_plot.reindex(ic_plot[sort_horizon].abs().sort_values(ascending=False).index)
    ic_plot.plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Spearman Information Coefficient by Feature and Horizon")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Spearman IC")
    ax.legend(title="Forward sessions")
    fig.tight_layout()
    plt.show()

# %%
if not predictive_tests.empty:
    spread_plot = predictive_tests.loc[predictive_tests["horizon_days"] == 5].copy()
    if not spread_plot.empty:
        spread_plot = spread_plot.sort_values("top_minus_bottom")
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.barh(spread_plot["label"], spread_plot["top_minus_bottom"])
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title("5-Day Forward UPRO Return: Top Minus Bottom Feature Quintile")
        ax.set_xlabel("Average return spread")
        fig.tight_layout()
        plt.show()

# %%
if not signal_backtest.empty:
    strategy_curve = (1 + signal_backtest["strategy_return"]).cumprod()
    upro_curve = (1 + signal_backtest["UPRO_return"].fillna(0)).cumprod().reindex(strategy_curve.index)
    fig, ax = plt.subplots()
    strategy_curve.plot(ax=ax, label="Signal long/flat UPRO")
    upro_curve.plot(ax=ax, label="Buy and hold UPRO", alpha=0.75)
    ax.set_title("Simple Signal Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    plt.show()

# %% [markdown]
# ## Limitations
#
# - This notebook cannot validate the open-interest or option-price thesis until
#   a historical BTC derivatives dataset is supplied. Without that file, the
#   rendered output is a BTC spot-control baseline plus a schema and test harness.
# - The local derivatives file may combine venues with different reporting times.
#   The default one-session lag is conservative but may be too conservative for
#   intraday derivatives marks known before the U.S. equity close.
# - Open interest can rise because of long or short positioning; direction is not
#   identifiable without additional long/short or dealer-positioning data.
# - Option implied volatility and skew can indicate either risk appetite or crash
#   demand. The univariate tests here should be followed with controls for BTC
#   spot momentum, BTC volatility, SPY/QQQ returns, and macro-volatility regimes.
# - UPRO is a 3x S&P 500 ETF. A Bitcoin-linked risk-on signal may map differently
#   to QQQ, TQQQ, SPY, or unlevered equity exposure.
# - Multiple comparisons are not adjusted for; p-values are exploratory.
#
# ## Conclusion
#
# This notebook creates the investigation framework for the thesis that Bitcoin
# open interest and option prices can predict UPRO forward returns. It aligns
# UPRO and BTC spot data to the equity close, documents the required derivatives
# schema, checks feature coverage, and runs rank-correlation, quintile-spread, and
# HAC regression diagnostics for each available feature and horizon. The thesis
# should be judged from the open-interest and option rows once the daily
# derivatives feature file is present.
#
# ## Next Research Ideas
#
# - Add venue-level BTC futures and perpetual open interest to separate CME,
#   Binance, OKX, and Deribit positioning.
# - Add option-surface features by tenor and delta: ATM IV, 25-delta risk
#   reversal, butterflies, term structure, and call/put open-interest imbalance.
# - Compare UPRO results with SPY, QQQ, TQQQ, and BTC itself to separate broad
#   risk-on effects from UPRO-specific leverage.
# - Use expanding or walk-forward models that choose features on an in-sample
#   window and evaluate them out of sample.
# - Add transaction costs and slippage if the diagnostics graduate into a live
#   UPRO strategy.
