"""Calendar-aligned origin features for global XGBoost V1.

Forecasts are issued after observing sales at origin T.
All historical predictors must use data available no later than T.
"""

import numpy as np
import pandas as pd


SERIES_KEYS = ["store_nbr", "family"]

# Frozen lag configuration from Notebook 05.
LAG_DAYS_V1 = (1, 2, 3, 7, 14, 21, 28, 35, 364)

# Preserve the exact origin-feature column order of Notebook 05.
ORIGIN_FEATURE_COLUMNS_V1 = (
    "store_nbr", "family", "forecast_origin", "sales_observed",
    "sales_at_origin", "sales_lag_1d", "sales_lag_2d",
    "sales_lag_3d", "sales_lag_7d", "sales_lag_14d",
    "sales_lag_21d", "sales_lag_28d", "sales_lag_35d",
    "sales_lag_364d", "rolling_mean_7d", "rolling_mean_14d",
    "rolling_mean_28d", "recent_level_ratio",
    "rolling_std_28d", "observed_days_28d", "zero_share_28d",
    "positive_rate_28d", "positive_sales_mean_28d",
    "days_since_last_positive", "history_days_available",
)


def regularize_sales_v1(
    sales: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Build a complete daily grid using only sales available by cutoff."""

    required = {*SERIES_KEYS, "date", "sales"}
    missing = required.difference(sales.columns)

    if missing:
        raise ValueError(f"Missing sales columns: {sorted(missing)}")

    if sales.empty:
        raise ValueError("Sales input must not be empty.")

    cutoff = pd.Timestamp(as_of_date)

    if pd.isna(cutoff):
        raise ValueError("as_of_date must be a valid date.")

    source = sales.loc[:, [*SERIES_KEYS, "date", "sales"]].copy()
    source["date"] = pd.to_datetime(source["date"], errors="raise")

    if source["date"].isna().any():
        raise ValueError("Sales dates cannot be missing.")

    # Exclude future observations before building historical features.
    # This is essential when the source contains data beyond the origin.
    source = source.loc[source["date"] <= cutoff].copy()

    if source.empty:
        raise ValueError("No sales observations at or before as_of_date.")

    # Duplicate business keys would make historical values ambiguous.
    if source.duplicated([*SERIES_KEYS, "date"]).any():
        raise ValueError("Duplicate store-family-date keys in sales input.")

    # Reproduce the data types and ordering of the validated notebook.
    source["store_nbr"] = source["store_nbr"].astype("int16")
    source["family"] = source["family"].astype("category")
    source["sales"] = source["sales"].astype("float32")

    source = source.sort_values(
        [*SERIES_KEYS, "date"]
    ).reset_index(drop=True)

    calendar = pd.date_range(
        source["date"].min(),
        cutoff,
        freq="D",
    )

    # Match V1: build the Cartesian product of stores, families and dates.
    # This can include combinations with no historical sales.
    full_index = pd.MultiIndex.from_product(
        [
            np.sort(source["store_nbr"].unique()),
            source["family"].cat.categories.tolist(),
            calendar,
        ],
        names=[*SERIES_KEYS, "date"],
    )

    # Missing calendar observations remain NaN, not zero.
    regularized = (
        source
        .set_index([*SERIES_KEYS, "date"])
        .reindex(full_index)
        .reset_index()
    )

    regularized["store_nbr"] = (
        regularized["store_nbr"].astype("int16")
    )
    regularized["family"] = (
        regularized["family"].astype("category")
    )
    regularized["sales"] = (
        regularized["sales"].astype("float32")
    )

    # An observed zero is valid data; a missing value is not observed.
    regularized["sales_observed"] = regularized["sales"].notna()

    return regularized


def _grouped_rolling(
    frame: pd.DataFrame,
    value_column: str,
    window: int,
    aggregation: str,
    min_periods: int = 1,
) -> pd.Series:
    """Compute rolling statistics independently for each demand series."""

    rolling = (
        frame.groupby(
            SERIES_KEYS, observed=True, sort=False
        )[value_column]
        .rolling(window=window, min_periods=min_periods)
    )

    if aggregation == "mean":
        values = rolling.mean()
    elif aggregation == "count":
        values = rolling.count()
    elif aggregation == "std":
        # ddof=0 matches the frozen V1 population-standard-deviation rule.
        values = rolling.std(ddof=0)
    else:
        raise ValueError(
            f"Unsupported rolling aggregation: {aggregation}"
        )

    # Restore the original row index after the grouped rolling operation.
    return (
        values
        .reset_index(level=SERIES_KEYS, drop=True)
        .sort_index()
    )


def build_origin_features_v1(
    regularized_sales: pd.DataFrame,
) -> pd.DataFrame:
    """Reproduce the historical demand features validated in Notebook 05."""

    required = {*SERIES_KEYS, "date", "sales", "sales_observed"}
    missing = required.difference(regularized_sales.columns)

    if missing:
        raise ValueError(
            f"Missing regularized columns: {sorted(missing)}"
        )

    if regularized_sales.empty:
        raise ValueError("Regularized sales must not be empty.")

    frame = (
        regularized_sales
        .sort_values([*SERIES_KEYS, "date"])
        .reset_index(drop=True)
        .copy()
    )

    if frame.duplicated([*SERIES_KEYS, "date"]).any():
        raise ValueError(
            "Duplicate store-family-date keys in regularized sales."
        )

    # Enforce calendar continuity before using row-based shifts.
    dates = (
        frame.groupby(SERIES_KEYS, observed=True, sort=False)["date"]
        .diff()
        .dropna()
    )

    if not dates.eq(pd.Timedelta(days=1)).all():
        raise ValueError(
            "Sales must have one row per calendar day per series."
        )

    # Origin T is included because forecasts are issued after daily sales.
    frame["sales_at_origin"] = frame["sales"].astype("float32")

    groups = frame.groupby(
        SERIES_KEYS, observed=True, sort=False
    )

    # Lag k now means exactly k calendar days before T.
    for lag in LAG_DAYS_V1:
        frame[f"sales_lag_{lag}d"] = (
            groups["sales"].shift(lag).astype("float32")
        )

    # Rolling statistics include T and ignore missing observations.
    for window in (7, 14, 28):
        frame[f"rolling_mean_{window}d"] = (
            _grouped_rolling(frame, "sales", window, "mean")
            .astype("float32")
        )

    frame["rolling_std_28d"] = (
        _grouped_rolling(
            frame, "sales", 28, "std", min_periods=2
        ).astype("float32")
    )

    # Compare short-term demand with the 28-day demand level.
    # Avoid unstable ratios when the denominator is effectively zero.
    rolling_7 = frame["rolling_mean_7d"]
    rolling_28 = frame["rolling_mean_28d"]

    ratio = pd.Series(
        np.nan, index=frame.index, dtype="float32"
    )

    valid = (
        rolling_7.notna()
        & rolling_28.notna()
        & rolling_28.abs().gt(1e-8)
    )

    ratio.loc[valid] = (
        rolling_7.loc[valid] / rolling_28.loc[valid]
    ).astype("float32")

    frame["recent_level_ratio"] = ratio

    # Keep missing days out of zero/positive demand proportions.
    frame["_zero_indicator"] = np.where(
        frame["sales"].isna(),
        np.nan,
        (frame["sales"] == 0).astype(float),
    )

    frame["_positive_indicator"] = np.where(
        frame["sales"].isna(),
        np.nan,
        (frame["sales"] > 0).astype(float),
    )

    frame["_positive_sales"] = (
        frame["sales"].where(frame["sales"] > 0)
    )

    # Count available observations within the last 28 calendar days.
    frame["observed_days_28d"] = (
        _grouped_rolling(frame, "sales", 28, "count")
        .astype("float32")
    )

    # Describe intermittent demand: zero days, positive days,
    # and average demand conditional on positive sales.
    indicators = (
        ("zero_share_28d", "_zero_indicator"),
        ("positive_rate_28d", "_positive_indicator"),
        ("positive_sales_mean_28d", "_positive_sales"),
    )

    for output, indicator in indicators:
        frame[output] = (
            _grouped_rolling(frame, indicator, 28, "mean")
            .astype("float32")
        )

    # Track the last date with strictly positive sales for each series.
    frame["_last_positive_date"] = (
        frame["date"].where(frame["sales"] > 0)
    )

    frame["_last_positive_date"] = (
        frame.groupby(
            SERIES_KEYS, observed=True, sort=False
        )["_last_positive_date"]
        .ffill()
    )

    frame["days_since_last_positive"] = (
        (frame["date"] - frame["_last_positive_date"])
        .dt.days
        .astype("float32")
    )

    # Preserve the historical count convention from Notebook 05.
    frame["history_days_available"] = (
        frame.groupby(
            SERIES_KEYS, observed=True, sort=False
        )
        .cumcount()
        .astype("int16")
    )

    # Exclude temporary indicators and raw sales from the final artifact.
    return (
        frame
        .rename(columns={"date": "forecast_origin"})
        .loc[:, list(ORIGIN_FEATURE_COLUMNS_V1)]
        .copy()
    )
