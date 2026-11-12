"""Target-aligned historical predictors for the frozen V1 model.

All historical lookups must use observations available by origin T.
Future target dates are allowed, but future sales labels are not.
"""

import numpy as np
import pandas as pd


SERIES_KEYS = ["store_nbr", "family"]

# Frozen seasonal configuration from Notebook 05.
ANNUAL_CYCLE_DAYS = 364
SEASONAL_CYCLES = (1, 2, 3, 4)
MIN_SEASONAL_WINDOW_OBS = 27

TARGET_FEATURES_V1 = (
    "target_sales_lag_364d",
    "target_sales_lag_1y_calendar",
    "target_sales_lag_728d",
    "historical_seasonal_level",
    "seasonal_level_trend_ratio",
    "seasonal_years_available",
)


def _historical_join(frame, history, dates, value_columns):
    """Attach historical values using series keys and lookup dates."""

    working = frame.copy()
    working["_lookup_date"] = dates

    source = history.rename(
        columns={"date": "_lookup_date"}
    )

    # many_to_one ensures that historical business keys are unique.
    working = working.merge(
        source[
            [*SERIES_KEYS, "_lookup_date", *value_columns]
        ],
        on=[*SERIES_KEYS, "_lookup_date"],
        how="left",
        validate="many_to_one",
        sort=False,
    )

    return working.drop(columns="_lookup_date")


def _seasonal_trend(levels):
    """Estimate the linear trend across annual seasonal levels."""

    values = levels.to_numpy(dtype=np.float32)

    # Yearly positions are ordered from oldest to most recent.
    x = np.arange(values.shape[1], dtype=np.float32)

    observed = np.isfinite(values)
    n = observed.sum(axis=1).astype(np.float32)

    # Missing seasonal levels must not contribute to the regression.
    safe = np.where(observed, values, 0.0)

    sum_x = (observed * x).sum(axis=1)
    sum_y = safe.sum(axis=1)
    sum_xy = (safe * x).sum(axis=1)
    sum_x2 = (observed * x**2).sum(axis=1)

    denominator = n * sum_x2 - sum_x**2

    trend = np.full(
        len(values), np.nan, dtype=np.float32
    )

    # A trend requires at least two usable historical years.
    valid = (n >= 2) & (denominator > 0)

    trend[valid] = (
        (
            n[valid] * sum_xy[valid]
            - sum_x[valid] * sum_y[valid]
        )
        / denominator[valid]
    )

    return trend


def build_target_aligned_features_v1(
    target_keys: pd.DataFrame,
    regularized_sales: pd.DataFrame,
    origin_features: pd.DataFrame,
    forecast_origin: str | pd.Timestamp,
) -> pd.DataFrame:
    """Build label-free annual and seasonal predictors for future dates."""

    cutoff = pd.Timestamp(forecast_origin)
    keys = [*SERIES_KEYS, "target_date"]

    if not set(keys).issubset(target_keys):
        raise ValueError("Target keys are missing required columns.")

    if target_keys.duplicated(keys).any():
        raise ValueError("Duplicate store-family-target_date keys.")

    features = (
        target_keys[keys]
        .copy()
        .reset_index(drop=True)
    )

    features["target_date"] = pd.to_datetime(
        features["target_date"]
    )

    # The frozen V1 only predicts the 14 days following origin T.
    if features.empty or not features["target_date"].between(
        cutoff + pd.Timedelta(days=1),
        cutoff + pd.Timedelta(days=14),
    ).all():
        raise ValueError(
            "Target dates must lie strictly in the 14-day future horizon."
        )

    # Fail before computation if any historical source contains future data.
    if pd.to_datetime(regularized_sales["date"]).max() > cutoff:
        raise ValueError(
            "Historical sales contain post-origin observations."
        )

    if pd.to_datetime(
        origin_features["forecast_origin"]
    ).max() > cutoff:
        raise ValueError(
            "Origin features contain post-origin observations."
        )

    # Prepare the two historical lookup tables.
    sales_lookup = regularized_sales[
        [*SERIES_KEYS, "date", "sales"]
    ].copy()
    sales_lookup["date"] = pd.to_datetime(
        sales_lookup["date"]
    )

    origin_lookup = origin_features[
        [
            *SERIES_KEYS,
            "forecast_origin",
            "rolling_mean_28d",
            "observed_days_28d",
        ]
    ].rename(
        columns={"forecast_origin": "date"}
    ).copy()

    origin_lookup["date"] = pd.to_datetime(
        origin_lookup["date"]
    )

    # Fixed annual lags: 364 and 728 days preserve weekday alignment.
    for days in (364, 728):
        name = f"target_sales_lag_{days}d"

        features = _historical_join(
            features,
            sales_lookup.rename(
                columns={"sales": name}
            ),
            features["target_date"] - pd.Timedelta(days=days),
            [name],
        )

        features[name] = features[name].astype("float32")

    # Calendar-year analogue, distinct from a fixed 364-day offset.
    name = "target_sales_lag_1y_calendar"

    features = _historical_join(
        features,
        sales_lookup.rename(columns={"sales": name}),
        features["target_date"] - pd.DateOffset(years=1),
        [name],
    )

    features[name] = features[name].astype("float32")

    # Retrieve the historical 28-day demand level around each
    # of the previous four annual seasonal positions.
    for cycle in SEASONAL_CYCLES:
        level = f"_seasonal_level_{cycle}y"

        working = _historical_join(
            features,
            origin_lookup,
            features["target_date"]
            - pd.Timedelta(days=ANNUAL_CYCLE_DAYS * cycle),
            ["rolling_mean_28d", "observed_days_28d"],
        )

        # A seasonal window is usable only if at least 27 of its
        # 28 calendar days have observed demand.
        features[level] = working["rolling_mean_28d"].where(
            working["observed_days_28d"]
            >= MIN_SEASONAL_WINDOW_OBS
        ).astype("float32")

    seasonal = [
        f"_seasonal_level_{cycle}y"
        for cycle in SEASONAL_CYCLES
    ]

    # Summarize both the seasonal level and its historical coverage.
    features["seasonal_years_available"] = (
        features[seasonal]
        .notna()
        .sum(axis=1)
        .astype("int8")
    )

    features["historical_seasonal_level"] = (
        features[seasonal]
        .mean(axis=1)
        .astype("float32")
    )

    # The trend runs chronologically: 4y -> 3y -> 2y -> 1y.
    trend = _seasonal_trend(features[seasonal[::-1]])
    mean = features["historical_seasonal_level"]

    # Normalize the slope so it is comparable across sales volumes.
    features["seasonal_level_trend_ratio"] = np.where(
        mean.notna() & (mean.abs() > 1e-8),
        trend / mean,
        np.nan,
    ).astype("float32")

    # Return only the six frozen predictors and their business keys.
    return features[
        keys + list(TARGET_FEATURES_V1)
    ]
