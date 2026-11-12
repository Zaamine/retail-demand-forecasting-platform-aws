"""Expand a V1 forecast origin into its 14 prediction horizons.

The function prepares target dates and calendar features without
using future sales observations. Historical demand features are
computed separately from data available at the forecast origin.
"""

import numpy as np
import pandas as pd


# Frozen forecasting configuration for the validated V1 model.
FORECAST_HORIZON_V1 = 14

# These columns uniquely identify a series at a forecast origin.
ORIGIN_KEYS = ("store_nbr", "family", "forecast_origin")


def expand_forecast_origin_v1(origin_state: pd.DataFrame) -> pd.DataFrame:
    """Expand each store-family origin into 14 future prediction rows."""

    # Validate the inputs before generating the forecast grid.
    # Missing keys or duplicate series could create incorrect predictions.
    missing = [
        column for column in ORIGIN_KEYS
        if column not in origin_state.columns
    ]

    if missing:
        raise ValueError(f"Missing origin keys: {missing}")

    if origin_state.empty:
        raise ValueError("Origin state must not be empty.")

    # Future observed sales must never enter the inference feature table.
    # This prevents an obvious form of target leakage.
    if "target_sales" in origin_state.columns:
        raise ValueError(
            "Future target labels are not allowed in origin features."
        )

    if origin_state.loc[:, list(ORIGIN_KEYS)].duplicated().any():
        raise ValueError("Duplicate store-family-origin rows.")

    # A batch represents a single forecasting run at one origin date.
    # All store-family series must therefore share the same origin.
    origin_dates = pd.to_datetime(
        origin_state["forecast_origin"],
        errors="raise",
    )

    if origin_dates.isna().any() or origin_dates.nunique() != 1:
        raise ValueError("Expected exactly one valid forecast origin.")

    # Repeat each historical state 14 times: one row per future horizon.
    # Historical features remain identical across these 14 rows.
    expanded = origin_state.loc[
        origin_state.index.repeat(FORECAST_HORIZON_V1)
    ].copy().reset_index(drop=True)

    expanded["forecast_origin"] = pd.Timestamp(origin_dates.iloc[0])

    # Assign horizons 1 through 14 independently to each store-family.
    expanded["horizon"] = np.tile(
        np.arange(1, FORECAST_HORIZON_V1 + 1, dtype=np.int8),
        len(origin_state),
    )

    # Convert relative horizons into actual target dates.
    expanded["target_date"] = (
        expanded["forecast_origin"]
        + pd.to_timedelta(expanded["horizon"], unit="D")
    )

    # Future calendar features are safe: their values are already known
    # at prediction time, unlike future sales or transactions.
    expanded["target_day_of_week"] = (
        expanded["target_date"].dt.dayofweek.astype("int8")
    )

    expanded["target_month"] = (
        expanded["target_date"].dt.month.astype("int8")
    )

    # pandas uses Monday=0 and Sunday=6.
    expanded["target_is_weekend"] = (
        expanded["target_day_of_week"].isin([5, 6]).astype("int8")
    )

    return expanded
