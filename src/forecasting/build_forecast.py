"""Build the complete V1 forecasting input frame without Jupyter."""

import numpy as np
import pandas as pd

from features.horizon import expand_forecast_origin_v1
from features.origin import (
    regularize_sales_v1,
    build_origin_features_v1,
)
from features.target_aligned import build_target_aligned_features_v1


def build_forecast_frame_v1(
    sales: pd.DataFrame,
    forecast_origin,
) -> pd.DataFrame:
    """Generate all V1 features for one 14-day batch forecast."""

    cutoff = pd.Timestamp(forecast_origin)

    # Construct a daily historical dataset using only sales known at T.
    regular = regularize_sales_v1(sales, cutoff)

    # Compute demand lags and rolling statistics for each series.
    origin = build_origin_features_v1(regular)

    # Extract the historical state available at the forecast origin.
    state = origin.loc[
        origin["forecast_origin"].eq(cutoff)
    ].copy()

    if state.empty:
        raise ValueError(
            "No origin features found at forecast origin."
        )

    # Create 14 future rows for every store-family series.
    expanded = expand_forecast_origin_v1(state)

    # Attach seasonal analogues aligned with each future target date.
    seasonal = build_target_aligned_features_v1(
        expanded[["store_nbr", "family", "target_date"]],
        regular,
        origin,
        cutoff,
    )

    expanded = expanded.merge(
        seasonal,
        on=["store_nbr", "family", "target_date"],
        how="left",
        validate="one_to_one",
        sort=False,
    )

    # Compare the current 28-day demand with its historical
    # seasonal level, preserving the frozen V1 definition.
    level = expanded["historical_seasonal_level"]

    expanded["current_vs_seasonal_ratio"] = np.where(
        level.notna() & (level.abs() > 1e-8),
        expanded["rolling_mean_28d"] / level,
        np.nan,
    ).astype("float32")

    # The future target is never needed to generate predictions.
    if "target_sales" in expanded:
        raise AssertionError(
            "Future outcome leaked into forecast features."
        )

    return expanded.sort_values(
        ["store_nbr", "family", "horizon"]
    ).reset_index(drop=True)
