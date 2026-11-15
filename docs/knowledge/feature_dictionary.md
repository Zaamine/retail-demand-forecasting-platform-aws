# Retail Forecasting — Feature Dictionary

**Knowledge ID:** `forecast.feature_dictionary.v1`
**Model version:** `point_forecast_v1`
**Status:** Draft — pending review
**Audience:** ML engineering and forecasting operations

## Feature Contract

The frozen forecasting model uses exactly 31 input variables.

All historical sales-based features must be constructed using data available no later than forecast origin T. Calendar properties of future prediction dates are known in advance.

Missing historical observations are not automatically equivalent to zero sales.

## Frozen Model Features

| Feature                      | Group                  | Definition                                                                                                                                       | Source                            |
|:-----------------------------|:-----------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------|
| store_nbr                    | Identity               | Store identifier.                                                                                                                                | src/features/origin.py            |
| family                       | Identity               | Product-family identifier.                                                                                                                       | src/features/origin.py            |
| horizon                      | Forecast calendar      | Number of days after the forecast origin, from 1 to 14.                                                                                          | src/features/horizon.py           |
| sales_at_origin              | Recent sales           | Observed sales on the forecast-origin date T.                                                                                                    | src/features/origin.py            |
| sales_lag_1d                 | Recent sales           | Observed sales 1 calendar days before the forecast origin T.                                                                                     | src/features/origin.py            |
| sales_lag_2d                 | Recent sales           | Observed sales 2 calendar days before the forecast origin T.                                                                                     | src/features/origin.py            |
| sales_lag_3d                 | Recent sales           | Observed sales 3 calendar days before the forecast origin T.                                                                                     | src/features/origin.py            |
| sales_lag_7d                 | Recent sales           | Observed sales 7 calendar days before the forecast origin T.                                                                                     | src/features/origin.py            |
| sales_lag_14d                | Recent sales           | Observed sales 14 calendar days before the forecast origin T.                                                                                    | src/features/origin.py            |
| sales_lag_21d                | Recent sales           | Observed sales 21 calendar days before the forecast origin T.                                                                                    | src/features/origin.py            |
| sales_lag_28d                | Recent sales           | Observed sales 28 calendar days before the forecast origin T.                                                                                    | src/features/origin.py            |
| sales_lag_35d                | Recent sales           | Observed sales 35 calendar days before the forecast origin T.                                                                                    | src/features/origin.py            |
| sales_lag_364d               | Recent sales           | Observed sales 364 calendar days before the forecast origin T.                                                                                   | src/features/origin.py            |
| rolling_mean_7d              | Recent sales           | Mean observed daily sales over the last 7 calendar days, including T; missing observations are ignored.                                          | src/features/origin.py            |
| rolling_mean_14d             | Recent sales           | Mean observed daily sales over the last 14 calendar days, including T; missing observations are ignored.                                         | src/features/origin.py            |
| rolling_mean_28d             | Recent sales           | Mean observed daily sales over the last 28 calendar days, including T; missing observations are ignored.                                         | src/features/origin.py            |
| recent_level_ratio           | Recent sales           | Ratio of the 7-day to the 28-day rolling mean; undefined when the 28-day mean is unavailable or near zero.                                       | src/features/origin.py            |
| rolling_std_28d              | Demand variability     | Population standard deviation of observed sales over the last 28 calendar days, including T; at least two observations required.                 | src/features/origin.py            |
| positive_rate_28d            | Demand intermittency   | Fraction of observed days with strictly positive sales in the last 28 calendar days.                                                             | src/features/origin.py            |
| positive_sales_mean_28d      | Demand intermittency   | Average sales conditional on strictly positive sales within the last 28 calendar days.                                                           | src/features/origin.py            |
| days_since_last_positive     | Demand intermittency   | Calendar days since the most recent strictly positive observed sales, measured at T.                                                             | src/features/origin.py            |
| target_day_of_week           | Forecast calendar      | Weekday of the predicted date; Monday is 0 and Sunday is 6.                                                                                      | src/features/horizon.py           |
| target_month                 | Forecast calendar      | Calendar month of the predicted date, from 1 to 12.                                                                                              | src/features/horizon.py           |
| target_is_weekend            | Forecast calendar      | Indicator equal to 1 when the predicted date is Saturday or Sunday.                                                                              | src/features/horizon.py           |
| target_sales_lag_364d        | Historical seasonality | Observed sales at the target date shifted backward by 364 calendar days.                                                                         | src/features/target_aligned.py    |
| target_sales_lag_1y_calendar | Historical seasonality | Observed sales at the target date shifted backward by one calendar year.                                                                         | src/features/target_aligned.py    |
| target_sales_lag_728d        | Historical seasonality | Observed sales at the target date shifted backward by 728 calendar days.                                                                         | src/features/target_aligned.py    |
| historical_seasonal_level    | Historical seasonality | Mean of available historical 28-day sales levels at up to four annual seasonal positions; each usable window requires at least 27 observed days. | src/features/target_aligned.py    |
| seasonal_level_trend_ratio   | Historical seasonality | Linear trend across available annual seasonal levels, divided by their mean; requires at least two usable years and a non-negligible mean.       | src/features/target_aligned.py    |
| seasonal_years_available     | Historical seasonality | Number of usable historical annual seasonal levels, between 0 and 4.                                                                             | src/features/target_aligned.py    |
| current_vs_seasonal_ratio    | Historical seasonality | Current 28-day rolling sales mean divided by the historical seasonal level, when the denominator is available and nonzero.                       | src/forecasting/build_forecast.py |

## Important Constraints

- Forecasts are generated for the 14 days following origin T.
- The feature order is defined by the frozen model contract.
- Future realized sales must never be used as predictors.
- Promotion information is not part of the selected V1 predictor schema.
- Feature availability and categorical compatibility must be validated during inference.
- This dictionary describes the V1 model, not all experimental features explored during research.

## Provenance

Model contract:

- `artifacts/global_xgboost/point_forecast_model_contract_v1.json`

Implementation:

- `src/features/origin.py`
- `src/features/horizon.py`
- `src/features/target_aligned.py`
- `src/forecasting/build_forecast.py`
- `src/forecasting/model_inputs.py`

Source Git revision: `d2d5a9c207735f541e06a42dd65457085e1d0c14`

Feature descriptions are derived from the frozen implementation and require review if the implementation or model version changes.
