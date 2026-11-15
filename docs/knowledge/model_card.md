# Retail Forecasting — Model Card

**Knowledge ID:** `forecast.model_card.v1`
**Model version:** `point_forecast_v1`
**Status:** Selected frozen forecasting champion
**Audience:** Retail planners and forecasting operations

## What does the system do?

It estimates daily sales for each store and product family over the next 14 days.

These forecasts provide a starting point for inventory and replenishment planning.

## How accurate are our forecasts?

Across 15 historical development tests, the average overall forecasting error was around 13% of sales.

In a particularly challenging August test period, the overall error was closer to 19%.

Forecasts are useful planning estimates, not guaranteed sales quantities.

## Can I trust every store and product forecast equally?

No. Forecast accuracy varies across stores, product families and time periods.

A good overall result does not guarantee equally accurate predictions for every individual product family.

## Does the system take promotions into account?

The currently selected forecasting model does not use promotion features.

Promotion-aware approaches were investigated separately but were not retained as the forecasting champion.

## Does low sales always mean low customer demand?

Not necessarily.

The available dataset records sales but does not provide inventory availability or confirmed stockout information.

Low sales could therefore reflect limited product availability rather than low customer demand.

## Can the forecasts determine exactly how much inventory to order?

No. They estimate future sales, but do not establish current stock levels, supplier constraints or optimal order quantities.

Retail planners must combine forecasts with the relevant operational information.

---

## Technical Evidence

| Property | Validated value |
|---|---|
| Model | Global direct multi-horizon XGBoost |
| Champion ID | `candidate_27` |
| Training policy | `last_2_years` |
| Forecast horizon | 14 days |
| Historical development windows | 15 |
| Historical mean fold WAPE | 13.2503% |
| Final holdout WAPE | 18.7725% |
| Final holdout normalized bias | +12.0925% |
| Final holdout period | 2017-08-02 to 2017-08-15 |

Historical development results and the final holdout represent different evaluation scopes.

The historical average is not a guarantee of future performance. The final holdout represents one particular historical period.

These metrics describe aggregate performance, not the accuracy of every individual store-family series.

The model is a validated historical forecasting artifact. This document does not establish that automated daily production forecasting is currently enabled.

## Provenance

Source Git revision: `d2d5a9c207735f541e06a42dd65457085e1d0c14`

- `artifacts/global_xgboost/point_forecast_model_contract_v1.json`
- `artifacts/aws/sagemaker_registry/champion_v1/evaluation_report_cloud.json`
- `docs/assets/global_forecasting/final_holdout_historical_context.csv`
- `docs/assets/global_forecasting/final_holdout_model_comparison.csv`
- `README.md`

The quantitative fields above are snapshots of the referenced model version. A future champion requires a separately reviewed Model Card.
