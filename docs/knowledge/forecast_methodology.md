# Retail Forecasting — Planning Methodology

**Knowledge ID:** `forecast.planning_methodology.v1`
**Audience:** Retail planners and forecasting operations
**Status:** Draft — pending source review
**Model scope:** `point_forecast_v1`

## What the Forecast Represents

The forecasting system estimates daily sales over the next 14 days for each store and product family.

These estimates help planners anticipate upcoming sales activity.

They are not inventory recommendations, confirmed customer orders or guaranteed sales volumes.

## How to Use Forecasts in Planning

Forecasts can support decisions about where future sales activity may be higher or lower.

However, a replenishment decision also requires information such as:

- Current inventory and incoming stock.
- Supplier lead times and delivery constraints.
- Product availability and operational requirements.

The current forecasting model does not account for all these factors.

Therefore, forecasts should support planning decisions rather than replace operational judgment.

## How Reliable Are the Forecasts?

Historical evaluation provides evidence that the model can produce useful sales estimates.

However, forecast accuracy varies across time periods, stores and product families.

A strong overall result does not guarantee accurate predictions for every individual series.

The Model Card contains the validated evaluation results and their scope.

## Understanding Forecasting Errors

Forecasts may be higher or lower than actual sales.

Repeated overestimation can contribute to unnecessary inventory if it influences replenishment decisions.

Repeated underestimation can increase the risk of insufficient stock.

These are potential planning consequences, not observed inventory outcomes in this dataset.

## Data Availability Limitations

The dataset contains observed sales but does not provide complete inventory availability or confirmed stockout information.

Consequently, the system cannot always distinguish low customer demand from sales constrained by product availability.

Forecasts estimate observed sales patterns rather than unconstrained customer demand.

## Promotions

The selected forecasting model does not explicitly use planned discounts or other promotion information as input features.

Separate promotion-aware experiments were conducted, but the promotion-aware extension was not retained as the forecasting champion.

The current forecasts should not be interpreted as validated estimates of incremental promotional impact.

## Practical Interpretation

A forecast should answer a practical planning question first.

For example, when a planner asks whether forecasts are reliable, the initial explanation should describe their usefulness and variability in plain business language.

Technical metrics, model architecture and statistical details can be provided when requested.

Important limitations should be mentioned when they materially affect a planning decision, without overwhelming the user with unnecessary details.

---

## Sources and Scope

- `README.md`
- `artifacts/global_xgboost/point_forecast_model_contract_v1.json`
- `docs/knowledge/model_card.md`
- `docs/assets/global_forecasting/final_holdout_historical_context.csv`
- `docs/assets/global_forecasting/final_holdout_model_comparison.csv`

This document describes how to interpret the current forecasting system. It does not establish that automated replenishment decisions or production inventory integrations are implemented.
