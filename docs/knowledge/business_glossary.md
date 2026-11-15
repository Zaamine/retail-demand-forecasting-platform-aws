# Retail Forecasting — Business Glossary

**Knowledge ID:** `forecast.business_glossary.v1`
**Audience:** Retail planners
**Status:** Draft — pending source review

## Sales forecast

An estimate of how much a store is expected to sell over a future period.

A forecast helps anticipate demand, but it is not a guaranteed sales quantity.

## Product family

A group of related products tracked together for forecasting.

Our current system forecasts at product-family level, not individual-product level.

## Forecast horizon

The period covered by a forecast.

Our system estimates daily sales for the next 14 days.

## Forecast accuracy

How closely forecasts match actual sales.

Accuracy can vary between stores, product families and periods. Good overall results do not guarantee equally reliable forecasts everywhere.

## Overforecasting

When forecasts are too high, the system predicts more sales than actually occur.

Persistent overforecasting may contribute to excess inventory if purchasing decisions rely too heavily on these predictions.

## Underforecasting

When forecasts are too low, the system predicts fewer sales than actually occur.

Persistent underforecasting may increase stockout risk if replenishment decisions do not account for unexpected demand.

## Stockout

A situation in which a product is unavailable for customers to purchase.

Observed sales may underestimate customer demand during stockouts.

## Observed sales versus customer demand

Sales measure what customers actually purchased.

Customer demand may be higher when products are unavailable. Our dataset does not provide the inventory information needed to measure this difference reliably.

## Promotion

A commercial action intended to encourage purchases, such as a discount or special offer.

The currently selected forecasting model does not use promotion features. Promotion effects were investigated separately.

## Replenishment planning

The process of deciding when and how much inventory to restock.

Sales forecasts can support replenishment planning, but inventory levels, supplier lead times and other operational constraints must also be considered.

---

## Sources and Scope

- `README.md`
- `artifacts/global_xgboost/point_forecast_model_contract_v1.json`
- `docs/knowledge/model_card.md`

Definitions describe the current forecasting use case and are not universal claims about all retail forecasting systems.
