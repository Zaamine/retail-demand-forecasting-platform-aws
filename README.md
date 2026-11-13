# Production-Ready Retail Demand Forecasting Platform on AWS

End-to-end retail demand forecasting project built around the Corporación Favorita — Store Sales: Time Series Forecasting dataset.

The project is designed as a decision-oriented Data Science product rather than a standalone forecasting exercise: forecasts support short-term retail planning, while a separate causal layer estimates the incremental impact of promotions.

## Business Problem

The project simulates a short-term retail demand-planning use case.

At the end of each day `t`, the system produces daily sales forecasts for each `store × product family` series over the next 14 days (`t+1` to `t+14`).

These forecasts are intended to support short-term inventory and replenishment planning by helping retailers:

- anticipate demand peaks and slowdowns;
- reduce stock-out risk;
- limit unnecessary overstock;
- improve inventory allocation across stores and product families.

Because the dataset contains observed sales but no inventory or stock-out information, sales are treated as a proxy for customer demand. In a production retailer setting, inventory availability would be required to distinguish genuinely low demand from constrained sales.

### Forecasting Setup

| Component | Definition |
| --- | --- |
| Target | Daily sales |
| Granularity | `store × product family × day` |
| Forecast horizon | 14 days |
| Refresh cadence | Daily |
| Forecast origin | End of day `t`, using information available up to that point |
| Primary business use case | Short-term inventory and replenishment planning |
| Primary evaluation metric | WAPE |
| Additional diagnostics | MAE, RMSE and forecast bias |

All model features must respect **point-in-time availability** at the forecast origin to prevent data leakage.

A separate promotion-analysis layer will later estimate counterfactual promotional uplift to support promotion-planning decisions.

## Objective

Build a production-oriented multi-series demand forecasting platform covering:

- data ingestion and transformation;
- temporal forecasting and rolling backtesting;
- promotion-aware demand modeling;
- counterfactual promotion analysis;
- AWS-based ML training, batch inference and monitoring;
- automated retraining and model lifecycle management;
- a tool-using forecast investigation agent.

## Dataset

The project uses the Corporación Favorita — Store Sales: Time Series Forecasting dataset.

Forecasting grain:

`date × store_nbr × family`

Approximate scale:

- 54 stores
- 33 product families
- 1,782 related daily demand series
- multi-year sales history
- promotion information
- store metadata
- holidays and events
- oil prices
- store transactions

## Architecture

The target architecture progressively combines:

`S3 → Glue / Athena / dbt → SageMaker → Model Registry → Batch Forecasting → Monitoring`

with a promotion/counterfactual analysis layer and, later, a Bedrock-based forecast investigation agent.

## Project Status

**AWS MLOps — Week 4B validated (November 2026).**

- Global XGBoost V1: 1,782 store-family series, 14-day forecasts,
  final holdout WAPE 18.77% versus 23.21% seasonal naive
  (19.12% relative improvement).
- SageMaker Processing: cloud artifact validation, batch inference,
  and reproducibility checks completed.
- SageMaker Model Registry: V1 champion and V2 challenger registered;
  V2 remains PendingManualApproval.
- Step Functions: V1/V2 forecasting workflows and CT-6C readiness
  orchestration demonstrated on AWS.
- Continuous training CT-2 through CT-5: guarded manual execution,
  with no automatic model promotion.
- CT-6A/6B/6C: checkpoint-aware readiness, DynamoDB concurrency lock,
  and successful cloud execution of the SKIPPED branch.
- CloudWatch monitoring, alerts, and GitHub Actions CI/CD integrated.

**Current limitation:** CT-6C deliberately blocks READY because
automatic CT-2 through CT-5 orchestration is not implemented.
EventBridge scheduling remains disabled.

**Next:** Forecast Investigation Agent, GenAI/LLMOps,
agent evaluation, and AgentOps.

## Repository Structure

```text
retail-demand-forecasting-platform-aws/
├── agent/
├── data/
├── dbt/
├── docs/
├── infrastructure/
├── notebooks/
├── pipelines/
├── src/
├── tests/
├── Makefile
├── pyproject.toml
└── README.md
```

## Design Principles

- Start from the business decision and forecasting constraints before selecting a model.
- Git contains code, configuration and reproducible transformations — not raw datasets or credentials.
- S3 is the source of truth for project data and generated ML artifacts.
- Forecast evaluation uses temporal backtesting rather than random train/test splits.
- Features must be available at forecast time to preserve point-in-time correctness.
- Forecast accuracy is evaluated alongside systematic forecast bias.
- The project prioritizes production-oriented design and depth over unnecessary architectural complexity.

## Roadmap

The project is developed incrementally over six weeks, from data foundations and forecasting baselines to global forecasting, promotion analysis, MLOps, monitoring and agentic AI.
