# Production-Ready Retail Demand Forecasting Platform on AWS

End-to-end retail demand forecasting project built around the Corporación Favorita — Store Sales: Time Series Forecasting dataset.

## Objective

Build a production-oriented multi-series demand forecasting platform covering:

- data ingestion and transformation;
- temporal forecasting and backtesting;
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

**Week 1 — Data, statistics and forecasting foundations**

Current focus:

- project repository setup;
- AWS cost controls;
- S3 data ingestion;
- schema and business-grain exploration;
- temporal forecasting baseline and backtesting.

## Repository Structure

```text
retail-demand-forecasting-platform-aws/
├── agent/
├── data/
├── dbt/
├── docs/
├── infrastructure/
├── pipelines/
├── src/
├── tests/
├── Makefile
├── pyproject.toml
└── README.md
```

## Design Principles

- Git contains code, configuration and reproducible transformations — not raw datasets or credentials.
- S3 is the source of truth for project data and generated ML artifacts.
- Forecast evaluation uses temporal backtesting rather than random train/test splits.
- The project prioritizes production-oriented design and depth over unnecessary architectural complexity.

## Roadmap

The project is developed incrementally over six weeks, from data foundations and forecasting baselines to global forecasting, promotion analysis, MLOps, monitoring and agentic AI.
