# Retail Forecasting — Operational Runbook

**Knowledge ID:** `forecast.operations_runbook.v1`
**Status:** Draft — validated historical project state
**Audience:** Forecasting operations and ML engineering

## System Scope

The project includes AWS-based model evaluation, batch inference, SageMaker Model Registry and guarded continuous-training workflows.

The validated forecasting champion is `point_forecast_v1`.

This runbook documents demonstrated capabilities and known constraints. It is not evidence of an active, fully automated production service.

## 1. Investigating Forecast Quality

When forecast quality is questioned:

1. Identify the model version and forecast origin.
2. Verify the forecast dates and store-family coverage.
3. Locate the matching forecasts and realized sales.
4. Use evaluation reports for the same model and observation period.
5. Check both overall forecasting error and systematic overestimation or underestimation.
6. Avoid extrapolating portfolio-wide accuracy to an individual store-family series.

Primary references:

- `artifacts/global_xgboost/point_forecast_model_contract_v1.json`
- `artifacts/aws/sagemaker_registry/champion_v1/evaluation_report_cloud.json`

## 2. Investigating Data and Feature Issues

Check that:

- Sales data contains the required store, family, date and sales fields.
- Store-family-date keys are unique.
- Historical observations do not extend beyond the forecast origin.
- Missing historical sales are not silently treated as observed zero sales.
- The exact frozen model feature contract is respected.
- The inference input does not contain future target values.

Primary references:

- `src/features/origin.py`
- `src/features/horizon.py`
- `src/features/target_aligned.py`
- `src/forecasting/model_inputs.py`

## 3. Continuous-Training State

The recorded checkpoint has the following historical state:

- Completed cutoff: `2017-08-15`
- Completed phase: `CHALLENGER_REGISTERED_PENDING`
- Challenger approval: `PendingManualApproval`
- Challenger promoted: `False`
- New out-of-sample performance available: `False`

The successful CT cycle produced a registered challenger awaiting manual approval.

It did not automatically promote a new forecasting model.

The latest validated CT-6C readiness workflow does not automatically execute the entire next training cycle.

## 4. Safe Operating Principles

- Do not automatically approve or promote a challenger based only on successful training or registration.
- Do not claim improved forecasting accuracy without appropriate evaluation evidence.
- Treat a blocked readiness gate as a reason to investigate the missing requirements, not as an instruction to bypass safeguards.
- Use versioned evaluation evidence when communicating model performance.
- Distinguish historical project execution from current service availability.

## 5. Known Operational Limitations

The project does not establish a fully automated production retraining and deployment lifecycle.

The forecasting system does not provide live inventory data or automatic replenishment decisions.

The operational investigation agent and its production-facing interfaces are developed separately.

## Provenance

- `README.md`
- `artifacts/aws/continuous_training/last_successful_ct.json`
- `pipelines/processing/ct_workflow_readiness.py`
- `artifacts/aws/sagemaker_registry/champion_v1/registry_record.json`

Source Git revision: `d2d5a9c207735f541e06a42dd65457085e1d0c14`

This runbook describes the recorded project state, not a live AWS health check or a replacement for deployment-specific incident procedures.
