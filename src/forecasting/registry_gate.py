"""Strict reproducibility audit for the frozen V1 final holdout.

This is an artifact-integrity/reproduction gate, NOT a model-selection
criterion calibrated from the untouched final holdout. Model selection
was frozen in notebook 05 before this final evaluation was opened.
"""

from __future__ import annotations

import math


def audit_frozen_champion(report: dict, policy: dict) -> dict:
    """Validate the exact out-of-time evaluation contract and return audit facts.

    Raises ValueError on a failed audit. Does not silently repair or impute data.
    """
    if report.get("status") != "PASSED":
        raise ValueError("Cloud evaluation report is not PASSED")
    if report.get("validation_type") != "frozen_v1_final_temporal_holdout":
        raise ValueError("Unexpected evaluation validation_type")

    for field, expected in (
        ("forecast_origin", policy["forecast_origin"]),
        ("holdout_start", policy["holdout_start"]),
        ("holdout_end", policy["holdout_end"]),
    ):
        if report.get(field) != expected:
            raise ValueError(f"Unexpected {field}: {report.get(field)} != {expected}")

    metrics = report.get("portfolio", {})
    if metrics.get("count") != policy["expected_rows"]:
        raise ValueError("Cloud report row count does not match frozen holdout")
    if report.get("series_count") != policy["expected_series"]:
        raise ValueError("Cloud report series count does not match 1782")
    if report.get("input_sha256", {}).get("forecasts") != policy["expected_forecasts_sha256"]:
        raise ValueError("Cloud evaluation did not use the frozen V1 forecast file")

    wape = metrics.get("wape")
    bias = metrics.get("normalized_bias")
    mae = metrics.get("mae")
    rmse = metrics.get("rmse")
    if not all(isinstance(v, (float, int)) and math.isfinite(v) for v in (wape, bias, mae, rmse)):
        raise ValueError("Non-finite or missing cloud evaluation metric")
    if wape < 0 or mae < 0 or rmse < 0:
        raise ValueError("Negative error metric")

    reference = policy["reference_wape"]
    tolerance = policy["wape_reproduction_tolerance"]
    difference = abs(wape - reference)
    if difference > tolerance:
        raise ValueError(
            f"Cloud WAPE does not reproduce frozen notebook 05: "
            f"{wape:.10f} vs {reference:.10f}, difference={difference:.10g}"
        )
    reported_reference = report.get("reference_wape")
    if (not isinstance(reported_reference, (float, int))
            or not math.isfinite(reported_reference)
            or abs(reported_reference - reference) > 1e-12):
        raise ValueError("Cloud evaluation used an unexpected reference WAPE")

    return {
        "status": "PASSED",
        "gate_type": "frozen_model_reproduction_audit_not_model_selection",
        "wape": float(wape),
        "normalized_bias": float(bias),
        "reference_wape": float(reference),
        "absolute_wape_difference": float(difference),
        "evaluated_rows": int(metrics["count"]),
        "evaluated_series": int(report["series_count"]),
    }
