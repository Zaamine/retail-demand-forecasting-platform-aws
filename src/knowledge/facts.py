"""Validated structured facts for the Retail Planner."""

import csv
import json
import math
from pathlib import Path


CONTRACT_PATH = (
    "artifacts/global_xgboost/"
    "point_forecast_model_contract_v1.json"
)
EVALUATION_PATH = (
    "artifacts/aws/sagemaker_registry/champion_v1/"
    "evaluation_report_cloud.json"
)
HISTORICAL_PATH = (
    "docs/assets/global_forecasting/"
    "final_holdout_historical_context.csv"
)


def get_validated_forecast_facts(topic, project_root):
    """Read and validate version-specific forecast facts."""
    root = Path(project_root)

    with (root / CONTRACT_PATH).open(encoding="utf-8") as file:
        contract = json.load(file)

    model_version = contract["artifact"]["model_version"]

    if topic == "model":
        return {
            "champion_id": contract["champion_id"],
            "model_version": model_version,
            "training_policy": contract["training_policy"],
            "forecast_horizon_days": contract["forecast_horizon_days"],
            "feature_count": len(contract["model_features"]),
            "sources": [CONTRACT_PATH],
        }

    if topic != "accuracy":
        raise ValueError(f"Unsupported fact topic: {topic}")

    with (root / EVALUATION_PATH).open(encoding="utf-8") as file:
        evaluation = json.load(file)

    with (root / HISTORICAL_PATH).open(
        encoding="utf-8", newline=""
    ) as file:
        historical = {
            row["metric"]: float(row["value"])
            for row in csv.DictReader(file)
        }

    contract_wape = float(
        contract["final_holdout_metrics"]["wape"]
    )
    cloud_wape = float(evaluation["portfolio"]["wape"])

    if evaluation["status"] != "PASSED":
        raise ValueError("Cloud evaluation has not passed.")

    if not math.isclose(
        contract_wape, cloud_wape, rel_tol=0, abs_tol=1e-8
    ):
        raise ValueError("Contract and cloud WAPE disagree.")

    if not math.isclose(
        contract_wape,
        historical["Final untouched holdout WAPE"],
        rel_tol=0,
        abs_tol=1e-8,
    ):
        raise ValueError("Historical and contract WAPE disagree.")

    if (
        contract["final_holdout_start"] != evaluation["holdout_start"]
        or contract["final_holdout_end"] != evaluation["holdout_end"]
    ):
        raise ValueError("Evaluation periods disagree.")

    return {
        "model_version": model_version,
        "historical_development_windows": int(
            historical["Historical deployment count"]
        ),
        "historical_mean_fold_wape": historical[
            "Historical mean fold WAPE"
        ],
        "final_holdout_wape": contract_wape,
        "final_holdout_normalized_bias": float(
            contract["final_holdout_metrics"]["normalized_bias"]
        ),
        "final_holdout_start": contract["final_holdout_start"],
        "final_holdout_end": contract["final_holdout_end"],
        "scope_note": (
            "Historical development results and the final holdout "
            "represent different evaluation scopes."
        ),
        "sources": [
            CONTRACT_PATH,
            EVALUATION_PATH,
            HISTORICAL_PATH,
        ],
    }
