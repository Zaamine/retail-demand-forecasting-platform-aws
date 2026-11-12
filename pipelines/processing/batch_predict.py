"""Load the frozen XGBoost V1 model and audit all 24,948 batch predictions.

The target sales are deliberately excluded from this prediction workflow.
"""
import argparse
import hashlib
import json
import platform
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
from forecasting.model_inputs import load_model_input_schema, prepare_xgboost_X
from forecasting.parity import align_forecasts

KEYS = ["store_nbr", "family", "forecast_origin", "horizon", "target_date"]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_v1_model(archive_path):
    """Read the single expected JSON file, without extracting tar paths."""
    with tarfile.open(archive_path, "r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        if len(members) != 1 or members[0].name != "xgboost_point_forecast_v1.json":
            raise ValueError("Unexpected model.tar.gz content")
        stream = archive.extractfile(members[0])
        if stream is None:
            raise ValueError("Cannot read frozen XGBoost JSON")
        model_data = bytearray(stream.read())
    model = XGBRegressor(enable_categorical=True, n_jobs=2)
    model.load_model(model_data)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("/opt/ml/processing/input")
    parser.add_argument("--features", type=Path, default=root / "features/features.parquet")
    parser.add_argument("--model", type=Path, default=root / "model/model.tar.gz")
    parser.add_argument("--reference", type=Path, default=root / "reference/final_holdout_forecast.parquet")
    parser.add_argument("--contract", type=Path, default=root / "contract/point_forecast_model_contract_v1.json")
    parser.add_argument("--categories", type=Path, default=root / "categories/categories_v1.json")
    parser.add_argument("--manifest", type=Path, default=root / "manifest/features.manifest.json")
    parser.add_argument("--output-dir", type=Path, default=Path("/opt/ml/processing/output"))
    parser.add_argument("--expected-rows", type=int, default=24948)
    parser.add_argument("--atol", type=float, default=0.0001)
    parser.add_argument("--rtol", type=float, default=0.000001)
    args = parser.parse_args()

    if xgb.__version__ != "3.4.1":
        raise RuntimeError(f"Expected XGBoost 3.4.1, found {xgb.__version__}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if sha256_file(args.features) != manifest["features_sha256"]:
        raise ValueError("Feature snapshot checksum mismatch")
    if sha256_file(args.reference) != manifest["reference_sha256"]:
        raise ValueError("Frozen reference checksum mismatch")
    schema = load_model_input_schema(args.contract, args.categories)
    frame = pd.read_parquet(args.features)
    if len(frame) != args.expected_rows or frame.duplicated(KEYS).any():
        raise ValueError("Invalid feature population")
    if any(column in frame.columns for column in ("sales", "target_sales", "actual_sales")):
        raise ValueError("Actual sales are forbidden in inference inputs")
    if not pd.to_datetime(frame["forecast_origin"]).eq(pd.Timestamp("2017-08-01")).all():
        raise ValueError("Unexpected forecast origin")

    matrix = prepare_xgboost_X(frame, schema)
    print("Python:", platform.python_version(), "XGBoost:", xgb.__version__)
    print("Inference rows:", len(matrix), "Predictors:", matrix.shape[1])
    model = load_v1_model(args.model)
    raw = model.predict(matrix).astype("float64")
    if not np.isfinite(raw).all():
        raise ValueError("Non-finite model predictions")
    output = frame[KEYS].copy()
    output["prediction_raw"] = raw
    output["prediction"] = np.clip(raw, a_min=0, a_max=None)
    compared = align_forecasts(output, pd.read_parquet(args.reference))

    raw_error = np.abs(compared["prediction_raw_aws"] - compared["prediction_raw_notebook"])
    clipped_error = np.abs(compared["prediction_aws"] - compared["prediction_notebook"])
    close_raw = np.isclose(
        compared["prediction_raw_aws"], compared["prediction_raw_notebook"],
        atol=args.atol, rtol=args.rtol,
    )
    close_clipped = np.isclose(
        compared["prediction_aws"], compared["prediction_notebook"],
        atol=args.atol, rtol=args.rtol,
    )
    failed_rows = int((~(close_raw & close_clipped)).sum())
    report = {
        "status": "PASSED" if not failed_rows else "FAILED",
        "validation_type": "frozen_v1_batch_inference_parity",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "forecast_origin": "2017-08-01",
        "forecast_rows": len(output),
        "feature_count": len(schema.features),
        "mismatched_rows": failed_rows,
        "maximum_raw_prediction_difference": float(raw_error.max()),
        "maximum_clipped_prediction_difference": float(clipped_error.max()),
        "atol": args.atol, "rtol": args.rtol,
        "model_sha256": sha256_file(args.model),
        "feature_snapshot_sha256": manifest["features_sha256"],
        "xgboost_version": xgb.__version__,
        "python_version": platform.python_version(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.sort_values(KEYS).to_parquet(
        args.output_dir / "batch_forecasts.parquet", index=False
    )
    (args.output_dir / "inference_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if failed_rows:
        raise AssertionError(f"Batch inference mismatch: {failed_rows} rows")
    print("SAGEMAKER BATCH INFERENCE: PASSED")


if __name__ == "__main__":
    main()
