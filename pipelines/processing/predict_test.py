"""Predict unseen test dates with the unchanged XGBoost V1 champion."""
import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from batch_predict import load_v1_model, sha256_file
from build_test_features import (
    KEYS,
    ORIGIN,
    assert_key_match,
    expected_test_rows,
)
from forecasting.model_inputs import (
    load_model_input_schema,
    prepare_xgboost_X,
)

ROOT = Path("/opt/ml/processing/input")

FROZEN_SHA = (
    "003f007a2057bf26b6926a292c9fad2f8e7869"
    "ca5e150a6d1b344a3e9a646b73"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=ROOT / "features/features.parquet",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "manifest/features.manifest.json",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=ROOT / "test/test.csv",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "model/model.tar.gz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/opt/ml/processing/output"),
    )
    args = parser.parse_args()

    if xgb.__version__ != "3.4.1":
        raise ValueError(f"Unexpected XGBoost: {xgb.__version__}")

    manifest = json.loads(
        args.manifest.read_text(encoding="utf-8")
    )

    if (
        manifest["status"] != "PASSED"
        or manifest["forecast_origin"] != str(ORIGIN.date())
        or manifest["max_sales_date"] != str(ORIGIN.date())
    ):
        raise ValueError("Feature manifest origin checks failed")

    if sha256_file(args.features) != manifest["features_sha256"]:
        raise ValueError("Feature checksum mismatch")

    if sha256_file(args.test) != manifest["test_sha256"]:
        raise ValueError("Kaggle test checksum mismatch")

    if sha256_file(args.model) != FROZEN_SHA:
        raise ValueError("Model is not the frozen champion V1")

    frame = pd.read_parquet(args.features)

    if len(frame) != 24948 or frame.duplicated(KEYS).any():
        raise ValueError("Incorrect forecast population")

    if any(
        col in frame.columns
        for col in ("sales", "target_sales", "actual_sales", "onpromotion")
    ):
        raise ValueError("Forbidden feature column")

    if not pd.to_datetime(
        frame["forecast_origin"]
    ).eq(ORIGIN).all():
        raise ValueError("Unexpected forecast origin")

    expected = expected_test_rows(args.test)
    assert_key_match(frame, expected)

    schema = load_model_input_schema(
        ROOT / "contract/point_forecast_model_contract_v1.json",
        ROOT / "categories/categories_v1.json",
    )

    matrix = prepare_xgboost_X(frame, schema)
    model = load_v1_model(args.model)
    raw = model.predict(matrix).astype("float64")

    if len(raw) != 24948 or not np.isfinite(raw).all():
        raise ValueError("Invalid raw predictions")

    output = frame[KEYS].copy()
    output["prediction_raw"] = raw
    output["prediction"] = np.maximum(raw, 0)

    output["family"] = output["family"].astype("string")
    output["forecast_origin"] = pd.to_datetime(
        output["forecast_origin"]
    )
    output["target_date"] = pd.to_datetime(
        output["target_date"]
    )

    output = expected.merge(
        output,
        on=KEYS,
        how="inner",
        validate="one_to_one",
    ).sort_values("id")

    if (
        len(output) != 24948
        or output["id"].isna().any()
        or output["id"].duplicated().any()
    ):
        raise ValueError("Missing or duplicate Kaggle IDs")

    if (
        (output["prediction"] < 0).any()
        or not np.isfinite(output["prediction"]).all()
    ):
        raise ValueError("Invalid final forecasts")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    destination = args.output_dir / "forecast_predictions.parquet"
    output.to_parquet(destination, index=False)

    report = {
        "status": "PASSED",
        "validation_type": "out_of_sample_v1_unlabeled_test",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "forecast_origin": str(ORIGIN.date()),
        "first_target_date": str(
            output["target_date"].min().date()
        ),
        "last_target_date": str(
            output["target_date"].max().date()
        ),
        "forecast_rows": len(output),
        "unique_series": int(
            output[["store_nbr", "family"]]
            .drop_duplicates().shape[0]
        ),
        "horizons": sorted(
            int(h) for h in output["horizon"].unique()
        ),
        "feature_count": len(schema.features),
        "model_sha256": FROZEN_SHA,
        "features_sha256": manifest["features_sha256"],
        "test_sha256": manifest["test_sha256"],
        "forecast_sha256": sha256_file(destination),
        "prediction_min": float(output["prediction"].min()),
        "prediction_max": float(output["prediction"].max()),
        "prediction_sum": float(output["prediction"].sum()),
        "xgboost_version": xgb.__version__,
        "python_version": platform.python_version(),
    }

    (
        args.output_dir / "prediction_report.json"
    ).write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print("OUT-OF-SAMPLE BATCH INFERENCE: PASSED")


if __name__ == "__main__":
    main()
