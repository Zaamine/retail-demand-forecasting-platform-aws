"""Materialize frozen label-free V1 features with the project's reusable code.

Uses historical retail sales only through the 2017-08-01 forecast origin.
Runs locally to avoid memory pressure on the quota-limited AWS instance.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from forecasting.build_forecast import build_forecast_frame_v1
from forecasting.model_inputs import load_model_input_schema, prepare_xgboost_X

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["store_nbr", "family", "forecast_origin", "horizon", "target_date"]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_keys(df):
    frame = df.loc[:, KEYS].copy()
    frame["family"] = frame["family"].astype("string")
    for name in ("forecast_origin", "target_date"):
        frame[name] = pd.to_datetime(frame[name])
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-sales", type=Path, default=ROOT / "data/raw/favorita/train/train.csv")
    parser.add_argument("--origin", default="2017-08-01")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts/aws/sagemaker_batch_inference/v1/features.parquet",
    )
    args = parser.parse_args()
    cutoff = pd.Timestamp(args.origin)
    if cutoff != pd.Timestamp("2017-08-01"):
        raise ValueError("The frozen V1 parity check requires origin 2017-08-01")
    if not args.raw_sales.is_file():
        raise FileNotFoundError(args.raw_sales)

    chunks = []
    for chunk in pd.read_csv(
        args.raw_sales,
        usecols=["date", "store_nbr", "family", "sales"],
        dtype={"store_nbr": "int16", "family": "string", "sales": "float32"},
        parse_dates=["date"], chunksize=250_000,
    ):
        before_cutoff = chunk.loc[chunk["date"] <= cutoff]
        if not before_cutoff.empty:
            chunks.append(before_cutoff)
    if not chunks:
        raise ValueError("No development sales were found")
    sales = pd.concat(chunks, ignore_index=True)
    del chunks
    if sales["date"].max() != cutoff:
        raise ValueError("Development sales must end exactly on 2017-08-01")
    if sales.duplicated(["date", "store_nbr", "family"]).any():
        raise ValueError("Duplicate development sales keys")

    print(f"Building V1 features from {len(sales):,} historical rows...")
    frame = build_forecast_frame_v1(sales, cutoff)
    schema = load_model_input_schema(
        ROOT / "artifacts/global_xgboost/point_forecast_model_contract_v1.json",
        ROOT / "configs/global_xgboost/categories_v1.json",
    )
    prepare_xgboost_X(frame, schema)
    if any(name in frame.columns for name in ("target_sales", "actual_sales", "sales")):
        raise ValueError("Forbidden actual-sales column in the prediction frame")
    if len(frame) != 24_948 or frame.duplicated(KEYS).any():
        raise ValueError("Forecast frame must contain 24,948 unique rows")
    if not pd.to_datetime(frame["forecast_origin"]).eq(cutoff).all():
        raise ValueError("Unexpected forecast origin")
    if set(frame["horizon"].astype(int)) != set(range(1, 15)):
        raise ValueError("Unexpected forecast horizon")

    reference = pd.read_parquet(ROOT / "artifacts/global_xgboost/final_holdout_forecast.parquet")
    overlap = normalize_keys(frame).merge(
        normalize_keys(reference), on=KEYS, how="outer", indicator=True, validate="one_to_one"
    )
    if len(overlap) != 24_948 or not overlap["_merge"].eq("both").all():
        raise ValueError("Generated inference keys differ from frozen notebook predictions")

    subset = list(dict.fromkeys(KEYS + list(schema.features)))
    frame = frame.loc[:, subset].sort_values(KEYS).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    reference_path = ROOT / "artifacts/global_xgboost/final_holdout_forecast.parquet"
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "forecast_origin": str(cutoff.date()),
        "source": "data/raw/favorita/train/train.csv",
        "latest_input_date": str(sales["date"].max().date()),
        "row_count": len(frame),
        "feature_count": len(schema.features),
        "features_sha256": sha256_file(args.output),
        "reference_sha256": sha256_file(reference_path),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    print("FROZEN LABEL-FREE FEATURE FRAME: PASSED")


if __name__ == "__main__":
    main()
