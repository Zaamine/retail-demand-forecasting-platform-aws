"""Build frozen, label-free V1 forecasting features from curated S3 sales.

The first end-to-end workflow rehearses forecast origin 2017-08-01.
The frozen forecast is used only to verify forecast keys and provenance,
never to derive predictors.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from forecasting.build_forecast import build_forecast_frame_v1
from forecasting.model_inputs import (
    load_model_input_schema,
    prepare_xgboost_X,
)


KEYS = [
    "store_nbr",
    "family",
    "forecast_origin",
    "horizon",
    "target_date",
]

INPUT = Path("/opt/ml/processing/input")


def sha256_file(path: Path) -> str:
    """Compute an artifact checksum without loading it fully into RAM."""
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize business keys for strict one-to-one reconciliation."""
    result = frame[KEYS].copy()
    result["family"] = result["family"].astype("string")
    result["forecast_origin"] = pd.to_datetime(result["forecast_origin"])
    result["target_date"] = pd.to_datetime(result["target_date"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", default="2017-08-01")
    parser.add_argument("--sales", type=Path, default=INPUT / "sales")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/opt/ml/processing/output"),
    )
    args = parser.parse_args()

    origin = pd.Timestamp(args.origin)

    # This workflow is deliberately a historical regression rehearsal.
    if origin != pd.Timestamp("2017-08-01"):
        raise ValueError("V1 historical parity requires origin 2017-08-01")

    # Filter at the Arrow layer before materializing sales in pandas.
    dataset = ds.dataset(str(args.sales), format="parquet")
    date_type = dataset.schema.field("date").type

    if pa.types.is_date(date_type):
        cutoff = origin.date()
    elif pa.types.is_timestamp(date_type):
        cutoff = origin.to_pydatetime()
    else:
        raise TypeError(f"Unsupported sales date type: {date_type}")

    sales_arrow = dataset.to_table(
        columns=["date", "store_nbr", "family", "sales"],
        filter=ds.field("date") <= cutoff,
    )

    sales = sales_arrow.to_pandas()
    sales["date"] = pd.to_datetime(sales["date"])
    sales["store_nbr"] = sales["store_nbr"].astype("int16")
    sales["family"] = sales["family"].astype("string")
    sales["sales"] = sales["sales"].astype("float32")

    if sales.empty or sales["date"].max() != origin:
        raise ValueError("Sales history does not end at the forecast origin")

    if sales.duplicated(["date", "store_nbr", "family"]).any():
        raise ValueError("Duplicate curated sales business keys")

    if sales["sales"].isna().any():
        raise ValueError("Missing historical sales")

    print(f"Curated historical sales: {len(sales):,} rows")

    # Reuse the same label-free feature engineering as local inference.
    features = build_forecast_frame_v1(sales, origin)

    schema = load_model_input_schema(
        INPUT / "contract/point_forecast_model_contract_v1.json",
        INPUT / "categories/categories_v1.json",
    )

    prepare_xgboost_X(features, schema)

    if any(
        column in features.columns
        for column in ("sales", "target_sales", "actual_sales")
    ):
        raise ValueError("Observed future sales cannot enter inference features")

    if len(features) != 24948 or features.duplicated(KEYS).any():
        raise ValueError("Expected exactly 24,948 unique forecast rows")

    if not pd.to_datetime(features["forecast_origin"]).eq(origin).all():
        raise ValueError("Unexpected forecast origin")

    if set(features["horizon"].astype(int)) != set(range(1, 15)):
        raise ValueError("Expected forecast horizons 1 through 14")

    # The reference checks historical keys only, not prediction features.
    reference_path = INPUT / "reference/final_holdout_forecast.parquet"
    reference = pd.read_parquet(reference_path)

    aligned = normalize_keys(features).merge(
        normalize_keys(reference),
        on=KEYS,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    if len(aligned) != 24948 or not aligned["_merge"].eq("both").all():
        raise ValueError("Generated keys differ from the frozen reference")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_columns = list(dict.fromkeys(KEYS + list(schema.features)))
    features_path = output_dir / "features.parquet"

    features[selected_columns].sort_values(KEYS).to_parquet(
        features_path,
        index=False,
    )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": (
            "s3://retail-demand-forecasting-platform-aws-zaamine/"
            "curated/stg_sales/"
        ),
        "forecast_origin": str(origin.date()),
        "latest_input_date": str(sales["date"].max().date()),
        "row_count": len(features),
        "feature_count": len(schema.features),
        "features_sha256": sha256_file(features_path),
        "reference_sha256": sha256_file(reference_path),
    }

    (output_dir / "features.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2))
    print("AWS FEATURE BUILD: PASSED")


if __name__ == "__main__":
    main()
