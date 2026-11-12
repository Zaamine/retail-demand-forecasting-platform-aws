"""Build label-free forecasting features for the 2017-08-15 origin."""

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


ROOT = Path("/opt/ml/processing/input")
ORIGIN = pd.Timestamp("2017-08-15")

KEYS = [
    "store_nbr",
    "family",
    "forecast_origin",
    "horizon",
    "target_date",
]


def checksum(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def expected_test_rows(path, origin=ORIGIN):
    """Read Kaggle identifiers and business keys, not future labels."""

    test = pd.read_csv(
        path,
        usecols=["id", "date", "store_nbr", "family"],
        parse_dates=["date"],
    )

    if (
        len(test) != 28512
        or test["date"].min() != origin + pd.Timedelta(days=1)
        or test["date"].max() != origin + pd.Timedelta(days=16)
    ):
        raise ValueError("Unexpected Kaggle test population or dates")

    if test["id"].isna().any() or test["id"].duplicated().any():
        raise ValueError("Missing or duplicate Kaggle IDs")

    if test.duplicated(["date", "store_nbr", "family"]).any():
        raise ValueError("Duplicate Kaggle business keys")

    test = test.loc[
        test["date"] <= origin + pd.Timedelta(days=14)
    ].copy()

    test["forecast_origin"] = origin
    test["horizon"] = (
        test["date"] - origin
    ).dt.days.astype("int16")

    test = test.rename(columns={"date": "target_date"})
    test["family"] = test["family"].astype("string")

    if (
        len(test) != 24948
        or set(test["horizon"]) != set(range(1, 15))
    ):
        raise ValueError("Expected exactly 24,948 forecast keys")

    return test[["id", *KEYS]]


def assert_key_match(left, right):
    """Check that every requested forecast key appears exactly once."""

    left = left[KEYS].copy()
    right = right[KEYS].copy()

    for frame in (left, right):
        frame["family"] = frame["family"].astype("string")
        frame["forecast_origin"] = pd.to_datetime(
            frame["forecast_origin"]
        )
        frame["target_date"] = pd.to_datetime(
            frame["target_date"]
        )

        if frame.duplicated(KEYS).any():
            raise ValueError("Duplicate forecast business keys")

    matched = left.merge(
        right,
        on=KEYS,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    if (
        len(matched) != 24948
        or not matched["_merge"].eq("both").all()
    ):
        raise ValueError("Forecast keys differ from Kaggle test keys")


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--sales",
        type=Path,
        default=ROOT / "sales",
    )

    parser.add_argument(
        "--test",
        type=Path,
        default=ROOT / "test/test.csv",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/opt/ml/processing/output"),
    )

    args = parser.parse_args()

    dataset = ds.dataset(str(args.sales), format="parquet")
    date_type = dataset.schema.field("date").type

    if pa.types.is_date(date_type):
        cutoff = ORIGIN.date()
    elif pa.types.is_timestamp(date_type):
        cutoff = ORIGIN.to_pydatetime()
    else:
        raise TypeError(f"Unsupported sales date type: {date_type}")

    table = dataset.to_table(
        columns=["date", "store_nbr", "family", "sales"],
        filter=ds.field("date") <= cutoff,
    )

    sales = table.to_pandas()

    sales["date"] = pd.to_datetime(sales["date"])
    sales["store_nbr"] = sales["store_nbr"].astype("int16")
    sales["family"] = sales["family"].astype("string")
    sales["sales"] = sales["sales"].astype("float32")

    if sales.empty or sales["date"].max() != ORIGIN:
        raise ValueError("Sales history must reach 2017-08-15")

    if (
        sales.duplicated(["date", "store_nbr", "family"]).any()
        or sales["sales"].isna().any()
    ):
        raise ValueError("Invalid historical sales")

    print(f"Historical sales: {len(sales):,} rows")

    features = build_forecast_frame_v1(sales, ORIGIN)

    schema = load_model_input_schema(
        ROOT / "contract/point_forecast_model_contract_v1.json",
        ROOT / "categories/categories_v1.json",
    )

    matrix = prepare_xgboost_X(features, schema)

    if matrix.shape != (24948, 31):
        raise ValueError("Expected 24,948 rows and 31 predictors")

    if any(
        column in features.columns
        for column in ("sales", "target_sales", "actual_sales")
    ):
        raise ValueError("Future sales cannot enter inference features")

    if (
        len(features) != 24948
        or not pd.to_datetime(
            features["forecast_origin"]
        ).eq(ORIGIN).all()
    ):
        raise ValueError("Unexpected forecast origin or population")

    expected = expected_test_rows(args.test)
    assert_key_match(features, expected)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    features_path = args.output_dir / "features.parquet"

    columns = list(dict.fromkeys(KEYS + list(schema.features)))

    features[columns].sort_values(KEYS).to_parquet(
        features_path,
        index=False,
    )

    manifest = {
        "status": "PASSED",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "forecast_origin": str(ORIGIN.date()),
        "max_sales_date": str(sales["date"].max().date()),
        "forecast_rows": len(features),
        "feature_count": len(schema.features),
        "features_sha256": checksum(features_path),
        "test_sha256": checksum(args.test),
    }

    (args.output_dir / "features.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2))
    print("OUT-OF-SAMPLE FEATURE BUILD: PASSED")


if __name__ == "__main__":
    main()
