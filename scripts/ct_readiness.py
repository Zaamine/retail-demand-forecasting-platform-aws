"""Read-only readiness gate for a 14-day forecasting retraining cycle."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LAST_TRAINING_DATE = pd.Timestamp("2017-08-01")
HORIZON = 14
EXPECTED_SERIES = 1782
KEYS = ["store_nbr", "family"]


def check_readiness(frame, last_training_date=LAST_TRAINING_DATE):
    cutoff = pd.Timestamp(last_training_date)
    end = cutoff + pd.Timedelta(days=HORIZON)

    required = {"date", "store_nbr", "family", "sales"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")

    data = frame[list(required)].copy()
    data["date"] = pd.to_datetime(data["date"])
    data["sales"] = pd.to_numeric(data["sales"], errors="coerce")

    if data["date"].isna().any():
        raise ValueError("Invalid dates")

    if data.duplicated(["date", *KEYS]).any():
        raise ValueError("Duplicate business keys")

    if data[KEYS].isna().any().any():
        raise ValueError("Missing series identifiers")

    if not np.isfinite(data["sales"].to_numpy(dtype=float)).all():
        raise ValueError("Missing or non-finite sales labels")

    baseline = data.loc[data["date"].eq(cutoff), KEYS]
    baseline_keys = set(map(tuple, baseline.itertuples(index=False, name=None)))

    if len(baseline_keys) != EXPECTED_SERIES:
        raise ValueError("Invalid baseline series population")

    latest = data["date"].max()

    report = {
        "last_training_date": str(cutoff.date()),
        "required_labeled_end": str(end.date()),
        "latest_observed_date": str(latest.date()),
        "expected_series": EXPECTED_SERIES,
        "horizon_days": HORIZON,
        "training_started": False,
        "champion_modified": False,
    }

    if latest < end:
        return {
            **report,
            "status": "SKIPPED",
            "reason": "INSUFFICIENT_NEW_LABELS",
        }

    window = data.loc[data["date"].between(
        cutoff + pd.Timedelta(days=1), end
    )]

    for day in pd.date_range(
        cutoff + pd.Timedelta(days=1), end
    ):
        actual_keys = set(map(
            tuple,
            window.loc[window["date"].eq(day), KEYS]
            .itertuples(index=False, name=None),
        ))

        if actual_keys != baseline_keys:
            return {
                **report,
                "status": "SKIPPED",
                "reason": "INCOMPLETE_SERIES_COVERAGE",
                "problem_date": str(day.date()),
                "observed_series": len(actual_keys),
            }

    return {
        **report,
        "status": "READY",
        "reason": "COMPLETE_NEW_LABELED_WINDOW",
        "new_labels_start": str(
            (cutoff + pd.Timedelta(days=1)).date()
        ),
        "new_labels_end": str(end.date()),
        "new_labeled_rows": len(window),
        "challenger_training_cutoff": str(end.date()),
        "next_step": "Build new supervised dataset and train challenger",
    }


def read_curated_s3(region):
    import boto3
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.fs as pafs

    session = boto3.Session(region_name=region)
    table = session.client("glue").get_table(
        DatabaseName="retail_demand_forecasting_curated",
        Name="stg_sales",
    )["Table"]

    uri = table["StorageDescriptor"]["Location"]
    filesystem, path = pafs.FileSystem.from_uri(uri)

    dataset = ds.dataset(
        path,
        filesystem=filesystem,
        format="parquet",
    )

    date_type = dataset.schema.field("date").type

    if pa.types.is_date(date_type):
        cutoff = LAST_TRAINING_DATE.date()
    elif pa.types.is_timestamp(date_type):
        cutoff = LAST_TRAINING_DATE.to_pydatetime()
    else:
        raise TypeError(f"Unsupported date type: {date_type}")

    result = dataset.to_table(
        columns=["date", "store_nbr", "family", "sales"],
        filter=ds.field("date") >= cutoff,
    )

    return result.to_pandas()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="eu-west-1")
    parser.add_argument(
        "--output",
        default="artifacts/aws/continuous_training/readiness_v1.json",
    )
    args = parser.parse_args()

    frame = read_curated_s3(args.region)
    report = check_readiness(frame)
    report["source"] = "glue:retail_demand_forecasting_curated.stg_sales"

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
