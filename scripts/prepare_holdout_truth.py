"""Export the untouched final holdout labels from the Glue curated sales table.

Only dates after the frozen development cutoff are selected. No labels are
fed back into the forecasting model or its feature construction.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/sagemaker/holdout_evaluation_v1.json"


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    session = boto3.Session(region_name=config["region"])
    catalog = config["glue"]
    table = session.client("glue").get_table(
        DatabaseName=catalog["database"], Name=catalog["table"]
    )["Table"]
    source_uri = table["StorageDescriptor"]["Location"]
    if urlparse(source_uri).netloc != config["bucket"]:
        raise ValueError("Curated table location is outside the project bucket")

    fs, path = pafs.FileSystem.from_uri(source_uri)
    dataset = ds.dataset(path, filesystem=fs, format="parquet")
    date_type = dataset.schema.field("date").type
    start = pd.Timestamp(config["holdout"]["start"])
    end = pd.Timestamp(config["holdout"]["end"])
    if pa.types.is_date(date_type):
        low, high = start.date(), end.date()
    elif pa.types.is_timestamp(date_type):
        low, high = start.to_pydatetime(), end.to_pydatetime()
    else:
        raise TypeError(f"Unsupported curated date type: {date_type}")

    # Read only held-out labels from the catalog-resolved curated Parquet layer.
    sales = dataset.to_table(
        columns=["date", "store_nbr", "family", "sales"],
        filter=(ds.field("date") >= low) & (ds.field("date") <= high),
    ).to_pandas()
    sales = sales.rename(columns={"date": "target_date", "sales": "target_sales"})
    sales["target_date"] = pd.to_datetime(sales["target_date"])
    sales["family"] = sales["family"].astype(str)
    keys = ["store_nbr", "family", "target_date"]
    if sales.duplicated(keys).any() or sales[keys + ["target_sales"]].isna().any().any():
        raise ValueError("Curated holdout contains duplicate keys or missing labels")
    if sales["target_date"].min() != start or sales["target_date"].max() != end:
        raise ValueError("Unexpected holdout date coverage")
    if len(sales) != config["holdout"]["expected_rows"]:
        raise ValueError(f"Unexpected holdout size: {len(sales)}")

    out = ROOT / "artifacts/aws/sagemaker_processing/evaluation_v1"
    out.mkdir(parents=True, exist_ok=True)
    parquet_path = out / "holdout_actuals.parquet"
    sales.sort_values(keys).to_parquet(parquet_path, index=False)
    manifest = {
        "source": source_uri,
        "glue_table": f"{catalog['database']}.{catalog['table']}",
        "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
        "date_start": str(start.date()),
        "date_end": str(end.date()),
        "row_count": len(sales),
        "note": "Labels reserved exclusively for final out-of-time evaluation",
    }
    (out / "holdout_actuals_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    print("Prepared:", parquet_path)


if __name__ == "__main__":
    main()
