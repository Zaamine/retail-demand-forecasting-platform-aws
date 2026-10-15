#!/usr/bin/env bash

set -euo pipefail

DATABASE="retail_demand_forecasting_raw"
TABLE="train"

RAW_TABLE="/tmp/train-glue-table.json"
TABLE_INPUT="/tmp/train-glue-table-input.json"

# Retrieve the current Glue table definition so all existing metadata is preserved.
aws glue get-table \
  --database-name "$DATABASE" \
  --name "$TABLE" \
  --output json > "$RAW_TABLE"

# Build a valid Glue TableInput and replace only the CSV serialization settings.
python - <<'PY'
import json

raw_path = "/tmp/train-glue-table.json"
output_path = "/tmp/train-glue-table-input.json"

with open(raw_path) as file:
    table = json.load(file)["Table"]

# Keep only fields accepted by the Glue UpdateTable API.
allowed_fields = [
    "Name",
    "Description",
    "Owner",
    "LastAccessTime",
    "LastAnalyzedTime",
    "Retention",
    "StorageDescriptor",
    "PartitionKeys",
    "ViewOriginalText",
    "ViewExpandedText",
    "TableType",
    "Parameters",
]

table_input = {
    key: table[key]
    for key in allowed_fields
    if key in table and table[key] is not None
}

# Use a quote-aware CSV parser for fields such as "LIQUOR,WINE,BEER".
table_input["StorageDescriptor"]["SerdeInfo"] = {
    "SerializationLibrary": "org.apache.hadoop.hive.serde2.OpenCSVSerde",
    "Parameters": {
        "separatorChar": ",",
        "quoteChar": '"',
        "escapeChar": "\\",
    },
}

with open(output_path, "w") as file:
    json.dump(table_input, file, indent=2)
PY

# Apply the corrected table definition to the Glue Data Catalog.
aws glue update-table \
  --database-name "$DATABASE" \
  --table-input "file://$TABLE_INPUT"

echo "Updated $DATABASE.$TABLE to use OpenCSVSerde."
