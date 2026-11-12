"""Freeze the categorical levels originally derived from development sales."""

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/global_xgboost_development_sales.csv"
OUTPUT = ROOT / "configs/global_xgboost/categories_v1.json"


def main():
    stores = set()
    families = set()

    # Read only identity columns to avoid loading the entire development history.
    for chunk in pd.read_csv(SOURCE, usecols=["store_nbr", "family"], chunksize=100_000):
        stores.update(chunk["store_nbr"].dropna().astype(int).tolist())
        families.update(chunk["family"].dropna().astype(str).tolist())

    # These counts match the fixed training population described in Notebook 05.
    if len(stores) != 54 or len(families) != 33:
        raise ValueError(
            f"Unexpected development categories: {len(stores)} stores, "
            f"{len(families)} families. Do not freeze this manifest yet."
        )

    manifest = {
        "schema_version": "v1",
        "source": "data/global_xgboost_development_sales.csv",
        "categorical_features": {
            "store_nbr": sorted(stores),
            "family": sorted(families),
            "target_day_of_week": list(range(7)),
            "target_month": list(range(1, 13)),
        },
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        existing = json.loads(OUTPUT.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("Existing V1 category manifest differs; refusing to overwrite.")
        print(f"Existing category manifest is unchanged: {OUTPUT}")
        return

    OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Frozen categories: {len(stores)} stores, {len(families)} families")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
