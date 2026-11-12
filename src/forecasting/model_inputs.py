"""Frozen input schema and predictor preparation for global XGBoost V1."""

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd


CATEGORICAL_COLUMNS_V1 = (
    "store_nbr",
    "family",
    "target_day_of_week",
    "target_month",
)
LABEL_COLUMN = "target_sales"


@dataclass(frozen=True)
class ModelInputSchema:
    """Feature order and categorical levels required by the V1 model."""

    features: tuple[str, ...]
    categorical_levels: dict[str, tuple]


def load_model_input_schema(contract_path: Path, categories_path: Path) -> ModelInputSchema:
    """Load the existing model contract and the new frozen category manifest."""
    with Path(contract_path).open(encoding="utf-8") as handle:
        contract = json.load(handle)

    with Path(categories_path).open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    features = tuple(contract["model_features"])
    levels = manifest["categorical_features"]

    if len(features) != 31 or len(set(features)) != 31 or LABEL_COLUMN in features:
        raise ValueError("Invalid V1 predictor contract: expected 31 unique label-free columns.")

    if set(levels) != set(CATEGORICAL_COLUMNS_V1):
        raise ValueError("Category manifest does not match the four V1 categorical predictors.")

    if any(column not in features for column in CATEGORICAL_COLUMNS_V1):
        raise ValueError("A required categorical predictor is absent from the model contract.")

    for column, values in levels.items():
        if not values or len(values) != len(set(values)):
            raise ValueError(f"Invalid categorical vocabulary for {column!r}.")

    return ModelInputSchema(
        features=features,
        categorical_levels={name: tuple(values) for name, values in levels.items()},
    )


def prepare_xgboost_X(frame: pd.DataFrame, schema: ModelInputSchema) -> pd.DataFrame:
    """Select predictors in V1 order and enforce frozen categorical vocabularies."""
    missing = [name for name in schema.features if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing V1 model predictors: {missing}")

    X = frame.loc[:, list(schema.features)].copy()

    for column, allowed_levels in schema.categorical_levels.items():
        observed = set(X[column].dropna().unique().tolist())
        unknown = observed.difference(allowed_levels)
        if unknown:
            raise ValueError(f"Unknown categories in {column!r}: {sorted(unknown)!r}")

        # The order of categories is part of the model's encoding contract.
        X[column] = X[column].astype(
            pd.CategoricalDtype(categories=list(allowed_levels), ordered=False)
        )

    return X


def prepare_xgboost_xy(frame: pd.DataFrame, schema: ModelInputSchema):
    """Prepare the same predictors plus labels for supervised training."""
    if LABEL_COLUMN not in frame.columns:
        raise ValueError(f"Missing training label: {LABEL_COLUMN!r}")

    X = prepare_xgboost_X(frame, schema)
    y = frame[LABEL_COLUMN].astype("float32").copy()
    return X, y
