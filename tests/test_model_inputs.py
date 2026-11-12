"""Contract tests; historical feature/prediction parity comes next."""

import unittest
from pathlib import Path

import pandas as pd

from forecasting.model_inputs import (
    load_model_input_schema,
    prepare_xgboost_X,
    prepare_xgboost_xy,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "artifacts/global_xgboost/point_forecast_model_contract_v1.json"
CATEGORIES = ROOT / "configs/global_xgboost/categories_v1.json"


class ModelInputsV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = load_model_input_schema(CONTRACT, CATEGORIES)

    def example_frame(self):
        values = {feature: [0.0] for feature in self.schema.features}
        for column, levels in self.schema.categorical_levels.items():
            values[column] = [levels[0]]
        values["target_sales"] = [17.25]
        values["debug_only"] = ["excluded"]
        return pd.DataFrame(values)

    def test_predictor_order_and_categorical_types(self):
        X = prepare_xgboost_X(self.example_frame(), self.schema)
        self.assertEqual(X.columns.tolist(), list(self.schema.features))
        self.assertEqual(X.shape, (1, 31))
        for column, levels in self.schema.categorical_levels.items():
            self.assertIsInstance(X[column].dtype, pd.CategoricalDtype)
            self.assertEqual(X[column].cat.categories.tolist(), list(levels))
        self.assertNotIn("target_sales", X.columns)
        self.assertNotIn("debug_only", X.columns)

    def test_label_separation(self):
        X, y = prepare_xgboost_xy(self.example_frame(), self.schema)
        self.assertEqual(X.shape, (1, 31))
        self.assertEqual(y.dtype.name, "float32")
        self.assertAlmostEqual(float(y.iloc[0]), 17.25)

    def test_unknown_store_fails_loudly(self):
        frame = self.example_frame()
        frame["store_nbr"] = [9999]
        with self.assertRaisesRegex(ValueError, "Unknown categories"):
            prepare_xgboost_X(frame, self.schema)

    def test_missing_feature_fails_loudly(self):
        frame = self.example_frame().drop(columns=["horizon"])
        with self.assertRaisesRegex(ValueError, "Missing V1 model predictors"):
            prepare_xgboost_X(frame, self.schema)


if __name__ == "__main__":
    unittest.main()
