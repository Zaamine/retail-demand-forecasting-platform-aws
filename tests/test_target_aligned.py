"""Integration tests for V1 seasonal features and forecast assembly."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from features.origin import (
    regularize_sales_v1,
    build_origin_features_v1,
)
from features.target_aligned import build_target_aligned_features_v1
from forecasting.build_forecast import build_forecast_frame_v1
from forecasting.model_inputs import (
    load_model_input_schema,
    prepare_xgboost_X,
)


class TargetAlignedV1Tests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Build a small multi-year series with deterministic daily sales."""

        # Several years are required to test annual seasonal analogues.
        dates = pd.date_range(
            "2013-01-01", "2017-08-01", freq="D"
        )

        sales = 10 + np.arange(len(dates)) % 7

        cls.history = pd.DataFrame({
            "store_nbr": 1,
            "family": "BEVERAGES",
            "date": dates,
            "sales": sales.astype("float32"),
        })

        cls.cutoff = pd.Timestamp("2017-08-01")

    def test_historical_analogue_is_calendar_aligned(self):
        """The 364-day lag must be relative to the target date."""

        regular = regularize_sales_v1(
            self.history, self.cutoff
        )
        origin = build_origin_features_v1(regular)

        target = pd.DataFrame({
            "store_nbr": [1],
            "family": ["BEVERAGES"],
            "target_date": [pd.Timestamp("2017-08-02")],
        })

        result = build_target_aligned_features_v1(
            target, regular, origin, self.cutoff
        )

        historical_date = (
            pd.Timestamp("2017-08-02")
            - pd.Timedelta(days=364)
        )

        expected = self.history.loc[
            self.history["date"].eq(historical_date),
            "sales",
        ].iloc[0]

        self.assertEqual(
            result.iloc[0]["target_sales_lag_364d"],
            expected,
        )

        # The fixture contains several usable annual seasonal windows.
        self.assertGreaterEqual(
            int(result.iloc[0]["seasonal_years_available"]),
            3,
        )

    def test_future_changes_cannot_affect_forecast(self):
        """Changing observations after T must not change input features."""

        baseline = build_forecast_frame_v1(
            self.history, self.cutoff
        )

        # Inject an extreme future observation into the input source.
        future = pd.DataFrame({
            "store_nbr": [1],
            "family": ["BEVERAGES"],
            "date": [self.cutoff + pd.Timedelta(days=1)],
            "sales": [999999.0],
        })

        augmented = pd.concat(
            [self.history, future],
            ignore_index=True,
        )

        changed = build_forecast_frame_v1(
            augmented, self.cutoff
        )

        # The entire feature frame must remain identical.
        pd.testing.assert_frame_equal(baseline, changed)

    def test_complete_predictor_contract(self):
        """The assembled frame must provide all 31 frozen V1 predictors."""

        root = Path(__file__).resolve().parents[1]

        schema = load_model_input_schema(
            root / "artifacts/global_xgboost/point_forecast_model_contract_v1.json",
            root / "configs/global_xgboost/categories_v1.json",
        )

        frame = build_forecast_frame_v1(
            self.history, self.cutoff
        )

        X = prepare_xgboost_X(frame, schema)

        # One series and 14 horizons produce 14 prediction rows.
        self.assertEqual(len(frame), 14)

        # Feature names and ordering must match the trained model.
        self.assertEqual(
            X.columns.tolist(),
            list(schema.features),
        )

        self.assertNotIn("target_sales", frame.columns)
        self.assertTrue(
            frame["target_date"].is_monotonic_increasing
        )

    def test_future_aware_source_is_rejected(self):
        """Seasonal building must reject post-origin source observations."""

        future_cutoff = self.cutoff + pd.Timedelta(days=1)

        future = pd.DataFrame({
            "store_nbr": [1],
            "family": ["BEVERAGES"],
            "date": [future_cutoff],
            "sales": [999.0],
        })

        augmented = pd.concat(
            [self.history, future],
            ignore_index=True,
        )

        regular = regularize_sales_v1(
            augmented, future_cutoff
        )
        origin = build_origin_features_v1(regular)

        target = pd.DataFrame({
            "store_nbr": [1],
            "family": ["BEVERAGES"],
            "target_date": [future_cutoff],
        })

        # We deliberately pass an origin older than the source cutoff.
        with self.assertRaisesRegex(ValueError, "post-origin"):
            build_target_aligned_features_v1(
                target, regular, origin, self.cutoff
            )


if __name__ == "__main__":
    unittest.main()
