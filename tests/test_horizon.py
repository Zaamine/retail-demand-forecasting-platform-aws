"""Unit tests for the V1 direct multi-horizon forecast expansion.

These tests use a small synthetic dataset to validate the forecast
grid without requiring AWS, historical sales files, or a trained model.
"""

import unittest

import pandas as pd

from features.horizon import expand_forecast_origin_v1


class HorizonExpansionV1Tests(unittest.TestCase):

    def setUp(self):
        """Create a fresh two-series example before each test."""

        # Two independent store-family series with the same origin date.
        # The different sales values help detect incorrect row replication.
        self.origin = pd.DataFrame({
            "store_nbr": [1, 2],
            "family": ["BEVERAGES", "DAIRY"],
            "forecast_origin": ["2017-08-01", "2017-08-01"],
            "sales_at_origin": [100.0, 60.0],
        })

    def test_horizon_grid(self):
        """Check that each series receives exactly 14 future horizons."""

        result = expand_forecast_origin_v1(self.origin)

        # Two series times 14 horizons must produce 28 prediction rows.
        self.assertEqual(len(result), 28)

        # Horizons must run from 1 to 14 for the first store-family.
        self.assertEqual(
            result["horizon"].iloc[:14].tolist(),
            list(range(1, 15)),
        )

        # Forecast dates must start one day after the origin and
        # finish 14 days after it.
        self.assertEqual(
            result["target_date"].iloc[0],
            pd.Timestamp("2017-08-02"),
        )
        self.assertEqual(
            result["target_date"].iloc[13],
            pd.Timestamp("2017-08-15"),
        )

        # Historical sales must remain unchanged across horizons.
        # Repeating rows must not alter the known demand state.
        self.assertTrue(
            (result["sales_at_origin"].iloc[:14] == 100).all()
        )

    def test_calendar_features(self):
        """Verify future calendar features against known dates."""

        result = expand_forecast_origin_v1(self.origin)

        # August 2, 2017 was Wednesday (pandas weekday=2).
        self.assertEqual(
            int(result.iloc[0]["target_day_of_week"]), 2
        )

        # August 5 and 6 were Saturday and Sunday respectively.
        self.assertEqual(
            int(result.iloc[3]["target_is_weekend"]), 1
        )
        self.assertEqual(
            int(result.iloc[4]["target_is_weekend"]), 1
        )

        # The first predicted date belongs to August.
        self.assertEqual(
            int(result.iloc[0]["target_month"]), 8
        )

    def test_future_labels_are_rejected(self):
        """Ensure observed future demand cannot enter inference inputs."""

        # Inject a label that should be unavailable at prediction time.
        origin = self.origin.assign(target_sales=10)

        # The function must fail instead of silently using future sales.
        with self.assertRaisesRegex(ValueError, "Future target labels"):
            expand_forecast_origin_v1(origin)

    def test_duplicate_series_are_rejected(self):
        """Prevent duplicate keys from generating duplicate forecasts."""

        # Duplicate the first series at the same forecast origin.
        origin = pd.concat(
            [self.origin, self.origin.iloc[[0]]],
            ignore_index=True,
        )

        # Duplicates would violate the expected business grain.
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            expand_forecast_origin_v1(origin)


if __name__ == "__main__":
    unittest.main()
