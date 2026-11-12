"""Tests for calendar-aware origin features and point-in-time safety."""

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from features.origin import (
    regularize_sales_v1,
    build_origin_features_v1,
)


class OriginFeatureV1Tests(unittest.TestCase):

    def setUp(self):
        """Create two observed series over 42 calendar days."""

        dates = pd.date_range(
            "2017-01-01", periods=42, freq="D"
        )

        records = []

        for store, family in (
            (1, "BEVERAGES"),
            (2, "DAIRY"),
        ):
            for index, date in enumerate(dates):

                # Simulate a globally missing date, as in Favorita.
                if date == pd.Timestamp("2017-01-10"):
                    continue

                # An observed zero must be distinct from missing sales.
                value = (
                    0.0
                    if date == pd.Timestamp("2017-01-12")
                    else float(10 + index)
                )

                records.append((store, family, date, value))

        self.sales = pd.DataFrame(
            records,
            columns=["store_nbr", "family", "date", "sales"],
        )

    def test_missing_date_is_preserved(self):
        """Missing calendar dates must not be silently treated as zero."""

        regular = regularize_sales_v1(
            self.sales, "2017-02-11"
        )

        # V1 uses the complete store x family Cartesian product.
        # Two stores and two families produce four possible series.
        self.assertEqual(len(regular), 4 * 42)

        missing_day = regular.loc[
            regular["date"].eq("2017-01-10")
        ]

        self.assertTrue(missing_day["sales"].isna().all())
        self.assertFalse(missing_day["sales_observed"].any())

    def test_calendar_lags_do_not_skip_missing_dates(self):
        """Lag 7 must mean seven calendar days, not seven observations."""

        regular = regularize_sales_v1(
            self.sales, "2017-02-11"
        )
        features = build_origin_features_v1(regular)

        row = features.loc[
            features["store_nbr"].eq(1)
            & features["forecast_origin"].eq("2017-01-17")
        ].iloc[0]

        # January 10 is missing, so the seven-day lag is missing.
        self.assertTrue(np.isnan(row["sales_lag_7d"]))

        # January 16 is observed with 25 units.
        self.assertEqual(row["sales_lag_1d"], 25.0)

    def test_rolling_counts_exclude_missing_but_keep_zero(self):
        """Rolling demand statistics must preserve missing/zero semantics."""

        regular = regularize_sales_v1(
            self.sales, "2017-02-11"
        )
        features = build_origin_features_v1(regular)

        row = features.loc[
            features["store_nbr"].eq(1)
            & features["forecast_origin"].eq("2017-01-12")
        ].iloc[0]

        # Twelve calendar days, one missing observation.
        self.assertEqual(row["observed_days_28d"], 11)

        # Ten positive observations out of eleven observed days.
        self.assertAlmostEqual(
            row["positive_rate_28d"], 10 / 11, places=6
        )

        # Last positive sale was January 11.
        self.assertEqual(row["days_since_last_positive"], 1)

    def test_future_changes_do_not_change_past_features(self):
        """Changing future sales must not affect features at an earlier T."""

        before = build_origin_features_v1(
            regularize_sales_v1(self.sales, "2017-01-20")
        )

        changed = self.sales.copy()

        # Deliberately corrupt every sales value after the cutoff.
        changed.loc[
            changed["date"].gt("2017-01-20"), "sales"
        ] = 999999.0

        after = build_origin_features_v1(
            regularize_sales_v1(changed, "2017-01-20")
        )

        # Compare every row, feature value, dtype and column order.
        assert_frame_equal(before, after)

    def test_duplicate_business_keys_are_rejected(self):
        """Duplicate source keys must fail before feature computation."""

        duplicate = pd.concat(
            [self.sales, self.sales.iloc[[0]]],
            ignore_index=True,
        )

        with self.assertRaisesRegex(ValueError, "Duplicate"):
            regularize_sales_v1(duplicate, "2017-02-11")


if __name__ == "__main__":
    unittest.main()
