"""Evaluate date/series alignment and the portfolio WAPE contract."""

import unittest

import pandas as pd

from pipelines.processing.evaluate_holdout import evaluate, metrics


class HoldoutEvaluationTest(unittest.TestCase):
    def setUp(self):
        origin = pd.Timestamp("2017-08-01")
        dates = [origin + pd.Timedelta(days=i) for i in range(1, 15)]
        self.forecasts = pd.DataFrame({
            "store_nbr": [1] * 14, "family": ["AUTOMOTIVE"] * 14,
            "forecast_origin": [origin] * 14,
            "horizon": range(1, 15), "target_date": dates,
            "prediction": [10.0] * 14,
        })
        self.actuals = pd.DataFrame({
            "store_nbr": [1] * 14, "family": ["AUTOMOTIVE"] * 14,
            "target_date": dates, "target_sales": [8.0] * 14,
        })

    def test_portfolio_metrics(self):
        aligned, report = evaluate(self.forecasts, self.actuals, "2017-08-01", 14)
        self.assertEqual(len(aligned), 14)
        self.assertAlmostEqual(report["portfolio"]["wape"], 0.25)
        self.assertAlmostEqual(report["portfolio"]["normalized_bias"], 0.25)

    def test_reject_missing_holdout_row(self):
        with self.assertRaises(ValueError):
            evaluate(self.forecasts, self.actuals.iloc[:-1], "2017-08-01", 14)

    def test_reject_bad_target_date(self):
        broken = self.forecasts.copy()
        broken.loc[0, "target_date"] += pd.Timedelta(days=1)
        with self.assertRaises(ValueError):
            evaluate(broken, self.actuals, "2017-08-01", 14)

    def test_zero_volume_wape_is_undefined(self):
        frame = pd.DataFrame({"prediction": [1.0], "target_sales": [0.0]})
        self.assertIsNone(metrics(frame)["wape"])


if __name__ == "__main__":
    unittest.main()
