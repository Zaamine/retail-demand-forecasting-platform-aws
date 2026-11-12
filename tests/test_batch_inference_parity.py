"""Guard against accidental key omission or duplicates during inference."""
import unittest
import pandas as pd
from forecasting.parity import align_forecasts


class BatchParityTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "store_nbr": [1, 2], "family": ["AUTOMOTIVE", "GROCERY I"],
            "forecast_origin": pd.to_datetime(["2017-08-01"] * 2),
            "horizon": [1, 1], "target_date": pd.to_datetime(["2017-08-02"] * 2),
            "prediction_raw": [1.0, 2.0], "prediction": [1.0, 2.0],
        })

    def test_reordered_reference(self):
        matched = align_forecasts(self.frame, self.frame.iloc[::-1])
        self.assertEqual(len(matched), 2)
        self.assertTrue((matched.prediction_aws == matched.prediction_notebook).all())

    def test_missing_key(self):
        with self.assertRaises(ValueError):
            align_forecasts(self.frame, self.frame.iloc[:1])

    def test_duplicate_key(self):
        with self.assertRaises(ValueError):
            align_forecasts(self.frame, pd.concat([self.frame, self.frame.iloc[:1]]))


if __name__ == "__main__":
    unittest.main()
