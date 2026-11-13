import unittest

import pandas as pd

from scripts.ct_readiness import check_readiness


def sample(days=15):
    dates = pd.date_range("2017-08-01", periods=days)
    index = pd.MultiIndex.from_product(
        [dates, range(1, 55), [f"F{i}" for i in range(33)]],
        names=["date", "store_nbr", "family"],
    )
    frame = index.to_frame(index=False)
    frame["sales"] = 10.0
    return frame


class CTReadinessTests(unittest.TestCase):

    def test_no_new_data(self):
        report = check_readiness(sample(1))
        self.assertEqual(report["status"], "SKIPPED")

    def test_incomplete_window(self):
        report = check_readiness(sample(14))
        self.assertEqual(report["status"], "SKIPPED")

    def test_complete_window(self):
        report = check_readiness(sample(15))
        self.assertEqual(report["status"], "READY")
        self.assertEqual(report["new_labeled_rows"], 24948)
        self.assertEqual(
            report["challenger_training_cutoff"], "2017-08-15"
        )

    def test_missing_series(self):
        report = check_readiness(sample(15).iloc[:-1])
        self.assertEqual(
            report["reason"], "INCOMPLETE_SERIES_COVERAGE"
        )

    def test_duplicate_keys(self):
        frame = sample(1)
        frame = pd.concat([frame, frame.iloc[:1]])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            check_readiness(frame)


if __name__ == "__main__":
    unittest.main()
