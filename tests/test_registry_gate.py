"""Unit tests for the frozen champion's cloud reproduction audit."""

import copy
import unittest

from forecasting.registry_gate import audit_frozen_champion


POLICY = {
    "forecast_origin": "2017-08-01",
    "holdout_start": "2017-08-02",
    "holdout_end": "2017-08-15",
    "expected_rows": 24948,
    "expected_series": 1782,
    "reference_wape": 0.1877245082,
    "wape_reproduction_tolerance": 1e-5,
    "expected_forecasts_sha256": "example-frozen-digest",
}

REPORT = {
    "status": "PASSED",
    "validation_type": "frozen_v1_final_temporal_holdout",
    "forecast_origin": "2017-08-01",
    "holdout_start": "2017-08-02",
    "holdout_end": "2017-08-15",
    "series_count": 1782,
    "portfolio": {
        "count": 24948,
        "wape": 0.18772450849472005,
        "normalized_bias": 0.12092500115095109,
        "mae": 86.11786714497063,
        "rmse": 271.99633906579334,
    },
    "reference_wape": 0.1877245082,
    "input_sha256": {"forecasts": "example-frozen-digest"},
}


class RegistryGateTests(unittest.TestCase):
    def test_matching_frozen_run_passes(self):
        gate = audit_frozen_champion(REPORT, POLICY)
        self.assertEqual(gate["status"], "PASSED")
        self.assertLess(gate["absolute_wape_difference"], 1e-8)

    def test_different_wape_rejected(self):
        report = copy.deepcopy(REPORT)
        report["portfolio"]["wape"] = 0.20
        with self.assertRaisesRegex(ValueError, "does not reproduce"):
            audit_frozen_champion(report, POLICY)

    def test_wrong_forecast_digest_rejected(self):
        report = copy.deepcopy(REPORT)
        report["input_sha256"]["forecasts"] = "different"
        with self.assertRaisesRegex(ValueError, "frozen V1 forecast"):
            audit_frozen_champion(report, POLICY)

    def test_wrong_holdout_origin_rejected(self):
        report = copy.deepcopy(REPORT)
        report["forecast_origin"] = "2017-08-02"
        with self.assertRaisesRegex(ValueError, "forecast_origin"):
            audit_frozen_champion(report, POLICY)

    def test_wrong_series_coverage_rejected(self):
        report = copy.deepcopy(REPORT)
        report["series_count"] = 1700
        with self.assertRaisesRegex(ValueError, "series count"):
            audit_frozen_champion(report, POLICY)

    def test_non_finite_metric_rejected(self):
        report = copy.deepcopy(REPORT)
        report["portfolio"]["wape"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            audit_frozen_champion(report, POLICY)


if __name__ == "__main__":
    unittest.main()
