"""No cloud calls; CT-4 guardrails for the historical inference window."""
import unittest
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipelines.processing.validate_ct_inference import (
    ROWS, KEYS, make_report, validate_features, validate_metadata,
)


class CTInferenceTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            'feature_sha256': 'features',
            'challenger_sha256': 'challenger',
            'ct2_manifest_sha256': 'dataset',
        }
        self.feature_manifest = dict(
            status='PASSED', forecast_origin='2017-08-15',
            max_sales_date='2017-08-15', forecast_rows=ROWS,
            feature_count=31, features_sha256='features',
        )
        self.training_report = dict(
            status='PASSED', mode='full', model_role='challenger_unapproved',
            training_cutoff='2017-08-15', dataset_rows_total=2_543_640,
            fit_rows=2_543_640, new_label_rows=44_550,
            sampled_forecast_origins=106, features=31, n_estimators=3200,
            xgboost_version='3.4.1',
            evaluation_type='training_integrity_only',
            has_new_out_of_sample_metric=False, promotion_eligible=False,
            model_archive_sha256='challenger', input_manifest_sha256='dataset',
        )

    def test_validate_metadata(self):
        validate_metadata(self.feature_manifest, self.training_report, self.cfg)

    def test_reject_promotion_eligible(self):
        self.training_report['promotion_eligible'] = True
        with self.assertRaisesRegex(ValueError, 'promotion_eligible'):
            validate_metadata(self.feature_manifest, self.training_report, self.cfg)

    def test_reject_wrong_feature_snapshot(self):
        self.feature_manifest['features_sha256'] = 'other'
        with self.assertRaisesRegex(ValueError, 'features_sha256'):
            validate_metadata(self.feature_manifest, self.training_report, self.cfg)

    def test_reject_tainted_horizon(self):
        dates = pd.date_range('2017-08-16', periods=14)
        series = pd.MultiIndex.from_product([list(range(1,55)), [f'F{i}' for i in range(33)], dates],
                                            names=['store_nbr','family','target_date']).to_frame(index=False)
        series['forecast_origin'] = pd.Timestamp('2017-08-15')
        series['horizon'] = (series['target_date'] - series['forecast_origin']).dt.days
        contract = {'model_features': ['p'+str(i) for i in range(31)]}
        for col in contract['model_features']:
            series[col] = 1
        validate_features(series, contract)
        series.loc[0,'target_sales'] = 1
        with self.assertRaisesRegex(ValueError, 'forbidden'):
            validate_features(series, contract)

    def test_difference_is_not_performance(self):
        frame = pd.DataFrame({k:[None]*ROWS for k in KEYS})
        champion = np.ones(ROWS)
        challenger = np.full(ROWS, 2.0)
        output, report = make_report(frame, champion, challenger, {})
        self.assertEqual(len(output), ROWS)
        self.assertEqual(report['evaluated_labels'], 0)
        self.assertEqual(report['mean_signed_prediction_delta'], 1.0)
        self.assertFalse(report['promotion_eligible'])
        self.assertFalse(report['has_new_out_of_sample_metric'])

if __name__ == '__main__':
    unittest.main()
