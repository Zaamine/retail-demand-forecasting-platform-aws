"""Pure gates for CT-3: no AWS calls, model fitting or paid resources."""
import unittest

from pipelines.processing.train_ct_challenger import check_manifest, model_config_from_contract
from scripts.launch_ct_training import build_request


FEATURES = [f'column_{i}' for i in range(31)]
CONTRACT = {
    'model_features': FEATURES, 'n_estimators': 3200, 'training_policy': 'last_2_years',
    'objective': 'reg:absoluteerror', 'max_depth': 8, 'learning_rate': 0.08,
    'min_child_weight': 1.0, 'subsample': 0.8, 'colsample_bytree': 1.0,
}
MANIFEST = {
    'status': 'PASSED', 'cutoff': '2017-08-15',
    'training_policy': 'last_2_years',
    'training_origin_sampling': 'weekly_rotating_weekday_global_anchor',
    'eligible_store_scope': 'positive_store_activity_at_origin',
    'training_origin_count': 106, 'training_rows': 2543640,
    'first_origin': '2015-08-16', 'last_origin': '2017-08-14',
    'xgboost_features': FEATURES,
    'parts': [
        {'filename': f'part-{i:04d}.parquet', 'origin': str(__import__('datetime').date(2015,8,16) + __import__('datetime').timedelta(days=i*7)), 'rows': 24000 if i < 105 else 23640}
        for i in range(106)
    ],
}
# The per-origin dates are only required to be monotone and unique by manifest.


class CTTrainingGates(unittest.TestCase):
    def test_valid_manifest(self):
        check_manifest(MANIFEST, CONTRACT)

    def test_wrong_feature_order_refused(self):
        incorrect = dict(MANIFEST, xgboost_features=FEATURES[::-1])
        with self.assertRaisesRegex(ValueError, 'Predictor order'):
            check_manifest(incorrect, CONTRACT)

    def test_incomplete_inventory_refused(self):
        incomplete = dict(MANIFEST, parts=MANIFEST['parts'][:-1])
        with self.assertRaisesRegex(ValueError, 'inventory'):
            check_manifest(incomplete, CONTRACT)

    def test_contract_no_tuning(self):
        params = model_config_from_contract(CONTRACT, n_estimators=3200, n_jobs=2)
        self.assertEqual(params['objective'], 'reg:absoluteerror')
        self.assertEqual(params['n_estimators'], 3200)
        self.assertEqual(params['tree_method'], 'hist')
        self.assertEqual(params['n_jobs'], 2)

    def test_request_is_processing_cpu_and_no_deploy(self):
        c = {'processing_instance_type': 'ml.t3.xlarge', 'volume_size_gb': 20,
             'max_runtime_seconds': 5400, 'smoke_max_runtime_seconds': 1200, 'xgboost_n_jobs': 2, 'cutoff': '2017-08-15',
             'ct2_dataset_uri': 's3://bucket/dataset/'}
        req = build_request(c, name='ct3-test', role_arn='arn:aws:iam::123:role/demo',
                            image='image@sha256:abc', code_uri='s3://bucket/code/',
                            output_uri='s3://bucket/output/', mode='smoke')
        self.assertEqual(req['ProcessingResources']['ClusterConfig']['InstanceType'], 'ml.t3.xlarge')
        self.assertEqual(req['AppSpecification']['ContainerArguments'][1], 'smoke')
        self.assertNotIn('ModelPackageGroupName', req)
        self.assertEqual(req['StoppingCondition']['MaxRuntimeInSeconds'], 1200)


if __name__ == '__main__':
    unittest.main()
