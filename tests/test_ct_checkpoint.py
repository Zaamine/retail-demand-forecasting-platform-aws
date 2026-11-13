import io
import json
import unittest
from botocore.exceptions import ClientError
from scripts.bootstrap_ct_checkpoint import (checkpoint_bytes, build_checkpoint,
                                             atomic_bootstrap, validate_registry_v2)


class FakeS3:
    def __init__(self, initial=None):
        self.data = initial
        self.puts = 0
        self.last_kwargs = None

    def get_object(self, **kwargs):
        if self.data is None:
            raise ClientError({'Error': {'Code': 'NoSuchKey', 'Message': 'missing'}}, 'GetObject')
        return {'Body': io.BytesIO(self.data)}

    def put_object(self, **kwargs):
        self.puts += 1
        self.last_kwargs = kwargs
        if self.data is not None:
            raise ClientError({'Error': {'Code': 'PreconditionFailed', 'Message': 'exists'}}, 'PutObject')
        self.data = kwargs['Body']


class FakeSM:
    def __init__(self, obj):
        self.obj = obj

    def describe_model_package(self, **kwargs):
        return self.obj


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.state = {'schema_version': 'ct_checkpoint_v1', 'last_successful_ct_cutoff': '2017-08-15'}

    def test_dry_run_does_not_write(self):
        client = FakeS3()
        self.assertEqual(atomic_bootstrap(client, 'bucket', 'key', self.state), 'DRY_RUN_WOULD_CREATE')
        self.assertEqual(client.puts, 0)

    def test_execute_conditional_write_and_readback(self):
        client = FakeS3()
        self.assertEqual(atomic_bootstrap(client, 'bucket', 'key', self.state, True), 'CREATED_AND_VERIFIED')
        self.assertEqual(client.last_kwargs['IfNoneMatch'], '*')
        self.assertEqual(json.loads(client.data), self.state)

    def test_repeat_identical_is_no_op(self):
        client = FakeS3(checkpoint_bytes(self.state))
        self.assertEqual(atomic_bootstrap(client, 'bucket', 'key', self.state, True), 'ALREADY_PRESENT_IDENTICAL')
        self.assertEqual(client.puts, 0)

    def test_refuse_conflicting_checkpoint(self):
        client = FakeS3(checkpoint_bytes({'last_successful_ct_cutoff': '2017-08-29'}))
        with self.assertRaisesRegex(RuntimeError, 'refusing to overwrite'):
            atomic_bootstrap(client, 'bucket', 'key', self.state, True)
        self.assertEqual(client.puts, 0)

    def test_deterministic_payload(self):
        self.assertEqual(checkpoint_bytes({'z':1,'a':2}), checkpoint_bytes({'a':2,'z':1}))
        self.assertTrue(checkpoint_bytes(self.state).endswith(b'\n'))

    def test_registry_rejects_approval(self):
        cfg = dict(bucket='bucket', registry_model_key='key', training_cutoff='2017-08-15', forecast_origin='2017-08-15',
                   challenger_model_sha256='hash', training_report_sha256='reportsha', ct2_manifest_sha256='manifestsha',
                   ct2_processing_job='ct2', ct3_processing_job='ct3', ct4_processing_job='ct4')
        version = 'arn:aws:sagemaker:eu-west-1:123:model-package/group/2'
        obj = {'ModelPackageArn':version, 'ModelPackageStatus':'Completed', 'ModelApprovalStatus':'Approved',
               'SourceUri':'s3://bucket/key', 'CustomerMetadataProperties':{}}
        with self.assertRaisesRegex(ValueError, 'approval'):
            validate_registry_v2(FakeSM(obj), cfg, version)

    def test_registry_rejects_wrong_hash(self):
        cfg = dict(bucket='bucket', registry_model_key='key', training_cutoff='2017-08-15', forecast_origin='2017-08-15',
                   challenger_model_sha256='hash', training_report_sha256='reportsha', ct2_manifest_sha256='manifestsha',
                   ct2_processing_job='ct2', ct3_processing_job='ct3', ct4_processing_job='ct4')
        version = 'arn:aws:sagemaker:eu-west-1:123:model-package/group/2'
        attrs = {'TrainingCutoff':'2017-08-15', 'ForecastOrigin':'2017-08-15','ModelSHA256':'BAD',
                 'TrainingReportSHA256':'reportsha', 'DatasetManifestSHA256':'manifestsha',
                 'PromotionEligible':'false', 'NewOutOfSampleWAPE':'unavailable',
                 'TrainingExecution':'sagemaker_processing_cpu','CT2ProcessingJob':'ct2',
                 'CT3ProcessingJob':'ct3','CT4ProcessingJob':'ct4'}
        obj = {'ModelPackageArn':version, 'ModelPackageStatus':'Completed', 'ModelApprovalStatus':'PendingManualApproval',
               'SourceUri':'s3://bucket/key', 'CustomerMetadataProperties': attrs}
        with self.assertRaisesRegex(ValueError, 'ModelSHA256'):
            validate_registry_v2(FakeSM(obj), cfg, version)


if __name__ == '__main__':
    unittest.main()
