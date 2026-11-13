"""CT-6C control-plane and dynamic checkpoint tests; no AWS calls."""
import hashlib
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import Mock

from scripts.ct_orchestration_lib import (
    CONFIG, asl_definition, code_assets, code_prefix, role_policies,
)
from scripts.deploy_ct_orchestrator import check_config, ensure_immutable_s3
from pipelines.processing.ct_workflow_readiness import parse_checkpoint, decide_readiness

CFG = json.loads(CONFIG.read_text())


def ckpt(cutoff='2017-08-15', version=2):
    return {
        'schema_version':'ct_checkpoint_v1',
        'last_successful_ct_cutoff':cutoff,
        'last_successful_cycle_phase':'CHALLENGER_REGISTERED_PENDING',
        'last_cycle_promoted':False,
        'last_cycle_new_out_of_sample_wape_available':False,
        'last_challenger_approval_status':'PendingManualApproval',
        'registered_champion_reference_arn':CFG['champion_package_arn'],
        'last_registered_challenger_arn':CFG['registry_group_arn_prefix']+str(version),
        'last_challenger_sha256':'a'*64,
    }


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.cfg=CFG.copy()
        account=self.cfg['account_id']; region=self.cfg['region']
        image=f"{account}.dkr.ecr.{region}.amazonaws.com/{self.cfg['ecr_repository']}@{self.cfg['ecr_image_digest']}"
        role=f"arn:aws:iam::{account}:role/{self.cfg['sagemaker_role_name']}"
        prefix=f"s3://{self.cfg['bucket']}/{code_prefix(self.cfg)}"
        self.asl=asl_definition(self.cfg,image,role,prefix)

    def test_dynamic_checkpoint_accepts_later_cutoff_and_version(self):
        state=ckpt('2017-08-29', 3)
        self.assertEqual(parse_checkpoint(json.dumps(state).encode(),CFG)[0],state)

    def test_rejects_older_state(self):
        with self.assertRaisesRegex(ValueError,'regressed'):
            parse_checkpoint(json.dumps(ckpt('2017-08-01')).encode(),CFG)

    def test_wrong_champion_rejected(self):
        state=ckpt(); state['registered_champion_reference_arn']='bad'
        with self.assertRaisesRegex(ValueError,'Champion'):
            parse_checkpoint(json.dumps(state).encode(),CFG)

    def test_disallow_model_promotion(self):
        state=ckpt(); state['last_cycle_promoted']=True
        with self.assertRaisesRegex(ValueError,'promote'):
            parse_checkpoint(json.dumps(state).encode(),CFG)

    def test_global_lock_conditional_and_owner_conditional_release(self):
        states=self.asl['States']
        self.assertEqual(states['AcquireGlobalLock']['Parameters']['ConditionExpression'],
                         'attribute_not_exists(LockName)')
        self.assertEqual(states['ReleaseGlobalLock']['Parameters']['ConditionExpression'],
                         '#owner = :owner')
        self.assertEqual(states['ReleaseGlobalLock']['Parameters']['ExpressionAttributeValues'][':owner']['S.$'],
                         '$$.Execution.Id')

    def test_ready_cannot_launch_any_training_or_register(self):
        states=self.asl['States']
        self.assertEqual(states['RouteReadinessStatus']['Choices'][2]['Next'],'OutcomeReadyBlocked')
        self.assertEqual(states['OutcomeReadyBlocked']['Parameters']['status'],'BLOCKED')
        blob=json.dumps(self.asl).lower()
        self.assertNotIn('createtrainingjob',blob)
        self.assertNotIn('createmodelpackage',blob)
        self.assertNotIn('updatemodelpackage',blob)
        self.assertNotIn('eventbridge:scheduler',blob)
        self.assertEqual(sum('createProcessingJob' in s.get('Resource','') for s in states.values()),1)

    def test_skip_releases_lock(self):
        self.assertEqual(self.asl['States']['OutcomeSkipped']['Next'],'ReleaseGlobalLock')
        self.assertEqual(self.asl['States']['ReleaseGlobalLock']['Next'],'ReturnOutcome')

    def test_processing_failure_retains_lock(self):
        self.assertNotIn('Catch', self.asl['States']['ReadCheckpointAndCuratedSales'])
        self.assertNotIn('Catch', self.asl['States']['ReadS3ReadinessReport'])

    def test_worker_uses_s3_checkpoint_dynamic(self):
        self.assertEqual(self.asl['States']['ReadCheckpointAndCuratedSales']['Parameters']['ProcessingResources']['ClusterConfig']['InstanceType'],'ml.t3.xlarge')
        data=decide_readiness(date(2017,8,29),date(2017,8,29),None)
        self.assertEqual(data['status'],'SKIPPED')
        self.assertEqual(data['next_required_labeled_end'],'2017-09-12')

    def test_asset_content_address_and_policy_scope(self):
        self.assertEqual(len(code_assets(CFG)),2)
        self.assertIn('/orchestration/code/',code_prefix(CFG))
        sf,sm=role_policies(CFG,'arn:aws:iam::634512762949:role/RetailForecastSageMakerExecutionRole')
        self.assertFalse(any('sagemaker:CreateTrainingJob' in json.dumps(s) for s in sf['Statement']))
        self.assertIsNone(sm)
        self.assertTrue(any(s.get('Action')=='glue:GetTable' for s in sf['Statement']))

    def test_config_guards(self):
        check_config(CFG)
        bad=CFG.copy();bad['processing_instance_type']='ml.m5.2xlarge'
        with self.assertRaises(ValueError):check_config(bad)

    def test_immutable_put_new(self):
        s3=Mock()
        self.assertEqual(ensure_immutable_s3(s3,'bucket','file.json',b'abc'),'CREATED')
        self.assertEqual(s3.put_object.call_args.kwargs['IfNoneMatch'],'*')


if __name__ == '__main__':
    unittest.main()

# Extra runtime tests independent of AWS credentials and pyarrow.
class CloudWorkerTests(unittest.TestCase):
    def test_worker_requires_no_boto3_or_cloud_sdk(self):
        from pathlib import Path
        blob=(Path(__file__).resolve().parents[1]/'pipelines/processing/ct_workflow_readiness.py').read_text()
        self.assertNotIn('import boto3',blob)
        self.assertNotIn('S3FileSystem',blob)

    def test_worker_reads_mount_and_returns_skipped(self):
        from unittest.mock import patch
        from pipelines.processing.ct_workflow_readiness import run
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'last_successful_ct.json'
            payload=json.dumps(ckpt()).encode()
            p.write_bytes(payload)
            with patch('pipelines.processing.ct_workflow_readiness.read_curated_dates_and_window',
                       return_value=(date(2017,8,15),None)):
                result=run(CFG,p,Path(td))
            self.assertEqual(result['status'],'SKIPPED')
            self.assertEqual(result['checkpoint_sha256'],hashlib.sha256(payload).hexdigest())

    def test_worker_blocks_stale_checkpoint_mismatch(self):
        from unittest.mock import patch
        from pipelines.processing.ct_workflow_readiness import run
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'last_successful_ct.json'
            p.write_bytes(json.dumps(ckpt()).encode())
            def mutate(*args):
                p.write_text('changed')
                return date(2017,8,15),None
            with patch('pipelines.processing.ct_workflow_readiness.read_curated_dates_and_window',
                       side_effect=mutate):
                with self.assertRaisesRegex(RuntimeError,'changed'):
                    run(CFG,p,Path(td))
