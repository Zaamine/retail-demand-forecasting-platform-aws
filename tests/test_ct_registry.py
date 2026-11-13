"""CT-5 registry governance tests (entirely offline, no AWS mutation)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts import register_ct_challenger as r

CFG = json.loads((Path(__file__).resolve().parents[1]
                  / 'configs/sagemaker/ct_registry_v1.json').read_text())


def training_document():
    return {
        'status': 'PASSED', 'mode': 'full', 'model_role': 'challenger_unapproved',
        'training_cutoff': '2017-08-15', 'dataset_rows_total': 2543640,
        'fit_rows': 2543640, 'new_label_rows': 44550,
        'sampled_forecast_origins': 106, 'features': 31, 'n_estimators': 3200,
        'xgboost_version': '3.4.1', 'training_policy': 'last_2_years',
        'evaluation_type': 'training_integrity_only',
        'input_manifest_sha256': CFG['ct2_manifest_sha256'],
        'model_archive_size_bytes': CFG['challenger_model_size_bytes'],
        'model_archive_sha256': CFG['challenger_model_sha256'],
        'has_new_out_of_sample_metric': False, 'promotion_eligible': False,
        'champion_v1_unchanged': True,
    }


def inference_document():
    return {
        'status': 'PASSED',
        'validation_type': 'same_snapshot_unlabeled_model_comparison',
        'forecast_origin': '2017-08-15', 'first_target_date': '2017-08-16',
        'last_target_date': '2017-08-29', 'forecast_rows': 24948,
        'unique_series': 1782, 'feature_count': 31, 'horizons': list(range(1, 15)),
        'model_champion_role': 'frozen_v1', 'model_challenger_role': 'v2_unapproved',
        'evaluated_labels': 0, 'has_new_out_of_sample_metric': False,
        'promotion_eligible': False,
        'champion_model_sha256': CFG['champion_v1_model_sha256'],
        'challenger_model_sha256': CFG['challenger_model_sha256'],
        'features_sha256': CFG['features_sha256'],
        'training_report_sha256': CFG['training_report_sha256'],
        'comparison_parquet_sha256': CFG['ct4_comparison_sha256'],
        'xgboost_version': '3.4.1',
        'v1_prediction_sum': 12346663.928606909,
        'v2_prediction_sum': 11196962.139246603,
        'mean_absolute_prediction_delta': 48.028968359161134,
        'mean_signed_prediction_delta': -46.08392614078511,
        'fraction_predictions_changed_gt_0_01': 0.9737854737854738,
    }


def manifest_document():
    return {
        'status': 'PASSED', 'cutoff': '2017-08-15',
        'training_policy': 'last_2_years', 'training_origin_count': 106,
        'training_rows': 2543640, 'last_origin': '2017-08-14',
        'parts': [{'rows': 23997}] * 105 + [{'rows': 2543640 - 23997*105}],
    }


class CT5Tests(unittest.TestCase):
    def test_manifest_valid(self):
        r.check_ct2_manifest(manifest_document(), CFG)

    def test_manifest_invalid(self):
        doc = manifest_document()
        doc['last_origin'] = '2017-08-15'
        with self.assertRaises(ValueError):
            r.check_ct2_manifest(doc, CFG)

    def test_training_valid(self):
        r.check_training_report(training_document(), CFG)

    def test_training_rejects_promotion_eligible(self):
        doc = training_document()
        doc['promotion_eligible'] = True
        with self.assertRaises(ValueError):
            r.check_training_report(doc, CFG)

    def test_inference_valid(self):
        r.check_inference_report(inference_document(), CFG, CFG['training_report_sha256'])

    def test_inference_rejects_claimed_new_metric(self):
        doc = inference_document()
        doc['has_new_out_of_sample_metric'] = True
        with self.assertRaises(ValueError):
            r.check_inference_report(doc, CFG, CFG['training_report_sha256'])

    def test_registry_request_cannot_approve_or_invent_metrics(self):
        attrs = {
            'ModelSHA256': CFG['challenger_model_sha256'],
            'TrainingReportSHA256': CFG['training_report_sha256'],
            'InferenceReportSHA256': 'a' * 64,
            'ChallengerModelS3URI': f"s3://{CFG['bucket']}/{CFG['ct3_prefix']}model.tar.gz",
            'RegistryModelS3URI': f"s3://{CFG['bucket']}/{CFG['registry_model_key']}",
        }
        req = r.build_request(CFG, attrs)
        self.assertEqual(req['ModelApprovalStatus'], 'PendingManualApproval')
        self.assertNotIn('ModelMetrics', req)
        self.assertNotIn('InferenceSpecification', req)
        self.assertEqual(req['SourceUri'], attrs['RegistryModelS3URI'])
        r.validate_request_shape(req)
        self.assertEqual(len(req['ClientToken']), 36)
        changed = dict(CFG, approval_status='Approved')
        with self.assertRaises(ValueError):
            r.build_request(changed, attrs)

    def test_sagemaker_source_uri_accepts_registry_copy_and_rejects_equals(self):
        original = f"s3://{CFG['bucket']}/{CFG['ct3_prefix']}model.tar.gz"
        compatible = f"s3://{CFG['bucket']}/{CFG['registry_model_key']}"
        with self.assertRaisesRegex(ValueError, 'forbidden'):
            r.validate_registry_source_uri(original)
        r.validate_registry_source_uri(compatible)

    def test_request_preflight_rejects_equals_and_oversized_metadata(self):
        attrs = {
            'ModelSHA256': CFG['challenger_model_sha256'],
            'TrainingReportSHA256': CFG['training_report_sha256'],
            'InferenceReportSHA256': 'a' * 64,
            'ChallengerModelS3URI': f"s3://{CFG['bucket']}/{CFG['ct3_prefix']}model.tar.gz",
            'RegistryModelS3URI': f"s3://{CFG['bucket']}/{CFG['registry_model_key']}",
        }
        request = r.build_request(CFG, attrs)
        r.validate_request_shape(request)
        broken = dict(request, SourceUri=attrs['ChallengerModelS3URI'])
        with self.assertRaises(ValueError):
            r.validate_request_shape(broken)
        broken = dict(request, CustomerMetadataProperties=dict(attrs, Extra='z'*257))
        with self.assertRaises(ValueError):
            r.validate_request_shape(broken)

    def test_client_token_deterministic(self):
        values = ('group', 'model', 'train', 'inference')
        self.assertEqual(r.registration_token(*values), r.registration_token(*values))
        self.assertNotEqual(r.registration_token(*values),
                            r.registration_token('group', 'model', 'train2', 'inference'))

    def test_duplicate_detection_fail_closed_on_changed_lineage(self):
        sm = MagicMock()
        sm.get_paginator.return_value.paginate.return_value = [
            {'ModelPackageSummaryList': [{'ModelPackageArn': 'arn:example'}]}]
        sm.describe_model_package.return_value = {
            'ModelPackageArn': 'arn:example',
            'CustomerMetadataProperties': {
                'ModelSHA256': 'modelhash', 'TrainingReportSHA256': 'wrong',
                'InferenceReportSHA256': 'auditsha',
            },
        }
        attrs = {'ModelSHA256': 'modelhash', 'TrainingReportSHA256': 'right',
                 'InferenceReportSHA256': 'auditsha', 'ChallengerModelS3URI': 's3://b/k',
                 'RegistryModelS3URI': 's3://b/registry/model.tar.gz'}
        with self.assertRaises(ValueError):
            r.matching_existing_package(sm, CFG, attrs)
        sm.get_paginator.return_value.paginate.assert_called_once_with(
            ModelPackageGroupName=CFG['model_package_group_name'],
            ModelPackageType='Versioned')

    def test_dry_run_reads_only_and_never_creates_registry_version(self):
        sts, s3, sm = MagicMock(), MagicMock(), MagicMock()
        sts.get_caller_identity.return_value = {'Account': CFG['account_id']}
        sm.describe_model_package_group.return_value = {
            'ModelPackageGroupName': CFG['model_package_group_name']}
        sm.describe_model_package.return_value = {
            'ModelPackageArn': CFG['champion_v1_package_arn'],
            'ModelPackageGroupName': CFG['model_package_group_name'],
            'ModelPackageVersion': 1,
            'SourceUri': f"s3://{CFG['bucket']}/{CFG['champion_v1_model_key']}",
            'CustomerMetadataProperties': {'ModelSHA256': CFG['champion_v1_model_sha256']},
            'ModelApprovalStatus': 'PendingManualApproval',
        }
        def describe_job(ProcessingJobName):
            for label in ('ct2', 'ct3', 'ct4'):
                if CFG[f'{label}_processing_job'] == ProcessingJobName:
                    prefix = CFG[f'{label}_prefix']
                    return {
                        'ProcessingJobStatus': 'Completed',
                        'ProcessingJobArn': (f"arn:aws:sagemaker:{CFG['region']}:"
                                             f"{CFG['account_id']}:processing-job/{ProcessingJobName}"),
                        'ProcessingOutputConfig': {'Outputs': [
                            {'S3Output': {'S3Uri': f"s3://{CFG['bucket']}/{prefix}"}}]},
                    }
            raise AssertionError('Unexpected processing job')
        sm.describe_processing_job.side_effect = describe_job
        sm.get_paginator.return_value.paginate.return_value = [{'ModelPackageSummaryList': []}]
        s3.head_object.return_value = {'ContentLength': CFG['challenger_model_size_bytes']}
        def fetch_json(s3client, bucket, key):
            if key.endswith('dataset_manifest.json'):
                return manifest_document(), CFG['ct2_manifest_sha256']
            if key.endswith('training_report.json'):
                return training_document(), CFG['training_report_sha256']
            if key.endswith('inference_report.json'):
                return inference_document(), 'f' * 64
            raise AssertionError('Unexpected JSON')
        session = MagicMock()
        session.client.side_effect = lambda name: {'sts': sts, 's3': s3, 'sagemaker': sm}[name]
        def get_sha(s3client, bucket, key):
            if key.endswith('model.tar.gz'):
                return CFG['challenger_model_sha256']
            if key.endswith('challenger_comparison.parquet'):
                return CFG['ct4_comparison_sha256']
            raise AssertionError('Unexpected object hash')
        with patch.object(r.boto3, 'Session', return_value=session), \
             patch.object(r, 'read_json_s3', side_effect=fetch_json), \
             patch.object(r, 'sha256_s3_object', side_effect=get_sha), \
             patch.object(sys, 'argv', ['register_ct_challenger.py']), \
             redirect_stdout(StringIO()) as out:
            r.main()
        sm.create_model_package.assert_not_called()
        sm.update_model_package.assert_not_called()
        self.assertIn('DRY RUN: read-only AWS verification', out.getvalue())

    def test_execute_requires_double_confirmation(self):
        with patch.object(sys, 'argv', ['script', '--execute']), \
             redirect_stdout(StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                r.main()
        self.assertEqual(ctx.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
