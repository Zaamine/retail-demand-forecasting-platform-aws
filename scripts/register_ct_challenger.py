"""CT-5: read-only evidence audit, then optional pending-only SageMaker Model Registry V2.

Default: entirely read-only AWS operations.  --execute --confirm-pending-registration
is required before creating one ModelPackage. NEVER approves/promotes a model,
updates a ModelPackage, deploys a model, or writes to S3/GitHub.

Registry package is intentionally model-only (no InferenceSpecification image),
mirroring the existing V1 registry contract. The CT-4 comparison has NO labels,
so no ModelQuality/WAPE metrics are attached to this new registration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/sagemaker/ct_registry_v1.json'
RECORD = ROOT / 'artifacts/aws/sagemaker_registry/challenger_ct_v2/registry_record.json'


def require_equal(value, expected, label):
    if value != expected:
        raise ValueError(f'{label}: expected {expected!r}, got {value!r}')


def require_false(document, field, label):
    require_equal(document.get(field), False, f'{label}.{field}')


def read_json_s3(s3, bucket, key):
    body = s3.get_object(Bucket=bucket, Key=key)['Body'].read()
    return json.loads(body), hashlib.sha256(body).hexdigest()


def sha256_s3_object(s3, bucket, key):
    h = hashlib.sha256()
    body = s3.get_object(Bucket=bucket, Key=key)['Body']
    with body:
        for block in body.iter_chunks(chunk_size=1024 * 1024):
            if block:
                h.update(block)
    return h.hexdigest()


def check_job(sm, cfg, job_name, prefix):
    response = sm.describe_processing_job(ProcessingJobName=job_name)
    require_equal(response['ProcessingJobStatus'], 'Completed', f'{job_name}.status')
    expected_arn = (f"arn:aws:sagemaker:{cfg['region']}:{cfg['account_id']}:"
                    f'processing-job/{job_name}')
    require_equal(response['ProcessingJobArn'], expected_arn, f'{job_name}.arn')
    expected_output = f"s3://{cfg['bucket']}/{prefix}"
    uris = [o['S3Output']['S3Uri'] for o in response['ProcessingOutputConfig']['Outputs']]
    if expected_output not in uris:
        raise ValueError(f'{job_name} outputs do not include expected {expected_output}: {uris}')
    return expected_arn


def check_training_report(report, cfg):
    expected = {
        'status': 'PASSED', 'mode': 'full',
        'model_role': 'challenger_unapproved',
        'training_cutoff': cfg['training_cutoff'],
        'dataset_rows_total': 2_543_640, 'fit_rows': 2_543_640,
        'new_label_rows': 44_550, 'sampled_forecast_origins': 106,
        'features': 31, 'n_estimators': 3200,
        'xgboost_version': '3.4.1',
        'training_policy': 'last_2_years',
        'evaluation_type': 'training_integrity_only',
        'input_manifest_sha256': cfg['ct2_manifest_sha256'],
        'model_archive_size_bytes': cfg['challenger_model_size_bytes'],
        'model_archive_sha256': cfg['challenger_model_sha256'],
    }
    for field, value in expected.items():
        require_equal(report.get(field), value, f'CT3.{field}')
    require_false(report, 'has_new_out_of_sample_metric', 'CT3')
    require_false(report, 'promotion_eligible', 'CT3')
    if report.get('champion_v1_unchanged') is not True:
        raise ValueError('CT3 did not certify champion V1 unchanged')


def check_inference_report(report, cfg, training_report_sha):
    expected = {
        'status': 'PASSED',
        'validation_type': 'same_snapshot_unlabeled_model_comparison',
        'forecast_origin': cfg['forecast_origin'],
        'first_target_date': '2017-08-16', 'last_target_date': '2017-08-29',
        'forecast_rows': 24_948, 'unique_series': 1_782,
        'feature_count': 31, 'horizons': list(range(1, 15)),
        'model_champion_role': 'frozen_v1',
        'model_challenger_role': 'v2_unapproved',
        'evaluated_labels': 0,
        'champion_model_sha256': cfg['champion_v1_model_sha256'],
        'challenger_model_sha256': cfg['challenger_model_sha256'],
        'features_sha256': cfg['features_sha256'],
        'training_report_sha256': training_report_sha,
        'comparison_parquet_sha256': cfg['ct4_comparison_sha256'],
        'xgboost_version': '3.4.1',
    }
    for field, value in expected.items():
        require_equal(report.get(field), value, f'CT4.{field}')
    require_false(report, 'has_new_out_of_sample_metric', 'CT4')
    require_false(report, 'promotion_eligible', 'CT4')
    for key in ('v1_prediction_sum', 'v2_prediction_sum',
                'mean_absolute_prediction_delta', 'mean_signed_prediction_delta',
                'fraction_predictions_changed_gt_0_01'):
        val = report.get(key)
        if not isinstance(val, (int, float)) or not __import__('math').isfinite(val):
            raise ValueError(f'CT4.{key} is missing or invalid')
    if report['v1_prediction_sum'] <= 0 or report['v2_prediction_sum'] <= 0:
        raise ValueError('CT4 prediction volumes must be positive')
    if not 0 <= report['fraction_predictions_changed_gt_0_01'] <= 1:
        raise ValueError('CT4 changed prediction fraction outside [0,1]')


def check_ct2_manifest(manifest, cfg):
    expected = {
        'status': 'PASSED', 'cutoff': cfg['training_cutoff'],
        'training_policy': 'last_2_years', 'training_origin_count': 106,
        'training_rows': 2_543_640, 'last_origin': '2017-08-14',
    }
    for field, value in expected.items():
        require_equal(manifest.get(field), value, f'CT2.{field}')
    if len(manifest.get('parts', [])) != 106:
        raise ValueError('CT2 part inventory is not 106')
    if sum(part['rows'] for part in manifest['parts']) != 2_543_640:
        raise ValueError('CT2 row inventory does not sum to expected population')


def registration_token(group, model_sha, train_sha, inference_sha):
    value = '\0'.join((group, model_sha, train_sha, inference_sha))
    return hashlib.sha256(value.encode('utf-8')).hexdigest()[:36]


def verify_champion_package(sm, cfg):
    record = sm.describe_model_package(ModelPackageName=cfg['champion_v1_package_arn'])
    require_equal(record['ModelPackageArn'], cfg['champion_v1_package_arn'], 'V1.arn')
    require_equal(record['ModelPackageGroupName'], cfg['model_package_group_name'], 'V1.group')
    require_equal(record['ModelPackageVersion'], 1, 'V1.version')
    require_equal(record.get('SourceUri'),
                  f"s3://{cfg['bucket']}/{cfg['champion_v1_model_key']}", 'V1.source')
    require_equal(record.get('CustomerMetadataProperties', {}).get('ModelSHA256'),
                  cfg['champion_v1_model_sha256'], 'V1.model_sha256')
    return record


def matching_existing_package(sm, cfg, attrs):
    group = cfg['model_package_group_name']
    packages = sm.get_paginator('list_model_packages').paginate(
        ModelPackageGroupName=group, ModelPackageType='Versioned')
    for page in packages:
        for candidate in page.get('ModelPackageSummaryList', []):
            desc = sm.describe_model_package(ModelPackageName=candidate['ModelPackageArn'])
            metadata = desc.get('CustomerMetadataProperties', {})
            if metadata.get('ModelSHA256') != attrs['ModelSHA256']:
                continue
            # For the same bytes, different lineage must be investigated, not hidden.
            for key in ('TrainingReportSHA256', 'InferenceReportSHA256'):
                require_equal(metadata.get(key), attrs[key], f'already_registered.{key}')
            require_equal(desc.get('SourceUri'), attrs['RegistryModelS3URI'],
                          'already_registered.source')
            return desc
    return None


def validate_registry_source_uri(uri):
    r"""ModelPackage.SourceUri forbids Unicode symbol characters such as '='.

    See AWS CreateModelPackage SourceUri pattern:
    [\p{L}\p{M}\p{Z}\p{N}\p{P}]{0,1024}
    """
    if not isinstance(uri, str) or not uri.startswith('s3://') or len(uri) > 1024:
        raise ValueError('Registry SourceUri must be a valid S3 URI of <=1024 chars')
    forbidden = sorted(set(char for char in uri
                           if unicodedata.category(char)[0] not in 'LMZNP'))
    if forbidden:
        raise ValueError('Registry SourceUri has AWS-forbidden characters: '
                         + repr(forbidden))


def validate_request_shape(request):
    """Preflight common CreateModelPackage constraints before any write."""
    validate_registry_source_uri(request['SourceUri'])
    if request['ModelApprovalStatus'] != 'PendingManualApproval':
        raise ValueError('CT-5 requires PendingManualApproval')
    if 'ModelMetrics' in request or 'InferenceSpecification' in request:
        raise ValueError('CT-5 must not publish invented metrics or inference image')
    attrs = request['CustomerMetadataProperties']
    if not 1 <= len(attrs) <= 50:
        raise ValueError('Invalid CustomerMetadataProperties count')
    allowed_meta_punctuation = set('_.:/=+-@')
    for key, value in attrs.items():
        for label, item, limit in [('metadata key', key, 128),
                                   ('metadata value', value, 256)]:
            if not isinstance(item, str) or not 1 <= len(item) <= limit:
                raise ValueError(f'{label} length invalid: {key}')
            if any(char not in allowed_meta_punctuation and
                   unicodedata.category(char)[0] not in 'LZN'
                   for char in item):
                raise ValueError(f'{label} has forbidden character: {key}')
    token = request['ClientToken']
    if not isinstance(token, str) or not 1 <= len(token) <= 36 or not all(
            c.isascii() and (c.isalnum() or c == '-') for c in token):
        raise ValueError('Invalid idempotency ClientToken')


def build_request(cfg, attrs):
    if cfg['approval_status'] != 'PendingManualApproval':
        raise ValueError('Only PendingManualApproval can be registered in CT-5')
    if cfg['task'] != 'REGRESSION':
        raise ValueError('Unexpected task')
    # ModelMetrics is deliberately OMITTED: labels 2017-08-16..29 do not exist.
    # InferenceSpecification omitted to match the existing nondeployable V1 package.
    return {
        'ModelPackageGroupName': cfg['model_package_group_name'],
        'ModelPackageDescription': (
            'CT challenger V2 trained with labels through 2017-08-15 on '
            'SageMaker Processing. Same-snapshot unlabeled inference passed; '
            'no new out-of-sample WAPE. Manual governance required. '
            'Model-only package, no endpoint inference image.'
        ),
        'ModelApprovalStatus': 'PendingManualApproval',
        'SourceUri': attrs['RegistryModelS3URI'],
        'CustomerMetadataProperties': attrs,
        'Task': 'REGRESSION',
        'ClientToken': registration_token(
            cfg['model_package_group_name'], attrs['ModelSHA256'],
            attrs['TrainingReportSHA256'], attrs['InferenceReportSHA256']),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--execute', action='store_true',
                        help='Create ONE PendingManualApproval model package')
    parser.add_argument('--confirm-pending-registration', action='store_true',
                        help='Required with --execute; never approves/promotes')
    args = parser.parse_args()
    if args.execute and not args.confirm_pending_registration:
        parser.error('--execute requires --confirm-pending-registration')
    if args.confirm_pending_registration and not args.execute:
        parser.error('--confirm-pending-registration requires --execute')
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    require_equal(cfg['region'], 'eu-west-1', 'region')
    require_equal(cfg['account_id'], '634512762949', 'account_id')
    require_equal(cfg['model_package_group_name'],
                  'retail-global-xgboost-forecasting', 'group')
    require_equal(cfg['approval_status'], 'PendingManualApproval', 'approval_status')
    session = boto3.Session(region_name=cfg['region'])
    account = session.client('sts').get_caller_identity()['Account']
    require_equal(account, cfg['account_id'], 'active AWS account')
    s3, sm = session.client('s3'), session.client('sagemaker')
    group = sm.describe_model_package_group(ModelPackageGroupName=cfg['model_package_group_name'])
    require_equal(group['ModelPackageGroupName'], cfg['model_package_group_name'], 'registry group')
    champion = verify_champion_package(sm, cfg)

    job_arns = {
        'ct2': check_job(sm, cfg, cfg['ct2_processing_job'], cfg['ct2_prefix']),
        'ct3': check_job(sm, cfg, cfg['ct3_processing_job'], cfg['ct3_prefix']),
        'ct4': check_job(sm, cfg, cfg['ct4_processing_job'], cfg['ct4_prefix']),
    }
    bucket = cfg['bucket']
    manifest, manifest_sha = read_json_s3(s3, bucket, cfg['ct2_prefix']+'dataset_manifest.json')
    require_equal(manifest_sha, cfg['ct2_manifest_sha256'], 'CT2 manifest sha256')
    check_ct2_manifest(manifest, cfg)
    training, train_sha = read_json_s3(s3, bucket, cfg['ct3_prefix']+'training_report.json')
    require_equal(train_sha, cfg['training_report_sha256'], 'CT3 training report sha256')
    check_training_report(training, cfg)
    inference, inference_sha = read_json_s3(s3, bucket, cfg['ct4_prefix']+'inference_report.json')
    check_inference_report(inference, cfg, train_sha)
    model_key = cfg['ct3_prefix']+'model.tar.gz'
    head = s3.head_object(Bucket=bucket, Key=model_key)
    require_equal(head['ContentLength'], cfg['challenger_model_size_bytes'], 'CT3 model bytes')
    actual_model_sha = sha256_s3_object(s3, bucket, model_key)
    require_equal(actual_model_sha, cfg['challenger_model_sha256'], 'CT3 model sha256')
    comparison_key = cfg['ct4_prefix']+'challenger_comparison.parquet'
    s3.head_object(Bucket=bucket, Key=comparison_key)
    require_equal(sha256_s3_object(s3, bucket, comparison_key),
                  cfg['ct4_comparison_sha256'], 'CT4 comparison parquet sha256')
    # The authoritative CT-3 artifact is immutable; this registration-only
    # S3 copy has the SAME bytes but an AWS SourceUri-compatible key (no '=').
    registry_key = cfg['registry_model_key']
    if not registry_key.startswith(
            'artifacts/global_xgboost/challenger_v2/sha256-'
            + cfg['challenger_model_sha256'] + '/'):
        raise ValueError('Registry model key is not content-addressed as expected')
    registry_uri = f's3://{bucket}/{registry_key}'
    validate_registry_source_uri(registry_uri)
    registry_head = s3.head_object(Bucket=bucket, Key=registry_key)
    require_equal(registry_head['ContentLength'], cfg['challenger_model_size_bytes'],
                  'Registry copy bytes')
    require_equal(sha256_s3_object(s3, bucket, registry_key),
                  cfg['challenger_model_sha256'], 'Registry copy SHA-256')

    model_uri = f's3://{bucket}/{model_key}'
    training_uri = f"s3://{bucket}/{cfg['ct3_prefix']}training_report.json"
    inference_uri = f"s3://{bucket}/{cfg['ct4_prefix']}inference_report.json"
    dataset_uri = f"s3://{bucket}/{cfg['ct2_prefix']}dataset_manifest.json"

    attrs = {
        'ModelSHA256': cfg['challenger_model_sha256'],
        'TrainingReportSHA256': train_sha,
        'InferenceReportSHA256': inference_sha,
        'DatasetManifestSHA256': manifest_sha,
        'ChallengerModelS3URI': model_uri,
        'RegistryModelS3URI': registry_uri,
        'TrainingReportS3URI': training_uri,
        'InferenceAuditS3URI': inference_uri,
        'DatasetManifestS3URI': dataset_uri,
        'TrainingCutoff': cfg['training_cutoff'],
        'ForecastOrigin': cfg['forecast_origin'],
        'ModelRole': 'challenger_v2_unapproved',
        'TrainingExecution': 'sagemaker_processing_cpu',
        'NewOutOfSampleWAPE': 'unavailable',
        'PromotionEligible': 'false',
        'InferenceImageValidated': 'false',
        'CT2ProcessingJob': cfg['ct2_processing_job'],
        'CT3ProcessingJob': cfg['ct3_processing_job'],
        'CT4ProcessingJob': cfg['ct4_processing_job'],
        'V1ChampionModelSHA256': cfg['champion_v1_model_sha256'],
    }
    request = build_request(cfg, attrs)
    validate_request_shape(request)
    existing = matching_existing_package(sm, cfg, attrs)
    changed_pct = 100 * (inference['v2_prediction_sum']/inference['v1_prediction_sum'] - 1)
    print('CT-5 evidence audit: PASSED')
    print('AWS region/account:', cfg['region'], account)
    print('Existing champion version:', champion['ModelPackageArn'],
          'approval:', champion.get('ModelApprovalStatus'))
    print('CT2/CT3/CT4 completed:', ', '.join(job_arns))
    print('CT3 new supervised examples:', training['new_label_rows'])
    print('CT4 predictions/series:', inference['forecast_rows'], '/', inference['unique_series'])
    print('V2 vs V1 prediction volume delta: %.3f%% (UNLABELED)' % changed_pct)
    print('New out-of-sample WAPE: NOT AVAILABLE')
    print('Promotion eligible: NO')
    print('Requested approval status:', request['ModelApprovalStatus'])
    print('Model metrics: deliberately omitted (no future labels)')
    print('InferenceSpecification: deliberately omitted (model-only, nondeployable)')
    print('Challenger SHA-256:', actual_model_sha)
    print('Registry copy SHA-256: VERIFIED')
    print('Registry SourceUri:', registry_uri)
    print('Training report SHA-256:', train_sha)
    print('Inference audit SHA-256:', inference_sha)
    if existing:
        print('Already registered identical challenger:', existing['ModelPackageArn'])
        print('Existing approval status:', existing.get('ModelApprovalStatus'))
        if existing.get('ModelApprovalStatus') != 'PendingManualApproval':
            raise RuntimeError('Existing challenger is no longer pending; manual review required')
        print('No new model version will be created')
        arn = existing['ModelPackageArn']
    elif not args.execute:
        print('DRY RUN: read-only AWS verification; no model package created')
        print(json.dumps(request, indent=2))
        return
    else:
        print('Creating ONE pending model version (no approval or deployment)...', flush=True)
        response = sm.create_model_package(**request)
        arn = response['ModelPackageArn']
        print('New challenger registry ARN:', arn)

    if not args.execute:
        print('DRY RUN: registry unchanged; no local record written')
        return
    # A user-invoked local provenance file; no cloud/GitHub writes here.
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    record = {
        'model_package_arn': arn,
        'model_package_group': cfg['model_package_group_name'],
        'approval_status_requested': 'PendingManualApproval',
        'model_role': 'challenger_v2_unapproved',
        'source_uri': registry_uri,
        'original_training_model_uri': model_uri,
        'model_sha256': actual_model_sha,
        'training_report_uri': training_uri,
        'training_report_sha256': train_sha,
        'inference_audit_uri': inference_uri,
        'inference_audit_sha256': inference_sha,
        'dataset_manifest_uri': dataset_uri,
        'dataset_manifest_sha256': manifest_sha,
        'training_cutoff': cfg['training_cutoff'],
        'forecast_origin': cfg['forecast_origin'],
        'forecast_window': '2017-08-16..2017-08-29',
        'has_new_out_of_sample_metric': False,
        'promotion_eligible': False,
        'training_execution': 'SageMaker Processing (CPU)',
        'inference_image_registered': False,
        'champion_v1_package_arn': cfg['champion_v1_package_arn'],
        'champion_v1_unchanged': True,
        'processing_jobs': job_arns,
        'volume_change_pct_descriptive_only': round(changed_pct, 6),
    }
    RECORD.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print('Local provenance recorded:', RECORD)
    print('CT-5 COMPLETE: challenger registered/reused pending; V1 untouched')


if __name__ == '__main__':
    main()
