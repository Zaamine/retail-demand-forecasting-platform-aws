"""CT-4: dry-run-first SageMaker Processing launcher for V1-V2 inference audit.

ALL external writes (S3 and SageMaker) are disabled unless --execute is passed.
No model training, no Model Registry operations, no changes to V1 forecasts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipelines.processing.validate_ct_inference import validate_metadata

CONFIG_PATH = ROOT/'configs/sagemaker/ct_inference_v1.json'


def s3_input(name: str, uri: str) -> dict:
    return {'InputName': name, 'S3Input': {
        'S3Uri': uri, 'LocalPath': f'/opt/ml/processing/input/{name}',
        'S3DataType': 'S3Prefix', 'S3InputMode': 'File',
    }}


def choose_feature_snapshot(s3, bucket: str, cfg: dict) -> str:
    """Find matching published V1 feature snapshot based on exact SHA-256."""
    prefix = cfg['feature_manifest_search_prefix']
    paginator = s3.get_paginator('list_objects_v2')
    matches = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if not key.endswith('/features.manifest.json'):
                continue
            payload = s3.get_object(Bucket=bucket, Key=key)['Body'].read()
            try:
                doc = json.loads(payload)
            except ValueError:
                continue
            if (doc.get('status') == 'PASSED'
                    and doc.get('forecast_origin') == '2017-08-15'
                    and doc.get('features_sha256') == cfg['feature_sha256']
                    and doc.get('forecast_rows') == 24_948
                    and doc.get('feature_count') == 31):
                features_key = key.removesuffix('features.manifest.json')+'features.parquet'
                try:
                    s3.head_object(Bucket=bucket, Key=features_key)
                except ClientError:
                    raise RuntimeError(f'Missing matching Parquet: {features_key}')
                matches.append((obj['LastModified'], key, features_key))
    if not matches:
        raise RuntimeError('No S3 feature snapshot matches the pinned 2017-08-15 SHA-256')
    matches.sort(key=lambda item: (item[0], item[1]), reverse=True)
    print('Matching validated S3 feature snapshots:', len(matches))
    return matches[0][1].removesuffix('features.manifest.json')


def local_files():
    parts = {
        'validate_ct_inference.py': ROOT/'pipelines/processing/validate_ct_inference.py',
        'forecasting/model_inputs.py': ROOT/'src/forecasting/model_inputs.py',
        'forecasting/__init__.py': ROOT/'src/forecasting/__init__.py',
        'contract/point_forecast_model_contract_v1.json': ROOT/'artifacts/global_xgboost/point_forecast_model_contract_v1.json',
        'categories/categories_v1.json': ROOT/'configs/global_xgboost/categories_v1.json',
        'config/ct_inference_v1.json': CONFIG_PATH,
    }
    for name, path in parts.items():
        if not path.is_file():
            raise FileNotFoundError(f'Missing: {name} ({path})')
    return parts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true', help='Upload code and create a billable Processing Job')
    args = p.parse_args()
    cfg = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    if cfg['processing_instance_type'] != 'ml.t3.xlarge':
        raise RuntimeError('Unexpected instance: only verified ml.t3.xlarge is allowed')
    if cfg['region'] != 'eu-west-1' or cfg['account_id'] != '634512762949':
        raise RuntimeError('AWS region/account configuration mismatch')
    session = boto3.Session(region_name=cfg['region'])
    account = session.client('sts').get_caller_identity()['Account']
    if account != cfg['account_id']:
        raise RuntimeError(f'Unexpected AWS account {account}')
    role = session.client('iam').get_role(RoleName=cfg['sagemaker_role_name'])['Role']['Arn']
    image = (f"{account}.dkr.ecr.{cfg['region']}.amazonaws.com/"
             f"{cfg['ecr_repository']}@{cfg['ecr_image_digest']}")
    s3 = session.client('s3')
    ecr = session.client('ecr')
    ecr_info = ecr.describe_images(repositoryName=cfg['ecr_repository'],
                                   imageIds=[{'imageDigest': cfg['ecr_image_digest']}])
    if ecr_info['imageDetails'][0]['imageDigest'] != cfg['ecr_image_digest']:
        raise RuntimeError('Pinned image digest not found')

    feature_prefix = choose_feature_snapshot(s3, cfg['bucket'], cfg)
    manifest = json.loads(s3.get_object(Bucket=cfg['bucket'], Key=feature_prefix+'features.manifest.json')['Body'].read())
    report_key = cfg['challenger_prefix']+'training_report.json'
    train_report = json.loads(s3.get_object(Bucket=cfg['bucket'], Key=report_key)['Body'].read())
    validate_metadata(manifest, train_report, cfg)
    model_key = cfg['challenger_prefix']+'model.tar.gz'
    object_info = s3.head_object(Bucket=cfg['bucket'], Key=model_key)
    if object_info['ContentLength'] != train_report['model_archive_size_bytes']:
        raise ValueError('Challenger S3 model size differs from training report')
    s3.head_object(Bucket=cfg['bucket'], Key=cfg['champion_model_key'])
    parts = local_files()

    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    run_id = stamp+'-'+uuid.uuid4().hex[:6]
    job = 'ct-inference-'+run_id
    code_prefix = cfg['code_prefix'].rstrip('/')+'/'+run_id+'/'
    output_prefix = cfg['output_prefix'].rstrip('/')+'/'+run_id+'/'
    bucket = cfg['bucket']
    source = lambda key: f's3://{bucket}/{key}'
    request = {
        'ProcessingJobName': job,
        'RoleArn': role,
        'AppSpecification': {
            'ImageUri': image,
            'ContainerEntrypoint': ['python', '-u', '/opt/ml/processing/input/code/validate_ct_inference.py'],
        },
        'ProcessingInputs': [
            s3_input('code', source(code_prefix)),
            s3_input('features', source(feature_prefix+'features.parquet')),
            s3_input('feature-manifest', source(feature_prefix+'features.manifest.json')),
            s3_input('champion', source(cfg['champion_model_key'])),
            s3_input('challenger', source(model_key)),
            s3_input('training-report', source(report_key)),
        ],
        'ProcessingOutputConfig': {'Outputs': [{
            'OutputName': 'audit',
            'S3Output': {
                'S3Uri': source(output_prefix),
                'LocalPath': '/opt/ml/processing/output',
                'S3UploadMode': 'EndOfJob',
            },
        }]},
        'ProcessingResources': {'ClusterConfig': {
            'InstanceCount': 1,
            'InstanceType': cfg['processing_instance_type'],
            'VolumeSizeInGB': cfg['volume_size_gb'],
        }},
        'StoppingCondition': {'MaxRuntimeInSeconds': cfg['max_runtime_seconds']},
        'Tags': [
            {'Key': 'Project', 'Value': 'retail-demand-forecasting'},
            {'Key': 'Stage', 'Value': 'ct-v2-inference-audit'},
            {'Key': 'TrainingCutoff', 'Value': '2017-08-15'},
        ],
    }
    print('Job:', job)
    print('Instance:', cfg['processing_instance_type'])
    print('Features source:', source(feature_prefix))
    print('Features SHA-256:', cfg['feature_sha256'])
    print('V1 model SHA-256:', cfg['champion_sha256'])
    print('V2 model SHA-256:', cfg['challenger_sha256'])
    print('Output:', source(output_prefix))
    print('Max runtime seconds:', cfg['max_runtime_seconds'])
    if not args.execute:
        print('DRY RUN: no S3 uploads, no SageMaker job, no registry mutation')
        print(json.dumps(request, indent=2))
        return
    # Immutable-by-convention job-specific inputs and outputs; fail on collision.
    for prefix in (code_prefix, output_prefix):
        if s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1).get('KeyCount', 0):
            raise FileExistsError('S3 run prefix is already populated: '+prefix)
    for name, local in parts.items():
        s3.upload_file(str(local), bucket, code_prefix+name)
    result = session.client('sagemaker').create_processing_job(**request)
    print('Submitted:', result['ProcessingJobArn'])
    print('Check: aws sagemaker describe-processing-job --processing-job-name', job,
          '--region', cfg['region'],
          '--query "{Status:ProcessingJobStatus,Reason:FailureReason}"')
    print('Read report after completion: aws s3 cp',
          source(output_prefix+'inference_report.json'), '-')

if __name__ == '__main__':
    main()
