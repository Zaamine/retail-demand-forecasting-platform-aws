"""Dry-run-first SageMaker Processing launcher for XGBoost challenger retraining.

This runs real CPU XGBoost training in a temporary Processing Job because
SageMaker Training CPU quotas are zero. It is NOT a SageMaker Training Job.
No registry changes, model promotion, or writes to champion V1 are performed.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipelines.processing.train_ct_challenger import check_manifest
CONFIG = ROOT / 'configs/sagemaker/ct_training_v1.json'


def collect_code_files():
    files = {
        'train_ct_challenger.py': ROOT/'pipelines/processing/train_ct_challenger.py',
        'contract/point_forecast_model_contract_v1.json': (
            ROOT/'artifacts/global_xgboost/point_forecast_model_contract_v1.json'),
        'categories/categories_v1.json': ROOT/'configs/global_xgboost/categories_v1.json',
        'forecasting/model_inputs.py': ROOT/'src/forecasting/model_inputs.py',
        'forecasting/__init__.py': ROOT/'src/forecasting/__init__.py',
    }
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f'Bundle file missing: {name} -> {path}')
    return files


def build_request(c, *, name, role_arn, image, code_uri, output_uri, mode):
    def source(mount, uri):
        return {
            'InputName': mount,
            'S3Input': {
                'S3Uri': uri,
                'LocalPath': f'/opt/ml/processing/input/{mount}',
                'S3DataType': 'S3Prefix',
                'S3InputMode': 'File',
            },
        }
    max_runtime = c.get('smoke_max_runtime_seconds', 1200) if mode == 'smoke' else c['max_runtime_seconds']
    return {
        'ProcessingJobName': name,
        'RoleArn': role_arn,
        'AppSpecification': {
            'ImageUri': image,
            'ContainerEntrypoint': [
                'python', '-u', '/opt/ml/processing/input/code/train_ct_challenger.py'
            ],
            'ContainerArguments': ['--mode', mode, '--n-jobs', str(c['xgboost_n_jobs'])],
        },
        'ProcessingInputs': [
            source('code', code_uri),
            source('dataset', c['ct2_dataset_uri']),
        ],
        'ProcessingOutputConfig': {
            'Outputs': [{
                'OutputName': 'challenger',
                'S3Output': {
                    'S3Uri': output_uri,
                    'LocalPath': '/opt/ml/processing/output',
                    'S3UploadMode': 'EndOfJob',
                },
            }]
        },
        'ProcessingResources': {
            'ClusterConfig': {
                'InstanceCount': 1,
                'InstanceType': c['processing_instance_type'],
                'VolumeSizeInGB': c['volume_size_gb'],
            }
        },
        'StoppingCondition': {'MaxRuntimeInSeconds': max_runtime},
        'Tags': [
            {'Key': 'Project', 'Value': 'retail-demand-forecasting'},
            {'Key': 'Stage', 'Value': 'ct-challenger-training'},
            {'Key': 'TrainingCutoff', 'Value': c['cutoff']},
            {'Key': 'TrainingMode', 'Value': mode},
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true',
                        help='Explicitly enable S3 uploads and SageMaker Processing job')
    parser.add_argument('--mode', choices=('smoke', 'full'), default='smoke')
    parser.add_argument('--confirm-full', action='store_true',
                        help='Required alongside --execute --mode full')
    args = parser.parse_args()
    c = json.loads(CONFIG.read_text(encoding='utf-8'))
    if c['processing_instance_type'] != 'ml.t3.xlarge':
        raise RuntimeError('Only the previously verified ml.t3.xlarge Processing quota is allowed')
    if c['cutoff'] != '2017-08-15' or c['account_id'] != '634512762949':
        raise RuntimeError('Not the frozen Week 4B retraining configuration')
    ready = json.loads((ROOT / c['readiness_report']).read_text(encoding='utf-8'))
    if (ready.get('status') != 'READY'
            or ready.get('challenger_training_cutoff') != c['cutoff']
            or ready.get('new_labeled_rows') != 24948):
        raise RuntimeError('CT readiness is not READY for validated 2017-08-15 labels')
    if args.execute and args.mode == 'full' and not args.confirm_full:
        raise RuntimeError('Full 3200-tree cloud training requires --confirm-full')
    session = boto3.Session(region_name=c['region'])
    account = session.client('sts').get_caller_identity()['Account']
    if account != c['account_id']:
        raise RuntimeError('AWS caller account mismatch')
    role = session.client('iam').get_role(RoleName=c['sagemaker_role_name'])['Role']['Arn']
    ecr_repo = session.client('ecr').describe_images(
        repositoryName=c['ecr_repository'],
        imageIds=[{'imageDigest': c['ecr_image_digest']}],
    )['imageDetails'][0]
    if ecr_repo['imageDigest'] != c['ecr_image_digest']:
        raise RuntimeError('ECR image digest changed unexpectedly')
    image = f"{account}.dkr.ecr.{c['region']}.amazonaws.com/{c['ecr_repository']}@{c['ecr_image_digest']}"
    uri = c['ct2_dataset_uri']
    expected_root = f"s3://{c['bucket']}/mlops/continuous_training/v1/datasets/cutoff={c['cutoff']}/"
    if not uri.startswith(expected_root) or not uri.endswith('/'):
        raise RuntimeError('CT-2 source URI outside frozen, versioned dataset prefix')
    key = uri.removeprefix(f"s3://{c['bucket']}/") + 'dataset_manifest.json'
    manifest_bytes = session.client('s3').get_object(Bucket=c['bucket'], Key=key)['Body'].read()
    manifest = json.loads(manifest_bytes)
    contract = json.loads((ROOT/'artifacts/global_xgboost/point_forecast_model_contract_v1.json').read_text())
    check_manifest(manifest, contract)
    code = collect_code_files()
    digest = hashlib.sha256()
    for name, file in sorted(code.items()):
        digest.update(name.encode() + b'\x00' + file.read_bytes())
    sha = digest.hexdigest()[:24]
    code_key = f"{c['code_prefix'].rstrip('/')}/{sha}/"
    code_uri = f"s3://{c['bucket']}/{code_key}"
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    run_id = stamp + '-' + uuid.uuid4().hex[:6]
    job = f'ct-train-{args.mode}-{run_id}'
    output_uri = (f"s3://{c['bucket']}/{c['output_prefix'].rstrip('/')}/"
                  f"cutoff={c['cutoff']}/mode={args.mode}/{run_id}/")
    request = build_request(c, name=job, role_arn=role, image=image,
                            code_uri=code_uri, output_uri=output_uri, mode=args.mode)
    print('CT-2 dataset:', uri)
    print('CT-2 manifest SHA-256:', hashlib.sha256(manifest_bytes).hexdigest())
    print('Job:', job)
    print('Mode:', args.mode)
    print('Number of trees:', 25 if args.mode == 'smoke' else 3200)
    print('Processing instance:', c['processing_instance_type'])
    print('Maximum runtime (seconds):', request['StoppingCondition']['MaxRuntimeInSeconds'])
    print('Pinned ECR image:', image)
    print('Code bundle:', code_uri)
    print('Output:', output_uri)
    if not args.execute:
        print('DRY RUN: no AWS writes or jobs started')
        print(json.dumps(request, indent=2))
        return
    # Never overwrite an existing run's output artifacts.
    output_prefix = output_uri.removeprefix(f"s3://{c['bucket']}/")
    if session.client('s3').list_objects_v2(Bucket=c['bucket'], Prefix=output_prefix,
                                            MaxKeys=1).get('KeyCount', 0):
        raise RuntimeError('Output prefix is already occupied')
    s3 = session.client('s3')
    for name, file in code.items():
        s3.upload_file(str(file), c['bucket'], code_key + name)
    resp = session.client('sagemaker').create_processing_job(**request)
    print('Submitted:', resp['ProcessingJobArn'])
    print('Check: aws sagemaker describe-processing-job --processing-job-name',
          job, '--region', c['region'],
          '--query "{Status:ProcessingJobStatus,Reason:FailureReason}"')


if __name__ == '__main__':
    main()
