"""Preview or launch a one-off immutable SageMaker CT dataset Processing Job.

Safe by default: without --execute, prints the request but uploads nothing.
The job builds labeled examples through 2017-08-15 in a NEW S3 prefix.
"""
import argparse
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/sagemaker/ct_dataset_v1.json'


def make_bundle():
    files = {
        'build_ct_dataset.py': ROOT / 'pipelines/processing/build_ct_dataset.py',
        'contract/point_forecast_model_contract_v1.json': (
            ROOT / 'artifacts/global_xgboost/point_forecast_model_contract_v1.json'
        ),
        'categories/categories_v1.json': (
            ROOT / 'configs/global_xgboost/categories_v1.json'
        ),
    }
    for package in ('features', 'forecasting'):
        for source in sorted((ROOT / 'src' / package).rglob('*.py')):
            files[source.relative_to(ROOT/'src').as_posix()] = source
    for key, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f'{key}: {path}')
    return files


def build_request(config, *, job_name, image_uri, role_arn, code_uri, output_uri):
    return {
        'ProcessingJobName': job_name,
        'RoleArn': role_arn,
        'AppSpecification': {
            'ImageUri': image_uri,
            'ContainerEntrypoint': [
                'python', '/opt/ml/processing/input/code/build_ct_dataset.py'
            ],
            'ContainerArguments': ['--cutoff', config['cutoff']],
        },
        'ProcessingInputs': [
            {
                'InputName': name,
                'S3Input': {
                    'S3Uri': s3_uri,
                    'LocalPath': f'/opt/ml/processing/input/{name}',
                    'S3DataType': 'S3Prefix',
                    'S3InputMode': 'File',
                },
            }
            for name, s3_uri in (
                ('code', code_uri),
                ('sales', f"s3://{config['bucket']}/{config['source_prefix']}"),
            )
        ],
        'ProcessingOutputConfig': {
            'Outputs': [{
                'OutputName': 'dataset',
                'S3Output': {
                    'S3Uri': output_uri,
                    'LocalPath': '/opt/ml/processing/output',
                    'S3UploadMode': 'EndOfJob',
                },
            }],
        },
        'ProcessingResources': {
            'ClusterConfig': {
                'InstanceCount': 1,
                'InstanceType': config['processing_instance_type'],
                'VolumeSizeInGB': config['volume_size_gb'],
            },
        },
        'StoppingCondition': {
            'MaxRuntimeInSeconds': config['max_runtime_seconds'],
        },
        'Tags': [
            {'Key':'Project','Value':'retail-demand-forecasting'},
            {'Key':'Stage','Value':'ct-dataset-build'},
            {'Key':'TrainingCutoff','Value':config['cutoff']},
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true',help='Explicitly permit AWS writes/jobs')
    args = p.parse_args()
    config = json.loads(CONFIG.read_text(encoding='utf-8'))
    report = json.loads((ROOT/config['readiness_report']).read_text())
    if (report.get('status')!='READY'
            or report.get('challenger_training_cutoff') != config['cutoff']
            or report.get('new_labeled_rows') != 24948):
        raise RuntimeError('CT readiness gate is not READY for the requested cutoff')

    session = boto3.Session(region_name=config['region'])
    account = session.client('sts').get_caller_identity()['Account']
    if account != config['account_id']:
        raise RuntimeError('Unexpected AWS account')
    role_arn = session.client('iam').get_role(
        RoleName=config['sagemaker_role_name']
    )['Role']['Arn']
    image = session.client('ecr').describe_images(
        repositoryName=config['ecr_repository'],
        imageIds=[{'imageTag':config['image_tag']}],
    )['imageDetails'][0]
    image_uri = (
        f"{account}.dkr.ecr.{config['region']}.amazonaws.com/"
        f"{config['ecr_repository']}@{image['imageDigest']}"
    )

    files = make_bundle()
    digest = hashlib.sha256()
    for key, path in sorted(files.items()):
        digest.update(key.encode()+b'\x00'+path.read_bytes())
    bundle_hash = digest.hexdigest()[:24]
    code_key = f"{config['code_prefix']}/{bundle_hash}/"
    code_uri = f"s3://{config['bucket']}/{code_key}"

    run_id = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]
    job_name = f'ct-dataset-{run_id}'
    output_uri = (
        f"s3://{config['bucket']}/{config['output_prefix']}/"
        f"cutoff={config['cutoff']}/{run_id}/"
    )
    request = build_request(
        config,job_name=job_name,image_uri=image_uri,
        role_arn=role_arn,code_uri=code_uri,output_uri=output_uri,
    )
    print('Job name:',job_name)
    print('Image (digest-pinned):',image_uri)
    print('Instance:',config['processing_instance_type'])
    print('Maximum runtime seconds:',config['max_runtime_seconds'])
    print('Dataset output:',output_uri)
    print('Source bundle:',code_uri)
    if not args.execute:
        print('DRY RUN: no S3 uploads and no SageMaker job')
        print(json.dumps(request,indent=2))
        return

    s3 = session.client('s3')
    for key,path in files.items():
        s3.upload_file(str(path),config['bucket'],code_key+key)
    sm = session.client('sagemaker')
    response = sm.create_processing_job(**request)
    print('SageMaker job submitted:',response['ProcessingJobArn'])
    print('To check: aws sagemaker describe-processing-job --processing-job-name',job_name,
          '--region',config['region'],'--query "{Status:ProcessingJobStatus,Reason:FailureReason}"')


if __name__=='__main__':
    main()
