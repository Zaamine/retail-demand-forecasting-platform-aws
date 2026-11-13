"""Dry-run-first deployment of CT-6C checkpoint-aware Step Functions gate.

Only --execute --confirm-deploy may create S3 code, DynamoDB table, IAM
policies/role, and new Step Functions state machine. NEVER schedules EventBridge,
starts a Processing/Training Job, registers models, or changes a checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from urllib.parse import unquote

import boto3
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.ct_orchestration_lib import (
    CONFIG, asl_definition, canonical, code_assets, code_prefix, role_policies,
)


def is_missing(exc, *names):
    return exc.response.get('Error', {}).get('Code') in set(names)


def check_config(c):
    if c['account_id'] != '634512762949' or c['region'] != 'eu-west-1':
        raise ValueError('Unapproved CT6C account / region')
    if c['processing_instance_type'] != 'ml.t3.xlarge':
        raise ValueError('Processing type must match confirmed ml.t3.xlarge quota')
    if c['max_runtime_seconds'] > 900 or c['max_runtime_seconds'] < 60:
        raise ValueError('Unexpected readiness processing runtime cap')
    if c['horizon_days'] != 14 or c['expected_series'] != 1782:
        raise ValueError('Frozen panel contract mismatch')
    if c['first_completed_ct_cutoff'] != '2017-08-15':
        raise ValueError('Bootstrap CT cutoff mismatch')
    if c['expected_source_uri'] != f"s3://{c['bucket']}/curated/stg_sales/":
        raise ValueError('Unexpected curated source')
    if not c['checkpoint_key'].startswith('mlops/continuous_training/v1/state/'):
        raise ValueError('Unexpected checkpoint location')
    if not c['code_prefix'].startswith('mlops/continuous_training/v1/orchestration/'):
        raise ValueError('Unexpected code prefix')
    if not c['output_prefix'].startswith('mlops/continuous_training/v1/orchestration/'):
        raise ValueError('Unexpected output prefix')


def ensure_immutable_s3(s3, bucket, key, body):
    """Create-only; verify same bytes when object already exists."""
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=body, IfNoneMatch='*',
                      ContentType='application/json' if key.endswith('.json') else 'text/x-python')
        return 'CREATED'
    except ClientError as exc:
        if not is_missing(exc, 'PreconditionFailed', '412'):
            raise
    old = s3.get_object(Bucket=bucket, Key=key)['Body'].read()
    if hashlib.sha256(old).digest() != hashlib.sha256(body).digest():
        raise RuntimeError(f'Conflicting existing immutable code asset: s3://{bucket}/{key}')
    return 'REUSED_IDENTICAL'


def ensure_lock_table(ddb, name):
    try:
        table = ddb.describe_table(TableName=name)['Table']
        if (table['KeySchema'] != [{'AttributeName':'LockName','KeyType':'HASH'}]
                or table['BillingModeSummary']['BillingMode'] != 'PAY_PER_REQUEST'):
            raise RuntimeError('Existing DynamoDB table does not match CT lock contract')
        if table['TableStatus'] != 'ACTIVE':
            ddb.get_waiter('table_exists').wait(TableName=name)
        return 'EXISTS'
    except ClientError as exc:
        if not is_missing(exc, 'ResourceNotFoundException'):
            raise
    ddb.create_table(
        TableName=name,
        AttributeDefinitions=[{'AttributeName':'LockName','AttributeType':'S'}],
        KeySchema=[{'AttributeName':'LockName','KeyType':'HASH'}],
        BillingMode='PAY_PER_REQUEST',
        Tags=[{'Key':'Project','Value':'retail-demand-forecasting'},
              {'Key':'Component','Value':'ct6c-lock'}],
    )
    ddb.get_waiter('table_exists').wait(TableName=name)
    return 'CREATED'


def ensure_role(iam, name, account):
    trust = {
        'Version':'2012-10-17',
        'Statement':[{'Effect':'Allow','Principal':{'Service':'states.amazonaws.com'},
                      'Action':'sts:AssumeRole'}],
    }
    try:
        result = iam.get_role(RoleName=name)['Role']
        actual = result['AssumeRolePolicyDocument']
        if isinstance(actual, str):
            actual = json.loads(unquote(actual))
        if 'states.amazonaws.com' not in canonical(actual):
            raise RuntimeError('Existing role trust policy incompatible')
        if result['Arn'] != f'arn:aws:iam::{account}:role/{name}':
            raise RuntimeError('Existing role ARN mismatch')
        return result['Arn'], 'EXISTS'
    except ClientError as exc:
        if not is_missing(exc, 'NoSuchEntity'):
            raise
    role = iam.create_role(RoleName=name,
                           AssumeRolePolicyDocument=canonical(trust),
                           Description='CT-6C read-only readiness orchestration with global DDB lock')
    return role['Role']['Arn'], 'CREATED'


def ensure_inline_policy(iam, role, name, document):
    try:
        current = iam.get_role_policy(RoleName=role, PolicyName=name)['PolicyDocument']
        if isinstance(current, str):
            current = json.loads(unquote(current))
        if canonical(current) != canonical(document):
            raise RuntimeError(f'Existing IAM policy differs: {role}/{name}; refusing overwrite')
        return 'REUSED_IDENTICAL'
    except ClientError as exc:
        if not is_missing(exc, 'NoSuchEntity'):
            raise
    iam.put_role_policy(RoleName=role, PolicyName=name, PolicyDocument=canonical(document))
    return 'CREATED'


def ensure_state_machine(sfn, c, role_arn, definition):
    name = c['state_machine_name']
    arn = f"arn:aws:states:{c['region']}:{c['account_id']}:stateMachine:{name}"
    try:
        item = sfn.describe_state_machine(stateMachineArn=arn)
        if item['type'] != 'STANDARD' or item['roleArn'] != role_arn:
            raise RuntimeError('Existing state machine conflicts with deployment')
        if canonical(json.loads(item['definition'])) != canonical(definition):
            raise RuntimeError('Existing state machine differs; no automatic update allowed')
        return arn, 'REUSED_IDENTICAL'
    except ClientError as exc:
        if not is_missing(exc,'StateMachineDoesNotExist'):
            raise
    result = sfn.create_state_machine(name=name, definition=canonical(definition),
                                      roleArn=role_arn, type='STANDARD',
                                      tags=[{'key':'Project','value':'retail-demand-forecasting'},
                                            {'key':'Stage','value':'ct6c-readiness'}])
    return result['stateMachineArn'], 'CREATED'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=CONFIG)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--confirm-deploy', action='store_true')
    p.add_argument('--aws-validate',action='store_true',help='Optional read-only AWS state machine validator')
    args=p.parse_args()
    if args.execute and not args.confirm_deploy:
        p.error('--execute requires --confirm-deploy')
    c=json.loads(args.config.read_text(encoding='utf-8'))
    check_config(c)
    session=boto3.Session(region_name=c['region'])
    account=session.client('sts').get_caller_identity()['Account']
    if account != c['account_id']:
        raise RuntimeError('AWS account mismatch')
    iam=session.client('iam')
    sm_role=iam.get_role(RoleName=c['sagemaker_role_name'])['Role']['Arn']
    if sm_role != f"arn:aws:iam::{account}:role/{c['sagemaker_role_name']}":
        raise RuntimeError('Unexpected SageMaker execution role')
    ecr=session.client('ecr').describe_images(
        repositoryName=c['ecr_repository'],
        imageIds=[{'imageDigest':c['ecr_image_digest']}],
    )['imageDetails'][0]
    if ecr['imageDigest'] != c['ecr_image_digest']:
        raise RuntimeError('Pinned ECR digest no longer present')
    image=f"{account}.dkr.ecr.{c['region']}.amazonaws.com/{c['ecr_repository']}@{c['ecr_image_digest']}"
    assets=code_assets(c)
    code_key=code_prefix(c)
    code_uri=f"s3://{c['bucket']}/{code_key}"
    definition=asl_definition(c, image, sm_role, code_uri)
    sf_policy, sm_policy=role_policies(c,sm_role)
    role_arn=f"arn:aws:iam::{account}:role/{c['state_machine_role_name']}"
    print('State machine:',c['state_machine_name'])
    print('Global lock DynamoDB:',c['lock_table_name'])
    print('Processing quota type:',c['processing_instance_type'])
    print('Image digest pinned:',image)
    print('Code:',code_uri)
    print('Ready branch: BLOCKED (no CT-2/3/4/5 orchestration yet)')
    print('SKIPPED branch: releases lock and succeeds')
    print('Unexpected failure: retains lock (manual recovery)')
    print('EventBridge schedule: NOT CREATED / NOT MODIFIED')
    if args.aws_validate:
        result=session.client('stepfunctions').validate_state_machine_definition(
            definition=canonical(definition), type='STANDARD', severity='ERROR')
        print('AWS state-machine validation:', result['result'])
        for diagnostic in result.get('diagnostics',[]):
            print(' ',diagnostic)
        if result['result'] != 'OK':
            raise RuntimeError('AWS rejected the ASL definition')
    if not args.execute:
        print('DRY RUN: NO AWS WRITES')
        print(json.dumps({'definition':definition,'stepfunctions_role_policy':sf_policy,
                          'sagemaker_existing_s3_ct_policy_required':'RetailForecastCTDataPrefixV1'},indent=2))
        return
    # Only this explicit branch mutates AWS. It NEVER executes the state machine.
    s3=session.client('s3')
    for name, body in assets.items():
        print('Code asset', name, ensure_immutable_s3(s3,c['bucket'],code_key+name,body))
    ddb=session.client('dynamodb')
    print('DynamoDB lock table:',ensure_lock_table(ddb,c['lock_table_name']))
    actual_role,status=ensure_role(iam,c['state_machine_role_name'],account)
    print('SFN execution role:',status)
    print('SFN inline IAM:',ensure_inline_policy(iam,c['state_machine_role_name'],'RetailForecastCT6CExecution',sf_policy))
    arn,status=ensure_state_machine(session.client('stepfunctions'),c,actual_role,definition)
    print('Step Functions state machine:',status,arn)
    print('DEPLOY COMPLETE. No execution started; EventBridge disabled/unmodified.')


if __name__ == '__main__':
    main()
