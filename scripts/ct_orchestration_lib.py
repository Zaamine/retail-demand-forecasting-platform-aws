"""CT-6C control-plane definitions. Pure builders; no AWS writes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/sagemaker/ct_orchestration_v1.json'
WORKER = ROOT / 'pipelines/processing/ct_workflow_readiness.py'


def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def code_assets(config):
    return {
        'ct_workflow_readiness.py': WORKER.read_bytes(),
        'ct_orchestration_v1.json': (canonical(config) + '\n').encode(),
    }


def code_prefix(config):
    assets = code_assets(config)
    h = hashlib.sha256()
    for name, blob in sorted(assets.items()):
        h.update(name.encode() + b'\x00' + blob)
    return f"{config['code_prefix'].rstrip('/')}/{h.hexdigest()[:24]}/"


def lock_release(config):
    return {
        'Type': 'Task', 'Resource': 'arn:aws:states:::dynamodb:deleteItem',
        'Parameters': {
            'TableName': config['lock_table_name'],
            'Key': {'LockName': {'S': 'ct-global-exclusive'}},
            'ConditionExpression': '#owner = :owner',
            'ExpressionAttributeNames': {'#owner': 'Owner'},
            'ExpressionAttributeValues': {':owner': {'S.$': '$$.Execution.Id'}},
        },
        'ResultPath': None,
        'Next': 'ReturnOutcome',
    }


def asl_definition(config, image_uri, sm_role_arn, source_code_uri):
    """Cloud decision gate. No READY path can start training or mutate registry."""
    bucket = config['bucket']
    base = f"s3://{bucket}/{config['output_prefix'].rstrip('/')}/"
    job = {
        'ProcessingJobName.$': '$.run.job_name',
        'RoleArn': sm_role_arn,
        'AppSpecification': {
            'ImageUri': image_uri,
            'ContainerEntrypoint': [
                'python', '-u', '/opt/ml/processing/input/code/ct_workflow_readiness.py'
            ],
            'ContainerArguments': [
                '--config', '/opt/ml/processing/input/code/ct_orchestration_v1.json',
                '--output-dir', '/opt/ml/processing/output',
            ],
        },
        'ProcessingInputs': [{
            'InputName': 'code',
            'S3Input': {
                'S3Uri': source_code_uri,
                'LocalPath': '/opt/ml/processing/input/code',
                'S3DataType': 'S3Prefix',
                'S3InputMode': 'File',
            },
        }, {
            'InputName': 'checkpoint',
            'S3Input': {
                'S3Uri': f"s3://{bucket}/{config['checkpoint_key']}",
                'LocalPath': '/opt/ml/processing/input/checkpoint',
                'S3DataType':'S3Prefix', 'S3InputMode':'File',
            },
        }, {
            'InputName': 'sales',
            'S3Input': {
                'S3Uri': config['expected_source_uri'],
                'LocalPath': '/opt/ml/processing/input/sales',
                'S3DataType':'S3Prefix', 'S3InputMode':'File',
            },
        }],
        'ProcessingOutputConfig': {'Outputs': [{
            'OutputName': 'readiness',
            'S3Output': {
                'S3Uri.$': f"States.Format('{base}{{}}/', $.run.job_name)",
                'LocalPath': '/opt/ml/processing/output',
                'S3UploadMode': 'EndOfJob',
            },
        }]},
        'ProcessingResources': {'ClusterConfig': {
            'InstanceCount': 1,
            'InstanceType': config['processing_instance_type'],
            'VolumeSizeInGB': config['volume_size_gb'],
        }},
        'StoppingCondition': {'MaxRuntimeInSeconds': config['max_runtime_seconds']},
        'Tags': [
            {'Key': 'Project', 'Value': 'retail-demand-forecasting'},
            {'Key': 'Stage', 'Value': 'ct-checkpoint-readiness'},
        ],
    }
    states = {
        'AcquireGlobalLock': {
            'Type': 'Task',
            'Resource': 'arn:aws:states:::dynamodb:putItem',
            'Parameters': {
                'TableName': config['lock_table_name'],
                'Item': {
                    'LockName': {'S': 'ct-global-exclusive'},
                    'Owner': {'S.$': '$$.Execution.Id'},
                    'CreatedAt': {'S.$': '$$.State.EnteredTime'},
                },
                'ConditionExpression': 'attribute_not_exists(LockName)',
            },
            'ResultPath': None,
            'Catch': [{
                'ErrorEquals': ['DynamoDB.ConditionalCheckFailedException'],
                'ResultPath': '$.lock_error',
                'Next': 'ConcurrentExecutionBlocked',
            }],
            'Next': 'CreateUniqueJobName',
        },
        'ConcurrentExecutionBlocked': {
            'Type': 'Pass',
            'Parameters': {
                'status': 'BLOCKED',
                'reason': 'GLOBAL_CT_LOCK_ALREADY_HELD',
            },
            'End': True,
        },
        'CreateUniqueJobName': {
            'Type': 'Pass',
            'Parameters': {
                'job_name.$': "States.Format('ct-gate-{}', States.UUID())",
            },
            'ResultPath': '$.run',
            'Next': 'VerifyGlueSource',
        },
        'VerifyGlueSource': {
            'Type':'Task',
            'Resource':'arn:aws:states:::aws-sdk:glue:getTable',
            'Parameters': {
                'DatabaseName': config['glue_database'],
                'Name': config['glue_table'],
            },
            'ResultPath':'$.catalog',
            'Next':'VerifyGlueLocation',
        },
        'VerifyGlueLocation': {
            'Type':'Choice',
            'Choices':[
                {'Variable':'$.catalog.Table.StorageDescriptor.Location',
                 'StringEquals':config['expected_source_uri'],
                 'Next':'ReadCheckpointAndCuratedSales'},
                {'Variable':'$.catalog.Table.StorageDescriptor.Location',
                 'StringEquals':config['expected_source_uri'].rstrip('/'),
                 'Next':'ReadCheckpointAndCuratedSales'},
            ],
            'Default':'UnexpectedGlueSourceFailClosed',
        },
        'UnexpectedGlueSourceFailClosed': {
            'Type':'Fail',
            'Error':'GlueSourceDrift',
            'Cause':'Glue source location changed; lock retained for investigation',
        },
        'ReadCheckpointAndCuratedSales': {
            'Type': 'Task',
            'Resource': 'arn:aws:states:::sagemaker:createProcessingJob.sync',
            'Parameters': job,
            'ResultPath': '$.job',
            'TimeoutSeconds': max(1800, config['max_runtime_seconds'] + 600),
            # Fail closed: on failure retain lock; a Processing Job may still be alive.
            'Next': 'ReadS3ReadinessReport',
        },
        'ReadS3ReadinessReport': {
            'Type': 'Task',
            'Resource': 'arn:aws:states:::aws-sdk:s3:getObject',
            'Parameters': {
                'Bucket': bucket,
                'Key.$': f"States.Format('{config['output_prefix'].rstrip('/')}/{{}}/readiness_report.json', $.run.job_name)",
            },
            'ResultSelector': {
                'result.$': 'States.StringToJson($.Body)',
            },
            'ResultPath': '$.readiness',
            'Next': 'ReadLiveCheckpoint',
        },
        'ReadLiveCheckpoint': {
            'Type':'Task',
            'Resource':'arn:aws:states:::aws-sdk:s3:getObject',
            'Parameters':{'Bucket':bucket, 'Key':config['checkpoint_key']},
            'ResultSelector':{'sha256.$':"States.Hash($.Body, 'SHA-256')"},
            'ResultPath':'$.live_checkpoint',
            'Next':'CheckpointStillCurrent',
        },
        'CheckpointStillCurrent': {
            'Type':'Choice',
            'Choices':[{'Variable':'$.live_checkpoint.sha256',
                        'StringEqualsPath':'$.readiness.result.checkpoint_sha256',
                        'Next':'RouteReadinessStatus'}],
            'Default':'ChangedCheckpointFailClosed',
        },
        'ChangedCheckpointFailClosed': {
            'Type':'Fail', 'Error':'CheckpointChanged',
            'Cause':'Checkpoint updated during workflow; lock retained',
        },
        'RouteReadinessStatus': {
            'Type': 'Choice',
            'Choices': [
                {'Variable': '$.readiness.result.status', 'StringEquals': 'SKIPPED', 'Next': 'OutcomeSkipped'},
                {'Variable': '$.readiness.result.status', 'StringEquals': 'BLOCKED', 'Next': 'OutcomeBlocked'},
                {'Variable': '$.readiness.result.status', 'StringEquals': 'READY', 'Next': 'OutcomeReadyBlocked'},
            ],
            'Default': 'UnexpectedReportFailClosed',
        },
        'OutcomeSkipped': {
            'Type': 'Pass',
            'Parameters': {
                'status': 'SKIPPED',
                'reason.$': '$.readiness.result.reason',
                'checkpoint_cutoff.$': '$.readiness.result.checkpoint_cutoff',
                'next_required_labeled_end.$': '$.readiness.result.next_required_labeled_end',
                'processing_job.$': '$.run.job_name',
            },
            'ResultPath': '$.outcome',
            'Next': 'ReleaseGlobalLock',
        },
        'OutcomeBlocked': {
            'Type': 'Pass',
            'Parameters': {
                'status': 'BLOCKED',
                'reason.$': '$.readiness.result.reason',
                'processing_job.$': '$.run.job_name',
            },
            'ResultPath': '$.outcome',
            'Next': 'ReleaseGlobalLock',
        },
        'OutcomeReadyBlocked': {
            'Type': 'Pass',
            'Parameters': {
                'status': 'BLOCKED',
                'reason': 'READY_DATA_BUT_DYNAMIC_CT2_CT5_NOT_YET_CONFIGURED',
                'next_training_cutoff.$': '$.readiness.result.next_training_cutoff',
                'processing_job.$': '$.run.job_name',
            },
            'ResultPath': '$.outcome',
            'Next': 'ReleaseGlobalLock',
        },
        'UnexpectedReportFailClosed': {
            'Type': 'Fail',
            'Error': 'InvalidReadinessStatus',
            'Cause': 'Unknown status. Global lock deliberately retained for investigation.',
        },
        'ReleaseGlobalLock': lock_release(config),
        'ReturnOutcome': {'Type': 'Pass', 'OutputPath': '$.outcome', 'End': True},
    }
    return {
        'Comment': 'CT-6C: checkpoint-aware gate; READY always blocked until dynamic CT2-CT5 is implemented; EventBridge OFF',
        'StartAt': 'AcquireGlobalLock',
        'TimeoutSeconds': max(2400, config['max_runtime_seconds'] + 1200),
        'States': states,
    }


def role_policies(config, sm_role_arn):
    account, region, bucket = config['account_id'], config['region'], config['bucket']
    table_arn = f"arn:aws:dynamodb:{region}:{account}:table/{config['lock_table_name']}"
    sf_policy = {
        'Version': '2012-10-17',
        'Statement': [
            {'Sid': 'LockAccess', 'Effect': 'Allow',
             'Action': ['dynamodb:PutItem','dynamodb:DeleteItem'], 'Resource': table_arn},
            {'Sid':'CreateOnlyCTReadinessJobs','Effect':'Allow',
             'Action':['sagemaker:CreateProcessingJob','sagemaker:DescribeProcessingJob',
                       'sagemaker:StopProcessingJob','sagemaker:AddTags','sagemaker:ListTags'],
             'Resource':f"arn:aws:sagemaker:{region}:{account}:processing-job/ct-gate-*"},
            {'Sid':'PassExistingSageMakerExecutionRole', 'Effect':'Allow',
             'Action':'iam:PassRole', 'Resource':sm_role_arn,
             'Condition':{'StringEquals':{'iam:PassedToService':'sagemaker.amazonaws.com'}}},
            {'Sid':'ValidateCuratedSourceInGlue','Effect':'Allow',
             'Action':'glue:GetTable', 'Resource':[
                 f'arn:aws:glue:{region}:{account}:catalog',
                 f"arn:aws:glue:{region}:{account}:database/{config['glue_database']}",
                 f"arn:aws:glue:{region}:{account}:table/{config['glue_database']}/{config['glue_table']}",
             ]},
            {'Sid':'ReadReadinessReports','Effect':'Allow', 'Action':'s3:GetObject',
             'Resource':[
                 f"arn:aws:s3:::{bucket}/{config['output_prefix'].rstrip('/')}/*",
                 f"arn:aws:s3:::{bucket}/{config['checkpoint_key']}",
             ]},
            {'Sid':'ProcessingJobSyncEventBridge', 'Effect':'Allow',
             'Action':['events:PutRule','events:PutTargets','events:DescribeRule'],
             'Resource':f"arn:aws:events:{region}:{account}:rule/StepFunctionsGetEventsForSageMaker*JobsRule"},
        ],
    }
    # No additional SageMaker permissions: use existing scoped CT S3 role.
    # Retained empty policy object for backwards-compatible test/preview shape.
    sm_policy = None
    return sf_policy, sm_policy
