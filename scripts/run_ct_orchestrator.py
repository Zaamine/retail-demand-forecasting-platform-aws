"""Manual CT-6C execution launcher. Default: read-only preview; NO schedule.

--execute --confirm-run starts ONE Standard Step Functions execution. Its only
cloud processing is checkpoint/Glue READINESS; READY is deliberately BLOCKED.
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG = ROOT / 'configs/sagemaker/ct_orchestration_v1.json'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=CONFIG)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--confirm-run',action='store_true')
    args=p.parse_args()
    if args.execute and not args.confirm_run:
        p.error('--execute requires --confirm-run')
    c=json.loads(args.config.read_text(encoding='utf-8'))
    session=boto3.Session(region_name=c['region'])
    account=session.client('sts').get_caller_identity()['Account']
    if account != c['account_id']:
        raise RuntimeError('Account mismatch')
    arn=f"arn:aws:states:{c['region']}:{account}:stateMachine:{c['state_machine_name']}"
    machine=session.client('stepfunctions').describe_state_machine(stateMachineArn=arn)
    if machine['type']!='STANDARD':
        raise RuntimeError('Unexpected state machine type')
    definition=json.loads(machine['definition'])
    # Refuse older/modified workflow, including any READY branch that launches jobs.
    from scripts.ct_orchestration_lib import asl_definition, code_prefix
    image=(f"{account}.dkr.ecr.{c['region']}.amazonaws.com/"
           f"{c['ecr_repository']}@{c['ecr_image_digest']}")
    role=f"arn:aws:iam::{account}:role/{c['sagemaker_role_name']}"
    code_uri=f"s3://{c['bucket']}/{code_prefix(c)}"
    expected=asl_definition(c,image,role,code_uri)
    if definition!=expected:
        raise RuntimeError('Deployed workflow differs from verified fail-closed definition')
    name='ct6c-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]
    print('State machine:',arn)
    print('New execution:',name)
    print('Processing:',c['processing_instance_type'],'max',c['max_runtime_seconds'],'seconds')
    print('READY is blocked (no training / registration)')
    if not args.execute:
        print('DRY RUN: no execution started, no AWS writes')
        return
    out=session.client('stepfunctions').start_execution(
        stateMachineArn=arn,
        name=name,
        input=json.dumps({'trigger':'manual','safety_mode':'readiness_only'}),
    )
    print('STARTED:',out['executionArn'])
    print('Check:', 'aws stepfunctions describe-execution --execution-arn',
          out['executionArn'], '--region',c['region'],
          '--query "{Status:status,Output:output,Error:error,Cause:cause}"')


if __name__=='__main__':
    main()
