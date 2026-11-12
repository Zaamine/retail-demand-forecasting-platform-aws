"""Deploy out-of-sample forecasting without training or holdout evaluation."""
import argparse
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/stepfunctions/test_forecast_v2.json"


def bundle_files():
    files = {
        "build_test_features.py":
            ROOT / "pipelines/processing/build_test_features.py",
        "predict_test.py":
            ROOT / "pipelines/processing/predict_test.py",
        "batch_predict.py":
            ROOT / "pipelines/processing/batch_predict.py",
        "contract/point_forecast_model_contract_v1.json":
            ROOT / "artifacts/global_xgboost/"
            "point_forecast_model_contract_v1.json",
        "categories/categories_v1.json":
            ROOT / "configs/global_xgboost/categories_v1.json",
        "test/test.csv":
            ROOT / "data/raw/favorita/test/test.csv",
    }

    for package in ("forecasting", "features"):
        folder = ROOT / "src" / package

        if not folder.is_dir():
            raise FileNotFoundError(folder)

        for path in sorted(folder.rglob("*.py")):
            name = path.relative_to(ROOT / "src").as_posix()
            files[name] = path

    return files


def mount(name, s3uri=None, expression=None):
    if (s3uri is None) == (expression is None):
        raise ValueError("Exactly one S3 locator required")

    item = {
        "LocalPath": f"/opt/ml/processing/input/{name}",
        "S3DataType": "S3Prefix",
        "S3InputMode": "File",
    }

    item["S3Uri" if s3uri else "S3Uri.$"] = (
        s3uri or expression
    )

    return {"InputName": name, "S3Input": item}


def expr(template):
    return f"States.Format('{template}', $.run_id)"


def processing(step, role, image, inputs, output_prefix, instance, entry):
    return {
        "ProcessingJobName.$": expr(f"retail-v2-{step}-{{}}"),
        "RoleArn": role,
        "AppSpecification": {
            "ImageUri": image,
            "ContainerEntrypoint": [
                "python",
                f"/opt/ml/processing/input/code/{entry}",
            ],
        },
        "ProcessingInputs": inputs,
        "ProcessingOutputConfig": {
            "Outputs": [{
                "OutputName": step,
                "S3Output": {
                    "LocalPath": "/opt/ml/processing/output",
                    "S3UploadMode": "EndOfJob",
                    "S3Uri.$": expr(output_prefix + "{}/"),
                },
            }]
        },
        "ProcessingResources": {
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": instance,
                "VolumeSizeInGB": 10,
            }
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": 900},
    }


def definition(config, image, code_prefix, role):
    bucket = config["bucket"]
    prefix = config["prefix"].rstrip("/")
    code = f"s3://{bucket}/{code_prefix}/"

    test = code + "test/test.csv"
    contract = (
        code + "contract/point_forecast_model_contract_v1.json"
    )
    categories = code + "categories/categories_v1.json"

    features = f"s3://{bucket}/{prefix}/features/"
    predictions = f"s3://{bucket}/{prefix}/predictions/"
    model = (
        f"s3://{bucket}/artifacts/global_xgboost/v1/model.tar.gz"
    )

    common = [
        mount("code", s3uri=code),
        mount("contract", s3uri=contract),
        mount("categories", s3uri=categories),
        mount("test", s3uri=test),
    ]

    first = processing(
        "feat",
        role,
        image,
        common + [
            mount(
                "sales",
                s3uri=f"s3://{bucket}/curated/stg_sales/",
            )
        ],
        features,
        "ml.t3.xlarge",
        "build_test_features.py",
    )

    second = processing(
        "pred",
        role,
        image,
        common + [
            mount("model", s3uri=model),
            mount(
                "features",
                expression=expr(features + "{}/features.parquet"),
            ),
            mount(
                "manifest",
                expression=expr(
                    features + "{}/features.manifest.json"
                ),
            ),
        ],
        predictions,
        "ml.t3.medium",
        "predict_test.py",
    )

    source = (
        f"{bucket}/{prefix}/predictions/"
        "{}/forecast_predictions.parquet"
    )
    destination = (
        "forecasts/published/v2/"
        f"origin={config['origin']}/"
        "{}/forecast_predictions.parquet"
    )

    return {
        "Comment": "Unlabeled test forecasting with frozen champion V1",
        "StartAt": "BuildOutOfSampleFeatures",
        "TimeoutSeconds": 3600,
        "States": {
            "BuildOutOfSampleFeatures": {
                "Type": "Task",
                "Resource": (
                    "arn:aws:states:::sagemaker:"
                    "createProcessingJob.sync"
                ),
                "Parameters": first,
                "ResultPath": "$.feature_result",
                "Next": "GenerateAndValidateTestForecasts",
            },
            "GenerateAndValidateTestForecasts": {
                "Type": "Task",
                "Resource": (
                    "arn:aws:states:::sagemaker:"
                    "createProcessingJob.sync"
                ),
                "Parameters": second,
                "ResultPath": "$.prediction_result",
                "Next": "PublishUnlabeledForecasts",
            },
            "PublishUnlabeledForecasts": {
                "Type": "Task",
                "Resource": (
                    "arn:aws:states:::aws-sdk:s3:copyObject"
                ),
                "Parameters": {
                    "Bucket": bucket,
                    "CopySource.$": expr(source),
                    "Key.$": expr(destination),
                },
                "End": True,
            },
        },
    }


def set_iam(iam, config, sm_arn):
    bucket = config["bucket"]
    prefix = config["prefix"].rstrip("/")
    account = config["account_id"]
    region = config["region"]

    sage_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject"],
                "Resource": f"arn:aws:s3:::{bucket}/{prefix}/*",
            },
            {
                "Effect": "Allow",
                "Action": "s3:ListBucket",
                "Resource": f"arn:aws:s3:::{bucket}",
                "Condition": {
                    "StringLike": {
                        "s3:prefix": [
                            f"{prefix}/*",
                            "curated/stg_sales/*",
                        ]
                    }
                },
            },
        ],
    }

    iam.put_role_policy(
        RoleName=config["sagemaker_role_name"],
        PolicyName="RetailForecastOutOfSampleProcessingV2",
        PolicyDocument=json.dumps(sage_policy),
    )

    workflow_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "sagemaker:CreateProcessingJob",
                    "sagemaker:DescribeProcessingJob",
                    "sagemaker:StopProcessingJob",
                    "sagemaker:AddTags",
                    "sagemaker:ListTags",
                ],
                "Resource": (
                    f"arn:aws:sagemaker:{region}:{account}:"
                    "processing-job/retail-v2-*"
                ),
            },
            {
                "Effect": "Allow",
                "Action": "iam:PassRole",
                "Resource": sm_arn,
                "Condition": {
                    "StringEquals": {
                        "iam:PassedToService":
                            "sagemaker.amazonaws.com"
                    }
                },
            },
            {
                "Effect": "Allow",
                "Action": [
                    "events:PutRule",
                    "events:PutTargets",
                    "events:DescribeRule",
                ],
                "Resource": (
                    f"arn:aws:events:{region}:{account}:rule/"
                    "StepFunctionsGetEventsForSageMakerProcessingJobsRule"
                ),
            },
            {
                "Effect": "Allow",
                "Action": "s3:GetObject",
                "Resource": (
                    f"arn:aws:s3:::{bucket}/{prefix}/predictions/*"
                ),
            },
            {
                "Effect": "Allow",
                "Action": "s3:PutObject",
                "Resource": (
                    f"arn:aws:s3:::{bucket}/forecasts/published/v2/*"
                ),
            },
        ],
    }

    iam.put_role_policy(
        RoleName=config["state_role_name"],
        PolicyName="RetailForecastOutOfSampleWorkflowV2",
        PolicyDocument=json.dumps(workflow_policy),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--deploy", action="store_true")
    modes.add_argument("--start", action="store_true")
    args = parser.parse_args()

    config = json.loads(CONFIG.read_text(encoding="utf-8"))

    if config["origin"] != "2017-08-15":
        raise ValueError("V2 is pinned to the Kaggle test origin")

    session = boto3.Session(region_name=config["region"])
    account = session.client("sts").get_caller_identity()["Account"]

    if account != config["account_id"]:
        raise ValueError("Unexpected AWS account")

    sfn = session.client("stepfunctions")
    state_arn = (
        f"arn:aws:states:{config['region']}:{account}:"
        f"stateMachine:{config['state_machine_name']}"
    )

    if args.start:
        state = sfn.describe_state_machine(
            stateMachineArn=state_arn
        )

        if state["status"] != "ACTIVE":
            raise RuntimeError("State machine is not ACTIVE")

        run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            + "-"
            + uuid.uuid4().hex[:6]
        )

        result = sfn.start_execution(
            stateMachineArn=state_arn,
            name=f"test-forecast-{run_id}",
            input=json.dumps({"run_id": run_id}),
        )

        print("Execution ARN:", result["executionArn"])
        print("Run ID:", run_id)
        print(
            "Expected S3 output IF SUCCEEDED:",
            f"s3://{config['bucket']}/forecasts/published/v2/"
            f"origin=2017-08-15/{run_id}/"
            "forecast_predictions.parquet",
        )
        return

    files = bundle_files()
    digest = hashlib.sha256()

    for name, path in sorted(files.items()):
        if not path.is_file():
            raise FileNotFoundError(path)

        digest.update(
            name.encode() + b"\0" + path.read_bytes()
        )

    bundle_sha = digest.hexdigest()[:24]
    code_prefix = (
        f"{config['prefix'].rstrip('/')}/"
        f"inputs/{bundle_sha}/code"
    )

    image = session.client("ecr").describe_images(
        repositoryName=config["ecr_repository"],
        imageIds=[{"imageTag": config["image_tag"]}],
    )["imageDetails"][0]

    image_uri = (
        f"{account}.dkr.ecr.{config['region']}.amazonaws.com/"
        f"{config['ecr_repository']}@{image['imageDigest']}"
    )

    iam = session.client("iam")
    sage_arn = iam.get_role(
        RoleName=config["sagemaker_role_name"]
    )["Role"]["Arn"]
    state_role_arn = iam.get_role(
        RoleName=config["state_role_name"]
    )["Role"]["Arn"]

    flow = definition(
        config, image_uri, code_prefix, sage_arn
    )
    flow_json = json.dumps(flow, indent=2)

    validation = sfn.validate_state_machine_definition(
        definition=flow_json,
        type="STANDARD",
    )

    print("Definition:", validation["result"])

    if validation["result"] != "OK":
        print(json.dumps(
            validation.get("diagnostics", []),
            indent=2,
        ))
        raise ValueError("Invalid Step Functions definition")

    print("Code bundle:", bundle_sha)
    print("Image:", image_uri)
    print(
        "Stages: cloud features -> "
        "out-of-sample inference -> S3 publish"
    )

    if args.dry_run:
        print(
            "DRY RUN: no S3 uploads, IAM updates, "
            "deployment or jobs"
        )
        return

    s3 = session.client("s3")

    for name, path in files.items():
        s3.upload_file(
            str(path),
            config["bucket"],
            f"{code_prefix}/{name}",
        )

    set_iam(iam, config, sage_arn)

    destination = (
        ROOT / "artifacts/aws/stepfunctions/"
        "test_forecast_v2.asl.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        flow_json + "\n",
        encoding="utf-8",
    )

    try:
        sfn.describe_state_machine(
            stateMachineArn=state_arn
        )
    except sfn.exceptions.StateMachineDoesNotExist:
        response = sfn.create_state_machine(
            name=config["state_machine_name"],
            definition=flow_json,
            roleArn=state_role_arn,
            type="STANDARD",
        )
        print("Created:", response["stateMachineArn"])
    else:
        sfn.update_state_machine(
            stateMachineArn=state_arn,
            definition=flow_json,
            roleArn=state_role_arn,
        )
        print("Updated:", state_arn)

    print("DEPLOYED: no Processing Jobs started")


if __name__ == "__main__":
    main()
