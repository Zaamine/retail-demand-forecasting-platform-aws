"""Deploy a curated-S3 -> cloud features -> XGBoost -> S3 workflow.

The initial run is a historical regression rehearsal at 2017-08-01.
It does not retrain the champion or re-evaluate holdout WAPE.
"""

import argparse
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import boto3


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/stepfunctions/forecast_workflow_v1.json"


def file_bundle():
    """Bundle Python packages while preserving their import paths."""
    files = {
        "build_cloud_features.py": (
            ROOT / "pipelines/processing/build_cloud_features.py"
        ),
        "batch_predict.py": (
            ROOT / "pipelines/processing/batch_predict.py"
        ),
        "contract/point_forecast_model_contract_v1.json": (
            ROOT / "artifacts/global_xgboost/"
            "point_forecast_model_contract_v1.json"
        ),
        "categories/categories_v1.json": (
            ROOT / "configs/global_xgboost/categories_v1.json"
        ),
    }

    source_root = ROOT / "src"

    for package in ("forecasting", "features"):
        package_dir = source_root / package

        if not package_dir.is_dir():
            raise FileNotFoundError(package_dir)

        for source_file in sorted(package_dir.rglob("*.py")):
            relative_path = source_file.relative_to(source_root)
            files[relative_path.as_posix()] = source_file

    return files


def mount(name, uri=None, expression=None):
    """Mount a static or dynamically resolved S3 input."""
    if (uri is None) == (expression is None):
        raise ValueError("Provide exactly one S3 locator")

    source = {
        "LocalPath": f"/opt/ml/processing/input/{name}",
        "S3DataType": "S3Prefix",
        "S3InputMode": "File",
    }

    source["S3Uri" if uri is not None else "S3Uri.$"] = (
        uri if uri is not None else expression
    )

    return {
        "InputName": name,
        "S3Input": source,
    }


def formatted(template):
    """Resolve the per-execution ID within AWS Step Functions."""
    return f"States.Format('{template}', $.run_id)"


def build_definition(config, image_uri, code_prefix, role_arn):
    """Build a three-stage workflow with explicit data dependencies."""
    bucket = config["bucket"]
    prefix = config["prefix"].rstrip("/")

    code_uri = f"s3://{bucket}/{code_prefix}/"

    contract_uri = (
        f"{code_uri}contract/point_forecast_model_contract_v1.json"
    )

    categories_uri = f"{code_uri}categories/categories_v1.json"

    reference_uri = (
        f"s3://{bucket}/forecasts/global_xgboost/v1/"
        "final_holdout_forecast.parquet"
    )

    model_uri = (
        f"s3://{bucket}/artifacts/global_xgboost/v1/model.tar.gz"
    )

    feature_prefix = f"s3://{bucket}/{prefix}/features/"
    prediction_prefix = f"s3://{bucket}/{prefix}/predictions/"

    feature_inputs = [
        mount("code", uri=code_uri),
        mount("sales", uri=f"s3://{bucket}/curated/stg_sales/"),
        mount("contract", uri=contract_uri),
        mount("categories", uri=categories_uri),
        mount("reference", uri=reference_uri),
    ]

    prediction_inputs = [
        mount("code", uri=code_uri),
        mount("contract", uri=contract_uri),
        mount("categories", uri=categories_uri),
        mount("reference", uri=reference_uri),
        mount("model", uri=model_uri),
        mount(
            "features",
            expression=formatted(
                feature_prefix + "{}/features.parquet"
            ),
        ),
        mount(
            "manifest",
            expression=formatted(
                feature_prefix + "{}/features.manifest.json"
            ),
        ),
    ]

    def processing_parameters(
        step,
        inputs,
        output_prefix,
        instance,
        entrypoint,
        arguments,
    ):
        return {
            "ProcessingJobName.$": formatted(
                f"retail-v1-orch-{step}-{{}}"
            ),
            "RoleArn": role_arn,
            "AppSpecification": {
                "ImageUri": image_uri,
                "ContainerEntrypoint": [
                    "python",
                    f"/opt/ml/processing/input/code/{entrypoint}",
                ],
                "ContainerArguments": arguments,
            },
            "ProcessingInputs": inputs,
            "ProcessingOutputConfig": {
                "Outputs": [
                    {
                        "OutputName": step,
                        "S3Output": {
                            "LocalPath": "/opt/ml/processing/output",
                            "S3UploadMode": "EndOfJob",
                            "S3Uri.$": formatted(
                                output_prefix + "{}/"
                            ),
                        },
                    }
                ]
            },
            "ProcessingResources": {
                "ClusterConfig": {
                    "InstanceCount": 1,
                    "InstanceType": instance,
                    "VolumeSizeInGB": 10,
                }
            },
            "StoppingCondition": {
                "MaxRuntimeInSeconds": 900
            },
        }

    feature_parameters = processing_parameters(
        "feat",
        feature_inputs,
        feature_prefix,
        "ml.t3.xlarge",
        "build_cloud_features.py",
        ["--origin", config["forecast_origin"]],
    )

    prediction_parameters = processing_parameters(
        "pred",
        prediction_inputs,
        prediction_prefix,
        "ml.t3.medium",
        "batch_predict.py",
        [
            "--expected-rows", "24948",
            "--atol", "0.0001",
            "--rtol", "1e-6",
        ],
    )

    source = (
        f"{bucket}/{prefix}/predictions/"
        "{}/batch_forecasts.parquet"
    )

    destination = (
        f"forecasts/published/v1/"
        f"origin={config['forecast_origin']}/"
        "{}/batch_forecasts.parquet"
    )

    return {
        "Comment": (
            "Curated sales -> cloud features -> "
            "batch inference -> validated publication"
        ),
        "StartAt": "BuildFeaturesFromCuratedSales",
        "TimeoutSeconds": 3600,
        "States": {
            "BuildFeaturesFromCuratedSales": {
                "Type": "Task",
                "Resource": (
                    "arn:aws:states:::sagemaker:"
                    "createProcessingJob.sync"
                ),
                "Parameters": feature_parameters,
                "ResultPath": "$.feature_job",
                "Next": "GenerateAndValidateForecasts",
            },
            "GenerateAndValidateForecasts": {
                "Type": "Task",
                "Resource": (
                    "arn:aws:states:::sagemaker:"
                    "createProcessingJob.sync"
                ),
                "Parameters": prediction_parameters,
                "ResultPath": "$.prediction_job",
                "Next": "PublishValidatedForecasts",
            },
            "PublishValidatedForecasts": {
                "Type": "Task",
                "Resource": (
                    "arn:aws:states:::aws-sdk:s3:copyObject"
                ),
                "Parameters": {
                    "Bucket": bucket,
                    "CopySource.$": formatted(source),
                    "Key.$": formatted(destination),
                },
                "ResultPath": "$.publish_result",
                "End": True,
            },
        },
    }


def ensure_policies(iam, config, sagemaker_role_arn):
    """Grant scoped access to existing SageMaker and workflow roles."""
    bucket = config["bucket"]
    prefix = config["prefix"].rstrip("/")
    region = config["region"]
    account = config["account_id"]

    sage_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": "s3:GetObject",
                "Resource": [
                    f"arn:aws:s3:::{bucket}/curated/stg_sales/*",
                    f"arn:aws:s3:::{bucket}/{prefix}/inputs/*",
                    f"arn:aws:s3:::{bucket}/{prefix}/features/*",
                ],
            },
            {
                "Effect": "Allow",
                "Action": "s3:PutObject",
                "Resource": [
                    f"arn:aws:s3:::{bucket}/{prefix}/features/*",
                    f"arn:aws:s3:::{bucket}/{prefix}/predictions/*",
                ],
            },
        ],
    }

    iam.put_role_policy(
        RoleName=config["sagemaker_role_name"],
        PolicyName="RetailForecastCloudFeatureAndPrediction",
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
                    "processing-job/retail-v1-orch-*"
                ),
            },
            {
                "Effect": "Allow",
                "Action": "iam:PassRole",
                "Resource": sagemaker_role_arn,
                "Condition": {
                    "StringEquals": {
                        "iam:PassedToService": "sagemaker.amazonaws.com"
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
                    f"arn:aws:s3:::{bucket}/forecasts/published/v1/*"
                ),
            },
        ],
    }

    iam.put_role_policy(
        RoleName=config["state_role_name"],
        PolicyName="RetailForecastPublishWorkflowV1",
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
    session = boto3.Session(region_name=config["region"])

    account = session.client("sts").get_caller_identity()["Account"]

    if account != config["account_id"]:
        raise ValueError("Unexpected AWS account")

    sfn = session.client("stepfunctions")

    state_arn = (
        f"arn:aws:states:{config['region']}:{account}:stateMachine:"
        f"{config['state_machine_name']}"
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
            name=f"forecast-{run_id}",
            input=json.dumps({"run_id": run_id}),
        )

        print("Execution ARN:", result["executionArn"])
        print("Run ID:", run_id)
        print(
            "Published forecast after successful execution:",
            f"s3://{config['bucket']}/forecasts/published/v1/"
            f"origin={config['forecast_origin']}/{run_id}/"
            "batch_forecasts.parquet",
        )
        return

    # Each source-code bundle is identified by its content.
    local_files = file_bundle()
    digest = hashlib.sha256()

    for name, path in sorted(local_files.items()):
        if not path.is_file():
            raise FileNotFoundError(path)

        digest.update(
            name.encode("utf-8") + b"\0" + path.read_bytes()
        )

    bundle_sha = digest.hexdigest()[:24]

    code_prefix = (
        f"{config['prefix'].rstrip('/')}/"
        f"inputs/{bundle_sha}/code"
    )

    # Reuse the image already published and validated in ECR.
    image = session.client("ecr").describe_images(
        repositoryName=config["ecr_repository"],
        imageIds=[{"imageTag": config["image_tag"]}],
    )["imageDetails"][0]

    image_uri = (
        f"{account}.dkr.ecr.{config['region']}.amazonaws.com/"
        f"{config['ecr_repository']}@{image['imageDigest']}"
    )

    iam = session.client("iam")

    sagemaker_role_arn = iam.get_role(
        RoleName=config["sagemaker_role_name"]
    )["Role"]["Arn"]

    state_role_arn = iam.get_role(
        RoleName=config["state_role_name"]
    )["Role"]["Arn"]

    definition = build_definition(
        config,
        image_uri,
        code_prefix,
        sagemaker_role_arn,
    )

    definition_json = json.dumps(definition, indent=2)

    check = sfn.validate_state_machine_definition(
        definition=definition_json,
        type="STANDARD",
    )

    print("Workflow definition:", check["result"])

    if check["result"] != "OK":
        print(json.dumps(check.get("diagnostics", []), indent=2))
        raise RuntimeError("Invalid workflow definition")

    print("Code bundle:", bundle_sha)
    print("Pinned image:", image_uri)
    print("Stages: build features -> infer and validate -> publish")

    if args.dry_run:
        print("DRY RUN: no S3 uploads, IAM updates or workflow creation")
        return

    # Publish the exact source files used by this workflow.
    s3 = session.client("s3")

    for name, path in local_files.items():
        s3.upload_file(
            str(path),
            config["bucket"],
            f"{code_prefix}/{name}",
        )

    ensure_policies(
        iam,
        config,
        sagemaker_role_arn,
    )

    output = (
        ROOT / "artifacts/aws/stepfunctions/"
        "forecast_workflow_v1.asl.json"
    )

    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        definition_json + "\n",
        encoding="utf-8",
    )

    try:
        sfn.describe_state_machine(
            stateMachineArn=state_arn
        )

    except sfn.exceptions.StateMachineDoesNotExist:
        response = sfn.create_state_machine(
            name=config["state_machine_name"],
            definition=definition_json,
            roleArn=state_role_arn,
            type="STANDARD",
        )

        print("Created:", response["stateMachineArn"])

    else:
        sfn.update_state_machine(
            stateMachineArn=state_arn,
            definition=definition_json,
            roleArn=state_role_arn,
        )

        print("Updated:", state_arn)

    print("DEPLOYED: no SageMaker jobs started")


if __name__ == "__main__":
    main()
