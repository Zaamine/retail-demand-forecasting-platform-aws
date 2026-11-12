"""Run a versioned, temporary SageMaker Processing batch-inference job.

A pinned ECR image digest and per-run immutable-by-convention S3 inputs
make each run traceable; --dry-run performs no S3 uploads/job creation.
"""
import argparse
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/sagemaker/batch_inference_v1.json"


def input_mount(name, uri):
    return {"InputName": name, "S3Input": {
        "S3Uri": uri,
        "LocalPath": f"/opt/ml/processing/input/{name}",
        "S3DataType": "S3Prefix", "S3InputMode": "File",
    }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    raw_config = args.config.read_bytes()
    config = json.loads(raw_config)
    region, bucket = config["region"], config["bucket"]
    session = boto3.Session(region_name=region)
    account = session.client("sts").get_caller_identity()["Account"]
    if args.role_arn.split(":")[4] != account:
        raise ValueError("Execution role is in a different AWS account")

    repository = config["ecr_repository"]
    tagged_image = session.client("ecr").describe_images(
        repositoryName=repository, imageIds=[{"imageTag": config["image_tag"]}]
    )["imageDetails"][0]
    image_uri = (
        f"{account}.dkr.ecr.{region}.amazonaws.com/"
        f"{repository}@{tagged_image['imageDigest']}"
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job_name = f"{config['job_name_prefix']}-{timestamp}-{uuid.uuid4().hex[:6]}"
    prefix = f"{config['inputs_prefix'].rstrip('/')}/{job_name}"
    output_uri = f"s3://{bucket}/{config['outputs_prefix'].rstrip('/')}/{job_name}/"

    local_files = {
        "code/batch_predict.py": ROOT / "pipelines/processing/batch_predict.py",
        "code/forecasting/__init__.py": ROOT / "src/forecasting/__init__.py",
        "code/forecasting/model_inputs.py": ROOT / "src/forecasting/model_inputs.py",
        "code/forecasting/parity.py": ROOT / "src/forecasting/parity.py",
        "contract/point_forecast_model_contract_v1.json": ROOT / "artifacts/global_xgboost/point_forecast_model_contract_v1.json",
        "categories/categories_v1.json": ROOT / "configs/global_xgboost/categories_v1.json",
        "features/features.parquet": ROOT / "artifacts/aws/sagemaker_batch_inference/v1/features.parquet",
        "manifest/features.manifest.json": ROOT / "artifacts/aws/sagemaker_batch_inference/v1/features.manifest.json",
        "config/batch_inference_v1.json": args.config.resolve(),
    }
    for key, path in local_files.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {key}: {path}")

    mount_files = {
        "contract": "point_forecast_model_contract_v1.json",
        "categories": "categories_v1.json",
        "features": "features.parquet",
        "manifest": "features.manifest.json",
        "config": "batch_inference_v1.json",
    }
    mounts = [input_mount("code", f"s3://{bucket}/{prefix}/code/")]
    mounts.extend(
        input_mount(name, f"s3://{bucket}/{prefix}/{name}/{filename}")
        for name, filename in mount_files.items()
    )
    mounts.append(input_mount("model", f"s3://{bucket}/{config['model_key']}"))
    mounts.append(input_mount("reference", f"s3://{bucket}/{config['reference_key']}"))

    request = {
        "ProcessingJobName": job_name,
        "RoleArn": args.role_arn,
        "AppSpecification": {
            "ImageUri": image_uri,
            "ContainerEntrypoint": [
                "python", "/opt/ml/processing/input/code/batch_predict.py"
            ],
            "ContainerArguments": [
                "--expected-rows", str(config["expected_rows"]),
                "--atol", str(config["absolute_tolerance"]),
                "--rtol", str(config["relative_tolerance"]),
            ],
        },
        "ProcessingInputs": mounts,
        "ProcessingOutputConfig": {"Outputs": [{
            "OutputName": "batch-forecasts",
            "S3Output": {
                "S3Uri": output_uri,
                "LocalPath": "/opt/ml/processing/output",
                "S3UploadMode": "EndOfJob",
            },
        }]},
        "ProcessingResources": {"ClusterConfig": {
            "InstanceCount": 1,
            "InstanceType": config["instance_type"],
            "VolumeSizeInGB": config["volume_size_gb"],
        }},
        "StoppingCondition": {"MaxRuntimeInSeconds": config["max_runtime_seconds"]},
        "Tags": [
            {"Key": "Project", "Value": "retail-demand-forecasting"},
            {"Key": "Stage", "Value": "batch-inference"},
            {"Key": "ConfigSHA256", "Value": hashlib.sha256(raw_config).hexdigest()},
        ],
    }
    print("Job:", job_name)
    print("Instance:", config["instance_type"])
    print("Pinned image:", image_uri)
    print("Output:", output_uri)
    if args.dry_run:
        print("DRY RUN: no inputs uploaded and no job submitted")
        print(json.dumps(request, indent=2))
        return
    s3 = session.client("s3")
    for key, path in local_files.items():
        s3.upload_file(str(path), bucket, f"{prefix}/{key}")
    response = session.client("sagemaker").create_processing_job(**request)
    print("Submitted:", response["ProcessingJobArn"])


if __name__ == "__main__":
    main()
