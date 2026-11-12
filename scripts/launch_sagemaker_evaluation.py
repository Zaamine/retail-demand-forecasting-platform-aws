"""Submit the versioned final-holdout evaluation as SageMaker Processing."""

import argparse
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/sagemaker/holdout_evaluation_v1.json"


def s3_input(name: str, uri: str) -> dict:
    return {"InputName": name, "S3Input": {
        "S3Uri": uri,
        "LocalPath": f"/opt/ml/processing/input/{name}",
        "S3DataType": "S3Prefix", "S3InputMode": "File",
    }}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    raw = args.config.read_bytes()
    config = json.loads(raw)
    compute, s3_cfg, holdout = config["compute"], config["s3"], config["holdout"]
    session = boto3.Session(region_name=config["region"])
    caller = session.client("sts").get_caller_identity()["Account"]
    if args.role_arn.split(":")[4] != caller:
        raise ValueError("Execution role must belong to active AWS account")

    inputs_local = {
        "code": ROOT / "pipelines/processing/evaluate_holdout.py",
        "actuals": ROOT / "artifacts/aws/sagemaker_processing/evaluation_v1/holdout_actuals.parquet",
        "manifest": ROOT / "artifacts/aws/sagemaker_processing/evaluation_v1/holdout_actuals_manifest.json",
        "config": args.config.resolve(),
    }
    for name, path in inputs_local.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} input: {path}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job = f"{config['job_name_prefix']}-{stamp}-{uuid.uuid4().hex[:6]}"
    bucket = config["bucket"]
    in_prefix = f"{s3_cfg['input_prefix'].rstrip('/')}/{job}"
    out_uri = f"s3://{bucket}/{s3_cfg['output_prefix'].rstrip('/')}/{job}/"
    inputs = [s3_input(name, f"s3://{bucket}/{in_prefix}/{name}/{path.name}")
              for name, path in inputs_local.items()]
    inputs.append(s3_input("forecasts", f"s3://{bucket}/{s3_cfg['forecast_key']}"))
    request = {
        "ProcessingJobName": job,
        "RoleArn": args.role_arn,
        "AppSpecification": {
            "ImageUri": config["image_uri"],
            "ContainerEntrypoint": ["python3", "/opt/ml/processing/input/code/evaluate_holdout.py"],
            "ContainerArguments": [
                "--origin", holdout["forecast_origin"],
                "--expected-rows", str(holdout["expected_rows"]),
                "--reference-wape", str(holdout["reference_wape"]),
                "--wape-tolerance", str(holdout["wape_tolerance"]),
            ],
        },
        "ProcessingInputs": inputs,
        "ProcessingOutputConfig": {"Outputs": [{
            "OutputName": "evaluation", "S3Output": {
                "S3Uri": out_uri,
                "LocalPath": "/opt/ml/processing/output",
                "S3UploadMode": "EndOfJob",
            },
        }]},
        "ProcessingResources": {"ClusterConfig": {
            "InstanceCount": compute["instance_count"],
            "InstanceType": compute["instance_type"],
            "VolumeSizeInGB": compute["volume_size_gb"],
        }},
        "StoppingCondition": {"MaxRuntimeInSeconds": compute["max_runtime_seconds"]},
        "Tags": [
            {"Key": "Project", "Value": "retail-demand-forecasting"},
            {"Key": "Stage", "Value": "final-holdout-evaluation"},
            {"Key": "ConfigSHA256", "Value": hashlib.sha256(raw).hexdigest()},
        ],
    }
    print("Job name:", job)
    print("Instance:", compute["instance_type"])
    print("Output:", out_uri)
    if args.dry_run:
        print("DRY RUN: no upload and no job submitted")
        print(json.dumps(request, indent=2))
        return

    s3 = session.client("s3")
    for name, path in inputs_local.items():
        s3.upload_file(str(path), bucket, f"{in_prefix}/{name}/{path.name}")
    response = session.client("sagemaker").create_processing_job(**request)
    print("Submitted ARN:", response["ProcessingJobArn"])


if __name__ == "__main__":
    main()
