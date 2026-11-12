"""Launch a configurable SageMaker artifact-validation Processing Job.

Infrastructure choices live in a version-controlled JSON file.
Each run receives its own S3 input and output prefixes.
"""

import argparse
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import boto3


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/sagemaker/bootstrap_v1.json"


def load_config(path: Path) -> tuple[dict, str]:
    """Read and validate deployment settings before any AWS mutation."""

    raw = path.read_bytes()
    config = json.loads(raw)

    required = {
        "region", "bucket", "job_name_prefix",
        "image_uri", "compute", "s3",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing config fields: {sorted(missing)}")

    for key in (
        "instance_type", "instance_count",
        "volume_size_gb", "max_runtime_seconds",
    ):
        if key not in config["compute"]:
            raise ValueError(f"Missing compute setting: {key}")

    for key in (
        "input_prefix", "output_prefix", "model_archive_key",
    ):
        if key not in config["s3"]:
            raise ValueError(f"Missing S3 setting: {key}")

    return config, hashlib.sha256(raw).hexdigest()


def processing_input(name: str, uri: str) -> dict:
    """Mount one S3 input under a dedicated Processing directory."""

    return {
        "InputName": name,
        "S3Input": {
            "S3Uri": uri,
            "LocalPath": f"/opt/ml/processing/input/{name}",
            "S3DataType": "S3Prefix",
            "S3InputMode": "File",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config, config_hash = load_config(args.config)

    # Explicit inputs make packaging predictable and auditable.
    local_inputs = {
        "code": ROOT / "pipelines/processing/validate_artifacts.py",
        "contract": (
            ROOT / "artifacts/global_xgboost/"
            "point_forecast_model_contract_v1.json"
        ),
        "categories": (
            ROOT / "configs/global_xgboost/categories_v1.json"
        ),
        "config": args.config.resolve(),
    }

    for name, path in local_inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")

    session = boto3.Session(region_name=config["region"])

    # Reject accidental use of an execution role from another account.
    account_id = session.client("sts").get_caller_identity()["Account"]
    role_account = args.role_arn.split(":")[4]

    if account_id != role_account:
        raise ValueError("Execution role and active AWS account differ.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    job_name = f"{config['job_name_prefix']}-{timestamp}-{suffix}"

    bucket = config["bucket"]
    s3_config = config["s3"]
    compute = config["compute"]

    # Isolate inputs by run instead of overwriting shared S3 objects.
    input_prefix = (
        f"{s3_config['input_prefix'].rstrip('/')}/{job_name}"
    )
    output_uri = (
        f"s3://{bucket}/"
        f"{s3_config['output_prefix'].rstrip('/')}/{job_name}/"
    )

    input_uris = {
        name: f"s3://{bucket}/{input_prefix}/{name}/{path.name}"
        for name, path in local_inputs.items()
    }

    inputs = [
        processing_input(name, uri)
        for name, uri in input_uris.items()
    ]

    model_uri = (
        f"s3://{bucket}/{s3_config['model_archive_key']}"
    )
    inputs.append(processing_input("model", model_uri))

    request = {
        "ProcessingJobName": job_name,
        "RoleArn": args.role_arn,
        "AppSpecification": {
            "ImageUri": config["image_uri"],
            "ContainerEntrypoint": [
                "python3",
                "/opt/ml/processing/input/code/validate_artifacts.py",
            ],
        },
        "ProcessingInputs": inputs,
        "ProcessingOutputConfig": {
            "Outputs": [{
                "OutputName": "validation-report",
                "S3Output": {
                    "S3Uri": output_uri,
                    "LocalPath": "/opt/ml/processing/output",
                    "S3UploadMode": "EndOfJob",
                },
            }],
        },
        "ProcessingResources": {
            "ClusterConfig": {
                "InstanceCount": compute["instance_count"],
                "InstanceType": compute["instance_type"],
                "VolumeSizeInGB": compute["volume_size_gb"],
            },
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": compute["max_runtime_seconds"],
        },
        "Tags": [
            {"Key": "Project", "Value": "retail-demand-forecasting"},
            {"Key": "Stage", "Value": "artifact-validation"},
            {"Key": "ConfigSHA256", "Value": config_hash},
        ],
    }

    print("Config SHA-256:", config_hash)
    print("Processing instance:", compute["instance_type"])
    print("Job name:", job_name)
    print("Output:", output_uri)

    # Preview the full AWS request without creating a billable job.
    if args.dry_run:
        print("DRY RUN: no files uploaded or job created.")
        print(json.dumps(request, indent=2))
        return

    s3 = session.client("s3")

    # Upload the exact script, contract, categories, and config.
    for name, path in local_inputs.items():
        key = f"{input_prefix}/{name}/{path.name}"
        s3.upload_file(str(path), bucket, key)

    response = session.client("sagemaker").create_processing_job(
        **request
    )

    print("Job submitted:", response["ProcessingJobArn"])


if __name__ == "__main__":
    main()
