"""Audit and register frozen XGBoost V1 in SageMaker Model Registry.

Registration references the existing S3 model.tar.gz and SageMaker-produced
holdout report. No training, Processing Job, endpoint or GPU is started.

An S3 SourceUri is used deliberately without an unverified inference image.
This creates a tracked version, not a deployable SageMaker endpoint package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from forecasting.registry_gate import audit_frozen_champion


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/sagemaker/champion_registry_v1.json"
DEFAULT_JOB = "retail-v1-holdout-eval-20261112-153624-4a0014"


def json_s3(s3, bucket: str, key: str) -> tuple[dict, str]:
    """Return JSON contents and SHA-256 of the original S3 object bytes."""

    data = s3.get_object(
        Bucket=bucket,
        Key=key,
    )["Body"].read()

    return json.loads(data), hashlib.sha256(data).hexdigest()


def verify_remote_model(
    s3,
    bucket: str,
    key: str,
    config: dict,
) -> None:
    """Verify exact model bytes rather than trusting its name or ETag."""

    info = s3.head_object(
        Bucket=bucket,
        Key=key,
    )

    if info["ContentLength"] != config["expected_model_size_bytes"]:
        raise ValueError(
            "S3 model archive has unexpected byte length"
        )

    digest = hashlib.sha256()

    stream = s3.get_object(
        Bucket=bucket,
        Key=key,
    )["Body"]

    with stream:
        for chunk in stream.iter_chunks(chunk_size=1024 * 1024):
            digest.update(chunk)

    if digest.hexdigest() != config["expected_model_sha256"]:
        raise ValueError(
            "S3 model archive SHA-256 differs from validated champion"
        )


def describe_existing_group(sagemaker, group_name: str) -> bool:
    """Return whether a SageMaker Model Package Group exists.

    SageMaker may report a missing group using either ResourceNotFound
    or ValidationException, depending on the API response.

    Only an explicit group-not-found error is treated as absence.
    Permission errors and unrelated validation failures must propagate.
    """

    try:
        sagemaker.describe_model_package_group(
            ModelPackageGroupName=group_name
        )

        return True

    except ClientError as exc:
        error = exc.response.get("Error", {})
        code = error.get("Code", "")
        message = error.get("Message", "").lower()

        if code in (
            "ResourceNotFound",
            "ResourceNotFoundException",
        ):
            return False

        if (
            code == "ValidationException"
            and "modelpackagegroup" in message
            and group_name.lower() in message
            and "does not exist" in message
        ):
            return False

        raise


def find_same_version(
    sagemaker,
    group: str,
    model_sha: str,
    metrics_sha: str,
):
    """Find an already registered version with identical artifacts.

    Matching the model and evaluation-report SHA-256 values prevents
    unnecessary duplicate versions on repeated executions.
    """

    paginator = sagemaker.get_paginator("list_model_packages")

    for page in paginator.paginate(
        ModelPackageGroupName=group
    ):
        for item in page.get("ModelPackageSummaryList", []):
            arn = item["ModelPackageArn"]

            record = sagemaker.describe_model_package(
                ModelPackageName=arn
            )

            attrs = record.get(
                "CustomerMetadataProperties",
                {},
            )

            if (
                attrs.get("ModelSHA256") == model_sha
                and attrs.get("EvaluationReportSHA256") == metrics_sha
            ):
                return record

    return None

def registration_client_token(
    group: str,
    model_sha: str,
    report_sha: str,
) -> str:
    """Build a stable SageMaker idempotency token (maximum 36 characters).

    The token uniquely identifies the model and its evaluation report.
    Repeated registration attempts use the same token.
    """
    payload = "\0".join(
        [group, model_sha, report_sha]
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()[:36]


def main() -> None:
    """Audit the frozen champion and optionally register it on AWS."""

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG,
        help="Version-controlled Model Registry configuration.",
    )

    parser.add_argument(
        "--evaluation-job",
        default=DEFAULT_JOB,
        help="Completed SageMaker Processing evaluation job.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the audit without creating AWS Registry resources.",
    )

    args = parser.parse_args()

    # Load the frozen registration contract.
    config = json.loads(
        args.config.read_text(encoding="utf-8")
    )

    session = boto3.Session(
        region_name=config["region"]
    )

    s3 = session.client("s3")
    sm = session.client("sagemaker")

    region = config["region"]
    bucket = config["bucket"]
    group = config["model_package_group_name"]
    model_key = config["model_artifact_key"]

    # Only a successfully completed AWS evaluation can be registered.
    job = sm.describe_processing_job(
        ProcessingJobName=args.evaluation_job
    )

    if job["ProcessingJobStatus"] != "Completed":
        raise RuntimeError(
            "Evaluation job is not Completed: "
            f"{job['ProcessingJobStatus']}"
        )

    if job["ProcessingJobArn"].split(":")[3] != region:
        raise ValueError(
            "Evaluation job is not in the configured region"
        )

    # Check that the evaluation outputs are stored at the expected path.
    expected_uri = (
        f"s3://{bucket}/"
        f"{config['evaluation_output_prefix'].rstrip('/')}/"
        f"{args.evaluation_job}/"
    )

    output_uris = [
        item["S3Output"]["S3Uri"]
        for item in job["ProcessingOutputConfig"]["Outputs"]
    ]

    if expected_uri not in output_uris:
        raise ValueError(
            "Unexpected SageMaker output location for evaluation job"
        )

    report_key = (
        f"{config['evaluation_output_prefix'].rstrip('/')}/"
        f"{args.evaluation_job}/evaluation_report.json"
    )

    # Retrieve the actual SageMaker-produced report and validate it.
    report, report_sha = json_s3(
        s3,
        bucket,
        report_key,
    )

    gate = audit_frozen_champion(
        report,
        config["holdout_audit"],
    )

    # Verify the frozen model archive stored in S3.
    verify_remote_model(
        s3,
        bucket,
        model_key,
        config,
    )

    model_uri = f"s3://{bucket}/{model_key}"
    report_uri = f"s3://{bucket}/{report_key}"

    # Persist audit provenance alongside the registered model.
    attrs = {
        "ModelSHA256": config["expected_model_sha256"],
        "EvaluationReportSHA256": report_sha,
        "EvaluationProcessingJob": args.evaluation_job,
        "FrozenNotebook": "05_global_xgboost",
        "EvaluationWAPE": f"{gate['wape']:.12f}",
        "ModelRuntime": "xgboost-3.4.1",
        "InferenceContainerValidated": "false",
    }

    # Make repeated registration requests idempotent at the API level.
    token = registration_client_token(
        group,
        attrs["ModelSHA256"],
        report_sha,
    )

    request = {
        "ModelPackageGroupName": group,
        "ModelPackageDescription": (
            "Frozen global direct multi-horizon XGBoost V1. "
            "Final holdout audited on SageMaker Processing; "
            "inference image pending."
        ),
        "ModelApprovalStatus": (
            config["registration"]["approval_status"]
        ),
        "SourceUri": model_uri,
        "ModelMetrics": {
            "ModelQuality": {
                "Statistics": {
                    "ContentType": "application/json",
                    "S3Uri": report_uri,
                }
            }
        },
        "CustomerMetadataProperties": attrs,
        "Task": config["registration"]["task"],
        "ClientToken": token,
    }

    # Display audit results before any AWS registry mutation.
    print("Cloud evaluation:", args.evaluation_job)
    print("Cloud WAPE:", f"{gate['wape']:.10f}")
    print(
        "Notebook parity difference:",
        f"{gate['absolute_wape_difference']:.3g}",
    )
    print(
        "S3 model SHA-256:",
        config["expected_model_sha256"],
    )
    print("Model Registry group:", group)
    print(
        "Registration status:",
        request["ModelApprovalStatus"],
    )
    print("Inference specification: deliberately not set")
    print("AUDIT PASSED")

    if args.dry_run:
        print(
            "DRY RUN: no model group or model version created."
        )
        print(json.dumps(request, indent=2))
        return

    # Create the group only if it does not already exist.
    # Missing-group ValidationException is now handled correctly.
    if not describe_existing_group(sm, group):
        created = sm.create_model_package_group(
            ModelPackageGroupName=group,
            ModelPackageGroupDescription=(
                "Versioned forecasting model artifacts "
                "and audited out-of-time metrics"
            ),
        )

        print(
            "Created Model Registry group:",
            created["ModelPackageGroupArn"],
        )

    # Reuse an identical registration rather than creating duplicates.
    existing = find_same_version(
        sm,
        group,
        config["expected_model_sha256"],
        report_sha,
    )

    if existing:
        arn = existing["ModelPackageArn"]
        version = existing.get("ModelPackageVersion")

        print(
            "Reusing identical registered model version:",
            arn,
        )

    else:
        response = sm.create_model_package(**request)

        arn = response["ModelPackageArn"]
        version = None

        print(
            "New Model Registry version submitted:",
            arn,
        )

    # Store technical provenance under artifacts/, never evidence/.
    dest = (
        ROOT
        / "artifacts/aws/sagemaker_registry/champion_v1"
    )

    dest.mkdir(
        parents=True,
        exist_ok=True,
    )

    record = {
        "model_package_arn": arn,
        "version": version,
        "model_package_group": group,
        "approval_status_requested": (
            request["ModelApprovalStatus"]
        ),
        "model_source_uri": model_uri,
        "evaluation_report_uri": report_uri,
        "evaluation_processing_job": args.evaluation_job,
        "evaluation_report_sha256": report_sha,
        "model_sha256": config["expected_model_sha256"],
        "audit": gate,
        "inference_image_registered": False,
    }

    (dest / "registry_record.json").write_text(
        json.dumps(record, indent=2) + "\n",
        encoding="utf-8",
    )

    (dest / "evaluation_report_cloud.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        "Recorded local provenance:",
        dest / "registry_record.json",
    )


if __name__ == "__main__":
    main()
