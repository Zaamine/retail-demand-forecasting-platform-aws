"""Safely bootstrap the CT cutoff checkpoint from an existing pending V2 registry package.

Read-only by default. --execute --confirm-bootstrap performs one conditional S3
PutObject only if no checkpoint exists; never modifies a model or launches jobs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/sagemaker/ct_state_v1.json"
REGISTRY_CONFIG = ROOT / "configs/sagemaker/ct_registry_v1.json"


def require(actual, expected, label):
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}; got {actual!r}")


def is_not_found(exc):
    err = exc.response.get("Error", {})
    return err.get("Code") in {"NoSuchKey", "404", "NotFound"}


def get_existing(s3, bucket, key):
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if is_not_found(exc):
            return None
        raise
    return json.loads(response["Body"].read())


def s3_json_sha(s3, bucket, key):
    content = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return json.loads(content), hashlib.sha256(content).hexdigest()


def require_job(sm, name, expected_output_uri):
    job = sm.describe_processing_job(ProcessingJobName=name)
    require(job["ProcessingJobStatus"], "Completed", f"job {name} status")
    outputs = [item["S3Output"]["S3Uri"] for item in job["ProcessingOutputConfig"]["Outputs"]]
    if expected_output_uri not in outputs:
        raise ValueError(f"Job {name} output S3 URI not verified: {outputs}")


def validate_registry_v2(sm, cfg, version):
    if not version.endswith("/2"):
        raise ValueError("Only the verified V2 bootstrap is supported in CT-6A")
    package = sm.describe_model_package(ModelPackageName=version)
    require(package["ModelPackageArn"], version, "V2 ARN")
    require(package["ModelPackageStatus"], "Completed", "V2 registration status")
    require(package["ModelApprovalStatus"], "PendingManualApproval", "V2 approval status")
    require(package["SourceUri"], f"s3://{cfg['bucket']}/{cfg['registry_model_key']}", "V2 registry SourceUri")
    if "InferenceSpecification" in package or "ModelMetrics" in package:
        raise ValueError("V2 must be model-only with no new performance metrics")
    attrs = package.get("CustomerMetadataProperties", {})
    expected = {
        "TrainingCutoff": cfg["training_cutoff"],
        "ForecastOrigin": cfg["forecast_origin"],
        "ModelSHA256": cfg["challenger_model_sha256"],
        "TrainingReportSHA256": cfg["training_report_sha256"],
        "DatasetManifestSHA256": cfg["ct2_manifest_sha256"],
        "PromotionEligible": "false",
        "NewOutOfSampleWAPE": "unavailable",
        "TrainingExecution": "sagemaker_processing_cpu",
        "CT2ProcessingJob": cfg["ct2_processing_job"],
        "CT3ProcessingJob": cfg["ct3_processing_job"],
        "CT4ProcessingJob": cfg["ct4_processing_job"],
    }
    for key, value in expected.items():
        require(attrs.get(key), value, f"V2 metadata {key}")
    return package, attrs


def build_checkpoint(cfg, state_cfg, attrs):
    """A deterministic checkpoint: repeated executions yield identical bytes."""
    return {
        "schema_version": "ct_checkpoint_v1",
        "last_successful_ct_cutoff": cfg["training_cutoff"],
        "last_successful_cycle_phase": "CHALLENGER_REGISTERED_PENDING",
        "last_registered_challenger_arn": state_cfg["challenger_package_arn"],
        "last_challenger_sha256": cfg["challenger_model_sha256"],
        "last_challenger_approval_status": "PendingManualApproval",
        "last_cycle_promoted": False,
        "last_cycle_new_out_of_sample_wape_available": False,
        "registered_champion_reference_arn": cfg["champion_v1_package_arn"],
        "registered_champion_reference_sha256": cfg["champion_v1_model_sha256"],
        "ct2_processing_job": attrs["CT2ProcessingJob"],
        "ct3_processing_job": attrs["CT3ProcessingJob"],
        "ct4_processing_job": attrs["CT4ProcessingJob"],
        "ct2_manifest_sha256": cfg["ct2_manifest_sha256"],
        "ct3_training_report_sha256": cfg["training_report_sha256"],
        "ct4_inference_report_sha256": attrs["InferenceReportSHA256"],
        "note": "Cutoff completed via registered pending challenger, not approved deployment or demonstrated forecast accuracy.",
    }


def checkpoint_bytes(state):
    return (json.dumps(state, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()


def atomic_bootstrap(s3, bucket, key, state, execute=False):
    """Create once; existing identical checkpoint is a safe no-op."""
    current = get_existing(s3, bucket, key)
    if current is not None:
        if current != state:
            raise RuntimeError("Existing CT checkpoint differs; refusing to overwrite it")
        return "ALREADY_PRESENT_IDENTICAL"
    if not execute:
        return "DRY_RUN_WOULD_CREATE"
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=checkpoint_bytes(state),
                      ContentType="application/json", IfNoneMatch="*")
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"PreconditionFailed", "412", "ConditionalRequestConflict", "409"}:
            raise RuntimeError("Concurrent checkpoint write detected: re-read and re-audit before any retry") from exc
        raise
    recorded = get_existing(s3, bucket, key)
    if recorded != state:
        raise RuntimeError("Checkpoint read-after-write verification failed")
    return "CREATED_AND_VERIFIED"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--registry-config", type=Path, default=REGISTRY_CONFIG)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-bootstrap", action="store_true")
    args = parser.parse_args()
    if args.execute != args.confirm_bootstrap:
        parser.error("--execute and --confirm-bootstrap must be supplied together")

    state_cfg = json.loads(args.config.read_text())
    cfg = json.loads(args.registry_config.read_text())
    for field in ("region", "account_id", "bucket"):
        require(state_cfg[field], cfg[field], f"state config {field}")
    require(cfg["training_cutoff"], "2017-08-15", "initial CT cutoff")
    require(state_cfg["challenger_package_arn"],
            f"arn:aws:sagemaker:{cfg['region']}:{cfg['account_id']}:model-package/{cfg['model_package_group_name']}/2",
            "challenger package ARN")
    require(state_cfg["checkpoint_key"],
            "mlops/continuous_training/v1/state/last_successful_ct.json", "checkpoint key")

    session = boto3.Session(region_name=cfg["region"])
    require(session.client("sts").get_caller_identity()["Account"], cfg["account_id"], "AWS account")
    sm, s3 = session.client("sagemaker"), session.client("s3")
    champion = sm.describe_model_package(ModelPackageName=cfg["champion_v1_package_arn"])
    require(champion["ModelPackageArn"], cfg["champion_v1_package_arn"], "V1 package ARN")
    require(champion["ModelPackageStatus"], "Completed", "V1 status")
    require(champion.get("CustomerMetadataProperties", {}).get("ModelSHA256"),
            cfg["champion_v1_model_sha256"], "V1 model sha256")
    _, attrs = validate_registry_v2(sm, cfg, state_cfg["challenger_package_arn"])
    bucket = cfg["bucket"]
    for name, prefix in ((cfg["ct2_processing_job"], cfg["ct2_prefix"]),
                         (cfg["ct3_processing_job"], cfg["ct3_prefix"]),
                         (cfg["ct4_processing_job"], cfg["ct4_prefix"])):
        require_job(sm, name, f"s3://{bucket}/{prefix}")
    for prefix, file, expected_sha in (
        (cfg["ct2_prefix"], "dataset_manifest.json", cfg["ct2_manifest_sha256"]),
        (cfg["ct3_prefix"], "training_report.json", cfg["training_report_sha256"]),
        (cfg["ct4_prefix"], "inference_report.json", attrs["InferenceReportSHA256"]),
    ):
        report, sha = s3_json_sha(s3, bucket, prefix + file)
        require(sha, expected_sha, f"{file} sha256")
        require(report.get("status"), "PASSED", f"{file} status")
    state = build_checkpoint(cfg, state_cfg, attrs)
    key = state_cfg["checkpoint_key"]
    print("Checkpoint target:", f"s3://{bucket}/{key}")
    print("CT registration: version 2 Completed / PendingManualApproval")
    print("Training cutoff:", state["last_successful_ct_cutoff"])
    print("Promotion eligible: NO; no new out-of-sample WAPE")
    print("Checkpoint preview:\n" + checkpoint_bytes(state).decode())
    action = atomic_bootstrap(s3, bucket, key, state, execute=args.execute)
    print("CT-6A checkpoint:", action)
    if not args.execute:
        print("DRY RUN: no AWS writes, Processing Jobs or Registry mutations")
    else:
        print("No Processing Job, Training Job or model promotion started")


if __name__ == "__main__":
    main()
