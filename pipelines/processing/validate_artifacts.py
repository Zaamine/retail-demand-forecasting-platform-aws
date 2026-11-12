"""Validate the frozen XGBoost V1 artifact bundle on SageMaker Processing.

This job checks the model input contract, its categorical
manifest, and the integrity of the compressed model archive.

It performs artifact validation, not model training or evaluation.
"""

import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path


INPUT = Path("/opt/ml/processing/input")
OUTPUT = Path("/opt/ml/processing/output")

EXPECTED_CATEGORICAL_FEATURES = {
    "store_nbr",
    "family",
    "target_day_of_week",
    "target_month",
}


def sha256_file(path: Path) -> str:
    """Compute a streaming SHA-256 digest without loading the file in RAM."""

    digest = hashlib.sha256()

    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> None:
    """Validate inputs and publish a machine-readable processing report."""

    contract_path = (
        INPUT / "contract" / "point_forecast_model_contract_v1.json"
    )
    categories_path = (
        INPUT / "categories" / "categories_v1.json"
    )
    archive_path = INPUT / "model" / "model.tar.gz"

    with contract_path.open(encoding="utf-8") as source:
        contract = json.load(source)

    with categories_path.open(encoding="utf-8") as source:
        categories = json.load(source)

    # The frozen champion must expose exactly its 31 input features.
    features = contract["model_features"]

    if len(features) != 31 or len(set(features)) != 31:
        raise ValueError("Unexpected V1 feature count or duplicate features.")

    if not EXPECTED_CATEGORICAL_FEATURES.issubset(features):
        raise ValueError("Missing categorical predictors in V1 contract.")

    if not isinstance(categories, dict) or not categories:
        raise ValueError("Invalid categorical manifest.")

    # Read the archive and validate that its gzip/tar structure is intact.
    # The archived model is not executed by this lightweight job.
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = archive.getmembers()

        if not members:
            raise ValueError("The frozen model archive is empty.")

        if not any(member.isfile() for member in members):
            raise ValueError("The archive contains no model file.")

        # Read archive members to force decompression and CRC verification.
        for member in members:
            if not member.isfile():
                continue

            source = archive.extractfile(member)

            if source is None:
                raise ValueError(f"Cannot read archive member: {member.name}")

            with source:
                for _ in iter(lambda: source.read(1024 * 1024), b""):
                    pass

        member_names = [member.name for member in members]

    report = {
        "status": "PASSED",
        "validation_type": "frozen_v1_artifact_integrity",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_count": len(features),
        "categorical_features": sorted(EXPECTED_CATEGORICAL_FEATURES),
        "category_manifest_keys": sorted(categories.keys()),
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256_file(archive_path),
        "archive_members": member_names,
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)

    report_path = OUTPUT / "validation_report.json"

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print("SageMaker V1 artifact validation: PASSED")


if __name__ == "__main__":
    main()
