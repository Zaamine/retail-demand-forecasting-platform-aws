"""Load verified local or published S3 forecasting snapshots."""

from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re

import pandas as pd

from agent.forecast_service import ForecastService
from agent.planning_calendar import PlanningCalendar


@dataclass(frozen=True)
class S3ForecastSnapshot:
    """Verified S3 data and provenance for downstream AWS status checks."""

    forecasts: pd.DataFrame
    forecast_service: ForecastService
    snapshot_bytes: bytes
    report: dict
    forecast_uri: str
    workflow_config: dict


def _read_report(root: Path) -> dict:
    report_path = (
        root / "artifacts/aws/stepfunctions/test_forecast_v2/"
        "prediction_report.json"
    )
    return json.loads(report_path.read_text(encoding="utf-8"))


def _verify_and_build(snapshot_bytes: bytes, report: dict, source: str):
    """Apply the same evidence and business checks to both modes."""
    if report["status"] != "PASSED":
        raise ValueError("Forecast publication was not validated.")

    actual_sha = sha256(snapshot_bytes).hexdigest()

    if actual_sha != report["forecast_sha256"]:
        raise ValueError("Forecast snapshot SHA-256 mismatch.")

    forecasts = pd.read_parquet(BytesIO(snapshot_bytes))

    if len(forecasts) != report["forecast_rows"]:
        raise ValueError("Forecast row count mismatch.")

    series_count = len(
        forecasts[["store_nbr", "family"]].drop_duplicates()
    )

    if series_count != report["unique_series"]:
        raise ValueError("Forecast series count mismatch.")

    calendar = PlanningCalendar(
        business_date=date.fromisoformat(report["forecast_origin"]),
        horizon_days=14,
    )

    service = ForecastService(
        forecasts=forecasts,
        calendar=calendar,
        source=source,
    )

    return forecasts, service


def load_verified_forecasts(project_root: Path):
    """Preserve the existing local, offline interface."""
    root = Path(project_root)
    snapshot_path = root / "data/forecasts/published_snapshot.parquet"

    if not snapshot_path.is_file():
        raise FileNotFoundError(
            f"Local forecast snapshot not found: {snapshot_path}"
        )

    report = _read_report(root)
    snapshot_bytes = snapshot_path.read_bytes()
    actual_sha = sha256(snapshot_bytes).hexdigest()

    return _verify_and_build(
        snapshot_bytes,
        report,
        source=f"verified_local_snapshot:sha256={actual_sha}",
    )


def load_verified_forecasts_s3(
    project_root: Path,
    *,
    run_id: str,
    s3_client,
) -> S3ForecastSnapshot:
    """Load one pinned S3 publication without falling back to local data."""
    root = Path(project_root)

    if not isinstance(run_id, str) or re.fullmatch(
        r"\d{8}-\d{6}-[0-9a-f]{6}", run_id
    ) is None:
        raise ValueError("Invalid forecast run ID.")

    config_path = root / "configs/stepfunctions/test_forecast_v2.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    report = _read_report(root)

    if report["status"] != "PASSED":
        raise ValueError("Forecast publication was not validated.")

    if config["origin"] != report["forecast_origin"]:
        raise ValueError(
            "Workflow origin does not match the validation report."
        )

    key = (
        f"forecasts/published/v2/origin={config['origin']}/"
        f"{run_id}/forecast_predictions.parquet"
    )
    uri = f"s3://{config['bucket']}/{key}"

    response = s3_client.get_object(
        Bucket=config["bucket"],
        Key=key,
    )

    body = response["Body"]
    try:
        snapshot_bytes = body.read()
    finally:
        body.close()

    forecasts, service = _verify_and_build(
        snapshot_bytes,
        report,
        source=uri,
    )

    return S3ForecastSnapshot(
        forecasts=forecasts,
        forecast_service=service,
        snapshot_bytes=snapshot_bytes,
        report=report,
        forecast_uri=uri,
        workflow_config=config,
    )
