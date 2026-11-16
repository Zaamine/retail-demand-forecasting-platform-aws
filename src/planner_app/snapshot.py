"""Load a verified local forecast snapshot without AWS calls."""

from datetime import date
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import pandas as pd

from agent.forecast_service import ForecastService
from agent.planning_calendar import PlanningCalendar


def load_verified_forecasts(project_root: Path):
    """Return validated forecast data and its query service."""
    root = Path(project_root)

    snapshot_path = root / "data/forecasts/published_snapshot.parquet"
    report_path = (
        root / "artifacts/aws/stepfunctions/test_forecast_v2/"
        "prediction_report.json"
    )

    if not snapshot_path.is_file():
        raise FileNotFoundError(
            f"Local forecast snapshot not found: {snapshot_path}"
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["status"] != "PASSED":
        raise ValueError("Forecast publication was not validated.")

    snapshot_bytes = snapshot_path.read_bytes()
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
        source=f"verified_local_snapshot:sha256={actual_sha}",
    )

    return forecasts, service
