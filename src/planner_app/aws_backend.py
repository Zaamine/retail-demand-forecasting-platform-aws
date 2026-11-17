"""Connect the Retail Planner to verified AWS forecasting outputs."""

from pathlib import Path

from agent.operations_service import PlanningStatusReader
from planner_app.snapshot import load_verified_forecasts_s3


def load_aws_planning_backend(
    project_root: Path,
    *,
    run_id: str,
    s3_client,
    stepfunctions_client,
):
    """Verify one published forecast and its successful AWS execution."""
    snapshot = load_verified_forecasts_s3(
        project_root,
        run_id=run_id,
        s3_client=s3_client,
    )

    reader = PlanningStatusReader(
        calendar=snapshot.forecast_service.calendar,
        forecasts=snapshot.forecasts,
        snapshot_bytes=snapshot.snapshot_bytes,
        report=snapshot.report,
        forecast_uri=snapshot.forecast_uri,
        workflow_config=snapshot.workflow_config,
        stepfunctions_client=stepfunctions_client,
    )

    if not reader.snapshot_verified:
        raise ValueError(
            "Published forecast evidence did not pass planning validation."
        )

    status = reader()

    if (
        status["status"] != "ready_for_historical_planning"
        or status["workflow"]["execution_status"] != "SUCCEEDED"
    ):
        raise RuntimeError(
            "The successful AWS forecast execution could not be verified."
        )

    return snapshot, reader, status
