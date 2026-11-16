"""Read-only operational status for a historical forecast snapshot."""

import hashlib
import re
from datetime import timezone

import pandas as pd
from botocore.exceptions import BotoCoreError, ClientError


class PlanningStatusReader:
    """Combine published-snapshot evidence with Step Functions status."""

    def __init__(
        self, *, calendar, forecasts, snapshot_bytes,
        report, forecast_uri, workflow_config, stepfunctions_client,
    ):
        pattern = (
            r"s3://([^/]+)/forecasts/published/v2/"
            r"origin=(\d{4}-\d{2}-\d{2})/"
            r"([^/]+)/forecast_predictions\.parquet"
        )
        match = re.fullmatch(pattern, forecast_uri)

        if not match:
            raise ValueError("Unexpected published forecast URI.")

        bucket, origin, run_id = match.groups()

        if (
            bucket != workflow_config["bucket"]
            or origin != workflow_config["origin"]
            or origin != calendar.business_date.isoformat()
        ):
            raise ValueError(
                "Forecast source does not match workflow configuration."
            )

        self.calendar = calendar
        self.forecast_uri = forecast_uri
        self.stepfunctions_client = stepfunctions_client

        self.execution_name = f"test-forecast-{run_id}"
        self.execution_arn = (
            f"arn:aws:states:{workflow_config['region']}:"
            f"{workflow_config['account_id']}:execution:"
            f"{workflow_config['state_machine_name']}:"
            f"{self.execution_name}"
        )

        # Verify the published snapshot once during initialization.
        dates = pd.to_datetime(forecasts["target_date"]).dt.date

        full_period = calendar.resolve("full horizon")
        first_day = full_period.start
        last_day = full_period.end

        series_counts = (
            forecasts.assign(_day=dates)
            .groupby(["store_nbr", "family"])["_day"]
            .agg(["size", "nunique"])
        )

        unique_series = len(series_counts)

        self.snapshot_verified = bool(
            report.get("status") == "PASSED"
            and report.get("forecast_origin") == origin
            and report.get("first_target_date") == first_day.isoformat()
            and report.get("last_target_date") == last_day.isoformat()
            and report.get("forecast_sha256")
                == hashlib.sha256(snapshot_bytes).hexdigest()
            and report.get("forecast_rows") == len(forecasts)
            and report.get("unique_series") == unique_series
            and not forecasts.empty
            and dates.between(first_day, last_day).all()
            and series_counts["size"].eq(calendar.horizon_days).all()
            and series_counts["nunique"].eq(calendar.horizon_days).all()
        )

        self.first_day = first_day
        self.last_day = last_day
        self.available_dates = set(dates)
        self.forecast_rows = len(forecasts)
        self.unique_series = unique_series

    def __call__(self) -> dict:
        """Return planning readiness and read-only AWS execution status."""

        week = self.calendar.resolve("next week")
        week_dates = set(
            pd.date_range(week.start, week.end).date
        )

        ready = (
            self.snapshot_verified
            and week_dates.issubset(self.available_dates)
        )

        workflow = {
            "execution_name": self.execution_name,
            "lookup": "unavailable",
            "execution_status": None,
        }

        try:
            execution = self.stepfunctions_client.describe_execution(
                executionArn=self.execution_arn
            )

            workflow.update({
                "lookup": "successful",
                "execution_status": execution["status"],
                "started_at_utc": (
                    execution["startDate"]
                    .astimezone(timezone.utc)
                    .isoformat()
                ),
                "stopped_at_utc": (
                    execution["stopDate"]
                    .astimezone(timezone.utc)
                    .isoformat()
                    if "stopDate" in execution else None
                ),
            })

        except (ClientError, BotoCoreError) as exc:
            workflow["lookup_error"] = (
                exc.response["Error"]["Code"]
                if isinstance(exc, ClientError)
                else type(exc).__name__
            )

        return {
            "status": (
                "ready_for_historical_planning"
                if ready else "not_validated"
            ),
            "business_date": self.calendar.business_date.isoformat(),
            "planning_period": {
                "start": week.start.isoformat(),
                "end": week.end.isoformat(),
            },
            "published_estimates": {
                "verified": self.snapshot_verified,
                "available_from": self.first_day.isoformat(),
                "available_through": self.last_day.isoformat(),
                "forecast_rows": self.forecast_rows,
                "store_family_series": self.unique_series,
                "source": self.forecast_uri,
            },
            "workflow": workflow,
            "context": (
                "Historical planning simulation. Actual AWS execution "
                "timestamps are separate from the simulated business date. "
                "Snapshot readiness does not depend on the current ability "
                "to retrieve the Step Functions execution history."
            ),
        }
