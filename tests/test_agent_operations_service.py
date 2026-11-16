"""Offline tests for published forecast and AWS workflow status."""

import hashlib
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest
from botocore.exceptions import ClientError

from agent.operations_service import PlanningStatusReader
from agent.planning_calendar import PlanningCalendar


class FakeStepFunctions:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def describe_execution(self, *, executionArn):
        self.calls.append(executionArn)

        if self.fail:
            raise ClientError(
                {
                    "Error": {
                        "Code": "AccessDeniedException",
                        "Message": "Access denied",
                    }
                },
                "DescribeExecution",
            )

        return {
            "status": "SUCCEEDED",
            "startDate": datetime(
                2026, 11, 12, 18, 54, tzinfo=timezone.utc
            ),
            "stopDate": datetime(
                2026, 11, 12, 18, 59, tzinfo=timezone.utc
            ),
        }


def make_reader(*, corrupt=False, incomplete=False, aws_error=False):
    records = [
        {
            "store_nbr": store,
            "family": "DAIRY",
            "target_date": (
                date(2017, 8, 15) + timedelta(days=day)
            ),
        }
        for store in (12, 44)
        for day in range(1, 15)
    ]

    forecasts = pd.DataFrame(records)

    if incomplete:
        forecasts = forecasts.iloc[:-1].copy()

    original_bytes = b"synthetic-forecast-snapshot"

    report = {
        "status": "PASSED",
        "forecast_origin": "2017-08-15",
        "first_target_date": "2017-08-16",
        "last_target_date": "2017-08-29",
        "forecast_rows": 28,
        "unique_series": 2,
        "forecast_sha256": hashlib.sha256(
            original_bytes
        ).hexdigest(),
    }

    config = {
        "region": "eu-west-1",
        "account_id": "123456789012",
        "bucket": "test-bucket",
        "origin": "2017-08-15",
        "state_machine_name": "test-workflow",
    }

    client = FakeStepFunctions(fail=aws_error)

    reader = PlanningStatusReader(
        calendar=PlanningCalendar(date(2017, 8, 15)),
        forecasts=forecasts,
        snapshot_bytes=(
            b"modified-snapshot" if corrupt else original_bytes
        ),
        report=report,
        forecast_uri=(
            "s3://test-bucket/forecasts/published/v2/"
            "origin=2017-08-15/test-run/"
            "forecast_predictions.parquet"
        ),
        workflow_config=config,
        stepfunctions_client=client,
    )

    return reader, client


def test_verified_snapshot_and_successful_workflow():
    reader, client = make_reader()
    result = reader()

    assert result["status"] == "ready_for_historical_planning"
    assert result["published_estimates"]["verified"]
    assert result["workflow"]["execution_status"] == "SUCCEEDED"

    assert len(client.calls) == 1
    assert client.calls[0].endswith(
        ":test-workflow:test-forecast-test-run"
    )


def test_modified_snapshot_is_rejected():
    reader, _ = make_reader(corrupt=True)
    result = reader()

    assert result["status"] == "not_validated"
    assert not result["published_estimates"]["verified"]


def test_missing_forecast_day_is_rejected():
    reader, _ = make_reader(incomplete=True)
    result = reader()

    assert result["status"] == "not_validated"
    assert not result["published_estimates"]["verified"]


def test_aws_lookup_failure_does_not_invalidate_snapshot():
    reader, _ = make_reader(aws_error=True)
    result = reader()

    assert result["status"] == "ready_for_historical_planning"
    assert result["published_estimates"]["verified"]
    assert result["workflow"]["lookup"] == "unavailable"
    assert result["workflow"]["execution_status"] is None
    assert (
        result["workflow"]["lookup_error"]
        == "AccessDeniedException"
    )
