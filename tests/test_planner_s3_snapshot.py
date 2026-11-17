"""Offline tests for S3 forecast loading and local compatibility."""

from datetime import date, timedelta
from hashlib import sha256
from io import BytesIO
import json

import pandas as pd
import pytest
from botocore.exceptions import ClientError

from planner_app.snapshot import (
    load_verified_forecasts,
    load_verified_forecasts_s3,
)


RUN_ID = "20261112-185450-b04315"


class FakeS3:
    def __init__(self, content=None, missing=False):
        self.content = content
        self.missing = missing
        self.calls = []

    def get_object(self, *, Bucket, Key):
        self.calls.append((Bucket, Key))

        if self.missing:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "Missing"}},
                "GetObject",
            )

        return {"Body": BytesIO(self.content)}


@pytest.fixture
def prepared_snapshot(tmp_path):
    data = pd.DataFrame({
        "store_nbr": [44] * 14,
        "family": ["DAIRY"] * 14,
        "target_date": [
            (date(2017, 8, 15) + timedelta(days=i)).isoformat()
            for i in range(1, 15)
        ],
        "prediction": [float(i) for i in range(1, 15)],
    })

    content = BytesIO()
    data.to_parquet(content, index=False)
    payload = content.getvalue()

    report = {
        "status": "PASSED",
        "forecast_origin": "2017-08-15",
        "first_target_date": "2017-08-16",
        "last_target_date": "2017-08-29",
        "forecast_sha256": sha256(payload).hexdigest(),
        "forecast_rows": 14,
        "unique_series": 1,
    }

    config = {
        "bucket": "test-retail-bucket",
        "region": "eu-west-1",
        "account_id": "123456789012",
        "origin": "2017-08-15",
        "state_machine_name": "retail-v2-test-forecast-workflow",
    }

    report_path = (
        tmp_path / "artifacts/aws/stepfunctions/test_forecast_v2/"
        "prediction_report.json"
    )
    config_path = tmp_path / "configs/stepfunctions/test_forecast_v2.json"
    local_path = tmp_path / "data/forecasts/published_snapshot.parquet"

    for path in (report_path, config_path, local_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    report_path.write_text(json.dumps(report), encoding="utf-8")
    config_path.write_text(json.dumps(config), encoding="utf-8")
    local_path.write_bytes(payload)

    return tmp_path, payload, report_path, config_path


def test_verified_s3_snapshot_uses_pinned_publication(prepared_snapshot):
    root, payload, _, _ = prepared_snapshot
    s3 = FakeS3(payload)

    result = load_verified_forecasts_s3(
        root, run_id=RUN_ID, s3_client=s3
    )

    expected_key = (
        "forecasts/published/v2/origin=2017-08-15/"
        f"{RUN_ID}/forecast_predictions.parquet"
    )

    assert s3.calls == [("test-retail-bucket", expected_key)]
    assert result.snapshot_bytes == payload
    assert result.forecast_uri == f"s3://test-retail-bucket/{expected_key}"
    assert len(result.forecasts) == 14
    assert result.forecast_service.source == result.forecast_uri

    total = result.forecast_service.sales_for_store(
        44, "DAIRY", "full horizon"
    )["total_predicted_sales"]

    assert total == 105.0


def test_local_interface_remains_unchanged(prepared_snapshot):
    root, _, _, _ = prepared_snapshot

    forecasts, service = load_verified_forecasts(root)

    assert len(forecasts) == 14
    assert service.source.startswith(
        "verified_local_snapshot:sha256="
    )


def test_corrupt_s3_bytes_fail_closed(prepared_snapshot):
    root, _, _, _ = prepared_snapshot

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_verified_forecasts_s3(
            root,
            run_id=RUN_ID,
            s3_client=FakeS3(b"tampered"),
        )


def test_missing_s3_object_does_not_use_local_file(prepared_snapshot):
    root, _, _, _ = prepared_snapshot

    with pytest.raises(ClientError):
        load_verified_forecasts_s3(
            root,
            run_id=RUN_ID,
            s3_client=FakeS3(missing=True),
        )


def test_invalid_run_id_fails_before_s3_call(prepared_snapshot):
    root, payload, _, _ = prepared_snapshot
    s3 = FakeS3(payload)

    with pytest.raises(ValueError, match="Invalid forecast run ID"):
        load_verified_forecasts_s3(
            root,
            run_id="../../other",
            s3_client=s3,
        )

    assert not s3.calls


def test_origin_mismatch_fails_before_s3_call(prepared_snapshot):
    root, payload, _, config_path = prepared_snapshot

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["origin"] = "2017-08-01"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    s3 = FakeS3(payload)

    with pytest.raises(ValueError, match="Workflow origin"):
        load_verified_forecasts_s3(
            root,
            run_id=RUN_ID,
            s3_client=s3,
        )

    assert not s3.calls


def test_report_rejection_fails_before_s3_call(prepared_snapshot):
    root, payload, report_path, _ = prepared_snapshot

    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["status"] = "FAILED"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    s3 = FakeS3(payload)

    with pytest.raises(ValueError, match="not validated"):
        load_verified_forecasts_s3(
            root,
            run_id=RUN_ID,
            s3_client=s3,
        )

    assert not s3.calls


class FakeStepFunctions:
    def __init__(self, status="SUCCEEDED"):
        self.status = status
        self.calls = []

    def describe_execution(self, *, executionArn):
        from datetime import datetime, timezone

        self.calls.append(executionArn)

        return {
            "status": self.status,
            "startDate": datetime(
                2026, 11, 12, 18, 54, tzinfo=timezone.utc
            ),
            "stopDate": datetime(
                2026, 11, 12, 19, 0, tzinfo=timezone.utc
            ),
        }


def test_aws_backend_validates_real_workflow_contract(prepared_snapshot):
    from planner_app.aws_backend import load_aws_planning_backend

    root, payload, _, _ = prepared_snapshot
    s3 = FakeS3(payload)
    stepfunctions = FakeStepFunctions()

    snapshot, reader, status = load_aws_planning_backend(
        root,
        run_id=RUN_ID,
        s3_client=s3,
        stepfunctions_client=stepfunctions,
    )

    assert reader.snapshot_verified
    assert status["status"] == "ready_for_historical_planning"
    assert status["workflow"]["execution_status"] == "SUCCEEDED"
    assert status["published_estimates"]["forecast_rows"] == 14
    assert snapshot.forecast_service.source == snapshot.forecast_uri

    assert len(s3.calls) == 1
    assert len(stepfunctions.calls) == 1
    assert stepfunctions.calls[0].endswith(
        f":retail-v2-test-forecast-workflow:test-forecast-{RUN_ID}"
    )


def test_aws_backend_rejects_unsuccessful_execution(prepared_snapshot):
    from planner_app.aws_backend import load_aws_planning_backend

    root, payload, _, _ = prepared_snapshot

    with pytest.raises(RuntimeError, match="successful AWS"):
        load_aws_planning_backend(
            root,
            run_id=RUN_ID,
            s3_client=FakeS3(payload),
            stepfunctions_client=FakeStepFunctions(status="FAILED"),
        )


def test_both_planner_modes_expose_injected_aws_status(prepared_snapshot):
    import asyncio
    import json
    from pathlib import Path

    from mcp import Client
    from planner_app.aws_backend import load_aws_planning_backend
    from planner_app.local_agent import build_local_planner
    from planner_app.bedrock_agent import build_bedrock_planner

    root, payload, _, _ = prepared_snapshot
    snapshot, reader, _ = load_aws_planning_backend(
        root,
        run_id=RUN_ID,
        s3_client=FakeS3(payload),
        stepfunctions_client=FakeStepFunctions(),
    )

    project_root = Path(__file__).resolve().parents[1]

    _, offline_server = build_local_planner(
        project_root,
        snapshot.forecasts,
        snapshot.forecast_service,
        status_reader=reader,
    )

    class ForbiddenBedrock:
        def converse(self, **kwargs):
            raise AssertionError("Bedrock must not be called.")

    _, bedrock_server, budget = build_bedrock_planner(
        project_root,
        snapshot.forecasts,
        snapshot.forecast_service,
        ForbiddenBedrock(),
        status_reader=reader,
        enabled=False,
    )

    async def check_status(server):
        async with Client(server, raise_exceptions=True) as client:
            result = await client.call_tool(
                "check_planning_data_status", {}
            )
            assert not result.is_error
            return json.loads(result.content[0].text)

    for server in (offline_server, bedrock_server):
        status = asyncio.run(check_status(server))
        assert status["published_estimates"]["verified"] is True
        assert status["workflow"]["execution_status"] == "SUCCEEDED"

    assert budget.attempts == 0
