"""Tests for business evidence completeness and numerical consistency."""

from datetime import date, timedelta

from agent.evidence_validator import (
    SalesEvidenceValidator,
    format_verified_sales,
)
from agent.planning_calendar import PlanningCalendar
from agent.planning_contract import PlanningContract


class FakeForecastService:
    source = "s3://synthetic-test-snapshot"

    def sales_for_store(self, store, family, when):
        start, end = (
            date.fromisoformat(value)
            for value in when.split("/")
        )

        return {
            "daily_predictions": [
                {
                    "date": (
                        start + timedelta(days=day)
                    ).isoformat(),
                    "predicted_sales": float(store + day),
                }
                for day in range((end - start).days + 1)
            ]
        }


def make_case():
    service = FakeForecastService()
    calendar = PlanningCalendar(date(2017, 8, 15))

    contract = PlanningContract.create(
        calendar,
        "next week",
        stores=[44],
        families=["DAIRY"],
    )

    rows = service.sales_for_store(
        44, "DAIRY", "2017-08-21/2017-08-27"
    )["daily_predictions"]

    lookup = {
        "round": 1,
        "name": "lookup_sales_forecast",
        "status": "success",
        "result": {
            "status": "available",
            "result_type": "period",
            "source": service.source,
            "data": {
                "store_nbr": 44,
                "family": "DAIRY",
                "period_start": "2017-08-21",
                "period_end": "2017-08-27",
                "daily_predictions": rows,
                "total_predicted_sales": sum(
                    row["predicted_sales"] for row in rows
                ),
            },
        },
    }

    readiness = {
        "round": 1,
        "name": "check_planning_data_status",
        "status": "success",
        "result": {
            "status": "ready_for_historical_planning",
            "published_estimates": {
                "verified": True,
                "source": service.source,
                "available_from": "2017-08-16",
                "available_through": "2017-08-29",
            },
        },
    }

    run = {
        "status": "completed",
        "tool_events": [readiness, lookup],
    }

    return SalesEvidenceValidator(service), contract, run


def test_complete_evidence():
    validator, contract, run = make_case()

    result = validator.validate(
        run, contract, require_readiness=True
    )

    assert result["evidence_complete"]
    assert result["covered"] == 7
    assert result["required"] == 7
    assert "Store 44" in format_verified_sales(result)


def test_incomplete_period_is_rejected():
    validator, contract, run = make_case()

    data = run["tool_events"][1]["result"]["data"]
    data["daily_predictions"] = data["daily_predictions"][:3]

    result = validator.validate(
        run, contract, require_readiness=True
    )

    assert not result["evidence_complete"]


def test_modified_sales_value_is_rejected():
    validator, contract, run = make_case()

    rows = run["tool_events"][1]["result"]["data"][
        "daily_predictions"
    ]
    rows[0]["predicted_sales"] += 100

    result = validator.validate(
        run, contract, require_readiness=True
    )

    assert not result["evidence_complete"]


def test_wrong_store_is_rejected():
    validator, contract, run = make_case()

    run["tool_events"][1]["result"]["data"][
        "store_nbr"
    ] = 45

    result = validator.validate(
        run, contract, require_readiness=True
    )

    assert not result["evidence_complete"]


def test_missing_readiness_is_rejected():
    validator, contract, run = make_case()

    run["tool_events"] = run["tool_events"][1:]

    result = validator.validate(
        run, contract, require_readiness=True
    )

    assert not result["evidence_complete"]
    assert not result["readiness_verified"]
