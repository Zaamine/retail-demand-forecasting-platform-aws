"""Tests for deterministic Retail Planner forecast queries."""

from datetime import date, timedelta

import pandas as pd
import pytest

from agent.planning_calendar import PlanningCalendar
from agent.forecast_service import ForecastService


@pytest.fixture
def snapshot():
    records = []

    for store in (12, 44):
        for family in ("DAIRY", "BEVERAGES"):
            for day in range(1, 15):
                records.append({
                    "store_nbr": store,
                    "family": family,
                    "forecast_origin": "2017-08-15",
                    "target_date": (
                        date(2017, 8, 15) + timedelta(days=day)
                    ),
                    "prediction": float(
                        store + day - 1
                        + (100 if family == "BEVERAGES" else 0)
                    ),
                })

    return pd.DataFrame(records)


@pytest.fixture
def service(snapshot):
    return ForecastService(
        snapshot,
        PlanningCalendar(date(2017, 8, 15)),
        source="synthetic-test-snapshot",
    )


def test_tomorrow(service):
    result = service.sales_for_store(12, "dairy", "tomorrow")

    assert result["total_predicted_sales"] == 12.0
    assert result["daily_predictions"][0]["date"] == "2017-08-16"


def test_next_week_ranking(service):
    result = service.analyze(
        "next week",
        ["store_nbr"],
        families=["DAIRY"],
    )

    assert [
        row["store_nbr"] for row in result["results"]
    ] == [44, 12]

    assert [
        row["predicted_sales"] for row in result["results"]
    ] == [364.0, 140.0]


def test_two_weeks(service):
    result = service.sales_for_store(
        12, "DAIRY", "next two weeks"
    )

    assert len(result["daily_predictions"]) == 14
    assert result["total_predicted_sales"] == 259.0


def test_mixed_product_units_rejected(service):
    with pytest.raises(ValueError, match="Group by family"):
        service.analyze("next week", ["store_nbr"])


def test_truncation_is_explicit(service):
    result = service.analyze(
        "next week",
        ["store_nbr"],
        families=["DAIRY"],
        top_n=1,
    )

    assert result["truncated"] is True
    assert result["groups_found"] == 2


def test_unknown_store(service):
    with pytest.raises(ValueError, match="No forecasts"):
        service.sales_for_store(99, "DAIRY", "tomorrow")


def test_duplicate_forecasts_rejected(snapshot):
    duplicated = pd.concat(
        [snapshot, snapshot.iloc[:1]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="Duplicate"):
        ForecastService(
            duplicated,
            PlanningCalendar(date(2017, 8, 15)),
            source="synthetic-test-snapshot",
        )


def test_incomplete_snapshot_rejected(snapshot):
    with pytest.raises(ValueError, match="Incomplete"):
        ForecastService(
            snapshot.iloc[1:],
            PlanningCalendar(date(2017, 8, 15)),
            source="synthetic-test-snapshot",
        )
