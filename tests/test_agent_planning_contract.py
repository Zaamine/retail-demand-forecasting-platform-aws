"""Tests for Retail Planner business request contracts."""

from datetime import date

import pytest

from agent.planning_calendar import PlanningCalendar
from agent.planning_contract import PlanningContract


@pytest.fixture
def contract():
    return PlanningContract.create(
        calendar=PlanningCalendar(date(2017, 8, 15)),
        when="next two weeks",
        stores=[12, 44],
        families=["dairy"],
    )


def test_business_scope_is_normalized(contract):
    assert contract.stores == frozenset({12, 44})
    assert contract.families == frozenset({"DAIRY"})
    assert contract.period.days == 14
    assert len(contract.required_keys()) == 28


def test_partial_store_lookup_is_eligible(contract):
    result = contract.inspect_evidence_scope(
        period=contract.period,
        stores=[12],
        families=["DAIRY"],
    )
    assert result["eligible"]


def test_missing_store_filter_is_detected(contract):
    result = contract.inspect_evidence_scope(
        period=contract.period,
        stores=None,
        families=["DAIRY"],
    )
    assert not result["eligible"]
    assert "Store filter missing." in result["issues"]


def test_wrong_family_is_detected(contract):
    result = contract.inspect_evidence_scope(
        period=contract.period,
        stores=[12],
        families=["BEVERAGES"],
    )
    assert not result["eligible"]


def test_partial_period_is_eligible(contract):
    calendar = PlanningCalendar(date(2017, 8, 15))
    result = contract.inspect_evidence_scope(
        period=calendar.resolve("next week"),
        stores=[12, 44],
        families=["DAIRY"],
    )
    assert result["eligible"]


def test_global_aggregation_is_not_store_specific(contract):
    result = contract.inspect_evidence_scope(
        period=contract.period,
        stores=None,
        families=None,
    )
    assert not result["eligible"]
    assert len(result["issues"]) == 2


def test_invalid_store_selection():
    with pytest.raises(ValueError):
        PlanningContract.create(
            PlanningCalendar(date(2017, 8, 15)),
            "next week",
            stores=[],
        )


def test_open_scope_cannot_enumerate_keys():
    contract = PlanningContract.create(
        PlanningCalendar(date(2017, 8, 15)),
        "next week",
        families=["DAIRY"],
    )

    with pytest.raises(ValueError):
        contract.required_keys()
