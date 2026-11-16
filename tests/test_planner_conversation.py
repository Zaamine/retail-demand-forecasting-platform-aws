"""Regression tests for Retail Planner conversational memory."""

import pytest

from agent.analysis_request import AnalysisRequestInterpreter
from agent.planning_calendar import PlanningCalendar
from datetime import date

from planner_app.conversation import (
    context_from_verified_result,
    plan_turn,
)


@pytest.fixture
def verified_result():
    return {
        "status": "verified_analysis_evidence",
        "request": {
            "status": "supported",
            "intent": "dependent_peak_day",
            "family": "DAIRY",
            "when": "next week",
            "top_n": 1,
        },
        "review": {"evidence_valid": True},
    }


@pytest.fixture
def known_families():
    return ["DAIRY", "BEVERAGES"]


@pytest.mark.parametrize(
    "question",
    ["Hey!", "Hello", "Good morning!", "Good evening"],
)
def test_greetings(question):
    action = plan_turn(question)

    assert action["mode"] == "reply"
    assert action["answer"]


@pytest.mark.parametrize(
    "question",
    ["What is your purpose?", "How can you help me?", "Help"],
)
def test_capability_questions(question):
    action = plan_turn(question)

    assert action["mode"] == "reply"
    assert "retail planners" in action["answer"].lower()


def test_verified_analysis_creates_context(verified_result):
    context = context_from_verified_result(verified_result)

    assert context == {
        "intent": "dependent_peak_day",
        "family": "DAIRY",
        "when": "next week",
        "top_n": 1,
    }


@pytest.mark.parametrize(
    "status,evidence_valid",
    [
        ("blocked_evidence", False),
        ("blocked_execution", True),
        ("verified_analysis_evidence", False),
    ],
)
def test_unverified_analysis_cannot_create_context(
    verified_result, status, evidence_valid
):
    verified_result["status"] = status
    verified_result["review"]["evidence_valid"] = evidence_valid

    assert context_from_verified_result(verified_result) is None


def test_family_follow_up_uses_previous_scope(
    verified_result, known_families
):
    context = context_from_verified_result(verified_result)

    action = plan_turn(
        "And what about beverages?",
        context,
        known_families,
    )

    assert action["mode"] == "agent"
    assert action["used_context"]

    interpreter = AnalysisRequestInterpreter(
        PlanningCalendar(date(2017, 8, 15)),
        known_families,
    )

    request = interpreter.extract(action["question"])

    assert request["status"] == "supported"
    assert request["intent"] == "dependent_peak_day"
    assert request["family"] == "BEVERAGES"
    assert request["when"] == "next week"


def test_period_follow_up_preserves_family(
    verified_result, known_families
):
    context = context_from_verified_result(verified_result)

    action = plan_turn(
        "What about next two weeks?",
        context,
        known_families,
    )

    assert action["mode"] == "agent"

    interpreter = AnalysisRequestInterpreter(
        PlanningCalendar(date(2017, 8, 15)),
        known_families,
    )
    request = interpreter.extract(action["question"])

    assert request["status"] == "supported"
    assert request["family"] == "DAIRY"
    assert request["when"] == "next two weeks"


def test_cleared_context_cannot_resolve_follow_up(known_families):
    action = plan_turn(
        "And what about beverages?",
        context=None,
        known_families=known_families,
    )

    assert action["mode"] == "reply"


def test_unknown_family_is_rejected(
    verified_result, known_families
):
    context = context_from_verified_result(verified_result)

    action = plan_turn(
        "What about imaginary products?",
        context,
        known_families,
    )

    assert action["mode"] == "reply"


def test_full_request_does_not_need_memory(known_families):
    action = plan_turn(
        "Which dairy store has the most sales next week?",
        context=None,
        known_families=known_families,
    )

    assert action["mode"] == "agent"
    assert action["used_context"] is False


def make_verified_sales_result(stores=(44,), families=("DAIRY",)):
    from agent.planning_contract import PlanningContract

    calendar = PlanningCalendar(date(2017, 8, 15))
    contract = PlanningContract.create(
        calendar,
        "next week",
        stores=list(stores),
        families=list(families),
    )

    return {
        "status": "verified_sales_evidence",
        "extraction": {
            "status": "supported",
            "contract": contract,
        },
        "review": {
            "evidence_complete": True,
            "period_start": "2017-08-21",
            "period_end": "2017-08-27",
            "verified_totals": [
                {
                    "store_nbr": store,
                    "family": family,
                    "predicted_sales": 100.0,
                }
                for store in stores
                for family in families
            ],
        },
    }


def test_sales_result_creates_verified_context():
    context = context_from_verified_result(
        make_verified_sales_result()
    )

    assert context == {
        "intent": "sales_total",
        "store_nbr": 44,
        "family": "DAIRY",
        "when": "2017-08-21/2017-08-27",
    }


def test_sales_follow_up_preserves_store_and_dates():
    from agent.request_interpreter import SalesRequestInterpreter

    context = context_from_verified_result(
        make_verified_sales_result()
    )

    action = plan_turn(
        "And what about beverages?",
        context,
        ["DAIRY", "BEVERAGES"],
    )

    assert action["mode"] == "agent"
    assert action["used_context"]

    interpreter = SalesRequestInterpreter(
        bedrock_client=object(),
        model_id="unused",
        calendar=PlanningCalendar(date(2017, 8, 15)),
        known_families=["DAIRY", "BEVERAGES"],
    )

    anchors, issue = interpreter._literal_anchors(
        action["question"]
    )

    assert issue is None
    assert anchors["stores"] == {44}
    assert anchors["families"] == {"BEVERAGES"}
    assert anchors["period"].start == date(2017, 8, 21)
    assert anchors["period"].end == date(2017, 8, 27)


def test_sales_follow_up_can_change_period():
    from agent.request_interpreter import SalesRequestInterpreter

    context = context_from_verified_result(
        make_verified_sales_result()
    )

    action = plan_turn(
        "What about next two weeks?",
        context,
        ["DAIRY", "BEVERAGES"],
    )

    interpreter = SalesRequestInterpreter(
        bedrock_client=object(),
        model_id="unused",
        calendar=PlanningCalendar(date(2017, 8, 15)),
        known_families=["DAIRY", "BEVERAGES"],
    )

    anchors, issue = interpreter._literal_anchors(
        action["question"]
    )

    assert issue is None
    assert anchors["stores"] == {44}
    assert anchors["families"] == {"DAIRY"}
    assert anchors["period"].start == date(2017, 8, 16)
    assert anchors["period"].end == date(2017, 8, 29)


@pytest.mark.parametrize(
    "status,evidence_complete",
    [
        ("blocked_evidence", False),
        ("blocked_execution", True),
        ("verified_sales_evidence", False),
    ],
)
def test_unverified_sales_does_not_create_memory(
    status, evidence_complete
):
    result = make_verified_sales_result()
    result["status"] = status
    result["review"]["evidence_complete"] = evidence_complete

    assert context_from_verified_result(result) is None


def test_multi_store_sales_context_is_not_simplified():
    result = make_verified_sales_result(stores=(12, 44))

    assert context_from_verified_result(result) is None


def test_sales_follow_up_after_clear_requires_context():
    action = plan_turn(
        "And what about beverages?",
        context=None,
        known_families=["DAIRY", "BEVERAGES"],
    )

    assert action["mode"] == "reply"
