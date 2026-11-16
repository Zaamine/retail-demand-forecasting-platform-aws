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
