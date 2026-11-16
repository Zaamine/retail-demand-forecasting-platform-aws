"""Offline tests for sales request extraction and answer gating."""

import asyncio
from datetime import date

from agent.planning_calendar import PlanningCalendar
from agent.planning_contract import PlanningContract
from agent.request_interpreter import SalesRequestInterpreter
from agent.verified_agent import VerifiedSalesAgent


class FakeBedrock:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)

        return {
            "stopReason": "tool_use",
            "output": {
                "message": {
                    "content": [{
                        "toolUse": {
                            "name": "submit_planning_contract",
                            "input": self.payload,
                        }
                    }]
                }
            },
        }


def make_interpreter(**changes):
    payload = {
        "intent": "sales_total",
        "stores": [44],
        "families": ["DAIRY"],
        "period": "next week",
        "requires_readiness": True,
        "reason": "",
    }

    payload.update(changes)

    bedrock = FakeBedrock(payload)

    interpreter = SalesRequestInterpreter(
        bedrock,
        "test-model",
        PlanningCalendar(date(2017, 8, 15)),
        known_families=["DAIRY", "BABY CARE"],
    )

    return interpreter, bedrock


def test_extract_verified_scope_and_readiness():
    interpreter, bedrock = make_interpreter()

    result = interpreter.extract(
        "Are next week's sales numbers ready, and how much dairy "
        "should store 44 expect to sell?"
    )

    assert result["status"] == "supported"
    assert result["contract"].stores == frozenset({44})
    assert result["contract"].families == frozenset({"DAIRY"})
    assert result["requires_readiness"] is True

    assert bedrock.calls[0]["toolConfig"]["toolChoice"] == {
        "tool": {"name": "submit_planning_contract"}
    }


def test_extract_two_store_question():
    interpreter, _ = make_interpreter(
        stores=[12, 44],
        period="next two weeks",
        requires_readiness=False,
    )

    result = interpreter.extract(
        "How much dairy should stores 12 and 44 expect "
        "to sell over the next two weeks?"
    )

    assert result["status"] == "supported"
    assert result["contract"].stores == frozenset({12, 44})
    assert result["contract"].period.days == 14


def test_hallucinated_store_is_rejected():
    interpreter, _ = make_interpreter(stores=[45])

    result = interpreter.extract(
        "How much dairy should store 44 expect to sell next week?"
    )

    assert result["status"] == "invalid_extraction"


def test_wrong_period_is_rejected():
    interpreter, _ = make_interpreter(
        period="next two weeks"
    )

    result = interpreter.extract(
        "How much dairy should store 44 expect to sell next week?"
    )

    assert result["status"] == "invalid_extraction"


def test_ranking_is_not_misclassified_as_sales_total():
    interpreter, bedrock = make_interpreter()

    result = interpreter.extract(
        "Which store will have the highest dairy sales next week?"
    )

    assert result["status"] == "unsupported"
    assert bedrock.calls == []


def test_missing_period_is_not_guessed():
    interpreter, bedrock = make_interpreter()

    result = interpreter.extract(
        "How much dairy should store 44 expect to sell?"
    )

    assert result["status"] == "unsupported"
    assert bedrock.calls == []


class FakeInterpreter:
    def __init__(self, status="supported"):
        self.status = status

    def extract(self, question):
        if self.status != "supported":
            return {
                "status": self.status,
                "reason": "Unsupported.",
            }

        return {
            "status": "supported",
            "contract": PlanningContract.create(
                PlanningCalendar(date(2017, 8, 15)),
                "next week",
                stores=[44],
                families=["DAIRY"],
            ),
            "requires_readiness": True,
        }


class FakeOrchestrator:
    def __init__(self, status="completed"):
        self.status = status

    async def run(self, question, mcp_client):
        return {
            "status": self.status,
            "answer": "Store 50 will sell $19,479.29 of dairy.",
            "tool_events": [],
            "model_text": [],
        }


class FakeValidator:
    def __init__(self, complete=True):
        self.complete = complete
        self.readiness_requested = None

    def validate(self, run, contract, *, require_readiness=False):
        self.readiness_requested = require_readiness

        return {
            "evidence_complete": self.complete,
            "verified_totals": [{
                "store_nbr": 44,
                "family": "DAIRY",
                "predicted_sales": 19479.29,
            }],
            "period_start": "2017-08-21",
            "period_end": "2017-08-27",
            "readiness_verified": True,
        }


def test_verified_answer_ignores_wrong_model_prose():
    validator = FakeValidator(complete=True)

    agent = VerifiedSalesAgent(
        FakeInterpreter(),
        FakeOrchestrator(),
        validator,
    )

    result = asyncio.run(agent.run("question", None))

    assert result["status"] == "verified_sales_evidence"
    assert "Store 44" in result["answer"]
    assert "Store 50" not in result["answer"]
    assert validator.readiness_requested is True

    # Original incorrect model output remains available for diagnosis.
    assert "Store 50" in result["model_run"]["answer"]


def test_incomplete_evidence_is_blocked():
    agent = VerifiedSalesAgent(
        FakeInterpreter(),
        FakeOrchestrator(),
        FakeValidator(complete=False),
    )

    result = asyncio.run(agent.run("question", None))

    assert result["status"] == "blocked_evidence"
    assert result["answer"] is None


def test_incomplete_execution_is_blocked():
    agent = VerifiedSalesAgent(
        FakeInterpreter(),
        FakeOrchestrator(status="bedrock_error"),
        FakeValidator(),
    )

    result = asyncio.run(agent.run("question", None))

    assert result["status"] == "blocked_execution"
    assert result["answer"] is None


def test_unsupported_request_is_blocked_before_execution():
    agent = VerifiedSalesAgent(
        FakeInterpreter(status="unsupported"),
        FakeOrchestrator(),
        FakeValidator(),
    )

    result = asyncio.run(agent.run("question", None))

    assert result["status"] == "blocked_request"
    assert result["model_run"] is None

def test_false_clarification_is_reconciled():
    interpreter, bedrock = make_interpreter(
        intent="needs_clarification",
        reason="Clarify dairy and next week.",
        requires_readiness=True,
    )

    result = interpreter.extract(
        "How much dairy should store 44 expect to sell next week?"
    )

    assert result["status"] == "supported"
    assert result["contract"].stores == frozenset({44})
    assert result["contract"].families == frozenset({"DAIRY"})
    assert result["model_intent"] == "needs_clarification"
    assert result["clarification_overridden"] is True
    assert result["requires_readiness"] is False
    assert len(bedrock.calls) == 1


def test_false_clarification_two_stores():
    interpreter, _ = make_interpreter(
        intent="needs_clarification",
        stores=[12, 44],
        period="next two weeks",
    )

    result = interpreter.extract(
        "How much dairy should stores 12 and 44 expect "
        "to sell over the next two weeks?"
    )

    assert result["status"] == "supported"
    assert result["contract"].stores == frozenset({12, 44})
    assert result["contract"].period.days == 14
    assert result["clarification_overridden"] is True


def test_false_clarification_with_wrong_scope_is_rejected():
    interpreter, _ = make_interpreter(
        intent="needs_clarification",
        stores=[45],
    )

    result = interpreter.extract(
        "How much dairy should store 44 expect to sell next week?"
    )

    assert result["status"] == "invalid_extraction"


def test_alternative_store_selection_is_rejected():
    interpreter, bedrock = make_interpreter()

    result = interpreter.extract(
        "How much dairy should store 44 or store 49 "
        "expect to sell next week?"
    )

    assert result["status"] == "unsupported"
    assert bedrock.calls == []


def test_conditional_sales_question_is_rejected():
    interpreter, bedrock = make_interpreter()

    result = interpreter.extract(
        "How much dairy should store 44 expect to sell "
        "next week if the store opens late?"
    )

    assert result["status"] == "unsupported"
    assert bedrock.calls == []


def test_model_unsupported_intent_is_not_overridden():
    interpreter, _ = make_interpreter(
        intent="unsupported",
        reason="Unsupported business operation.",
    )

    result = interpreter.extract(
        "How much dairy should store 44 expect to sell next week?"
    )

    assert result["status"] == "unsupported"
