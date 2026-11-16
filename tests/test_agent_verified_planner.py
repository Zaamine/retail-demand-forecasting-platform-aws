"""Offline routing and fail-closed tests for the verified planner."""

import asyncio
from datetime import date

import pytest

from agent.analysis_request import AnalysisRequestInterpreter
from agent.planning_calendar import PlanningCalendar
from agent.verified_planner import VerifiedPlannerAgent


@pytest.fixture
def interpreter():
    return AnalysisRequestInterpreter(
        calendar=PlanningCalendar(date(2017, 8, 15)),
        known_families=["DAIRY", "BEVERAGES", "BABY CARE"],
    )


@pytest.mark.parametrize(
    "question,intent,top_n",
    [
        (
            "Which three stores should expect the most dairy "
            "sales next week?",
            "store_ranking", 3,
        ),
        (
            "Which store should expect the most dairy sales "
            "next week?",
            "store_ranking", 1,
        ),
        (
            "Which dairy store should expect the most sales "
            "next week, and which day will be its busiest?",
            "dependent_peak_day", 1,
        ),
    ],
)
def test_supported_analyses(
    interpreter, question, intent, top_n
):
    result = interpreter.extract(question)

    assert result["status"] == "supported"
    assert result["intent"] == intent
    assert result["top_n"] == top_n
    assert result["family"] == "DAIRY"
    assert result["period_start"] == "2017-08-21"
    assert result["period_end"] == "2017-08-27"


@pytest.mark.parametrize(
    "question",
    [
        "Which three stores have the most dairy sales?",
        "Which three stores have the most dairy and "
        "beverages sales next week?",
        "Which three stores have the most dairy sales "
        "next week and tomorrow?",
        "Which three stores generate the highest dairy "
        "revenue next week?",
        "Which store 44 or store 49 has the most dairy "
        "sales next week?",
    ],
)
def test_unsupported_analyses_are_rejected(interpreter, question):
    assert interpreter.extract(question)["status"] == "unsupported"


def test_sales_request_uses_existing_route(interpreter):
    result = interpreter.extract(
        "How much dairy should store 44 expect to sell next week?"
    )

    assert result["status"] == "not_applicable"


class FakeSalesAgent:
    def __init__(self):
        self.calls = 0

    async def run(self, question, client):
        self.calls += 1

        return {
            "status": "verified_sales_evidence",
            "answer": "Store 44: 19,479.29 sales units",
            "model_run": {"status": "completed"},
            "review": {"evidence_complete": True},
        }


class FakeOrchestrator:
    def __init__(self, status="completed"):
        self.status = status
        self.calls = 0

    async def run(self, question, client):
        self.calls += 1

        return {
            "status": self.status,
            "answer": "Store 50 leads the dairy ranking.",
            "tool_events": [],
        }


class FakeAnalysisValidator:
    def __init__(self, valid=True):
        self.valid = valid
        self.calls = []

    def validate(self, run, **kwargs):
        self.calls.append(kwargs)

        return {
            "evidence_valid": self.valid,
            "verified_answer": (
                "1. Store 44: 19,479.29 sales units"
                if self.valid else None
            ),
            "issues": [] if self.valid else [
                "Missing scoped evidence."
            ],
        }


def make_planner(interpreter, *, valid=True, model_status="completed"):
    sales = FakeSalesAgent()
    orchestrator = FakeOrchestrator(model_status)
    validator = FakeAnalysisValidator(valid)

    planner = VerifiedPlannerAgent(
        sales_agent=sales,
        analysis_interpreter=interpreter,
        analysis_validator=validator,
        orchestrator=orchestrator,
    )

    return planner, sales, orchestrator, validator


def test_sales_is_routed_without_analysis(interpreter):
    planner, sales, orchestrator, validator = make_planner(
        interpreter
    )

    result = asyncio.run(planner.run(
        "How much dairy should store 44 sell next week?",
        None,
    ))

    assert result["route"] == "sales"
    assert result["status"] == "verified_sales_evidence"
    assert sales.calls == 1
    assert orchestrator.calls == 0
    assert validator.calls == []


def test_ranking_uses_verified_answer(interpreter):
    planner, sales, orchestrator, validator = make_planner(
        interpreter
    )

    result = asyncio.run(planner.run(
        "Which three stores should expect the most "
        "dairy sales next week?",
        None,
    ))

    assert result["status"] == "verified_analysis_evidence"
    assert result["route"] == "analysis"

    assert "Store 44" in result["answer"]
    assert "Store 50" not in result["answer"]

    # Preserve the incorrect model output for diagnostics.
    assert "Store 50" in result["model_run"]["answer"]

    assert validator.calls[0]["top_n"] == 3
    assert validator.calls[0]["require_busiest"] is False
    assert sales.calls == 0


def test_dependent_request_without_mcp_is_blocked(interpreter):
    """Missing MCP access must not produce a verified answer."""
    planner, sales, orchestrator, validator = make_planner(
        interpreter
    )

    result = asyncio.run(planner.run(
        "Which dairy store should expect the most sales "
        "next week, and which day will be its busiest?",
        None,
    ))

    assert result["status"] == "blocked_execution"
    assert result["answer"] is None
    assert result["model_run"]["converse_calls"] == 0
    assert sales.calls == 0
    assert orchestrator.calls == 0
    assert validator.calls == []


def test_incomplete_analysis_evidence_is_blocked(interpreter):
    planner, _, _, _ = make_planner(
        interpreter, valid=False
    )

    result = asyncio.run(planner.run(
        "Which three stores should expect the most "
        "dairy sales next week?",
        None,
    ))

    assert result["status"] == "blocked_evidence"
    assert result["answer"] is None


def test_bedrock_failure_is_not_approved(interpreter):
    planner, _, _, validator = make_planner(
        interpreter, model_status="bedrock_error"
    )

    result = asyncio.run(planner.run(
        "Which three stores should expect the most "
        "dairy sales next week?",
        None,
    ))

    assert result["status"] == "blocked_execution"
    assert result["answer"] is None
    assert validator.calls == []


def test_ambiguous_analysis_never_calls_bedrock(interpreter):
    planner, sales, orchestrator, validator = make_planner(
        interpreter
    )

    result = asyncio.run(planner.run(
        "Which stores have the most dairy sales next week?",
        None,
    ))

    assert result["status"] == "blocked_request"
    assert result["answer"] is None
    assert sales.calls == 0
    assert orchestrator.calls == 0
    assert validator.calls == []


@pytest.mark.parametrize("leading_store", [44, 49])
def test_dependent_workflow_uses_verified_leading_store(
    interpreter, leading_store
):
    """Dependent analysis must use the validated ranking result."""

    from types import SimpleNamespace

    class RecordingMCPClient:
        def __init__(self):
            self.calls = []

        async def call_tool(self, name, arguments):
            self.calls.append({
                "name": name,
                "arguments": arguments,
            })

            if len(self.calls) == 1:
                payload = {
                    "status": "available",
                    "results": [
                        {"store_nbr": leading_store}
                    ],
                }
            else:
                payload = {
                    "status": "available",
                    "results": [
                        {"target_date": "2017-08-27"}
                    ],
                }

            return SimpleNamespace(
                is_error=False,
                structured_content=payload,
                content=[],
            )

    class RecordingValidator:
        def __init__(self):
            self.calls = []

        def validate(self, run, **kwargs):
            self.calls.append({
                "run": run,
                **kwargs,
            })

            return {
                "evidence_valid": True,
                "issues": [],
                "leading_store": leading_store,
                "busiest_day": (
                    "2017-08-27"
                    if kwargs["require_busiest"]
                    else None
                ),
                "verified_answer": (
                    f"Verified Store {leading_store}"
                ),
            }

    planner, sales, orchestrator, _ = make_planner(
        interpreter
    )

    client = RecordingMCPClient()
    validator = RecordingValidator()
    planner.analysis_validator = validator

    result = asyncio.run(planner.run(
        "Which dairy store should expect the most sales "
        "next week, and which day will be its busiest?",
        client,
    ))

    assert result["status"] == "verified_analysis_evidence"
    assert sales.calls == 0
    assert orchestrator.calls == 0

    assert len(client.calls) == 2

    first, second = client.calls

    assert first["name"] == "analyze_sales_forecasts"
    assert first["arguments"]["group_by"] == [
        "store_nbr", "family"
    ]
    assert first["arguments"]["top_n"] == 1
    assert first["arguments"].get("stores") is None

    assert second["name"] == "analyze_sales_forecasts"
    assert second["arguments"]["group_by"] == [
        "target_date"
    ]
    assert second["arguments"]["stores"] == [leading_store]
    assert second["arguments"]["top_n"] == 1

    assert [
        call["require_busiest"]
        for call in validator.calls
    ] == [False, True]

    assert result["model_run"]["converse_calls"] == 0
