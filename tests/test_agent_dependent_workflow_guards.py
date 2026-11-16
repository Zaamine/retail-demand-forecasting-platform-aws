"""Offline failure-injection tests for dependent MCP analysis."""

import asyncio
from datetime import date
from types import SimpleNamespace

import pytest

from agent.analysis_request import AnalysisRequestInterpreter
from agent.planning_calendar import PlanningCalendar
from agent.verified_planner import VerifiedPlannerAgent


QUESTION = (
    "Which dairy store should expect the most sales "
    "next week, and which day will be its busiest?"
)


class ForbiddenBedrock:
    def __init__(self):
        self.calls = 0

    async def run(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("Bedrock must not be called.")


class FakeMCP:
    def __init__(self, failure=None, failing_round=1):
        self.calls = []
        self.failure = failure
        self.failing_round = failing_round

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        round_number = len(self.calls)

        if (
            self.failure is not None
            and round_number == self.failing_round
        ):
            if self.failure == "exception":
                raise RuntimeError("Simulated MCP failure")

            if self.failure == "malformed":
                return SimpleNamespace(
                    structured_content=["invalid"],
                    content=[],
                    is_error=False,
                )

            return SimpleNamespace(
                structured_content={
                    "status": "tool_error"
                },
                content=[],
                is_error=True,
            )

        if round_number == 1:
            payload = {
                "status": "available",
                "results": [{"store_nbr": 44}],
            }
        else:
            payload = {
                "status": "available",
                "results": [{
                    "target_date": "2017-08-27"
                }],
            }

        return SimpleNamespace(
            structured_content=payload,
            content=[],
            is_error=False,
        )


class FakeValidator:
    def __init__(
        self, ranking_valid=True, daily_valid=True,
        leading_store=44,
    ):
        self.ranking_valid = ranking_valid
        self.daily_valid = daily_valid
        self.leading_store = leading_store
        self.stages = []

    def validate(self, run, **kwargs):
        daily = kwargs["require_busiest"]
        self.stages.append(daily)

        complete = (
            len(run["tool_events"]) == 2
            and all(
                event["status"] == "success"
                for event in run["tool_events"]
            )
        )

        valid = (
            self.daily_valid and complete
            if daily else self.ranking_valid
        )

        return {
            "evidence_valid": valid,
            "leading_store": self.leading_store,
            "verified_answer": (
                "Independently verified answer"
                if valid else None
            ),
            "issues": [] if valid else [
                "Simulated invalid evidence."
            ],
        }


def make_system(client, validator):
    interpreter = AnalysisRequestInterpreter(
        calendar=PlanningCalendar(date(2017, 8, 15)),
        known_families=["DAIRY"],
    )
    orchestrator = ForbiddenBedrock()

    planner = VerifiedPlannerAgent(
        sales_agent=ForbiddenBedrock(),
        analysis_interpreter=interpreter,
        analysis_validator=validator,
        orchestrator=orchestrator,
    )

    return planner, orchestrator


def execute(client, validator):
    planner, orchestrator = make_system(
        client, validator
    )
    result = asyncio.run(
        planner.run(QUESTION, client)
    )
    assert orchestrator.calls == 0
    return result


@pytest.mark.parametrize(
    "failure",
    ["exception", "malformed", "tool_error"],
)
def test_first_tool_failure_blocks_second_call(failure):
    client = FakeMCP(failure=failure)
    validator = FakeValidator()

    result = execute(client, validator)

    assert result["status"] == "blocked_execution"
    assert result["answer"] is None
    assert len(client.calls) == 1
    assert validator.stages == []


def test_unverified_ranking_blocks_daily_analysis():
    client = FakeMCP()
    validator = FakeValidator(ranking_valid=False)

    result = execute(client, validator)

    assert result["status"] == "blocked_evidence"
    assert result["answer"] is None
    assert len(client.calls) == 1
    assert validator.stages == [False, True]


@pytest.mark.parametrize(
    "leading_store",
    [None, -1, True, "44"],
)
def test_invalid_verified_store_blocks_daily_analysis(
    leading_store,
):
    client = FakeMCP()
    validator = FakeValidator(
        leading_store=leading_store
    )

    result = execute(client, validator)

    assert result["status"] == "blocked_execution"
    assert result["answer"] is None
    assert len(client.calls) == 1
    assert validator.stages == [False]


@pytest.mark.parametrize(
    "failure",
    ["exception", "malformed", "tool_error"],
)
def test_second_tool_failure_blocks_final_answer(failure):
    client = FakeMCP(
        failure=failure, failing_round=2
    )
    validator = FakeValidator()

    result = execute(client, validator)

    assert result["status"] == "blocked_execution"
    assert result["answer"] is None
    assert len(client.calls) == 2
    assert validator.stages == [False]


def test_final_evidence_rejection_blocks_answer():
    client = FakeMCP()
    validator = FakeValidator(daily_valid=False)

    result = execute(client, validator)

    assert result["status"] == "blocked_evidence"
    assert result["answer"] is None
    assert len(client.calls) == 2
    assert validator.stages == [False, True]


def test_verified_workflow_has_zero_converse_calls():
    client = FakeMCP()
    validator = FakeValidator()

    result = execute(client, validator)

    assert result["status"] == "verified_analysis_evidence"
    assert result["answer"] == "Independently verified answer"
    assert result["model_run"]["converse_calls"] == 0
    assert (
        result["model_run"]["execution_mode"]
        == "deterministic_mcp"
    )
    assert len(client.calls) == 2
    assert validator.stages == [False, True]
