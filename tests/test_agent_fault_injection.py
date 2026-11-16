"""Offline Bedrock and MCP failure-injection tests."""

import asyncio
from datetime import date
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError

from agent.evidence_validator import SalesEvidenceValidator
from agent.orchestrator import BedrockMCPOrchestrator
from agent.planning_calendar import PlanningCalendar
from agent.planning_contract import PlanningContract

from test_agent_reliability_matrix import fixture_service


def use_tool():
    return {
        "stopReason": "tool_use",
        "output": {
            "message": {
                "role": "assistant",
                "content": [{
                    "toolUse": {
                        "toolUseId": "tool-1",
                        "name": "lookup_sales_forecast",
                        "input": {
                            "when": "next week",
                            "store_nbr": 44,
                            "family": "DAIRY",
                        },
                    }
                }],
            }
        },
    }


def finish(answer="Store 44 should sell 999,999 dairy units."):
    return {
        "stopReason": "end_turn",
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": answer}],
            }
        },
    }


class FakeBedrock:
    def __init__(self, items):
        self.items = list(items)
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        result = self.items.pop(0)

        if isinstance(result, Exception):
            raise result

        return result


class FakeMCP:
    def __init__(self, mode="business_error"):
        self.mode = mode

    async def list_tools(self):
        if self.mode == "discovery_error":
            raise ConnectionError("MCP registry unavailable")

        return SimpleNamespace(tools=[
            SimpleNamespace(
                name="lookup_sales_forecast",
                description="Sales lookup",
                input_schema={
                    "type": "object",
                    "properties": {
                        "when": {"type": "string"},
                        "store_nbr": {"type": "integer"},
                        "family": {"type": "string"},
                    },
                },
            )
        ])

    async def call_tool(self, name, args):
        if self.mode == "transport_error":
            raise ConnectionError("MCP connection lost")

        if self.mode == "malformed_json":
            return SimpleNamespace(
                is_error=False,
                structured_content=None,
                content=[
                    SimpleNamespace(
                        type="text",
                        text="not valid JSON",
                    )
                ],
            )

        return SimpleNamespace(
            is_error=False,
            structured_content={
                "status": "invalid_request",
                "message": "Invalid date range.",
            },
            content=[],
        )


def run_case(
    items,
    mcp_mode="business_error",
    max_calls=4,
):
    runner = BedrockMCPOrchestrator(
        bedrock_client=FakeBedrock(items),
        model_id="fixture-model",
        calendar=PlanningCalendar(date(2017, 8, 15)),
        max_calls=max_calls,
    )

    return asyncio.run(
        runner.run(
            "How much dairy should store 44 sell next week?",
            FakeMCP(mcp_mode),
        )
    )


def test_bedrock_model_failure_returns_no_answer():
    error = ClientError(
        {
            "Error": {
                "Code": "ModelErrorException",
                "Message": "Invalid tool sequence.",
            }
        },
        "Converse",
    )

    result = run_case([error])

    assert result["status"] == "bedrock_error"
    assert result["answer"] is None


@pytest.mark.parametrize(
    "mode",
    ["transport_error", "malformed_json", "business_error"],
)
def test_mcp_fault_is_reported_as_tool_error(mode):
    result = run_case(
        [use_tool(), finish()],
        mcp_mode=mode,
    )

    # Bedrock may still finish and generate an unsupported claim.
    assert result["status"] == "completed"
    assert result["tool_events"][0]["status"] == "error"
    assert result["answer"] is not None

    # Evidence validation must not approve that claim.
    contract = PlanningContract.create(
        PlanningCalendar(date(2017, 8, 15)),
        "next week",
        stores=[44],
        families=["DAIRY"],
    )

    review = SalesEvidenceValidator(
        fixture_service()
    ).validate(result, contract)

    assert review["evidence_complete"] is False


def test_round_limit_does_not_release_an_answer():
    result = run_case(
        [use_tool()],
        max_calls=1,
    )

    assert result["status"] == "round_limit"
    assert result["answer"] is None


def test_mcp_discovery_failure_is_bounded():
    result = run_case(
        [],
        mcp_mode="discovery_error",
    )

    assert result["status"] == "tool_discovery_error"
    assert result["answer"] is None
