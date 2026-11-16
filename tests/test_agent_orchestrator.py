"""Offline tests of Bedrock-MCP conversation control."""

import asyncio
from copy import deepcopy
from datetime import date
from types import SimpleNamespace

from botocore.exceptions import ClientError

from agent.orchestrator import BedrockMCPOrchestrator
from agent.planning_calendar import PlanningCalendar


class FakeMCP:
    def __init__(self):
        self.calls = []

    async def list_tools(self):
        return SimpleNamespace(tools=[
            SimpleNamespace(
                name="lookup_sales_forecast",
                description="Look up expected sales.",
                input_schema={
                    "type": "object",
                    "title": "ToolArguments",
                    "properties": {
                        "when": {"type": "string"}
                    },
                },
            )
        ])

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))

        if arguments["when"] == "invalid":
            payload = {
                "status": "invalid_request",
                "message": "Invalid planning period.",
            }
        else:
            payload = {
                "status": "available",
                "data": {"predicted_sales": 42},
            }

        return SimpleNamespace(
            is_error=False,
            structured_content=payload,
            content=[],
        )


class FakeBedrock:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        item = self.responses.pop(0)

        if isinstance(item, Exception):
            raise item

        return item


def response(stop_reason, content):
    return {
        "stopReason": stop_reason,
        "output": {
            "message": {
                "role": "assistant",
                "content": content,
            }
        },
    }


def tool_use(when):
    return {
        "toolUse": {
            "toolUseId": "test-tool-use-1",
            "name": "lookup_sales_forecast",
            "input": {"when": when},
        }
    }


def make_runner(bedrock, max_calls=8):
    return BedrockMCPOrchestrator(
        bedrock_client=bedrock,
        model_id="test-model",
        calendar=PlanningCalendar(date(2017, 8, 15)),
        max_calls=max_calls,
    )


def test_successful_tool_round_trip():
    bedrock = FakeBedrock([
        response("tool_use", [
            {"text": "<thinking>Use the tool.</thinking>"},
            tool_use("tomorrow"),
        ]),
        response("end_turn", [
            {"text": "Expected sales: 42 units."}
        ]),
    ])

    result = asyncio.run(
        make_runner(bedrock).run(
            "How much should the store expect to sell?",
            FakeMCP(),
        )
    )

    assert result["status"] == "completed"
    assert result["converse_calls"] == 2
    assert result["answer"] == "Expected sales: 42 units."
    assert result["tool_events"][0]["status"] == "success"

    assert "<thinking>" in result["model_text"][0]["text"]

    tool_response = bedrock.calls[1]["messages"][-1]
    assert (
        tool_response["content"][0]["toolResult"]["status"]
        == "success"
    )

    schema = (
        bedrock.calls[0]["toolConfig"]["tools"][0]
        ["toolSpec"]["inputSchema"]["json"]
    )
    assert "title" not in schema


def test_business_error_is_sent_to_bedrock():
    bedrock = FakeBedrock([
        response("tool_use", [tool_use("invalid")]),
        response("end_turn", [
            {"text": "Those dates are not available."}
        ]),
    ])

    result = asyncio.run(
        make_runner(bedrock).run(
            "Check the requested dates.",
            FakeMCP(),
        )
    )

    assert result["tool_events"][0]["status"] == "error"

    tool_response = bedrock.calls[1]["messages"][-1]
    assert (
        tool_response["content"][0]["toolResult"]["status"]
        == "error"
    )


def test_bedrock_error_is_captured():
    error = ClientError(
        {
            "Error": {
                "Code": "ModelErrorException",
                "Message": "Invalid tool-use sequence.",
            }
        },
        "Converse",
    )

    result = asyncio.run(
        make_runner(FakeBedrock([error])).run(
            "Check expected sales.",
            FakeMCP(),
        )
    )

    assert result["status"] == "bedrock_error"
    assert result["answer"] is None


def test_conversation_call_limit():
    mcp = FakeMCP()

    result = asyncio.run(
        make_runner(
            FakeBedrock([
                response("tool_use", [
                    tool_use("tomorrow")
                ])
            ]),
            max_calls=1,
        ).run("Check expected sales.", mcp)
    )

    assert result["status"] == "round_limit"
    assert result["answer"] is None
    assert mcp.calls == []
