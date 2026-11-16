"""Bedrock Converse orchestrator for MCP-provided Retail Planner tools.

This is an execution engine, NOT a business-answer validator.
"""

import json

from botocore.exceptions import BotoCoreError, ClientError


BUSINESS_ERRORS = {
    "invalid_request",
    "unsupported_date_expression",
    "outside_forecast_horizon",
    "rejected_request",
    "tool_error",
}


def mcp_tool_config(tools):
    """Adapt MCP input schemas for Bedrock's client-side tool calling."""
    specs = []

    for tool in tools:
        schema = dict(tool.input_schema)
        schema.pop("title", None)

        if schema.get("type") != "object":
            raise ValueError(
                f"Invalid MCP input schema for {tool.name}"
            )

        specs.append({
            "toolSpec": {
                "name": tool.name,
                "description": (
                    tool.description or f"Call {tool.name}"
                ),
                "inputSchema": {"json": schema},
            }
        })

    if not specs:
        raise ValueError("No MCP tools were discovered.")

    return {"tools": specs}


def decode_mcp_result(result):
    """Prefer structured MCP content, with JSON-text compatibility."""

    structured = getattr(result, "structured_content", None)

    if isinstance(structured, dict):
        return structured

    texts = [
        item.text
        for item in getattr(result, "content", [])
        if getattr(item, "type", None) == "text"
    ]

    if not getattr(result, "is_error", False) and len(texts) == 1:
        try:
            payload = json.loads(texts[0])

            if isinstance(payload, dict):
                return payload

        except (ValueError, TypeError):
            pass

    return {
        "status": "tool_error",
        "message": (
            " ".join(texts)
            or "MCP returned no usable JSON object."
        ),
    }


class BedrockMCPOrchestrator:
    """Run one conversation using dynamically discovered MCP tools."""

    def __init__(
        self,
        bedrock_client,
        model_id,
        calendar,
        *,
        max_calls=8,
    ):
        if type(max_calls) is not int or max_calls < 1:
            raise ValueError("max_calls must be a positive integer.")

        self.bedrock = bedrock_client
        self.model_id = model_id
        self.max_calls = max_calls

        full_period = calendar.resolve("full horizon")

        self.system_prompt = (
            "You assist retail planners in plain business English. "
            f"The simulated business date is {calendar.business_date}. "
            f"Published sales estimates cover {full_period.start} "
            f"through {full_period.end}. "
            "Next week means the next calendar Monday through Sunday. "
            "Use MCP evidence for factual claims and sales quantities. "
            "Sales estimates represent sales volumes, not money, "
            "inventory on hand, or guaranteed customer demand. "
            "Do not infer delivery quantities without inventory and "
            "other operational information. "
            "When investigating a specific store, retain that store "
            "filter in any subsequent analysis."
        )

    async def run(self, question, mcp_client):
        """Return model output and traces without approving factual claims."""

        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be nonempty.")

        try:
            discovered = await mcp_client.list_tools()
            tool_config = mcp_tool_config(discovered.tools)
        except Exception as exc:
            return {
                "status": "tool_discovery_error",
                "answer": None,
                "error": (
                    "MCP tool discovery failed: "
                    f"{type(exc).__name__}"
                ),
                "converse_calls": 0,
                "tool_events": [],
                "model_text": [],
            }

        allowed = {tool.name for tool in discovered.tools}

        messages = [{
            "role": "user",
            "content": [{"text": question}],
        }]

        events = []
        model_text = []

        def finish(status, *, answer=None, error=None, calls=0):
            return {
                "status": status,
                "answer": answer,
                "error": error,
                "converse_calls": calls,
                "tool_events": events,
                "model_text": model_text,
            }

        for round_num in range(1, self.max_calls + 1):
            try:
                reply = self.bedrock.converse(
                    modelId=self.model_id,
                    system=[{"text": self.system_prompt}],
                    messages=messages,
                    toolConfig=tool_config,
                    inferenceConfig={
                        "maxTokens": 3000,
                        "temperature": 0.0,
                    },
                    additionalModelRequestFields={
                        "inferenceConfig": {"topK": 1}
                    },
                )

            except (ClientError, BotoCoreError) as exc:
                return finish(
                    "bedrock_error",
                    error=str(exc),
                    calls=round_num,
                )

            stop_reason = reply.get("stopReason")
            assistant = reply["output"]["message"]

            messages.append(assistant)
            blocks = assistant.get("content", [])

            for block in blocks:
                if "text" in block:
                    model_text.append({
                        "round": round_num,
                        "text": block["text"],
                    })

            requests = [
                block["toolUse"]
                for block in blocks
                if "toolUse" in block
            ]

            if stop_reason == "end_turn" and not requests:
                answer = "\n".join(
                    block["text"]
                    for block in blocks
                    if "text" in block
                ).strip()

                return finish(
                    "completed",
                    answer=answer,
                    calls=round_num,
                )

            if stop_reason != "tool_use" or not requests:
                return finish(
                    "incomplete",
                    error=(
                        f"Unexpected stop reason: {stop_reason}"
                    ),
                    calls=round_num,
                )

            if round_num == self.max_calls:
                return finish(
                    "round_limit",
                    calls=round_num,
                )

            responses = []

            for request in requests:
                name = request["name"]
                arguments = request.get("input") or {}

                if name not in allowed:
                    payload = {
                        "status": "tool_error",
                        "message": "Unknown tool.",
                    }
                    failed = True

                else:
                    try:
                        mcp_result = await mcp_client.call_tool(
                            name,
                            arguments,
                        )

                        payload = decode_mcp_result(mcp_result)
                        failed = bool(mcp_result.is_error)

                    except Exception as exc:
                        payload = {
                            "status": "tool_error",
                            "message": (
                                "MCP call failed: "
                                f"{type(exc).__name__}"
                            ),
                        }
                        failed = True

                failed = (
                    failed
                    or payload.get("status") in BUSINESS_ERRORS
                )

                status = "error" if failed else "success"

                events.append({
                    "round": round_num,
                    "name": name,
                    "arguments": arguments,
                    "status": status,
                    "result": payload,
                })

                responses.append({
                    "toolResult": {
                        "toolUseId": request["toolUseId"],
                        "content": [{"json": payload}],
                        "status": status,
                    }
                })

            messages.append({
                "role": "user",
                "content": responses,
            })

        return finish(
            "round_limit",
            calls=self.max_calls,
        )
