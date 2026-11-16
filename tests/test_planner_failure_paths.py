"""Offline failure-path tests for the verified Retail Planner."""

import asyncio
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from agent.planning_calendar import PlanningCalendar
from planner_app.bedrock_agent import (
    build_bedrock_planner,
    ask_bedrock_planner,
)
from planner_app.conversation import context_from_verified_result
from planner_app.live_quota import LocallyCappedBedrock


QUESTION = "How much dairy should store 44 expect to sell next week?"
ROOT = Path(__file__).resolve().parents[1]


class FakeForecastService:
    """Small trusted forecast fixture, independent of the model."""

    source = "offline_test_snapshot"

    def __init__(self):
        self.calendar = PlanningCalendar(date(2017, 8, 15))

    def sales_for_store(self, store_nbr, family, when):
        period = self.calendar.resolve(when)
        daily = [
            {
                "date": (
                    period.start + timedelta(days=offset)
                ).isoformat(),
                "predicted_sales": float(store_nbr + offset),
            }
            for offset in range(period.days)
        ]

        return {
            "store_nbr": store_nbr,
            "family": family,
            "period_start": period.start.isoformat(),
            "period_end": period.end.isoformat(),
            "daily_predictions": daily,
            "total_predicted_sales": sum(
                row["predicted_sales"] for row in daily
            ),
            "source": self.source,
        }


class ScriptedBedrock:
    """Return controlled model replies without AWS access."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        assert self.replies, "Unexpected additional Converse call."

        reply = self.replies.pop(0)

        if isinstance(reply, Exception):
            raise reply

        return {
            **reply,
            "usage": {
                "inputTokens": 10,
                "outputTokens": 5,
            },
        }


def contract_reply(store=44):
    return {
        "stopReason": "tool_use",
        "output": {
            "message": {
                "role": "assistant",
                "content": [{
                    "toolUse": {
                        "toolUseId": "contract-1",
                        "name": "submit_planning_contract",
                        "input": {
                            "intent": "sales_total",
                            "stores": [store],
                            "families": ["DAIRY"],
                            "period": "next week",
                            "requires_readiness": False,
                            "reason": "",
                        },
                    },
                }],
            },
        },
    }


def lookup_reply(store=44):
    return {
        "stopReason": "tool_use",
        "output": {
            "message": {
                "role": "assistant",
                "content": [{
                    "toolUse": {
                        "toolUseId": "lookup-1",
                        "name": "lookup_sales_forecast",
                        "input": {
                            "store_nbr": store,
                            "family": "DAIRY",
                            "when": "next week",
                        },
                    },
                }],
            },
        },
    }


def final_reply(text="Sales will be 999,999 units."):
    return {
        "stopReason": "end_turn",
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": text}],
            },
        },
    }


def run_request(bedrock_client):
    forecasts = pd.DataFrame({
        "family": ["DAIRY", "BEVERAGES"],
    })

    planner, server, budget = build_bedrock_planner(
        ROOT,
        forecasts,
        FakeForecastService(),
        bedrock_client,
        enabled=True,
        max_calls=6,
    )

    result = asyncio.run(
        ask_bedrock_planner(QUESTION, planner, server)
    )

    return result, budget


def assert_rejected(result):
    assert result["status"] not in (
        "verified_sales_evidence",
        "verified_analysis_evidence",
    )
    assert result["answer"] is None
    assert context_from_verified_result(result) is None


def test_model_cannot_change_explicit_store():
    fake = ScriptedBedrock([contract_reply(store=99)])

    result, budget = run_request(fake)

    assert result["status"] == "blocked_request"
    assert_rejected(result)
    assert budget.attempts == 1
    assert len(fake.calls) == 1


def test_model_text_without_mcp_evidence_is_rejected():
    fake = ScriptedBedrock([
        contract_reply(),
        final_reply(),
    ])

    result, budget = run_request(fake)

    assert result["status"] == "blocked_evidence"
    assert_rejected(result)
    assert result["model_run"]["tool_events"] == []
    assert budget.attempts == 2


def test_wrong_store_mcp_lookup_is_rejected():
    fake = ScriptedBedrock([
        contract_reply(),
        lookup_reply(store=45),
        final_reply(),
    ])

    result, budget = run_request(fake)

    assert result["status"] == "blocked_evidence"
    assert_rejected(result)
    assert len(result["model_run"]["tool_events"]) == 1
    assert budget.attempts == 3


def test_global_quota_blocks_without_provider_call(tmp_path):
    path = tmp_path / "quota"
    path.write_text("1", encoding="utf-8")

    fake = ScriptedBedrock([contract_reply()])
    capped = LocallyCappedBedrock(
        fake, path, max_calls=1
    )

    result, budget = run_request(capped)

    assert result["status"] == "blocked_execution"
    assert_rejected(result)
    assert budget.attempts == 1
    assert fake.calls == []
    assert path.read_text(encoding="utf-8") == "1"


def test_provider_exception_blocks_execution():
    fake = ScriptedBedrock([
        RuntimeError("Simulated Bedrock outage")
    ])

    result, budget = run_request(fake)

    assert result["status"] == "blocked_execution"
    assert_rejected(result)
    assert budget.attempts == 1
    assert budget.records[0]["status"] == "error"


def test_verified_mcp_evidence_overrides_model_prose():
    fake = ScriptedBedrock([
        contract_reply(),
        lookup_reply(store=44),
        final_reply(),
    ])

    result, budget = run_request(fake)

    assert result["status"] == "verified_sales_evidence"
    assert result["review"]["evidence_complete"] is True

    # Trusted fixture: 44 + 45 + ... + 50 = 329 units.
    assert "329.00 sales units" in result["answer"]
    assert "999,999" not in result["answer"]

    assert context_from_verified_result(result) == {
        "intent": "sales_total",
        "store_nbr": 44,
        "family": "DAIRY",
        "when": "2017-08-21/2017-08-27",
    }
    assert budget.attempts == 3


def modified_contract(field, value):
    from copy import deepcopy

    reply = deepcopy(contract_reply())
    payload = reply["output"]["message"]["content"][0]["toolUse"]["input"]

    if value is None:
        payload.pop(field)
    else:
        payload[field] = value

    return reply


def simulated_throttling():
    from botocore.exceptions import ClientError

    return ClientError(
        {
            "Error": {
                "Code": "ThrottlingException",
                "Message": "Simulated throttling",
            }
        },
        "Converse",
    )


def test_missing_contract_period_fails_closed():
    fake = ScriptedBedrock([modified_contract("period", None)])
    result, budget = run_request(fake)

    assert result["status"] == "blocked_request"
    assert_rejected(result)
    assert budget.attempts == 1


def test_wrong_contract_family_fails_closed():
    fake = ScriptedBedrock([
        modified_contract("families", ["BEVERAGES"])
    ])
    result, budget = run_request(fake)

    assert result["status"] == "blocked_request"
    assert_rejected(result)
    assert budget.attempts == 1


def test_wrong_contract_period_fails_closed():
    fake = ScriptedBedrock([
        modified_contract("period", "tomorrow")
    ])
    result, budget = run_request(fake)

    assert result["status"] == "blocked_request"
    assert_rejected(result)
    assert budget.attempts == 1


def test_unstructured_contract_fails_closed():
    fake = ScriptedBedrock([
        final_reply("The expected sales are 999,999 units.")
    ])
    result, budget = run_request(fake)

    assert result["status"] == "blocked_request"
    assert_rejected(result)
    assert budget.attempts == 1


def test_incomplete_orchestrator_response_fails_closed():
    incomplete = {
        "stopReason": "max_tokens",
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "999,999 units"}],
            }
        },
    }

    fake = ScriptedBedrock([
        contract_reply(),
        incomplete,
    ])

    result, budget = run_request(fake)

    assert result["status"] == "blocked_execution"
    assert_rejected(result)
    assert budget.attempts == 2


def test_malformed_orchestrator_response_fails_closed():
    fake = ScriptedBedrock([
        contract_reply(),
        {"stopReason": "tool_use"},
    ])

    result, budget = run_request(fake)

    assert result["status"] == "blocked_execution"
    assert_rejected(result)
    assert budget.attempts == 2


def test_unknown_mcp_tool_cannot_produce_sales_answer():
    reply = lookup_reply()
    reply["output"]["message"]["content"][0]["toolUse"][
        "name"
    ] = "not_a_registered_tool"

    fake = ScriptedBedrock([
        contract_reply(),
        reply,
        final_reply(),
    ])

    result, budget = run_request(fake)

    assert_rejected(result)
    assert budget.attempts <= 3


def test_invalid_mcp_period_cannot_produce_sales_answer():
    reply = lookup_reply()
    reply["output"]["message"]["content"][0]["toolUse"][
        "input"
    ]["when"] = "2030-01-01"

    fake = ScriptedBedrock([
        contract_reply(),
        reply,
        final_reply(),
    ])

    result, budget = run_request(fake)

    assert_rejected(result)
    assert budget.attempts <= 3


def test_throttling_during_extraction_fails_closed():
    fake = ScriptedBedrock([
        simulated_throttling(),
    ])

    result, budget = run_request(fake)

    assert result["status"] == "blocked_request"
    assert_rejected(result)
    assert budget.attempts == 1
    assert budget.records[0]["status"] == "error"


def test_throttling_during_orchestration_fails_closed():
    fake = ScriptedBedrock([
        contract_reply(),
        simulated_throttling(),
    ])

    result, budget = run_request(fake)

    assert result["status"] == "blocked_execution"
    assert_rejected(result)
    assert budget.attempts == 2
    assert budget.records[-1]["status"] == "error"


def test_global_quota_exhausted_mid_request(tmp_path):
    from planner_app.live_quota import LocallyCappedBedrock

    fake = ScriptedBedrock([contract_reply()])
    path = tmp_path / "quota"

    capped = LocallyCappedBedrock(
        fake,
        path,
        max_calls=1,
    )

    result, budget = run_request(capped)

    assert result["status"] == "blocked_execution"
    assert_rejected(result)

    # Contract extraction succeeded, but the next call was denied.
    assert budget.attempts == 2
    assert len(fake.calls) == 1
    assert path.read_text(encoding="utf-8") == "1"


def test_parallel_sessions_share_persistent_quota(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock
    from planner_app.live_quota import (
        LiveQuotaExceeded,
        LocallyCappedBedrock,
    )

    class CountingBedrock:
        def __init__(self):
            self.calls = 0
            self.lock = Lock()

        def converse(self, **kwargs):
            with self.lock:
                self.calls += 1

            return {"usage": {"inputTokens": 10}}

    fake = CountingBedrock()
    path = tmp_path / "shared_quota"

    def attempt_call(_):
        client = LocallyCappedBedrock(
            fake,
            path,
            max_calls=3,
        )

        try:
            client.converse(modelId="test")
            return True
        except LiveQuotaExceeded:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(attempt_call, range(12)))

    assert sum(outcomes) == 3
    assert fake.calls == 3
    assert path.read_text(encoding="utf-8") == "3"
