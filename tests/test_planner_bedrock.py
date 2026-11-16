"""Offline tests for bounded Bedrock integration."""

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from agent.planning_calendar import PlanningCalendar
from planner_app.bedrock_agent import (
    MODEL_ID,
    ConverseBudget,
    build_bedrock_planner,
)


class FakeBedrock:
    def __init__(self):
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "usage": {
                "inputTokens": 100,
                "outputTokens": 25,
            }
        }


def test_disabled_by_default():
    fake = FakeBedrock()
    budget = ConverseBudget(fake)

    with pytest.raises(RuntimeError, match="disabled"):
        budget.converse(modelId=MODEL_ID)

    assert budget.attempts == 0
    assert fake.calls == []


def test_budget_counts_calls_and_tokens():
    fake = FakeBedrock()
    budget = ConverseBudget(
        fake, max_calls=2, enabled=True
    )

    budget.converse(modelId=MODEL_ID)
    budget.converse(modelId=MODEL_ID)

    assert budget.attempts == 2
    assert len(fake.calls) == 2
    assert sum(
        record["input_tokens"] for record in budget.records
    ) == 200

    with pytest.raises(RuntimeError, match="exhausted"):
        budget.converse(modelId=MODEL_ID)

    assert len(fake.calls) == 2


def test_wrong_model_is_rejected():
    fake = FakeBedrock()
    budget = ConverseBudget(fake, enabled=True)

    with pytest.raises(ValueError, match="Unexpected"):
        budget.converse(modelId="other-model")

    assert fake.calls == []
    assert budget.attempts == 0


def test_planner_assembly_makes_no_model_calls():
    fake = FakeBedrock()
    root = Path(__file__).resolve().parents[1]

    forecasts = pd.DataFrame({"family": ["DAIRY"]})
    service = SimpleNamespace(
        calendar=PlanningCalendar(date(2017, 8, 15))
    )

    planner, server, budget = build_bedrock_planner(
        project_root=root,
        forecasts=forecasts,
        forecast_service=service,
        bedrock_client=fake,
    )

    assert planner.sales_agent.interpreter.bedrock is budget
    assert planner.orchestrator.bedrock is budget
    assert planner.sales_agent.orchestrator is planner.orchestrator
    assert server is not None
    assert budget.enabled is False
    assert budget.attempts == 0
    assert fake.calls == []


def test_disabled_sales_request_fails_closed():
    import asyncio
    from planner_app.bedrock_agent import ask_bedrock_planner

    fake = FakeBedrock()
    root = Path(__file__).resolve().parents[1]
    forecasts = pd.DataFrame({"family": ["DAIRY"]})
    service = SimpleNamespace(
        calendar=PlanningCalendar(date(2017, 8, 15))
    )

    planner, server, budget = build_bedrock_planner(
        root, forecasts, service, fake,
        enabled=False,
    )

    result = asyncio.run(
        ask_bedrock_planner(
            "How much dairy should store 44 expect to sell next week?",
            planner,
            server,
        )
    )

    assert result["status"] == "blocked_execution"
    assert result["answer"] is None
    assert budget.attempts == 0
    assert fake.calls == []


def test_exhausted_budget_fails_closed():
    import asyncio
    from planner_app.bedrock_agent import ask_bedrock_planner

    fake = FakeBedrock()
    root = Path(__file__).resolve().parents[1]
    forecasts = pd.DataFrame({"family": ["DAIRY"]})
    service = SimpleNamespace(
        calendar=PlanningCalendar(date(2017, 8, 15))
    )

    planner, server, budget = build_bedrock_planner(
        root, forecasts, service, fake,
        enabled=True,
        max_calls=2,
    )

    budget.converse(modelId=MODEL_ID)
    budget.converse(modelId=MODEL_ID)

    result = asyncio.run(
        ask_bedrock_planner(
            "How much dairy should store 44 expect to sell next week?",
            planner,
            server,
        )
    )

    assert result["status"] == "blocked_execution"
    assert result["answer"] is None
    assert budget.attempts == 2
    assert len(fake.calls) == 2
