"""Offline integration tests for the four-tool MCP server."""

import asyncio
import json
from datetime import date, timedelta

import pandas as pd

from mcp import Client

from agent.forecast_service import ForecastService
from agent.mcp_server import build_retail_planner_mcp_server
from agent.planning_calendar import PlanningCalendar


class FakeKnowledgeRetriever:
    def build_context(self, question, top_k=3):
        return {
            "evidence": [{
                "passage_id": "test::promotions",
                "section": "Promotions",
                "content": (
                    "Planned discounts are not explicitly included "
                    "in the current sales estimates."
                ),
                "source_path": "test/business_knowledge.md",
                "source_sha256": "synthetic-test-hash",
            }]
        }


def fake_status_reader():
    return {
        "status": "ready_for_historical_planning",
        "planning_period": {
            "start": "2017-08-21",
            "end": "2017-08-27",
        },
        "published_estimates": {
            "verified": True,
        },
    }


def make_server():
    records = []

    for store in (12, 44, 49):
        for day in range(1, 15):
            records.append({
                "store_nbr": store,
                "family": "DAIRY",
                "forecast_origin": "2017-08-15",
                "target_date": (
                    date(2017, 8, 15)
                    + timedelta(days=day)
                ),
                "prediction": float(store + day),
            })

    service = ForecastService(
        pd.DataFrame(records),
        PlanningCalendar(date(2017, 8, 15)),
        source="synthetic-test-snapshot",
    )

    return build_retail_planner_mcp_server(
        forecast_service=service,
        knowledge_retriever=FakeKnowledgeRetriever(),
        status_reader=fake_status_reader,
    )


def decode_result(result):
    assert not result.is_error
    return json.loads(result.content[0].text)


async def exercise_mcp_server():
    server = make_server()

    async with Client(server, raise_exceptions=True) as client:
        tools = await client.list_tools()

        names = {tool.name for tool in tools.tools}

        assert names == {
            "lookup_sales_forecast",
            "analyze_sales_forecasts",
            "consult_business_knowledge",
            "check_planning_data_status",
        }

        lookup = decode_result(await client.call_tool(
            "lookup_sales_forecast",
            {
                "store_nbr": 12,
                "family": "dairy",
                "when": "next two weeks",
            },
        ))

        assert lookup["status"] == "available"
        assert len(lookup["data"]["daily_predictions"]) == 14

        ranking = decode_result(await client.call_tool(
            "analyze_sales_forecasts",
            {
                "when": "next week",
                "group_by": ["store_nbr"],
                "families": ["DAIRY"],
                "top_n": 3,
            },
        ))

        assert ranking["status"] == "available"
        assert [
            row["store_nbr"] for row in ranking["results"]
        ] == [49, 44, 12]

        knowledge = decode_result(await client.call_tool(
            "consult_business_knowledge",
            {
                "question": (
                    "Does the planned dairy discount affect "
                    "the expected sales numbers?"
                )
            },
        ))

        assert knowledge["status"] == "available"
        assert knowledge["evidence"][0]["section"] == "Promotions"
        assert knowledge["evidence"][0]["source_path"]

        status = decode_result(await client.call_tool(
            "check_planning_data_status", {}
        ))

        assert status["status"] == "ready_for_historical_planning"
        assert status["published_estimates"]["verified"]

        invalid = decode_result(await client.call_tool(
            "lookup_sales_forecast",
            {
                "store_nbr": 12,
                "family": "DAIRY",
                "when": "2017-08-30",
            },
        ))

        assert invalid["status"] == "invalid_request"


def test_four_tool_mcp_server():
    asyncio.run(exercise_mcp_server())
