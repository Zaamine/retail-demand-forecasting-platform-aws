"""Assemble the Retail Planner for local, offline investigations."""

from pathlib import Path

from mcp import Client

from agent.analysis_request import AnalysisRequestInterpreter
from agent.analysis_validator import AnalysisEvidenceValidator
from agent.mcp_server import build_retail_planner_mcp_server
from agent.verified_planner import VerifiedPlannerAgent
from knowledge.retrieval import BusinessKnowledgeRetriever


class OfflineSalesRoute:
    """Reject requests that require Bedrock in offline mode."""

    async def run(self, question, client):
        return {
            "status": "blocked_request",
            "reason": "This request requires the Bedrock-enabled agent.",
            "answer": None,
            "model_run": None,
            "review": None,
        }


class OfflineOrchestrator:
    """Prevent model orchestration during local-only testing."""

    async def run(self, question, client):
        return {
            "status": "offline_disabled",
            "answer": None,
            "converse_calls": 0,
            "tool_events": [],
        }


def build_local_planner(project_root, forecasts, forecast_service):
    """Inject verified local dependencies into the existing agent."""
    root = Path(project_root)

    interpreter = AnalysisRequestInterpreter(
        calendar=forecast_service.calendar,
        known_families=forecasts["family"].unique(),
    )

    validator = AnalysisEvidenceValidator(forecast_service)

    planner = VerifiedPlannerAgent(
        sales_agent=OfflineSalesRoute(),
        analysis_interpreter=interpreter,
        analysis_validator=validator,
        orchestrator=OfflineOrchestrator(),
    )

    knowledge = BusinessKnowledgeRetriever(root)

    def local_status():
        return {
            "status": "not_checked",
            "message": "AWS operational status is not checked in local mode.",
        }

    server = build_retail_planner_mcp_server(
        forecast_service=forecast_service,
        knowledge_retriever=knowledge,
        status_reader=local_status,
    )

    return planner, server


async def ask_local_planner(question, planner, server):
    """Execute one question with a fresh local MCP session."""
    async with Client(server, raise_exceptions=True) as client:
        return await planner.run(question, client)
