"""Assemble the existing verified agent with bounded Bedrock access."""

import logging
from pathlib import Path

from mcp import Client

from agent.analysis_request import AnalysisRequestInterpreter
from agent.analysis_validator import AnalysisEvidenceValidator
from agent.evidence_validator import SalesEvidenceValidator
from agent.mcp_server import build_retail_planner_mcp_server
from agent.orchestrator import BedrockMCPOrchestrator
from agent.request_interpreter import SalesRequestInterpreter
from agent.verified_agent import VerifiedSalesAgent
from agent.verified_planner import VerifiedPlannerAgent
from knowledge.retrieval import BusinessKnowledgeRetriever


MODEL_ID = "eu.amazon.nova-micro-v1:0"


logger = logging.getLogger(__name__)


class BedrockDisabledError(RuntimeError):
    """Raised when model access is not authorized."""


class BedrockBudgetExhausted(RuntimeError):
    """Raised when the request exceeds its call limit."""


class ConverseBudget:
    """Require explicit activation and bound Converse calls per request."""

    def __init__(
        self, client, *, model_id=MODEL_ID,
        max_calls=6, enabled=False,
    ):
        if type(max_calls) is not int or not 2 <= max_calls <= 12:
            raise ValueError("max_calls must be between 2 and 12.")

        self.client = client
        self.model_id = model_id
        self.max_calls = max_calls
        self.enabled = enabled
        self.attempts = 0
        self.records = []

    def converse(self, **kwargs):
        if not self.enabled:
            raise BedrockDisabledError("Bedrock access is disabled.")

        if kwargs.get("modelId") != self.model_id:
            raise ValueError("Unexpected Bedrock model.")

        if self.attempts >= self.max_calls:
            raise BedrockBudgetExhausted("Bedrock call budget exhausted.")

        self.attempts += 1

        try:
            response = self.client.converse(**kwargs)
        except Exception:
            self.records.append({
                "status": "error",
                "input_tokens": 0,
                "output_tokens": 0,
            })
            raise

        usage = response.get("usage") or {}

        self.records.append({
            "status": "success",
            "input_tokens": usage.get("inputTokens", 0),
            "output_tokens": usage.get("outputTokens", 0),
        })

        return response


def build_bedrock_planner(
    project_root,
    forecasts,
    forecast_service,
    bedrock_client,
    *,
    enabled=False,
    max_calls=6,
    status_reader=None,
):
    """Inject a bounded client into the existing agent components."""
    root = Path(project_root)
    calendar = forecast_service.calendar
    families = sorted(forecasts["family"].unique())

    budget = ConverseBudget(
        bedrock_client,
        max_calls=max_calls,
        enabled=enabled,
    )

    orchestrator = BedrockMCPOrchestrator(
        bedrock_client=budget,
        model_id=MODEL_ID,
        calendar=calendar,
        max_calls=max_calls - 1,
    )

    sales_agent = VerifiedSalesAgent(
        interpreter=SalesRequestInterpreter(
            bedrock_client=budget,
            model_id=MODEL_ID,
            calendar=calendar,
            known_families=families,
        ),
        orchestrator=orchestrator,
        validator=SalesEvidenceValidator(forecast_service),
    )

    planner = VerifiedPlannerAgent(
        sales_agent=sales_agent,
        analysis_interpreter=AnalysisRequestInterpreter(
            calendar=calendar,
            known_families=families,
        ),
        analysis_validator=AnalysisEvidenceValidator(
            forecast_service
        ),
        orchestrator=orchestrator,
    )

    knowledge = BusinessKnowledgeRetriever(root)

    def local_status():
        return {
            "status": "not_checked",
            "message": (
                "AWS workflow status is not available "
                "in this local application."
            ),
        }

    server = build_retail_planner_mcp_server(
        forecast_service=forecast_service,
        knowledge_retriever=knowledge,
        status_reader=(
            status_reader if status_reader is not None else local_status
        ),
    )

    return planner, server, budget


async def ask_bedrock_planner(question, planner, server):
    """Execute one MCP request, rejecting unsafe model failures."""
    try:
        async with Client(server, raise_exceptions=True) as client:
            return await planner.run(question, client)

    except BedrockDisabledError:
        return {
            "status": "blocked_execution",
            "answer": None,
            "reason": "Bedrock is disabled for this request.",
        }

    except BedrockBudgetExhausted:
        return {
            "status": "blocked_execution",
            "answer": None,
            "reason": "The model call budget was exhausted.",
        }

    except Exception:
        logger.exception("Retail Planner Bedrock execution failed")
        return {
            "status": "blocked_execution",
            "answer": None,
            "reason": "The request could not be completed safely.",
        }
