"""Verified application entry point for supported Retail Planner requests."""

import json
from numbers import Integral


class VerifiedPlannerAgent:
    """Route supported requests and release only evidence-grounded answers."""

    def __init__(
        self,
        *,
        sales_agent,
        analysis_interpreter,
        analysis_validator,
        orchestrator,
    ):
        self.sales_agent = sales_agent
        self.analysis_interpreter = analysis_interpreter
        self.analysis_validator = analysis_validator
        self.orchestrator = orchestrator

    @staticmethod
    def _read_mcp_payload(raw):
        """Decode the structured or JSON-text MCP response."""
        payload = getattr(raw, "structured_content", None)

        if payload is None:
            blocks = [
                block.text
                for block in (getattr(raw, "content", None) or [])
                if isinstance(getattr(block, "text", None), str)
            ]

            if len(blocks) != 1:
                raise ValueError("Expected one MCP JSON block.")

            payload = json.loads(blocks[0])

        if not isinstance(payload, dict):
            raise ValueError("MCP payload must be a dictionary.")

        return payload

    async def _run_dependent_analysis(self, request, client):
        """Execute a verified two-stage MCP dependency."""

        run = {
            "status": "completed",
            "answer": "",
            "tool_events": [],
            "model_text": [],
            "converse_calls": 0,
            "error": None,
            "execution_mode": "deterministic_mcp",
        }

        async def call_analysis(arguments, round_number):
            try:
                raw = await client.call_tool(
                    "analyze_sales_forecasts", arguments
                )
                payload = self._read_mcp_payload(raw)
                status = (
                    "error"
                    if getattr(raw, "is_error", True)
                    else "success"
                )
            except Exception as exc:
                payload = {
                    "status": "tool_error",
                    "message": str(exc),
                }
                status = "error"

            run["tool_events"].append({
                "round": round_number,
                "name": "analyze_sales_forecasts",
                "arguments": arguments,
                "status": status,
                "result": payload,
            })

            if status != "success":
                run["status"] = "mcp_error"
                run["error"] = (
                    f"MCP analysis failed in round {round_number}."
                )
                return False

            return True

        ranking_args = {
            "when": request["when"],
            "group_by": ["store_nbr", "family"],
            "families": [request["family"]],
            "top_n": 1,
            "order": "highest",
        }

        if not await call_analysis(ranking_args, 1):
            return run

        # The leading store must come from independently
        # validated ranking evidence, not an LLM guess.
        try:
            ranking_review = self.analysis_validator.validate(
                run,
                when=request["when"],
                family=request["family"],
                top_n=1,
                require_busiest=False,
            )
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            run["status"] = "validation_error"
            run["error"] = str(exc)
            return run

        run["ranking_review"] = ranking_review

        if not ranking_review.get("evidence_valid"):
            # The application will reject incomplete evidence.
            # Crucially, the second MCP call never happens.
            return run

        leading_store = ranking_review.get("leading_store")

        if (
            not isinstance(leading_store, Integral)
            or isinstance(leading_store, bool)
            or leading_store <= 0
        ):
            run["status"] = "validation_error"
            run["error"] = "Invalid verified leading store."
            return run

        daily_args = {
            "when": request["when"],
            "group_by": ["target_date"],
            "stores": [int(leading_store)],
            "families": [request["family"]],
            "top_n": 1,
            "order": "highest",
        }

        await call_analysis(daily_args, 2)

        return run

    async def run(self, question, mcp_client):
        request = self.analysis_interpreter.extract(question)

        # Preserve the existing Bedrock-based sales contract route.
        if request["status"] == "not_applicable":
            result = await self.sales_agent.run(
                question, mcp_client
            )

            return {
                **result,
                "route": "sales",
            }

        if request["status"] != "supported":
            return {
                "route": "analysis",
                "status": "blocked_request",
                "reason": request.get(
                    "reason", "Unsupported request."
                ),
                "answer": None,
                "request": request,
                "model_run": None,
                "review": None,
            }

        if request["intent"] == "dependent_peak_day":
            model_run = await self._run_dependent_analysis(
                request, mcp_client
            )
        else:
            model_run = await self.orchestrator.run(
                question, mcp_client
            )

        if model_run.get("status") != "completed":
            return {
                "route": "analysis",
                "status": "blocked_execution",
                "reason": "Agent execution did not complete.",
                "answer": None,
                "request": request,
                "model_run": model_run,
                "review": None,
            }

        try:
            review = self.analysis_validator.validate(
                model_run,
                when=request["when"],
                family=request["family"],
                top_n=request["top_n"],
                require_busiest=request["require_busiest"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            return {
                "route": "analysis",
                "status": "blocked_evidence",
                "reason": f"Evidence validation failed: {exc}",
                "answer": None,
                "request": request,
                "model_run": model_run,
                "review": None,
            }

        if (
            not review["evidence_valid"]
            or not review.get("verified_answer")
        ):
            return {
                "route": "analysis",
                "status": "blocked_evidence",
                "reason": (
                    "Missing, inconsistent or incorrectly "
                    "scoped analysis evidence."
                ),
                "answer": None,
                "request": request,
                "model_run": model_run,
                "review": review,
            }

        return {
            "route": "analysis",
            "status": "verified_analysis_evidence",
            "reason": None,
            "answer": review["verified_answer"],
            "request": request,
            "model_run": model_run,
            "review": review,
        }
