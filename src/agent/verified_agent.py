"""Fail-closed business-answer gate for explicitly scoped sales totals."""

from agent.evidence_validator import format_verified_sales


class VerifiedSalesAgent:
    """Evidence-backed route for retail sales-total questions."""

    def __init__(self, interpreter, orchestrator, validator):
        self.interpreter = interpreter
        self.orchestrator = orchestrator
        self.validator = validator

    async def run(self, question, mcp_client):
        extraction = self.interpreter.extract(question)

        if extraction["status"] != "supported":
            return {
                "status": "blocked_request",
                "reason": extraction.get(
                    "reason", "Unsupported request."
                ),
                "answer": None,
                "extraction": extraction,
                "model_run": None,
                "review": None,
            }

        model_run = await self.orchestrator.run(
            question,
            mcp_client,
        )

        if model_run.get("status") != "completed":
            return {
                "status": "blocked_execution",
                "reason": "Agent execution did not complete.",
                "answer": None,
                "extraction": extraction,
                "model_run": model_run,
                "review": None,
            }

        try:
            review = self.validator.validate(
                model_run,
                extraction["contract"],
                require_readiness=extraction[
                    "requires_readiness"
                ],
            )

        except (TypeError, ValueError, KeyError) as exc:
            return {
                "status": "blocked_evidence",
                "reason": (
                    f"Evidence validation failed: {exc}"
                ),
                "answer": None,
                "extraction": extraction,
                "model_run": model_run,
                "review": None,
            }

        if not review["evidence_complete"]:
            return {
                "status": "blocked_evidence",
                "reason": (
                    "Missing, incorrect or unverified "
                    "sales evidence."
                ),
                "answer": None,
                "extraction": extraction,
                "model_run": model_run,
                "review": review,
            }

        return {
            "status": "verified_sales_evidence",
            "answer": format_verified_sales(review),
            "reason": None,
            "extraction": extraction,
            "model_run": model_run,
            "review": review,
        }
