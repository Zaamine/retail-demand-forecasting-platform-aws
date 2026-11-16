"""MCP tools for the Retail Planner application.

Dependencies are injected. The server does not load forecasts,
create AWS clients, or initialize the knowledge index itself.
"""

from collections.abc import Callable

from mcp.server import MCPServer


def build_retail_planner_mcp_server(
    forecast_service,
    knowledge_retriever,
    status_reader: Callable[[], dict],
) -> MCPServer:
    """Build the four-tool Retail Planner MCP server."""

    if not callable(status_reader):
        raise TypeError("status_reader must be callable.")

    server = MCPServer("RetailPlanner")

    @server.tool()
    def lookup_sales_forecast(
        store_nbr: int,
        family: str,
        when: str,
    ) -> dict:
        """Get expected sales for a store and product family.

        Use for a specific store's daily or period sales estimates.
        Supported periods include tomorrow, next week,
        next 7 days, next two weeks, next 14 days, full horizon,
        and dates or date ranges.
        """
        try:
            result = forecast_service.sales_for_store(
                store_nbr, family, when
            )
        except (ValueError, TypeError) as exc:
            return {
                "status": "invalid_request",
                "message": str(exc),
            }

        if len(result["daily_predictions"]) == 1:
            day = result["daily_predictions"][0]
            return {
                "status": "available",
                "result_type": "single_day",
                "data": {
                    "store_nbr": result["store_nbr"],
                    "family": result["family"],
                    "date": day["date"],
                    "predicted_sales": day["predicted_sales"],
                },
                "source": result["source"],
            }

        return {
            "status": "available",
            "result_type": "period",
            "data": result,
            "source": result["source"],
        }

    @server.tool()
    def analyze_sales_forecasts(
        when: str,
        group_by: list[str],
        stores: list[int] | None = None,
        families: list[str] | None = None,
        top_n: int = 20,
        order: str = "highest",
    ) -> dict:
        """Compare and analyze expected sales by store, family or day.

        Filter by stores and families when a planner asks about
        particular locations or products.

        Use group_by=['store_nbr'] for store comparisons and
        group_by=['target_date'] for daily sales patterns.

        When identifying a store's busiest day, include that
        store in the stores filter.
        """
        try:
            return forecast_service.analyze(
                when=when,
                group_by=group_by,
                stores=stores,
                families=families,
                top_n=top_n,
                order=order,
            )
        except (ValueError, TypeError) as exc:
            return {
                "status": "invalid_request",
                "message": str(exc),
            }

    @server.tool()
    def consult_business_knowledge(question: str) -> dict:
        """Retrieve documented guidance for retail planning.

        Use for questions about promotions, model limitations,
        expected sales, inventory, replenishment and reliability.
        Return source-backed evidence rather than inventing policy.
        """
        context = knowledge_retriever.build_context(
            question=question,
            top_k=3,
        )

        evidence = [
            {
                key: item[key]
                for key in (
                    "passage_id",
                    "section",
                    "content",
                    "source_path",
                    "source_sha256",
                )
                if key in item
            }
            for item in context["evidence"]
        ]

        return {
            "status": (
                "available" if evidence else "no_relevant_evidence"
            ),
            "question": question,
            "evidence": evidence,
        }

    @server.tool()
    def check_planning_data_status() -> dict:
        """Check if published sales estimates are ready for planning.

        Use when a planner asks whether sales numbers are available,
        which dates they cover, or whether preparation completed.

        The injected reader must be read-only. This tool must
        never start, rerun or modify AWS workflows.
        """
        result = status_reader()

        if not isinstance(result, dict):
            raise TypeError(
                "The operational status reader must return a dictionary."
            )

        return result

    return server
