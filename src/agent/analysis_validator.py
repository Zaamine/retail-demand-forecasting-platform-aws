"""Validate store rankings and dependent daily analyses.

The validator compares MCP evidence with independent ForecastService
calculations. It does not certify arbitrary generated prose.
"""

import math
import re


def final_response_text(text):
    """Exclude diagnostic model text from answer checks."""
    text = re.sub(
        r"<thinking>.*?</thinking>",
        "",
        text or "",
        flags=re.S | re.I,
    )
    return re.sub(r"</?response>", "", text, flags=re.I).strip()


class AnalysisEvidenceValidator:
    def __init__(self, forecast_service):
        self.service = forecast_service

    @staticmethod
    def _same_number(a, b):
        try:
            return math.isclose(
                float(a), float(b), rel_tol=0, abs_tol=0.05
            )
        except (TypeError, ValueError, OverflowError):
            return False

    def _tool_matches(
        self, event, *, when, family, store=None, daily=False
    ):
        if (
            event.get("name") != "analyze_sales_forecasts"
            or event.get("status") != "success"
        ):
            return False

        args = event.get("arguments") or {}
        result = event.get("result") or {}

        if result.get("status") != "available":
            return False

        try:
            expected_period = self.service.calendar.resolve(when)
            tool_period = self.service.calendar.resolve(args["when"])
        except (KeyError, TypeError, ValueError):
            return False

        grouping = args.get("group_by") or []
        expected_group = "target_date" if daily else "store_nbr"

        if expected_group not in grouping:
            return False

        if daily != ("target_date" in grouping):
            return False

        if not set(grouping).issubset({
            "store_nbr", "family", "target_date"
        }):
            return False

        if args.get("order", "highest") != "highest":
            return False

        selected_families = {
            str(f).strip().upper()
            for f in (args.get("families") or [])
        }

        if selected_families != {family}:
            return False

        if store is None:
            if args.get("stores") is not None:
                return False
        elif args.get("stores") != [store]:
            return False

        return (
            expected_period == tool_period
            and result.get("period_start")
                == expected_period.start.isoformat()
            and result.get("period_end")
                == expected_period.end.isoformat()
            and result.get("source") == self.service.source
        )

    def validate(
        self, run, *, when, family, top_n=3,
        require_busiest=False,
    ):
        """Validate tool evidence and construct factual output."""

        if type(top_n) is not int or not 1 <= top_n <= 20:
            raise ValueError("top_n must be between 1 and 20.")

        family = family.strip().upper()

        reference = self.service.analyze(
            when=when,
            group_by=["store_nbr", "family"],
            families=[family],
            top_n=top_n,
            order="highest",
        )

        expected = reference["results"]
        ranking_event = None

        for event in run.get("tool_events", []):
            if not self._tool_matches(
                event, when=when, family=family, daily=False
            ):
                continue

            result = event["result"]
            rows = result.get("results") or []

            if (
                result.get("rows_analyzed")
                    != reference["rows_analyzed"]
                or result.get("groups_found")
                    != reference["groups_found"]
                or len(rows) < len(expected)
            ):
                continue

            matches = all(
                row.get("store_nbr") == ref["store_nbr"]
                and str(row.get("family", family)).upper()
                    == family
                and self._same_number(
                    row.get("predicted_sales"),
                    ref["predicted_sales"],
                )
                for row, ref in zip(rows[:top_n], expected)
            )

            if matches:
                ranking_event = event
                break

        issues = []

        if ranking_event is None:
            issues.append(
                "No correctly scoped, numerically verified "
                "store ranking."
            )

        leading_store = (
            expected[0]["store_nbr"] if expected else None
        )

        daily_reference = None
        daily_event = None

        if require_busiest and ranking_event is not None:
            daily_reference = self.service.analyze(
                when=when,
                group_by=["target_date"],
                stores=[leading_store],
                families=[family],
                top_n=1,
                order="highest",
            )

            for event in run.get("tool_events", []):
                if (
                    event.get("round", 0)
                    <= ranking_event.get("round", 0)
                ):
                    continue

                if not self._tool_matches(
                    event,
                    when=when,
                    family=family,
                    store=leading_store,
                    daily=True,
                ):
                    continue

                result = event["result"]
                rows = result.get("results") or []
                peak = daily_reference["results"][0]

                if (
                    result.get("rows_analyzed")
                        == daily_reference["rows_analyzed"]
                    and result.get("groups_found")
                        == daily_reference["groups_found"]
                    and rows
                    and rows[0].get("target_date")
                        == peak["target_date"]
                    and self._same_number(
                        rows[0].get("predicted_sales"),
                        peak["predicted_sales"],
                    )
                ):
                    daily_event = event
                    break

            if daily_event is None:
                issues.append(
                    "Missing later store-specific daily analysis "
                    "for the leading store."
                )

        evidence_valid = (
            ranking_event is not None
            and (not require_busiest or daily_event is not None)
        )

        # Diagnostic check only: this does NOT certify all model prose.
        answer = final_response_text(run.get("answer"))
        reported_stores = [
            int(value)
            for value in re.findall(
                r"\bstore\s*(?:number\s*|#\s*)?(\d+)\b",
                answer,
                flags=re.I,
            )
        ]
        expected_stores = [
            int(row["store_nbr"]) for row in expected
        ]

        store_sequence_matches = (
            reported_stores == expected_stores
            if not require_busiest else None
        )

        if (
            not require_busiest
            and evidence_valid
            and not store_sequence_matches
        ):
            issues.append(
                "Final model answer does not match "
                "the validated store ranking."
            )

        verified_answer = None

        if evidence_valid:
            if require_busiest:
                peak = daily_reference["results"][0]

                verified_answer = (
                    f"Store {leading_store} has the highest "
                    f"expected {family} sales for {when}: "
                    f"{expected[0]['predicted_sales']:,.2f} "
                    "sales units. "
                    f"Its busiest day is {peak['target_date']}, "
                    f"with {peak['predicted_sales']:,.2f} "
                    "expected sales units. "
                    "These sales estimates alone do not "
                    "determine delivery quantities."
                )
            else:
                lines = [
                    f"{rank}. Store {row['store_nbr']}: "
                    f"{row['predicted_sales']:,.2f} sales units"
                    for rank, row in enumerate(expected, 1)
                ]

                verified_answer = (
                    f"The leading stores by expected {family} "
                    "sales are:\n" + "\n".join(lines)
                )

        return {
            "evidence_valid": evidence_valid,
            "reported_store_sequence_matches":
                store_sequence_matches,
            "issues": issues,
            "verified_answer": verified_answer,
            "leading_store": (
                leading_store if evidence_valid else None
            ),
            "busiest_day": (
                daily_reference["results"][0]["target_date"]
                if evidence_valid and require_busiest
                else None
            ),
        }
