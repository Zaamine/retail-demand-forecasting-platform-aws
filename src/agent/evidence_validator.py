"""Independent verification of sales evidence returned by MCP tools.

This module does not approve unrestricted LLM-generated prose.
"""

import math
from datetime import date, timedelta


class SalesEvidenceValidator:
    """Compare observed MCP lookup evidence with the trusted forecast service."""

    def __init__(self, forecast_service, tolerance=1e-6):
        self.forecast_service = forecast_service
        self.tolerance = tolerance

    def validate(self, run, contract, *, require_readiness=False):
        required = contract.required_keys()
        expected = {}

        # Build the numerical reference independently of the LLM.
        for store in contract.stores:
            for family in contract.families:
                period = (
                    f"{contract.period.start.isoformat()}/"
                    f"{contract.period.end.isoformat()}"
                )

                reference = self.forecast_service.sales_for_store(
                    store, family, period
                )

                for row in reference["daily_predictions"]:
                    expected[(store, family, row["date"])] = float(
                        row["predicted_sales"]
                    )

        if set(expected) != required:
            raise ValueError(
                "Trusted snapshot does not cover the contract."
            )

        accepted = {}
        issues = []
        readiness = False

        for event in run.get("tool_events", []):
            name = event.get("name")
            payload = event.get("result") or {}

            if name == "check_planning_data_status":
                published = payload.get("published_estimates") or {}

                readiness = readiness or (
                    event.get("status") == "success"
                    and payload.get("status")
                        == "ready_for_historical_planning"
                    and published.get("verified") is True
                    and published.get("source")
                        == self.forecast_service.source
                    and published.get(
                        "available_from", "9999-12-31"
                    ) <= contract.period.start.isoformat()
                    and published.get(
                        "available_through", "0000-01-01"
                    ) >= contract.period.end.isoformat()
                )
                continue

            if name != "lookup_sales_forecast":
                continue

            if (
                event.get("status") != "success"
                or payload.get("status") != "available"
            ):
                issues.append(
                    "Unsuccessful lookup was not counted."
                )
                continue

            data = payload.get("data") or {}

            if payload.get("source") != self.forecast_service.source:
                issues.append(
                    "Lookup source does not match trusted snapshot."
                )
                continue

            try:
                store = data["store_nbr"]
                family = data["family"]

                if (
                    type(store) is not int
                    or not isinstance(family, str)
                ):
                    raise ValueError("Invalid store or family.")

                family = family.strip().upper()
                result_type = payload.get("result_type")

                if result_type == "single_day":
                    rows = [{
                        "date": data["date"],
                        "predicted_sales": data["predicted_sales"],
                    }]

                elif result_type == "period":
                    rows = data["daily_predictions"]

                    dates = [
                        date.fromisoformat(row["date"])
                        for row in rows
                    ]

                    start = date.fromisoformat(
                        data["period_start"]
                    )
                    end = date.fromisoformat(
                        data["period_end"]
                    )

                    complete_dates = {
                        start + timedelta(days=offset)
                        for offset in range(
                            (end - start).days + 1
                        )
                    }

                    if (
                        start > end
                        or len(dates) != len(set(dates))
                        or set(dates) != complete_dates
                    ):
                        raise ValueError(
                            "Incomplete or duplicate period rows."
                        )

                    total = float(
                        data["total_predicted_sales"]
                    )

                    if not math.isclose(
                        sum(
                            float(row["predicted_sales"])
                            for row in rows
                        ),
                        total,
                        rel_tol=0,
                        abs_tol=self.tolerance,
                    ):
                        raise ValueError(
                            "Daily values disagree with period total."
                        )

                else:
                    raise ValueError(
                        "Unsupported lookup result type."
                    )

                candidate = {}

                for row in rows:
                    day = date.fromisoformat(
                        row["date"]
                    ).isoformat()

                    key = (store, family, day)
                    value = float(row["predicted_sales"])

                    if not math.isfinite(value) or key not in required:
                        raise ValueError(
                            "Evidence outside requested scope."
                        )

                    if not math.isclose(
                        value,
                        expected[key],
                        rel_tol=0,
                        abs_tol=self.tolerance,
                    ):
                        raise ValueError(
                            "Value differs from trusted snapshot."
                        )

                    candidate[key] = value

                accepted.update(candidate)

            except (
                KeyError, TypeError, ValueError, OverflowError
            ) as exc:
                issues.append(
                    f"Lookup evidence rejected: {exc}"
                )

        missing = sorted(required - set(accepted))

        complete = (
            not missing
            and (readiness or not require_readiness)
        )

        totals = []

        if complete:
            required_days = sorted({
                key[2] for key in required
            })

            for store in sorted(contract.stores):
                for family in sorted(contract.families):
                    total = sum(
                        expected[(store, family, day)]
                        for day in required_days
                    )

                    totals.append({
                        "store_nbr": store,
                        "family": family,
                        "predicted_sales": total,
                    })

        return {
            "evidence_complete": complete,
            "covered": len(accepted),
            "required": len(required),
            "readiness_verified": readiness,
            "missing": missing,
            "issues": issues,
            "period_start": contract.period.start.isoformat(),
            "period_end": contract.period.end.isoformat(),
            "verified_totals": totals,
        }


def format_verified_sales(review):
    """Build factual output from verified evidence, not LLM prose."""

    if not review["evidence_complete"]:
        raise ValueError(
            "Cannot report sales without complete evidence."
        )

    heading = (
        "Expected sales from "
        f"{review['period_start']} to "
        f"{review['period_end']}:"
    )

    lines = [
        f"- Store {row['store_nbr']}, {row['family']}: "
        f"{row['predicted_sales']:,.2f} sales units"
        for row in review["verified_totals"]
    ]

    if review["readiness_verified"]:
        heading = (
            "Published sales estimates are ready for planning.\n"
            + heading
        )

    return heading + "\n" + "\n".join(lines)
