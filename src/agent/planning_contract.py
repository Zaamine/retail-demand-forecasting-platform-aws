"""Business request scope and evidence eligibility checks."""

from dataclasses import dataclass
from datetime import timedelta

from agent.planning_calendar import PlanningCalendar, PlanningPeriod


@dataclass(frozen=True)
class PlanningContract:
    """Requested planning scope, independent of LLM tool arguments."""

    period: PlanningPeriod
    stores: frozenset[int] | None = None
    families: frozenset[str] | None = None

    @classmethod
    def create(
        cls,
        calendar: PlanningCalendar,
        when: str,
        stores: list[int] | None = None,
        families: list[str] | None = None,
    ):
        if stores is not None:
            if not stores or any(
                type(store) is not int or store <= 0
                for store in stores
            ):
                raise ValueError("Invalid store selection.")
            stores = frozenset(stores)

        if families is not None:
            if not families or any(
                not isinstance(f, str) or not f.strip()
                for f in families
            ):
                raise ValueError("Invalid product family selection.")
            families = frozenset(
                family.strip().upper()
                for family in families
            )

        return cls(
            period=calendar.resolve(when),
            stores=stores,
            families=families,
        )

    def inspect_evidence_scope(
        self,
        period: PlanningPeriod,
        stores: list[int] | None,
        families: list[str] | None,
    ) -> dict:
        """Check whether a tool result can support this request.

        Partial periods and store subsets are allowed.
        Complete coverage is checked separately.
        """
        issues = []

        if (
            period.start < self.period.start
            or period.end > self.period.end
        ):
            issues.append("Period outside requested scope.")

        if self.stores is not None:
            if stores is None:
                issues.append("Store filter missing.")
            elif not set(stores).issubset(self.stores):
                issues.append("Store filter exceeds requested scope.")

        if self.families is not None:
            if families is None:
                issues.append("Product family filter missing.")
            elif not {
                f.strip().upper() for f in families
            }.issubset(self.families):
                issues.append("Product family outside requested scope.")

        return {
            "eligible": not issues,
            "issues": issues,
        }

    def required_keys(self) -> set[tuple[int, str, str]]:
        """Enumerate required store-family-date observations.

        Available only when stores and families are explicitly scoped.
        """
        if self.stores is None or self.families is None:
            raise ValueError(
                "Explicit stores and families are required "
                "to enumerate evidence keys."
            )

        return {
            (store, family, current.isoformat())
            for store in self.stores
            for family in self.families
            for offset in range(self.period.days)
            for current in [
                self.period.start + timedelta(days=offset)
            ]
        }
