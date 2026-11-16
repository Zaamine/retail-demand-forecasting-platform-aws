"""Deterministic planning calendar for retail sales estimates."""

from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class PlanningPeriod:
    start: date
    end: date

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1


@dataclass(frozen=True)
class PlanningCalendar:
    business_date: date
    horizon_days: int = 14

    def resolve(self, expression: str) -> PlanningPeriod:
        """Resolve a planning expression relative to the business date."""

        when = " ".join(expression.strip().lower().split())
        first_day = self.business_date + timedelta(days=1)
        last_day = self.business_date + timedelta(
            days=self.horizon_days
        )

        if when == "tomorrow":
            start = end = first_day

        elif when == "next week":
            # Next Monday through Sunday.
            start = self.business_date + timedelta(
                days=7 - self.business_date.weekday()
            )
            end = start + timedelta(days=6)

        elif when == "next 7 days":
            start = first_day
            end = start + timedelta(days=6)

        elif when in {
            "next 14 days",
            "next two weeks",
            "next 2 weeks",
            "full horizon",
        }:
            start = first_day
            end = last_day

        elif "/" in when:
            parts = when.split("/")
            if len(parts) != 2:
                raise ValueError("Invalid date range.")

            start, end = (
                date.fromisoformat(part.strip())
                for part in parts
            )

        else:
            start = end = date.fromisoformat(when)

        if start > end:
            raise ValueError("Period start is after period end.")

        if start < first_day or end > last_day:
            raise ValueError(
                f"Requested dates must be between "
                f"{first_day} and {last_day}."
            )

        return PlanningPeriod(start=start, end=end)
