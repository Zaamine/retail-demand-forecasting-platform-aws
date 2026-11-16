"""Conservative request interpretation for ranking and dependent analysis."""

import re


class AnalysisRequestInterpreter:
    """Recognize explicitly scoped retail analysis requests."""

    PERIODS = (
        (r"\bnext (?:two|2) weeks\b", "next two weeks"),
        (r"\bnext 14 days\b", "next two weeks"),
        (r"\bnext (?:seven|7) days\b", "next 7 days"),
        (r"\bnext week(?:'s)?\b", "next week"),
        (r"\btomorrow\b", "tomorrow"),
    )

    NUMBERS = {
        "one": 1, "two": 2, "three": 3,
        "four": 4, "five": 5,
        "1": 1, "2": 2, "3": 3,
        "4": 4, "5": 5,
    }

    def __init__(self, calendar, known_families):
        self.calendar = calendar
        self.known_families = sorted(
            {str(family).strip().upper() for family in known_families}
        )

    def extract(self, question):
        if not isinstance(question, str) or not question.strip():
            return {
                "status": "unsupported",
                "reason": "Empty question.",
            }

        # These words signal a ranking-related question.
        ranking = re.search(
            r"\b(?:top|most|highest|leading|busiest|"
            r"best-selling|highest-selling)\b",
            question,
            flags=re.I,
        )

        if not ranking:
            return {"status": "not_applicable"}

        def reject(reason):
            return {"status": "unsupported", "reason": reason}

        if not re.search(r"\bstores?\b", question, re.I):
            return reject("A store-ranking question is required.")

        # Named-store comparisons are not general network rankings.
        if re.search(
            r"\bstores?\s*(?:number\s*|#\s*)?\d+\b",
            question, re.I,
        ):
            return reject(
                "Named-store comparisons are not supported here."
            )

        if re.search(
            r"\b(?:revenue|profit|price|discount|promotion|"
            r"inventory|stock|order|orders|deliver|delivery|"
            r"deliveries|replenishment)\b|[$€]",
            question, re.I,
        ):
            return reject(
                "This request involves another business operation."
            )

        families = []

        for family in self.known_families:
            pattern = (
                r"(?<!\w)"
                + re.escape(family).replace(r"\ ", r"\s+")
                + r"(?!\w)"
            )

            if re.search(pattern, question, re.I):
                families.append(family)

        if len(families) != 1:
            return reject(
                "Exactly one product family must be explicit."
            )

        periods = [
            expression
            for pattern, expression in self.PERIODS
            if re.search(pattern, question, re.I)
        ]

        if (
            len(periods) != 1
            or re.search(r"\b\d{4}-\d{2}-\d{2}\b", question)
        ):
            return reject(
                "Exactly one supported planning period is required."
            )

        when = periods[0]

        try:
            period = self.calendar.resolve(when)
        except (ValueError, TypeError):
            return reject(
                "Planning period is outside published forecasts."
            )

        daily = bool(
            re.search(
                r"\b(?:day|date|daily|when)\b",
                question, re.I,
            )
        )

        if daily:
            if not re.search(
                r"\b(?:leading|top|busiest|highest-selling|most)\b",
                question, re.I,
            ):
                return reject(
                    "The leading-store dependency is not explicit."
                )

            intent = "dependent_peak_day"
            top_n = 1

        else:
            count_match = re.search(
                r"\btop\s+(one|two|three|four|five|[1-5])\b"
                r"|\b(?:which|what|name|list)\s+"
                r"(one|two|three|four|five|[1-5])\s+stores\b",
                question, re.I,
            )

            if count_match:
                token = (
                    count_match.group(1) or count_match.group(2)
                ).lower()

                top_n = self.NUMBERS[token]

            elif re.search(
                r"\bwhich\s+(?:\w+\s+)?store\b",
                question, re.I,
            ):
                top_n = 1

            else:
                return reject(
                    "An explicit ranking size is required."
                )

            intent = "store_ranking"

        return {
            "status": "supported",
            "intent": intent,
            "family": families[0],
            "when": when,
            "period_start": period.start.isoformat(),
            "period_end": period.end.isoformat(),
            "top_n": top_n,
            "require_busiest": daily,
        }
