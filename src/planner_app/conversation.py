"""Manage simple dialogue and verified Retail Planner context."""

import re

from agent.planning_contract import PlanningContract


GREETINGS = {
    "hi", "hello", "hey",
    "good morning", "good afternoon", "good evening",
}

HELP_REQUESTS = {
    "who are you", "what is your purpose",
    "what's your purpose", "what can you do",
    "how can you help me", "help",
}

PERIODS = {
    "tomorrow": "tomorrow",
    "next week": "next week",
    "next 7 days": "next 7 days",
    "next two weeks": "next two weeks",
    "next 14 days": "next two weeks",
}


def context_from_verified_result(result):
    """Extract business context only from verified agent results."""
    if result.get("status") == "verified_sales_evidence":
        extraction = result.get("extraction") or {}
        review = result.get("review") or {}
        contract = extraction.get("contract")

        if (
            extraction.get("status") != "supported"
            or review.get("evidence_complete") is not True
            or not isinstance(contract, PlanningContract)
        ):
            return None

        if (
            contract.stores is None
            or contract.families is None
            or len(contract.stores) != 1
            or len(contract.families) != 1
        ):
            return None

        store = next(iter(contract.stores))
        family = next(iter(contract.families))
        start = contract.period.start.isoformat()
        end = contract.period.end.isoformat()
        totals = review.get("verified_totals") or []

        if (
            review.get("period_start") != start
            or review.get("period_end") != end
            or len(totals) != 1
            or totals[0].get("store_nbr") != store
            or totals[0].get("family") != family
        ):
            return None

        return {
            "intent": "sales_total",
            "store_nbr": store,
            "family": family,
            "when": f"{start}/{end}",
        }

    if result.get("status") != "verified_analysis_evidence":
        return None

    request = result.get("request") or {}
    review = result.get("review") or {}

    if not review.get("evidence_valid"):
        return None

    if request.get("status") != "supported":
        return None

    if request.get("intent") not in {
        "dependent_peak_day", "store_ranking"
    }:
        return None

    if not isinstance(request.get("family"), str):
        return None

    if not isinstance(request.get("when"), str):
        return None

    top_n = request.get("top_n")
    if type(top_n) is not int or not 1 <= top_n <= 5:
        return None

    return {
        "intent": request["intent"],
        "family": request["family"],
        "when": request["when"],
        "top_n": top_n,
    }


def plan_turn(question, context=None, known_families=()):
    """Resolve a message into a direct reply or an agent request."""
    message = " ".join(question.strip().split())
    normalized = message.casefold().rstrip(" .!?")

    if normalized in GREETINGS:
        return {
            "mode": "reply",
            "answer": (
                "Hello! I'm your Retail Planner assistant. "
                "How can I help you today?"
            ),
        }

    if normalized in HELP_REQUESTS:
        return {
            "mode": "reply",
            "answer": (
                "I help retail planners explore expected sales "
                "across stores and product families. "
                "In this preview, I can identify leading stores "
                "and their busiest forecast days. "
                "I cannot determine inventory orders from "
                "sales forecasts alone."
            ),
        }

    match = re.fullmatch(
        r"(?:and\s+)?(?:what|how)\s+about\s+(.+)",
        normalized,
    )

    if match is None:
        return {
            "mode": "agent",
            "question": message,
            "used_context": False,
        }

    if context is None:
        return {
            "mode": "reply",
            "answer": (
                "Could you first ask a complete forecast question?"
            ),
        }

    subject = match.group(1).strip()
    family = context["family"]
    when = context["when"]

    if subject in PERIODS:
        when = PERIODS[subject]
    elif subject.upper() in {
        str(name).upper() for name in known_families
    }:
        family = subject.upper()
    else:
        return {
            "mode": "reply",
            "answer": (
                "Could you specify a known product family "
                "or a supported forecast period?"
            ),
        }

    if context["intent"] == "sales_total":
        if "/" in when:
            start, end = when.split("/", 1)
            period_phrase = f"from {start} to {end}"
        else:
            period_phrase = when

        expanded = (
            f"How much {family.lower()} should store "
            f"{context['store_nbr']} expect to sell "
            f"{period_phrase}?"
        )

    elif context["intent"] == "dependent_peak_day":
        expanded = (
            f"Which {family.lower()} store should expect "
            f"the most sales {when}, "
            "and which day will be its busiest?"
        )
    else:
        top_n = context["top_n"]
        if top_n == 1:
            expanded = (
                f"Which store should expect the most "
                f"{family.lower()} sales {when}?"
            )
        else:
            expanded = (
                f"Which top {top_n} stores should expect "
                f"the most {family.lower()} sales {when}?"
            )

    return {
        "mode": "agent",
        "question": expanded,
        "used_context": True,
    }
