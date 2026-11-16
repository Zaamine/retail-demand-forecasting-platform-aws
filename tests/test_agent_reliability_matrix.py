"""Twenty offline evidence-regression scenarios.

These cases evaluate validators, not empirical LLM accuracy.
No AWS or Bedrock calls are made.
"""

from copy import deepcopy
from datetime import date, timedelta

import pandas as pd
import pytest

from agent.analysis_validator import AnalysisEvidenceValidator
from agent.evidence_validator import SalesEvidenceValidator
from agent.forecast_service import ForecastService
from agent.planning_calendar import PlanningCalendar
from agent.planning_contract import PlanningContract


def fixture_service():
    """Complete synthetic snapshot for three stores and fourteen days."""

    calendar = PlanningCalendar(date(2017, 8, 15))

    rows = [
        {
            "store_nbr": store,
            "family": "DAIRY",
            "forecast_origin": "2017-08-15",
            "target_date": date(2017, 8, 15) + timedelta(days=day),
            "prediction": float(store * 10 + day),
        }
        for store in (12, 44, 49)
        for day in range(1, 15)
    ]

    return ForecastService(
        pd.DataFrame(rows),
        calendar,
        source="fixture://published-sales",
    )


def lookup_event(service, store=44, when="next week"):
    data = service.sales_for_store(store, "DAIRY", when)

    return {
        "round": 1,
        "name": "lookup_sales_forecast",
        "status": "success",
        "arguments": {
            "store_nbr": store,
            "family": "DAIRY",
            "when": when,
        },
        "result": {
            "status": "available",
            "result_type": "period",
            "source": service.source,
            "data": data,
        },
    }


def readiness_event(service):
    return {
        "round": 1,
        "name": "check_planning_data_status",
        "status": "success",
        "arguments": {},
        "result": {
            "status": "ready_for_historical_planning",
            "published_estimates": {
                "verified": True,
                "source": service.source,
                "available_from": "2017-08-16",
                "available_through": "2017-08-29",
            },
        },
    }


def ranking_event(service):
    args = {
        "when": "next week",
        "group_by": ["store_nbr", "family"],
        "families": ["DAIRY"],
        "top_n": 3,
        "order": "highest",
    }

    return {
        "round": 1,
        "name": "analyze_sales_forecasts",
        "status": "success",
        "arguments": args,
        "result": service.analyze(**args),
    }


def daily_event(service, store=49, round_no=2):
    args = {
        "when": "next week",
        "group_by": ["target_date"],
        "stores": [store],
        "families": ["DAIRY"],
        "top_n": 1,
        "order": "highest",
    }

    return {
        "round": round_no,
        "name": "analyze_sales_forecasts",
        "status": "success",
        "arguments": args,
        "result": service.analyze(**args),
    }


# Questions are fixed annotations.
# The test traces below are synthetic or deliberately corrupted.
CASES = [
    (
        "01_sales_ready",
        "Are next week's dairy sales for store 44 ready, and how much will it sell?",
        "sales", "ready", True,
    ),
    (
        "02_sales_no_readiness",
        "How much dairy should store 44 expect to sell next week?",
        "sales", "clean", True,
    ),
    (
        "03_sales_two_stores",
        "How much dairy should stores 12 and 44 expect to sell over the next two weeks?",
        "sales", "two_stores", True,
    ),
    (
        "04_sales_split",
        "How many dairy units will store 44 sell over the next two weeks?",
        "sales", "split", True,
    ),
    (
        "05_sales_readiness_missing",
        "Are dairy sales for store 44 ready for next week?",
        "sales", "missing_ready", False,
    ),
    (
        "06_sales_wrong_day_value",
        "What are store 44's expected dairy sales next week?",
        "sales", "altered_day", False,
    ),
    (
        "07_sales_wrong_total",
        "What dairy sales total should store 44 expect next week?",
        "sales", "altered_total", False,
    ),
    (
        "08_sales_wrong_store",
        "How much dairy will store 44 sell next week?",
        "sales", "wrong_store", False,
    ),
    (
        "09_sales_wrong_family",
        "How much dairy should store 44 sell during next week?",
        "sales", "wrong_family", False,
    ),
    (
        "10_sales_missing_day",
        "Give me store 44's expected dairy sales for the coming week.",
        "sales", "missing_day", False,
    ),
    (
        "11_sales_wrong_source",
        "How many dairy sales units can store 44 expect next week?",
        "sales", "wrong_source", False,
    ),
    (
        "12_sales_overlap",
        "What are the two-week dairy sales estimates for store 44?",
        "sales", "overlap_gap", False,
    ),
    (
        "13_ranking_correct",
        "Which three stores should expect the most dairy sales next week?",
        "ranking", "correct", True,
    ),
    (
        "14_ranking_wrong_final",
        "Name the top three stores for dairy sales next week.",
        "ranking", "wrong_answer", True,
    ),
    (
        "15_ranking_wrong_tool",
        "Which stores lead dairy sales next week?",
        "ranking", "wrong_order", False,
    ),
    (
        "16_ranking_unscoped",
        "Where will dairy sales be highest next week?",
        "ranking", "missing_family_filter", False,
    ),
    (
        "17_dependent_correct",
        "Which dairy store is busiest next week and on which day?",
        "dependent", "correct", True,
    ),
    (
        "18_dependent_same_round",
        "Find the busiest dairy store and its peak day next week.",
        "dependent", "same_round", False,
    ),
    (
        "19_dependent_network_day",
        "Which day is busiest for the leading dairy store next week?",
        "dependent", "network_daily", False,
    ),
    (
        "20_dependent_wrong_store",
        "When is next week's busiest day for our highest-selling dairy store?",
        "dependent", "wrong_store", False,
    ),
]


def evaluate_case(case):
    _, _, kind, variant, expected_valid = case

    service = fixture_service()

    if kind == "sales":
        stores = [12, 44] if variant == "two_stores" else [44]

        when = (
            "next two weeks"
            if variant in ("two_stores", "split", "overlap_gap")
            else "next week"
        )

        contract = PlanningContract.create(
            service.calendar,
            when,
            stores=stores,
            families=["DAIRY"],
        )

        if variant == "split":
            events = [
                lookup_event(service, when="next 7 days"),
                lookup_event(
                    service,
                    when="2017-08-23/2017-08-29",
                ),
            ]

        elif variant == "overlap_gap":
            events = [
                lookup_event(service, when="next 7 days"),
                lookup_event(service, when="next week"),
            ]

        else:
            events = [
                lookup_event(service, store=store, when=when)
                for store in stores
            ]

        if variant == "ready":
            events.insert(0, readiness_event(service))

        target = next(
            event for event in events
            if event["name"] == "lookup_sales_forecast"
        )

        payload = target["result"]
        data = payload["data"]

        if variant == "altered_day":
            data["daily_predictions"][0]["predicted_sales"] += 100

        elif variant == "altered_total":
            data["total_predicted_sales"] += 100

        elif variant == "wrong_store":
            data["store_nbr"] = 12

        elif variant == "wrong_family":
            data["family"] = "BEVERAGES"

        elif variant == "missing_day":
            data["daily_predictions"].pop()

        elif variant == "wrong_source":
            payload["source"] = "fixture://other-snapshot"

        review = SalesEvidenceValidator(service).validate(
            {"status": "completed", "tool_events": events},
            contract,
            require_readiness=variant in (
                "ready", "missing_ready"
            ),
        )

        actual = review["evidence_complete"]

    else:
        ranking = ranking_event(service)
        events = [ranking]
        answer = "Store 49, Store 44, Store 12"

        if kind == "ranking":
            if variant == "wrong_answer":
                answer = "Store 49, Store 44, Store 50"

            elif variant == "wrong_order":
                rows = ranking["result"]["results"]
                rows[0], rows[1] = rows[1], rows[0]

            elif variant == "missing_family_filter":
                ranking["arguments"]["families"] = None

            review = AnalysisEvidenceValidator(service).validate(
                {
                    "status": "completed",
                    "tool_events": events,
                    "answer": answer,
                },
                when="next week",
                family="DAIRY",
                top_n=3,
            )

            actual = review["evidence_valid"]

            if variant == "wrong_answer":
                assert (
                    review["reported_store_sequence_matches"]
                    is False
                )
                assert review["verified_answer"] is not None

        else:
            if variant == "same_round":
                detail = daily_event(service, round_no=1)

            elif variant == "network_daily":
                detail = daily_event(service, round_no=2)
                detail["arguments"]["stores"] = None
                detail["result"] = service.analyze(
                    when="next week",
                    group_by=["target_date"],
                    families=["DAIRY"],
                    top_n=1,
                    order="highest",
                )

            elif variant == "wrong_store":
                detail = daily_event(service, store=44)

            else:
                detail = daily_event(service, store=49)

            events.append(detail)

            review = AnalysisEvidenceValidator(service).validate(
                {
                    "status": "completed",
                    "tool_events": events,
                    "answer": "",
                },
                when="next week",
                family="DAIRY",
                top_n=1,
                require_busiest=True,
            )

            actual = review["evidence_valid"]

    return actual, expected_valid


@pytest.mark.parametrize(
    "case",
    CASES,
    ids=[item[0] for item in CASES],
)
def test_fixed_evidence_scenario(case):
    actual, expected = evaluate_case(deepcopy(case))

    assert actual is expected, (
        f"Unexpected verdict for {case[0]}"
    )


def test_benchmark_has_unique_questions_and_fixed_size():
    assert len(CASES) == 20
    assert len({item[0] for item in CASES}) == len(CASES)
    assert len({item[1] for item in CASES}) == len(CASES)
    assert {
        item[2] for item in CASES
    } == {"sales", "ranking", "dependent"}
