"""Tests for deterministic Retail Planner date interpretation."""

from datetime import date

import pytest

from agent.planning_calendar import PlanningCalendar


@pytest.fixture
def calendar():
    return PlanningCalendar(date(2017, 8, 15))


@pytest.mark.parametrize(
    ("expression", "expected_start", "expected_end"),
    [
        ("tomorrow", "2017-08-16", "2017-08-16"),
        ("next week", "2017-08-21", "2017-08-27"),
        ("next 7 days", "2017-08-16", "2017-08-22"),
        ("next two weeks", "2017-08-16", "2017-08-29"),
        ("next 14 days", "2017-08-16", "2017-08-29"),
        ("full horizon", "2017-08-16", "2017-08-29"),
        ("2017-08-16/2017-08-29", "2017-08-16", "2017-08-29"),
    ],
)
def test_planning_periods(
    calendar, expression, expected_start, expected_end
):
    result = calendar.resolve(expression)

    assert result.start.isoformat() == expected_start
    assert result.end.isoformat() == expected_end


@pytest.mark.parametrize(
    "expression",
    [
        "2017-08-30",
        "2017-08-15",
        "2017-08-20/2017-08-19",
        "2017-08-16-2017-08-29",
    ],
)
def test_invalid_periods_are_rejected(calendar, expression):
    with pytest.raises(ValueError):
        calendar.resolve(expression)


def test_calendar_is_configurable():
    calendar = PlanningCalendar(date(2017, 8, 20))

    result = calendar.resolve("next week")

    assert result.start == date(2017, 8, 21)
    assert result.end == date(2017, 8, 27)
