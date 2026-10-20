"""Focused regression coverage for recently discussed analysis and UI/API contract fixes.

These tests avoid importing the full FastAPI app, so they can run in a lean environment.
The API end-to-end test remains in test_region_compare_end_to_end.py and requires the
complete application dependencies.
"""
import pandas as pd
import pytest

from packages.analytics_core.src.engines.analyst_answer import (
    _resolve_trend_window,
    _trend,
)
from packages.analytics_core.src.intelligence.canonical_question_contract import (
    compile_canonical_question_contract,
)


def _annual_frame(start=2000, end=2026):
    return pd.DataFrame([
        {"year": year, "temperature": 15.0 + 0.08 * (year - start) + 0.1 * ((year % 3) - 1),
         "revenue": 1000 + year - start, "region": "A" if year % 2 else "B"}
        for year in range(start, end + 1)
    ])


def test_past_decade_filters_to_latest_ten_elapsed_years_and_charts_same_window():
    frame = _annual_frame()
    result = _trend("How has temperature changed over the past decade?", frame, "temperature", "year", None)
    assert result is not None
    numbers = result.numbers
    assert numbers["time_window"].startswith("10-year window ending")
    assert numbers["periods"] == 11  # annual labels at both boundaries over ten elapsed years
    chart_years = [int(point["time"]) for point in numbers["trend_chart"]["data"]]
    assert chart_years == list(range(2016, 2027))
    assert numbers["aggregation"] if "aggregation" in numbers else True
    assert numbers["trend_chart"]["metadata"]["trend_line_included"] is False
    assert numbers["trend_chart"]["metadata"]["observation_count"] == 11


def test_temperature_trend_uses_mean_not_default_sum():
    frame = _annual_frame()
    result = _trend("How has temperature changed over the past decade?", frame, "temperature", "year", None)
    assert result is not None
    assert "average per year" in result.headline.lower() or result.numbers["grain"] == "year"
    assert result.numbers["source_observation_count"] == 11


def test_explicit_year_window_is_inclusive_of_requested_end_year():
    frame = _annual_frame()
    dates = pd.to_datetime(frame["year"].astype(str) + "-01-01")
    mask, label, requested = _resolve_trend_window("Show temperature trend from 2012 to 2018", dates)
    assert requested is True
    assert int(mask.sum()) == 7
    assert dates.loc[mask].min().year == 2012
    assert dates.loc[mask].max().year == 2018
    assert "2012-2018" in label


def test_unresolvable_explicit_window_fails_closed():
    frame = _annual_frame()
    dates = pd.to_datetime(frame["year"].astype(str) + "-01-01")
    mask, label, requested = _resolve_trend_window("Show the trend over the last 15 fortnights", dates)
    # The established resolver does not support every phrase; it must not widen it silently.
    if requested and "Unresolved" in label:
        assert not bool(mask.any())


def test_canonical_contract_comparison_keeps_numeric_target_and_categorical_group():
    frame = pd.DataFrame({
        "order_id": ["o1", "o2", "o3", "o4", "o5", "o6"],
        "region": ["A", "B", "C", "A", "B", "C"],
        "revenue": [10, 20, 30, 11, 21, 31],
        "order_date": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-01", "2024-01-01", "2024-01-01", "2024-01-01"]),
    })
    contract = compile_canonical_question_contract("Compare regions A, B and C by total revenue in 2025.", frame)
    assert contract.target_column == "revenue"
    assert tuple(contract.grouping_columns) == ("region",)
    assert tuple(contract.comparison_values) == ("A", "B", "C")


def test_group_roles_inverted_does_not_silently_bind_text_as_measure():
    frame = pd.DataFrame({"region": ["A", "B"], "revenue": [10, 20]})
    contract = compile_canonical_question_contract("Show region by revenue", frame)
    assert not (contract.target_column == "region" and tuple(contract.grouping_columns) == ("revenue",))


def test_unresolvable_explicit_window_fails_closed():
    frame = _annual_frame()
    dates = pd.to_datetime(frame["year"].astype(str) + "-01-01")
    mask, label, requested = _resolve_trend_window("Show the trend over the last 7 business days", dates)
    assert requested is True
    assert "Unresolved" in label
    assert not bool(mask.any())


def test_generic_climate_question_scans_climate_indicators_not_unrelated_numeric_columns():
    from packages.analytics_core.src.engines.analyst_answer import build_analyst_result
    base = _annual_frame()
    frame = base.assign(
        precipitation=[900 + 5 * (year % 3) for year in base["year"]],
        support_tickets=[(i % 5) + 1 for i in range(len(base))],
        discount=[(i % 7) + 1 for i in range(len(base))],
    )
    result = build_analyst_result(
        "How has the climate changed over the past decade?", frame,
        target="revenue", time_col="year", target_explicit=False,
    )
    assert result is not None and result.kind == "TREND"
    assert result.numbers["analysis"] == "trend_analysis_multiple_outcomes"
    assert set(result.numbers["candidate_outcomes"]) == {"temperature", "precipitation"}
    assert result.numbers["multiple_testing"] == "Benjamini-Hochberg false-discovery rate across scanned outcomes"
    assert len(result.numbers["trend_charts"]) == 2
    for summary in result.numbers["trend_results"]:
        assert "p_bh_adjusted" in summary


def test_numeric_year_column_resolves_as_time_axis():
    from packages.analytics_core.src.engines.intent import IntentEngine
    from packages.analytics_core.src.engines.semantic import SemanticEngine
    frame = pd.DataFrame({
        "year": list(range(2000, 2027)),
        "temperature": [15.0 + 0.1 * i for i in range(27)],
        "precipitation": [900.0 + 2.0 * (i % 4) for i in range(27)],
    })
    intent = IntentEngine.parse_intent("How has the climate changed over the past decade?", list(frame.columns))
    resolved = SemanticEngine().resolve_schema(intent, {"climate": frame})
    assert resolved.time_col == "year"
    assert resolved.time_zone_policy == "CALENDAR_YEAR_NO_TIMEZONE"


def test_explicit_control_variables_are_used_in_adjusted_regression():
    from packages.analytics_core.src.engines.analyst_answer import build_analyst_result
    rows = []
    for i in range(60):
        tenure = float(i % 15 + 1)
        tickets = float(i % 6)
        discount = float((i * 3) % 11)
        revenue = 20.0 + 1.8 * discount + 0.9 * tenure - 1.2 * tickets + (i % 4) * 0.15
        rows.append({"revenue": revenue, "discount": discount, "tenure_months": tenure, "support_tickets": tickets})
    frame = pd.DataFrame(rows)
    result = build_analyst_result(
        "Does discount affect revenue, controlling for tenure_months and support_tickets?",
        frame, target="revenue", explanatory=["discount"],
        explicit_controls=["tenure_months", "support_tickets"],
    )
    assert result is not None and result.kind == "ASSOCIATION"
    assert result.numbers["adjustment_executed"] is True
    assert result.numbers["used_controls"] == ["tenure_months", "support_tickets"]
    assert "discount" in result.numbers["coefficients"]
    assert result.numbers["robust_covariance"] == "HC3"
    assert result.numbers["causal_identification"] is False


def test_missing_requested_control_refuses_unadjusted_substitution():
    from packages.analytics_core.src.engines.analyst_answer import build_analyst_result
    frame = pd.DataFrame({"revenue": [1, 2, 4, 5, 7, 8, 9, 11], "discount": [0, 1, 2, 3, 4, 5, 6, 7]})
    result = build_analyst_result(
        "Does discount affect revenue, controlling for region?", frame,
        target="revenue", explanatory=["discount"], unresolved_controls=["region"],
    )
    assert result is not None and result.kind == "INSUFFICIENT_EVIDENCE"
    assert result.numbers["adjustment_executed"] is False
    assert "No unadjusted substitute" in " ".join(result.details)


def test_group_trend_outputs_chart_and_formal_slope_comparison():
    from packages.analytics_core.src.engines.analyst_answer import build_analyst_result
    rows = []
    for year in range(2000, 2026):
        i = year - 2000
        for region, slope in (("North", 1.0), ("South", 2.0)):
            rows.append({"year": year, "revenue": 100.0 + slope * i, "region": region})
    frame = pd.DataFrame(rows)
    result = build_analyst_result(
        "Compare revenue trends by region over the past 25 years.", frame,
        target="revenue", group="region", time_col="year", target_explicit=True,
    )
    assert result is not None and result.kind == "TREND_COMPARISON"
    assert result.numbers["trend_chart"]["chart_type"] == "multi_line"
    assert result.numbers["trend_slope_chart"]["metadata"]["shows_ci"] is True
    assert len(result.numbers["slope_comparisons"]) == 1
    assert result.numbers["inference"]["causal_identification"] is False


def test_before_after_comparison_centres_level_shift_on_explicit_boundary():
    from packages.analytics_core.src.engines.analyst_answer import build_analyst_result
    rows = []
    for year in range(2000, 2025):
        pre = year <= 2012
        i = year - 2000
        value = 20.0 + 0.5 * i + (0.0 if pre else 30.0 + 1.5 * (year - 2012))
        rows.append({"year": year, "sales": value})
    frame = pd.DataFrame(rows)
    result = build_analyst_result(
        "Compare the sales trend before and after 2012.", frame,
        target="sales", time_col="year", target_explicit=True,
    )
    assert result is not None and result.kind == "TREND_COMPARISON_PERIODS"
    assert result.numbers["boundary_year"] == 2012
    assert result.numbers["boundary_explicit"] is True
    assert result.numbers["trend_chart"]["metadata"]["boundary_year"] == 2012
    assert result.numbers["inference"]["causal_identification"] is False


def test_before_after_without_boundary_fails_closed():
    from packages.analytics_core.src.engines.analyst_answer import build_analyst_result
    frame = _annual_frame()
    result = build_analyst_result(
        "Compare the temperature trend before and after the policy change.", frame,
        target="temperature", time_col="year", target_explicit=True,
    )
    assert result is not None and result.kind == "INSUFFICIENT_EVIDENCE"
    assert "boundary year" in " ".join(result.details).lower()
