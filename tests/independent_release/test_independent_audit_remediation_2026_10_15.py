"""Regression tests for defects independently reproduced in the 2026-10-15 audit."""

import pandas as pd
import pytest

from packages.analytics_core.src.engines.stopping import StoppingEngine
from packages.analytics_core.src.intelligence.canonical_question_contract import extract_filter_conditions
from packages.analytics_core.src.profiling.data_quality_gate import DataQualityGate
from packages.analytics_core.src.runtime.state_reconstruction import classify_evidence_for_hypothesis
from packages.analytics_core.src.ingestion.robust_loader import RobustFileLoader


def test_failed_verification_cannot_become_supporting_due_to_nonzero_effect():
    assert classify_evidence_for_hypothesis("FAILED", 0.42) == "contradicting"
    assert classify_evidence_for_hypothesis("FAILED", 0.0) == "contradicting"


def test_unknown_verification_status_does_not_infer_support_from_effect_magnitude():
    assert classify_evidence_for_hypothesis(None, 0.42) is None
    assert classify_evidence_for_hypothesis("PENDING", -0.8) is None


def test_verified_evidence_is_classified_supporting():
    assert classify_evidence_for_hypothesis("VERIFIED", 0.0) == "supporting"


def test_stopping_fails_closed_when_verification_argument_is_omitted():
    decision = StoppingEngine.evaluate_stopping(
        leading_posterior=0.99,
        entropy_delta=-1.0,
        experiments_completed=5,
        counter_hypothesis_evaluated=True,
    )
    assert decision.should_stop is False
    assert decision.criteria_breakdown["verification_passed"] is False


def test_stopping_cannot_claim_resolution_when_verification_failed():
    decision = StoppingEngine.evaluate_stopping(
        leading_posterior=0.99,
        entropy_delta=-1.0,
        experiments_completed=5,
        all_verifications_passed=False,
        counter_hypothesis_evaluated=True,
    )
    assert decision.should_stop is False


def test_explicit_numeric_and_categorical_filters_resolve_arbitrary_columns():
    filters = extract_filter_conditions(
        "Show revenue where discount_rate > 0.1 and region = 'North'",
        ["revenue", "discount_rate", "region", "order_date"],
    )
    assert { (f["column"], f["operator"], f["value"]) for f in filters } == {
        ("discount_rate", ">", 0.1), ("region", "=", "North")
    }


def test_calendar_year_filter_becomes_half_open_date_range():
    filters = extract_filter_conditions("Show orders placed in 2025", ["order_date", "revenue"])
    assert filters == (
        {"column": "order_date", "operator": ">=", "value": "2025-01-01"},
        {"column": "order_date", "operator": "<", "value": "2026-01-01"},
    )


def test_iso_date_range_includes_the_full_final_day():
    filters = extract_filter_conditions(
        "Show orders between 2025-01-01 and 2025-03-31", ["order_date", "revenue"]
    )
    assert filters[0]["value"] == "2025-01-01"
    assert filters[1]["value"] == "2025-04-01"


def test_natural_language_region_and_completed_status_filters_resolve():
    region = extract_filter_conditions("Show sales in the North region", ["region", "sales"])
    status = extract_filter_conditions("Show completed orders", ["order_status", "revenue"])
    assert {"column": "region", "operator": "=", "value": "North"} in region
    assert {"column": "order_status", "operator": "=", "value": "completed"} in status


def test_malformed_unterminated_csv_quote_fails_closed():
    content = b"order_id,region,revenue\n1,North,10\n2,South,20\n3,\"North,30\n4,West,40\n"
    with pytest.raises(ValueError, match="unterminated quoted field"):
        RobustFileLoader().load(file_bytes=content, filename="malformed.csv")


def test_sparse_period_medians_do_not_create_scale_change_blocker():
    # Each date has only 3 rows; the apparent median ratio is 21x, but this is too
    # sparse to hard-block the whole dataset on the temporal-magnitude heuristic.
    frame = pd.DataFrame({
        "date": ["2025-01-01"] * 3 + ["2025-01-02"] * 3,
        "revenue": [1.0, 1.0, 1.0, 21.0, 21.0, 21.0],
        "order_id": list(range(1, 7)),
    })
    result = DataQualityGate.evaluate_fitness(frame, time_col="date", metric_col="revenue")
    assert not any("median jumps" in issue for issue in result.critical_issues)


def test_controller_stopping_call_passes_accumulated_verification_state():
    from pathlib import Path
    source = Path(__file__).parents[2] / "packages/analytics_core/src/runtime/controller.py"
    text = source.read_text(encoding="utf-8")
    start = text.index("stop_decision = StoppingEngine.evaluate_stopping(")
    block = text[start:text.index("\n            )", start) + 14]
    assert "all_verifications_passed=all_verifications_passed" in block


def test_semantic_engine_has_no_unmatched_first_metric_fallback():
    from pathlib import Path
    source = Path(__file__).parents[2] / "packages/analytics_core/src/engines/semantic.py"
    text = source.read_text(encoding="utf-8")
    assert "target_col = next(m for m in world_model.metrics if m.table_name == primary_table).column_name" not in text


def test_schema_derived_dimensions_support_unfamiliar_enterprise_columns():
    from packages.analytics_core.src.intelligence.canonical_question_contract import compile_canonical_question_contract
    frame = pd.DataFrame({"sales_territory": ["East", "West"], "customer_cohort": ["New", "Returning"], "net_margin": [2.0, 3.0]})
    territory = compile_canonical_question_contract("What is average net margin by sales territory?", frame)
    cohort = compile_canonical_question_contract("What is average net margin by customer cohort?", frame)
    assert territory.target_column == "net_margin"
    assert territory.grouping_columns == ("sales_territory",)
    assert cohort.target_column == "net_margin"
    assert cohort.grouping_columns == ("customer_cohort",)


def test_forecasting_series_is_sorted_by_time_not_input_row_order():
    from packages.analytics_core.src.forecasting.engine import ForecastingEngine
    frame = pd.DataFrame({"date": ["2025-03-01", "2025-01-01", "2025-02-01"], "revenue": [30.0, 10.0, 20.0]})
    prepared = ForecastingEngine._prepare_series(frame, "date", "revenue")
    assert list(pd.to_datetime(prepared["date"])) == list(pd.to_datetime(["2025-01-01", "2025-02-01", "2025-03-01"]))
    assert list(prepared["revenue"]) == [10.0, 20.0, 30.0]


def test_parsed_filters_apply_to_the_intended_population():
    from packages.analytics_core.src.intelligence.intent_state import filter_condition_mask
    frame = pd.DataFrame({
        "region": ["North", "North", "South"],
        "discount_rate": [0.2, 0.05, 0.3],
        "revenue": [100.0, 200.0, 300.0],
    })
    filters = extract_filter_conditions(
        "Show revenue where discount_rate > 0.1 and region = 'North'", list(frame.columns)
    )
    population = frame.copy()
    for condition in filters:
        population = population.loc[filter_condition_mask(
            population[condition["column"]], condition["operator"], condition.get("value")
        )]
    assert len(population) == 1
    assert population["revenue"].sum() == 100.0


def test_clause_aware_filter_and_grouping_parsing():
    from packages.analytics_core.src.intelligence.canonical_question_contract import compile_canonical_question_contract
    from packages.analytics_core.src.intelligence.analytical_operations import aggregate_frame

    frame = pd.DataFrame({
        "revenue": [100.0, 200.0, 300.0, 400.0],
        "region": ["East", "West", "East", "South"],
        "channel": ["Web", "Store", "Web", "Store"],
        "status": ["active", "inactive", "active", "active"],
    })

    # Case 1: "Calculate total revenue by region for active status"
    c1 = compile_canonical_question_contract("Calculate total revenue by region for active status", df=frame)
    assert c1.target_column == "revenue"
    assert c1.grouping_columns == ("region",)
    assert { (f["column"], f["operator"], f["value"]) for f in c1.filter_conditions } == {("status", "=", "active")}
    res1 = aggregate_frame(frame, operation="SUM", target_column=c1.target_column, grouping_columns=c1.grouping_columns, filters=c1.filter_conditions)
    assert res1["population_row_count"] == 3
    assert { (r["group"]["region"], r["value"]) for r in res1["records"] } == {("East", 400.0), ("South", 400.0)}

    # Case 2: "Calculate revenue by region West split by channel"
    c2 = compile_canonical_question_contract("Calculate revenue by region West split by channel", df=frame)
    assert c2.target_column == "revenue"
    assert c2.grouping_columns == ("channel",)
    assert { (f["column"], f["operator"], f["value"]) for f in c2.filter_conditions } == {("region", "=", "West")}
    res2 = aggregate_frame(frame, operation="SUM", target_column=c2.target_column, grouping_columns=c2.grouping_columns, filters=c2.filter_conditions)
    assert res2["population_row_count"] == 1
    assert res2["records"] == [{"group": {"channel": "Store"}, "value": 200.0, "row_count": 1}]

    # Case 3: "Calculate total revenue by region and channel for East region"
    c3 = compile_canonical_question_contract("Calculate total revenue by region and channel for East region", df=frame)
    assert c3.target_column == "revenue"
    assert c3.grouping_columns == ("region", "channel")
    assert { (f["column"], f["operator"], f["value"]) for f in c3.filter_conditions } == {("region", "=", "East")}
    res3 = aggregate_frame(frame, operation="SUM", target_column=c3.target_column, grouping_columns=c3.grouping_columns, filters=c3.filter_conditions)
    assert res3["population_row_count"] == 2
    assert res3["records"] == [{"group": {"region": "East", "channel": "Web"}, "value": 400.0, "row_count": 2}]

    # Case 4: "Show revenue by region excluding East"
    c4 = compile_canonical_question_contract("Show revenue by region excluding East", df=frame)
    assert c4.target_column == "revenue"
    assert c4.grouping_columns == ("region",)
    assert { (f["column"], f["operator"], f["value"]) for f in c4.filter_conditions } == {("region", "!=", "East")}
    res4 = aggregate_frame(frame, operation="SUM", target_column=c4.target_column, grouping_columns=c4.grouping_columns, filters=c4.filter_conditions)
    assert res4["population_row_count"] == 2
    assert { (r["group"]["region"], r["value"]) for r in res4["records"] } == {("West", 200.0), ("South", 400.0)}

    # The same exclusion must survive a comma that separates a clause from its grouping.
    c5 = compile_canonical_question_contract("Show revenue by region, excluding East", df=frame)
    assert c5.completeness == "COMPLETE"
    assert c5.grouping_columns == ("region",)
    assert { (f["column"], f["operator"], f["value"]) for f in c5.filter_conditions } == {("region", "!=", "East")}

