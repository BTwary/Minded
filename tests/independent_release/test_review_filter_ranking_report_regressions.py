"""Regression coverage for review guidance, explicit population filters and exports."""
from __future__ import annotations

from dataclasses import replace

import pandas as pd

from packages.analytics_core.src.intelligence.canonical_question_contract import compile_canonical_question_contract
from packages.analytics_core.src.governance.guided_investigation import ResolverStatus, apply_human_guidance_to_contract
from packages.analytics_core.src.reporting.exporter import _extract_table_data, _report_chart_axis_bounds
from packages.analytics_core.src.reporting.narrative_composer import compose_report_narrative


def _commerce_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "order_id": ["A", "B", "C", "D", "E", "F"],
        "region": ["South", "North", "East", "West", "South", "North"],
        "satisfaction_score": [3.1, 4.2, 3.9, 4.0, 3.6, 4.1],
        "order_status": ["Delivered", "Delivered", "Delivered", "Cancelled", "Delivered", "Delivered"],
        "returned": [0, 0, 1, 0, 0, 1],
        "support_load_band": ["High", "Low", "High", "High", "High", "Low"],
        # This overlapping token used to cause the word “high” to be discarded as a
        # possible band value even when the question explicitly said “high support load”.
        "high_support_flag": [True, False, True, True, True, False],
        "realized_revenue": [100.0, 90.0, 150.0, 0.0, 120.0, 75.0],
        "net_revenue": [95.0, 88.0, 145.0, 0.0, 110.0, 72.0],
        "customer_tenure_months": [12, 4, 18, 5, 7, 2],
    })


def test_high_support_load_qualifier_binds_to_the_named_band_even_with_high_support_flag_column():
    frame = _commerce_frame()
    question = "Among orders with high support load that were delivered and not returned, which region had the lowest average satisfaction score?"
    contract = compile_canonical_question_contract(question, frame)
    filters = {(f["column"], str(f["operator"]).upper(), str(f["value"]).casefold()) for f in contract.filter_conditions}
    assert ("support_load_band", "=", "high") in filters
    assert ("order_status", "=", "delivered") in filters
    assert ("returned", "=", "0") in filters
    assert contract.target_column == "satisfaction_score"
    assert contract.grouping_columns == ("region",)
    assert contract.requested_aggregation == "MEAN"
    assert contract.task_family == "RANKING"
    assert contract.ranking_direction == "ASC"
    assert contract.completeness == "COMPLETE"


def test_quoted_redundant_exclusion_is_accepted_and_keeps_ranking_contract():
    frame = _commerce_frame()
    question = "After excluding returns and cancellations, which region generated the most realized revenue?"
    base = compile_canonical_question_contract(question, frame)
    # Mirror the realistic schema-grounded contract from the 9,000-row test CSV,
    # where returns/cancellations are encoded by the shared lifecycle status field.
    base = replace(base, filter_conditions=({"column": "order_status", "operator": "NOT IN", "value": ["Returned", "Cancelled"]},))
    outcome = apply_human_guidance_to_contract(
        question,
        frame,
        {
            "selected_metric": "realized_revenue",
            "selected_aggregation": "SUM",
            "selected_group_dimension": "region",
            "custom_instruction": "'Exclude returns and cancellations'",
        },
        previous_contract=base,
    )
    assert outcome.status == ResolverStatus.RESOLVED
    assert outcome.contract is not None
    contract = outcome.contract
    assert contract.task_family == "RANKING"
    assert contract.estimand == "ranking"
    assert contract.target_column == "realized_revenue"
    assert contract.grouping_columns == ("region",)
    assert contract.ranking_direction == "DESC"
    assert len(contract.filter_conditions) == 1
    assert contract.filter_conditions[0]["column"] == "order_status"
    assert contract.filter_conditions[0]["operator"] == "NOT IN"
    assert contract.filter_conditions[0]["value"] == ["Returned", "Cancelled"]
    assert contract.completeness == "COMPLETE"


def test_saved_parameters_are_only_restored_for_a_clear_canonical_preview():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    panel = (root / "apps/web/src/components/ProposedAnalysisPanel.tsx").read_text(encoding="utf-8")
    assert 'const canRestoreSavedParameters = plan.ambiguity_status === "CLEAR"' in panel
    assert "savedMetric = canRestoreSavedParameters ? saved.metric : undefined" in panel
    assert "savedAggregation = canRestoreSavedParameters ? saved.aggregation : undefined" in panel
    assert "savedGroup = canRestoreSavedParameters ? saved.group : undefined" in panel
    assert "canRestoreSavedParameters && saved.metric !== undefined" in panel
    assert "canRestoreSavedParameters && saved.group !== undefined" in panel


def test_positive_only_bar_chart_axis_uses_zero_baseline_and_donut_has_no_numeric_scale():
    low, high = _report_chart_axis_bounds("bar", [3.2, 8.0, 4.1])
    assert low == 0.0
    assert high > 8.0
    low_negative, high_negative = _report_chart_axis_bounds("bar", [-8.0, -3.0])
    assert low_negative < -8.0
    assert high_negative == 0.0
    assert _report_chart_axis_bounds("donut", [2.0, 7.0]) == (2.0, 7.0)


def test_lowest_average_question_gets_lowest_first_table_without_misleading_share_column():
    meta = _extract_table_data(
        "DESCRIPTIVE",
        {
            "operation": "MEAN",
            "table": {
                "East": {"value": 3.81, "n": 200},
                "South": {"value": 3.71, "n": 180},
                "North": {"value": 3.77, "n": 190},
                "West": {"value": 3.79, "n": 185},
            },
        },
        "Among orders with high support load, which region had the lowest average satisfaction score?",
    )
    assert meta is not None
    assert meta["title"] == "Ranking by Metric"
    assert meta["headers"] == ["Rank", "Group / Entity", "Average Value", "Rows"]
    assert [row[1] for row in meta["rows"]] == ["South", "North", "West", "East"]
    assert "Share" not in meta["headers"]


def test_report_narrative_answers_lowest_instead_of_always_claiming_highest():
    result = {
        "kind": "DESCRIPTIVE",
        "headline": "Average satisfaction by region.",
        "numbers": {
            "operation": "MEAN",
            "target_column": "satisfaction_score",
            "grouping_columns": ["region"],
            "table": {
                "East": {"value": 3.81, "n": 200},
                "South": {"value": 3.71, "n": 180},
                "North": {"value": 3.77, "n": 190},
                "West": {"value": 3.79, "n": 185},
            },
        },
    }
    evidence = [{"id": "e-sat", "validation_status": "VERIFIED", "statement": "Verified region satisfaction averages"}]
    execution = {"aggregation": "MEAN", "contract_status": "MATCHED", "verification_status": "VERIFIED"}
    narrative = compose_report_narrative(
        question="Among orders with high support load that were delivered and not returned, which region had the lowest average satisfaction score?",
        analyst_result=result,
        execution_record=execution,
        evidence=evidence,
        status="COMPLETED",
    )
    assert narrative["status"] == "READY"
    assert "South recorded the lowest mean of satisfaction_score" in narrative["summary"]
    assert "East recorded the highest mean of satisfaction_score" not in narrative["summary"]
    assert narrative["validation"]["status"] == "PASSED"
