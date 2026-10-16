"""Regression coverage for the controller's deterministic analyst-answer path.

The production controller calls ``build_analyst_result`` after resolving its
columns.  These tests exercise that same contract compiler -> executor seam,
rather than a parser-only or hand-written SQL path.
"""
from dataclasses import replace

import pandas as pd

from packages.analytics_core.src.engines.analyst_answer import build_analyst_result
from packages.analytics_core.src.intelligence.canonical_question_contract import compile_canonical_question_contract


def _execute(question, frame):
    contract = compile_canonical_question_contract(question, frame)
    result = build_analyst_result(
        question, frame, target=contract.target_column,
        group=contract.grouping_columns[0] if contract.grouping_columns else None,
    )
    return contract, result


def test_grouped_row_count_is_count_star_and_preserves_missing_group():
    frame = pd.DataFrame({"plan_tier": ["basic", "pro", None, "basic"], "monthly_price": [10, 20, None, 10]})
    contract, result = _execute("Count the records in each plan_tier.", frame)
    assert contract.requested_aggregation == "ROW_COUNT"
    assert contract.grouping_columns == ("plan_tier",)
    assert result.numbers["table"] == {
        "basic": {"value": 2, "n": 2}, "pro": {"value": 1, "n": 1}, "<missing>": {"value": 1, "n": 1},
    }
    assert result.numbers["verification"]["status"] == "verified"


def test_grouped_sum_and_explicit_exclusion_are_executed_and_recorded():
    frame = pd.DataFrame({"region": ["east", "east", "west"], "revenue": [100, 50, 75], "status": ["active", "cancelled", "active"]})
    contract, result = _execute("Calculate total revenue by region, excluding cancelled subscriptions.", frame)
    assert contract.completeness == "COMPLETE"
    assert result.numbers["table"] == {"east": {"value": 100.0, "n": 1}, "west": {"value": 75.0, "n": 1}}
    assert result.numbers["filters"] == [{"column": "status", "value": "cancelled", "operator": "!="}]


def test_unresolved_exclusion_is_not_represented_as_an_executed_answer():
    frame = pd.DataFrame({"region": ["east", "west"], "revenue": [100, 75]})
    contract, result = _execute("Calculate total revenue by region, excluding cancelled subscriptions.", frame)
    assert contract.completeness == "INCOMPLETE"
    assert "unresolved_role:filter_instruction" in contract.unresolved_semantics
    assert result is None


def test_approved_structured_choice_overrides_the_original_question_at_execution():
    frame = pd.DataFrame({"plan_tier": ["basic", "pro", "basic"], "monthly_price": [10, 20, 10]})
    proposed = compile_canonical_question_contract("Calculate total monthly_price by plan_tier.", frame)
    approved = replace(
        proposed,
        task_family="COUNT",
        estimand="count",
        target_column=None,
        requested_aggregation="ROW_COUNT",
        grouping_columns=("plan_tier",),
        completeness="COMPLETE",
        resolution_evidence={"source": "validated_human_guidance"},
    )
    result = build_analyst_result(
        proposed.question_text, frame, target="monthly_price", group="plan_tier", approved_contract=approved,
    )
    assert result.numbers["operation"] == "ROW_COUNT"
    assert {key: item["value"] for key, item in result.numbers["table"].items()} == {"basic": 2, "pro": 1}
    assert result.numbers["canonical_contract"]["resolution_evidence"]["source"] == "validated_human_guidance"

