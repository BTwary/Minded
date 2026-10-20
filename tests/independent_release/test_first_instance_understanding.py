"""A question should be understood at first instance; when it cannot be, ask the fewest, shortest questions."""
import pytest

from tests.independent_release._first_instance_data import make
from packages.analytics_core.src.governance.guided_investigation import (
    apply_human_guidance_to_contract, evaluate_guided_investigation,
)
from packages.analytics_core.src.intelligence.canonical_question_contract import compile_canonical_question_contract
from packages.analytics_core.src.intelligence.complex_objective_planner import plan_compound_question

SCREENSHOT_Q = ("Among Corporate orders that were delivered and not returned, "
                "which sales channel had the highest average gross sales per order?")


@pytest.fixture(scope="module")
def df():
    return make()


def test_screenshot_question_is_understood_without_any_question(df):
    plan = evaluate_guided_investigation(SCREENSHOT_Q, df=df)
    assert plan.can_auto_proceed and not plan.clarification_options
    sql = plan.planned_calculation
    assert 'AVG("gross_sales")' in sql and 'GROUP BY "sales_channel"' in sql and "ORDER BY metric_value DESC" in sql
    for predicate in ('"customer_segment" = \'Corporate\'', '"order_status" = \'Delivered\'', '"order_status" != \'Returned\''):
        assert predicate in sql                      # no population qualifier silently dropped
    assert any("customer_segment = Corporate" in a for a in plan.assumptions)   # and the user can see them


def test_scope_clause_is_not_a_second_operation():
    assert not plan_compound_question(SCREENSHOT_Q).is_compound
    assert plan_compound_question("What is the total revenue and which region is best?").is_compound


def test_ambiguous_value_is_never_bound_to_a_guessed_column(df):
    d = df.copy()
    d["billing_tier"] = "Corporate"                   # same literal now lives in two columns
    c = compile_canonical_question_contract("Average gross_sales for Corporate orders by sales_channel", d)
    assert not any(f["value"] == "Corporate" for f in c.filter_conditions)


def test_count_ranking_counts_instead_of_summing_another_measure(df):
    for q, direction in (("Which region has the most orders?", "DESC"), ("Which region has the fewest orders?", "ASC")):
        plan = evaluate_guided_investigation(q, df=df)
        assert plan.can_auto_proceed
        assert 'COUNT(DISTINCT "order_id")' in plan.planned_calculation and direction in plan.planned_calculation
        assert "SUM(" not in plan.planned_calculation


def test_multi_part_question_asks_one_question_with_the_parts_as_options(df):
    q = "What is the total gross_sales and which region has the most orders?"
    plan = evaluate_guided_investigation(q, df=df)
    assert not plan.can_auto_proceed and plan.pending_intent_slot == "operation_graph"
    assert len(plan.clarification_options) <= 3 and all(len(o) < 80 for o in plan.clarification_options)
    assert not set(plan.clarification_options) & set(df.columns)        # never a column dump
    for patch in plan.clarification_option_patches:                     # a click resolves it
        out = apply_human_guidance_to_contract(q, df, {"intent_patch": patch["patch"]})
        assert out.status.name == "RESOLVED" and out.contract.completeness == "COMPLETE"


def test_typed_answer_and_ordinal_also_work_for_multi_part_question(df):
    q = "What is the total gross_sales and which region has the most orders?"
    typed = apply_human_guidance_to_contract(q, df, {"clarification_response": "average gross_sales by region", "pending_intent_slot": "operation_graph"})
    assert typed.status.name == "RESOLVED" and typed.contract.grouping_columns == ("region",)
    first = apply_human_guidance_to_contract(q, df, {"clarification_response": "1", "pending_intent_slot": "operation_graph"})
    assert first.status.name == "RESOLVED" and first.contract.requested_aggregation == "SUM"
    again = apply_human_guidance_to_contract(q, df, {"clarification_response": "total gross_sales and also which region leads", "pending_intent_slot": "operation_graph"})
    assert again.status.name != "RESOLVED"        # still two asks: refused, not half-answered


def test_per_customer_ratio_offers_two_executable_readings_not_columns(df):
    q = "What is the average gross_sales per customer?"
    plan = evaluate_guided_investigation(q, df=df)
    assert not plan.can_auto_proceed and len(plan.clarification_options) == 2
    assert not set(plan.clarification_options) & set(df.columns)
    for patch in plan.clarification_option_patches:
        out = apply_human_guidance_to_contract(q, df, {"intent_patch": patch["patch"]})
        assert out.status.name == "RESOLVED"


def test_unresolved_column_slots_suggest_at_most_three_ranked_columns(df):
    from packages.analytics_core.src.intelligence.intent_state import compact_resolution_candidates
    for slot in ("target_column", "grouping_columns", "denominator_column", "time_column"):
        assert len(compact_resolution_candidates(slot, df, "average gross sales by channel")) <= 3
    assert compact_resolution_candidates("target_column", df, "average gross sales")[0] == "gross_sales"
    assert "order_id" not in compact_resolution_candidates("target_column", df, "order")      # identifiers are not measures
