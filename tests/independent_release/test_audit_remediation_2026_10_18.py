import pandas as pd
import pytest
from packages.analytics_core.src.governance.guided_investigation import evaluate_guided_investigation
from packages.analytics_core.src.intelligence.canonical_question_contract import compile_canonical_question_contract
from packages.analytics_core.src.governance.verification_gate import (
    analyst_group_comparison_dual_statistical_confirmation,
    analyst_group_comparison_independently_verified,
)


def test_h1_compound_plans_and_supported_graph():
    df = pd.DataFrame({
        "region": ["East", "West", "East"],
        "revenue": [10.0, 20.0, 30.0],
        "profit": [1.0, 2.0, 3.0],
        "gross_sales": [15.0, 25.0, 35.0],
        "order_id": ["O1", "O2", "O3"],
    })

    # Unsupported multi-metric compound question is refused:
    p_diff = evaluate_guided_investigation("Rank revenue by region and compare profit by region", df=df)
    assert p_diff.can_auto_proceed is False
    assert p_diff.canonical_contract.operation_graph is None
    assert p_diff.planned_calculation.startswith("Not executable")

    # Multi-part question asks with parts as options:
    p_multi = evaluate_guided_investigation("What is the total gross_sales and which region has the most orders?", df=df)
    assert p_multi.can_auto_proceed is False
    assert p_multi.pending_intent_slot == "operation_graph"

    # Supported compound question is approved as aggregate-then-rank graph:
    p_graph = evaluate_guided_investigation("Calculate total revenue by region and rank regions by revenue", df=df)
    assert p_graph.can_auto_proceed is True
    assert p_graph.canonical_contract.completeness == "COMPLETE"
    assert p_graph.canonical_contract.operation_graph is not None
    assert "OPERATION GRAPH" in p_graph.planned_calculation


def test_h2_filter_rules_anchor_and_never_overwrite_existing():
    df = pd.DataFrame({
        "realized_revenue": [10.0, 20.0, 30.0, 40.0],
        "region": ["North", "South", "North", "South"],
        "customer_segment": ["Corporate", "Consumer", "Corporate", "Consumer"],
        "order_status": ["Returned", "Cancelled", "Delivered", "Shipped"],
    })

    # Question with existing compiler filter NOT IN:
    q_exclude = (
        "Calculate the total realized_revenue for each region, excluding records where "
        "order_status is Returned or Cancelled. Return the total for every region."
    )
    p_exclude = evaluate_guided_investigation(q_exclude, df=df)
    assert p_exclude.canonical_contract.filter_conditions == ({
        "column": "order_status", "operator": "NOT IN", "value": ["Returned", "Cancelled"]
    },)

    # Disjunction should not become an equality string 'North or South':
    p_or = evaluate_guided_investigation("Show total realized_revenue where region is North or South", df=df)
    filters_or = p_or.canonical_contract.filter_conditions
    assert not any(f.get("value") == "North or South" for f in filters_or)

    # Conjunction should not become 'North and customer_segment = Corporate':
    p_and = evaluate_guided_investigation("Show total realized_revenue where region = North and customer_segment = Corporate", df=df)
    filters_and = p_and.canonical_contract.filter_conditions
    assert not any("and" in str(f.get("value")) for f in filters_and)

    # Simple single equality filter is preserved:
    p_simple = evaluate_guided_investigation("Show total realized_revenue where region = North", df=df)
    assert p_simple.canonical_contract.filter_conditions == ({"column": "region", "operator": "=", "value": "North"},)


def test_h3_top_n_multi_ranking_and_out_of_bounds_blocked():
    df = pd.DataFrame({
        "region": ["East", "West"],
        "revenue": [100.0, 200.0],
        "profit": [10.0, 20.0],
    })

    # Multi-ranking question blocked:
    p_multi_rank = evaluate_guided_investigation("Top 3 region by revenue and bottom 2 region by profit", df=df)
    assert p_multi_rank.can_auto_proceed is False
    assert p_multi_rank.pending_intent_slot == "operation_graph"

    # Out-of-bounds limit (>100) blocked:
    p_oob = evaluate_guided_investigation("Top 500 region by revenue", df=df)
    assert p_oob.can_auto_proceed is False
    assert p_oob.pending_intent_slot == "ranking_limit"

    # Valid Top-N within 1-100 allowed:
    p_valid = evaluate_guided_investigation("Top 3 region by revenue", df=df)
    assert p_valid.can_auto_proceed is True
    assert p_valid.canonical_contract.ranking_limit == 3
    assert p_valid.canonical_contract.target_column == "revenue"
    assert p_valid.canonical_contract.grouping_columns == ("region",)

    # Explicit ranking Top region by gross_sales consistency across preview and authoritative resolver:
    df_rank = pd.DataFrame({"region": ["North", "South"], "gross_sales": [100.0, 200.0]})
    p_rank = evaluate_guided_investigation("Top region by gross_sales", df=df_rank)
    from packages.analytics_core.src.governance.guided_investigation import apply_human_guidance_to_contract, ResolverStatus
    r_rank = apply_human_guidance_to_contract(question="Top region by gross_sales", data_context=df_rank, guidance={})
    assert p_rank.can_auto_proceed is True
    assert p_rank.canonical_contract.target_column == "gross_sales"
    assert p_rank.canonical_contract.grouping_columns == ("region",)
    assert r_rank.status == ResolverStatus.RESOLVED
    assert r_rank.contract.target_column == "gross_sales"
    assert r_rank.contract.grouping_columns == ("region",)


def test_h4_average_order_value_scalar_and_customer_grouping():
    from packages.analytics_core.src.governance.guided_investigation import apply_human_guidance_to_contract, ResolverStatus
    # Scalar AOV without customer_id
    df_no_cust = pd.DataFrame({
        "order_id": [1, 2, 3],
        "order_total": [100.0, 200.0, 300.0],
    })
    p_scalar = evaluate_guided_investigation("average order value", df=df_no_cust)
    assert p_scalar.can_auto_proceed is True
    assert p_scalar.canonical_contract.target_column == "order_total"
    assert p_scalar.canonical_contract.grouping_columns == ()
    assert p_scalar.canonical_contract.requested_aggregation == "MEAN"

    # Scalar AOV with revenue
    df_rev = pd.DataFrame({
        "order_id": [1, 2, 3],
        "revenue": [100.0, 200.0, 300.0],
    })
    p_rev = evaluate_guided_investigation("average order value", df=df_rev)
    assert p_rev.can_auto_proceed is True
    assert p_rev.canonical_contract.target_column == "revenue"
    assert p_rev.canonical_contract.grouping_columns == ()

    # AOV per customer groups by customer
    df_cust = pd.DataFrame({
        "order_id": [1, 2, 3],
        "customer_id": ["c1", "c2", "c1"],
        "order_total": [100.0, 200.0, 300.0],
    })
    p_cust = evaluate_guided_investigation("average order value per customer", df=df_cust)
    assert p_cust.can_auto_proceed is True
    assert p_cust.canonical_contract.target_column == "order_total"
    assert p_cust.canonical_contract.grouping_columns == ("customer_id",)
    assert p_cust.canonical_contract.requested_aggregation == "MEAN"

    # Multi-amount consistency across preview and resolver on frame with revenue and order_total:
    df_multi_amount = pd.DataFrame({
        "order_id": [1, 2, 3],
        "revenue": [100.0, 200.0, 300.0],
        "order_total": [100.0, 200.0, 300.0],
        "customer_id": ["c1", "c2", "c1"],
    })
    p_multi_aov = evaluate_guided_investigation("average order value", df=df_multi_amount)
    r_multi_aov = apply_human_guidance_to_contract(question="average order value", data_context=df_multi_amount, guidance={})
    assert p_multi_aov.canonical_contract.target_column == "order_total"
    assert r_multi_aov.contract.target_column == "order_total"
    assert r_multi_aov.status == ResolverStatus.RESOLVED

    # AOV per customer on multi-amount frame consistency:
    p_multi_cust = evaluate_guided_investigation("average order value per customer", df=df_multi_amount)
    r_multi_cust = apply_human_guidance_to_contract(question="average order value per customer", data_context=df_multi_amount, guidance={})
    assert p_multi_cust.can_auto_proceed is True
    assert p_multi_cust.canonical_contract.target_column == "order_total"
    assert p_multi_cust.canonical_contract.grouping_columns == ("customer_id",)
    assert r_multi_cust.status == ResolverStatus.RESOLVED
    assert r_multi_cust.contract.target_column == "order_total"
    assert r_multi_cust.contract.grouping_columns == ("customer_id",)


def test_m1_group_comparison_statistical_confirmation():
    # Check that dual statistical confirmation function works and legacy alias exists
    good = {"anova_p": 0.01, "kruskal_p": 0.02, "robust": True, "adequate_power": True}
    assert analyst_group_comparison_dual_statistical_confirmation("GROUP_COMPARISON", good) is True
    assert analyst_group_comparison_independently_verified("GROUP_COMPARISON", good) is True
    assert analyst_group_comparison_dual_statistical_confirmation("GROUP_COMPARISON", {**good, "robust": False}) is False


def test_m2_partial_year_warning_in_assumptions():
    from packages.analytics_core.src.governance.guided_investigation import apply_human_guidance_to_contract, ResolverStatus
    df = pd.DataFrame({
        "order_date": ["2025-01-01", "2025-06-01", "2026-01-01", "2026-03-09"],
        "revenue": [100.0, 200.0, 300.0, 400.0],
    })
    p = evaluate_guided_investigation("Compare revenue this year and last year", df=df)
    assert any("partial period" in a.lower() for a in p.assumptions)
    r_m2 = apply_human_guidance_to_contract(question="Compare revenue this year and last year", data_context=df, guidance={})
    assert p.can_auto_proceed is True
    assert p.canonical_contract.target_column == "revenue"
    assert p.canonical_contract.time_column == "order_date"
    assert r_m2.status == ResolverStatus.RESOLVED
    assert r_m2.contract.target_column == "revenue"
    assert r_m2.contract.time_column == "order_date"


def test_m3_no_dead_end_clarification_options():
    df = pd.DataFrame({"region": ["East", "West"], "revenue": [100.0, 200.0], "profit": [10.0, 20.0]})

    p1 = evaluate_guided_investigation("Top 500 region by revenue", df=df)
    assert "Clarify the unresolved analytical requirement" not in p1.clarification_options
    assert len(p1.clarification_options) >= 2

    p2 = evaluate_guided_investigation("Compare revenue", df=df)
    assert "Clarify the unresolved analytical requirement" not in p2.clarification_options
    assert len(p2.clarification_options) >= 2
    # Measures like revenue and profit must never be offered as grouping choices:
    assert not any("revenue" in opt.lower() or "profit" in opt.lower() for opt in p2.clarification_options)
    assert any("region" in opt.lower() for opt in p2.clarification_options)

    df_no_amount = pd.DataFrame({"order_id": [1, 2], "name": ["a", "b"]})
    p3 = evaluate_guided_investigation("average order value", df=df_no_amount)
    assert "Clarify the unresolved analytical requirement" not in p3.clarification_options
    assert len(p3.clarification_options) >= 2
