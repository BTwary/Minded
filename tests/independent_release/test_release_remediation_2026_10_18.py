"""Regression tests for combined v1.0.0 release-blocker remediation."""
import pandas as pd
from packages.analytics_core.src.governance.guided_investigation import evaluate_guided_investigation
from packages.analytics_core.src.intelligence.complex_objective_planner import plan_compound_question
from packages.analytics_core.src.governance.verification_gate import analyst_group_comparison_independently_verified


def frame():
    return pd.DataFrame({"region": ["North", "South", "East", "West"] * 5,
        "gross_sales": [100, 200, 150, 80] * 5, "revenue": [100, 200, 150, 80] * 5,
        "profit": [20, 30, 25, 10] * 5, "order_id": range(20),
        "customer_id": [1, 2, 3, 4, 5] * 4, "order_total": [100, 200, 150, 80] * 5,
        "order_date": pd.date_range("2024-01-01", periods=20, freq="100D")})


def test_bare_compare_does_not_auto_execute_empty_contract():
    r = evaluate_guided_investigation("Compare revenue", df=frame())
    assert not r.can_auto_proceed
    assert r.canonical_contract.completeness == "INCOMPLETE"
    assert "grouping_dimension_or_comparison_values" in r.canonical_contract.unresolved_roles


def test_rank_roles_bind_measure_and_group_correctly():
    r = evaluate_guided_investigation("Top region by gross_sales", df=frame())
    c = r.canonical_contract
    assert r.can_auto_proceed and c.target_column == "gross_sales"
    assert c.grouping_columns == ("region",) and c.requested_aggregation == "SUM"
    assert 'GROUP BY "region"' in r.planned_calculation and 'SUM("gross_sales")' in r.planned_calculation


def test_top_n_limit_is_hashed_and_reaches_relational_plan():
    d = frame()
    r = evaluate_guided_investigation("Top 3 regions by revenue", df=d)
    c = r.canonical_contract
    assert r.can_auto_proceed and c.target_column == "revenue"
    assert c.grouping_columns == ("region",) and c.ranking_limit == 3
    assert r.planned_calculation.endswith("LIMIT 3")
    assert c.contract_hash() == c.contract_hash()


def test_total_revenue_filter_is_preserved():
    r = evaluate_guided_investigation("Show total revenue where region = North", df=frame())
    c = r.canonical_contract
    assert r.can_auto_proceed and c.target_column == "revenue" and c.requested_aggregation == "SUM"
    assert {"column": "region", "operator": "=", "value": "North"} in c.filter_conditions
    assert 'WHERE "region" = \'North\'' in r.planned_calculation


def test_aov_never_binds_to_order_identifier():
    r = evaluate_guided_investigation("average order value per customer", df=frame())
    c = r.canonical_contract
    assert r.can_auto_proceed and c.target_column == "order_total"
    assert c.grouping_columns == ("customer_id",) and c.requested_aggregation == "MEAN"
    assert c.target_column != "order_id"


def test_current_vs_prior_year_has_both_explicit_periods():
    r = evaluate_guided_investigation("Compare revenue between this year and last year", df=frame())
    c = r.canonical_contract
    assert r.can_auto_proceed and c.time_column == "order_date"
    assert c.time_range["start"] and c.time_range["end"] and c.time_range["baseline_start"] and c.time_range["baseline_end"]
    assert "current_period" in r.planned_calculation and "baseline_period" in r.planned_calculation
    assert "UNION ALL" in r.planned_calculation


def test_compound_planner_keeps_both_requested_metrics():
    p = plan_compound_question("Compare revenue and profit by region")
    assert p.is_compound and not p.validation_issues
    assert [o.statement for o in p.objectives] == ["Compare revenue by region", "Compare profit by region"]


def test_group_comparison_verification_fails_closed():
    good = {"anova_p": .01, "kruskal_p": .02, "robust": True, "adequate_power": True}
    assert analyst_group_comparison_independently_verified("GROUP_COMPARISON", good)
    assert not analyst_group_comparison_independently_verified("GROUP_COMPARISON", {**good, "robust": False})
    assert not analyst_group_comparison_independently_verified("GROUP_COMPARISON", {**good, "anova_p": .2})
    assert not analyst_group_comparison_independently_verified("GROUP_COMPARISON", {**good, "kruskal_p": float("nan")})


def test_compound_preview_approves_all_resolved_objectives_not_one_clause():
    r = evaluate_guided_investigation("Compare revenue and profit by region", df=frame())
    assert r.can_auto_proceed
    assert r.canonical_contract.operation_graph["compound_objectives"]
    assert "STEP 1: Compare revenue by region" in r.planned_calculation
    assert "STEP 2: Compare profit by region" in r.planned_calculation
    assert "No objective is omitted" in r.proposed_interpretation
