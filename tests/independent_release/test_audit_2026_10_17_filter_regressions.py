"""Regression tests for the 2026-10-17 audit of the filter/query-output polish."""
import pandas as pd
import pytest

from packages.analytics_core.src.governance.guided_investigation import (
    evaluate_guided_investigation,
    parse_relational_literal,
)
from packages.analytics_core.src.intelligence.canonical_question_contract import extract_filter_conditions
from packages.analytics_core.src.runtime.controller_scope import validate_scope_filter_expression

pytest.importorskip("sqlglot")
COLS = ["profit", "region", "name", "order_status", "realized_revenue"]


def test_trailing_task_clause_is_not_swallowed_into_in_list():
    q = "Among orders where order_status is Delivered, calculate the total realized_revenue for each region."
    conds = extract_filter_conditions(q, COLS)
    assert [(c["column"], c["operator"], c.get("value")) for c in conds] == [("order_status", "=", "Delivered")]


def test_real_alternatives_still_become_in_list():
    q = "Calculate total realized_revenue where order_status is Returned or Cancelled, return every region."
    conds = extract_filter_conditions(q, COLS)
    assert conds and conds[0]["operator"] == "IN" and conds[0]["value"] == ["Returned", "Cancelled"]


@pytest.mark.parametrize("expr", ["profit < -100", "profit BETWEEN -5 AND 10", "profit IN (-1, 2)", "name = 'a;b'", "name = 'x--y'"])
def test_validator_accepts_negative_numbers_and_punctuation_in_literals(expr):
    assert validate_scope_filter_expression(expr, COLS)


@pytest.mark.parametrize("expr", [
    "profit = 1; DROP TABLE t", "profit = 1 -- x", "profit = 1 /* x */", "profit > -(1)", "profit > -profit",
    "profit > NULL", "region IN ('A', NULL)", "profit BETWEEN NULL AND 3", "profit BETWEEN 1 AND region",
    "profit = 1 OR 1 = 1", "profit > (SELECT 1)",
])
def test_validator_still_rejects_unsafe_or_null_forms(expr):
    with pytest.raises(ValueError):
        validate_scope_filter_expression(expr, COLS)


def test_relational_literal_numeric_grammar():
    assert parse_relational_literal("1e3") == 1000.0
    assert parse_relational_literal("-5") == -5
    assert parse_relational_literal("nan") == "nan"
    assert parse_relational_literal("inf") == "inf"


def test_distinct_values_preview_does_not_sum_a_category():
    df = pd.DataFrame({"plan_tier": ["Basic", "Basic", "Pro", "Pro", "Enterprise"], "customer_id": range(5)})
    plan = evaluate_guided_investigation("what are the different tiers of subscriptions?", df)
    assert "SUM" not in plan.planned_calculation


def _recon_contract(**overrides):
    from types import SimpleNamespace
    base = dict(filter_conditions=(), time_range=None, time_column=None, comparison_values=(), grouping_columns=("channel",),
                target_column="amount", requested_aggregation="SUM", numerator_column=None, denominator_column=None,
                share_of_total=False, requested_outputs=(), ranking_direction="DESC", task_family="DESCRIPTIVE")
    base.update(overrides)
    return SimpleNamespace(**base)


def _recon(contract, spec_overrides=None):
    from types import SimpleNamespace
    from packages.analytics_core.src.runtime.execution_reconciliation import reconcile_approved_execution
    spec = dict(target_column="amount", aggregation="SUM", grouping_columns=["channel"], filters=[], share_of_total=False,
                time_range=None, comparison_values=[], requested_outputs=[], ranking_direction=None)
    spec.update(spec_overrides or {})
    plan = SimpleNamespace(to_sql=lambda: "SELECT 1")
    result = SimpleNamespace(source_datasets=["d"], source_dataset_fingerprints={}, source_dataset_versions={},
                             input_content_fingerprints={}, executed_query="SELECT 1")
    return reconcile_approved_execution(contract, SimpleNamespace(specification=spec, plan=plan), result, dataset_name="d",
                                        expected_fingerprints={}, expected_versions={}, expected_inputs={})


def test_reconciliation_accepts_rank_instruction_on_descriptive_grouped_task():
    # "total amount by channel, rank highest first": contract carries DESC, spec records None for non-RANKING tasks.
    assert _recon(_recon_contract())["status"] == "MATCHED"


def test_reconciliation_still_rejects_wrong_direction_for_ranking_task():
    out = _recon(_recon_contract(task_family="RANKING"), {"ranking_direction": "ASC"})
    assert out["status"] == "MISMATCH" and "ranking_direction_matches" in out["failed_checks"]


def test_reconciliation_fails_closed_without_grouping_dimension_for_comparison():
    out = _recon(_recon_contract(grouping_columns=(), comparison_values=("A", "B")), {"grouping_columns": [], "comparison_values": ["A", "B"]})
    assert out["status"] == "MISMATCH"


# ---- second pass: causal wording, row-unique grouping, forecast horizon, quoted identifiers ----
def _contract(question, df):
    from packages.analytics_core.src.intelligence.canonical_question_contract import compile_canonical_question_contract
    return compile_canonical_question_contract(question, df)


def test_why_question_across_group_is_not_a_plain_grouped_aggregate():
    df = pd.DataFrame({"datacenter_region": ["a", "b"] * 50, "cost_metric": range(100)})
    c = _contract("Why did cost_metric surge across datacenter_region?", df)
    assert c.family == "NONE" and c.task_family == "ROOT_CAUSE"


def test_plain_grouped_aggregate_is_unaffected_by_causal_guard():
    df = pd.DataFrame({"datacenter_region": ["a", "b"] * 50, "cost_metric": range(100)})
    c = _contract("Total cost_metric by datacenter_region", df)
    assert c.family == "SIMPLE_AGGREGATE" and list(c.grouping_columns) == ["datacenter_region"]


def test_per_row_unique_identifier_is_not_bound_as_grouping():
    df = pd.DataFrame({"order_id": range(20), "revenue": range(20), "region": ["n", "s"] * 10})
    avg = _contract("What is the average revenue per order?", df)
    assert "order_id" not in list(avg.grouping_columns)            # never one-row groups
    assert avg.completeness == "COMPLETE" and avg.target_column == "revenue"   # order grain: per order = per row
    total = _contract("What is the total revenue per order?", df)
    assert total.completeness != "COMPLETE"                         # no unambiguous meaning: not guessed


def test_per_customer_grouping_still_works_when_customers_repeat():
    df = pd.DataFrame({"customer_id": [1, 1, 2, 2, 3, 3], "revenue": range(6)})
    c = _contract("What is the average revenue per customer?", df)
    assert list(c.grouping_columns) == ["customer_id"] and c.completeness == "COMPLETE"


def test_recompiler_handles_quoted_identifiers_but_not_literals():
    import polars as pl
    from packages.analytics_core.src.engines.polars_query_recompiler import execute_known_query, _unquote_simple_identifiers
    df = pl.from_pandas(pd.DataFrame({"order_date": ["2026-01-01", "2026-01-01", "2026-01-02"], "revenue": [1.0, 2.0, 3.0]}))
    out = execute_known_query(
        "SELECT DATE_TRUNC('day', TRY_CAST(order_date AS DATE)) AS period, SUM(\"revenue\") AS \"revenue\" "
        "FROM data_table GROUP BY period ORDER BY period ASC", df)
    assert out is not None and out.shape == (2, 2)
    assert _unquote_simple_identifiers("x = 'say \"hi\"' AND \"y\" = 1") == "x = 'say \"hi\"' AND y = 1"
    assert '"b c"' in _unquote_simple_identifiers('SELECT SUM("a") AS "b c" FROM t')  # not simple: stays quoted


@pytest.mark.parametrize("q,expected", [
    ("Forecast next month's revenue trajectory", True), ("Forecast revenue for the next 3 months", True),
    ("Forecast sales next quarter", True), ("Forecast revenue", False),
])
def test_forecast_horizon_regex_accepts_implicit_single_period(q, expected):
    import re
    src = open("packages/analytics_core/src/runtime/controller.py", encoding="utf-8").read()
    pat = re.search(r'"SPECIFIED" if re\.search\(\s*r"(.+?)",\s*question', src, re.S).group(1)
    assert bool(re.search(pat, q, re.I)) is expected
