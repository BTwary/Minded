from __future__ import annotations

from io import BytesIO
import duckdb
import pandas as pd
import pytest
from pypdf import PdfReader
import docx

from packages.analytics_core.src.engines.dataset_provider import InMemoryDatasetProvider
from packages.analytics_core.src.governance.guided_investigation import (
    ResolverStatus,
    apply_human_guidance_to_contract,
    parse_relational_filters,
)
from packages.analytics_core.src.reporting.exporter import (
    export_investigation_to_docx,
    export_investigation_to_pdf,
)
from packages.analytics_core.src.runtime.approved_execution import (
    compile_approved_aggregate,
)


def test_parse_relational_filters_all_operators() -> None:
    available_cols = {"order_status", "amount", "category", "created_date"}

    # Equality = and ==
    f1, err1 = parse_relational_filters("order_status = 'Delivered'", available_cols)
    assert err1 is None
    assert f1 == [{"column": "order_status", "operator": "=", "value": "Delivered"}]

    f2, err2 = parse_relational_filters("order_status == Delivered", available_cols)
    assert err2 is None
    assert f2 == [{"column": "order_status", "operator": "=", "value": "Delivered"}]

    # Inequality != and <>
    f3, err3 = parse_relational_filters("order_status != 'Cancelled'", available_cols)
    assert err3 is None
    assert f3 == [{"column": "order_status", "operator": "!=", "value": "Cancelled"}]

    f4, err4 = parse_relational_filters("order_status <> Cancelled", available_cols)
    assert err4 is None
    assert f4 == [{"column": "order_status", "operator": "!=", "value": "Cancelled"}]

    # Comparisons <, <=, >, >=
    f5, err5 = parse_relational_filters("amount > 100", available_cols)
    assert err5 is None
    assert f5 == [{"column": "amount", "operator": ">", "value": 100.0}]

    f6, err6 = parse_relational_filters("amount >= 50.5", available_cols)
    assert err6 is None
    assert f6 == [{"column": "amount", "operator": ">=", "value": 50.5}]

    f7, err7 = parse_relational_filters("amount < 500", available_cols)
    assert err7 is None
    assert f7 == [{"column": "amount", "operator": "<", "value": 500.0}]

    f8, err8 = parse_relational_filters("amount <= 200", available_cols)
    assert err8 is None
    assert f8 == [{"column": "amount", "operator": "<=", "value": 200.0}]

    # IS NULL and IS NOT NULL
    f9, err9 = parse_relational_filters("category is null", available_cols)
    assert err9 is None
    assert f9 == [{"column": "category", "operator": "IS NULL"}]

    f10, err10 = parse_relational_filters("category is not null", available_cols)
    assert err10 is None
    assert f10 == [{"column": "category", "operator": "IS NOT NULL"}]

    # IN and NOT IN
    f11, err11 = parse_relational_filters("order_status IN ('Delivered', 'Shipped')", available_cols)
    assert err11 is None
    assert f11 == [{"column": "order_status", "operator": "IN", "value": ["Delivered", "Shipped"]}]

    f12, err12 = parse_relational_filters("amount NOT IN (10, 20, 30)", available_cols)
    assert err12 is None
    assert f12 == [{"column": "amount", "operator": "NOT IN", "value": [10.0, 20.0, 30.0]}]

    # BETWEEN
    f13, err13 = parse_relational_filters("amount BETWEEN 25 AND 100", available_cols)
    assert err13 is None
    assert f13 == [{"column": "amount", "operator": "BETWEEN", "value": [25, 100]}]

    # Multi-condition with AND and leading WHERE
    f14, err14 = parse_relational_filters(
        "WHERE order_status = 'Delivered' AND amount >= 100 AND category IS NOT NULL",
        available_cols,
    )
    assert err14 is None
    assert len(f14) == 3
    assert f14[0] == {"column": "order_status", "operator": "=", "value": "Delivered"}
    assert f14[1] == {"column": "amount", "operator": ">=", "value": 100}
    assert f14[2] == {"column": "category", "operator": "IS NOT NULL"}


def test_freeform_custom_instruction_relational_filter() -> None:
    frame = pd.DataFrame(
        {
            "order_status": ["Delivered", "Pending", "Delivered"],
            "amount": [100.0, 50.0, 150.0],
            "region": ["East", "West", "East"],
        }
    )
    context = InMemoryDatasetProvider({"orders": frame}).acquire_context("project")

    # Custom instruction: "order_status is Delivered" (reproducing user screenshot 1)
    outcome = apply_human_guidance_to_contract(
        "What is total amount by region?",
        context,
        {"custom_instruction": "order_status is Delivered"},
    )
    assert outcome.status is ResolverStatus.RESOLVED
    assert outcome.contract.filter_conditions == ({"column": "order_status", "operator": "=", "value": "Delivered"},)

    # Custom instruction: relational expression with operator
    outcome2 = apply_human_guidance_to_contract(
        "What is total amount by region?",
        context,
        {"custom_instruction": "amount > 75"},
    )
    assert outcome2.status is ResolverStatus.RESOLVED
    assert outcome2.contract.filter_conditions == ({"column": "amount", "operator": ">", "value": 75},)


def test_approved_execution_relational_operators() -> None:
    frame = pd.DataFrame(
        {
            "category": ["A", "B", "C", "A", None],
            "amount": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    context = InMemoryDatasetProvider({"orders": frame}).acquire_context("project")

    # Test != in contract and execution
    outcome_ineq = apply_human_guidance_to_contract(
        "What is total amount?",
        context,
        {"selected_scope_filter": "category != 'A'"},
    )
    assert outcome_ineq.status is ResolverStatus.RESOLVED
    binding_ineq = compile_approved_aggregate(outcome_ineq.contract, frame)
    sql_ineq = binding_ineq.plan.to_sql()
    assert "WHERE" in sql_ineq
    assert '"category" != \'A\'' in sql_ineq

    con = duckdb.connect()
    con.register("data_table", frame)
    res_ineq = con.execute(sql_ineq).df()
    assert res_ineq["value"].iloc[0] == 50.0  # 20.0 + 30.0

    # Test BETWEEN
    outcome_bet = apply_human_guidance_to_contract(
        "What is total amount?",
        context,
        {"selected_scope_filter": "amount BETWEEN 15 AND 35"},
    )
    assert outcome_bet.status is ResolverStatus.RESOLVED
    binding_bet = compile_approved_aggregate(outcome_bet.contract, frame)
    sql_bet = binding_bet.plan.to_sql()
    assert "BETWEEN" not in sql_bet  # compiled into >= and <=
    assert '>= 15' in sql_bet and '<= 35' in sql_bet

    res_bet = con.execute(sql_bet).df()
    assert res_bet["value"].iloc[0] == 50.0  # 20.0 + 30.0

    # Test IS NOT NULL
    outcome_not_null = apply_human_guidance_to_contract(
        "How many records?",
        context,
        {"selected_scope_filter": "category IS NOT NULL"},
    )
    assert outcome_not_null.status is ResolverStatus.RESOLVED
    binding_not_null = compile_approved_aggregate(outcome_not_null.contract, frame)
    sql_not_null = binding_not_null.plan.to_sql()
    assert "IS NOT NULL" in sql_not_null

    res_not_null = con.execute(sql_not_null).df()
    assert res_not_null["value"].iloc[0] == 4


def test_export_with_list_of_dicts_and_special_xml_chars() -> None:
    # Contains <, >, &, and list-of-dicts table
    investigation = {
        "id": "inv-relational-001",
        "question": "What is the revenue where margin > 20% & region < 'South'?",
        "status": "COMPLETED",
        "verdict_type": "OBSERVED",
        "reproducible_manifest_hash": "deadbeef1234567890abcdefdeadbeef1234567890abcdefdeadbeef12345678",
        "analyst_result": {
            "kind": "TABULAR",
            "headline": "High-margin revenue summary (margin > 20% & active = true)",
            "numbers": {
                "rows": [
                    {"region": "North <Tier 1>", "revenue": 10500.5, "status": "active & verified"},
                    {"region": "East <Tier 2>", "revenue": 8900.0, "status": "active & verified"},
                ]
            },
            "sql_query": "SELECT region, SUM(revenue) FROM orders WHERE margin > 0.20 AND status = 'active' GROUP BY region",
            "details": ["Analysis includes only records where margin > 20% & status is verified."],
            "caveats": ["Filtered records where amount < 0 or status = 'cancelled'."],
        },
    }

    # Test PDF generation (would fail with ExpatError if xml unescaped)
    pdf_bytes = export_investigation_to_pdf(investigation)
    assert len(pdf_bytes) > 0
    reader = PdfReader(BytesIO(pdf_bytes))
    text = "".join(page.extract_text() or "" for page in reader.pages)
    assert "High-margin revenue summary" in text
    assert "North <Tier 1>" in text

    # Test DOCX generation
    docx_bytes = export_investigation_to_docx(investigation)
    assert len(docx_bytes) > 0
    doc = docx.Document(BytesIO(docx_bytes))
    docx_text = "\n".join(p.text for p in doc.paragraphs)
    assert "High-margin revenue summary" in docx_text


def test_comma_separated_filters_arbitrary_count() -> None:
    available_cols = {
        "order_status", "amount", "region", "category", "created_date",
        "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10"
    }

    # Comma-separated filters with multiple operators
    raw = "order_status = 'Delivered', amount > 100, region != 'North', category IN ('A', 'B'), amount BETWEEN 10 AND 50"
    filters, err = parse_relational_filters(raw, available_cols)
    assert err is None
    assert len(filters) == 5
    assert filters[0] == {"column": "order_status", "operator": "=", "value": "Delivered"}
    assert filters[1] == {"column": "amount", "operator": ">", "value": 100}
    assert filters[2] == {"column": "region", "operator": "!=", "value": "North"}
    assert filters[3] == {"column": "category", "operator": "IN", "value": ["A", "B"]}
    assert filters[4] == {"column": "amount", "operator": "BETWEEN", "value": [10, 50]}

    # Any number of filters (e.g. 10 filters)
    raw_10 = "c1 = 1, c2 = 2, c3 = 3, c4 = 4, c5 = 5, c6 = 6, c7 = 7, c8 = 8, c9 = 9, c10 = 10"
    filters_10, err_10 = parse_relational_filters(raw_10, available_cols)
    assert err_10 is None
    assert len(filters_10) == 10
    for i in range(1, 11):
        assert filters_10[i - 1] == {"column": f"c{i}", "operator": "=", "value": i}


def test_compound_analyst_question_with_filters_group_and_ranking() -> None:
    from packages.analytics_core.src.governance.guided_investigation import (
        evaluate_guided_investigation,
        AmbiguityStatus,
    )

    df = pd.DataFrame({
        "order_status": ["Delivered", "Cancelled", "Delivered", "Delivered"],
        "realized_revenue": [100.0, 50.0, 200.0, 150.0],
        "region": ["North", "North", "South", "East"],
    })

    q = (
        "Among orders where order_status is Delivered, calculate the total "
        "realized_revenue for each region. Return every region, ranked from highest to lowest total."
    )

    # 1. Preview evaluation
    preview = evaluate_guided_investigation(q, df=df)
    assert preview.ambiguity_status == AmbiguityStatus.CLEAR
    assert preview.can_auto_proceed is True
    assert preview.target_metric == "realized_revenue"
    assert preview.group_dimension == "region"
    assert preview.scope == "order_status = 'Delivered'"
    assert "WHERE \"order_status\" = 'Delivered'" in preview.planned_calculation
    assert "GROUP BY \"region\"" in preview.planned_calculation
    assert "ORDER BY metric_value DESC" in preview.planned_calculation

    # 2. Canonical contract resolution
    assert preview.canonical_contract is not None
    contract = preview.canonical_contract
    assert contract.task_family == "RANKING"
    assert contract.target_column == "realized_revenue"
    assert contract.grouping_columns == ("region",)
    assert contract.requested_aggregation == "SUM"
    assert contract.ranking_direction == "DESC"
    assert contract.filter_conditions == ({"column": "order_status", "operator": "=", "value": "Delivered"},)

    # 3. Approved AST compilation & DuckDB execution
    binding = compile_approved_aggregate(contract, df)
    sql = binding.plan.to_sql()
    assert 'WHERE ("order_status" = \'Delivered\')' in sql
    assert "GROUP BY region" in sql
    assert "ORDER BY value DESC" in sql

    con = duckdb.connect()
    con.register("data_table", df)
    res = con.execute(sql).df()

    # South: 200, East: 150, North: 100 (Cancelled order in North was excluded!)
    assert len(res) == 3
    assert res.iloc[0]["region"] == "South" and res.iloc[0]["value"] == 200.0
    assert res.iloc[1]["region"] == "East" and res.iloc[1]["value"] == 150.0
    assert res.iloc[2]["region"] == "North" and res.iloc[2]["value"] == 100.0
