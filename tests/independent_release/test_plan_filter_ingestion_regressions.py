"""Release regressions for plan/filter/pre-analysis parity and fresh EXE frontend builds."""
from pathlib import Path

import pandas as pd

from packages.analytics_core.src.governance.guided_investigation import (
    ResolverStatus,
    apply_human_guidance_to_contract,
    compute_resolved_plan_hash,
    evaluate_guided_investigation,
)
from packages.analytics_core.src.intelligence.analytical_operations import aggregate_frame
from packages.analytics_core.src.intelligence.canonical_question_contract import (
    compile_canonical_question_contract,
)

ROOT = Path(__file__).parents[2]


def sample_frame():
    return pd.DataFrame({
        "revenue": [100.0, 200.0, 300.0, 400.0],
        "region": ["East", "West", "East", "South"],
        "channel": ["Web", "Store", "Web", "Store"],
        "status": ["active", "inactive", "active", "active"],
    })


def test_comma_before_exclusion_is_preserved_in_canonical_contract():
    frame = sample_frame()
    contract = compile_canonical_question_contract("show revenue by region, excluding East", frame)
    assert contract.completeness == "COMPLETE"
    assert not contract.unresolved_roles
    assert contract.grouping_columns == ("region",)
    assert { (f["column"], f["operator"], f["value"]) for f in contract.filter_conditions } == {
        ("region", "!=", "East")
    }

    result = aggregate_frame(
        frame, operation="SUM", target_column=contract.target_column,
        grouping_columns=contract.grouping_columns, filters=contract.filter_conditions,
    )
    assert result["population_row_count"] == 2
    assert { (r["group"]["region"], r["value"]) for r in result["records"] } == {
        ("West", 200.0), ("South", 400.0)
    }


def test_filter_then_split_uses_filter_dimension_as_predicate_not_grouping():
    frame = sample_frame()
    contract = compile_canonical_question_contract(
        "Calculate revenue by region West split by channel", frame
    )
    assert contract.completeness == "COMPLETE"
    assert contract.target_column == "revenue"
    assert contract.grouping_columns == ("channel",)
    assert { (f["column"], f["operator"], f["value"]) for f in contract.filter_conditions } == {
        ("region", "=", "West")
    }
    result = aggregate_frame(
        frame, operation="SUM", target_column=contract.target_column,
        grouping_columns=contract.grouping_columns, filters=contract.filter_conditions,
    )
    assert result["population_row_count"] == 1
    assert result["records"] == [{"group": {"channel": "Store"}, "value": 200.0, "row_count": 1}]


def test_excluding_two_status_values_becomes_one_not_in_predicate():
    frame = pd.DataFrame({
        "realized_revenue": [10.0, 20.0, 30.0, 40.0],
        "region": ["North", "South", "North", "South"],
        "order_status": ["Returned", "Cancelled", "Delivered", "Shipped"],
    })
    question = (
        "Calculate the total realized_revenue for each region, excluding records where "
        "order_status is Returned or Cancelled. Return the total for every region."
    )
    contract = compile_canonical_question_contract(question, frame)
    assert contract.completeness == "COMPLETE"
    assert contract.target_column == "realized_revenue"
    assert contract.requested_aggregation == "SUM"
    assert contract.grouping_columns == ("region",)
    assert contract.filter_conditions == ({
        "column": "order_status", "operator": "NOT IN", "value": ["Returned", "Cancelled"]
    },)

    preview = evaluate_guided_investigation(question, df=frame)
    resolved = apply_human_guidance_to_contract(question, {"sales": frame}, {})
    assert preview.can_auto_proceed is True
    assert preview.canonical_contract is not None
    assert preview.canonical_contract.filter_conditions == contract.filter_conditions
    assert resolved.status == ResolverStatus.RESOLVED
    assert resolved.contract is not None
    assert resolved.contract.filter_conditions == contract.filter_conditions
    from packages.analytics_core.src.governance.guided_investigation import render_contract_preview_from_contract
    planned_sql = render_contract_preview_from_contract(resolved.contract)
    assert planned_sql is not None
    assert "\"order_status\" NOT IN ('Returned', 'Cancelled')" in planned_sql

    result = aggregate_frame(
        frame, operation="SUM", target_column=contract.target_column,
        grouping_columns=contract.grouping_columns, filters=contract.filter_conditions,
    )
    assert result["population_row_count"] == 2
    assert {(item["group"]["region"], item["value"]) for item in result["records"]} == {
        ("North", 30.0), ("South", 40.0)
    }


def test_positive_same_column_or_values_become_in_list_not_raw_or():
    frame = pd.DataFrame({"revenue": [10, 20, 30], "status": ["Delivered", "Shipped", "Cancelled"]})
    contract = compile_canonical_question_contract(
        "Calculate total revenue where status is Delivered or Shipped", frame
    )
    assert contract.filter_conditions == ({
        "column": "status", "operator": "IN", "value": ["Delivered", "Shipped"]
    },)


def test_preview_and_creation_resolver_agree_for_filtered_descriptive_question():
    frame = sample_frame()
    question = "show revenue by region, excluding East"
    preview = evaluate_guided_investigation(question, df=frame)
    resolved = apply_human_guidance_to_contract(question, {"sales": frame}, {})

    assert preview.can_auto_proceed is True
    assert preview.canonical_contract is not None
    assert resolved.status == ResolverStatus.RESOLVED
    assert resolved.contract is not None
    assert resolved.contract.requested_aggregation == "SUM"
    assert resolved.contract.grouping_columns == ("region",)
    assert resolved.contract.filter_conditions == preview.canonical_contract.filter_conditions
    assert compute_resolved_plan_hash(resolved.contract, "same-dataset-fingerprint") == compute_resolved_plan_hash(
        preview.canonical_contract, "same-dataset-fingerprint"
    )


def test_user_supplied_scope_filter_is_added_without_dropping_question_filter():
    frame = sample_frame()
    question = "show revenue by region excluding East"
    outcome = apply_human_guidance_to_contract(
        question,
        {"sales": frame},
        {"scope_filter": "channel = 'Web'"},
    )
    assert outcome.status == ResolverStatus.RESOLVED
    assert outcome.contract is not None
    filters = {
        (f["column"], f["operator"], f["value"])
        for f in outcome.contract.filter_conditions
    }
    assert ("region", "!=", "East") in filters
    assert ("channel", "=", "Web") in filters

    result = aggregate_frame(
        frame, operation="SUM", target_column=outcome.contract.target_column,
        grouping_columns=outcome.contract.grouping_columns,
        filters=outcome.contract.filter_conditions,
    )
    assert result["population_row_count"] == 0
    # Zero matches are explicit; the user's filters must never be dropped to manufacture output.


def test_preanalysis_route_blocks_partial_dataset_context_before_planning():
    source = (ROOT / "apps/api/src/api/v1/investigations.py").read_text(encoding="utf-8")
    endpoint = source.index('def interpret_investigation_question(')
    body = source[endpoint:source.index('@router.post("/{investigation_id}/guidance")', endpoint)]
    guard = body.index("if context.load_errors or context.unavailable_requested_ids or context.scope_errors:")
    selection = body.index("select_primary_dataset(payload.question, context.datasets_map or {})")
    assert guard < selection
    assert "PREANALYSIS_DATASET_CONTEXT_INCOMPLETE" in body


def test_plan_panel_keeps_natural_language_filters_visible_in_preview():
    source = (ROOT / "apps/web/src/components/ProposedAnalysisPanel.tsx").read_text(encoding="utf-8")
    assert "const filters = questionFilters.map(sqlFilter)" in source
    assert "Question filter (preserved):" in source
    assert "Keep question-defined scope" in source
    assert 'scope_filter: selectedScope === "CUSTOM"' in source


def test_windows_exe_build_never_reuses_stale_next_static_export():
    source = (ROOT / "packaging/windows/build_exe_scope.ps1").read_text(encoding="utf-8")
    assert "Remove-Item -Recurse -Force $StaticOut" in source
    assert "npm ci --no-audit --ignore-scripts" in source
    assert "npm run build" in source
    build_section = source[source.index("$StaticOut = Join-Path"):source.index('if (-not (Test-Path $StaticOut -PathType Container)) { throw')]
    assert "if (-not (Test-Path $StaticOut -PathType Container))" not in build_section


def test_filter_parser_handles_between_and_escaped_apostrophes():
    from packages.analytics_core.src.governance.guided_investigation import parse_relational_filters

    parsed, error = parse_relational_filters(
        "revenue BETWEEN 100 AND 400 AND customer_name = 'O''Reilly'",
        ["revenue", "customer_name"],
    )
    assert error is None
    assert parsed == [
        {"column": "revenue", "operator": "BETWEEN", "value": [100, 400]},
        {"column": "customer_name", "operator": "=", "value": "O'Reilly"},
    ]


def test_between_is_rendered_and_applied_in_the_typed_filter_path():
    from packages.analytics_core.src.intelligence.intent_state import filter_condition_mask
    from packages.analytics_core.src.governance.guided_investigation import parse_relational_filters

    frame = sample_frame()
    parsed, error = parse_relational_filters("revenue BETWEEN 200 AND 400", list(frame.columns))
    assert error is None
    assert parsed == [{"column": "revenue", "operator": "BETWEEN", "value": [200, 400]}]
    mask = filter_condition_mask(frame["revenue"], parsed[0]["operator"], parsed[0]["value"])
    assert frame.loc[mask, "revenue"].tolist() == [200.0, 300.0, 400.0]

    resolved = apply_human_guidance_to_contract(
        "show revenue by region",
        {"sales": frame},
        {"scope_filter": "revenue BETWEEN 200 AND 400"},
    )
    assert resolved.status == ResolverStatus.RESOLVED
    assert resolved.contract is not None
    from packages.analytics_core.src.governance.guided_investigation import render_contract_preview_from_contract
    sql = render_contract_preview_from_contract(resolved.contract)
    assert sql is not None
    assert '"revenue" BETWEEN 200 AND 400' in sql


def test_plan_panel_uses_schema_bound_filter_builder_and_explains_sql_semantics():
    source = (ROOT / "apps/web/src/components/ProposedAnalysisPanel.tsx").read_text(encoding="utf-8")
    assert "compileEditableFilters" in source
    assert "BETWEEN" in source
    assert 'setFilterMode(filterMode === "builder" ? "sql" : "builder")' in source
    assert "Conditions are combined with AND" in source
    assert "backend validates the expression against the selected dataset" in source
    assert "scope_filter: selectedScope === \"CUSTOM\" ? activeCustomFilter" in source
    assert "lower BETWEEN bound must not exceed the upper bound" in source
    assert "is not available in the selected dataset" in source
    assert "region = 'West' AND status IN ('Delivered', 'Shipped')" in source
    assert "region = 'West' OR status IN ('Delivered', 'Shipped')" not in source
    assert "proposed calculation preview, not a claim that execution has succeeded" in source
    result_source = (ROOT / "apps/web/src/components/StructuredAnalysisAnswer.tsx").read_text(encoding="utf-8")
    assert "function timeRangeDisplay(value: unknown)" in result_source
    assert "end exclusive" in result_source


def test_result_output_shows_saved_execution_scope_near_the_answer():
    source = (ROOT / "apps/web/src/components/StructuredAnalysisAnswer.tsx").read_text(encoding="utf-8")
    assert 'aria-label="Analysis at a glance"' in source
    assert "execution.population_row_count" in source
    assert "execution.result_row_count" in source
    assert "filterDisplay(execution.filters)" in source
    assert "verification === \"VERIFIED\"" in source


def test_numeric_threshold_is_bound_to_explicit_numeric_column_or_refused():
    frame = pd.DataFrame({"revenue": [200.0, 600.0], "status": ["paid", "paid"]})
    contract = compile_canonical_question_contract("How many orders had revenue above 500?", frame)
    assert { (f["column"], f["operator"], f["value"]) for f in contract.filter_conditions } == {
        ("revenue", ">", 500.0)
    }
    # This question's entity-count semantics are not yet fully bound, so the contract
    # must remain incomplete rather than answering an unfiltered count.
    assert contract.completeness == "INCOMPLETE"


def test_exclusion_inflection_resolves_to_observed_category_value():
    frame = pd.DataFrame({"revenue": [10.0, 20.0, 30.0], "status": ["paid", "refunded", "paid"]})
    contract = compile_canonical_question_contract("Calculate total revenue excluding refunds", frame)
    assert contract.completeness == "COMPLETE"
    assert contract.filter_conditions == ({"column": "status", "operator": "!=", "value": "refunded"},)


def test_two_numeric_thresholds_must_both_be_bound():
    frame = pd.DataFrame({"revenue": [100.0, 600.0], "quantity": [1, 4]})
    contract = compile_canonical_question_contract(
        "How many orders had revenue above 500 and quantity below 3?", frame
    )
    predicates = {(f["column"], f["operator"], float(f["value"])) for f in contract.filter_conditions}
    assert ("revenue", ">", 500.0) in predicates
    assert ("quantity", "<", 3.0) in predicates
    assert len([x for x in contract.intent_ledger if x["kind"] == "numeric_threshold"]) == 2
    assert all(x["status"] == "CONSUMED" for x in contract.intent_ledger if x["kind"] == "numeric_threshold")


def test_ambiguous_exclusion_does_not_choose_a_column_by_score():
    frame = pd.DataFrame({
        "order_status": ["refunded", "paid"],
        "payment_status": ["refunded", "settled"],
        "revenue": [10.0, 20.0],
    })
    contract = compile_canonical_question_contract("Calculate total revenue excluding refunds", frame)
    assert contract.completeness == "INCOMPLETE"
    assert "exclusion_value_or_column_ambiguous" in contract.unresolved_roles
    assert any(x["kind"] == "categorical_exclusion" and x["status"] == "LEFTOVER" for x in contract.intent_ledger)


def test_exclusion_inflection_requires_unambiguous_observed_category():
    frame = pd.DataFrame({"revenue": [10.0, 20.0, 30.0], "status": ["paid", "refunded", "paid"]})
    contract = compile_canonical_question_contract("Calculate total revenue excluding refunds", frame)
    assert contract.filter_conditions == ({"column": "status", "operator": "!=", "value": "refunded"},)
    assert any(x["kind"] == "categorical_exclusion" and x["status"] == "CONSUMED" for x in contract.intent_ledger)


def test_each_numeric_threshold_ledger_entry_must_match_its_own_value_and_operator():
    frame = pd.DataFrame({"revenue": [100, 600], "quantity": [1, 5]})
    contract = compile_canonical_question_contract(
        "How many orders had revenue above 500 and quantity below 3?", frame
    )
    ledger = [item for item in contract.intent_ledger if item["kind"] == "numeric_threshold"]
    assert len(ledger) == 2
    assert all(item["status"] == "CONSUMED" for item in ledger)
    assert {
        (item["column"], item["operator"], float(item["value"]))
        for item in contract.filter_conditions
    } >= {("revenue", ">", 500.0), ("quantity", "<", 3.0)}



def test_investigation_graph_describes_priority_as_priority_not_probability():
    source = (ROOT / "apps/web/src/components/InvestigationGraphView.tsx").read_text(encoding="utf-8")
    assert "Relative priority meter" in source
    assert "Scores prioritize checks; they are not probabilities" in source
    assert "Bayesian Belief Distribution" not in source
    assert "setSelectedHypothesis" in source and "useEffect" in source


def test_three_condition_subset_is_preserved_and_applied_before_aggregation():
    from packages.analytics_core.src.governance.guided_investigation import parse_relational_filters

    frame = pd.DataFrame({
        "revenue": [100.0, 200.0, 300.0, 400.0],
        "region": ["West", "East", "West", "South"],
        "customer_segment": ["Small Business", "Enterprise", "Small Business", "Small Business"],
        "quarter": ["Q3", "Q3", "Q2", "Q3"],
        "delivery_completed_flag": [1, 1, 1, 0],
    })
    predicate = "customer_segment = 'Small Business' AND quarter = 'Q3' AND delivery_completed_flag = 1"
    parsed, error = parse_relational_filters(predicate, list(frame.columns))
    assert error is None
    assert parsed == [
        {"column": "customer_segment", "operator": "=", "value": "Small Business"},
        {"column": "quarter", "operator": "=", "value": "Q3"},
        {"column": "delivery_completed_flag", "operator": "=", "value": 1},
    ]

    resolved = apply_human_guidance_to_contract(
        "Calculate total revenue by region",
        {"orders": frame},
        {"scope_filter": predicate},
    )
    assert resolved.status == ResolverStatus.RESOLVED
    assert resolved.contract is not None
    assert list(resolved.contract.filter_conditions) == parsed
    result = aggregate_frame(
        frame,
        operation="SUM",
        target_column="revenue",
        grouping_columns=["region"],
        filters=resolved.contract.filter_conditions,
    )
    assert result["population_row_count"] == 1
    assert result["records"] == [{"group": {"region": "West"}, "value": 100.0, "row_count": 1}]

    # Runtime independently validates and applies the same predicate before analysis.
    # Assert that the execution path invokes this fail-closed validator; the optional
    # sqlglot dependency is not installed in every source-inspection test environment.
    runtime_source = (ROOT / "packages/analytics_core/src/runtime/controller.py").read_text(encoding="utf-8")
    assert "validate_scope_filter_expression(clean_filter, list(primary_df.columns))" in runtime_source
    assert "SELECT * FROM primary_df WHERE {validated_filter}" in runtime_source
