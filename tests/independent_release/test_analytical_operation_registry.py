"""Acceptance tests for the typed aggregation registry and persisted operation graph."""
import pandas as pd
import pytest

from packages.analytics_core.src.intelligence.analytical_operations import (
    AnalyticalOperationError, OperationGraph, OperationNode, aggregate_frame,
    execute_operation_graph, validate_operation_graph,
)


def test_human_question_grouped_revenue_by_region_and_channel_includes_missing_groups():
    # Expected values are calculated by hand from the six source rows.
    frame = pd.DataFrame({
        "region": ["East", "East", "West", "West", None, "East"],
        "channel": ["web", "store", "web", "store", "web", "web"],
        "revenue": [100, 50, 70, 30, 20, None],
        "cancelled": [False, False, False, True, False, False],
    })
    result = aggregate_frame(
        frame, operation="SUM", target_column="revenue",
        grouping_columns=("region", "channel"),
        filters=({"column": "cancelled", "operator": "=", "value": False},),
    )
    actual = {(tuple(row["group"].values())): row["value"] for row in result["records"]}
    assert actual[("East", "web")] == 100.0
    assert actual[("East", "store")] == 50.0
    assert actual[("West", "web")] == 70.0
    assert actual[("<missing>", "web")] == 20.0
    assert result["population_row_count"] == 5
    assert result["verification"]["status"] == "verified"


def test_variance_standard_deviation_and_percentile_have_explicit_sample_semantics():
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    assert aggregate_frame(frame, operation="VARIANCE", target_column="x")["value"] == pytest.approx(5 / 3)
    assert aggregate_frame(frame, operation="STDDEV", target_column="x")["value"] == pytest.approx((5 / 3) ** 0.5)
    assert aggregate_frame(frame, operation="PERCENTILE", target_column="x", percentile=50)["value"] == pytest.approx(2.5)


def test_aggregation_fails_closed_on_unknown_columns_unsupported_operations_and_missing_percentile():
    frame = pd.DataFrame({"x": [1, 2], "g": ["a", "b"]})
    with pytest.raises(AnalyticalOperationError, match="Unknown grouping"):
        aggregate_frame(frame, operation="SUM", target_column="x", grouping_columns=("ghost",))
    with pytest.raises(AnalyticalOperationError, match="Unsupported aggregation"):
        aggregate_frame(frame, operation="MAGIC", target_column="x")
    with pytest.raises(AnalyticalOperationError, match="explicit percentile"):
        aggregate_frame(frame, operation="PERCENTILE", target_column="x")
    with pytest.raises(AnalyticalOperationError, match="numeric"):
        aggregate_frame(pd.DataFrame({"x": ["one", "two"]}), operation="MEAN", target_column="x")


def test_persisted_operation_graph_aggregates_then_ranks_with_stable_hash():
    frame = pd.DataFrame({"region": ["East", "West", "East", "North"], "revenue": [5, 8, 7, 3]})
    graph = OperationGraph(
        version=1,
        nodes=(
            OperationNode("revenue_by_region", "AGGREGATE", {
                "aggregation": "SUM", "target_column": "revenue", "grouping_columns": ["region"]
            }),
            OperationNode("rank_regions", "RANK", {"direction": "DESC"}, ("revenue_by_region",)),
        ),
        requested_outputs=("revenue_by_region", "rank_regions"),
    )
    assert graph.graph_hash() == graph.graph_hash()
    assert validate_operation_graph(graph, list(frame.columns)) == ("revenue_by_region", "rank_regions")
    result = execute_operation_graph(frame, graph)
    ranked = result["outputs"]["rank_regions"]["records"]
    assert [(x["group"]["region"], x["value"], x["rank"]) for x in ranked] == [
        ({"region": "East"}["region"], 12, 1), ("West", 8, 2), ("North", 3, 3)
    ]
    assert result["verification"]["graph_validated"] is True


def test_operation_graph_rejects_cycles_unknown_dependencies_and_invalid_rank_inputs():
    frame = pd.DataFrame({"region": ["East"], "revenue": [5]})
    cycle = OperationGraph(1, (
        OperationNode("a", "AGGREGATE", {"aggregation": "SUM", "target_column": "revenue"}, ("b",)),
        OperationNode("b", "RANK", {"direction": "DESC"}, ("a",)),
    ), ("b",))
    with pytest.raises(AnalyticalOperationError, match="cycle"):
        validate_operation_graph(cycle, list(frame.columns))
    unknown = OperationGraph(1, (
        OperationNode("a", "AGGREGATE", {"aggregation": "SUM", "target_column": "revenue"}, ("missing",)),
    ), ("a",))
    with pytest.raises(AnalyticalOperationError, match="missing dependency"):
        validate_operation_graph(unknown, list(frame.columns))
    invalid_rank = OperationGraph(1, (OperationNode("rank", "RANK", {}, ()),), ("rank",))
    with pytest.raises(AnalyticalOperationError, match="exactly one"):
        validate_operation_graph(invalid_rank, list(frame.columns))


def test_production_descriptive_contract_executor_uses_registry_for_multidimensional_grouping():
    from packages.analytics_core.src.engines.analyst_answer import _execute_canonical_descriptive_contract
    from packages.analytics_core.src.intelligence.canonical_question_contract import CanonicalQuestionContract

    frame = pd.DataFrame({
        "region": ["East", "East", "West", "West"],
        "channel": ["web", "store", "web", "store"],
        "revenue": [10.0, 20.0, 30.0, 40.0],
    })
    contract = CanonicalQuestionContract(
        question_text="Calculate total revenue by region and channel",
        task_family="DESCRIPTIVE", claim_type="OBSERVATION", estimand="sum",
        target_column="revenue", grouping_columns=("region", "channel"),
        requested_aggregation="SUM", family="SIMPLE_AGGREGATE", completeness="COMPLETE",
    )
    result = _execute_canonical_descriptive_contract(contract, frame)
    assert result is not None
    assert result.numbers["verification"]["status"] == "verified"
    assert result.numbers["population_row_count"] == 4
    assert result.numbers["table"]["region=East | channel=web"]["value"] == 10.0
    assert result.numbers["table"]["region=West | channel=store"]["value"] == 40.0


def test_production_descriptive_contract_executor_supports_variance_and_stddev():
    from packages.analytics_core.src.engines.analyst_answer import _execute_canonical_descriptive_contract
    from packages.analytics_core.src.intelligence.canonical_question_contract import CanonicalQuestionContract

    frame = pd.DataFrame({"revenue": [1.0, 2.0, 3.0, 4.0]})
    for operation, expected in (("VARIANCE", 5 / 3), ("STDDEV", (5 / 3) ** 0.5)):
        contract = CanonicalQuestionContract(
            question_text=f"Calculate {operation.lower()} revenue",
            task_family="DESCRIPTIVE", claim_type="OBSERVATION", estimand="descriptive",
            target_column="revenue", requested_aggregation=operation,
            family="SIMPLE_AGGREGATE", completeness="COMPLETE",
        )
        result = _execute_canonical_descriptive_contract(contract, frame)
        assert result is not None
        assert result.numbers["value"] == pytest.approx(expected)
        assert result.numbers["verification"]["status"] == "verified"


def test_real_natural_language_questions_compile_and_execute_exact_aggregations():
    from packages.analytics_core.src.engines.analyst_answer import build_analyst_result
    from packages.analytics_core.src.intelligence.canonical_question_contract import compile_canonical_question_contract

    frame = pd.DataFrame({
        "region": ["East", "East", "West", "West"],
        "channel": ["web", "store", "web", "store"],
        "revenue": [10.0, 20.0, 30.0, 40.0],
    })
    cases = [
        ("Calculate total revenue by region and channel", "SUM", ("region", "channel")),
        ("Calculate variance of revenue by region", "VARIANCE", ("region",)),
        ("Calculate standard deviation of revenue", "STDDEV", ()),
    ]
    for question, expected_operation, expected_groups in cases:
        contract = compile_canonical_question_contract(question, frame)
        assert contract.completeness == "COMPLETE", (question, contract.to_dict())
        assert contract.requested_aggregation == expected_operation
        assert contract.grouping_columns == expected_groups
        answer = build_analyst_result(
            question, frame, target=contract.target_column,
            group=(expected_groups[0] if expected_groups else None), approved_contract=contract,
        )
        assert answer is not None, question
        assert answer.numbers["operation"] == expected_operation
        assert answer.numbers["verification"]["status"] == "verified"
        if expected_operation == "SUM":
            assert answer.numbers["table"]["region=East | channel=web"]["value"] == 10.0
            assert answer.numbers["table"]["region=West | channel=store"]["value"] == 40.0
        elif expected_operation == "VARIANCE":
            assert answer.numbers["table"]["region=East"]["value"] == pytest.approx(50.0)
            assert answer.numbers["table"]["region=West"]["value"] == pytest.approx(50.0)
        else:
            assert answer.numbers["value"] == pytest.approx((500 / 3) ** 0.5)


def test_literal_missing_label_does_not_collide_with_actual_null_group():
    frame = pd.DataFrame({"region": ["<missing>", None], "revenue": [7, 11]})
    result = aggregate_frame(frame, operation="SUM", target_column="revenue", grouping_columns=("region",))
    values = [row["value"] for row in result["records"]]
    assert values == [7, 11]


def test_rank_keeps_undefined_aggregate_values_last_for_both_directions():
    frame = pd.DataFrame({"region": ["East", "West", "North"], "revenue": [10.0, 3.0, None]})
    graph = OperationGraph(1, (
        OperationNode("agg", "AGGREGATE", {"aggregation": "SUM", "target_column": "revenue", "grouping_columns": ["region"]}),
        OperationNode("rank", "RANK", {"direction": "DESC"}, ("agg",)),
    ), ("rank",))
    ranked = execute_operation_graph(frame, graph)["outputs"]["rank"]["records"]
    assert [r["group"]["region"] for r in ranked] == ["East", "West", "North"]


def test_ranking_language_distinguishes_entity_dimension_from_measure():
    from packages.analytics_core.src.engines.analyst_answer import build_analyst_result
    from packages.analytics_core.src.intelligence.canonical_question_contract import compile_canonical_question_contract

    frame = pd.DataFrame({"region": ["East", "West", "East"], "revenue": [10.0, 20.0, 30.0]})
    for question in ("Rank regions by total revenue", "Rank revenue by region"):
        contract = compile_canonical_question_contract(question, frame)
        assert contract.task_family == "RANKING"
        assert contract.target_column == "revenue"
        assert contract.grouping_columns == ("region",)
        assert contract.requested_aggregation == "SUM"
        answer = build_analyst_result(
            question, frame, target=contract.target_column, group="region", approved_contract=contract
        )
        assert answer is not None
        assert answer.numbers["table"]["East"]["value"] == 40.0
        assert answer.numbers["table"]["West"]["value"] == 20.0


def test_guided_preview_preserves_all_group_dimensions_and_new_aggregation_semantics():
    from packages.analytics_core.src.governance.guided_investigation import evaluate_guided_investigation

    frame = pd.DataFrame({"region": ["East", "West"], "channel": ["web", "store"], "revenue": [10.0, 20.0]})
    multi = evaluate_guided_investigation("Calculate total revenue by region and channel", df=frame)
    assert multi.can_auto_proceed is True
    assert multi.canonical_contract.grouping_columns == ("region", "channel")
    assert 'GROUP BY "region", "channel"' in multi.planned_calculation
    assert multi.group_dimension == "region, channel"

    variance = evaluate_guided_investigation("Calculate variance of revenue by region", df=frame)
    assert variance.canonical_contract.requested_aggregation == "VARIANCE"
    assert "VAR_SAMP(\"revenue\")" in variance.planned_calculation


def test_operation_graph_is_canonical_contract_data_and_executes_only_when_approved(monkeypatch):
    from packages.analytics_core.src.engines.analyst_answer import (
        _execute_canonical_descriptive_contract, build_analyst_result,
    )
    from packages.analytics_core.src.intelligence import canonical_question_contract as contract_module
    from packages.analytics_core.src.intelligence.canonical_question_contract import (
        CanonicalQuestionContract, canonical_question_contract_from_dict,
    )

    frame = pd.DataFrame({"region": ["East", "West", "East"], "revenue": [5.0, 8.0, 7.0]})
    graph = OperationGraph(1, (
        OperationNode("sum_by_region", "AGGREGATE", {
            "aggregation": "SUM", "target_column": "revenue", "grouping_columns": ["region"]
        }),
        OperationNode("rank_regions", "RANK", {"direction": "DESC"}, ("sum_by_region",)),
    ), ("rank_regions",))
    contract = CanonicalQuestionContract(
        question_text="Aggregate revenue by region and rank regions",
        task_family="DESCRIPTIVE", claim_type="OBSERVATION", estimand="sum",
        target_column="revenue", grouping_columns=("region",), requested_aggregation="SUM",
        family="SIMPLE_AGGREGATE", completeness="COMPLETE", operation_graph=graph.to_dict(),
    )
    restored = canonical_question_contract_from_dict(contract.to_dict())
    assert restored.contract_hash() == contract.contract_hash()
    assert restored.operation_graph == graph.to_dict()

    # If a graph-bearing contract appears only as a fresh parse, the public answer path refuses it.
    monkeypatch.setattr(contract_module, "compile_canonical_question_contract", lambda question, df: restored)
    assert build_analyst_result(contract.question_text, frame, target="revenue", group="region") is None

    # The approved-contract executor can run the persisted graph and returns ranked output.
    approved = _execute_canonical_descriptive_contract(restored, frame)
    assert approved is not None
    graph_output = approved.numbers["operation_graph"]["outputs"]["rank_regions"]["records"]
    assert [(row["group"]["region"], row["value"], row["rank"]) for row in graph_output] == [
        ("East", 12.0, 1), ("West", 8.0, 2)
    ]


def test_supported_compound_human_question_is_approved_as_graph_and_executes_both_steps():
    from packages.analytics_core.src.engines.analyst_answer import build_analyst_result
    from packages.analytics_core.src.governance.guided_investigation import evaluate_guided_investigation

    frame = pd.DataFrame({"region": ["East", "West", "East"], "revenue": [10.0, 20.0, 30.0]})
    question = "Calculate total revenue by region and rank regions by revenue"
    preview = evaluate_guided_investigation(question, df=frame)
    assert preview.can_auto_proceed is True
    contract = preview.canonical_contract
    assert contract.completeness == "COMPLETE"
    assert contract.operation_graph is not None
    assert "OPERATION GRAPH" in preview.planned_calculation
    assert contract.contract_hash()

    result = build_analyst_result(
        question, frame, target=contract.target_column, group="region", approved_contract=contract
    )
    assert result is not None
    graph_result = result.numbers["operation_graph"]
    aggregate = graph_result["outputs"]["aggregate"]["records"]
    ranked = graph_result["outputs"]["rank"]["records"]
    assert {(r["group"]["region"], r["value"]) for r in aggregate} == {("East", 40.0), ("West", 20.0)}
    assert [(r["group"]["region"], r["value"], r["rank"]) for r in ranked] == [
        ("East", 40.0, 1), ("West", 20.0, 2)
    ]


def test_different_metric_compound_question_still_refuses_partial_execution():
    from packages.analytics_core.src.governance.guided_investigation import evaluate_guided_investigation

    frame = pd.DataFrame({"region": ["East", "West"], "revenue": [10.0, 20.0], "profit": [2.0, 4.0]})
    preview = evaluate_guided_investigation(
        "Rank revenue by region and compare profit by region", df=frame
    )
    assert preview.can_auto_proceed is False
    assert preview.canonical_contract.operation_graph is None
    assert preview.planned_calculation.startswith("Not executable")


def test_zero_matching_rows_reports_unverified_not_verified():
    from packages.analytics_core.src.intelligence.analytical_operations import aggregate_frame, OperationGraph, OperationNode, execute_operation_graph

    frame = pd.DataFrame({"region": ["East", "West"], "revenue": [10.0, 20.0]})
    # Filter that matches zero rows
    filters = [{"column": "region", "operator": "=", "value": "Nonexistent"}]

    grouped = aggregate_frame(frame, operation="SUM", target_column="revenue", grouping_columns=("region",), filters=filters)
    assert grouped["population_row_count"] == 0
    assert grouped["records"] == []
    assert grouped["verification"]["status"] == "unverified_zero_matches"
    assert grouped["verification"]["matches"] is False
    assert grouped["verification"]["zero_matching_rows"] is True

    ungrouped = aggregate_frame(frame, operation="SUM", target_column="revenue", filters=filters)
    assert ungrouped["population_row_count"] == 0
    assert ungrouped["value"] is None
    assert ungrouped["verification"]["status"] == "unverified_zero_matches"
    assert ungrouped["verification"]["matches"] is False
    assert ungrouped["verification"]["zero_matching_rows"] is True

    # Graph with zero matching rows
    graph = OperationGraph(1, (
        OperationNode("agg", "AGGREGATE", {"aggregation": "SUM", "target_column": "revenue", "grouping_columns": ["region"], "filters": filters}),
        OperationNode("rank", "RANK", {"direction": "DESC"}, ("agg",)),
    ), requested_outputs=("agg", "rank"))
    graph_res = execute_operation_graph(frame, graph)
    assert graph_res["verification"]["graph_validated"] is True
    assert graph_res["verification"]["analytical_verified"] is False
    assert graph_res["verification"]["status"] == "unverified_zero_matches"
    assert graph_res["verification"]["zero_matching_rows"] is True

