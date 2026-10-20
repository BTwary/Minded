from __future__ import annotations

import pandas as pd

from packages.analytics_core.src.engines.dataset_provider import InMemoryDatasetProvider
from packages.analytics_core.src.engines.analyst_answer import build_analyst_result
from packages.analytics_core.src.governance.guided_investigation import (
    ResolverStatus,
    apply_human_guidance_to_contract,
)


def test_grouped_row_count_is_not_a_numeric_aggregation() -> None:
    frame = pd.DataFrame(
        {"plan_tier": ["Basic", "Pro", None], "revenue": [None, 20.0, 30.0]}
    )
    context = InMemoryDatasetProvider({"subscriptions": frame}).acquire_context("project")

    outcome = apply_human_guidance_to_contract(
        "How many records by plan tier?", context, {}
    )

    assert outcome.status is ResolverStatus.RESOLVED
    assert outcome.contract.requested_aggregation == "ROW_COUNT"
    assert outcome.contract.target_column is None
    assert outcome.contract.grouping_columns == ("plan_tier",)


def test_structured_categorical_count_is_resolved() -> None:
    frame = pd.DataFrame({"plan_tier": ["Basic", "Pro", "Basic"]})
    context = InMemoryDatasetProvider({"subscriptions": frame}).acquire_context("project")

    outcome = apply_human_guidance_to_contract(
        "Count subscriptions by plan tier",
        context,
        {
            "selected_metric": "plan_tier",
            "selected_aggregation": "COUNT",
            "selected_group_dimension": "plan_tier",
        },
    )

    assert outcome.status is ResolverStatus.RESOLVED
    assert outcome.contract.requested_aggregation == "ROW_COUNT"
    assert outcome.contract.target_column is None
    assert outcome.contract.grouping_columns == ("plan_tier",)


def test_freeform_guidance_resolves_aggregate_filter_rank_and_share() -> None:
    frame = pd.DataFrame(
        {
            "region": ["North", "South", "North"],
            "net_revenue": [100.0, 50.0, 25.0],
            "status": ["active", "cancelled", "active"],
        }
    )
    context = InMemoryDatasetProvider({"subscriptions": frame}).acquire_context("project")

    outcome = apply_human_guidance_to_contract(
        "How are subscriptions performing?",
        context,
        {
            "custom_instruction": (
                "Calculate net revenue by region, exclude cancelled subscriptions, "
                "rank highest first, and show each region's share of revenue."
            )
        },
    )

    assert outcome.status is ResolverStatus.RESOLVED
    assert outcome.contract.target_column == "net_revenue"
    assert outcome.contract.grouping_columns == ("region",)
    assert outcome.contract.requested_aggregation == "SUM"
    assert outcome.contract.ranking_direction == "DESC"
    assert outcome.contract.filter_conditions == (
        {"column": "status", "operator": "!=", "value": "cancelled"},
    )
    assert outcome.contract.share_of_total is True


def test_structured_and_equivalent_freeform_guidance_converge() -> None:
    frame = pd.DataFrame(
        {"region": ["North", "South"], "net_revenue": [100.0, 50.0]}
    )
    context = InMemoryDatasetProvider({"orders": frame}).acquire_context("project")
    structured = apply_human_guidance_to_contract(
        "How are orders performing?",
        context,
        {
            "selected_metric": "net_revenue",
            "selected_aggregation": "SUM",
            "selected_group_dimension": "region",
        },
    )
    freeform = apply_human_guidance_to_contract(
        "How are orders performing?",
        context,
        {"custom_instruction": "Calculate total net revenue by region"},
    )

    assert structured.status is ResolverStatus.RESOLVED
    assert freeform.status is ResolverStatus.RESOLVED
    assert (
        structured.contract.target_column,
        structured.contract.requested_aggregation,
        structured.contract.grouping_columns,
    ) == (
        freeform.contract.target_column,
        freeform.contract.requested_aggregation,
        freeform.contract.grouping_columns,
    )


def test_conflicting_structured_and_freeform_aggregation_is_explicit() -> None:
    frame = pd.DataFrame({"revenue": [10.0, 20.0]})
    context = InMemoryDatasetProvider({"orders": frame}).acquire_context("project")

    outcome = apply_human_guidance_to_contract(
        "What is the total revenue?",
        context,
        {
            "selected_aggregation": "SUM",
            "custom_instruction": "Calculate average revenue",
        },
    )

    assert outcome.status is ResolverStatus.CONFLICT
    assert outcome.conflict is not None
    assert outcome.conflict.field == "aggregation"
    assert outcome.conflict.structured_value == "SUM"
    assert outcome.conflict.freeform_value == "MEAN"


def test_unknown_freeform_instruction_is_not_reported_as_resolved() -> None:
    frame = pd.DataFrame({"revenue": [10.0, 20.0]})
    context = InMemoryDatasetProvider({"orders": frame}).acquire_context("project")

    outcome = apply_human_guidance_to_contract(
        "What is the total revenue?",
        context,
        {"custom_instruction": "Apply a proprietary scoring function to prioritize accounts"},
    )

    assert outcome.status is ResolverStatus.UNSUPPORTED
    assert outcome.original_instruction == "Apply a proprietary scoring function to prioritize accounts"


def test_multi_dataset_scope_does_not_silently_select_first_dataset() -> None:
    context = InMemoryDatasetProvider(
        {
            "orders": pd.DataFrame({"revenue": [10.0]}),
            "subscriptions": pd.DataFrame({"revenue": [20.0]}),
        }
    ).acquire_context("project")

    outcome = apply_human_guidance_to_contract(
        "What is the total revenue?", context, {}
    )

    assert outcome.status is ResolverStatus.NEEDS_CLARIFICATION
    assert set(outcome.suggested_resolutions) == {"orders", "subscriptions"}


def test_explicit_correction_replaces_prior_aggregation() -> None:
    frame = pd.DataFrame({"revenue": [10.0, 20.0], "region": ["N", "S"]})
    context = InMemoryDatasetProvider({"orders": frame}).acquire_context("project")
    initial = apply_human_guidance_to_contract(
        "What is the total revenue by region?",
        context,
        {"selected_aggregation": "SUM"},
    )

    corrected = apply_human_guidance_to_contract(
        "What is the total revenue by region?",
        context,
        {"selected_aggregation": "MEAN"},
        previous_contract=initial.contract,
    )

    assert initial.status is ResolverStatus.RESOLVED
    assert corrected.status is ResolverStatus.RESOLVED
    assert corrected.contract.requested_aggregation == "MEAN"


def test_grouped_share_contract_is_executed_and_independently_verified() -> None:
    frame = pd.DataFrame({"region": ["North", "South", "North"], "revenue": [60.0, 30.0, 10.0]})
    context = InMemoryDatasetProvider({"orders": frame}).acquire_context("project")
    outcome = apply_human_guidance_to_contract(
        "How much revenue by region?",
        context,
        {"custom_instruction": "Calculate total revenue by region and show each share of total"},
    )

    assert outcome.status is ResolverStatus.RESOLVED
    result = build_analyst_result(
        outcome.contract.question_text,
        frame,
        target=outcome.contract.target_column,
        group=outcome.contract.grouping_columns[0],
        approved_contract=outcome.contract,
    )

    assert result is not None
    assert result.numbers["table"]["North"]["value"] == 70.0
    assert result.numbers["table"]["South"]["value"] == 30.0
    assert result.numbers["table"]["North"]["share_of_total"] == 0.7
    assert result.numbers["table"]["South"]["share_of_total"] == 0.3
    assert result.numbers["verification"]["shares_match"] is True



def test_scope_filter_accepts_is_alias_for_custom_dataset_columns() -> None:
    frame = pd.DataFrame(
        {
            "region": ["East", "West", "East"],
            "realized_revenue": [10.0, 20.0, 30.0],
            "order_status": ["Delivered", "Returned", "Delivered"],
        }
    )

    outcome = apply_human_guidance_to_contract(
        "Calculate total realized_revenue by region",
        frame,
        {
            "selected_metric": "realized_revenue",
            "selected_aggregation": "SUM",
            "selected_group_dimension": "region",
            "scope_filter": "order_status is Delivered",
        },
    )

    assert outcome.status is ResolverStatus.RESOLVED
    assert outcome.contract.target_column == "realized_revenue"
    assert outcome.contract.grouping_columns == ("region",)
    assert {"column": "order_status", "operator": "=", "value": "Delivered"} in list(
        outcome.contract.filter_conditions
    )


def test_scope_filter_accepts_multiple_literal_exclusions_joined_by_and() -> None:
    frame = pd.DataFrame(
        {
            "region": ["East", "West", "East", "South"],
            "realized_revenue": [10.0, 20.0, 30.0, 40.0],
            "order_status": ["Delivered", "Returned", "Cancelled", "Delivered"],
        }
    )

    outcome = apply_human_guidance_to_contract(
        "Calculate total realized_revenue by region",
        frame,
        {
            "selected_metric": "realized_revenue",
            "selected_aggregation": "SUM",
            "selected_group_dimension": "region",
            "scope_filter": "order_status is not 'Returned' AND order_status != 'Cancelled'",
        },
    )

    assert outcome.status is ResolverStatus.RESOLVED
    assert {"column": "order_status", "operator": "!=", "value": "Returned"} in list(
        outcome.contract.filter_conditions
    )
    assert {"column": "order_status", "operator": "!=", "value": "Cancelled"} in list(
        outcome.contract.filter_conditions
    )


def test_scope_filter_rejects_or_and_sql_expressions() -> None:
    frame = pd.DataFrame(
        {
            "region": ["East", "West"],
            "realized_revenue": [10.0, 20.0],
            "order_status": ["Delivered", "Returned"],
        }
    )

    for unsafe_filter in (
        "order_status = 'Delivered' OR order_status = 'Returned'",
        "order_status = 'Delivered'; DROP TABLE data",
    ):
        outcome = apply_human_guidance_to_contract(
            "Calculate total realized_revenue by region",
            frame,
            {
                "selected_metric": "realized_revenue",
                "selected_aggregation": "SUM",
                "selected_group_dimension": "region",
                "scope_filter": unsafe_filter,
            },
        )
        assert outcome.status is ResolverStatus.UNSUPPORTED
