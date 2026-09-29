"""Production-path invariants for analyst-selected multi-dataset scope."""
from __future__ import annotations

from unittest.mock import Mock

import pandas as pd
import pytest

from packages.analytics_core.src.engines.dataset_provider import InMemoryDatasetProvider
from packages.analytics_core.src.engines.intent import IntentEngine
from packages.analytics_core.src.engines.semantic import SemanticEngine


def test_provider_loads_multiple_selected_datasets_without_scope_expansion():
    provider = InMemoryDatasetProvider(
        {
            "customers": pd.DataFrame({"customer_id": [1, 2], "region": ["A", "B"]}),
            "orders": pd.DataFrame({"customer_id": [1, 1, 2], "revenue": [10, 20, 30]}),
            "costs": pd.DataFrame({"customer_id": [1, 2], "cost": [3, 7]}),
        }
    )
    ctx = provider.acquire_context("project-1", dataset_ids=["customers", "orders"])
    assert sorted(ctx.datasets_map) == ["customers", "orders"]
    assert ctx.unavailable_requested_ids == []


def test_provider_reports_missing_requested_dataset_instead_of_expanding_scope():
    provider = InMemoryDatasetProvider(
        {
            "customers": pd.DataFrame({"customer_id": [1, 2]}),
            "orders": pd.DataFrame({"customer_id": [1, 2], "revenue": [10, 20]}),
        }
    )
    ctx = provider.acquire_context("project-1", dataset_ids=["customers", "finance"])
    assert sorted(ctx.datasets_map) == ["customers"]
    assert ctx.unavailable_requested_ids == ["finance"]


def test_provider_can_switch_scope_between_questions():
    provider = InMemoryDatasetProvider(
        {
            "sales": pd.DataFrame({"id": [1], "revenue": [100]}),
            "support": pd.DataFrame({"id": [1], "tickets": [4]}),
        }
    )
    first = provider.acquire_context("project-1", dataset_ids=["sales"])
    second = provider.acquire_context("project-1", dataset_ids=["support"])
    both = provider.acquire_context("project-1", dataset_ids=["sales", "support"])
    assert sorted(first.datasets_map) == ["sales"]
    assert sorted(second.datasets_map) == ["support"]
    assert sorted(both.datasets_map) == ["sales", "support"]


def test_semantic_engine_can_bind_a_multi_dataset_question_to_a_relational_scope():
    customers = pd.DataFrame({"customer_id": [1, 2, 3], "segment": ["A", "B", "A"]})
    orders = pd.DataFrame({"order_id": [10, 11, 12], "customer_id": [1, 2, 3], "revenue": [100, 200, 300]})
    question = "Which customer segment generated the highest revenue?"
    intent = IntentEngine.parse_intent(question, available_columns=list(customers.columns) + list(orders.columns))
    semantic = SemanticEngine.resolve_schema_static(intent, {"customers": customers, "orders": orders})
    assert semantic.primary_dataset_name == "orders"
    assert semantic.relational_access is not None
    assert {semantic.relational_access.base_table, semantic.relational_access.joined_table} == {"orders", "customers"}
    assert semantic.relational_access.join_hops
    assert semantic.target_metric_col == "revenue"


def test_semantic_engine_proves_a_three_table_relational_path():
    customers = pd.DataFrame({"customer_id": [1, 2, 3], "segment": ["A", "B", "A"]})
    orders = pd.DataFrame({"order_id": [10, 11, 12], "customer_id": [1, 2, 3], "product_id": [100, 101, 100], "revenue": [100, 200, 300]})
    products = pd.DataFrame({"product_id": [100, 101], "category": ["Premium", "Basic"]})
    question = "Which customer segment generated the highest revenue from the Premium product category?"
    intent = IntentEngine.parse_intent(question, available_columns=list(customers.columns) + list(orders.columns) + list(products.columns))
    semantic = SemanticEngine.resolve_schema_static(intent, {"customers": customers, "orders": orders, "products": products})
    assert semantic.relational_access is not None
    assert len(semantic.relational_access.join_hops) == 2
    assert set(semantic.relational_access.joined_tables) == {"customers", "products"}
    assert semantic.relational_access.base_table == "orders"


def test_semantic_engine_records_explicit_multi_dataset_references_without_guessing_a_join():
    left = pd.DataFrame({"customer_id": [1, 2], "revenue": [10, 20]})
    right = pd.DataFrame({"ticket_id": [8, 9], "support_cost": [3, 4]})
    question = "Compare revenue and support_cost"
    intent = IntentEngine.parse_intent(question, available_columns=list(left.columns) + list(right.columns))
    semantic = SemanticEngine.resolve_schema_static(intent, {"sales": left, "support": right})
    assert semantic.relational_access is None
    assert semantic.question_referenced_datasets == ["sales", "support"]
