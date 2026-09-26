import pandas as pd

from packages.analytics_core.src.relational.semantic_planner import (
    discover_join_candidates,
    find_safe_two_table_plan,
)
from packages.analytics_core.src.semantic.world_model import SemanticWorldModelBuilder


def test_discover_join_candidates_is_runtime_safe_and_deterministic():
    orders = pd.DataFrame({
        "order_id": [1, 2, 3, 4],
        "customer_id": [10, 20, 10, 30],
        "revenue": [10.0, 20.0, 15.0, 30.0],
    })
    customers = pd.DataFrame({
        "customer_id": [10, 20, 30],
        "segment": ["A", "B", "A"],
    })

    plans1 = discover_join_candidates({"orders": orders, "customers": customers})
    plans2 = discover_join_candidates({"customers": customers, "orders": orders})
    assert plans1
    assert plans1 == plans2
    assert any(p.left_key == "customer_id" and p.right_key == "customer_id" and p.status == "SAFE" for p in plans1)


def test_find_safe_two_table_plan_uses_actual_cardinality():
    orders = pd.DataFrame({"customer_id": [10, 20, 10], "revenue": [10., 20., 15.]})
    customers = pd.DataFrame({"customer_id": [10, 20], "segment": ["A", "B"]})
    plan = find_safe_two_table_plan(
        {"orders": orders, "customers": customers},
        target_column="revenue",
        grouping_column="segment",
    )
    assert plan is not None
    assert plan.status == "SAFE"
    assert "cardinality=many_to_one" in plan.rationale


def test_world_model_promotes_only_safe_observed_relationships():
    orders = pd.DataFrame({"order_id": [1, 2, 3, 4], "customer_id": [10, 20, 10, 30], "revenue": [1., 2., 3., 4.]})
    customers = pd.DataFrame({"customer_id": [10, 20, 30], "segment": ["A", "B", "A"]})
    unrelated = pd.DataFrame({"customer_id": [999, 998, 997], "label": ["x", "y", "z"]})
    model = SemanticWorldModelBuilder().build_world_model({
        "orders": orders,
        "customers": customers,
        "unrelated": unrelated,
    })
    assert any(
        r.source_table == "orders"
        and r.target_table == "customers"
        and r.source_column == "customer_id"
        and r.target_column == "customer_id"
        for r in model.relationships
    )
    assert all(r.join_safety_score is not None and 0.0 <= r.join_safety_score <= 1.0 for r in model.relationships)
