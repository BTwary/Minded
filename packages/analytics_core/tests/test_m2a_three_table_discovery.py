import pandas as pd

from packages.analytics_core.src.relational.compiler import JoinHop, RelationalCompiler, RelationalPlan
from packages.analytics_core.src.relational.expressions import AggExpr, ColumnRef
from packages.analytics_core.src.relational.join_safety import assess_join_safety, assess_metric_aggregation_safety
from packages.analytics_core.src.relational.semantic_planner import find_safe_three_table_plan


def test_discovers_unambiguous_grain_safe_three_table_path():
    tables = {
        "order_items": pd.DataFrame({"item_id": ["I1", "I2", "I3"], "order_id": ["O1", "O1", "O2"], "line_amount": [40, 60, 80]}),
        "orders": pd.DataFrame({"order_id": ["O1", "O2"], "customer_id": ["C1", "C2"]}),
        "customers": pd.DataFrame({"customer_id": ["C1", "C2"], "segment": ["Enterprise", "SMB"]}),
    }
    plan = find_safe_three_table_plan(tables, target_column="line_amount", grouping_column="segment")
    assert plan is not None
    assert plan.left_dataset == "order_items"
    assert [(hop["left_table"], hop["right_table"]) for hop in plan.join_hops] == [("order_items", "orders"), ("orders", "customers")]


def test_rejects_equal_ranked_three_table_paths_as_ambiguous():
    tables = {
        "line_items": pd.DataFrame({"item_id": ["I1", "I2"], "order_id": ["O1", "O2"], "line_amount": [10, 20]}),
        "orders": pd.DataFrame({"order_id": ["O1", "O2"], "customer_a_id": ["A1", "A2"], "customer_b_id": ["B1", "B2"]}),
        "customers_a": pd.DataFrame({"customer_a_id": ["A1", "A2"], "segment": ["Enterprise", "SMB"]}),
        "customers_b": pd.DataFrame({"customer_b_id": ["B1", "B2"], "segment": ["Enterprise", "SMB"]}),
    }
    assert find_safe_three_table_plan(tables, target_column="line_amount", grouping_column="segment") is None


def test_detects_fanout_risk_for_raw_metric_aggregation():
    orders = pd.DataFrame({"order_id": ["O1", "O2"], "order_amount": [100, 80]})
    items = pd.DataFrame({"item_id": ["I1", "I2", "I3"], "order_id": ["O1", "O1", "O2"], "product_id": ["P1", "P2", "P1"]})
    report = assess_join_safety(orders, items, "order_id", "order_id", "orders", "order_items")
    aggregate_safety = assess_metric_aggregation_safety("orders", [report])
    assert report.fanout_risk is True
    assert aggregate_safety.safe is False


def test_rejects_raw_order_grain_sum_after_three_table_fanout():
    tables = {
        "orders": pd.DataFrame({"order_id": ["O1", "O2"], "customer_segment": ["Enterprise", "SMB"], "order_amount": [100, 80]}),
        "order_items": pd.DataFrame({"item_id": ["I1", "I2", "I3"], "order_id": ["O1", "O1", "O2"], "product_id": ["P1", "P2", "P1"]}),
        "products": pd.DataFrame({"product_id": ["P1", "P2"], "category": ["Software", "Services"]}),
    }
    plan = RelationalPlan(
        base_table="orders",
        hops=[JoinHop("orders", "order_items", "order_id", "order_id"), JoinHop("order_items", "products", "product_id", "product_id")],
        select_columns=["products.category AS category"],
        aggregations={"total_order_amount": AggExpr("SUM", ColumnRef("order_amount", "orders"))},
        group_by=["products.category"],
    )
    result = RelationalCompiler(tables).compile_and_execute(plan)
    assert result.safe is False
    assert "can multiply source rows" in result.blocked_reason
