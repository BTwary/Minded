import pandas as pd

from packages.analytics_core.src.relational.semantic_planner import find_safe_n_table_plan


def test_finds_safe_four_table_path():
    tables = {
        "line_items": pd.DataFrame({"line_id": [1,2,3], "order_id": [10,10,11], "amount": [40,60,80]}),
        "orders": pd.DataFrame({"order_id": [10,11], "customer_id": [100,101]}),
        "customers": pd.DataFrame({"customer_id": [100,101], "account_id": [1000,1001]}),
        "accounts": pd.DataFrame({"account_id": [1000,1001], "segment": ["Enterprise","SMB"]}),
    }
    plan = find_safe_n_table_plan(tables, target_column="amount", grouping_column="segment", max_hops=3)
    assert plan is not None
    assert len(plan.join_hops) == 3
    assert plan.left_dataset == "line_items"
    assert plan.right_dataset == "accounts"
