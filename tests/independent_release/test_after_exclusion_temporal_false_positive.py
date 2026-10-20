"""Regression tests for 'After excluding returns and cancellations' being parsed as time scope."""
from __future__ import annotations

import pandas as pd

from packages.analytics_core.src.governance.guided_investigation import evaluate_guided_investigation
from packages.analytics_core.src.intelligence.canonical_question_contract import compile_canonical_question_contract


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "region": ["East", "West", "North", "South", "East", "West"],
        "realized_revenue": [100.25, 80.0, 40.0, 75.5, 25.75, 20.0],
        "order_status": ["Delivered", "Shipped", "Returned", "Cancelled", "Delivered", "Shipped"],
    })


def test_after_excluding_is_not_a_temporal_scope_and_both_statuses_are_excluded():
    question = "After excluding returns and cancellations, which region generated the most realized revenue?"
    df = _frame()

    contract = compile_canonical_question_contract(question, df)
    assert contract.time_range is None
    assert "time_range" not in contract.unresolved_roles
    assert contract.filter_conditions == ({
        "column": "order_status",
        "operator": "NOT IN",
        "value": ["Returned", "Cancelled"],
    },)

    preview = evaluate_guided_investigation(question, df=df)
    assert preview.can_auto_proceed is True
    assert preview.clarification_prompt is None
    assert preview.canonical_contract is not None
    assert preview.canonical_contract.filter_conditions == contract.filter_conditions
    assert "NOT IN ('Returned', 'Cancelled')" in preview.planned_calculation
    assert "Not executable yet" not in preview.planned_calculation


def test_coordinated_exclusion_keeps_other_valid_lifecycle_states():
    question = "After excluding returns and cancellations, which region generated the most realized revenue?"
    df = _frame()
    preview = evaluate_guided_investigation(question, df=df)
    assert preview.can_auto_proceed is True

    eligible = df.loc[~df["order_status"].isin(["Returned", "Cancelled"])]
    # Shipped and Delivered remain eligible. An exclusion is not equivalent to
    # forcing order_status = Delivered.
    assert set(eligible["order_status"]) == {"Delivered", "Shipped"}
    totals = eligible.groupby("region")["realized_revenue"].sum().sort_values(ascending=False)
    assert totals.iloc[0] == 126.0
    assert totals.index[0] == "East"

