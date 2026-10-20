"""Regression coverage for the native-window startup hang and schema-safe ranking parse."""
from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

from packages.analytics_core.src.governance.guided_investigation import evaluate_guided_investigation
from packages.analytics_core.src.intelligence.canonical_question_contract import (
    compile_canonical_question_contract,
    resolve_phrase_to_column,
)

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / "packaging" / "windows" / "minded_entry.py"


def test_pywebview_bridge_does_not_hold_native_window_reference() -> None:
    """A window->js_api->window graph caused get_functions() to recurse indefinitely."""
    source = ENTRY.read_text(encoding="utf-8")
    tree = ast.parse(source)
    api_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DesktopExportAPI"
    )

    # The exposed instance must not assign any object reference (especially Window) to
    # itself. Keep the native Window only in a module global, outside pywebview's walk.
    assert not any(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr in {"window", "_window", "native_window", "_native_window"}
        for node in ast.walk(api_class)
    )
    assert "export_api._window = window" not in source
    assert "_desktop_window = window" in source
    assert "desktop_window = _desktop_window" in source
    assert "Local API imports completed in" in source


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "region": ["East", "West", "North", "South", "East", "West"],
        "realized_revenue": [100.25, 80.0, 40.0, 75.5, 25.75, 20.0],
        "order_status": ["Delivered", "Shipped", "Returned", "Cancelled", "Delivered", "Shipped"],
    })


def test_ranking_with_trailing_exclusion_is_still_compiled_as_one_question() -> None:
    question = "Which region generated the most revenue after excluding cancelled orders?"
    frame = _frame()

    contract = compile_canonical_question_contract(question, frame)
    assert contract.task_family == "RANKING"
    assert contract.target_column == "realized_revenue"
    assert contract.grouping_columns == ("region",)
    assert contract.filter_conditions == ({
        "column": "order_status", "operator": "!=", "value": "Cancelled",
    },)
    assert contract.completeness == "COMPLETE"

    preview = evaluate_guided_investigation(question, df=frame)
    assert preview.can_auto_proceed is True
    assert preview.clarification_prompt is None
    assert preview.canonical_contract is not None
    assert preview.canonical_contract.target_column == "realized_revenue"
    assert 'WHERE "order_status" != \'Cancelled\'' in preview.planned_calculation
    assert 'ORDER BY metric_value DESC' in preview.planned_calculation


def test_coordinated_exclusions_and_revenue_ranking_are_preserved() -> None:
    question = "Which region generated the most revenue after excluding returns and cancellations?"
    frame = _frame()
    contract = compile_canonical_question_contract(question, frame)

    assert contract.target_column == "realized_revenue"
    assert contract.grouping_columns == ("region",)
    assert contract.filter_conditions == ({
        "column": "order_status",
        "operator": "NOT IN",
        "value": ["Returned", "Cancelled"],
    },)
    preview = evaluate_guided_investigation(question, df=frame)
    assert preview.can_auto_proceed is True
    assert "NOT IN ('Returned', 'Cancelled')" in preview.planned_calculation


def test_generic_revenue_binds_only_when_the_schema_is_unambiguous() -> None:
    assert resolve_phrase_to_column("revenue", ["region", "realized_revenue"]) == "realized_revenue"
    assert resolve_phrase_to_column(
        "revenue", ["region", "net_revenue", "realized_revenue"]
    ) is None

    frame = pd.DataFrame({
        "region": ["East", "West", "East", "West"],
        "net_revenue": [10.0, 20.0, 30.0, 40.0],
        "realized_revenue": [11.0, 21.0, 31.0, 41.0],
        "order_status": ["Delivered", "Cancelled", "Delivered", "Returned"],
    })
    preview = evaluate_guided_investigation(
        "Which region generated the most revenue after excluding cancelled orders?", df=frame
    )
    assert preview.can_auto_proceed is False
    assert preview.ambiguity_status.value == "AMBIGUOUS_METRIC"
    assert preview.target_metric is None


def test_ranking_without_a_resolved_target_is_never_executable() -> None:
    frame = pd.DataFrame({"region": ["East", "West"], "order_status": ["Delivered", "Cancelled"]})
    question = "Which region generated the most revenue?"
    contract = compile_canonical_question_contract(question, frame)
    preview = evaluate_guided_investigation(question, df=frame)

    assert contract.target_column is None
    assert "target" in contract.unresolved_roles
    assert preview.can_auto_proceed is False
    assert preview.plan_hash is None


def test_plain_language_compare_by_dimension_compiles_to_an_aggregate() -> None:
    frame = _frame()
    question = "How does revenue compare by region?"
    contract = compile_canonical_question_contract(question, frame)

    assert contract.task_family == "DESCRIPTIVE"
    assert contract.target_column == "realized_revenue"
    assert contract.grouping_columns == ("region",)
    assert contract.requested_aggregation == "SUM"

    preview = evaluate_guided_investigation(question, df=frame)
    assert preview.can_auto_proceed is True
    assert preview.ambiguity_status.value == "CLEAR"
    assert 'GROUP BY "region"' in preview.planned_calculation
