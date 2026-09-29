import ast
from pathlib import Path

import pandas as pd

from packages.analytics_core.src.relational.join_safety import (
    Cardinality,
    SafetyStatus,
    assess_declared_join_hops,
)


ROOT = Path(__file__).resolve().parents[1]


def test_declared_join_hops_safe_many_to_one():
    datasets = {
        "orders": pd.DataFrame({"customer_id": ["C1", "C1", "C2"]}),
        "customers": pd.DataFrame({"customer_id": ["C1", "C2"]}),
    }
    reports = assess_declared_join_hops(datasets, [{
        "left_table": "orders",
        "right_table": "customers",
        "left_key": "customer_id",
        "right_key": "customer_id",
    }])
    assert len(reports) == 1
    assert reports[0].status == SafetyStatus.SAFE
    assert reports[0].cardinality == Cardinality.MANY_TO_ONE


def test_declared_join_hops_unknown_is_not_safe():
    datasets = {
        "orders": pd.DataFrame({"customer_id": pd.Series(dtype="object")}),
        "customers": pd.DataFrame({"customer_id": ["C1", "C2"]}),
    }
    reports = assess_declared_join_hops(datasets, [{
        "left_table": "orders",
        "right_table": "customers",
        "left_key": "customer_id",
        "right_key": "customer_id",
    }])
    assert reports[0].status == SafetyStatus.UNKNOWN


def test_declared_join_hops_missing_metadata_fails_closed():
    datasets = {
        "orders": pd.DataFrame({"customer_id": ["C1"]}),
        "customers": pd.DataFrame({"customer_id": ["C1"]}),
    }
    reports = assess_declared_join_hops(datasets, [{"left_table": "orders"}])
    assert reports[0].status == SafetyStatus.UNSAFE
    assert reports[0].fanout_risk is True


def test_controller_contains_fail_closed_join_gate():
    source = (ROOT / "packages/analytics_core/src/runtime/controller.py").read_text()
    tree = ast.parse(source)
    constants = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "JOIN_SAFETY_METADATA_MISSING" in constants
    assert "JOIN_SAFETY_GATE_BLOCKED" in constants
    assert "assess_declared_join_hops" in source
    assert "r.status != SafetyStatus.SAFE" in source
