"""SQL WHERE expression guard tests; run in the locked environment with sqlglot installed."""
import pytest

pytest.importorskip("sqlglot")

from packages.analytics_core.src.runtime.controller_scope import validate_scope_filter_expression


def test_accepts_supported_and_between_in_and_null_predicates():
    columns = ["order_status", "realized_revenue", "region", "active_flag", "returned_at"]
    normalized = validate_scope_filter_expression(
        "order_status != 'Cancelled' AND realized_revenue BETWEEN 100 AND 500",
        columns,
    )
    assert isinstance(normalized, str) and normalized
    assert "BETWEEN" in normalized.upper() and "AND" in normalized.upper()
    validate_scope_filter_expression("region IN ('East', 'West')", columns)
    validate_scope_filter_expression("region NOT IN ('East', 'West')", columns)
    validate_scope_filter_expression("active_flag = TRUE", columns)
    validate_scope_filter_expression("returned_at IS NOT NULL", columns)
    validate_scope_filter_expression("returned_at IS NULL", columns)


def test_rejects_unknown_qualified_and_column_to_column_filters():
    columns = ["region", "channel", "revenue"]
    with pytest.raises(ValueError, match="does not exist"):
        validate_scope_filter_expression("ghost = 'x'", columns)
    with pytest.raises(ValueError, match="Qualified column"):
        validate_scope_filter_expression("sales.region = 'East'", columns)
    with pytest.raises(ValueError, match="literal value"):
        validate_scope_filter_expression("revenue > channel", columns)


def test_rejects_boolean_bypass_constant_comparison_or_and_functions():
    columns = ["region", "revenue"]
    with pytest.raises(ValueError):
        validate_scope_filter_expression("region = 'East' OR TRUE", columns)
    with pytest.raises(ValueError):
        validate_scope_filter_expression("1 = 1", columns)
    with pytest.raises(ValueError):
        validate_scope_filter_expression("read_csv('private.csv') IS NOT NULL", columns)
    with pytest.raises(ValueError):
        validate_scope_filter_expression("region = 'East'; SELECT * FROM data", columns)


def test_null_comparison_requires_explicit_null_operator():
    with pytest.raises(ValueError, match="IS NULL"):
        validate_scope_filter_expression("region = NULL", ["region"])
    with pytest.raises(ValueError, match="IS NULL"):
        validate_scope_filter_expression("region != NULL", ["region"])
