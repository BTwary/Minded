from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def governance():
    try:
        from packages.analytics_core.src.security import governance as module
    except ModuleNotFoundError as exc:
        pytest.skip(f"governance dependencies unavailable: {exc.name}")
    return module


def test_nested_subquery_never_receives_first_where_string_injection(governance):
    GovernanceEngine = governance.GovernanceEngine
    sql = """
    SELECT o.region, SUM(o.revenue) AS total_revenue
    FROM orders o
    JOIN (SELECT customer_id FROM customers WHERE active = true) c
      ON o.customer_id = c.customer_id
    GROUP BY o.region
    """
    out = GovernanceEngine.inject_rbac_filters(sql, {"tenant_id": "tenant_A"})
    import sqlglot
    from sqlglot import exp

    parsed = sqlglot.parse_one(out, dialect="duckdb")
    # sqlglot's internal AST key for the FROM clause differs by version
    # (e.g. "from_" in sqlglot 30.x, not "from") -- use the public,
    # version-stable exp.Select.find(exp.From) instead of reading
    # .args.get("from") directly, matching the fix in governance.py.
    selects = [s for s in parsed.find_all(exp.Select) if s.find(exp.From) is not None]
    assert len(selects) == 2
    assert all(select.args.get("where") is not None for select in selects)

    inner_sql = [s.sql(dialect="duckdb") for s in selects if "active" in s.sql(dialect="duckdb").lower()][0].lower()
    assert "tenant_id = 'tenant_a'" in inner_sql
    assert "active = true" in inner_sql

    outer_sql = [s.sql(dialect="duckdb") for s in selects if "sum(o.revenue)" in s.sql(dialect="duckdb").lower()][0].lower()
    assert "tenant_id = 'tenant_a'" in outer_sql


def test_malformed_sql_fails_closed(governance):
    GovernanceEngine = governance.GovernanceEngine
    with pytest.raises(governance.RLSQueryRewriteError):
        GovernanceEngine.inject_rbac_filters("SELECT * FROM (SELECT", {"tenant_id": "tenant_A"})


def test_sqlglot_unavailable_fails_closed(monkeypatch, governance):
    GovernanceEngine = governance.GovernanceEngine
    monkeypatch.setattr(governance, "_HAS_SQLGLOT", False)
    with pytest.raises(governance.RLSQueryRewriteError, match="SQLGlot is required"):
        GovernanceEngine.inject_rbac_filters(
            "SELECT SUM(revenue) FROM orders", {"tenant_id": "tenant_A"}
        )


def test_sqlglot_rewrite_exception_fails_closed(monkeypatch, governance):
    GovernanceEngine = governance.GovernanceEngine
    monkeypatch.setattr(governance, "_HAS_SQLGLOT", True)

    class BrokenSqlglot:
        @staticmethod
        def parse_one(*args, **kwargs):
            raise RuntimeError("synthetic parser failure")

    monkeypatch.setattr(governance, "sqlglot", BrokenSqlglot)
    with pytest.raises(governance.RLSQueryRewriteError, match="rewrite failed closed"):
        GovernanceEngine.inject_rbac_filters(
            "SELECT SUM(revenue) FROM orders", {"tenant_id": "tenant_A"}
        )


def test_empty_user_context_is_noop(governance):
    sql = "SELECT SUM(revenue) FROM orders"
    assert governance.GovernanceEngine.inject_rbac_filters(sql, None) == sql
    assert governance.GovernanceEngine.inject_rbac_filters(sql, {}) == sql
