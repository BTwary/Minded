import os
import ast
from pathlib import Path

def test_production_api_docs_disabled_and_cors_not_wildcard():
    src = Path("apps/api/src/main.py").read_text()
    assert "docs_url=None if is_production" in src
    assert 'allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]' in src
    assert 'allow_headers=["Authorization", "Content-Type", "Accept", "X-Request-ID"]' in src

def test_production_schema_guard_is_fail_closed():
    src = Path("apps/api/src/core/database.py").read_text()
    tree = ast.parse(src)
    assert "_assert_production_schema_current" in src
    assert "alembic_version" in src
    # The guard used to hard-code a specific expected revision (formerly
    # "8c1d2e4f7a90" here), which goes stale on every new migration that
    # lands. It was intentionally rewritten to resolve the authoritative
    # head from Alembic's own ScriptDirectory at runtime instead -- assert
    # on that design (dynamic resolution + explicit single-head guard +
    # fail-closed on mismatch) rather than pinning a revision string that
    # this test would otherwise have to be updated by hand for every future
    # migration.
    assert "ScriptDirectory" in src and "get_heads()" in src
    assert "must have exactly one head" in src
    assert "does not match Alembic head" in src
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_assert_production_schema_current")
    fn_src = ast.get_source_segment(src, fn) or ""
    assert "hard-cod" in fn_src.lower() or "hard cod" in fn_src.lower(), (
        "guard should keep documenting why it resolves the head dynamically instead of hard-coding a revision"
    )

def test_direct_analysis_script_fails_on_exception():
    src = Path("scripts/test_direct_analysis.py").read_text()
    assert "raise SystemExit(1)" in src
