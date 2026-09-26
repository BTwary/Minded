"""Independent regression checks for the 2026-09-09 P0 audit closures.

These checks intentionally avoid importing DuckDB-backed runtime modules so they
remain executable in environments where optional analytical engines are absent.
"""
from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[2]
CONTROLLER = ROOT / "packages/analytics_core/src/runtime/controller.py"
AI_RUNTIME = ROOT / "apps/api/src/ai/runtime.py"
DATABASE = ROOT / "apps/api/src/core/database.py"
PACKAGE = ROOT / "scripts/package_zip.py"
README = ROOT / "README.md"


def _tree(path):
    return ast.parse(path.read_text())


def _nested_function_defs(tree, name):
    return [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]


def test_stable_order_defined_before_first_call():
    tree = _tree(CONTROLLER)
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "execute_investigation")
    body = list(method.body)
    def_index = next(i for i,n in enumerate(body) if isinstance(n, ast.FunctionDef) and n.name == "_stable_hypothesis_order")
    call_lines = [n.lineno for n in ast.walk(method) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_stable_hypothesis_order"]
    assert call_lines
    assert body[def_index].lineno < min(call_lines)


def test_scope_records_initialized_before_experiment_function():
    text = CONTROLLER.read_text()
    init = text.find("experiment_scope_records: List[Dict[str, Any]] = []")
    fn = text.find("def run_selected_experiment(session: Session):")
    assert init >= 0 and fn > init


def test_ev_entity_created_before_verification_use():
    text = CONTROLLER.read_text()
    verify = text.find("# Store Verification Proof")
    create = text.find("ev_entity = Evidence(")
    assert create >= 0 and verify > create


def test_settings_imported_in_runtime():
    assert "from apps.api.src.core.config import settings" in AI_RUNTIME.read_text()


def test_release_packager_excludes_runtime_db():
    text = PACKAGE.read_text()
    assert '"autonomous_analyst.db"' in text
    assert '".db"' in text


def test_readme_does_not_claim_unconditional_runtime_passes():
    text = README.read_text(encoding="utf-8")
    assert "7%2F7_PASSING" not in text
    assert "9%2F9_PASSING" not in text
    assert "6%2F6_PASSING" not in text
