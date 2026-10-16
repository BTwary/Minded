"""Static guard that the typed conditional policy remains wired into the controller.

This complements (but does not replace) runtime/controller tests, which require the
full locked application dependencies including DuckDB and Polars.
"""
from pathlib import Path


def test_controller_resolves_and_persists_typed_branch_decisions_before_specialist_work():
    root = Path(__file__).resolve().parents[2]
    source = (root / "packages/analytics_core/src/runtime/controller.py").read_text(encoding="utf-8")
    assert "resolve_plan_branches(_typed_plan, _typed_facts)" in source
    assert "conditional_execution_policy(_typed_task, _typed_branch_decisions)" in source
    assert 'event_type="investigation.typed_conditional_plan.resolved"' in source
    assert 'event_type="investigation.typed_conditional_plan.enforced"' in source
    assert 'and (_typed_task != "CAUSAL_EFFECT" or _typed_branch_runs("B-ID"))' in source
    assert '"horizon_status"' in source
    assert '"y_type"' in source
    assert '"group_dimension_col"' in source


def test_release_bootstrap_keeps_hashes_and_fails_closed():
    root = Path(__file__).resolve().parents[2]
    source = (root / "scripts/bootstrap_release_env.ps1").read_text(encoding="utf-8")
    assert "--require-hashes -r" in source
    assert "pip check" in source
    assert "release_gate.py" in source
    assert "do not remove hashes or relax pins" in source
