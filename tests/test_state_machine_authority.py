"""P0 vocabulary/authority test for the canonical lifecycle state machine."""
from pathlib import Path

from packages.analytics_core.src.execution.state_machine import AnalyticalPhase, InvestigationStateMachine

ROOT = Path(__file__).resolve().parents[1]


def test_execution_state_machine_defines_canonical_investigation_lifecycle():
    assert hasattr(AnalyticalPhase, "QUESTION_UNDERSTANDING")
    assert hasattr(AnalyticalPhase, "VERIFICATION")
    assert hasattr(AnalyticalPhase, "BELIEF_UPDATE")
    assert hasattr(AnalyticalPhase, "STOPPING")
    assert hasattr(AnalyticalPhase, "VERDICT")
    assert InvestigationStateMachine.can_transition("RUNNING", "VERIFYING")


def test_legacy_analytical_state_machine_has_no_production_imports():
    legacy = "packages/analytics_core/src/runtime/analytical_state_machine.py"
    hits = []
    for path in ROOT.glob("**/*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith("tests/") or rel == legacy:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "runtime.analytical_state_machine" in text:
            hits.append(rel)
    assert hits == []


def test_semantic_graph_unknown_defaults_are_non_optimistic():
    from packages.schemas.src.semantic_graph import MetricNode, RelationshipEdge

    metric = MetricNode(metric_name="ambiguous", table_name="t")
    rel = RelationshipEdge(source_table="a", target_table="b", source_column="id", target_column="id")
    assert metric.aggregation_type == "UNKNOWN"
    assert rel.join_safety_score is None
    assert rel.confidence is None
    assert rel.fanout_risk is None


def test_semantic_resolution_does_not_infer_naive_timestamp_timezone(monkeypatch):
    import pandas as pd
    from packages.analytics_core.src.engines.intent import IntentEngine
    from packages.analytics_core.src.engines.semantic import SemanticEngine

    monkeypatch.delenv("AAOS_BUSINESS_TIMEZONE", raising=False)
    df = pd.DataFrame({
        "order_date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
        "revenue": [10.0, 20.0, 30.0],
    })
    intent = IntentEngine.parse_intent("What was revenue in January?", available_columns=list(df.columns))
    resolved = SemanticEngine().resolve_schema(intent, {"orders": df})
    assert resolved.time_col == "order_date"
    assert resolved.time_zone_policy == "UNRESOLVED_NAIVE_TIMEZONE"
