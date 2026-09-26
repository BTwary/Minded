"""Reproduction test verifying backward-compatibility preservation across schemas and consumers."""
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from packages.schemas.src.analysis import (
    AnalysisResponse,
    AnalysisStatus,
    AnalyticalVerdict,
    FindingSchema,
    MultiVectorConfidence,
)
from packages.schemas.src.semantic_graph import (
    RelationshipEdge,
    SemanticWorldModelSchema,
)


def test_confidence_breakdown_loss():
    print("[TEST 1] Reproducing confidence_breakdown on AnalysisResponse...")
    mb = MultiVectorConfidence(
        evidence_quality=95,
        data_quality=98,
        statistical_strength=92,
        causal_evidence="OBSERVATIONAL",
        model_reliability=90,
        overall_verdict="CONFIRMED",
    )
    # Instantiate AnalysisResponse with confidence_breakdown
    resp = AnalysisResponse(
        id="ANA-001",
        project_id="PRJ-001",
        question="Why did revenue drop?",
        status=AnalysisStatus.COMPLETED,
        verdict=AnalyticalVerdict.CONFIRMED,
        confidence_breakdown=mb,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    assert hasattr(resp, "confidence_breakdown"), "confidence_breakdown attribute missing!"
    assert resp.confidence_breakdown is not None, "confidence_breakdown was None!"
    assert resp.confidence_breakdown.evidence_quality == 95, "evidence_quality mismatch!"
    print(" -> PASSED: confidence_breakdown successfully preserved on AnalysisResponse.")


def test_recommended_actions_loss():
    print("[TEST 2] Reproducing recommended_actions on FindingSchema...")
    finding = FindingSchema(
        id="FND-001",
        title="Revenue Contraction",
        summary="Region B dropped 40%",
        importance_score=0.9,
        recommended_actions=["Audit Region B logistics pipeline.", "Verify discount approvals."],
    )
    assert hasattr(finding, "recommended_actions"), "recommended_actions attribute missing!"
    assert len(finding.recommended_actions) == 2, "recommended_actions length mismatch!"
    print(" -> PASSED: recommended_actions successfully preserved on FindingSchema.")


def test_semantic_builder_field_loss():
    print("[TEST 3] Reproducing semantic-builder fields on RelationshipEdge and SemanticWorldModelSchema...")
    edge = RelationshipEdge(
        source_table="orders",
        target_table="customers",
        source_column="customer_id",
        target_column="customer_id",
        cardinality="many_to_one",
        join_safety_score=0.95,
        fanout_risk=False,
        confidence=0.95,
    )
    assert hasattr(edge, "join_safety_score"), "join_safety_score missing on RelationshipEdge!"
    assert hasattr(edge, "fanout_risk"), "fanout_risk missing on RelationshipEdge!"
    assert edge.join_safety_score == 0.95

    swm = SemanticWorldModelSchema(
        relationships=[edge],
        verified_grains={"orders": ["order_id"]},
    )
    assert "orders" in swm.verified_grains
    assert swm.relationships[0].fanout_risk is False
    print(" -> PASSED: semantic builder fields successfully preserved.")


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING SCHEMA COMPATIBILITY REPRODUCTION SUITE")
    print("=" * 70)
    try:
        test_confidence_breakdown_loss()
    except Exception as e:
        print(f" -> FAILED (reproduced): {e}")

    try:
        test_recommended_actions_loss()
    except Exception as e:
        print(f" -> FAILED (reproduced): {e}")

    try:
        test_semantic_builder_field_loss()
    except Exception as e:
        print(f" -> FAILED (reproduced): {e}")
