from packages.analytics_core.src.governance.claim_gate import (
    ClaimType,
    DesignStatus,
    EvidenceLevel,
    evaluate_gate,
)


def test_refuse_structured_gate_cannot_be_positive():
    result = evaluate_gate(
        claim_type=ClaimType.CAUSAL,
        evidence_level=EvidenceLevel.ASSOCIATION,
        design_status=DesignStatus.KNOWN,
        assumptions=[],
        identification_strategy=None,
        verification_passed=True,
    )
    assert result.outcome == "REFUSE"
    assert result.requested_claim == ClaimType.CAUSAL


def test_qualified_gate_is_distinct_from_answer():
    result = evaluate_gate(
        claim_type=ClaimType.PREDICTION,
        evidence_level=EvidenceLevel.PREDICTION,
        design_status=DesignStatus.KNOWN,
        assumptions=[],
        identification_strategy=None,
        verification_passed=True,
        prediction_out_of_sample=False,
    )
    assert result.outcome == "QUALIFIED_ANSWER"
    assert result.outcome != "ANSWER"
