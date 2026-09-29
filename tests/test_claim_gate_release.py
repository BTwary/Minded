from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packages.analytics_core.src.governance.claim_gate import (  # noqa: E402
    ClaimType,
    DesignStatus,
    EvidenceLevel,
    evaluate_gate,
)


def test_descriptive_validated_answer():
    r = evaluate_gate(
        claim_type=ClaimType.OBSERVATION,
        evidence_level=EvidenceLevel.VALIDATED_COMPUTATION,
        design_status=DesignStatus.KNOWN,
        assumptions=[],
        identification_strategy=None,
        verification_passed=True,
    )
    assert r.outcome == "ANSWER"


def test_clean_association_answer():
    r = evaluate_gate(
        claim_type=ClaimType.ASSOCIATION,
        evidence_level=EvidenceLevel.ASSOCIATION,
        design_status=DesignStatus.KNOWN,
        assumptions=[],
        identification_strategy=None,
        verification_passed=True,
    )
    assert r.outcome == "ANSWER"


def test_association_with_unresolved_confounder_is_qualified():
    r = evaluate_gate(
        claim_type=ClaimType.ASSOCIATION,
        evidence_level=EvidenceLevel.ASSOCIATION,
        design_status=DesignStatus.ASSUMED,
        assumptions=["Unmeasured confounding may remain."],
        identification_strategy=None,
        assumption_risk=True,
        verification_passed=True,
    )
    assert r.outcome == "QUALIFIED_ANSWER"
    assert r.blocked_claim


def test_prediction_requires_out_of_sample_validation():
    r = evaluate_gate(
        claim_type=ClaimType.PREDICTION,
        evidence_level=EvidenceLevel.VALIDATED_COMPUTATION,
        design_status=DesignStatus.KNOWN,
        assumptions=[],
        identification_strategy=None,
        prediction_out_of_sample=False,
        verification_passed=True,
    )
    assert r.outcome == "QUALIFIED_ANSWER"
    assert any("holdout" in x.lower() for x in r.recovery_actions)


def test_prediction_with_holdout_answers():
    r = evaluate_gate(
        claim_type=ClaimType.PREDICTION,
        evidence_level=EvidenceLevel.PREDICTION,
        design_status=DesignStatus.KNOWN,
        assumptions=[],
        identification_strategy=None,
        prediction_out_of_sample=True,
        verification_passed=True,
    )
    assert r.outcome == "ANSWER"


def test_causal_without_identification_refuses_and_exposes_recovery():
    r = evaluate_gate(
        claim_type=ClaimType.CAUSAL,
        evidence_level=EvidenceLevel.ASSOCIATION,
        design_status=DesignStatus.UNKNOWN,
        assumptions=[],
        identification_strategy=None,
        verification_passed=True,
        computed_evidence={"estimate": 0.082},
        refusal_id="RG-test",
    )
    assert r.outcome == "REFUSE"
    assert r.refusal_id == "RG-test"
    assert r.max_supported_level == EvidenceLevel.ASSOCIATION
    assert r.recovery_actions
    assert r.missing_evidence


def test_causal_identified_but_not_fully_estimated_is_qualified():
    r = evaluate_gate(
        claim_type=ClaimType.CAUSAL,
        evidence_level=EvidenceLevel.CAUSAL_IDENTIFICATION,
        design_status=DesignStatus.KNOWN,
        assumptions=["No unmeasured confounding."],
        identification_strategy="IDENTIFIED_BACKDOOR",
        verification_passed=True,
    )
    assert r.outcome == "QUALIFIED_ANSWER"


def test_weak_recommendation_is_qualified_not_refused():
    r = evaluate_gate(
        claim_type=ClaimType.RECOMMENDATION,
        evidence_level=EvidenceLevel.ASSOCIATION,
        design_status=DesignStatus.KNOWN,
        assumptions=["Utility is decision-dependent."],
        identification_strategy=None,
        assumption_risk=False,
        verification_passed=True,
    )
    assert r.outcome == "QUALIFIED_ANSWER"


def test_result_is_json_serializable():
    r = evaluate_gate(
        claim_type=ClaimType.ASSOCIATION,
        evidence_level=EvidenceLevel.ASSOCIATION,
        design_status=DesignStatus.KNOWN,
        assumptions=[],
        identification_strategy=None,
        verification_passed=True,
    )
    json.dumps(r.to_dict())


def test_failed_verification_cannot_support_answer():
    r = evaluate_gate(
        claim_type=ClaimType.ASSOCIATION,
        evidence_level=EvidenceLevel.ASSOCIATION,
        design_status=DesignStatus.KNOWN,
        assumptions=[],
        identification_strategy=None,
        verification_passed=False,
    )
    assert r.outcome in {"QUALIFIED_ANSWER", "REFUSE"}
    assert r.evidence_level == EvidenceLevel.RAW_OBSERVATION


def test_refusal_exposes_recovery_not_synthetic_evidence():
    r = evaluate_gate(
        claim_type=ClaimType.CAUSAL,
        evidence_level=EvidenceLevel.ASSOCIATION,
        design_status=DesignStatus.UNKNOWN,
        assumptions=[],
        identification_strategy=None,
        verification_passed=False,
        computed_evidence={"estimate": 0.082},
    )
    assert r.outcome == "REFUSE"
    assert r.computed_evidence["estimate"] == 0.082
    assert r.recovery_actions

