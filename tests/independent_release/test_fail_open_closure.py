"""Deterministic failure-boundary proofs for positive-claim closure."""
from packages.analytics_core.src.governance.claim_gate import ClaimType, DesignStatus, EvidenceLevel, evaluate_gate
from packages.analytics_core.src.intelligence.analytical_identity import FinalAnalyticalContract, AnalyticalIdentityError


def test_missing_verification_never_allows_positive_association_claim():
    gate = evaluate_gate(
        claim_type=ClaimType.ASSOCIATION,
        evidence_level=EvidenceLevel.ASSOCIATION,
        design_status=DesignStatus.KNOWN,
        assumptions=[], identification_strategy=None,
        verification_passed=False,
    )
    assert gate.outcome in {"QUALIFIED_ANSWER", "REFUSE"}
    assert gate.outcome != "ANSWER"


def test_missing_dataset_identity_cannot_form_contract():
    try:
        FinalAnalyticalContract(
            dataset_identity="",
            problem_class="ASSOCIATION", canonical_task="CORRELATION", objective="COMPARE",
            target_column="y", predictor_columns=("x",), selected_method_code=None,
            claim_ceiling=None, verification_regime=None, method_family="CORRELATION",
            bindings=(),
        )
    except AnalyticalIdentityError:
        return
    raise AssertionError("empty dataset identity must fail closed")
