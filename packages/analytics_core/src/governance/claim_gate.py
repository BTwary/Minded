"""Deterministic scientific claim admissibility gate.

This module contains two compatible layers:

* ``admit_positive_claim`` is the legacy boolean gate used by the controller.
* ``evaluate_gate`` is the release-facing three-state Claim Gate:
  ANSWER / QUALIFIED_ANSWER / REFUSE.

The gate is deterministic. LLMs may explain a decision but never choose it.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import IntEnum, Enum
from typing import Any, Iterable, Literal, Optional


class EvidenceLevel(IntEnum):
    RAW_OBSERVATION = 0
    VALIDATED_COMPUTATION = 1
    DESCRIPTIVE_CLAIM = 2
    ASSOCIATION = 3
    ROBUST_ASSOCIATION = 4
    PREDICTION = 5
    CAUSAL_IDENTIFICATION = 6
    CAUSAL_ESTIMATION = 7
    RECOMMENDATION = 8


class ClaimType(IntEnum):
    OBSERVATION = 0
    ASSOCIATION = 3
    PREDICTION = 5
    CAUSAL = 7
    RECOMMENDATION = 8


class DesignStatus(str, Enum):
    KNOWN = "KNOWN"
    ASSUMED = "ASSUMED"
    UNKNOWN = "UNKNOWN"


GateOutcome = Literal["ANSWER", "QUALIFIED_ANSWER", "REFUSE"]


@dataclass(frozen=True)
class GateResult:
    outcome: GateOutcome
    requested_claim: ClaimType
    evidence_level: EvidenceLevel
    design_status: DesignStatus
    assumptions: list[str]
    allowed_claim: str
    blocked_claim: Optional[str]
    reason: str
    recovery_actions: list[str]
    computed_evidence: dict[str, Any]
    refusal_id: Optional[str] = None
    requested_level: Optional[int] = None
    max_supported_level: Optional[int] = None
    identification_strategy: Optional[str] = None
    blocking_conditions: list[str] = None  # normalized in __post_init__
    missing_evidence: list[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "blocking_conditions", list(self.blocking_conditions or []))
        object.__setattr__(self, "missing_evidence", list(self.missing_evidence or []))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["requested_claim"] = self.requested_claim.name
        payload["evidence_level"] = self.evidence_level.name
        payload["design_status"] = self.design_status.value
        return payload


@dataclass(frozen=True)
class ClaimAdmission:
    allowed: bool
    ceiling: str
    reasons: tuple[str, ...] = ()


def _reason_actions(reason_codes: Iterable[str]) -> list[str]:
    mapping = {
        "causal_identification_missing": "Provide a recognized identification strategy such as a randomized design, valid pre-period/control, instrument, or explicit causal DAG.",
        "causal_effect_estimate_missing": "Run a validated causal effect estimator after identification is established.",
        "verified_evidence_missing": "Complete independent verification of the underlying computation before making a stronger claim.",
        "selection_bias": "Resolve the sampling/eligibility mechanism or restrict the claim to the observed sample.",
        "assumptions_unvalidated": "Validate the material assumptions or report the result only conditionally on them.",
        "prediction_holdout_missing": "Evaluate the model on a genuine holdout or time-respecting out-of-sample split.",
        "design_unknown": "Supply or resolve the study/data design before making the requested inferential claim.",
        "claim_exceeds_evidence": "Lower the requested claim level or run additional evidence-producing analyses.",
    }
    return list(dict.fromkeys(mapping[r] for r in reason_codes if r in mapping))


def evaluate_gate(
    *,
    claim_type: ClaimType,
    evidence_level: EvidenceLevel,
    design_status: DesignStatus,
    assumptions: list[str],
    identification_strategy: str | None,
    computed_evidence: Optional[dict[str, Any]] = None,
    assumption_risk: bool = False,
    prediction_out_of_sample: bool = False,
    selection_bias: bool = False,
    verification_passed: bool = True,
    refusal_id: Optional[str] = None,
) -> GateResult:
    """Evaluate the release-facing claim gate.

    The decision is deliberately conservative. The numerical evidence is never
    manufactured by this function; callers must supply the achieved evidence
    level from already-existing analytical components.
    """
    evidence_level = EvidenceLevel(int(evidence_level))
    claim_type = ClaimType(int(claim_type))
    design_status = DesignStatus(design_status)
    assumptions = list(assumptions or [])
    computed_evidence = dict(computed_evidence or {})
    blocking: list[str] = []
    missing: list[str] = []

    # A positive claim cannot rest on an unverified computation.
    if not verification_passed and evidence_level >= EvidenceLevel.VALIDATED_COMPUTATION:
        evidence_level = EvidenceLevel.RAW_OBSERVATION
        blocking.append("verified evidence is missing")

    # Explicitly requested causal inference is special: a causal claim requires
    # an identification strategy, even if a numerical association is strong.
    if claim_type == ClaimType.CAUSAL and not identification_strategy:
        blocking.append("no recognized causal identification strategy")
        missing.extend([
            "pre-period + control group",
            "randomized holdout",
            "valid instrument",
        ])
        return GateResult(
            outcome="REFUSE",
            requested_claim=claim_type,
            evidence_level=evidence_level,
            design_status=design_status,
            assumptions=assumptions,
            allowed_claim="Only the computed observational/descriptive result may be reported.",
            blocked_claim="A causal claim is not supported without a recognized identification strategy.",
            reason="The requested causal conclusion does not follow from the available evidence.",
            recovery_actions=_reason_actions(["causal_identification_missing"]),
            computed_evidence=computed_evidence,
            refusal_id=refusal_id,
            requested_level=claim_type.value,
            max_supported_level=evidence_level.value,
            identification_strategy=None,
            blocking_conditions=blocking,
            missing_evidence=missing,
        )

    if claim_type == ClaimType.CAUSAL and identification_strategy and evidence_level < EvidenceLevel.CAUSAL_ESTIMATION:
        return GateResult(
            outcome="QUALIFIED_ANSWER",
            requested_claim=claim_type,
            evidence_level=evidence_level,
            design_status=design_status,
            assumptions=assumptions,
            allowed_claim="Estimated causal effect under the stated identification and modeling assumptions.",
            blocked_claim="An unconditional causal claim.",
            reason="A causal identification strategy is present, but the available evidence does not justify an unconditional causal statement.",
            recovery_actions=_reason_actions(["causal_effect_estimate_missing"]),
            computed_evidence=computed_evidence,
            refusal_id=refusal_id,
            requested_level=claim_type.value,
            max_supported_level=evidence_level.value,
            identification_strategy=identification_strategy,
            blocking_conditions=["causal estimate / stronger evidence remains conditional"],
            missing_evidence=[],
        )

    if claim_type == ClaimType.PREDICTION and not prediction_out_of_sample:
        return GateResult(
            outcome="QUALIFIED_ANSWER",
            requested_claim=claim_type,
            evidence_level=min(evidence_level, EvidenceLevel.PREDICTION),
            design_status=design_status,
            assumptions=assumptions,
            allowed_claim="Model-based prediction subject to stated validation limitations.",
            blocked_claim="A demonstrated out-of-sample performance claim.",
            reason="Prediction evidence exists without a genuine out-of-sample validation record.",
            recovery_actions=_reason_actions(["prediction_holdout_missing"]),
            computed_evidence=computed_evidence,
            refusal_id=refusal_id,
            requested_level=claim_type.value,
            max_supported_level=evidence_level.value,
            identification_strategy=identification_strategy,
            blocking_conditions=["out-of-sample validation missing"],
            missing_evidence=["holdout or rolling-origin validation"],
        )

    if selection_bias and claim_type in {ClaimType.ASSOCIATION, ClaimType.PREDICTION, ClaimType.RECOMMENDATION}:
        blocking.append("selection/sampling bias remains unresolved")

    gap = claim_type.value - evidence_level.value
    # Recommendations are inherently decision-context dependent. Evidence at
    # association/prediction level can support a qualified recommendation, but
    # does not silently become an unconditional decision claim.
    if claim_type == ClaimType.RECOMMENDATION and evidence_level >= EvidenceLevel.ASSOCIATION:
        outcome: GateOutcome = "QUALIFIED_ANSWER"
        reason = "The evidence can inform a recommendation, but the preferred action depends on explicit utility, risk, and decision-context assumptions."
        allowed = "A recommendation conditional on the stated decision assumptions."
        blocked = "An unconditional claim that one action is objectively best."
    elif assumption_risk or blocking:
        outcome = "QUALIFIED_ANSWER"
        reason = "The requested claim is usable only under explicit assumptions or unresolved design conditions."
        if blocking:
            reason += " " + "; ".join(blocking) + "."
        allowed = "The result may be reported with the stated assumptions and limitations."
        blocked = "A stronger unconditional claim."
    elif gap <= 0:
        outcome = "ANSWER"
        reason = "The achieved evidence level meets or exceeds the requested claim level under the supplied design context."
        allowed = "The requested claim is supported by the current evidence."
        blocked = None
    elif gap <= 1 or (gap == 2 and design_status == DesignStatus.KNOWN):
        outcome = "QUALIFIED_ANSWER"
        reason = "The requested claim is usable only under explicit assumptions or unresolved design conditions."
        if blocking:
            reason += " " + "; ".join(blocking) + "."
        allowed = "The result may be reported with the stated assumptions and limitations."
        blocked = "A stronger unconditional claim."
        if assumption_risk:
            blocking.append("material assumptions remain unvalidated")
    else:
        outcome = "REFUSE"
        blocking.append("requested claim exceeds supported evidence level")
        reason = "The requested conclusion does not follow from the current evidence."
        allowed = "Only claims at or below the achieved evidence level may be reported."
        blocked = "The requested stronger claim."

    reason_codes = []
    if selection_bias:
        reason_codes.append("selection_bias")
    if assumption_risk:
        reason_codes.append("assumptions_unvalidated")
    if outcome == "REFUSE":
        reason_codes.append("claim_exceeds_evidence")

    return GateResult(
        outcome=outcome,
        requested_claim=claim_type,
        evidence_level=evidence_level,
        design_status=design_status,
        assumptions=assumptions,
        allowed_claim=allowed,
        blocked_claim=blocked,
        reason=reason,
        recovery_actions=_reason_actions(reason_codes),
        computed_evidence=computed_evidence,
        refusal_id=refusal_id,
        requested_level=claim_type.value,
        max_supported_level=evidence_level.value,
        identification_strategy=identification_strategy,
        blocking_conditions=blocking,
        missing_evidence=missing,
    )


def admit_positive_claim(
    *,
    verdict_type: str,
    directly_tested: bool,
    verified_evidence: bool,
    missingness_classification: Optional[str] = None,
    causal_identifiable: bool = False,
    requested_causal: bool = False,
    causal_effect_estimated: bool = False,
    method_claim_ceiling: Optional[str] = None,
    required_assumptions_satisfied: bool = True,
    probability_calibrated: bool = False,
    selection_bias_indicators: Optional[Iterable[str]] = None,
    unvalidated_high_risk_assumptions: int = 0,
    full_scope: bool = True,
    sampling_policy: str = "NONE",
    sampling_reason: Optional[str] = None,
) -> ClaimAdmission:
    """Legacy positive-claim gate retained for controller compatibility."""
    reasons: list[str] = []
    if not directly_tested:
        reasons.append("leading_hypothesis_not_directly_tested")
    if not verified_evidence:
        reasons.append("verified_evidence_missing")
    if missingness_classification in {"SENSITIVE", "UNIDENTIFIABLE", "INSUFFICIENT_EVIDENCE"}:
        reasons.append(f"missingness_{missingness_classification.lower()}")
    selection_indicators = tuple(str(x) for x in (selection_bias_indicators or ()) if str(x).strip())
    if selection_indicators and verdict_type in {"DIAGNOSED", "STATISTICALLY_SIGNIFICANT", "SUPPORTED"}:
        reasons.append("selection_bias_indicators_present")
    if requested_causal and not causal_identifiable:
        reasons.append("causal_effect_not_identified")
    if requested_causal and causal_identifiable and not causal_effect_estimated:
        reasons.append("causal_effect_estimate_missing")
    if not required_assumptions_satisfied:
        reasons.append("required_method_assumptions_not_satisfied")
    if int(unvalidated_high_risk_assumptions or 0) > 0 and verdict_type in {"DIAGNOSED", "STATISTICALLY_SIGNIFICANT", "SUPPORTED", "OBSERVED"}:
        reasons.append("material_high_risk_assumptions_unvalidated")
    if (not full_scope or str(sampling_policy or "NONE").upper() != "NONE") and verdict_type in {"DIAGNOSED", "STATISTICALLY_SIGNIFICANT", "SUPPORTED"}:
        reasons.append("sampled_analysis_scope")
    calibration_status = "CALIBRATED" if probability_calibrated else "MODEL_BASED_BELIEF"
    admission_reasons = tuple(r for r in reasons if r != "posterior_probability_not_empirically_calibrated")
    ceiling = method_claim_ceiling or ("CAUSAL" if requested_causal and causal_identifiable else "ASSOCIATIONAL")
    positive = verdict_type in {"DIAGNOSED", "STATISTICALLY_SIGNIFICANT", "SUPPORTED", "OBSERVED"}
    return ClaimAdmission(allowed=positive and not admission_reasons, ceiling=ceiling, reasons=admission_reasons)
