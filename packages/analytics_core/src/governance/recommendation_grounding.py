"""Authoritative recommendation-grounding admissibility rules.

A business recommendation is allowed to become GROUNDED only when the
supporting evidence lineage is explicit and independently verified.  Merely
having evidence IDs, or having an evidence row whose validation flag happens
to be non-empty, is not sufficient.

This module is deliberately independent of SQLAlchemy.  Callers provide the
current canonical evidence/verification status maps, which lets the same rule
be used by the controller and API without duplicating the admissibility logic.
"""
from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple


VERIFIED_EVIDENCE_STATUS = {"VERIFIED", "PASSED"}


@dataclass(frozen=True)
class RecommendationGrounding:
    status: str  # GROUNDED, UNSUPPORTED, INCONCLUSIVE
    supporting_evidence_ids: Tuple[str, ...]
    verified_supporting_evidence_ids: Tuple[str, ...]
    unverified_supporting_evidence_ids: Tuple[str, ...]
    verified_contradicting_evidence_ids: Tuple[str, ...]
    reason: str


def evaluate_recommendation_grounding(
    supporting_evidence_ids: Sequence[str],
    contradicting_evidence_ids: Sequence[str],
    evidence_validation_status_by_id: Mapping[str, str],
    latest_verification_status_by_id: Mapping[str, str],
) -> RecommendationGrounding:
    """Return the canonical admissibility state for a recommendation.

    Rules:
      1. At least one supporting evidence item must be explicitly linked.
      2. Every supporting evidence item must currently be marked VERIFIED/PASSED.
      3. Every supporting evidence item must have a latest VERIFIED/PASSED
         independent verification record.
      4. Any latest VERIFIED/PASSED contradiction blocks grounding and makes
         the recommendation INCONCLUSIVE rather than silently choosing a side.

    The rule is intentionally fail-closed. Unknown/missing statuses are not
    interpreted as verified.
    """
    support = tuple(dict.fromkeys(str(x) for x in (supporting_evidence_ids or []) if x))
    contradict = tuple(dict.fromkeys(str(x) for x in (contradicting_evidence_ids or []) if x))

    if not support:
        return RecommendationGrounding(
            status="UNSUPPORTED",
            supporting_evidence_ids=(),
            verified_supporting_evidence_ids=(),
            unverified_supporting_evidence_ids=(),
            verified_contradicting_evidence_ids=(),
            reason="No supporting evidence is explicitly linked to the recommendation's hypothesis.",
        )

    verified_support: list[str] = []
    unverified_support: list[str] = []
    for evidence_id in support:
        evidence_status = str(evidence_validation_status_by_id.get(evidence_id, "")).upper()
        verification_status = str(latest_verification_status_by_id.get(evidence_id, "")).upper()
        if evidence_status in VERIFIED_EVIDENCE_STATUS and verification_status in VERIFIED_EVIDENCE_STATUS:
            verified_support.append(evidence_id)
        else:
            unverified_support.append(evidence_id)

    verified_contradictions = tuple(
        evidence_id
        for evidence_id in contradict
        if str(evidence_validation_status_by_id.get(evidence_id, "")).upper() in VERIFIED_EVIDENCE_STATUS
        and str(latest_verification_status_by_id.get(evidence_id, "")).upper() in VERIFIED_EVIDENCE_STATUS
    )

    if verified_contradictions:
        return RecommendationGrounding(
            status="INCONCLUSIVE",
            supporting_evidence_ids=support,
            verified_supporting_evidence_ids=tuple(verified_support),
            unverified_supporting_evidence_ids=tuple(unverified_support),
            verified_contradicting_evidence_ids=verified_contradictions,
            reason=(
                "Recommendation grounding is blocked because verified evidence also contradicts "
                "the grounded hypothesis."
            ),
        )

    if unverified_support:
        return RecommendationGrounding(
            status="UNSUPPORTED",
            supporting_evidence_ids=support,
            verified_supporting_evidence_ids=tuple(verified_support),
            unverified_supporting_evidence_ids=tuple(unverified_support),
            verified_contradicting_evidence_ids=(),
            reason=(
                "Recommendation grounding is blocked because at least one supporting evidence item "
                "is not currently independently verified."
            ),
        )

    return RecommendationGrounding(
        status="GROUNDED",
        supporting_evidence_ids=support,
        verified_supporting_evidence_ids=tuple(verified_support),
        unverified_supporting_evidence_ids=(),
        verified_contradicting_evidence_ids=(),
        reason="All explicitly linked supporting evidence is independently verified and no verified contradiction is present.",
    )
