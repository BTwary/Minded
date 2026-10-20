"""Fail-closed gate for hypothesis verdicts at the final answer boundary.

This module intentionally contains pure, deterministic policy helpers so the
policy can be tested independently of the orchestration/database stack.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

HYPOTHESIS_VERDICTS = frozenset({
    "DIAGNOSED",
    "STATISTICALLY_SIGNIFICANT",
    "NO_DETECTABLE_EFFECT",
})


def has_current_verified_supporting_evidence(
    supporting_evidence_ids: Iterable[str] | None,
    *,
    evidence_validation_status_by_id: Mapping[str, object] | None,
    latest_verification_status_by_id: Mapping[str, object] | None,
) -> bool:
    """Return True only if at least one *supporting* item has current proof.

    Contradictory evidence must never satisfy this condition. Both the evidence
    row's validation state and its latest independent verification record must
    be VERIFIED. A stale earlier VERIFIED record cannot override a newer FAIL.
    """
    evidence_ids = {str(value) for value in (supporting_evidence_ids or []) if value is not None}
    evidence_statuses = evidence_validation_status_by_id or {}
    verification_statuses = latest_verification_status_by_id or {}
    return any(
        str(evidence_statuses.get(evidence_id, "")).strip().upper() == "VERIFIED"
        and str(verification_statuses.get(evidence_id, "")).strip().upper() == "VERIFIED"
        for evidence_id in evidence_ids
    )


def verification_gate_reason(
    verdict_type: str,
    *,
    all_verifications_passed: bool,
    has_verified_evidence: bool,
) -> str | None:
    """Return a refusal reason unless a hypothesis verdict has verified support.

    Use identity checks, not truthiness: values such as the string ``"false"``
    and integers are not valid proof that a verification gate passed.
    Descriptive OBSERVED results are handled by their separate contract gate.
    """
    if str(verdict_type or "").strip().upper() not in HYPOTHESIS_VERDICTS:
        return None
    if all_verifications_passed is not True:
        return (
            "verification gate failed: one or more experiments were failed, "
            "blocked, partial, or otherwise not verified"
        )
    if has_verified_evidence is not True:
        return "verification gate failed: no independently verified supporting evidence for the leading hypothesis"
    return None


def analyst_group_comparison_dual_statistical_confirmation(kind: object, numbers: Mapping[str, object] | None, alpha: float = 0.05) -> bool:
    """True only when an analyst-owned group-difference claim has dual statistical confirmation.

    The hypothesis loop can test a different claim (e.g. concentration) than the question asked
    ("does X differ by region?"), leaving the leading hypothesis without supporting evidence even
    though the analyst layer recomputed the comparison itself. That recomputation counts as
    verified support only if the parametric (ANOVA) and rank-based (Kruskal-Wallis) tests both
    reject at ``alpha``, the outlier/robustness check held and power was adequate. Anything
    missing, non-finite or unflagged is NOT verified (identity checks, as elsewhere in this module).
    """
    if str(kind or "").upper() != "GROUP_COMPARISON" or not isinstance(numbers, Mapping):
        return False
    try:
        anova_p = float(numbers.get("anova_p"))
        kruskal_p = float(numbers.get("kruskal_p"))
    except (TypeError, ValueError):
        return False
    if not (0.0 <= anova_p < alpha and 0.0 <= kruskal_p < alpha):
        return False
    return numbers.get("robust") is True and numbers.get("adequate_power") is True


# Backward compatibility alias
analyst_group_comparison_independently_verified = analyst_group_comparison_dual_statistical_confirmation

