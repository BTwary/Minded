"""Downstream data-quality decision controls for autonomous investigations.

This module turns contextual data-quality findings into explicit analytical
constraints. Findings are not merely reported: when they are relevant to the
current task, they require a documented review/sensitivity path and can
prevent a high-confidence conclusion until that review resolves the concern.

The module is deterministic and depends only on the existing
DataQualityAssessment/findings plus explicitly supplied sensitivity outcomes.
It intentionally does not infer an unobserved missingness mechanism or causal
bias from metadata alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from packages.analytics_core.src.profiling.data_quality_gate import (
    DataQualityAssessment,
    DataQualityFinding,
)


@dataclass(frozen=True)
class DataQualityDecision:
    """Task-scoped downstream decision state derived from quality findings."""

    task: str
    relevant_findings: List[Dict[str, Any]] = field(default_factory=list)
    required_reviews: List[Dict[str, Any]] = field(default_factory=list)
    unresolved_findings: List[str] = field(default_factory=list)
    resolved_findings: List[str] = field(default_factory=list)
    claim_qualification_required: bool = False
    recommendation_allowed: bool = True
    blocking_reasons: List[str] = field(default_factory=list)

    @property
    def has_relevant_findings(self) -> bool:
        return bool(self.relevant_findings)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "relevant_findings": list(self.relevant_findings),
            "required_reviews": list(self.required_reviews),
            "unresolved_findings": list(self.unresolved_findings),
            "resolved_findings": list(self.resolved_findings),
            "claim_qualification_required": self.claim_qualification_required,
            "recommendation_allowed": self.recommendation_allowed,
            "blocking_reasons": list(self.blocking_reasons),
        }


_FINDING_ACTIONS: Dict[str, Dict[str, str]] = {
    "MISSINGNESS_CONCENTRATED_BY_GROUP": {
        "review_code": "MISSINGNESS_SENSITIVITY_REVIEW",
        "review": "Run claim-specific missingness sensitivity analysis and require the conclusion to remain stable under the documented bound.",
        "question": "Could concentrated missingness materially change the requested finding?",
    },
    "OUTLIERS_CONCENTRATED_BY_GROUP": {
        "review_code": "OUTLIER_ROBUSTNESS_REVIEW",
        "review": "Repeat the affected analysis with a documented robust/rank-based or trimmed sensitivity check.",
        "question": "Does the requested finding survive a robust treatment of the concentrated outliers?",
    },
    "SELECTION_BIAS_INDICATOR": {
        "review_code": "SELECTION_BIAS_REVIEW",
        "review": "Define the target population and test whether the conclusion remains stable after accounting for the observed selection mechanism.",
        "question": "Could the observed selection mechanism materially change the requested finding?",
    },
    "TEMPORAL_DATA_QUALITY": {
        "review_code": "TEMPORAL_VALIDITY_REVIEW",
        "review": "Resolve or explicitly exclude invalid time records before interpreting temporal effects.",
        "question": "Could invalid time records materially change the requested temporal finding?",
    },
}


def _task_relevant(finding: DataQualityFinding, task: str) -> bool:
    task_upper = str(task or "").upper()
    affected = {str(x).upper() for x in (finding.affected_methods or [])}
    return task_upper in affected or "ALL" in affected


def assess_downstream_impact(
    quality_assessment: Optional[DataQualityAssessment],
    task: str,
    *,
    missingness_classification: Optional[str] = None,
    outlier_robust: Optional[bool] = None,
    selection_bias_resolved: bool = False,
    temporal_issues_resolved: bool = False,
    include_unresolved_without_runtime_evidence: bool = True,
) -> DataQualityDecision:
    """Translate relevant quality findings into downstream controls.

    ``include_unresolved_without_runtime_evidence`` is used during plan
    compilation: the system records the review requirement before the
    corresponding sensitivity evidence exists. Final controller evaluation
    supplies the runtime outcomes and can clear individual findings.
    """
    if quality_assessment is None:
        return DataQualityDecision(task=str(task or ""))

    relevant: List[Dict[str, Any]] = []
    required_reviews: List[Dict[str, Any]] = []
    unresolved: List[str] = []
    resolved: List[str] = []
    blocking: List[str] = []

    for finding in quality_assessment.contextual_findings:
        if not _task_relevant(finding, task):
            continue
        spec = _FINDING_ACTIONS.get(finding.finding_id)
        if spec is None:
            # Unknown future finding types remain conservative: surface them
            # to the caller and require a review rather than silently ignoring
            # a new quality condition.
            spec = {
                "review_code": f"QUALITY_REVIEW_{finding.finding_id}",
                "review": finding.recommended_action,
                "question": f"Could data-quality finding {finding.finding_id} materially change the requested finding?",
            }

        relevant.append({
            "finding_id": finding.finding_id,
            "severity": finding.severity,
            "issue": finding.issue,
            "scope": dict(finding.scope),
            "evidence": dict(finding.evidence),
            "potential_bias": finding.potential_bias,
            "affected_methods": list(finding.affected_methods),
            "recommended_action": finding.recommended_action,
        })
        required_reviews.append({
            "code": spec["review_code"],
            "finding_id": finding.finding_id,
            "required_action": spec["review"],
            "question": spec["question"],
        })

        is_resolved = False
        if finding.finding_id == "MISSINGNESS_CONCENTRATED_BY_GROUP":
            is_resolved = str(missingness_classification or "").upper() == "ROBUST"
        elif finding.finding_id == "OUTLIERS_CONCENTRATED_BY_GROUP":
            is_resolved = outlier_robust is True
        elif finding.finding_id == "SELECTION_BIAS_INDICATOR":
            is_resolved = bool(selection_bias_resolved)
        elif finding.finding_id == "TEMPORAL_DATA_QUALITY":
            is_resolved = bool(temporal_issues_resolved)

        if is_resolved:
            resolved.append(finding.finding_id)
        else:
            # Missing/unknown runtime resolution is itself unresolved.  The
            # downstream controller must never interpret absence of a
            # sensitivity result as evidence that the quality concern is safe.
            unresolved.append(finding.finding_id)
            blocking.append(
                f"{finding.finding_id} requires the documented downstream review before a high-confidence conclusion is admitted."
            )

    # Every relevant unresolved finding is a claim-quality issue. A finding
    # that affects only an unrelated method does not enter this decision.
    qualification_required = bool(unresolved)
    recommendation_allowed = not qualification_required

    return DataQualityDecision(
        task=str(task or ""),
        relevant_findings=relevant,
        required_reviews=required_reviews,
        unresolved_findings=list(dict.fromkeys(unresolved)),
        resolved_findings=list(dict.fromkeys(resolved)),
        claim_qualification_required=qualification_required,
        recommendation_allowed=recommendation_allowed,
        blocking_reasons=list(dict.fromkeys(blocking)),
    )
