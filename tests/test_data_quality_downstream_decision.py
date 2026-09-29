from packages.analytics_core.src.profiling.data_quality_gate import (
    DataQualityAssessment,
    DataQualityFinding,
)
from packages.analytics_core.src.profiling.data_quality_decision import (
    assess_downstream_impact,
)


def _assessment(*findings):
    return DataQualityAssessment(
        dataset_name="demo",
        row_count=100,
        column_count=4,
        overall_quality_score=85.0,
        fitness_verdict="CAUTION",
        can_proceed=True,
        contextual_findings=list(findings),
    )


def _finding(fid, methods=("COMPARISON",), severity="HIGH"):
    return DataQualityFinding(
        finding_id=fid,
        severity=severity,
        issue=f"issue:{fid}",
        scope={"x": "y"},
        evidence={"observed": True},
        potential_bias="potential distortion",
        affected_methods=list(methods),
        recommended_action="review it",
    )


def test_relevant_missingness_finding_requires_review_and_blocks_recommendation():
    result = assess_downstream_impact(_assessment(_finding("MISSINGNESS_CONCENTRATED_BY_GROUP")), "COMPARISON")
    assert result.claim_qualification_required
    assert not result.recommendation_allowed
    assert "MISSINGNESS_CONCENTRATED_BY_GROUP" in result.unresolved_findings
    assert result.required_reviews[0]["code"] == "MISSINGNESS_SENSITIVITY_REVIEW"


def test_robust_missingness_sensitivity_resolves_missingness_constraint():
    result = assess_downstream_impact(
        _assessment(_finding("MISSINGNESS_CONCENTRATED_BY_GROUP")),
        "COMPARISON",
        missingness_classification="ROBUST",
        include_unresolved_without_runtime_evidence=False,
    )
    assert not result.claim_qualification_required
    assert result.recommendation_allowed
    assert result.resolved_findings == ["MISSINGNESS_CONCENTRATED_BY_GROUP"]


def test_relevant_outlier_finding_requires_robust_check():
    result = assess_downstream_impact(
        _assessment(_finding("OUTLIERS_CONCENTRATED_BY_GROUP", methods=("ASSOCIATION",))),
        "ASSOCIATION",
        include_unresolved_without_runtime_evidence=True,
    )
    assert result.claim_qualification_required
    assert result.required_reviews[0]["code"] == "OUTLIER_ROBUSTNESS_REVIEW"


def test_robust_outlier_result_resolves_outlier_constraint():
    result = assess_downstream_impact(
        _assessment(_finding("OUTLIERS_CONCENTRATED_BY_GROUP")),
        "ASSOCIATION",
        outlier_robust=True,
        include_unresolved_without_runtime_evidence=False,
    )
    assert not result.claim_qualification_required
    assert result.recommendation_allowed


def test_unrelated_quality_finding_does_not_block_task():
    result = assess_downstream_impact(
        _assessment(_finding("OUTLIERS_CONCENTRATED_BY_GROUP", methods=("PREDICTION",))),
        "COMPARISON",
        include_unresolved_without_runtime_evidence=False,
    )
    assert not result.has_relevant_findings
    assert result.recommendation_allowed


def test_selection_bias_remains_conservative_without_resolution_evidence():
    result = assess_downstream_impact(
        _assessment(_finding("SELECTION_BIAS_INDICATOR")),
        "COMPARISON",
        include_unresolved_without_runtime_evidence=False,
    )
    assert result.claim_qualification_required
    assert not result.recommendation_allowed


def test_temporal_quality_remains_blocked_until_temporal_records_are_resolved():
    result = assess_downstream_impact(
        _assessment(_finding("TEMPORAL_DATA_QUALITY", methods=("FORECAST",))),
        "FORECAST",
        temporal_issues_resolved=False,
        include_unresolved_without_runtime_evidence=False,
    )
    assert result.claim_qualification_required
    assert result.required_reviews[0]["code"] == "TEMPORAL_VALIDITY_REVIEW"
