"""Regression coverage for common rank-direction wording and the human approval hand-off."""
from __future__ import annotations

import pandas as pd

from packages.analytics_core.src.intelligence.canonical_question_contract import (
    compile_canonical_question_contract,
    has_multiple_ranking_directives,
)
from packages.analytics_core.src.governance.guided_investigation import (
    AmbiguityStatus,
    ResolverStatus,
    apply_human_guidance_to_contract,
    evaluate_guided_investigation,
)


def test_highest_to_lowest_is_one_direction_and_not_a_filter() -> None:
    frame = pd.DataFrame({
        "realized_revenue": [100.0, 200.0, 150.0, 50.0],
        "region": ["North", "South", "East", "West"],
        "net_revenue": [90.0, 185.0, 140.0, 45.0],
        "customer_tenure_months": [12, 24, 8, 36],
        "order_id": ["A", "B", "C", "D"],
    })
    question = "Calculate total realized_revenue by region, ranked from highest to lowest."

    contract = compile_canonical_question_contract(question, frame)
    assert contract.target_column == "realized_revenue"
    assert contract.grouping_columns == ("region",)
    assert contract.requested_aggregation == "SUM"
    assert contract.ranking_direction == "DESC"
    assert contract.completeness == "COMPLETE"
    assert "filter_instruction" not in contract.unresolved_roles

    preview = evaluate_guided_investigation(question, df=frame)
    assert preview.ambiguity_status == AmbiguityStatus.CLEAR
    assert preview.can_auto_proceed is True
    assert preview.canonical_contract is not None
    assert preview.canonical_contract.target_column == "realized_revenue"
    assert preview.canonical_contract.grouping_columns == ("region",)
    assert "GROUP BY \"region\"" in preview.planned_calculation
    assert "ORDER BY metric_value DESC" in preview.planned_calculation

    resolved = apply_human_guidance_to_contract(question, {"orders": frame}, {})
    assert resolved.status == ResolverStatus.RESOLVED
    assert resolved.contract is not None
    assert resolved.contract.target_column == "realized_revenue"
    assert resolved.contract.grouping_columns == ("region",)


def test_conflicting_ranking_commands_are_still_rejected() -> None:
    assert has_multiple_ranking_directives("Rank revenue from highest to lowest") is False
    assert has_multiple_ranking_directives("Rank revenue top 5 and bottom 5") is True


def test_panel_preserves_guidance_when_a_new_plan_remounts() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    panel = (root / "apps/web/src/components/ProposedAnalysisPanel.tsx").read_text(encoding="utf-8")
    page = (root / "apps/web/src/app/analyst/page.tsx").read_text(encoding="utf-8")
    assert 'retainedGuidance.clarification_response || ""' in panel
    assert "guidanceNeedsPreview ? \"Apply Decisions & Refresh Preview\" : \"Confirm & Run Analysis\"" in panel
    assert "retainedGuidance.selected_metric ?? contractMetric ?? plan.target_metric ?? savedMetric" in panel
    assert "retainedGuidance.selected_group_dimension ?? contractGroup ?? plan.group_dimension ?? savedGroup" in panel
    assert 'const canRestoreSavedParameters = plan.ambiguity_status === "CLEAR"' in panel
    assert "const savedMetric = canRestoreSavedParameters ? saved.metric : undefined" in panel
    assert "const savedGroup = canRestoreSavedParameters ? saved.group : undefined" in panel
    assert "hasExplicitAggregation" in panel
    assert "guidance_context: guidanceFields" in page
    assert "previewedGuidanceKey={previewedGuidanceKey}" in page
