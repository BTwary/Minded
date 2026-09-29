"""Builds the canonical SemanticBindingSet from AA-OS's existing,
authoritative semantic resolution (v20-A section 8/9).

This module deliberately implements NO new column-selection heuristic. It
only classifies the analytical *role* of columns that
``packages.analytics_core.src.engines.semantic.SemanticEngine`` and
``packages.analytics_core.src.intelligence.universal_question_planner``
have already resolved. If a raw ``semantic.*`` field disagrees with the
canonical binding produced here, the canonical binding -- not the raw
field -- is what executable experiments must honor going forward.
"""
from __future__ import annotations

from typing import Any, List, Optional

from packages.analytics_core.src.intelligence.semantic_builder_vocabulary import (
    CHURN_STATUS_MAP,
    TARGET_STYLE_TASKS,
)
from packages.schemas.src.semantic_binding import SemanticBinding, SemanticBindingSet
from packages.schemas.src.semantic_role import SemanticRole, ResolutionStatus


def _status_for(value: Optional[str], *, ambiguous: bool = False) -> ResolutionStatus:
    if ambiguous:
        return ResolutionStatus.AMBIGUOUS
    if not value:
        return ResolutionStatus.UNRESOLVED
    return ResolutionStatus.RESOLVED


def build_binding_set(
    semantic: Any,
    problem_class: str,
    *,
    explanatory_columns: Optional[List[str]] = None,
) -> SemanticBindingSet:
    """Build the canonical SemanticBindingSet for one contract version.

    Args:
        semantic: the resolved ``SemanticResolution`` (engines/semantic.py).
        problem_class: the compiler task string (e.g. "ASSOCIATION",
            "FORECAST", "SEGMENTATION", ...).
        explanatory_columns: additional explanatory columns resolved by the
            compiler's SemanticContract, if any beyond what SemanticResolution
            itself exposes (e.g. secondary_metric_col for correlation).
    """
    bindings: List[SemanticBinding] = []
    table = getattr(semantic, "primary_dataset_name", "") or ""
    task = (problem_class or "").upper()

    target_col = getattr(semantic, "target_metric_col", None)
    churn_col = getattr(semantic, "churn_event_col", None)
    churn_status_raw = getattr(semantic, "churn_event_resolution_status", "RESOLVED")

    # -- Primary outcome/target -------------------------------------------
    if churn_col:
        churn_status = CHURN_STATUS_MAP.get(str(churn_status_raw).upper(), ResolutionStatus.RESOLVED)
        bindings.append(SemanticBinding(
            column=churn_col, table=table, role=SemanticRole.OUTCOME,
            resolution_status=churn_status, confidence=1.0 if churn_status == ResolutionStatus.RESOLVED else 0.0,
            semantic_type="churn_event",
            provenance={"source": "semantic.churn_event_col"},
        ))
    elif target_col:
        role = SemanticRole.TARGET if task in TARGET_STYLE_TASKS else SemanticRole.OUTCOME
        bindings.append(SemanticBinding(
            column=target_col, table=table, role=role,
            resolution_status=_status_for(target_col),
            confidence=1.0,
            provenance={"source": "semantic.target_metric_col"},
        ))

    # -- Explanatory variable(s) --------------------------------------------
    secondary_col = getattr(semantic, "secondary_metric_col", None)
    if secondary_col:
        bindings.append(SemanticBinding(
            column=secondary_col, table=table, role=SemanticRole.EXPLANATORY_VARIABLE,
            resolution_status=_status_for(secondary_col),
            confidence=1.0,
            provenance={"source": "semantic.secondary_metric_col"},
        ))
    for col in (explanatory_columns or []):
        if not col:
            continue
        if any(b.column == col and b.role == SemanticRole.EXPLANATORY_VARIABLE for b in bindings):
            continue
        bindings.append(SemanticBinding(
            column=col, table=table, role=SemanticRole.EXPLANATORY_VARIABLE,
            resolution_status=_status_for(col),
            confidence=0.9,
            provenance={"source": "compiler.explanatory_columns"},
        ))

    # -- Grouping dimension --------------------------------------------------
    group_col = getattr(semantic, "group_dimension_col", None)
    if group_col:
        bindings.append(SemanticBinding(
            column=group_col, table=table, role=SemanticRole.GROUPING_DIMENSION,
            resolution_status=_status_for(group_col),
            confidence=1.0,
            provenance={"source": "semantic.group_dimension_col"},
        ))

    # -- Time variable ---------------------------------------------------
    time_col = getattr(semantic, "time_col", None)
    if time_col:
        bindings.append(SemanticBinding(
            column=time_col, table=table, role=SemanticRole.TIME_VARIABLE,
            resolution_status=_status_for(time_col),
            confidence=1.0,
            provenance={"source": "semantic.time_col"},
        ))

    # -- Churn survival-analysis auxiliary roles -----------------------------
    exposure_col = getattr(semantic, "churn_exposure_col", None)
    if exposure_col:
        bindings.append(SemanticBinding(
            column=exposure_col, table=table, role=SemanticRole.EXPOSURE,
            resolution_status=_status_for(exposure_col),
            confidence=1.0,
            provenance={"source": "semantic.churn_exposure_col"},
        ))
    censored_col = getattr(semantic, "churn_censored_col", None)
    if censored_col:
        bindings.append(SemanticBinding(
            column=censored_col, table=table, role=SemanticRole.CENSORED_TIME,
            resolution_status=_status_for(censored_col),
            confidence=1.0,
            provenance={"source": "semantic.churn_censored_col"},
        ))
    for confounder in (getattr(semantic, "churn_confounder_cols", None) or []):
        if not confounder:
            continue
        bindings.append(SemanticBinding(
            column=confounder, table=table, role=SemanticRole.EXPLANATORY_VARIABLE,
            resolution_status=_status_for(confounder),
            confidence=0.7,
            provenance={"source": "semantic.churn_confounder_cols"},
        ))

    return SemanticBindingSet(bindings=bindings)
