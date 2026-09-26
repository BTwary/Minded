"""Builds the richer v20-C1 CanonicalSemanticResolution from AA-OS's
existing, authoritative semantic resolution -- the same no-new-heuristic
principle as semantic_binding_builder.py's build_binding_set(), extended to
preserve the resolution story (ambiguity candidates, confounder/exposure/
censoring collections) that a flat SemanticBindingSet cannot represent.

This module does NOT replace build_binding_set() or SemanticBindingSet --
CanonicalSemanticResolution.bindings IS a SemanticBindingSet, built the
same way, and remains the sole source of truth for execution-gate
questions (validate_experiment_against_contract). This module only adds
the richer resolution-story fields alongside it, for consumers
(hypothesis/experiment synthesis) that need more than "is this column
bound and executable?".
"""
from __future__ import annotations

from typing import Any, List, Optional

from packages.analytics_core.src.intelligence.semantic_binding_builder import build_binding_set
from packages.analytics_core.src.intelligence.semantic_builder_vocabulary import (
    CHURN_STATUS_MAP as _CHURN_STATUS_MAP,
    TARGET_STYLE_TASKS as _TARGET_STYLE_TASKS,
)
from packages.schemas.src.semantic_binding import SemanticBindingSet
from packages.schemas.src.semantic_resolution_contract import (
    CanonicalSemanticResolution,
    ResolvedSemanticField,
    SemanticCandidateSet,
    QuestionRoleProposal,
)
from packages.schemas.src.semantic_role import SemanticRole, ResolutionStatus


def _resolve_single_field(
    role: SemanticRole,
    explicit_value: Optional[str],
    candidate_pool: Optional[List[str]],
    source: str,
) -> ResolvedSemanticField:
    """Generalizes predictive_hypothesis.resolve_group_dimension's
    fail-closed 0/1/>1 resolution rule to any single-value semantic field:
    an explicit upstream value always wins; absent that, exactly one
    candidate resolves, zero is UNRESOLVED, and two-or-more is AMBIGUOUS
    (never a first-wins guess).
    """
    if explicit_value:
        return ResolvedSemanticField(
            role=role, value=explicit_value, status=ResolutionStatus.RESOLVED,
            candidates=[explicit_value], provenance={"source": source},
        )
    candidates = list(candidate_pool or [])
    if len(candidates) == 1:
        return ResolvedSemanticField(
            role=role, value=candidates[0], status=ResolutionStatus.RESOLVED,
            candidates=candidates, provenance={"source": source},
        )
    if len(candidates) == 0:
        return ResolvedSemanticField(
            role=role, value=None, status=ResolutionStatus.UNRESOLVED,
            candidates=[], provenance={"source": source},
        )
    return ResolvedSemanticField(
        role=role, value=None, status=ResolutionStatus.AMBIGUOUS,
        candidates=candidates, provenance={"source": source},
    )


def _resolve_outcome_field(semantic: Any, task: str) -> ResolvedSemanticField:
    """Unlike build_binding_set()'s outcome/target branch (which only ever
    creates a binding when churn_event_col is already a resolved,
    non-empty string), this preserves the AMBIGUOUS/UNRESOLVED churn
    outcome cases too -- a churn investigation whose outcome column could
    not be identified is still "churn was the relevant outcome mechanism
    here", not silently reclassified as an ordinary target_metric_col
    question.
    """
    churn_col = getattr(semantic, "churn_event_col", None)
    churn_status_raw = str(getattr(semantic, "churn_event_resolution_status", "RESOLVED")).upper()
    churn_ambiguity = list(getattr(semantic, "churn_event_ambiguity", None) or [])
    churn_outcome_available = getattr(semantic, "churn_outcome_available", True)

    churn_relevant = bool(churn_col) or bool(churn_ambiguity) or (not churn_outcome_available) or churn_status_raw != "RESOLVED"

    if churn_relevant:
        status = _CHURN_STATUS_MAP.get(churn_status_raw, ResolutionStatus.RESOLVED if churn_col else ResolutionStatus.UNRESOLVED)
        if status == ResolutionStatus.RESOLVED and not churn_col:
            # A claimed RESOLVED status with no actual column is not
            # actually resolved -- fail closed rather than trust the flag.
            status = ResolutionStatus.UNRESOLVED
        if status == ResolutionStatus.AMBIGUOUS:
            candidates = churn_ambiguity
        elif status == ResolutionStatus.RESOLVED:
            candidates = [churn_col]
        else:
            candidates = []
        return ResolvedSemanticField(
            role=SemanticRole.OUTCOME,
            value=churn_col if status == ResolutionStatus.RESOLVED else None,
            status=status, candidates=candidates,
            provenance={"source": "semantic.churn_event_col"},
        )

    target_col = getattr(semantic, "target_metric_col", None)
    role = SemanticRole.TARGET if task in _TARGET_STYLE_TASKS else SemanticRole.OUTCOME
    return _resolve_single_field(role, target_col, None, "semantic.target_metric_col")


def build_canonical_semantic_resolution(
    semantic: Any,
    problem_class: str,
    *,
    explanatory_columns: Optional[List[str]] = None,
    bindings: Optional[SemanticBindingSet] = None,
    question_roles: Optional[QuestionRoleProposal] = None,
) -> CanonicalSemanticResolution:
    """Build the v20-C1 CanonicalSemanticResolution for one contract
    version.

    Args:
        semantic: the resolved SemanticResolution (engines/semantic.py).
        problem_class: the compiler task string, same as
            build_binding_set()'s.
        explanatory_columns: forwarded to build_binding_set() unchanged.
        bindings: an already-built SemanticBindingSet to reuse instead of
            building a second one (e.g. controller.py already builds one
            for the persisted contract) -- avoids computing the same flat
            binding list twice for the same semantic resolution.
        question_roles: v20-C4.2.1. The compiler's explicit question-role
            proposal. Only columns the question itself referenced become
            ``requested_explanatory`` (authoritative); compiler columns
            that were merely inferred are kept out of it, and the
            discovered ``secondary_metric`` is never promoted into it.
            When omitted, ``requested_explanatory`` is None (legacy
            behavior, unchanged).
    """
    task = (problem_class or "").upper()
    binding_set = bindings if bindings is not None else build_binding_set(
        semantic, problem_class, explanatory_columns=explanatory_columns,
    )

    outcome = _resolve_outcome_field(semantic, task)
    dimension = _resolve_single_field(
        SemanticRole.GROUPING_DIMENSION,
        getattr(semantic, "group_dimension_col", None),
        getattr(semantic, "available_categorical_cols", None),
        "semantic.group_dimension_col",
    )
    time = _resolve_single_field(
        SemanticRole.TIME_VARIABLE,
        getattr(semantic, "time_col", None),
        None,
        "semantic.time_col",
    )
    secondary_metric = _resolve_single_field(
        SemanticRole.EXPLANATORY_VARIABLE,
        getattr(semantic, "secondary_metric_col", None),
        None,
        "semantic.secondary_metric_col",
    )
    exposure = _resolve_single_field(
        SemanticRole.EXPOSURE,
        getattr(semantic, "churn_exposure_col", None),
        None,
        "semantic.churn_exposure_col",
    )
    censored = _resolve_single_field(
        SemanticRole.CENSORED_TIME,
        getattr(semantic, "churn_censored_col", None),
        None,
        "semantic.churn_censored_col",
    )
    confounders = SemanticCandidateSet(
        role=SemanticRole.EXPLANATORY_VARIABLE,
        purpose="CONFOUNDER",
        columns=list(getattr(semantic, "churn_confounder_cols", None) or []),
        provenance={"source": "semantic.churn_confounder_cols"},
    )

    requested_explanatory = None
    if question_roles is not None:
        explicit = question_roles.explicit_explanatory_columns()
        # The question's own outcome is never its own predictor, even if
        # canonical resolved a different outcome column than the compiler.
        outcome_col = outcome.value if outcome.is_resolved() else None
        explicit = [c for c in explicit if c != outcome_col]
        requested_explanatory = SemanticCandidateSet(
            role=SemanticRole.EXPLANATORY_VARIABLE,
            purpose="REQUESTED_PREDICTOR",
            columns=explicit,
            provenance={
                "source": "compiler.explanatory_columns",
                "basis": "question_referenced_columns",
                "inferred_not_authoritative": question_roles.inferred_explanatory_columns(),
            },
        )

    return CanonicalSemanticResolution(
        bindings=binding_set,
        outcome=outcome,
        dimension=dimension,
        time=time,
        secondary_metric=secondary_metric,
        exposure=exposure,
        censored=censored,
        confounders=confounders,
        available_categorical_candidates=list(getattr(semantic, "available_categorical_cols", None) or []),
        requested_explanatory=requested_explanatory,
    )
