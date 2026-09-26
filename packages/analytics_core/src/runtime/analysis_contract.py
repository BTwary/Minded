"""
AnalysisContract: Versioned, central analytical contract for AA-OS investigations.

The AnalysisContract is the authoritative artifact of an investigation, defining:
- Question interpretation & confidence
- Semantic resolution & grain binding
- Data readiness constraints & method permissions
- Formal estimand, claim type, and claim ceiling
- Competing hypotheses & selected method
- Experiment execution plan & budget
- Evidence requirements & stopping conditions

Every replanning round creates a new immutable version (v1 -> v2 -> ...),
preserving the complete reasoning trail.
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class ContractConstraintRecord:
    check_id: str
    severity: str
    finding: str
    affected_methods: List[str]
    required_action: str
    can_proceed: bool


@dataclass
class AnalysisContractVersionData:
    contract_id: str
    investigation_id: str
    version: int
    parent_version: Optional[int]
    created_at_iso: str
    trigger_reason: str

    # Question & scope
    original_question: str
    normalized_question: str
    question_confidence: float
    ambiguity_notes: List[str]

    # Semantic bindings
    primary_dataset: str
    grain: str
    unit_of_analysis: List[str]
    target_column: Optional[str]
    explanatory_columns: List[str]
    time_column: Optional[str]
    group_dimension: Optional[str]
    population_scope: str

    # Problem definition
    problem_class: str
    estimand: str
    claim_type: str
    claim_ceiling: str
    assumptions: List[Dict[str, Any]]

    # Method & readiness
    candidate_methods: List[str]
    selected_method: str
    method_rejection_reasons: Dict[str, str]
    blocking_constraints: List[Dict[str, Any]]
    active_warnings: List[str]

    # Execution policy
    planned_experiment_codes: List[str]
    max_experiments: int
    evidence_requirements: List[str]
    verification_required: bool
    adversarial_required: bool
    stopping_criteria: List[str]

    # Epistemic tier & semantic authority
    target_epistemic_source: Optional[str] = None
    metric_aggregation_type: Optional[str] = None
    grain_proven: bool = False
    epistemic_manifest: Dict[str, Any] = field(default_factory=dict)
    # v28 SemanticBindingSet authority: canonical bindings carry full provenance
    # and drive target/explanatory/time/group compatibility projections
    semantic_bindings: Optional[List[Dict[str, Any]]] = None

    @property
    def semantic_binding_set(self) -> Any:
        from packages.schemas.src.semantic_binding import SemanticBindingSet
        return SemanticBindingSet.from_dict_list(self.semantic_bindings or [])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> AnalysisContractVersionData:
        import dataclasses
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


class AnalysisContractManager:
    """Manages versioned analytical contracts for an investigation."""

    @staticmethod
    def create_v1(
        investigation_id: str,
        question: str,
        problem_class: str,
        estimand: str,
        claim_type: str,
        claim_ceiling: str,
        primary_dataset: str,
        grain: str,
        unit_of_analysis: List[str],
        target_column: Optional[str],
        explanatory_columns: List[str],
        time_column: Optional[str],
        group_dimension: Optional[str],
        selected_method: str,
        candidate_methods: List[str],
        planned_experiment_codes: List[str],
        max_experiments: int = 5,
        population_scope: str = "entire_dataset",
        question_confidence: float = 1.0,
        ambiguity_notes: Optional[List[str]] = None,
        assumptions: Optional[List[Dict[str, Any]]] = None,
        method_rejection_reasons: Optional[Dict[str, str]] = None,
        blocking_constraints: Optional[List[Dict[str, Any]]] = None,
        active_warnings: Optional[List[str]] = None,
        evidence_requirements: Optional[List[str]] = None,
        verification_required: bool = True,
        adversarial_required: bool = True,
        stopping_criteria: Optional[List[str]] = None,
        target_epistemic_source: Optional[str] = None,
        metric_aggregation_type: Optional[str] = None,
        grain_proven: bool = False,
        epistemic_manifest: Optional[Dict[str, Any]] = None,
        semantic_bindings: Optional[List[Dict[str, Any]]] = None,
    ) -> AnalysisContractVersionData:
        if semantic_bindings:
            from packages.schemas.src.semantic_binding import SemanticBindingSet
            sbs = SemanticBindingSet.from_dict_list(semantic_bindings)
            if target_column is None:
                target_column = sbs.projected_target_column()
            if not explanatory_columns:
                explanatory_columns = sbs.projected_explanatory_columns()
            if time_column is None:
                time_column = sbs.projected_time_column()
            if group_dimension is None:
                group_dimension = sbs.projected_group_dimension()

        return AnalysisContractVersionData(
            contract_id=f"CTR-{investigation_id}",
            investigation_id=investigation_id,
            version=1,
            parent_version=None,
            created_at_iso=datetime.now(timezone.utc).isoformat(),
            trigger_reason="INITIAL_CONTRACT_COMMIT",
            original_question=question,
            normalized_question=question.strip().lower(),
            question_confidence=question_confidence,
            ambiguity_notes=ambiguity_notes or [],
            primary_dataset=primary_dataset,
            grain=grain,
            unit_of_analysis=unit_of_analysis,
            target_column=target_column,
            explanatory_columns=explanatory_columns,
            time_column=time_column,
            group_dimension=group_dimension,
            population_scope=population_scope,
            problem_class=problem_class,
            estimand=estimand,
            claim_type=claim_type,
            claim_ceiling=claim_ceiling,
            assumptions=assumptions or [],
            candidate_methods=candidate_methods,
            selected_method=selected_method,
            method_rejection_reasons=method_rejection_reasons or {},
            blocking_constraints=blocking_constraints or [],
            active_warnings=active_warnings or [],
            planned_experiment_codes=planned_experiment_codes,
            max_experiments=max_experiments,
            evidence_requirements=evidence_requirements or ["dual_engine_verification"],
            verification_required=verification_required,
            adversarial_required=adversarial_required,
            stopping_criteria=stopping_criteria or ["ANSWER_ESTABLISHED", "UNCERTAINTY_STAGNATED"],
            target_epistemic_source=target_epistemic_source,
            metric_aggregation_type=metric_aggregation_type,
            grain_proven=grain_proven,
            epistemic_manifest=epistemic_manifest or {},
            semantic_bindings=semantic_bindings,
        )

    @staticmethod
    def spawn_replan(
        previous: AnalysisContractVersionData,
        trigger_reason: str,
        updated_experiment_codes: Optional[List[str]] = None,
        new_hypotheses_or_assumptions: Optional[List[Dict[str, Any]]] = None,
        new_active_warnings: Optional[List[str]] = None,
        adjusted_claim_ceiling: Optional[str] = None,
        new_semantic_bindings: Optional[List[Dict[str, Any]]] = None,
    ) -> AnalysisContractVersionData:
        """Spawns an immutable next version (e.g. v2) preserving replanning lineage."""
        semantic_bindings = new_semantic_bindings if new_semantic_bindings is not None else previous.semantic_bindings
        target_col = previous.target_column
        expl_cols = list(previous.explanatory_columns)
        time_col = previous.time_column
        grp_dim = previous.group_dimension
        if new_semantic_bindings is not None:
            from packages.schemas.src.semantic_binding import SemanticBindingSet
            sbs = SemanticBindingSet.from_dict_list(new_semantic_bindings)
            target_col = sbs.projected_target_column() or target_col
            expl_cols = sbs.projected_explanatory_columns() or expl_cols
            time_col = sbs.projected_time_column() or time_col
            grp_dim = sbs.projected_group_dimension() or grp_dim

        return AnalysisContractVersionData(
            contract_id=previous.contract_id,
            investigation_id=previous.investigation_id,
            version=previous.version + 1,
            parent_version=previous.version,
            created_at_iso=datetime.now(timezone.utc).isoformat(),
            trigger_reason=trigger_reason,
            original_question=previous.original_question,
            normalized_question=previous.normalized_question,
            question_confidence=previous.question_confidence,
            ambiguity_notes=list(previous.ambiguity_notes),
            primary_dataset=previous.primary_dataset,
            grain=previous.grain,
            unit_of_analysis=list(previous.unit_of_analysis),
            target_column=target_col,
            explanatory_columns=expl_cols,
            time_column=time_col,
            group_dimension=grp_dim,
            population_scope=previous.population_scope,
            problem_class=previous.problem_class,
            estimand=previous.estimand,
            claim_type=previous.claim_type,
            claim_ceiling=adjusted_claim_ceiling or previous.claim_ceiling,
            assumptions=list(previous.assumptions) + (new_hypotheses_or_assumptions or []),
            candidate_methods=list(previous.candidate_methods),
            selected_method=previous.selected_method,
            method_rejection_reasons=dict(previous.method_rejection_reasons),
            blocking_constraints=list(previous.blocking_constraints),
            active_warnings=list(previous.active_warnings) + (new_active_warnings or []),
            planned_experiment_codes=updated_experiment_codes if updated_experiment_codes is not None else list(previous.planned_experiment_codes),
            max_experiments=previous.max_experiments,
            evidence_requirements=list(previous.evidence_requirements),
            verification_required=previous.verification_required,
            adversarial_required=previous.adversarial_required,
            stopping_criteria=list(previous.stopping_criteria),
            target_epistemic_source=previous.target_epistemic_source,
            metric_aggregation_type=previous.metric_aggregation_type,
            grain_proven=previous.grain_proven,
            epistemic_manifest=dict(previous.epistemic_manifest),
            semantic_bindings=semantic_bindings,
        )
