from datetime import datetime, timezone
import hashlib
import json
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    elif hasattr(obj, "isoformat"):
        return obj.isoformat()
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    elif pd.isna(obj):
        return None
    return str(obj) if type(obj).__module__ != "builtins" else obj


from apps.api.src.core.database import SessionLocal
from apps.api.src.services.dataset_service import DatasetService
from apps.api.src.models.entities import (
    Investigation,
    InvestigationObjective,
    Hypothesis,
    Experiment,
    Observation,
    Evidence,
    EvidenceVerification,
    BeliefUpdate,
    InvestigationVerdict,
    InvestigationGraphEdge,
    DecisionRecommendationRecord,
    Prediction,
    InvestigationContract,
    InvestigationDataReadiness,
    ClaimGateDecision,
    gen_uuid,
)
from packages.analytics_core.src.engines.variable_resolution_gate import (
    detect_unresolved_requested_variables,
    detect_rate_question_with_no_column_reference,
)
from packages.analytics_core.src.intelligence.hypothesis_identity import compute_semantic_identity
from packages.analytics_core.src.runtime.hypothesis_persistence import get_or_create_hypothesis_row
from packages.analytics_core.src.execution.state_machine import (
    InvestigationState,
    ExecutionState,
    FailureTaxonomy,
    VerificationStatus,
    AnalyticalPhase,
    transition_investigation,
    transition_execution,
)
from packages.analytics_core.src.execution.checkpointer import (
    InvestigationCheckpointer,
    DatasetUnavailableError,
)
from packages.analytics_core.src.engines.dataset_provider import (
    BaseDatasetProvider,
    DatabaseDatasetProvider,
    InvestigationDataContext,
)
from packages.analytics_core.src.engines.intent import IntentEngine
from packages.analytics_core.src.engines.method_selection import (
    MethodSelectionEngine, MethodRegistry, ProblemClass, CANONICAL_TASK_AUTHORITY,
)
from packages.analytics_core.src.engines.temporal import TemporalResolver
from packages.analytics_core.src.engines.semantic import SemanticEngine, SemanticResolution
from packages.analytics_core.src.engines.execution_provider import (
    BaseExecutionProvider,
    DuckDBExecutionProvider,
    QueryExecutionTimeoutError,
)
from packages.analytics_core.src.engines.evidence import EvidenceEngine
from packages.analytics_core.src.engines.evidence_ledger import EvidenceLedger
from packages.analytics_core.src.graph.evidence_identity import compute_evidence_identity, canonical_dataset_identity
from packages.analytics_core.src.engines.verification import VerificationEngine
from packages.analytics_core.src.relational.join_safety import assess_declared_join_hops, SafetyStatus
from packages.analytics_core.src.relational.plan_serialization import plan_to_dict
from packages.analytics_core.src.engines.belief import BeliefEngine, compute_shannon_entropy
from packages.analytics_core.src.engines.stopping import StoppingEngine, StoppingDecision
from packages.analytics_core.src.engines.human_decision import HumanDecisionController
from packages.analytics_core.src.engines.verdict import VerdictEngine
from packages.analytics_core.src.engines.provenance import ProvenanceEngine
from packages.analytics_core.src.engines.analyst_answer import build_analyst_result
from packages.analytics_core.src.engines.decision_utility import DecisionUtilityEngine
from packages.analytics_core.src.engines.assumption_ledger import AssumptionLedgerEngine
from packages.analytics_core.src.profiling.data_quality_gate import DataQualityGate
from packages.analytics_core.src.profiling.data_quality_decision import assess_downstream_impact, DataQualityDecision
from packages.analytics_core.src.data.dataset_change_assessment import (
    DatasetChangeAssessment,
    assess_dataset_change,
    infer_stable_key_columns,
)
from packages.analytics_core.src.causal.identifiability_gate import CausalIdentifiabilityGate
from packages.analytics_core.src.semantic.metric_semantics import MetricDefinition, MetricSemanticsResolver
from packages.analytics_core.src.semantic.missingness_sensitivity import (
    MissingnessSensitivityEngine,
    ROBUST as MISSINGNESS_ROBUST,
    SENSITIVE as MISSINGNESS_SENSITIVE,
    UNIDENTIFIABLE as MISSINGNESS_UNIDENTIFIABLE,
    INSUFFICIENT_EVIDENCE as MISSINGNESS_INSUFFICIENT_EVIDENCE,
)
from packages.analytics_core.src.runtime.state import InvestigationStateManager
from packages.analytics_core.src.runtime.state_reconstruction import (
    reconstruct_investigation_state,
    classify_evidence_for_hypothesis,
)

from packages.schemas.src.analysis import (
    AggregationType,
    CausalIdentifiabilityStatus,
    DecisionRecommendation,
    EpistemicClaimType,
    ExpectedUtilityCalculation,
    FirstClassExperiment,
    GrainPreservationStatus,
    GrainRef,
    MetricRef,
    ObjectiveType,
    TypedAnalyticalIntent,
    VariableRef,
    CalculationTraceSchema,
)
from packages.analytics_core.src.validation.ir_validator import AnalyticalIRValidator
from packages.analytics_core.src.sql.relational_compiler import RelationalCompiler

from packages.analytics_core.src.intelligence.predictive_hypothesis import HypothesisSynthesizer, PredictiveHypothesis, resolve_group_dimension
from packages.analytics_core.src.intelligence.experiment_synthesizer import (
    ExperimentSynthesizer,
    CandidateExperiment,
    UncertaintyState,
)
from packages.analytics_core.src.intelligence.semantic_binding_builder import build_binding_set
from packages.analytics_core.src.intelligence.experiment_contract_validation import (
    AnalyticalContractSnapshot,
    detect_plan_final_contract_conflicts,
    validate_experiment_against_contract,
)
from packages.schemas.src.semantic_binding import SemanticBindingSet
from packages.schemas.src.semantic_role import ExperimentRole
from packages.analytics_core.src.intelligence.eig_optimizer import EIGOptimizer
from packages.analytics_core.src.intelligence.prediction_engine import (
    PredictionSynthesizer,
    PredictionEvaluator,
    StructuredPrediction,
)
from packages.analytics_core.src.intelligence.hypothesis_revision import HypothesisRevisionEngine
from packages.analytics_core.src.intelligence.adversarial_attacker import AdversarialAttacker
from packages.analytics_core.src.intelligence.multiverse_engine import MultiverseEngine
from packages.analytics_core.src.intelligence.epistemic_calibration import EpistemicCalibrationEngine
from packages.analytics_core.src.statistics.probability_calibration import load_calibration_profile
from packages.analytics_core.src.intelligence.universal_question_planner import UniversalQuestionCompiler, EvidenceRequirement
from packages.analytics_core.src.intelligence.complex_objective_planner import plan_compound_question
from packages.schemas.src.semantic_resolution_contract import QuestionRoleProposal
from packages.analytics_core.src.intelligence.universal_specialized import run_specialized_analysis, run_predictive_risk_analysis, run_forecast_analysis
from packages.analytics_core.src.intelligence.transition import ScientificTransitionService
from packages.analytics_core.src.intelligence.state_validation import validate_state
from packages.analytics_core.src.governance.recommendation_grounding import evaluate_recommendation_grounding
from packages.analytics_core.src.governance.claim_gate import (
    admit_positive_claim,
    ClaimAdmission,
    ClaimType as GateClaimType,
    DesignStatus as GateDesignStatus,
    EvidenceLevel as GateEvidenceLevel,
    evaluate_gate,
)
from packages.analytics_core.src.runtime.scientific_state_snapshot import build_scientific_state_snapshot
from packages.analytics_core.src.intelligence.contract_authority import (
    finalize_contract, load_final_contract, set_phase, supersede_and_replan,
)
from packages.analytics_core.src.intelligence.analytical_identity import (
    AnalyticalIdentityError, FinalAnalyticalContract, hypothesis_identity_for, stamp_experiment_identity,
)
from packages.analytics_core.src.statistics.churn_estimands import analyze_churn_identifiability, ChurnVerdict
from packages.shared.src.constants import MAX_INVESTIGATION_RUNTIME_SECONDS, MAX_INVESTIGATION_EXPERIMENTS


def _is_binary_mirror_hypothesis(counter_hyp: Any, tested_hyp: Any) -> bool:
    """True only when counter_hyp is the direct binary mirror of tested_hyp
    (same target metric/dimension/value) -- e.g. "no material association"
    vs "revenue and cost are correlated". Structurally distinct alternative
    hypotheses (a different mechanism, a different confound entirely, such
    as churn's stratification-confound or exposure-imbalance hypotheses)
    must NOT be auto-credited with evidence they were never actually tested
    by -- each needs its own dedicated experiment. Without this scoping, a
    single verified experiment against one hypothesis could prematurely
    satisfy the "every counter hypothesis has direct evidence" stopping
    gate for unrelated counter hypotheses, skipping real confound-check
    experiments the investigation still needed to run.
    """
    if counter_hyp is None or tested_hyp is None:
        return False
    return (
        getattr(counter_hyp, "target_metric", None) == getattr(tested_hyp, "target_metric", None)
        and getattr(counter_hyp, "target_dimension", None) == getattr(tested_hyp, "target_dimension", None)
    )


def compute_task_compatible_candidates(
    candidates: List["CandidateExperiment"],
    task: str,
    semantic: "SemanticResolution",
) -> Tuple[List["CandidateExperiment"], Optional[Dict[str, Any]]]:
    """P0-B (controller hardening): the single authoritative rule for
    filtering candidate experiments down to the ones compatible with the
    selected analytical task.

    Returns (filtered_candidates, diagnostic). filtered_candidates is NEVER
    silently widened back to the full input when filtering eliminates
    everything -- an analytically incompatible candidate (e.g. a
    correlation candidate for a FORECAST task) must never be restored and
    executed merely because nothing matched. When filtering does eliminate
    every candidate, diagnostic is a NO_COMPATIBLE_EXPERIMENT_CANDIDATE
    payload the caller should record to provenance before replanning or
    stopping; otherwise diagnostic is None.

    Extracted as a standalone, independently-testable function (rather than
    the previous in-loop closure) so this invariant can be unit tested
    without running a full investigation.
    """
    if not candidates:
        return [], None
    keep_prefixes = None
    if task == "ASSOCIATION":
        # For a churn association question, do not spend the first
        # experiment on baseline uniformity/censoring diagnostics.
        # EXP-COND-* (the Simpson's-paradox conditional adversarial
        # candidate built by build_simpsons_conditional_candidate) is always
        # allowed once it exists -- its own construction is already gated on
        # a genuinely resolved primary dimension and a detected paradox, so
        # it must not be silently stripped out by this task-family filter
        # the way "filtered or candidates" used to accidentally preserve it.
        # EXP-CORR is the general (non-churn-specific) Pearson/Spearman
        # correlation candidate ExperimentSynthesizer.synthesize_correlation_experiment
        # emits for any ASSOCIATION hypothesis -- it is a legitimate member
        # of this task family and was previously only reachable through the
        # same "filtered or candidates" fallback bug fixed here; excluding
        # it would silently break non-churn association questions once that
        # bug was fixed, so it is included explicitly instead.
        keep_prefixes = ("EXP-CHURN-CRUDE", "EXP-CHURN-STRAT", "EXP-CHURN-EXPOSURE", "EXP-COND", "EXP-CORR")
        if not getattr(semantic, "churn_confounder_cols", None):
            keep_prefixes = ("EXP-CHURN-CRUDE", "EXP-CHURN-EXPOSURE", "EXP-COND", "EXP-CORR")
        if not getattr(semantic, "churn_exposure_col", None):
            keep_prefixes = tuple(x for x in keep_prefixes if x != "EXP-CHURN-EXPOSURE")
    elif task == "FORECAST":
        keep_prefixes = ("EXP-FORECAST",)
    elif task == "CAUSAL":
        keep_prefixes = ("EXP-CAUSAL", "EXP-IV", "EXP-DID", "EXP-RDD")
    if keep_prefixes is None:
        return candidates, None
    filtered = [c for c in candidates if any(str(c.code).startswith(prefix) for prefix in keep_prefixes)]
    if filtered:
        return filtered, None
    diagnostic = {
        "diagnostic": "NO_COMPATIBLE_EXPERIMENT_CANDIDATE",
        "analytical_task": task,
        "allowed_prefixes": list(keep_prefixes),
        "candidate_codes": [str(c.code) for c in candidates],
        "rejected_candidate_codes": [str(c.code) for c in candidates],
        "reason": (
            f"No candidate experiment code matched the compatible method-family "
            f"prefixes {list(keep_prefixes)} for task {task}; incompatible candidates "
            f"were not restored/executed."
        ),
        "replanning_status": "CALLER_MUST_REPLAN_OR_STOP",
    }
    return [], diagnostic


def build_simpsons_conditional_candidate(
    semantic: "SemanticResolution",
    conf_dim: str,
    leading_hypothesis_code: str,
    target_metric_col_label: Optional[str] = None,
) -> Tuple[Optional["CandidateExperiment"], Optional[Dict[str, Any]]]:
    """P0-B (controller hardening): construct the conditional adversarial
    (Simpson's-paradox) experiment ONLY from a genuinely resolved primary
    grouping dimension and resolved metric aggregation semantics.

    Returns (candidate_or_None, diagnostic_or_None). Exactly one of the two
    is non-None:
      - metric aggregation unresolved -> (None, {"status": "UNRESOLVED_METRIC_SEMANTICS", ...})
      - primary dimension UNRESOLVED  -> (None, {"status": "UNRESOLVED_PRIMARY_DIMENSION", ...})
      - primary dimension AMBIGUOUS   -> (None, {"status": "AMBIGUOUS_PRIMARY_DIMENSION", ...})
      - both resolved                 -> (CandidateExperiment(...), None)

    Never falls back to a literal placeholder dimension (e.g. "segment")
    and never silently picks one of several ambiguous candidates.
    Extracted as a standalone function so this invariant is independently
    unit-testable without running a full investigation loop.
    """
    primary_dim, primary_dim_status, primary_dim_candidates = resolve_group_dimension(semantic)
    _metric_def = semantic.metric_definition
    if _metric_def is None or getattr(_metric_def, "aggregation_type", None) is None:
        return None, {
            "confounding_dimension": conf_dim,
            "type": "simpsons_paradox",
            "status": "UNRESOLVED_METRIC_SEMANTICS",
            "reason": "Conditional adversarial test could not be constructed without a resolved metric aggregation.",
        }
    if primary_dim_status != "RESOLVED":
        return None, {
            "confounding_dimension": conf_dim,
            "type": "simpsons_paradox",
            "status": f"{primary_dim_status}_PRIMARY_DIMENSION",
            "candidate_dimensions": primary_dim_candidates,
            "reason": (
                "Conditional adversarial test could not be constructed: the primary "
                f"grouping dimension is {primary_dim_status} (candidates: "
                f"{primary_dim_candidates}); the controller does not choose one "
                "implicitly and did not execute a conditional grouping query."
            ),
        }
    _metric_expr = MetricSemanticsResolver.sql_aggregation_expression(_metric_def, "total_metric")
    _agg_str = _metric_def.aggregation_type.value.upper()
    candidate = CandidateExperiment(
        code=f"EXP-COND-{conf_dim.upper()}",
        target_hypothesis_code=leading_hypothesis_code,
        hypothesis_ids=[leading_hypothesis_code],
        tool_name="duckdb_sql",
        # Session 8 (DEFECT-023): the multi-row-result guard in execution_provider
        # rejects any grouped result without an ORDER BY, so this experiment
        # failed on every run (invisibly to the analyst). Same deterministic
        # ordering the synthesizer's EXP-COND variant already uses.
        query_sql=f"SELECT {primary_dim}, {conf_dim}, {_metric_expr} FROM data_table GROUP BY {primary_dim}, {conf_dim} ORDER BY total_metric DESC, {primary_dim}, {conf_dim}",
        description=f"Conditional test on {conf_dim} to address adversarial Simpson's paradox ({target_metric_col_label or semantic.target_metric_col} aggregated as {_agg_str}).",
        aggregation_type=_agg_str,
        discriminating_power=0.99,
        discrimination_value=0.99,
        estimated_cost=1.4,
        reliability_weight=0.96,
        decision_relevance=0.95,
        target_dimension=f"{primary_dim}_{conf_dim}",
        target_metric=semantic.target_metric_col,
        metrics=[semantic.target_metric_col],
        dimensions=[primary_dim, conf_dim],
        adversarial_value=0.95,
        # v20-B1: this is a Simpson's-paradox counter-hypothesis challenge by
        # construction -- always ADVERSARIAL in intent, regardless of
        # prediction linkage. Tag it explicitly here rather than relying on
        # the gate-time UNASSIGNED backfill (which had no way to distinguish
        # this from an ordinary PRIMARY/SUPPORTING candidate and would
        # mis-tag it as PRIMARY).
        experiment_role=ExperimentRole.ADVERSARIAL.value,
        target_uncertainty=f"Simpson's paradox challenge on {conf_dim}",
        provenance={"attack_type": "simpsons_paradox", "confounding_dimension": conf_dim},
    )
    return candidate, None


class InvestigationController:
    """Canonical brain of AA-OS executing the recursive autonomous, predictive experimentation and verification loop."""

    # Phase 11 (Metric Semantics Engine): full string->AggregationType mapping,
    # replacing the pre-Phase-11 binary "SUM if SUM else MEAN" collapse that
    # silently discarded COUNT/COUNT_DISTINCT/RATE/RATIO/PROPORTION/WEIGHTED_MEAN
    # semantics whenever an experiment's own aggregation_type wasn't literally "SUM".
    _AGG_STR_TO_TYPE = {
        "SUM": AggregationType.SUM,
        "MEAN": AggregationType.MEAN,
        "AVG": AggregationType.MEAN,
        "COUNT": AggregationType.COUNT,
        "COUNT_DISTINCT": AggregationType.COUNT_DISTINCT,
        "RATE": AggregationType.RATE,
        "RATIO": AggregationType.RATIO,
        "PROPORTION": AggregationType.PROPORTION,
        "WEIGHTED_MEAN": AggregationType.WEIGHTED_MEAN,
        "MEDIAN": AggregationType.MEDIAN,
        "MIN": AggregationType.MIN,
        "MAX": AggregationType.MAX,
    }

    def __init__(
        self,
        session_factory=SessionLocal,
        checkpointer: Optional[InvestigationCheckpointer] = None,
        dataset_provider: Optional[BaseDatasetProvider] = None,
        execution_provider: Optional[BaseExecutionProvider] = None,
        semantic_engine: Optional[SemanticEngine] = None,
        ai_provider: Optional[Any] = None,
    ):
        self.session_factory = session_factory
        self.checkpointer = checkpointer or InvestigationCheckpointer(session_factory=session_factory)
        self.dataset_provider = dataset_provider or DatabaseDatasetProvider(session_factory=session_factory)
        self.execution_provider = execution_provider or DuckDBExecutionProvider()
        self.semantic_engine = semantic_engine or SemanticEngine()
        self.ai_provider = ai_provider
        self.ir_validator = AnalyticalIRValidator()
        self.compiler = RelationalCompiler()
        self.causal_gate = CausalIdentifiabilityGate()

    def compile_experiment_intent(
        self,
        exp: CandidateExperiment,
        semantic: SemanticResolution,
        investigation_id: str,
    ) -> str:
        """Compile CandidateExperiment through TypedAnalyticalIntent and RelationalCompiler."""
        dim_ref = [VariableRef(table=semantic.primary_dataset_name, column=exp.target_dimension)] if exp.target_dimension else []
        grain_keys = (
            semantic.world_model.verified_grains.get(semantic.primary_dataset_name, [semantic.target_metric_col])
            if semantic.world_model and hasattr(semantic.world_model, "verified_grains")
            else [semantic.target_metric_col]
        )
        resolved_aggregation = self._AGG_STR_TO_TYPE.get(str(exp.aggregation_type).upper())
        if resolved_aggregation is None:
            raise ValueError(f"Analytical compilation refused unresolved aggregation semantics: {exp.aggregation_type!r}")

        exp_intent = TypedAnalyticalIntent(
            intent_id=f"EXP_INT_{exp.code}_{investigation_id[:6]}",
            objective=ObjectiveType.COMPARE if exp.target_dimension else ObjectiveType.DESCRIBE,
            target_metric=MetricRef(
                name=exp.target_metric or semantic.target_metric_col,
                table=semantic.primary_dataset_name,
                column=exp.target_metric or semantic.target_metric_col,
                aggregation=resolved_aggregation,
            ),
            unit_of_analysis=GrainRef(
                table=semantic.primary_dataset_name,
                keys=grain_keys,
            ),
            dimensions=dim_ref,
            source_question=exp.description,
        )
        val_res = self.ir_validator.validate(exp_intent, semantic.world_model)
        if not val_res.is_valid:
            err_msgs = "; ".join([e.message for e in val_res.errors])
            raise ValueError(f"Analytical IR Validation Failed for {exp.code}: {err_msgs}")

        compiled = self.compiler.compile(exp_intent)
        if not compiled or not compiled.sql_query:
            raise ValueError(f"Analytical Compilation Failed for {exp.code}: Empty physical plan.")

        return compiled.sql_query

    def execute_investigation(
        self,
        investigation_id: str,
        worker_id: str,
        attempt: int = 1,
        cancellation_check: Optional[Callable[[], bool]] = None,
    ) -> bool:
        """Execute the autonomous multi-turn recursive scientific investigation loop."""
        exec_id = self.checkpointer.get_or_create_execution(
            investigation_id=investigation_id,
            worker_id=worker_id,
            attempt=attempt,
        )
        investigation_started_monotonic = time.monotonic()
        max_runtime_seconds = int(os.getenv("AAOS_MAX_INVESTIGATION_RUNTIME_SECONDS", str(MAX_INVESTIGATION_RUNTIME_SECONDS)))

        # Scope/sampling authority ledger: every executed experiment appends its
        # analysis_row_count / full_scope / sampling_policy / sampling_reason /
        # sampling_method record here (see run_selected_experiment below), and the
        # final verdict's scope summary is derived from this list. Must exist
        # before any nested function in this method can reference it.
        experiment_scope_records: List[Dict[str, Any]] = []

        def _stable_hypothesis_order(hypotheses):
            """Return hypotheses in deterministic scientific order.

            Controller behavior must not depend on SQL row order, insertion order,
            event arrival order, or dictionary ordering. Canonical identity is the
            primary stable key; hypothesis code is the fallback for legacy rows.

            Defined near the top of execute_investigation (rather than further
            down, where it originally lived) because the restart/reconstruction
            branch calls this before reaching that later point in the method
            body; Python does not hoist nested-function definitions, so calling
            it before its `def` executes raises NameError. See
            AAOS_FORENSIC_AUDIT_2026-09-09.md, P0 finding #2.
            """
            return sorted(
                list(hypotheses or []),
                key=lambda h: (
                    str(getattr(h, "canonical_identity", "") or ""),
                    str(getattr(h, "hypothesis_code", "") or ""),
                ),
            )

        def _complete_inconclusive(reason: str, *, failure_class: str = FailureTaxonomy.DATA_QUALITY_FAILURE) -> bool:
            """Terminate a non-executable investigation honestly without marking it FAILED."""
            message = f"Investigation inconclusive: {reason}"
            self.checkpointer.record_event(
                investigation_id=investigation_id, execution_id=exec_id,
                event_type="investigation.inconclusive_completion",
                payload={"reason": reason, "failure_class": failure_class},
            )
            self.checkpointer.complete_execution(
                investigation_id=investigation_id, execution_id=exec_id,
                final_status="COMPLETED", error_message=message,
            )
            with self.session_factory() as terminal_session:
                terminal_inv = terminal_session.query(Investigation).filter(Investigation.id == investigation_id).first()
                if terminal_inv:
                    terminal_inv.status = InvestigationState.COMPLETED
                    terminal_inv.verdict_type = "INCONCLUSIVE"
                    terminal_inv.direct_answer = message
                    terminal_inv.main_finding = reason
                    terminal_inv.confidence_score = 0.0
                    terminal_session.commit()
            return True

        def _runtime_budget_exceeded(stage: str) -> bool:
            elapsed = time.monotonic() - investigation_started_monotonic
            if elapsed >= max_runtime_seconds:
                try:
                    self.checkpointer.record_event(
                        investigation_id=investigation_id,
                        event_type="investigation.runtime_budget_exceeded",
                        payload={"stage": stage, "elapsed_seconds": round(elapsed, 3), "budget_seconds": max_runtime_seconds},
                        execution_id=exec_id,
                    )
                except Exception:
                    pass
                return True
            return False

        with self.session_factory() as session:
            inv = session.query(Investigation).filter(Investigation.id == investigation_id).first()
            if not inv:
                raise ValueError(f"Investigation {investigation_id} not found.")

            if inv.status == InvestigationState.CANCEL_REQUESTED:
                transition_investigation(db=session, investigation_id=investigation_id, to_state=InvestigationState.CANCELLED, actor=worker_id)
                session.commit()
                return False

            if inv.status == InvestigationState.PLANNED:
                transition_investigation(db=session, investigation_id=investigation_id, to_state=InvestigationState.QUEUED, actor=worker_id)
            if inv.status in (InvestigationState.QUEUED, InvestigationState.CLAIMED, InvestigationState.PLANNED):
                transition_investigation(db=session, investigation_id=investigation_id, to_state=InvestigationState.RUNNING, actor=worker_id)
                session.commit()

            question = inv.question
            project_id = inv.project_id
            parent_user_id = inv.user_id
            requested_dataset_ids = list(inv.requested_dataset_ids_json) if inv.requested_dataset_ids_json else None

        # 1. Dataset Context Acquisition
        if cancellation_check and cancellation_check():
            return False

        try:
            data_context = self.dataset_provider.acquire_context(project_id, dataset_ids=requested_dataset_ids)
            if requested_dataset_ids and data_context.unavailable_requested_ids:
                raise DatasetUnavailableError(
                    "Requested dataset scope could not be fully loaded. "
                    f"Unavailable dataset IDs: {sorted(set(data_context.unavailable_requested_ids))}. "
                    "AA-OS refuses to continue with a silently reduced dataset population."
                )
            if data_context.load_errors:
                raise DatasetUnavailableError(
                    "Dataset scope contains load failures. "
                    f"Load errors: {data_context.load_errors}. "
                    "AA-OS refuses to continue until the complete requested scope is available."
                )
            if getattr(data_context, "scope_errors", None):
                raise DatasetUnavailableError(
                    "Dataset scope is ambiguous or unsafe. "
                    f"Scope errors: {data_context.scope_errors}. "
                    "AA-OS refuses to continue rather than silently collapsing selected datasets."
                )
            if data_context.is_empty:
                raise DatasetUnavailableError(
                    f"No datasets available for project {project_id}. "
                    "Investigation aborted under AA-OS Zero-Synthetic-Data Invariant."
                )
        except Exception as e:
            self.checkpointer.fail_execution(
                investigation_id=investigation_id,
                execution_id=exec_id,
                error_message=f"Failed to acquire dataset context: {str(e)}",
                failure_class=FailureTaxonomy.RESOURCE_UNAVAILABLE,
            )
            return False

        # 1.4 Centralized pre-analysis variable-resolution gate. Runs before
        # semantic/intent resolution and before any task-family synthesizer
        # gets a chance to silently bind a fallback column, so no family
        # (correlation, rate/proportion, trend, ranking) can substitute a
        # different variable and report a confident answer to a question
        # that was never asked. See variable_resolution_gate.py for the
        # detection rule and its deliberately narrow scope; churn-flavored
        # questions are excluded here because they already have their own
        # dedicated fail-closed path further below in this method.
        _all_available_columns: set = set()
        for _ds_df in data_context.datasets_map.values():
            _all_available_columns.update(str(c) for c in getattr(_ds_df, "columns", []))
        _missing_requested_vars = detect_unresolved_requested_variables(question, _all_available_columns)
        if not _missing_requested_vars and detect_rate_question_with_no_column_reference(question, _all_available_columns):
            _missing_requested_vars = ["the requested rate/proportion outcome"]
        if _missing_requested_vars:
            named = ", ".join(_missing_requested_vars)
            message = (
                f"The requested variable(s) {named} could not be resolved to any column "
                "in the available dataset. AA-OS will not substitute a different column and "
                "run an analysis the question did not ask for; no experiment was run."
            )
            self.checkpointer.record_event(
                investigation_id=investigation_id,
                execution_id=exec_id,
                event_type="investigation.variable_not_found",
                payload={"missing_variables": _missing_requested_vars},
            )
            self.checkpointer.complete_execution(
                investigation_id=investigation_id, execution_id=exec_id,
                final_status="COMPLETED", error_message=message,
            )
            with self.session_factory() as terminal_session:
                terminal_inv = terminal_session.query(Investigation).filter(Investigation.id == investigation_id).first()
                if terminal_inv:
                    terminal_inv.status = InvestigationState.COMPLETED
                    terminal_inv.verdict_type = "VARIABLE_NOT_FOUND"
                    terminal_inv.direct_answer = message
                    terminal_inv.main_finding = message
                    terminal_inv.confidence_score = 0.0
                    terminal_session.commit()
            return True

        # 1.5 Compound analytical objective orchestration. A genuinely compound
        # question is never collapsed into whichever lexical task happens to win.
        # Independent clauses become durable child investigations under the same
        # dataset scope. Dependent clauses are refused rather than executed as if
        # a prior result were already established.
        compound_plan = plan_compound_question(question)
        if compound_plan.is_compound:
            with self.session_factory() as objective_session:
                existing = (
                    objective_session.query(InvestigationObjective)
                    .filter(InvestigationObjective.investigation_id == investigation_id)
                    .all()
                )
                existing_statements = {str(o.statement) for o in existing}
                for spec in compound_plan.objectives:
                    if spec.statement not in existing_statements:
                        objective_session.add(
                            InvestigationObjective(
                                id=gen_uuid(),
                                investigation_id=investigation_id,
                                statement=spec.statement,
                                priority_rank=spec.ordinal,
                                status="BLOCKED_DEPENDENCY" if spec.depends_on_prior_objective else "PLANNED",
                            )
                        )
                objective_session.commit()

            compound_results: list[dict[str, Any]] = []
            latest_cohort_entity: Optional[str] = None

            for spec in compound_plan.objectives:
                effective_statement = spec.statement
                if spec.depends_on_prior_objective:
                    # Check whether prior objectives established a verified empirical finding
                    prior_successful = [
                        r for r in compound_results
                        if r.get("status") == "COMPLETED" and r.get("verdict_type") in ("OBSERVED", "STATISTICALLY_SIGNIFICANT", "DIAGNOSED", "NO_DETECTABLE_EFFECT")
                    ]
                    if not prior_successful:
                        block_msg = (
                            f"Dependent analytical objective {spec.ordinal} blocked: prior objective "
                            f"did not establish an empirical finding for {spec.dependency_reason or 'context reference'}."
                        )
                        compound_results.append({
                            "ordinal": spec.ordinal,
                            "statement": spec.statement,
                            "child_investigation_id": None,
                            "status": "BLOCKED_DEPENDENCY",
                            "verdict_type": "INCONCLUSIVE",
                            "confidence_score": 0.0,
                            "direct_answer": block_msg,
                        })
                        with self.session_factory() as child_session:
                            objective = (
                                child_session.query(InvestigationObjective)
                                .filter(
                                    InvestigationObjective.investigation_id == investigation_id,
                                    InvestigationObjective.statement == spec.statement,
                                )
                                .order_by(InvestigationObjective.priority_rank.asc())
                                .first()
                            )
                            if objective is not None:
                                objective.status = "BLOCKED_DEPENDENCY"
                            child_session.commit()
                        self.checkpointer.record_event(
                            investigation_id=investigation_id,
                            execution_id=exec_id,
                            event_type="investigation.compound_objective.blocked",
                            payload={"ordinal": spec.ordinal, "statement": spec.statement, "reason": block_msg},
                        )
                        continue

                    # Ground the dependent statement with the empirical cohort discovered in prior stages
                    if latest_cohort_entity:
                        grounded = re.sub(
                            r"\b(?:for|among|in|of)\s+(?:those|these|such|the same)\s+(?:customers|clients|accounts|users|cohorts|segments|groups)?\b",
                            f"for {latest_cohort_entity} customers",
                            spec.statement,
                            flags=re.I,
                        )
                        grounded = re.sub(
                            r"\b(?:those|these|such)\s+(?:customers|clients|accounts|users|cohorts|segments|groups)\b",
                            f"{latest_cohort_entity} customers",
                            grounded,
                            flags=re.I,
                        )
                        effective_statement = grounded
                        self.checkpointer.record_event(
                            investigation_id=investigation_id,
                            execution_id=exec_id,
                            event_type="investigation.compound_objective.grounded",
                            payload={"ordinal": spec.ordinal, "original": spec.statement, "grounded": effective_statement, "cohort_entity": latest_cohort_entity},
                        )

                child_id = f"{investigation_id}-OBJ{spec.ordinal}-{gen_uuid()[:8]}"
                with self.session_factory() as child_session:
                    child_session.add(
                        Investigation(
                            id=child_id,
                            project_id=project_id,
                            user_id=parent_user_id,
                            question=effective_statement,
                            status=InvestigationState.PLANNED,
                            verdict_type="INCONCLUSIVE",
                            confidence_score=0.0,
                            requested_dataset_ids_json=list(requested_dataset_ids or []) or None,
                            parent_investigation_id=investigation_id,
                            is_subinvestigation=True,
                            created_at=datetime.now(timezone.utc),
                            updated_at=datetime.now(timezone.utc),
                        )
                    )
                    child_session.commit()
                self.checkpointer.record_event(
                    investigation_id=investigation_id,
                    execution_id=exec_id,
                    event_type="investigation.compound_objective.started",
                    payload={"ordinal": spec.ordinal, "statement": effective_statement, "original_statement": spec.statement, "child_investigation_id": child_id},
                )
                try:
                    self.execute_investigation(
                        child_id,
                        worker_id=f"{worker_id}:objective-{spec.ordinal}",
                        attempt=1,
                        cancellation_check=cancellation_check,
                    )
                except Exception as child_exc:
                    compound_results.append({
                        "ordinal": spec.ordinal,
                        "statement": spec.statement,
                        "effective_statement": effective_statement,
                        "child_investigation_id": child_id,
                        "status": "FAILED",
                        "verdict_type": "INCONCLUSIVE",
                        "direct_answer": f"{type(child_exc).__name__}: {child_exc}",
                    })
                    continue

                with self.session_factory() as child_session:
                    child = child_session.query(Investigation).filter(Investigation.id == child_id).first()
                    result = {
                        "ordinal": spec.ordinal,
                        "statement": spec.statement,
                        "effective_statement": effective_statement,
                        "child_investigation_id": child_id,
                        "status": str(child.status) if child else "FAILED",
                        "verdict_type": child.verdict_type if child else "INCONCLUSIVE",
                        "confidence_score": float(child.confidence_score or 0.0) if child else 0.0,
                        "direct_answer": child.direct_answer if child else "Child investigation record was not found.",
                    }
                    objective = (
                        child_session.query(InvestigationObjective)
                        .filter(
                            InvestigationObjective.investigation_id == investigation_id,
                            InvestigationObjective.statement == spec.statement,
                        )
                        .order_by(InvestigationObjective.priority_rank.asc())
                        .first()
                    )
                    if objective is not None:
                        objective.status = "COMPLETED" if child and str(child.status) == str(InvestigationState.COMPLETED) else "INCONCLUSIVE"
                    child_session.commit()

                    # Extract cohort entity for forward-chaining
                    if child and child.direct_answer:
                        ans_text = child.direct_answer
                        m = re.search(r"['\"]([A-Za-z0-9_-]+)['\"]\s+accounts for", ans_text)
                        if not m:
                            m = re.search(r"^([A-Za-z0-9_-]+)\s+has the highest", ans_text)
                        if not m:
                            m = re.search(r"concentrated in\s+['\"]?([A-Za-z0-9_-]+)['\"]?", ans_text, re.I)
                        if not m:
                            m = re.search(r"([A-Za-z0-9_-]+)\s+vs runner-up", ans_text)
                        if m:
                            latest_cohort_entity = m.group(1)

                compound_results.append(result)
                self.checkpointer.record_event(
                    investigation_id=investigation_id,
                    execution_id=exec_id,
                    event_type="investigation.compound_objective.completed",
                    payload=result,
                )

            all_completed = (
                bool(compound_results)
                and all(
                    r.get("status") == "COMPLETED" and r.get("verdict_type") in ("OBSERVED", "STATISTICALLY_SIGNIFICANT", "DIAGNOSED", "NO_DETECTABLE_EFFECT")
                    for r in compound_results
                )
            )
            any_blocked = any(r.get("status") == "BLOCKED_DEPENDENCY" for r in compound_results)

            if all_completed and any(o.depends_on_prior_objective for o in compound_plan.objectives):
                summary_lines = [f"Sequential compound investigation resolved across {len(compound_results)} analytical stages:"]
                for result in compound_results:
                    summary_lines.append(
                        f"Stage {result['ordinal']}: {result['statement']} -> "
                        f"{result.get('verdict_type', 'INCONCLUSIVE')}: {result.get('direct_answer') or 'No direct answer recorded.'}"
                    )
                final_verdict = "OBSERVED" if any(r.get("verdict_type") == "OBSERVED" for r in compound_results) else "STATISTICALLY_SIGNIFICANT"
                final_conf = min(float(r.get("confidence_score", 1.0)) for r in compound_results)
                final_finding = f"All {len(compound_results)} sequential compound objectives resolved with verified empirical findings across stages."
                stopping_rationale = "COMPOUND_OBJECTIVES_RESOLVED_SEQUENTIALLY"
            elif any_blocked:
                summary_lines = ["Compound investigation halted: dependent premise not established:"]
                for result in compound_results:
                    summary_lines.append(
                        f"Stage {result['ordinal']}: {result['statement']} -> "
                        f"{result.get('verdict_type', 'INCONCLUSIVE')}: {result.get('direct_answer') or 'No direct answer recorded.'}"
                    )
                final_verdict = "INCONCLUSIVE"
                final_conf = 0.0
                final_finding = "Investigation ended inconclusive because an earlier dependent premise was not established in data."
                stopping_rationale = "COMPOUND_DEPENDENCY_PREMISE_NOT_MET"
            else:
                summary_lines = [f"Compound question decomposed into {len(compound_results)} independent analytical objectives."]
                for result in compound_results:
                    summary_lines.append(
                        f"Objective {result['ordinal']}: {result['statement']} -> "
                        f"{result.get('verdict_type', 'INCONCLUSIVE')}: {result.get('direct_answer') or 'No direct answer recorded.'}"
                    )
                summary_lines.append(
                    "Overall status: INCONCLUSIVE at parent level; each objective retains its own contract, evidence, verification and provenance."
                )
                final_verdict = "INCONCLUSIVE"
                final_conf = 0.0
                final_finding = "Independent objective results are preserved separately; the parent question is not collapsed into a single unsupported verdict."
                stopping_rationale = "COMPOUND_OBJECTIVES_PRESERVED_SEPARATELY"

            self.checkpointer.record_event(
                investigation_id=investigation_id,
                execution_id=exec_id,
                event_type="investigation.compound_objective.synthesized",
                payload={"objective_count": len(compound_results), "verdict_type": final_verdict},
            )
            with self.session_factory() as parent_session:
                parent = parent_session.query(Investigation).filter(Investigation.id == investigation_id).first()
                if parent:
                    parent.status = InvestigationState.COMPLETED
                    parent.verdict_type = final_verdict
                    parent.direct_answer = "\n".join(summary_lines)
                    parent.main_finding = final_finding
                    parent.confidence_score = final_conf
                    parent.stopping_criteria_met = True
                    parent.stopping_rationale = stopping_rationale
                    parent_session.commit()
            self.checkpointer.complete_execution(
                investigation_id=investigation_id,
                execution_id=exec_id,
                final_status="COMPLETED",
                error_message=f"Compound objectives completed ({stopping_rationale}).",
            )
            return True

        # 2. Intent Parsing & Semantic Resolution
        intent = IntentEngine.parse_intent(question)
        semantic = self.semantic_engine.resolve_schema(intent, data_context.datasets_map)

        # Human-selected multi-dataset scopes are allowed to contain datasets
        # that are merely workspace context. However, when the question
        # explicitly names physical fields from multiple datasets, a missing
        # relational plan is a correctness failure, not permission to fall back
        # to the primary table. This closes the silent single-table-collapse
        # failure mode for real analysts.
        explicit_required_datasets = list(getattr(semantic, "question_referenced_datasets", []) or [])
        primary_cols = set(data_context.datasets_map.get(semantic.primary_dataset_name, pd.DataFrame()).columns)
        primary_fully_covers = (
            bool(semantic.target_metric_col and semantic.target_metric_col in primary_cols)
            and (semantic.group_dimension_col is None or semantic.group_dimension_col in primary_cols)
        )
        if (
            requested_dataset_ids
            and len(set(explicit_required_datasets)) >= 2
            and not primary_fully_covers
            and getattr(semantic, "relational_access", None) is None
        ):
            message = (
                "The question explicitly references fields from multiple selected datasets, "
                "but AA-OS could not construct a deterministic safe relational plan. "
                f"Referenced datasets: {explicit_required_datasets}. "
                "AA-OS refused to analyze only the primary dataset."
            )
            self.checkpointer.record_event(
                investigation_id=investigation_id,
                execution_id=exec_id,
                event_type="investigation.cross_dataset.plan_missing",
                payload={
                    "requested_dataset_ids": list(requested_dataset_ids),
                    "referenced_datasets": explicit_required_datasets,
                    "reason": "explicit_multi_dataset_reference_without_relational_plan",
                },
            )
            return _complete_inconclusive(message)

        # 2.25 Temporal Intent Resolution (closes DEFECT-002): resolve any
        # natural-language time reference in the question ("March", "last
        # quarter", "trailing 12 months", ...) against the ACTUAL date range
        # of the semantically-resolved time column, not wall-clock time.
        # Recorded on the investigation for provenance below; threaded into
        # ExperimentSynthesizer so diagnostic/comparative experiments are
        # scoped to the period the question actually asked about instead of
        # aggregating across the dataset's entire history.
        _primary_df_for_temporal = data_context.datasets_map.get(semantic.primary_dataset_name)
        if _primary_df_for_temporal is None:
            self.checkpointer.fail_execution(
                investigation_id=investigation_id,
                execution_id=exec_id,
                error_message=(
                    f"Semantically selected primary dataset {semantic.primary_dataset_name!r} "
                    "is unavailable in the acquired data context; refusing to analyze a different dataset."
                ),
                failure_class=FailureTaxonomy.RESOURCE_UNAVAILABLE,
            )
            return False
        temporal_scope = TemporalResolver.resolve(question, _primary_df_for_temporal, semantic.time_col)
        temporal_provenance_note: Optional[str] = None
        if temporal_scope.found:
            temporal_provenance_note = (
                f"Temporal scope: {temporal_scope.matched_expression!r} resolved to "
                f"[{temporal_scope.period_start.isoformat()}, {temporal_scope.period_end.isoformat()}) "
                f"via {temporal_scope.resolution_method}"
                + (f" -- AMBIGUOUS: {temporal_scope.ambiguity_note}" if temporal_scope.ambiguous else "")
            )

        # 2.5 Pre-Investigation Data Quality Fitness Gate
        primary_df = data_context.datasets_map.get(semantic.primary_dataset_name)
        if primary_df is None:
            self.checkpointer.fail_execution(
                investigation_id=investigation_id,
                execution_id=exec_id,
                error_message=(
                    f"Primary dataset {semantic.primary_dataset_name!r} disappeared from the acquired context; "
                    "refusing to fall back to another dataset."
                ),
                failure_class=FailureTaxonomy.RESOURCE_UNAVAILABLE,
            )
            return False
        quality_assessment = DataQualityGate.evaluate_fitness(
            primary_df,
            semantic.primary_dataset_name,
            time_col=semantic.time_col,
            metric_col=semantic.target_metric_col,
        )
        if not quality_assessment.can_proceed:
            self.checkpointer.fail_execution(
                investigation_id=investigation_id,
                execution_id=exec_id,
                error_message=f"Data Quality Gate rejected dataset: {'; '.join(quality_assessment.critical_issues)}",
                failure_class=FailureTaxonomy.DATA_QUALITY_FAILURE,
            )
            return False

        # Universal Question -> Analysis contract. This is the deterministic
        # "compiler" between natural language and the scientific runtime: it
        # makes claim type, estimand, evidence requirements, stopping rule,
        # and specialist path explicit BEFORE hypotheses are generated.
        analysis_plan = UniversalQuestionCompiler.compile(
            question,
            semantic=semantic,
            df=primary_df,
            quality_assessment=quality_assessment,
            ai_provider=self.ai_provider,
        )

        # Contextual data-quality findings are decision controls, not merely
        # diagnostics.  At plan time, require the corresponding sensitivity
        # or remediation review so the autonomous investigation cannot silently
        # continue as though the quality issue were irrelevant.
        plan_quality_decision = assess_downstream_impact(
            quality_assessment, analysis_plan.task,
            include_unresolved_without_runtime_evidence=True,
        )
        for review in plan_quality_decision.required_reviews:
            requirement = EvidenceRequirement(
                code=str(review["code"]),
                requirement=str(review["required_action"]),
                mandatory=True,
                rationale=f"Triggered by contextual data-quality finding {review['finding_id']}: {review['question']}",
            )
            if not any(r.code == requirement.code for r in analysis_plan.required_evidence):
                analysis_plan.required_evidence.append(requirement)
            if review["question"] not in analysis_plan.unresolved_questions:
                analysis_plan.unresolved_questions.append(review["question"])
        for reason in plan_quality_decision.blocking_reasons:
            limitation = f"Data-quality safeguard: {reason}"
            if limitation not in analysis_plan.limitations:
                analysis_plan.limitations.append(limitation)
        if plan_quality_decision.has_relevant_findings:
            self.checkpointer.record_event(
                investigation_id=investigation_id,
                execution_id=exec_id,
                event_type="investigation.data_quality.downstream_constraints",
                payload=plan_quality_decision.to_dict(),
            )

        # Autonomous dataset-version intelligence: before hypotheses are generated,
        # inspect the latest immutable version against the immediately prior version
        # when this is a database-backed dataset.  The assessment is deliberately
        # conservative: a material/relevant change creates a review requirement,
        # but never becomes a causal claim that the refresh caused the outcome.
        version_change_assessments: Dict[str, Dict[str, Any]] = {}
        relevant_columns = set(getattr(analysis_plan.semantics, "referenced_columns", []) or [])
        relevant_columns.update(
            c for c in (
                getattr(analysis_plan.semantics, "target_column", None),
                getattr(analysis_plan.semantics, "time_column", None),
                *(getattr(analysis_plan.semantics, "grouping_columns", []) or []),
                *(getattr(analysis_plan.semantics, "explanatory_columns", []) or []),
            ) if c
        )
        if getattr(semantic, "secondary_metric_col", None):
            relevant_columns.add(semantic.secondary_metric_col)

        if isinstance(self.dataset_provider, DatabaseDatasetProvider):
            try:
                with self.session_factory() as version_session:
                    for dataset_name, dataset_df in data_context.datasets_map.items():
                        version_meta = dict((data_context.dataset_versions or {}).get(dataset_name) or {})
                        dataset_id = version_meta.get("dataset_id")
                        current_version = int(version_meta.get("dataset_version") or 1)
                        if not dataset_id or current_version <= 1:
                            continue
                        previous = (
                            version_session.query(DatasetVersion)
                            .filter(
                                DatasetVersion.dataset_id == str(dataset_id),
                                DatasetVersion.version_number == current_version - 1,
                            )
                            .first()
                        )
                        if previous is None:
                            version_change_assessments[dataset_name] = {
                                "status": "PREVIOUS_VERSION_UNAVAILABLE",
                                "current_version": current_version,
                                "notes": ["Current version is greater than 1 but the immediately prior version record is unavailable."],
                            }
                            continue
                        previous_df = DatasetService(version_session).storage.load_dataframe(previous.file_path)
                        key_columns = infer_stable_key_columns(semantic, dataset_name, dataset_df)
                        assessment = assess_dataset_change(
                            previous_df,
                            dataset_df,
                            before_version=current_version - 1,
                            after_version=current_version,
                            question_columns=sorted(relevant_columns) if dataset_name == semantic.primary_dataset_name else None,
                            time_column=getattr(analysis_plan.semantics, "time_column", None) if dataset_name == semantic.primary_dataset_name else None,
                            key_columns=key_columns or None,
                            task=str(getattr(analysis_plan, "task", "")),
                        )
                        version_change_assessments[dataset_name] = assessment.to_dict()
            except Exception as version_exc:
                self.checkpointer.record_event(
                    investigation_id=investigation_id,
                    execution_id=exec_id,
                    event_type="investigation.dataset_version_change_assessment_failed",
                    payload={"error": f"{type(version_exc).__name__}: {version_exc}"},
                )
        else:
            version_change_assessments[semantic.primary_dataset_name] = {
                "status": "NOT_DATABASE_BACKED",
                "notes": ["Autonomous version comparison requires an immutable database-backed version history; in-memory benchmark datasets have no prior-version source."],
            }

        primary_version_assessment = version_change_assessments.get(semantic.primary_dataset_name)
        if primary_version_assessment and primary_version_assessment.get("needs_investigation"):
            review_requirement = EvidenceRequirement(
                code="DATASET_VERSION_CHANGE_REVIEW",
                requirement=(
                    "Review the immediately prior dataset version because the current refresh contains a material change "
                    "relevant to the requested analysis; establish whether the finding is stable to the version change."
                ),
                mandatory=True,
                rationale=(
                    "Autonomous dataset-version assessment detected a material and question-relevant change. "
                    "This is a review trigger, not evidence that the refresh caused the observed outcome."
                ),
            )
            if not any(r.code == review_requirement.code for r in analysis_plan.required_evidence):
                analysis_plan.required_evidence.append(review_requirement)
            review_question = "Could the dataset version change materially affect the requested finding?"
            if review_question not in analysis_plan.unresolved_questions:
                analysis_plan.unresolved_questions.append(review_question)
            analysis_plan.limitations.append(
                "The latest dataset version contains a material change relevant to this question; version-change review is required before treating the finding as stable to the refresh."
            )

        for dataset_name, assessment_payload in version_change_assessments.items():
            self.checkpointer.record_event(
                investigation_id=investigation_id,
                execution_id=exec_id,
                event_type="investigation.dataset_version_change_assessment",
                payload={
                    "dataset_name": dataset_name,
                    "question": question,
                    "assessment": assessment_payload,
                },
            )

        # Temporal claims must not silently inherit the host machine timezone.
        # Require either timezone-aware source timestamps or an explicit business
        # timezone declaration when the question actually invokes temporal scope.
        temporal_task = str(getattr(analysis_plan, "task", "")).upper() in {"FORECAST", "FORECASTING", "PREDICTION", "CAUSAL"}
        if semantic.time_col and (temporal_scope.found or temporal_task) and semantic.time_zone_policy == "UNRESOLVED_NAIVE_TIMEZONE":
            self.checkpointer.fail_execution(
                investigation_id=investigation_id,
                execution_id=exec_id,
                error_message=(
                    f"Temporal analysis requires an explicit business timezone for naive time column '{semantic.time_col}'. "
                    "Declare AAOS_BUSINESS_TIMEZONE or dataset attrs['business_timezone']; AA-OS will not infer it from host locale."
                ),
                failure_class=FailureTaxonomy.DATA_QUALITY_FAILURE,
            )
            return False

        # v20-A / v28: canonical semantic-role binding set for this contract
        # version, built from the already-resolved semantic model (no new
        # column-selection heuristic). Rejected/conflicting role assignments
        # are recorded but do not by themselves stop compilation here -- the
        # hard gate is enforced per-experiment at execution time via
        # validate_experiment_against_contract(), so a conflict discovered
        # here is exactly the information that gate will use.
        semantic_binding_set = build_binding_set(
            semantic,
            analysis_plan.task,
            explanatory_columns=list(analysis_plan.semantics.explanatory_columns),
        )
        semantic_binding_conflicts = semantic_binding_set.validate()
        if semantic_binding_conflicts:
            self.checkpointer.record_event(
                investigation_id=investigation_id, execution_id=exec_id,
                event_type="investigation.semantic_binding.conflict",
                payload={"conflicts": [c.to_dict() for c in semantic_binding_conflicts]},
            )

        # v28 SemanticBindingSet authority cutover: from here on, all legacy
        # target/grouping/time/explanatory fields are strictly projected from
        # the canonical SemanticBindingSet rather than being parallel authorities.
        canonical_target_col = semantic_binding_set.projected_target_column()
        canonical_group_dimension_col = semantic_binding_set.projected_group_dimension()
        canonical_time_col = semantic_binding_set.projected_time_column()
        canonical_explanatory_cols = semantic_binding_set.projected_explanatory_columns()

        # Persist the compiled contract as a first-class, versioned artifact.
        # The event stream remains useful for replay, but the DB row is now the
        # durable source of truth for what AA-OS intended to investigate.
        contract_dict = analysis_plan.to_dict()
        with self.session_factory() as contract_session:
            existing_contract = (contract_session.query(InvestigationContract)
                                 .filter(InvestigationContract.investigation_id == investigation_id)
                                 .order_by(InvestigationContract.version.desc()).first())

            target_data = {"target": canonical_target_col}
            grain_data = {"grain": analysis_plan.semantics.grain}
            scope_data = {"dataset": semantic.primary_dataset_name}

            if semantic.world_model:
                for m in getattr(semantic.world_model, "metrics", []) or []:
                    if m.table_name == semantic.primary_dataset_name and m.column_name == canonical_target_col:
                        target_data["epistemic_source"] = getattr(m.epistemic_source, "value", str(m.epistemic_source))
                        target_data["aggregation_type"] = getattr(m, "aggregation_type", "SUM")
                        target_data["unit"] = getattr(m, "unit", "UNKNOWN")
                        target_data["additivity"] = getattr(m.additivity, "value", str(m.additivity))
                        break
                for e in getattr(semantic.world_model, "entities", []) or []:
                    if e.table_name == semantic.primary_dataset_name:
                        grain_data["grain_proven"] = getattr(e, "grain_proven", False)
                        grain_data["epistemic_source"] = getattr(e.epistemic_source, "value", str(e.epistemic_source))
                        grain_data["primary_key"] = getattr(e, "primary_key", None)
                        break
                scope_data["verified_grains"] = getattr(semantic.world_model, "verified_grains", {}).get(semantic.primary_dataset_name, [])
                scope_data["epistemic_manifest"] = getattr(semantic.world_model, "epistemic_manifest", {})

            contract = InvestigationContract(
                investigation_id=investigation_id,
                version=(existing_contract.version + 1 if existing_contract else 1),
                parent_contract_id=(existing_contract.id if existing_contract else None),
                original_question=question,
                normalized_question=" ".join((question or "").split()),
                problem_class=analysis_plan.task,
                claim_type=analysis_plan.claim_type,
                target_json=target_data,
                explanatory_variables_json=canonical_explanatory_cols,
                population_json={
                    "scope": "requested_datasets" if requested_dataset_ids else "project_datasets",
                    "requested_dataset_ids": list(requested_dataset_ids or []),
                    "resolved_dataset_names": sorted(list(data_context.datasets_map.keys())),
                    "resolved_dataset_versions": dict(data_context.dataset_versions),
                    "dataset_fingerprints": dict(data_context.dataset_fingerprints),
                    "dataset_identities": {
                        name: {
                            "dataset_id": meta.get("dataset_id"),
                            "dataset_version_id": meta.get("dataset_version_id"),
                            "dataset_version": meta.get("dataset_version"),
                            "fingerprint": (data_context.dataset_fingerprints or {}).get(name),
                        }
                        for name, meta in (data_context.dataset_versions or {}).items()
                    },
                    "primary_dataset": semantic.primary_dataset_name,
                },
                grain_json=grain_data,
                scope_json={
                    **dict(scope_data or {}),
                    "dataset_scope": {
                        "requested_dataset_ids": list(requested_dataset_ids or []),
                        "resolved_dataset_names": sorted(list(data_context.datasets_map.keys())),
                        "primary_dataset": semantic.primary_dataset_name,
                    },
                },
                time_window_json={"time_column": canonical_time_col},
                estimand_json=dict(analysis_plan.estimand),
                assumptions_json=list(analysis_plan.limitations),
                data_requirements_json=[e.to_dict() for e in analysis_plan.required_evidence],
                candidate_methods_json=[e.to_dict() for e in analysis_plan.experiments],
                # v20-C4.2.3: the compiler's experiment is a PROPOSAL.  selected_method_json
                # stays NULL until finalize_contract() writes the reconciled decision.
                proposed_method_json=(analysis_plan.experiments[0].to_dict() if analysis_plan.experiments else None),
                selected_method_json=None,
                evidence_requirements_json=[e.to_dict() for e in analysis_plan.required_evidence],
                stopping_criteria_json={"rule": analysis_plan.stopping_rule},
                ambiguity_state_json={"status": analysis_plan.decision_status},
                semantic_interpretations_json=list(analysis_plan.semantics.notes),
                unresolved_questions_json=list(analysis_plan.unresolved_questions),
                semantic_bindings_json=semantic_binding_set.to_dict_list(),
                confidence=None,
            )
            contract_session.add(contract)
            contract_session.flush()
            contract_id = contract.id
            contract_version = contract.version
            inv_contract = contract_session.query(Investigation).filter(Investigation.id == investigation_id).first()
            if inv_contract is not None:
                inv_contract.active_contract_id = contract_id
                inv_contract.contract_revision = contract.version
                inv_contract.current_phase = AnalyticalPhase.CONTRACT_COMPILED
            for dataset_name, dataset_df in data_context.datasets_map.items():
                dataset_quality = (
                    quality_assessment
                    if dataset_name == quality_assessment.dataset_name
                    else DataQualityGate.evaluate_fitness(dataset_df, dataset_name)
                )
                version_meta = dict((data_context.dataset_versions or {}).get(dataset_name) or {})
                contract_session.add(InvestigationDataReadiness(
                    investigation_id=investigation_id,
                    contract_id=contract_id,
                    dataset_name=dataset_name,
                    dataset_version_ids_json=[version_meta.get("dataset_version_id")] if version_meta.get("dataset_version_id") else [],
                    source_fingerprints_json={dataset_name: (data_context.dataset_fingerprints or {}).get(dataset_name)},
                    row_count=int(dataset_quality.row_count),
                    column_count=int(dataset_quality.column_count),
                    overall_quality_score=float(dataset_quality.overall_quality_score),
                    fitness_verdict=dataset_quality.fitness_verdict,
                    can_proceed=bool(dataset_quality.can_proceed),
                    checks_json={
                        "missingness": dataset_quality.missingness_summary,
                        "duplicates": dataset_quality.duplicate_rows_count,
                        "duplicate_keys": dataset_quality.duplicate_key_columns,
                        "outliers": dataset_quality.outlier_columns,
                        "class_imbalance": dataset_quality.class_imbalance,
                        "temporal_issues": dataset_quality.temporal_issues,
                        "selection_bias_indicators": dataset_quality.selection_bias_indicators,
                        "leakage_indicators": dataset_quality.leakage_indicators,
                        "unit_consistency_indicators": dataset_quality.unit_consistency_indicators,
                        "contextual_findings": [
                            {
                                "finding_id": f.finding_id,
                                "severity": f.severity,
                                "issue": f.issue,
                                "scope": f.scope,
                                "evidence": f.evidence,
                                "potential_bias": f.potential_bias,
                                "affected_methods": f.affected_methods,
                                "recommended_action": f.recommended_action,
                            }
                            for f in dataset_quality.contextual_findings
                        ],
                        "dataset_version_change_assessment": version_change_assessments.get(dataset_name),
                    },
                    critical_issues_json=list(dataset_quality.critical_issues),
                    warnings_json=list(dataset_quality.warnings),
                    recommendations_json=list(dataset_quality.recommendations),
                ))
            contract_session.commit()

        for phase, payload in [
            (AnalyticalPhase.QUESTION_UNDERSTANDING, {"question": question}),
            (AnalyticalPhase.SEMANTIC_MODEL, {"dataset": semantic.primary_dataset_name, "grain": semantic.table_grain}),
            (AnalyticalPhase.DATA_READINESS, {"fitness": quality_assessment.fitness_verdict, "score": quality_assessment.overall_quality_score}),
            (AnalyticalPhase.CONTRACT_COMPILED, {"contract_id": contract_id, "task": analysis_plan.task}),
        ]:
            self.checkpointer.record_event(investigation_id=investigation_id, execution_id=exec_id, event_type=f"investigation.phase.{phase}", payload=payload)
        self.checkpointer.record_event(
            investigation_id=investigation_id,
            execution_id=exec_id,
            event_type="investigation.universal_analysis_plan",
            payload=analysis_plan.to_dict(),
        )
        self.checkpointer.record_event(
            investigation_id=investigation_id,
            execution_id=exec_id,
            event_type="investigation.dataset_scope_resolved",
            payload={
                "requested_dataset_ids": list(requested_dataset_ids or []),
                "resolved_dataset_names": sorted(data_context.datasets_map.keys()),
                "resolved_dataset_versions": dict(data_context.dataset_versions),
                "dataset_fingerprints": dict(data_context.dataset_fingerprints),
                "primary_dataset": semantic.primary_dataset_name,
                "relational_access": (
                    {
                        "base_table": semantic.relational_access.base_table,
                        "joined_table": semantic.relational_access.joined_table,
                        "join_hops": list(semantic.relational_access.join_hops),
                        "joined_tables": list(getattr(semantic.relational_access, "joined_tables", []) or []),
                    }
                    if getattr(semantic, "relational_access", None) is not None else None
                ),
            },
        )

        # Canonical analytical policy spine: classify problem, bind the estimand,
        # and select admissible method family/claim ceiling exactly once.
        try:
            method_decision = MethodSelectionEngine.decide(
                intent=intent,
                semantic=semantic,
                question=question,
                temporal_scope=temporal_scope,
                investigation_id=investigation_id,
                primary_df=primary_df,
                # v20-C4.2.1: hand the compiler's question-grounded roles to
                # canonical resolution (compile() ran above), so the
                # estimand reflects the variable the user actually asked
                # about rather than a generic discovered secondary metric.
                question_roles=QuestionRoleProposal.from_plan_semantics(getattr(analysis_plan, "semantics", None)),
            )
        except Exception as method_exc:
            method_decision = MethodSelectionEngine.fallback(
                f"controller_method_selection_exception:{type(method_exc).__name__}:{method_exc}"
            )
        # The universal analysis contract is the authoritative input to method
        # selection. The legacy intent decision supplies compatibility metadata; the
        # registry then ranks the admissible methods for this concrete question/data
        # state and records the winning method for downstream execution.
        try:
            # v20-C4.2.3: the plan is a PROPOSAL.  Selection consumes the reconciled view
            # (predictors canonical resolution rejected are removed); analysis_plan itself
            # stays untouched for conflict detection against the final contract.
            method_decision = MethodSelectionEngine.select_for_plan(
                method_decision,
                MethodSelectionEngine.canonical_plan_view(method_decision, analysis_plan),
                semantic, quality_assessment, primary_df,
            )
        except Exception as registry_exc:
            method_decision.rationale += (
                f" Registry selection unavailable: {type(registry_exc).__name__}: {registry_exc}."
            )

        # v19 section 8/13: hard authority-conflict assertion. select_for_plan
        # always sets method_decision.canonical_task = analysis_plan.task on
        # every path it takes (including its own exception handling above,
        # via method_decision.fallback()/re-raise). If they disagree here it
        # means some earlier path returned a decision without going through
        # that reconciliation -- never silently pick one and continue.
        if method_decision.canonical_task != analysis_plan.task:
            self.checkpointer.record_event(
                investigation_id=investigation_id,
                execution_id=exec_id,
                event_type="investigation.analytical_authority_conflict",
                payload={
                    "conflict": "ANALYTICAL_AUTHORITY_CONFLICT",
                    "contract_problem_class": analysis_plan.task,
                    "method_decision_canonical_task": method_decision.canonical_task,
                    "method_decision_problem_class": method_decision.problem_class.value,
                },
            )
            return _complete_inconclusive(
                "Internal analytical authority conflict: the canonical analysis plan "
                f"(task={analysis_plan.task}) and the method-selection decision "
                f"(canonical_task={method_decision.canonical_task}) disagree. Refusing to "
                "execute rather than silently choosing one."
            )

        self.checkpointer.record_event(
            investigation_id=investigation_id,
            execution_id=exec_id,
            event_type="investigation.method_selection",
            payload=method_decision.to_provenance_dict(),
        )

        # v20-C4.2 (detection increment): record, but do not act on, any
        # disagreement between plan.semantics (what admissibility/scoring
        # above just used) and decision.estimand (the canonical role
        # identity). See MethodSelectionEngine.detect_canonical_plan_role_disagreement
        # and AAOS_V20C4_2_CONTRACT_AUTHORITY_AUDIT.md for why this is
        # deliberately non-blocking for now.
        try:
            role_conflicts = MethodSelectionEngine.detect_canonical_plan_role_disagreement(method_decision, analysis_plan)
            if role_conflicts:
                self.checkpointer.record_event(
                    investigation_id=investigation_id,
                    execution_id=exec_id,
                    event_type="investigation.canonical_plan_role_disagreement",
                    payload={"conflicts": role_conflicts},
                )
        except Exception:
            # Best-effort instrumentation; never block execution on it.
            pass

        # v20-C4.2.3: FINALIZE the analytical contract.  The method decision is now
        # reconciled, so the durable contract row receives the FINAL decision
        # (problem class, estimand target/predictors, selected method, claim
        # ceiling, verification regime, deterministic analytical identity).
        # Unlike the removed best-effort reconciled_method_decision write, this is
        # fail-closed: if no final contract can be established, nothing executes.
        try:
            _final_predictors = list(method_decision.estimand.predictor_columns or []) or list(
                analysis_plan.semantics.explanatory_columns or []
            )
            final_binding_set = build_binding_set(
                semantic, analysis_plan.task, explanatory_columns=_final_predictors,
            )
            final_contract = FinalAnalyticalContract.from_method_decision(
                method_decision,
                dataset_fingerprints=data_context.dataset_fingerprints,
                semantic_bindings=final_binding_set,
            )
            plan_final_conflicts = detect_plan_final_contract_conflicts(analysis_plan, final_contract)
            if plan_final_conflicts:
                self.checkpointer.record_event(
                    investigation_id=investigation_id,
                    execution_id=exec_id,
                    event_type="investigation.plan_final_contract_conflict",
                    payload={"conflicts": plan_final_conflicts, "authority": "FINAL_CONTRACT"},
                )
            if any(c["severity"] == "BLOCKING" for c in plan_final_conflicts):
                return _complete_inconclusive(
                    "Analytical authority conflict: the compiler proposal disagrees with the "
                    "final reconciled contract on " + ", ".join(
                        sorted({c["field"] for c in plan_final_conflicts if c["severity"] == "BLOCKING"})
                    ) + ". Refusing to execute rather than let a stale plan redefine the analysis."
                )
            with self.session_factory() as authority_session:
                finalize_contract(authority_session, contract_id, final_contract)
        except AnalyticalIdentityError as identity_exc:
            self.checkpointer.record_event(
                investigation_id=investigation_id,
                execution_id=exec_id,
                event_type="investigation.final_contract_failed",
                payload={"error": str(identity_exc), "error_type": type(identity_exc).__name__},
            )
            return _complete_inconclusive(
                f"No final analytical contract could be established ({identity_exc}); refusing to execute."
            )

        # Method admissibility gate: the compiled analyst question and data-readiness
        # state must agree with the declared capability before any specialist method
        # is allowed to execute. A method can be available in the codebase and still
        # be BLOCKED for this particular question/data combination.
        method_gate_ok, method_gate = MethodRegistry.admissibility_for_plan(
            analysis_plan, semantic, quality_assessment, primary_df
        )
        self.checkpointer.record_event(
            investigation_id=investigation_id,
            execution_id=exec_id,
            event_type="investigation.method_admissibility",
            payload=method_gate,
        )
        # A proven typed relational aggregation is a deterministic descriptive
        # computation, not an inferential group-effect test.  Its admissibility
        # is established by join safety and dual-engine verification below;
        # do not reject it solely because the single-table statistical preflight
        # cannot see the dimension column across the join.
        if not method_gate_ok and getattr(semantic, "relational_access", None) is None:
            # A per-customer churn-risk question ("Which customers are likely
            # to churn?") is compiled by UniversalQuestionCompiler as task
            # "PREDICTION" (entity-level risk ranking), which is a distinct
            # analytical task from aggregate churn-rate comparison. But when
            # the underlying dataset has no identifiable churn/cancellation/
            # attrition event at all, the generic "target_not_binary"
            # preflight message this gate produces is misleading -- there is
            # no target to be non-binary about, the outcome simply does not
            # exist. IntentEngine's lexical CHURN classification (based on
            # churn/attrition/cancellation vocabulary) is independent of the
            # PREDICTION-vs-CHURN task split above, so use it here to detect
            # this case and give the same honest, specific explanation the
            # canonical churn-identifiability path (is_churn_unidentifiable,
            # later in this method) gives when it IS reached -- rather than
            # a generic method-admissibility failure that reads as a data
            # quality/shape problem instead of a data-availability one.
            if (
                str(getattr(intent, "intent_type", "") or "").upper() == "CHURN"
                and (
                    not getattr(semantic, "churn_outcome_available", True)
                    or getattr(semantic, "churn_event_col", None) is None
                )
            ):
                return _complete_inconclusive(
                    "the dataset contains no identifiable churn outcome, so AA-OS cannot "
                    "estimate or predict churn from this data. This is a data-availability "
                    "limitation, not evidence that churn did or did not occur."
                )
            return _complete_inconclusive(
                f"Method admissibility gate blocked {method_gate.get('method', 'requested method')}: "
                + "; ".join(method_gate.get("errors", []))
            )


        # v19 section 8: UNSUPPORTED_ANALYTICAL_TASK gate. This must run before
        # the SPECIALIST AUTHORITY GATE below, using the same
        # CANONICAL_TASK_AUTHORITY registry that gate now reads from, so
        # there is exactly one place that decides "does AA-OS have any
        # analytical method for this task at all".
        if method_decision.analytical_task_status == "UNSUPPORTED_ANALYTICAL_TASK":
            self.checkpointer.record_event(
                investigation_id=investigation_id, execution_id=exec_id,
                event_type="investigation.unsupported_analytical_task",
                payload={"task": analysis_plan.task, "rationale": method_decision.rationale},
            )
            return _complete_inconclusive(
                f"UNSUPPORTED_ANALYTICAL_TASK: '{analysis_plan.task}' has no registered "
                "analytical method in this release; no result can be produced."
            )

        # SPECIALIST AUTHORITY GATE
        #
        # Specialists are computational engines, never epistemic authorities.
        # A specialist result is admissible to the scientific state only when it
        # supplies a separately executed verification proof AND the canonical
        # stopping/claim gate can consume that proof. The previous implementation
        # persisted specialist output directly as Evidence/EvidenceVerification
        # and could therefore turn `validation=PASSED` into a completed verdict
        # without the canonical loop. That is prohibited.
        #
        # v19 section 7/8: this used to be a separate hardcoded
        # {"RECONCILIATION", "DATA_QUALITY", "GOVERNANCE"} set maintained
        # independently of method_selection.py's registry -- a second,
        # competing authority over the same question ("can this task run?").
        # It now reads canonical_transition_wired off the single
        # CANONICAL_TASK_AUTHORITY registry instead. DATA_QUALITY/GOVERNANCE
        # never reach this point now (blocked above as UNSUPPORTED); this
        # gate's remaining real job is RECONCILIATION, which does have a
        # registered method but isn't wired to the canonical transition yet.
        _task_entry = CANONICAL_TASK_AUTHORITY.get(analysis_plan.task)
        if _task_entry is not None and not _task_entry.canonical_transition_wired:
            self.checkpointer.record_event(
                investigation_id=investigation_id, execution_id=exec_id,
                event_type="investigation.specialist_authority.blocked",
                payload={
                    "task": analysis_plan.task,
                    "reason": "specialist_requires_canonical_transition",
                    "required_chain": [
                        "computation", "independent_verification", "evidence",
                        "belief_or_uncertainty", "adversarial_if_applicable",
                        "stopping", "claim_gate", "verdict", "provenance",
                    ],
                },
            )
            return _complete_inconclusive(
                f"Specialist task {analysis_plan.task} has no canonical transition executor; "
                "no authoritative specialist result was produced."
            )

        # Build World Model AST
        unit_of_analysis_keys = [canonical_target_col]
        if semantic.world_model and hasattr(semantic.world_model, "verified_grains"):
            # .get()'s default only applies when the key is absent -- if the
            # dataset IS present in verified_grains but mapped to an empty
            # list (grain not yet established), the default never kicks in
            # and an empty keys list reaches GrainRef, which requires at
            # least one key. Fall back explicitly whenever the resolved
            # value is empty, not just when the key is missing.
            resolved_keys = semantic.world_model.verified_grains.get(semantic.primary_dataset_name)
            if resolved_keys:
                unit_of_analysis_keys = resolved_keys

        typed_intent = TypedAnalyticalIntent(
            intent_id=f"INTENT_{investigation_id[:8]}",
            objective=method_decision.objective,
            target_metric=MetricRef(
                name=canonical_target_col,
                table=semantic.primary_dataset_name,
                column=canonical_target_col,
                # Phase 11: use the metric's actually-resolved semantics instead
                # of hardcoding SUM for every investigation regardless of what
                # the target metric is (a rate, a mean, a count, ...).
                aggregation=(
                    semantic.metric_definition.aggregation_type
                    if semantic.metric_definition else AggregationType.SUM
                ),
            ),
            unit_of_analysis=GrainRef(
                table=semantic.primary_dataset_name,
                keys=unit_of_analysis_keys,
            ),
            temporal_scope=(
                temporal_scope.to_temporal_scope(
                    semantic.primary_dataset_name,
                    canonical_time_col,
                )
                if getattr(temporal_scope, "found", False)
                else None
            ),
            causal_intent=method_decision.causal_intent,
            dimensions=[
                VariableRef(
                    table=(
                        semantic.relational_access.joined_table
                        if getattr(semantic, "relational_access", None) is not None
                        else semantic.primary_dataset_name
                    ),
                    column=canonical_group_dimension_col,
                )
            ] if canonical_group_dimension_col else [],
            source_question=question,
        )

        ir_val_res = self.ir_validator.validate(typed_intent, semantic.world_model)
        if not ir_val_res.is_valid:
            # BUGFIX (Phase 10): this used to `raise ValueError(...)` here,
            # uncaught by any surrounding try/except in this method. That
            # left the Investigation stuck in RUNNING forever (never
            # transitioned to FAILED) instead of reaching a safe terminal
            # state -- a hostile-but-common case (e.g. a 0/1 rate/percentage
            # metric requested for SUM aggregation) crashed the safety net
            # instead of being reported as a controlled data-quality failure.
            err_msgs = "; ".join([e.message for e in ir_val_res.errors])
            self.checkpointer.fail_execution(
                investigation_id=investigation_id,
                execution_id=exec_id,
                error_message=f"Analytical IR Validation Failed on Question Intent: {err_msgs}",
                failure_class=FailureTaxonomy.DATA_QUALITY_FAILURE,
            )
            return False

        # Production Causal Gate
        # The semantic world model is not a causal graph. For causal questions on
        # a single resolved dataframe, use the conservative PC discovery path.
        # It may still return OBSERVATIONAL_ONLY when the graph is only partially
        # oriented; that is intentional and prevents an observational CPDAG from
        # being silently promoted into a unique causal DAG.
        if method_decision.causal_intent is not None:
            try:
                causal_treatment = method_decision.causal_intent.treatment.column
                causal_outcome = method_decision.causal_intent.outcome.column
                causal_table = method_decision.causal_intent.treatment.table
                same_primary_table = causal_table in (semantic.primary_dataset_name, "df")
                if same_primary_table and causal_treatment in primary_df.columns and causal_outcome in primary_df.columns:
                    causal_gate_res = self.causal_gate.discover_and_evaluate(
                        primary_df,
                        treatment=causal_treatment,
                        outcome=causal_outcome,
                        feature_columns=semantic.available_numeric_cols,
                    )
                else:
                    causal_gate_res = self.causal_gate.evaluate_identifiability(
                        method_decision.causal_intent, None
                    )
            except Exception as causal_exc:
                causal_gate_res = self.causal_gate.evaluate_identifiability(
                    method_decision.causal_intent, None
                )
                self.checkpointer.record_event(
                    investigation_id=investigation_id,
                    execution_id=exec_id,
                    event_type="investigation.causal_gate.fallback",
                    payload={
                        "reason": f"discovery_failed:{type(causal_exc).__name__}",
                        "message": str(causal_exc),
                        "status": getattr(causal_gate_res.status, "value", causal_gate_res.status),
                    },
                )
        else:
            causal_gate_res = self.causal_gate.evaluate_identifiability(None, None)

        self.checkpointer.record_event(
            investigation_id=investigation_id,
            execution_id=exec_id,
            event_type="investigation.causal_gate.evaluated",
            payload={
                "status": getattr(causal_gate_res.status, "value", causal_gate_res.status),
                "is_identifiable": causal_gate_res.is_identifiable,
                "adjustment_set": causal_gate_res.backdoor_adjustment_set,
                "sensitivity_e_value": causal_gate_res.sensitivity_e_value,
                "diagnostic_message": causal_gate_res.diagnostic_message,
                "proof": causal_gate_res.proof.model_dump(),
            },
        )

        # Phase 23 causal estimation: identification alone is not an effect estimate.
        # When a single-table treatment/outcome is identified, run a conservative
        # AIPW estimate with cross-fitted nuisance models and reproducible bootstrap
        # uncertainty. Any estimation failure is recorded and never promoted to a
        # causal claim.
        causal_effect_res = None
        causal_effect_estimated = False
        if (
            method_decision.causal_intent is not None
            and causal_gate_res.is_identifiable
            and semantic.primary_dataset_name in (method_decision.causal_intent.treatment.table, "df")
        ):
            try:
                from packages.analytics_core.src.causal.effect_estimation import estimate_binary_treatment_effect
                causal_effect_res = estimate_binary_treatment_effect(
                    primary_df,
                    treatment_col=method_decision.causal_intent.treatment.column,
                    outcome_col=method_decision.causal_intent.outcome.column,
                    adjustment_set=causal_gate_res.backdoor_adjustment_set,
                    estimand="ATE",
                    estimator="AIPW",
                    confidence=0.95,
                    bootstrap_resamples=500,
                    seed=20260908,
                )
                causal_effect_estimated = True
                self.checkpointer.record_event(
                    investigation_id=investigation_id,
                    execution_id=exec_id,
                    event_type="investigation.causal_effect.estimated",
                    payload={
                        "effect": causal_effect_res.effect,
                        "estimand": causal_effect_res.estimand,
                        "estimator": causal_effect_res.estimator,
                        "confidence_interval": causal_effect_res.confidence_interval,
                        "standard_error": causal_effect_res.standard_error,
                        "n": causal_effect_res.n,
                        "treated_n": causal_effect_res.treated_n,
                        "control_n": causal_effect_res.control_n,
                        "positivity": causal_effect_res.positivity,
                        "balance": causal_effect_res.balance,
                        "formulas": causal_effect_res.formulas,
                        "assumptions": causal_effect_res.assumptions,
                        "limitations": causal_effect_res.limitations,
                        "diagnostics": causal_effect_res.diagnostics,
                        "reproducibility": causal_effect_res.reproducibility,
                    },
                )
            except Exception as causal_est_exc:
                self.checkpointer.record_event(
                    investigation_id=investigation_id,
                    execution_id=exec_id,
                    event_type="investigation.causal_effect.estimation_failed",
                    payload={
                        "reason": f"estimation_failed:{type(causal_est_exc).__name__}",
                        "message": str(causal_est_exc),
                        "identification_status": getattr(causal_gate_res.status, "value", causal_gate_res.status),
                    },
                )

        # Scientific transparency layer: make the assumptions and limitations
        # of the selected analysis explicit before hypotheses/experiments run.
        # This does not alter evidence; it creates a durable review surface.
        assumption_items, quality_card = AssumptionLedgerEngine.build(
            question=question,
            method_decision=method_decision,
            semantic=semantic,
            quality_assessment=quality_assessment,
            temporal_scope=temporal_scope,
            causal_gate_result=causal_gate_res,
        )
        # The FORECAST_GENERALIZATION assumption ("historical patterns are
        # assumed to provide useful information about the forecast
        # horizon") is necessarily built as unvalidated: AssumptionLedgerEngine.build()
        # runs before any experiment executes, so there is no backtest
        # evidence yet to check it against, and AssumptionItem is a frozen
        # dataclass that cannot be mutated afterward. Left as-is, this
        # "high" sensitivity-risk, permanently-unvalidated assumption
        # silently caps every forecast investigation at INCONCLUSIVE
        # forever -- even given an overwhelming, dual-engine-verified,
        # backtest-confirmed trend -- which contradicts method_selection's
        # own explicit STATISTICALLY_SIGNIFICANT ceiling for FORECAST
        # (parity with CORRELATION, not an unreachable one). The rolling-
        # origin backtest a forecast experiment actually runs (comparing
        # the fitted trend's out-of-sample error against a naive baseline)
        # is direct empirical evidence for exactly this assumption, so
        # track whether that evidence was observed during execution and
        # treat the assumption as validated by it at count time below.
        forecast_generalization_validated = False

        self.checkpointer.record_event(
            investigation_id=investigation_id,
            execution_id=exec_id,
            event_type="investigation.assumption_ledger.created",
            payload={
                "quality_card": quality_card.to_dict(),
                "assumptions": [item.to_dict() for item in assumption_items],
            },
        )

        def persist_assumption_ledger(session: Session):
            from apps.api.src.models.entities import Assumption
            for item in assumption_items:
                existing = (
                    session.query(Assumption)
                    .filter(Assumption.investigation_id == investigation_id, Assumption.statement == item.statement)
                    .first()
                )
                if existing is None:
                    session.add(Assumption(
                        id=gen_uuid(),
                        investigation_id=investigation_id,
                        statement=f"[{item.code}] {item.statement}",
                        is_validated=item.is_validated,
                        sensitivity_risk=item.sensitivity_risk,
                    ))

        self.checkpointer.execute_step_transactionally(
            investigation_id=investigation_id,
            execution_id=exec_id,
            step_index=2,
            step_type="ASSUMPTION_LEDGER",
            step_code="ASSUMPTION-LEDGER",
            step_fn=persist_assumption_ledger,
        )

        # Initialize State Manager & Evidence Ledger
        state_mgr = InvestigationStateManager(investigation_id, question, typed_intent.objective)
        evidence_ledger = EvidenceLedger(investigation_id)

        decision_eval = HumanDecisionController.evaluate_ambiguity(
            available_numeric_cols=semantic.available_numeric_cols,
            resolved_metric=canonical_target_col,
            user_specified_metric=bool(intent.target_metric_hint),
        )
        if decision_eval.policy == "ASK_USER":
            self.checkpointer.request_human_decision(
                investigation_id=investigation_id,
                execution_id=exec_id,
                question=decision_eval.question or "Please confirm target metric",
                options=decision_eval.options or [canonical_target_col],
            )
            return False

        def step1_objective(session: Session):
            existing_obj = (
                session.query(InvestigationObjective)
                .filter(InvestigationObjective.investigation_id == investigation_id)
                .first()
            )
            if not existing_obj:
                obj = InvestigationObjective(
                    id=gen_uuid(),
                    investigation_id=investigation_id,
                    statement=question,
                    target_variable=canonical_target_col,
                    priority_rank=1,
                    status="active",
                )
                session.add(obj)
            return {
                "objective": question,
                "target_metric": canonical_target_col,
                "ir_valid": ir_val_res.is_valid,
            }

        self.checkpointer.execute_step_transactionally(
            investigation_id=investigation_id,
            execution_id=exec_id,
            step_index=1,
            step_type="OBJECTIVE_DECOMPOSITION",
            step_code="OBJ-01",
            step_fn=step1_objective,
        )

        # 3. Dynamic Predictive Hypothesis Graph Synthesis / restart reconstruction
        if cancellation_check and cancellation_check():
            return False

        prior_hyp_step_completed = self.checkpointer.is_step_completed(
            investigation_id, f"{investigation_id}:HYPOTHESIS_GENERATION:HYP-GEN"
        )
        prior_execution_attempt = attempt > 1
        if prior_hyp_step_completed or prior_execution_attempt:
            # Reconstruct the scientific runtime from durable projections + immutable events.
            # This is the authoritative restart path; do not synthesize a fresh graph.
            with self.session_factory() as reconstruction_session:
                state_mgr = reconstruct_investigation_state(
                    investigation_id=investigation_id,
                    session=reconstruction_session,
                    question=question,
                )
            current_hyps = _stable_hypothesis_order(state_mgr.get_active_hypotheses())
            initial_priors = [h.prior_probability for h in current_hyps]
            initial_entropy = compute_shannon_entropy(initial_priors) if initial_priors else 0.0
            candidate_hyps = list(current_hyps)
            reconstructed = True
        else:
            candidate_hyps = HypothesisSynthesizer.synthesize_competing_hypotheses(semantic, question, intent=intent, decision=method_decision)
            initial_priors = [h.prior_probability for h in candidate_hyps]
            initial_entropy = compute_shannon_entropy(initial_priors)
            reconstructed = False

        # v20-C4.2.3: stamp each hypothesis with the deterministic analytical
        # identity it belongs to, from its STRUCTURED target/predictor fields
        # (never its narrative).  Hypotheses that match no pair stay unlinked.
        for _h in candidate_hyps:
            _h.analytical_identity = hypothesis_identity_for(final_contract, _h)

        pending_events: List[Dict[str, Any]] = []

        def _queue_event(event_type: str, payload: Dict[str, Any]) -> None:
            pending_events.append({"event_type": event_type, "payload": payload})

        def _flush_pending_events() -> None:
            while pending_events:
                evt = pending_events.pop(0)
                self.checkpointer.record_event(
                    investigation_id=investigation_id,
                    execution_id=exec_id,
                    event_type=evt["event_type"],
                    payload=evt["payload"],
                )
                # Persist phase transitions in the authoritative contract.
                evt_phase = evt.get("payload", {}).get("phase")
                if evt_phase:
                    try:
                        with self.session_factory() as phase_session:
                            set_phase(phase_session, investigation_id, str(evt_phase), state_patch={
                                "event_type": evt["event_type"],
                                "execution_id": exec_id,
                            })
                    except Exception as phase_exc:
                        self.checkpointer.record_event(
                            investigation_id=investigation_id,
                            execution_id=exec_id,
                            event_type="investigation.contract_phase_persistence_failed",
                            payload={"phase": evt_phase, "error": str(phase_exc)},
                        )

        def _leading_hypothesis(hypotheses):
            """Deterministically select a leader; ties are resolved by identity."""
            ordered = list(hypotheses or [])
            if not ordered:
                return None
            return sorted(
                ordered,
                key=lambda h: (
                    -float(getattr(h, "posterior_probability", 0.0)),
                    str(getattr(h, "canonical_identity", "") or ""),
                    str(getattr(h, "hypothesis_code", "") or ""),
                ),
            )[0]

        prediction_seq = [0]

        def _next_prediction_id() -> str:
            prediction_seq[0] += 1
            return f"{investigation_id[:8]}_PRED-{prediction_seq[0]:03d}"

        def _synthesize_and_register_prediction(session: Session, h: PredictiveHypothesis, step_idx: int) -> StructuredPrediction:
            pred = PredictionSynthesizer.synthesize_prediction_for_hypothesis(
                hypothesis=h,
                prediction_id=_next_prediction_id(),
                hypothesis_entity_id=f"{investigation_id}_{h.hypothesis_code}",
            )
            pred_entity = Prediction(
                id=pred.prediction_id,
                hypothesis_id=pred.hypothesis_id,
                statement=pred.statement,
                expected_direction=pred.expected_direction or "invariant",
                target_metric=pred.target_metric,
                target_dimension=pred.target_dimension,
                expected_value=str(pred.expected_value) if pred.expected_value is not None else None,
                threshold=pred.threshold,
                expected_relationship=pred.expected_relationship,
                expected_effect=pred.expected_effect,
                confidence=pred.confidence,
                testability=pred.testability,
                status="PENDING",
            )
            session.merge(pred_entity)
            state_mgr.create_prediction(pred)
            if pred.prediction_id not in h.prediction_ids:
                h.prediction_ids.append(pred.prediction_id)

            _queue_event(
                "prediction.created",
                {
                    "prediction_id": pred.prediction_id,
                    "hypothesis_code": h.hypothesis_code,
                    "statement": pred.statement,
                    "threshold": pred.threshold,
                    "expected_direction": pred.expected_direction,
                },
            )
            return pred

        def _apply_hypothesis_fields(
            row: Hypothesis,
            resolved: PredictiveHypothesis,
            canonical_code: str,
            generated_reason: Optional[str] = None,
            *,
            preserve_scientific_state: bool = False,
        ) -> None:
            """Populate descriptive hypothesis fields without overwriting accumulated scientific state.

            On first insert, all fields are initialized from the resolved hypothesis.
            On same-identity rediscovery, posterior/belief/status/counts and provenance
            already persisted on the canonical row remain authoritative; only
            descriptive/semantic fields are refreshed and provenance is unioned.
            """
            row.hypothesis_code = canonical_code
            if getattr(resolved, "analytical_identity", "") and not row.analytical_identity:
                row.analytical_identity = resolved.analytical_identity
            row.statement = resolved.claim or row.statement
            row.rationale = resolved.mechanism or row.rationale

            if not preserve_scientific_state:
                row.prior_probability = resolved.prior_probability
                row.posterior_probability = getattr(
                    resolved, "posterior_probability", resolved.prior_probability
                )
                row.belief_state = getattr(resolved, "belief_state", None) or "active"
                row.is_counter_hypothesis = resolved.is_counter_hypothesis
                row.status = "ACTIVE"

            row.target_metric = resolved.target_metric or row.target_metric
            row.target_dimension = resolved.target_dimension or row.target_dimension
            target_value = getattr(resolved, "target_value", None)
            if target_value is not None:
                row.target_value = str(target_value)
            row.mechanism_detail = getattr(resolved, "mechanism_detail", "") or row.mechanism_detail

            if generated_reason and not row.generated_reason:
                row.generated_reason = generated_reason

            # Provenance is append-only under canonical identity. Never replace
            # existing evidence/parent references during rediscovery.
            existing_evidence = list(row.source_evidence_json or [])
            for item in list(getattr(resolved, "source_evidence", []) or []):
                if item not in existing_evidence:
                    existing_evidence.append(item)
            row.source_evidence_json = existing_evidence

            existing_parents = list(row.parent_hypotheses_json or [])
            for item in list(getattr(resolved, "parent_hypotheses", []) or []):
                if item not in existing_parents:
                    existing_parents.append(item)
            row.parent_hypotheses_json = existing_parents

        def _persist_canonical_hypothesis(
            session: Session,
            resolved: PredictiveHypothesis,
            canonical_code: str,
            generated_reason: Optional[str] = None,
        ) -> Hypothesis:
            """The single write path for hypothesis rows. Resolves through the
            database's own UNIQUE(investigation_id, canonical_identity)
            constraint (see hypothesis_persistence.get_or_create_hypothesis_row)
            rather than trusting the in-memory registry's hypothesis_code
            alone -- so even a genuine cross-process race on the same
            proposition still lands on exactly one persisted row. If some
            other writer already won that race, ``resolved`` (and therefore
            every downstream reference in this call, e.g. the prediction
            synthesized right after) is repointed onto the surviving code.
            """
            canonical_identity = getattr(resolved, "canonical_identity", None) or compute_semantic_identity(resolved)
            resolved.canonical_identity = canonical_identity

            def _build() -> Hypothesis:
                entity = Hypothesis(
                    id=f"{investigation_id}_{canonical_code}",
                    investigation_id=investigation_id,
                    canonical_identity=canonical_identity,
                )
                _apply_hypothesis_fields(entity, resolved, canonical_code, generated_reason, preserve_scientific_state=False)
                return entity

            def _update(row: Hypothesis) -> None:
                _apply_hypothesis_fields(row, resolved, row.hypothesis_code, generated_reason, preserve_scientific_state=True)

            row, _created = get_or_create_hypothesis_row(
                session, investigation_id, canonical_identity, _build, _update
            )
            if row.hypothesis_code != canonical_code:
                resolved.hypothesis_code = row.hypothesis_code
                resolved.id = row.id
            return row

        def step2_hypotheses(session: Session):
            for h in candidate_hyps:
                # Route through the single authoritative consolidation path
                # even for the initial competing-hypothesis batch, so H1/H2/H3
                # (or their persisted counterparts) can never silently diverge
                # from whatever identity rule governs later rounds.
                action, canonical_code, resolved = state_mgr.create_hypothesis(h)
                row = _persist_canonical_hypothesis(session, resolved, canonical_code)
                canonical_code = row.hypothesis_code
                _queue_event(
                    "hypothesis.created" if action == "created" else "hypothesis.consolidated",
                    {"hypothesis_code": canonical_code, "statement": resolved.claim, "prior": resolved.prior_probability},
                )
                _synthesize_and_register_prediction(session, resolved, step_idx=2)

            inv_rec = session.query(Investigation).filter(Investigation.id == investigation_id).first()
            if inv_rec:
                inv_rec.entropy_initial = initial_entropy
                inv_rec.entropy_current = initial_entropy

            return {"hypotheses": [h.hypothesis_code for h in candidate_hyps], "initial_entropy": initial_entropy}

        if not reconstructed:
            self.checkpointer.execute_step_transactionally(
                investigation_id=investigation_id,
                execution_id=exec_id,
                step_index=2,
                step_type="HYPOTHESIS_GENERATION",
                step_code="HYP-GEN",
                step_fn=step2_hypotheses,
            )
            _flush_pending_events()
            state_mgr.sync_hypotheses(candidate_hyps)
        else:
            _queue_event(
                "investigation.state.reconstructed",
                {
                    "source": "durable_projections_and_event_stream",
                    "hypotheses": len(state_mgr.get_active_hypotheses()),
                    "executed_experiments": len(state_mgr.get_executed_experiments()),
                },
            )
            _flush_pending_events()

        # 4. Recursive Autonomous Reasoning Loop
        current_hyps = state_mgr.get_active_hypotheses()
        all_verifications_passed = True
        primary_df = data_context.datasets_map[semantic.primary_dataset_name]
        # Real sample statistics for the EIG likelihood model (see
        # likelihood_model.py): computed once from the actual loaded data so
        # concentration/isolation candidates get a genuine, data-dependent
        # expected-information-gain instead of an unavailable/zero value.
        # Cardinality failures (e.g. an unhashable column) degrade to
        # "unknown" for that dimension rather than crashing the investigation.
        _category_cardinalities: Dict[str, int] = {}
        for _dim in set(filter(None, [canonical_group_dimension_col, *(semantic.available_categorical_cols or [])])):
            try:
                if _dim in primary_df.columns:
                    _category_cardinalities[_dim] = int(primary_df[_dim].nunique())
            except Exception:
                pass
        candidate_sample_stats = {
            "row_count": int(len(primary_df)) if primary_df is not None else 0,
            "category_cardinalities": _category_cardinalities,
        }
        step_counter = 3
        max_safety_budget_experiments = int(os.getenv("AAOS_MAX_INVESTIGATION_EXPERIMENTS", str(MAX_INVESTIGATION_EXPERIMENTS)))
        if max_safety_budget_experiments < 1:
            max_safety_budget_experiments = 1
        self.checkpointer.record_event(
            investigation_id=investigation_id, execution_id=exec_id,
            event_type="investigation.runtime_budget_configured",
            payload={"max_runtime_seconds": max_runtime_seconds, "max_experiments": max_safety_budget_experiments},
        )
        last_result_df = None
        last_executed_candidate = None
        unresolved_adversarial_issues: List[Dict[str, Any]] = []
        blocked_experiment_codes: set[str] = set()
        # Tracks every hypothesis code that a candidate experiment has ever
        # actually targeted, across all rounds -- used to distinguish "this
        # counter hypothesis was never directly tested because no experiment
        # was generated for it" (an unavoidable design limitation, must not
        # block stopping) from "an experiment existed to test it and none was
        # ever run against it" (a real gap that should still gate stopping).
        hypothesis_codes_ever_targeted: set[str] = set()
        # Phase 12: most recent missingness/selection-bias sensitivity result
        # for the leading conclusion, refreshed every loop iteration and
        # consulted by both in-loop stopping and the final verdict.
        latest_missingness_result = None
        stopping_reason = "MAX_SAFETY_BUDGET_REACHED"
        authoritative_stop_decision: Optional[StoppingDecision] = None
        # The Universal Question Compiler is the primary scientific planner.
        # The legacy EIG machinery remains the optimizer, but it is only
        # allowed to choose from experiments compatible with the question
        # contract and within its planned scientific budget.
        question_experiment_budget = min(
            max_safety_budget_experiments,
            UniversalQuestionCompiler.experiment_budget(analysis_plan.task, semantics=analysis_plan.semantics),
        )

        def _question_filter_candidates(candidates: List[CandidateExperiment]) -> List[CandidateExperiment]:
            filtered, diagnostic = compute_task_compatible_candidates(candidates, analysis_plan.task, semantic)
            if diagnostic is not None:
                self.checkpointer.record_event(
                    investigation_id=investigation_id,
                    execution_id=exec_id,
                    event_type="investigation.no_compatible_experiment_candidate",
                    payload=diagnostic,
                )
            return filtered

        # Per-round Shannon-entropy trace (round 0 first) consumed by the
        # StoppingEngine's recursive uncertainty gate (see engines/stopping.py).
        entropy_trace: List[float] = [float(initial_entropy)]

        while len(state_mgr.get_executed_experiments()) < question_experiment_budget:
            if _runtime_budget_exceeded("investigation_loop_start"):
                stopping_reason = "RUNTIME_BUDGET_EXCEEDED"
                break
            if cancellation_check and cancellation_check():
                return False

            current_hyps = _stable_hypothesis_order(state_mgr.get_active_hypotheses())
            current_posteriors = [h.posterior_probability for h in current_hyps]
            executed_exp_ids = state_mgr.get_executed_experiments()
            executed_fps = state_mgr.get_executed_fingerprints()

            predictions_by_hyp: Dict[str, str] = {}
            predictions_map: Dict[str, List[str]] = {}
            for h in current_hyps:
                pending_for_h = [
                    p for p in state_mgr.get_predictions_for_hypothesis(h.hypothesis_code)
                    if p.status == "PENDING"
                ]
                if pending_for_h:
                    predictions_by_hyp[h.hypothesis_code] = pending_for_h[0].prediction_id
                    predictions_map[h.hypothesis_code] = [p.prediction_id for p in pending_for_h]

            _queue_event("investigation.progress", {
                "phase": "REPLANNING",
                "executed_experiments": len(executed_exp_ids),
                "max_experiments": max_safety_budget_experiments,
                "elapsed_seconds": round(time.monotonic() - investigation_started_monotonic, 3),
            })
            try:
                replan = ExperimentSynthesizer.dynamically_replan_candidates(
                    hypotheses=current_hyps,
                    semantic=semantic,
                    executed_codes=executed_exp_ids,
                    current_posteriors=current_posteriors,
                    last_result_df=last_result_df,
                    last_experiment=last_executed_candidate,
                    unresolved_adversarial_issues=unresolved_adversarial_issues,
                    time_filter_sql=(temporal_scope.sql_filter(canonical_time_col) if temporal_scope.found and canonical_time_col else None),
                    decision=method_decision,
                    sample_stats=candidate_sample_stats,
                )
                available_candidates = [c for c in replan.candidates if c.code not in blocked_experiment_codes]
                available_candidates = MethodSelectionEngine.bind_candidates(available_candidates, method_decision.selected_method_code)
                try:
                    with self.session_factory() as contract_session:
                        active_hyp_codes = [h.hypothesis_code for h in current_hyps]
                        new_contract_id = supersede_and_replan(
                            contract_session,
                            investigation_id,
                            replan_reason=str(getattr(replan, "trigger_reason", "evidence_changed")),
                            candidate_experiments=[c.to_dict() for c in available_candidates],
                            active_hypotheses=active_hyp_codes,
                            phase=AnalyticalPhase.REPLANNING,
                        )
                        # v20-C4.2.3: subsequent experiments must be validated against and
                        # attributed to the ACTIVE contract, not the superseded parent.
                        contract_id = new_contract_id
                        _adopted = contract_session.query(InvestigationContract).filter(
                            InvestigationContract.id == new_contract_id
                        ).first()
                        contract_version = _adopted.version
                        final_contract = load_final_contract(contract_session, new_contract_id)
                    _queue_event("investigation.contract.replanned", {
                        "contract_id": new_contract_id,
                        "candidate_count": len(available_candidates),
                        "reason": str(getattr(replan, "trigger_reason", "evidence_changed")),
                    })
                except Exception as contract_exc:
                    self.checkpointer.record_event(
                        investigation_id=investigation_id, execution_id=exec_id,
                        event_type="investigation.contract_replan_failed",
                        payload={"error": str(contract_exc)},
                    )
                    stopping_reason = "CONTRACT_REPLAN_FAILED"
                    break
            except Exception as adaptive_exc:
                # Dynamic replanning failed; record why before falling back to
                # non-adaptive candidate synthesis so this degradation is
                # observable instead of silently swallowed.
                self.checkpointer.record_event(
                    investigation_id=investigation_id,
                    execution_id=exec_id,
                    event_type="investigation.adaptive_replan_failed",
                    payload={
                        "reason": f"replan_failed:{type(adaptive_exc).__name__}",
                        "message": str(adaptive_exc),
                    },
                )
                available_candidates = [
                    c for c in ExperimentSynthesizer.synthesize_candidate_experiments(
                        current_hyps,
                        semantic,
                        unresolved_adversarial_issues=unresolved_adversarial_issues,
                        predictions_by_hyp=predictions_by_hyp,
                        predictions_map=predictions_map,
                        time_filter_sql=(temporal_scope.sql_filter(canonical_time_col) if temporal_scope.found and canonical_time_col else None),
                        decision=method_decision,
                        sample_stats=candidate_sample_stats,
                    )
                    if not state_mgr.has_experiment_been_executed(c.code, c.fingerprint)
                    and c.code not in blocked_experiment_codes
                ]
                available_candidates = MethodSelectionEngine.bind_candidates(available_candidates, method_decision.selected_method_code)
                replan = None

            available_candidates = _question_filter_candidates(available_candidates)

            # Register any emergent hypotheses discovered from evidence patterns.
            # replan.new_hypotheses has already been through same-batch
            # consolidation inside HypothesisSynthesizer.synthesize_emergent_hypotheses;
            # here each surviving candidate is checked against the EXISTING
            # (cross-round) registry through the same authoritative merge path.
            if replan is not None and replan.new_hypotheses:
                for new_h in replan.new_hypotheses:
                    original_code = new_h.hypothesis_code
                    action, canonical_code, resolved = state_mgr.expand_hypothesis_space(new_h)

                    if action == "merged" and canonical_code != original_code:
                        # Re-point any freshly-synthesized candidate experiments
                        # that were planned against the now-retired code onto
                        # the surviving canonical hypothesis instead of
                        # dropping them -- the informative test is still valid,
                        # it just targets the pre-existing claim.
                        for cand in available_candidates:
                            if getattr(cand, "target_hypothesis_code", None) == original_code:
                                cand.target_hypothesis_code = canonical_code
                            if hasattr(cand, "hypothesis_ids") and cand.hypothesis_ids:
                                cand.hypothesis_ids = [
                                    canonical_code if hid == original_code else hid
                                    for hid in cand.hypothesis_ids
                                ]

                    # Refresh from the registry rather than hand-mutating a
                    # local list, so consolidation state (merged evidence,
                    # consolidation_log, posterior_history) is never stale.
                    current_hyps = _stable_hypothesis_order(state_mgr.get_active_hypotheses())

                    if action == "created":
                        # expand_hypothesis_space has already allocated bounded model
                        # mass while preserving the accumulated posterior odds and
                        # original prior semantics. Never re-normalize by simply
                        # replacing priors with posteriors here.
                        current_hyps = _stable_hypothesis_order(state_mgr.get_active_hypotheses())
                        resolved = state_mgr.get_hypothesis(canonical_code) or resolved

                    def persist_hypothesis_registration(session: Session, h=resolved, act=action):
                        pre_persist_code = h.hypothesis_code
                        # _persist_canonical_hypothesis resolves through the
                        # database's own UNIQUE(investigation_id,
                        # canonical_identity) constraint (not just the
                        # in-memory registry's `act`/`canonical_code`), and
                        # when act == "merged" this updates the SURVIVOR's
                        # existing row (with the now-unioned provenance
                        # fields) rather than inserting a second row for the
                        # absorbed hypothesis.
                        row = _persist_canonical_hypothesis(
                            session, h, pre_persist_code,
                            generated_reason=getattr(h, "generated_reason", None) or replan.trigger_reason,
                        )
                        if row.hypothesis_code != pre_persist_code:
                            # Lost a DB-level race the in-memory registry
                            # couldn't see -- repoint any candidates already
                            # targeting the pre-race code onto the real
                            # surviving row, same as the in-memory "merged"
                            # repoint above.
                            for cand in available_candidates:
                                if getattr(cand, "target_hypothesis_code", None) == pre_persist_code:
                                    cand.target_hypothesis_code = row.hypothesis_code
                                if hasattr(cand, "hypothesis_ids") and cand.hypothesis_ids:
                                    cand.hypothesis_ids = [
                                        row.hypothesis_code if hid == pre_persist_code else hid
                                        for hid in cand.hypothesis_ids
                                    ]
                        act = "created" if act == "created" and row.hypothesis_code == pre_persist_code else "merged"
                        _queue_event(
                            "hypothesis.created" if act == "created" else "hypothesis.consolidated",
                            {
                                "hypothesis_code": h.hypothesis_code,
                                "statement": h.claim,
                                "prior": h.prior_probability,
                                "parent_hypotheses": list(getattr(h, "parent_hypotheses", []) or []),
                                "consolidated_evidence_contributions": len(getattr(h, "consolidation_log", []) or []),
                            },
                        )
                        new_pred = _synthesize_and_register_prediction(session, h, step_idx=step_counter)
                        for cand in available_candidates:
                            if cand.target_hypothesis_code == h.hypothesis_code:
                                cand.target_prediction_id = new_pred.prediction_id
                                if new_pred.prediction_id not in cand.prediction_ids:
                                    cand.prediction_ids.append(new_pred.prediction_id)
                        return {"hypothesis": h.hypothesis_code, "action": act, "new_prediction": new_pred.prediction_id}

                    self.checkpointer.execute_step_transactionally(
                        investigation_id=investigation_id,
                        execution_id=exec_id,
                        step_index=step_counter,
                        step_type="ADAPTIVE_HYPOTHESIS_REVISION" if action == "created" else "ADAPTIVE_HYPOTHESIS_CONSOLIDATION",
                        step_code=f"REVISE-{resolved.hypothesis_code}",
                        step_fn=persist_hypothesis_registration,
                    )
                    _flush_pending_events()
                    step_counter += 1

            # Filter candidates against authoritative executed experiments and fingerprints
            available_candidates = [
                c for c in available_candidates
                if not state_mgr.has_experiment_been_executed(c.code, c.fingerprint)
            ]

            if not available_candidates:
                stopping_reason = "NO_INFORMATIVE_EXPERIMENT_REMAINS"
                break

            # In-Loop Adversarial Challenge if a leading explanation emerges
            leading_h = _leading_hypothesis(current_hyps)
            if leading_h.posterior_probability >= 0.65 and len(current_hyps) >= 2:
                counter_hyps = [h for h in current_hyps if h.is_counter_hypothesis]
                counter_h = _stable_hypothesis_order(counter_hyps)[0] if counter_hyps else _stable_hypothesis_order([h for h in current_hyps if h.hypothesis_code != leading_h.hypothesis_code])[0]
                attack_res = AdversarialAttacker.execute_adversarial_attack(
                    df=primary_df,
                    leading_hypothesis=leading_h,
                    counter_hypothesis=counter_h,
                    semantic=semantic,
                    execution_provider=self.execution_provider,
                )
                if attack_res.simpsons_paradox_detected:
                    conf_dim = attack_res.details.get("secondary_dimension")
                    if conf_dim and not any(issue.get("confounding_dimension") == conf_dim for issue in unresolved_adversarial_issues):
                        unresolved_adversarial_issues.append({"confounding_dimension": conf_dim, "type": "simpsons_paradox"})
                        cond_candidate, cond_diagnostic = build_simpsons_conditional_candidate(
                            semantic, conf_dim, leading_h.hypothesis_code,
                        )
                        if cond_diagnostic is not None:
                            unresolved_adversarial_issues.append(cond_diagnostic)
                        if cond_candidate is not None:
                            available_candidates.insert(0, cond_candidate)

            available_candidates = _question_filter_candidates(available_candidates)
            # P0-B: the fail-closed _question_filter_candidates above may now
            # legitimately return [] (no incompatible candidate is silently
            # restored). Without this guard, EIGOptimizer.select_next_experiment
            # would raise on an empty candidate list instead of the controller
            # replanning/stopping explicitly.
            if not available_candidates:
                stopping_reason = "NO_COMPATIBLE_EXPERIMENT_CANDIDATE"
                break
            hypothesis_codes_ever_targeted.update(
                c.target_hypothesis_code for c in available_candidates if getattr(c, "target_hypothesis_code", None)
            )

            # Select next optimal test via EIG & Multi-Objective scoring
            selected_exp, selection_exp = EIGOptimizer.select_next_experiment(
                available_candidates,
                current_hyps,
                executed_codes=state_mgr.get_executed_experiments(),
                executed_fingerprints=state_mgr.get_executed_fingerprints(),
            )
            binding_ok, binding_detail = MethodSelectionEngine.assert_candidate_binding(
                selected_exp, method_decision.selected_method_code
            )
            if not binding_ok:
                self.checkpointer.record_event(
                    investigation_id=investigation_id, execution_id=exec_id,
                    event_type="investigation.method_binding.blocked",
                    payload={"experiment_code": selected_exp.code, "selected_method": method_decision.selected_method_code, "detail": binding_detail},
                )
                blocked_experiment_codes.add(selected_exp.code)
                all_verifications_passed = False
                stopping_reason = "METHOD_EXECUTOR_BINDING_BLOCKED"
                continue
            self.checkpointer.record_event(
                investigation_id=investigation_id, execution_id=exec_id,
                event_type="investigation.method_binding.approved",
                payload={"experiment_code": selected_exp.code, "method_code": method_decision.selected_method_code, "executor_id": binding_detail},
            )

            # v20-A hard execution gate: no experiment may reach the executor
            # without passing validate_experiment_against_contract() against
            # the canonical contract for this round.
            # v20-C4.2.3: an UNASSIGNED experiment is no longer silently promoted to
            # PRIMARY.  Every synthesis site tags its role; an untagged candidate
            # fails validation (EXPERIMENT_ROLE_UNASSIGNED) instead of executing as
            # the contract's primary test by accident.
            _hyp_identity_by_code = {
                h.hypothesis_code: hypothesis_identity_for(final_contract, h)
                for h in list(candidate_hyps) + list(current_hyps)
            }
            stamp_experiment_identity(final_contract, selected_exp, _hyp_identity_by_code)
            # Runtime authority for the PRIMARY-per-analytical-identity invariant
            # (the partial-unique DB index uq_experiments_primary_identity is the
            # final concurrency guard).  An analytical identity has exactly ONE
            # PRIMARY experiment: the first test that answers the contract.  Any
            # later candidate that is a *different* test (identical code/fingerprint
            # candidates were already filtered out above) but carries the same
            # identity is a follow-up refinement of that claim -- e.g. the segment
            # isolation test after the concentration screen -- and is executed as
            # SUPPORTING evidence instead of being dropped or written as a second
            # PRIMARY.  Its Bayesian weight and provenance are unchanged; the
            # demotion is persisted and audited, so nothing is silent.
            if selected_exp.experiment_role == ExperimentRole.PRIMARY.value and selected_exp.analytical_identity:
                existing_primary_id = (
                    session.query(Experiment.id)
                    .filter(
                        Experiment.investigation_id == investigation_id,
                        Experiment.experiment_role == ExperimentRole.PRIMARY.value,
                        Experiment.analytical_identity == selected_exp.analytical_identity,
                    )
                    .first()
                )
                if existing_primary_id:
                    self.checkpointer.record_event(
                        investigation_id=investigation_id, execution_id=exec_id,
                        event_type="investigation.primary_identity.demoted_to_supporting",
                        payload={
                            "analytical_identity": selected_exp.analytical_identity,
                            "existing_primary_experiment_id": existing_primary_id[0],
                            "experiment_code": selected_exp.code,
                            "reason": (
                                "PRIMARY analytical identity already has a persisted PRIMARY experiment; "
                                "this later, distinct test is executed as SUPPORTING follow-up evidence."
                            ),
                        },
                    )
                    selected_exp.experiment_role = ExperimentRole.SUPPORTING.value
            contract_method_family = getattr(method_decision, "method_family", None)
            contract_estimand = ""
            if method_decision.selected_method_code:
                _cap = MethodRegistry.get(method_decision.selected_method_code)
                contract_estimand = getattr(_cap, "estimand", "") if _cap else ""
            contract_snapshot = AnalyticalContractSnapshot(
                contract_id=contract_id,
                contract_version=contract_version,
                problem_class=final_contract.canonical_task,
                method_family=contract_method_family,
                selected_method=final_contract.selected_method_code,
                estimand=contract_estimand,
                semantic_bindings=final_binding_set,
                final_contract=final_contract,
                population_scope="primary_dataset",
                grain=analysis_plan.semantics.grain,
                time_scope=(
                    {"time_column": analysis_plan.semantics.time_column}
                    if analysis_plan.semantics.time_column else None
                ),
            )
            contract_validation = validate_experiment_against_contract(selected_exp, contract_snapshot)
            if not contract_validation.compatible:
                self.checkpointer.record_event(
                    investigation_id=investigation_id, execution_id=exec_id,
                    event_type="investigation.contract_validation.blocked",
                    payload={
                        "experiment_code": selected_exp.code,
                        "experiment_role": selected_exp.experiment_role,
                        "contract_id": contract_id,
                        "contract_version": contract_version,
                        **contract_validation.to_dict(),
                    },
                )
                blocked_experiment_codes.add(selected_exp.code)
                all_verifications_passed = False
                stopping_reason = "INCOMPATIBLE_WITH_ANALYTICAL_CONTRACT"
                continue
            self.checkpointer.record_event(
                investigation_id=investigation_id, execution_id=exec_id,
                event_type="investigation.contract_validation.approved",
                payload={
                    "experiment_code": selected_exp.code,
                    "experiment_role": selected_exp.experiment_role,
                    "contract_id": contract_id,
                    "contract_version": contract_version,
                    "warnings": contract_validation.warnings,
                },
            )
            selected_exp.contract_id = contract_id
            selected_exp.contract_version = contract_version

            last_executed_candidate = selected_exp
            exp_code = selected_exp.code
            _queue_event("investigation.progress", {
                "phase": "EXECUTION_SELECTED",
                "experiment_code": exp_code,
                "executed_experiments": len(executed_exp_ids),
                "candidate_count": len(available_candidates),
                "elapsed_seconds": round(time.monotonic() - investigation_started_monotonic, 3),
            })
            _flush_pending_events()
            if _runtime_budget_exceeded(f"before_{exp_code}"):
                stopping_reason = "RUNTIME_BUDGET_EXCEEDED"
                break

            # Canonical relational-plan authority.
            #
            # When a CandidateExperiment carries a typed RelationalPlan,
            # that plan is the source of truth. SQL is compiled FROM the
            # plan and is therefore only the execution representation.
            #
            # Legacy experiments without relational_plan retain the existing
            # query_sql / analytical-compilation path for compatibility.
            relational_plan = getattr(
                selected_exp,
                "relational_plan",
                None,
            )

            if relational_plan is not None:
                plan_errors = relational_plan.validate()
                if plan_errors:
                    raise ValueError(
                        "RelationalPlan validation failed for "
                        f"{exp_code}: {'; '.join(plan_errors)}"
                    )

                effective_sql = relational_plan.to_sql()

                # The typed plan is authoritative for declared join hops.
                # Keep join_hops synchronized as compatibility metadata so
                # downstream audit records remain consistent.
                plan_hops = [
                    {
                        "left_table": hop.left_table,
                        "right_table": hop.right_table,
                        "left_key": hop.left_key,
                        "right_key": hop.right_key,
                        "join_type": hop.join_type,
                    }
                    for hop in (relational_plan.hops or [])
                ]

                selected_exp.join_hops = plan_hops

                selected_exp.executable_spec = dict(
                    getattr(selected_exp, "executable_spec", {}) or {}
                )
                selected_exp.executable_spec["relational_plan"] = plan_to_dict(relational_plan)

            elif selected_exp.query_sql.startswith("SELECT"):
                effective_sql = selected_exp.query_sql

            else:
                effective_sql = self.compile_experiment_intent(
                    selected_exp,
                    semantic,
                    investigation_id,
                )
            # DEFECT-014 closure: JOIN-bearing experiments MUST carry explicit
            # hop metadata and pass the same cardinality/key/type safety proof
            # against the ACTUAL datasets before DuckDB executes the SQL.
            # Never infer safety from the SQL string alone and never treat an
            # unknown join as safe.
            join_hops_for_safety = (
                [
                    {
                        "left_table": hop.left_table,
                        "right_table": hop.right_table,
                        "left_key": hop.left_key,
                        "right_key": hop.right_key,
                        "join_type": hop.join_type,
                    }
                    for hop in (relational_plan.hops or [])
                ]
                if relational_plan is not None
                else (
                    getattr(selected_exp, "join_hops", None)
                    or getattr(
                        selected_exp,
                        "executable_spec",
                        {},
                    ).get("join_hops", [])
                )
            )

            join_reports = assess_declared_join_hops(
                data_context.datasets_map,
                join_hops_for_safety,
                authorize_many_to_many=False,
            )
            if "JOIN" in effective_sql.upper():
                if not join_reports:
                    blocked_experiment_codes.add(exp_code)
                    stopping_reason = "JOIN_SAFETY_METADATA_MISSING"
                    all_verifications_passed = False
                    continue
                unsafe_join_reports = [r for r in join_reports if r.status != SafetyStatus.SAFE]
                if unsafe_join_reports:
                    blocked_experiment_codes.add(exp_code)
                    stopping_reason = "JOIN_SAFETY_GATE_BLOCKED"
                    all_verifications_passed = False
                    # Keep an auditable record of the blocked analytical path.
                    self.checkpointer.record_event(
                        investigation_id,
                        "relational.join.safety.blocked",
                        {
                            "experiment_code": exp_code,
                            "effective_sql": effective_sql,
                            "join_safety_reports": [
                            {
                                "left_table": r.left_table,
                                "right_table": r.right_table,
                                "left_key": r.left_key,
                                "right_key": r.right_key,
                                "cardinality": getattr(r.cardinality, "value", r.cardinality),
                                "status": getattr(r.status, "value", r.status),
                                "fanout_risk": r.fanout_risk,
                                "key_type_compatible": r.key_type_compatible,
                                "keys_exist": r.keys_exist,
                                "reason": r.reason,
                            }
                                for r in join_reports
                            ],
                        },
                    )
                    continue

            is_grouped = "GROUP BY" in effective_sql.upper()

            # Evidence provenance must identify only the physical datasets that
            # actually contributed to this experiment. The investigation scope
            # may contain additional datasets that were selected by the analyst
            # but not used by this particular claim. For relational experiments,
            # the typed join hops are the authoritative source list; for
            # single-table experiments the semantic primary dataset is the
            # complete source. Never label every selected dataset as evidence
            # merely because it was available in the workspace.
            experiment_source_datasets = {semantic.primary_dataset_name}
            if relational_plan is not None:
                for hop in (relational_plan.hops or []):
                    experiment_source_datasets.add(hop.left_table)
                    experiment_source_datasets.add(hop.right_table)
            experiment_source_datasets = sorted(
                name for name in experiment_source_datasets
                if name in data_context.datasets_map
            )
            experiment_source_fingerprints = {
                name: data_context.dataset_fingerprints[name]
                for name in experiment_source_datasets
                if name in data_context.dataset_fingerprints
            }

            # Persist the EIG actually computed by the optimizer, never the legacy
            # discriminating-power heuristic. Recompute from the same current
            # posterior and explicit predictive likelihoods used for selection.
            ordered_hypotheses = sorted(
                current_hyps,
                key=lambda h: (
                    str(getattr(h, "canonical_identity", "") or ""),
                    str(getattr(h, "hypothesis_code", "") or ""),
                ),
            )
            hyp_index_map = {h.hypothesis_code: i for i, h in enumerate(ordered_hypotheses)}
            selected_target_idx = hyp_index_map.get(selected_exp.target_hypothesis_code)
            selected_eig = 0.0
            if selected_target_idx is not None and selected_exp.likelihood_if_true is not None and selected_exp.likelihood_if_false is not None:
                selected_eig = EIGOptimizer.compute_expected_entropy_reduction(
                    current_probs=[h.posterior_probability for h in ordered_hypotheses],
                    target_hyp_index=selected_target_idx,
                    discriminating_power=0.0,
                    likelihood_if_true=selected_exp.likelihood_if_true,
                    likelihood_if_false=selected_exp.likelihood_if_false,
                )

            def run_selected_experiment(session: Session):
                _target_hyp = next(
                    (h for h in ordered_hypotheses if h.hypothesis_code == selected_exp.target_hypothesis_code), None
                )
                exp_entity = Experiment(
                    id=f"{investigation_id}_{exp_code}",
                    investigation_id=investigation_id,
                    experiment_role=selected_exp.experiment_role,
                    analytical_identity=selected_exp.analytical_identity or None,
                    hypothesis_canonical_identity=(getattr(_target_hyp, "canonical_identity", None) or None),
                    contract_id=contract_id,
                    hypothesis_id=f"{investigation_id}_{selected_exp.target_hypothesis_code}",
                    target_prediction_id=selected_exp.target_prediction_id,
                    test_code=exp_code,
                    tool_name=selected_exp.tool_name,
                    arguments_json={
                        "sql": effective_sql,
                        "relational_plan": plan_to_dict(relational_plan) if relational_plan is not None else None,
                        "selected_method_code": method_decision.selected_method_code,
                        "executor_id": (MethodRegistry.executor_id(method_decision.selected_method_code) if method_decision.selected_method_code else None),
                        # Phase 11 Part 14: full metric-semantics decision trail
                        # attached to the experiment record that used it, so a
                        # reviewer can see exactly what metric was identified,
                        # why that aggregation was chosen, what numerator/
                        # denominator/weight was used, and how it was verified
                        # -- not merely the final number.
                        "metric_semantics": (
                            semantic.metric_definition.to_provenance_dict()
                            if semantic.metric_definition else None
                        ),
                        "join_hops": [dict(h) for h in (getattr(selected_exp, "join_hops", None) or [])],
                        "source_datasets": list(experiment_source_datasets),
                        "source_dataset_fingerprints": dict(experiment_source_fingerprints),
                        "join_safety_preflight": [
                            {
                                "left_table": r.left_table,
                                "right_table": r.right_table,
                                "left_key": r.left_key,
                                "right_key": r.right_key,
                                "cardinality": r.cardinality.value,
                                "status": r.status.value,
                                "fanout_risk": r.fanout_risk,
                                "reason": r.reason,
                            }
                            for r in join_reports
                        ],
                    },
                    rationale=selection_exp,
                    expected_information_gain=float(selected_eig),
                    adversarial_value=getattr(selected_exp, "adversarial_value", 0.0),
                    robustness_value=getattr(selected_exp, "robustness_value", 0.0),
                    causal_value=getattr(selected_exp, "causal_value", 0.0),
                    selection_rationale=selected_exp.selection_rationale,
                    test_cost=selected_exp.estimated_cost,
                    test_reliability=selected_exp.reliability_weight,
                    status="EXECUTED",
                )
                session.merge(exp_entity)

                # Explicitly freeze analysis-scope metadata before execution.
                scope_record = {
                    "experiment_id": exp_entity.id,
                    "analysis_row_count": int(len(primary_df) if primary_df is not None else 0),
                    "full_scope": bool(getattr(selected_exp, "full_scope", True)),
                    "sampling_policy": str(getattr(selected_exp, "sampling_policy", "NONE") or "NONE"),
                    "sampling_reason": getattr(selected_exp, "sampling_reason", None),
                    "sampling_method": getattr(selected_exp, "sampling_method", None),
                }
                experiment_scope_records.append(scope_record)
                exp_entity.arguments_json = {**(exp_entity.arguments_json or {}), "analysis_scope": scope_record}

                # Execute primary query via DuckDB
                exec_res = self.execution_provider.execute_query(
                    query=effective_sql,
                    data_context=data_context,
                    primary_table=semantic.primary_dataset_name,
                    aggregation_type=getattr(selected_exp, "aggregation_type", None),
                    primary_result_column=getattr(selected_exp, "primary_result_column", None),
                )
                nonlocal last_result_df
                last_result_df = exec_res.result_df

                primary_val = float(exec_res.primary_value)
                row_cnt = int(exec_res.row_count)
                dur_ms = float(exec_res.execution_time_ms)
                if getattr(exec_res, "primary_result_column", None):
                    exp_entity.arguments_json = {
                        **(exp_entity.arguments_json or {}),
                        "primary_result_column": exec_res.primary_result_column,
                    }

                # Execute Canonical 7-Step Post-Execution Transition
                target_pids = list(selected_exp.prediction_ids) if selected_exp.prediction_ids else ([selected_exp.target_prediction_id] if selected_exp.target_prediction_id else [])
                # DEFECT-005: pass the target hypothesis's claim_type through
                # so ScientificTransitionService can use real ANOVA (with
                # subgroup sample-adequacy) for OBSERVATION/segmentation
                # hypotheses instead of the concentration-style share-of-
                # total heuristic, which produced a false-positive DIAGNOSED
                # verdict on a dataset with no genuine statistical
                # heterogeneity (verified independently: scipy ANOVA
                # p=0.886) purely because one category happened to hold a
                # larger share of a tiny, sparsely-populated total.
                _target_hyp = next((h for h in current_hyps if h.hypothesis_code == selected_exp.target_hypothesis_code), None)

                # FIX-02: scientific evidence identity is independent of the
                # experiment UUID/hypothesis. Identical computation against the
                # same dataset snapshot and scope is ONE evidence item and may
                # contribute at most one Bayesian update within this investigation.
                _metric_formula = getattr(semantic.metric_definition, "sql_formula", None)
                _identity_parameters = {
                    "aggregation_type": selected_exp.aggregation_type,
                    "metric": canonical_target_col,
                    "dimension": selected_exp.target_dimension or canonical_group_dimension_col,
                    "grouped": bool(is_grouped),
                    "unit_of_analysis_keys": list(unit_of_analysis_keys),
                    "method_code": getattr(method_decision, "selected_method_code", None),
                }
                evidence_identity = compute_evidence_identity(
                    canonical_dataset_identity(experiment_source_fingerprints),
                    effective_sql,
                    "duckdb_sql",
                    semantic_scope=_identity_parameters,
                    estimator=getattr(method_decision, "selected_method_code", None),
                    formula=_metric_formula,
                    parameters=_identity_parameters,
                    output_columns=list(exec_res.result_df.columns) if exec_res.result_df is not None else [],
                    input_row_count=len(primary_df) if primary_df is not None else row_cnt,
                    output_row_count=len(exec_res.result_df) if exec_res.result_df is not None else 0,
                )
                # Store only the ID (not the full ORM object) so that the
                # object cannot become detached/expired across the
                # begin_nested() SAVEPOINT boundary below. The live row will
                # be fetched again inside the else-branch via session.get().
                # (Named `existing_evidence`, holding just the scalar id, to
                # match the check-then-act pair described below.)
                existing_evidence = (
                    session.query(Evidence.id)
                    .filter(
                        Evidence.investigation_id == investigation_id,
                        Evidence.evidence_identity_hash == evidence_identity,
                    )
                    .scalar()
                )
                evidence_is_duplicate = existing_evidence is not None

                transition = ScientificTransitionService.apply_post_execution_transition(
                    state_mgr,
                    experiment_id=exp_entity.id,
                    experiment_code=exp_code,
                    target_prediction_ids=target_pids,
                    target_hypothesis_code=selected_exp.target_hypothesis_code,
                    result_df=exec_res.result_df,
                    primary_value=primary_val,                    row_count=row_cnt,
                    duration_ms=dur_ms,
                    primary_df=primary_df,
                    target_metric_col=canonical_target_col,
                    group_dimension_col=selected_exp.target_dimension or canonical_group_dimension_col,
                    aggregation_type=selected_exp.aggregation_type,
                    effective_sql=effective_sql,
                    is_grouped=is_grouped,
                    unit_of_analysis_keys=unit_of_analysis_keys,
                    metric_definition=semantic.metric_definition,
                    dataset_fingerprints=experiment_source_fingerprints,
                    primary_result_truncated=bool(getattr(exec_res, "is_truncated", False)),
                    claim_type=getattr(_target_hyp, "claim_type", None),
                    # Thread the already-resolved semantic world model into the
                    # canonical transition. Churn identifiability must use the
                    # positively-resolved churn event binding rather than
                    # re-inferring it from column names. Omitting this argument
                    # silently disabled that authority on every controller run.
                    semantic=semantic,
                    relational_plan=relational_plan,
                    relation_tables=data_context.datasets_map,
                    join_safety_reports=join_reports,
                    evidence_identity=evidence_identity,
                    evidence_is_duplicate=evidence_is_duplicate,
                    primary_result_column=getattr(selected_exp, "primary_result_column", None) or getattr(exec_res, "primary_result_column", None),
                )

                if transition.verification_status != "VERIFIED":
                    nonlocal all_verifications_passed
                    all_verifications_passed = False

                # See the FORECAST_GENERALIZATION note near the assumption
                # ledger construction above: a rolling-origin backtest that
                # shows the fitted trend materially beating a naive baseline
                # out-of-sample is direct empirical evidence that historical
                # patterns generalize to the forecast horizon for this
                # dataset. Require the verified/duplicate-safe path (mirrors
                # the all_verifications_passed check above) so a failed or
                # skipped recomputation cannot count as validating evidence.
                if (
                    transition.verification_status == "VERIFIED"
                    and getattr(transition, "forecast_backtest_trend_mae", None) is not None
                    and getattr(transition, "forecast_backtest_naive_mae", None) is not None
                    and transition.forecast_backtest_trend_mae < transition.forecast_backtest_naive_mae
                ):
                    nonlocal forecast_generalization_validated
                    forecast_generalization_validated = True

                # Enrich the targeted hypothesis with the concrete driving
                # value once evidence names it. Without this, an investigation
                # can reach a correctly DIAGNOSED, dual-engine-verified verdict
                # and still never say WHICH category is responsible: the
                # concentration experiment's structured observation already
                # computes the dominant value (structured_result['top_value']),
                # but nothing previously propagated it onto the hypothesis, so
                # the final direct_answer stayed generic ("concentrated within
                # specific high-volume categories") instead of naming it.
                structured_res = transition.raw_observation.structured_result or {}
                top_value = structured_res.get("top_value")
                top_share_pct = structured_res.get("top_share_pct")
                target_hyp = state_mgr.get_hypothesis(selected_exp.target_hypothesis_code)
                if (
                    target_hyp is not None
                    and top_value
                    and top_share_pct is not None
                    and not getattr(target_hyp, "target_value", None)
                ):
                    target_hyp.target_value = top_value
                    target_hyp.claim = (
                        f"{target_hyp.claim} Specifically, '{top_value}' accounts for "
                        f"{top_share_pct:.1f}% of the observed {canonical_target_col}."
                    )
                    hyp_row = session.query(Hypothesis).filter(
                        Hypothesis.id == f"{investigation_id}_{target_hyp.hypothesis_code}"
                    ).first()
                    if hyp_row:
                        hyp_row.target_value = str(top_value)
                        hyp_row.statement = target_hyp.claim
                elif (
                    target_hyp is not None
                    and top_value
                    and structured_res.get("is_additive") is False
                    and not getattr(target_hyp, "target_value", None)
                ):
                    # Phase 11 Part 12: non-additive metric (rate/ratio/mean/
                    # weighted-mean) -- narrate as "highest value", never as a
                    # share of a summed total, which would misstate the metric.
                    top_metric_value = structured_res.get("top_metric_value")
                    target_hyp.target_value = top_value
                    metric_label = semantic.metric_definition.semantic_type if semantic.metric_definition else "metric"
                    target_hyp.claim = (
                        f"{target_hyp.claim} Specifically, '{top_value}' has the highest observed "
                        f"{canonical_target_col} ({metric_label} = {top_metric_value}), "
                        f"not merely the largest share of a summed total."
                    )
                    hyp_row = session.query(Hypothesis).filter(
                        Hypothesis.id == f"{investigation_id}_{target_hyp.hypothesis_code}"
                    ).first()
                    if hyp_row:
                        hyp_row.target_value = str(top_value)
                        hyp_row.statement = target_hyp.claim

                # Store Observation Model
                trace_payload = transition.calculation_trace or {}
                exp_entity.arguments_json = {
                    **(exp_entity.arguments_json or {}),
                    "calculation_trace_id": trace_payload.get("trace_id"),
                    "calculation_trace_hash": trace_payload.get("canonical_hash"),
                }

                obs_entity = Observation(
                    id=transition.raw_observation.observation_id,
                    experiment_id=exp_entity.id,
                    # Phase 11 Part 14: persist the structured, metric-
                    # semantics-aware finding (top value / share-of-total for
                    # additive metrics, or top/bottom value + spread for
                    # non-additive metrics) alongside the primary value --
                    # previously only the primary scalar was persisted, so a
                    # reviewer inspecting DB state after the run had no way
                    # to see WHAT was concluded, only the raw number.
                    result_json=_json_safe({
                        **transition.raw_observation.result_summary,
                        "structured_result": transition.raw_observation.structured_result,
                        "result_rows": (
                            exec_res.result_df.to_dict(orient="records")
                            if getattr(exec_res, "result_df", None) is not None
                            else []
                        ),
                        "calculation_trace": trace_payload,
                    }),
                    row_count_analyzed=row_cnt,
                    execution_time_ms=dur_ms,
                    sql_executed=effective_sql,
                )
                session.merge(obs_entity)

                # DEFECT-007 / audit P0: resolve the targeted hypothesis BEFORE
                # building the Evidence row, both to link evidence (as before)
                # and to source its confidence_score from the same epistemic
                # engine used for the final verdict, rather than a hard-coded
                # verified/failed switch.
                evidence_target_hyp = state_mgr.get_hypothesis(selected_exp.target_hypothesis_code)

                # Store Evidence Model
                #
                # Audit P0 fix (2026-09-09 forensic pass): this block, and the
                # `ev_entity` it creates/resolves, must run BEFORE "Store
                # Verification Proof" below, since that block references
                # ev_entity.id. It previously ran after, which meant every
                # execution of this branch raised
                # NameError: name 'ev_entity' is not defined.
                # See AAOS_FORENSIC_AUDIT_2026-09-09.md, P0 finding #3.
                ev_statement = f"Empirical evaluation of {selected_exp.target_hypothesis_code} using {exp_code} -> value={primary_val:.2f}."
                if top_value and top_share_pct is not None:
                    ev_statement += f" Leading category: '{top_value}' ({top_share_pct:.1f}% share)."

                # Audit P0 fix: confidence_score used to be a hard-coded
                # 0.95 (VERIFIED) / 0.40 (FAILED) switch, which conflated
                # "did the two computation engines agree?" (a verification
                # question) with "how strong is this evidence?" (an
                # epistemic-strength question). It now comes from
                # EpistemicCalibrationEngine: evidence_strength (grounded in
                # variance explained when a group dimension is available,
                # else a neutral prior) discounted by model_uncertainty
                # (how much the dual-engine paths disagreed). A clean
                # dual-engine agreement no longer manufactures a flat 0.95
                # regardless of how weak the underlying signal actually is,
                # and a disagreement still drives confidence toward zero.
                _posterior_for_evidence = (
                    evidence_target_hyp.posterior_probability if evidence_target_hyp is not None else 0.5
                )
                _var_explained_local = None
                if canonical_group_dimension_col and canonical_group_dimension_col in primary_df.columns:
                    _var_explained_local = BeliefEngine.compute_variance_explained(
                        primary_df, canonical_group_dimension_col, canonical_target_col
                    )
                _evidence_epistemic_vec = EpistemicCalibrationEngine.calibrate_vectors(
                    df=exec_res.result_df,
                    posterior_probability=_posterior_for_evidence,
                    verification_delta_pct=transition.delta_pct,
                    variance_explained_pct=_var_explained_local,
                    multiverse_robustness_pct=0.0,  # not yet assessed at experiment-execution time
                    causal_identifiable=causal_gate_res.is_identifiable,
                )
                _evidence_confidence = round(
                    float(np.clip(
                        _evidence_epistemic_vec.evidence_strength * (1.0 - _evidence_epistemic_vec.model_uncertainty),
                        0.0, 1.0,
                    )),
                    4,
                )

                if existing_evidence is None:
                    # P2-1 fix: the `existing_evidence` SELECT above and this
                    # INSERT are a check-then-act pair. Under genuine
                    # concurrent execution, two executions can both pass the
                    # SELECT before either commits, and the DB's
                    # UniqueConstraint("investigation_id",
                    # "evidence_identity_hash") (apps/api/src/models/entities.py)
                    # would previously raise an unhandled IntegrityError here,
                    # failing the whole experiment step instead of being
                    # treated as "already known" evidence. Insert inside a
                    # SAVEPOINT so a losing race only unwinds this insert --
                    # not the rest of the in-progress transaction (exp_entity,
                    # hypothesis updates, etc. already added to `session`) --
                    # then defer to the winning row, mirroring the same
                    # get-or-create-under-constraint pattern already used for
                    # hypotheses in runtime/hypothesis_persistence.py.
                    ev_entity = Evidence(
                        id=f"EV_{exp_entity.id}",
                        investigation_id=investigation_id,
                        experiment_id=exp_entity.id,
                        hypothesis_id=f"{investigation_id}_{selected_exp.target_hypothesis_code}",
                        statement=ev_statement,
                        evidence_type="DIRECT_MEASUREMENT",
                        analytical_identity=exp_entity.analytical_identity,
                        confidence_score=_evidence_confidence,
                        validation_status=transition.verification_status,
                        evidence_identity_hash=evidence_identity,
                    )
                    try:
                        with session.begin_nested():
                            session.add(ev_entity)
                            session.flush()
                    except IntegrityError:
                        # Lost the race: some other writer committed the
                        # canonical Evidence row for this identity between
                        # our SELECT and our INSERT. Discard our attempted
                        # insert and defer to theirs -- do not retry the
                        # insert, and do not create a second evidence node.
                        winner = (
                            session.query(Evidence)
                            .filter(
                                Evidence.investigation_id == investigation_id,
                                Evidence.evidence_identity_hash == evidence_identity,
                            )
                            .first()
                        )
                        if winner is None:
                            # The constraint fired for a reason other than
                            # this identity (or the survivor isn't yet
                            # visible under this isolation level) -- surface
                            # the original failure rather than silently
                            # dropping the evidence.
                            raise
                        ev_entity = winner
                        evidence_is_duplicate = True
                        transition.evidence_is_duplicate = True
                else:
                    # Reuse the canonical Evidence row. A duplicate experiment
                    # remains auditable through its Observation/experiment edge,
                    # but cannot create a second scientific evidence node.
                    # Fetch a fresh, live ORM instance within this session so
                    # that attribute access on ev_entity.id (used immediately
                    # below for EvidenceVerification) never raises
                    # DetachedInstanceError — we stored only the scalar ID
                    # above specifically to avoid carrying an expired ORM
                    # object across the begin_nested() SAVEPOINT boundary.
                    ev_entity = session.get(Evidence, existing_evidence)
                    if ev_entity is None:
                        # Shouldn't happen (we just read this ID), but be safe.
                        raise RuntimeError(
                            f"Evidence {existing_evidence!r} disappeared between identity check and reuse."
                        )


                # Store Verification Proof
                verif_entity = EvidenceVerification(
                    id=gen_uuid(),
                    evidence_id=ev_entity.id,
                    primary_tool="duckdb_sql",
                    secondary_tool=getattr(transition, "secondary_tool", None) or "polars_vectorized",
                    observed_delta_pct=transition.delta_pct,
                    status=VerificationStatus.VERIFIED if transition.verification_status == "VERIFIED" else VerificationStatus.FAILED,
                )
                session.add(verif_entity)

                # DEFECT-007: link this evidence onto the SPECIFIC hypothesis
                # it targeted (selected_exp.target_hypothesis_code), in-memory,
                # right when it's created -- mirroring exactly what
                # state_reconstruction.py already does when rebuilding state
                # from DB rows after a restart (same verified/effect_size ->
                # supporting, failed -> contradicting rule). Without this, a
                # live (non-restarted) run's in-memory hypothesis never
                # recorded which evidence tested it, so "has this hypothesis
                # actually been directly tested" was unanswerable during the
                # run itself -- only after a restart-and-reconstruct round
                # trip. Consistent behavior pre/post restart is invariant 11.
                if evidence_target_hyp is not None:
                    has_supported_pred = any(p.status == "SUPPORTED" for p in transition.evaluated_predictions)
                    has_refuted_pred = any(p.status == "REFUTED" for p in transition.evaluated_predictions)
                    # Explicit prediction outcomes are the most direct signal
                    # when they exist (churn/segmentation-style experiments
                    # that generate concrete predicted_observables). Many
                    # experiment types (correlation, several segmentation
                    # paths) never generate predictions at all, so falling
                    # through to "no evidence either way" for them was the
                    # root cause of a hypothesis that was genuinely, directly
                    # tested never being recognized as tested. Delegate to
                    # the same canonical classify_evidence_for_hypothesis()
                    # that state_reconstruction.py uses after a restart, so
                    # live-run and post-restart classification can never
                    # diverge (invariant 11).
                    if has_supported_pred:
                        classification = "supporting"
                    elif has_refuted_pred:
                        classification = "contradicting"
                    else:
                        classification = classify_evidence_for_hypothesis(
                            transition.verification_status, ev_entity.effect_size
                        )
                    if classification == "supporting":
                        if ev_entity.id not in evidence_target_hyp.supporting_evidence_ids:
                            evidence_target_hyp.supporting_evidence_ids.append(ev_entity.id)
                        for ch in current_hyps:
                            if (
                                getattr(ch, "is_counter_hypothesis", False)
                                and ch.hypothesis_code != evidence_target_hyp.hypothesis_code
                                and _is_binary_mirror_hypothesis(ch, evidence_target_hyp)
                            ):
                                if ev_entity.id not in ch.contradicting_evidence_ids:
                                    ch.contradicting_evidence_ids.append(ev_entity.id)
                    elif classification == "contradicting":
                        if ev_entity.id not in evidence_target_hyp.contradicting_evidence_ids:
                            evidence_target_hyp.contradicting_evidence_ids.append(ev_entity.id)
                        for ch in current_hyps:
                            if (
                                getattr(ch, "is_counter_hypothesis", False)
                                and ch.hypothesis_code != evidence_target_hyp.hypothesis_code
                                and _is_binary_mirror_hypothesis(ch, evidence_target_hyp)
                            ):
                                if ev_entity.id not in ch.supporting_evidence_ids:
                                    ch.supporting_evidence_ids.append(ev_entity.id)

                # Register in Canonical Evidence Ledger with Epistemic Typing
                claim_type = method_decision.claim_type_for(is_grouped, selected_exp.aggregation_type)
                evidence_ledger.record_claim(
                    claim_statement=ev_statement,
                    claim_type=claim_type,
                    source_experiment_id=exp_code,
                    source_datasets=experiment_source_datasets,
                    source_columns=list(dict.fromkeys([
                        c for c in (
                            [canonical_target_col]
                            + ([canonical_group_dimension_col] if canonical_group_dimension_col else [])
                            + list(getattr(selected_exp, "metrics", []) or [])
                        ) if c
                    ])),
                    row_count_evaluated=row_cnt,
                    computation_proof={
                        "primary_value": primary_val,
                        "delta": transition.delta_pct,
                        "calculation_trace": trace_payload,
                    },
                    verification_status=transition.verification_status,
                    calculation_trace=(
                        CalculationTraceSchema.model_validate(trace_payload)
                        if trace_payload else None
                    ),
                    evidence_identity=evidence_identity,
                    identity_parameters=_identity_parameters,
                )
                self.checkpointer.record_event(
                    investigation_id=investigation_id,
                    execution_id=exec_id,
                    event_type="calculation.trace.created",
                    payload=trace_payload,
                )

                # Graph Edge
                edge = InvestigationGraphEdge(
                    id=gen_uuid(),
                    investigation_id=investigation_id,
                    source_node_type="EXPERIMENT",
                    source_node_id=exp_entity.id,
                    target_node_type="EVIDENCE",
                    target_node_id=ev_entity.id,
                    relationship_type="PRODUCES",
                )
                session.add(edge)

                # Persist evaluated predictions & emit events
                for ev_rec in transition.evaluated_predictions:
                    if ev_rec.prediction_id:
                        pred_entity = session.query(Prediction).filter(Prediction.id == ev_rec.prediction_id).first()
                        if pred_entity:
                            pred_entity.status = ev_rec.status
                            pred_entity.actual_observed_result_json = ev_rec.actual_observed_result
                            pred_entity.evaluation_reason = ev_rec.reason
                            pred_entity.target_experiment_id = exp_entity.id
                        _queue_event(f"prediction.{ev_rec.status.lower()}", {"prediction_id": ev_rec.prediction_id, "status": ev_rec.status, "reason": ev_rec.reason})

                # Persist hypothesis revisions
                for revision in transition.hypothesis_revisions:
                    hyp_code = revision["hypothesis_code"]
                    hyp_entity = session.query(Hypothesis).filter(Hypothesis.id == f"{investigation_id}_{hyp_code}").first()
                    runtime_h = state_mgr.get_hypothesis(hyp_code)
                    if hyp_entity and runtime_h is not None:
                        hyp_entity.status = revision["new_status"]
                        hyp_entity.supporting_prediction_count = runtime_h.supporting_prediction_count
                        hyp_entity.refuted_prediction_count = runtime_h.refuted_prediction_count
                        hyp_entity.unresolved_prediction_count = runtime_h.unresolved_prediction_count
                    if revision.get("changed"):
                        _queue_event("hypothesis.revised", {"hypothesis_code": hyp_code, "new_status": revision["new_status"], "reason": revision["reason"]})

                # Persist Bayesian belief update
                bu = transition.belief_update or {}
                posteriors = bu.get("posteriors", [])
                bayes_factors = bu.get("bayes_factors", [])
                delta_entropy = bu.get("delta_entropy", 0.0)

                for idx, h in enumerate(current_hyps):
                    h_entity = session.query(Hypothesis).filter(Hypothesis.id == f"{investigation_id}_{h.hypothesis_code}").first()
                    if h_entity:
                        h_entity.posterior_probability = h.posterior_probability
                        h_entity.belief_state = h.belief_state

                if current_hyps and not transition.evidence_is_duplicate:
                    # Persist the Bayesian contribution against the HYPOTHESIS
                    # THAT THIS EXPERIMENT ACTUALLY TESTED.  The previous
                    # current_hyps[0] implementation made database provenance
                    # depend on hypothesis arrival order and could attach a
                    # belief update to an unrelated hypothesis.
                    target_idx = next(
                        (i for i, h in enumerate(current_hyps)
                         if h.hypothesis_code == selected_exp.target_hypothesis_code),
                        None,
                    )
                    if target_idx is not None:
                        priors = bu.get("priors", []) or []
                        target_prior = priors[target_idx] if target_idx < len(priors) else current_hyps[target_idx].prior_probability
                        target_bayes_factor = bayes_factors[target_idx] if target_idx < len(bayes_factors) else 1.0
                        target_posterior = posteriors[target_idx] if target_idx < len(posteriors) else current_hyps[target_idx].posterior_probability
                        b_update = BeliefUpdate(
                            id=gen_uuid(),
                            investigation_id=investigation_id,
                            hypothesis_id=f"{investigation_id}_{current_hyps[target_idx].hypothesis_code}",
                            evidence_id=ev_entity.id,
                            prior_probability=target_prior,
                            likelihood_p=None,
                            bayes_factor=target_bayes_factor,
                            posterior_probability=target_posterior,
                            entropy_delta=delta_entropy,
                            update_step_index=step_counter,
                        )
                        session.add(b_update)

                return {
                    "experiment": exp_code,
                    "metric_value": primary_val,
                    "verified": transition.verification_status == "VERIFIED",
                    "delta_entropy": delta_entropy,
                    "posteriors": posteriors,
                }

            try:
                step_result = self.checkpointer.execute_step_transactionally(
                    investigation_id=investigation_id,
                    execution_id=exec_id,
                    step_index=step_counter,
                    step_type="EXPERIMENT_EXECUTION",
                    step_code=exp_code,
                    step_fn=run_selected_experiment,
                )
            except QueryExecutionTimeoutError as timeout_exc:
                _queue_event("investigation.progress", {
                    "phase": "EXECUTION_TIMEOUT",
                    "experiment_code": exp_code,
                    "error": str(timeout_exc),
                    "elapsed_seconds": round(time.monotonic() - investigation_started_monotonic, 3),
                })
                _flush_pending_events()
                stopping_reason = "QUERY_EXECUTION_TIMEOUT"
                all_verifications_passed = False
                break
            except Exception as experiment_exc:
                _queue_event("investigation.progress", {
                    "phase": "EXPERIMENT_FAILED",
                    "experiment_code": exp_code,
                    "error": f"{type(experiment_exc).__name__}: {experiment_exc}",
                    "elapsed_seconds": round(time.monotonic() - investigation_started_monotonic, 3),
                })
                _flush_pending_events()
                stopping_reason = "EXPERIMENT_EXECUTION_FAILED"
                all_verifications_passed = False
                break
            _flush_pending_events()
            step_counter += 1

            # In-Flight Fail-Closed State Consistency Validation
            in_flight_report = state_mgr.validate_consistency()
            if not in_flight_report.is_consistent:
                errors = [e.message for e in in_flight_report.errors]
                self.checkpointer.record_event(
                    investigation_id=investigation_id,
                    execution_id=exec_id,
                    event_type="investigation.state_inconsistent",
                    payload={"error_count": len(errors), "errors": errors},
                )
                self.checkpointer.fail_execution(
                    investigation_id=investigation_id,
                    execution_id=exec_id,
                    error_message=f"In-flight canonical state consistency validation failed: {'; '.join(errors)}",
                    failure_class=FailureTaxonomy.ANALYTICAL_ERROR,
                )
                return False

            # Refresh runtime state from authoritative state manager
            current_hyps = _stable_hypothesis_order(state_mgr.get_active_hypotheses())
            current_posteriors = [h.posterior_probability for h in current_hyps]

            # Phase 12: Missingness & Selection-Bias Sensitivity (Sections 5-10, 22, 24).
            # Re-evaluated every iteration against the CURRENT leading hypothesis so a
            # late-emerging leader is checked too, not just whatever led at loop start.
            leading_h_loop = _leading_hypothesis(current_hyps)
            counter_h_loop_candidates = _stable_hypothesis_order([h for h in current_hyps if h.hypothesis_code != leading_h_loop.hypothesis_code])
            if canonical_target_col in primary_df.columns:
                leader_val = getattr(leading_h_loop, "target_value", None)
                counter_val = getattr(counter_h_loop_candidates[0], "target_value", None) if counter_h_loop_candidates else None
                if (
                    canonical_group_dimension_col
                    and leader_val and counter_val and leader_val != counter_val
                    and canonical_group_dimension_col in primary_df.columns
                    and leader_val in set(primary_df[canonical_group_dimension_col].astype(str))
                    and counter_val in set(primary_df[canonical_group_dimension_col].astype(str))
                ):
                    latest_missingness_result = MissingnessSensitivityEngine.analyze_group_ranking(
                        primary_df.assign(**{canonical_group_dimension_col: primary_df[canonical_group_dimension_col].astype(str)}),
                        semantic.metric_definition, canonical_target_col,
                        canonical_group_dimension_col, str(leader_val), str(counter_val),
                    )
                else:
                    latest_missingness_result = MissingnessSensitivityEngine.analyze(
                        primary_df, semantic.metric_definition, canonical_target_col,
                        group_col=canonical_group_dimension_col,
                    )
            # In-Loop Dynamic Stopping Evaluation
            current_probs = [h.posterior_probability for h in current_hyps]
            current_ent = compute_shannon_entropy(current_probs)
            entropy_trace.append(float(current_ent))

            # A competing hypothesis counts as evaluated only when the live
            # canonical state contains direct evidence linked to that specific
            # alternative. Merely existing as a counter hypothesis, being
            # considered by the adversarial attacker, or inheriting posterior
            # mass is not empirical evaluation. This mirrors DEFECT-007's
            # direct-test rule and prevents normal scientific stopping before
            # an alternative explanation has actually been tested.
            #
            # ALL counter hypotheses must carry direct evidence, not just any
            # one of them -- with evidence cross-linking now correctly scoped
            # to true binary mirrors only (see _is_binary_mirror_hypothesis),
            # a verified experiment against the leading hypothesis no longer
            # auto-credits structurally distinct alternatives (e.g. a
            # confound/exposure hypothesis with its own target dimension).
            # Requiring `any()` here would let the investigation stop the
            # moment its binary-mirror null was tested, silently skipping the
            # dedicated confound-check experiments other counter hypotheses
            # still need.
            counter_hypotheses_for_stop = [h for h in current_hyps if getattr(h, "is_counter_hypothesis", False)]
            if counter_hypotheses_for_stop:
                counter_hypothesis_evaluated = all(
                    bool(getattr(h, "supporting_evidence_ids", None) or getattr(h, "contradicting_evidence_ids", None))
                    # A counter hypothesis that no candidate experiment has
                    # ever targeted cannot be blocked on -- there is no
                    # dedicated test the investigation could have run
                    # against it, only cross-linked evidence from testing
                    # the leading hypothesis (see _is_binary_mirror_hypothesis
                    # for when that cross-link legitimately applies).
                    or h.hypothesis_code not in hypothesis_codes_ever_targeted
                    for h in counter_hypotheses_for_stop
                )
            else:
                counter_hypothesis_evaluated = False
            if not counter_hypothesis_evaluated and len(current_hyps) > 1 and not counter_hypotheses_for_stop:
                # Non-counter alternatives can also serve as the explicit
                # competing explanation when no dedicated null/counter node
                # exists. They must still have direct evidence.
                _non_leading = [h for h in current_hyps if h.hypothesis_code != _leading_hypothesis(current_hyps).hypothesis_code]
                counter_hypothesis_evaluated = bool(_non_leading) and all(
                    bool(getattr(h, "supporting_evidence_ids", None) or getattr(h, "contradicting_evidence_ids", None))
                    for h in _non_leading
                )

            # A right-censoring column present in the data is a structural
            # fact about how the churn outcome was measured, not merely one
            # more optional experiment -- a crude rate computed without ever
            # looking at how much of the "no churn" mass is actually
            # unresolved (censored) observation can be systematically
            # biased. When the semantic layer has identified a censoring
            # column, do not allow normal (non-safety-budget) stopping until
            # EXP-CHURN-CENSOR has actually run at least once, regardless of
            # how quickly the posterior otherwise concentrates.
            if getattr(semantic, "churn_censored_col", None):
                censoring_diagnostic_done = any(
                    "EXP-CHURN-CENSOR" in eid for eid in state_mgr.get_executed_experiments()
                )
                if not censoring_diagnostic_done:
                    counter_hypothesis_evaluated = False

            stop_decision = StoppingEngine.evaluate_stopping(
                current_entropy=current_ent,
                initial_entropy=initial_entropy,
                entropy_history=entropy_trace,
                iteration_count=len(state_mgr.get_executed_experiments()),
                max_iterations=max_safety_budget_experiments,
                current_posteriors=current_probs,
                unresolved_adversarial_issues=unresolved_adversarial_issues,
                counter_hypothesis_evaluated=counter_hypothesis_evaluated,
                missingness_classification=(latest_missingness_result.classification if latest_missingness_result else None),
            )
            authoritative_stop_decision = stop_decision
            if stop_decision.should_stop:
                stopping_reason = stop_decision.reason
                break

        # 5. Canonical-State Consistency Validation (Fail Closed)
        consistency_report = state_mgr.validate_consistency()
        if not consistency_report.is_consistent:
            errors = [e.message for e in consistency_report.errors]
            self.checkpointer.record_event(
                investigation_id=investigation_id,
                execution_id=exec_id,
                event_type="investigation.state_inconsistent",
                payload={"error_count": len(errors), "errors": errors},
            )
            self.checkpointer.fail_execution(
                investigation_id=investigation_id,
                execution_id=exec_id,
                error_message=f"Canonical state consistency validation failed: {'; '.join(errors)}",
                failure_class=FailureTaxonomy.ANALYTICAL_ERROR,
            )
            return False

        # 6. Claim-specific adversarial challenge. A global attack is required
        # only where it can materially change the requested claim; direct
        # descriptive/predictive questions use their own validation methods.
        if _runtime_budget_exceeded("before_adversarial_review"):
            stopping_reason = "RUNTIME_BUDGET_EXCEEDED"
        if cancellation_check and cancellation_check():
            return False

        leading_hyp = _leading_hypothesis(current_hyps)
        counter_hyps = [h for h in current_hyps if h.is_counter_hypothesis]
        counter_hyp = _stable_hypothesis_order(counter_hyps)[0] if counter_hyps else _stable_hypothesis_order([h for h in current_hyps if h.hypothesis_code != leading_hyp.hypothesis_code])[0]
        adversarial_required = analysis_plan.task in {"CAUSAL", "DIAGNOSTIC", "COMPARISON", "GENERAL_EXPLORATION", "PRESCRIPTIVE"}

        if adversarial_required:
            attack_res = AdversarialAttacker.execute_adversarial_attack(
                df=primary_df, leading_hypothesis=leading_hyp, counter_hypothesis=counter_hyp,
                semantic=semantic, execution_provider=self.execution_provider,
            )

            def step_adversarial(session: Session):
                if attack_res.is_falsified:
                    leading_hyp.belief_state = "falsified"
                    # Principled Bayesian update on adversarial falsification:
                    # Treat falsification as an adversarial likelihood weight (BF < 1).
                    # A decisive Simpson's reversal or counter-evidence provides strong
                    # evidence against the leading hypothesis.
                    # BeliefEngine.compute_bayesian_posteriors ensures all hypothesis
                    # probabilities are properly updated and normalized (sum = 1.0).
                    priors = [float(h.posterior_probability) for h in current_hyps]
                    adv_weight = 0.05 if attack_res.simpsons_paradox_detected else 0.15
                    likelihoods = []
                    for h in current_hyps:
                        if h.hypothesis_code == leading_hyp.hypothesis_code:
                            likelihoods.append(adv_weight)
                        elif getattr(h, "is_counter_hypothesis", False) or h.hypothesis_code == getattr(counter_hyp, "hypothesis_code", None):
                            likelihoods.append(1.0 / adv_weight if adv_weight > 0 else 1.0)
                        else:
                            likelihoods.append(1.0)
                    try:
                        new_posteriors, _ = BeliefEngine.compute_bayesian_posteriors(priors, likelihoods)
                        for idx, h in enumerate(current_hyps):
                            h.posterior_probability = float(new_posteriors[idx])
                    except Exception:
                        leading_hyp.posterior_probability = max(0.05, leading_hyp.posterior_probability * 0.10)
                    # Persist the updated belief states and normalized posteriors
                    for h in current_hyps:
                        _h_row = session.query(Hypothesis).filter(
                            Hypothesis.id == f"{investigation_id}_{h.hypothesis_code}"
                        ).first()
                        if _h_row is not None:
                            if h.hypothesis_code == leading_hyp.hypothesis_code:
                                _h_row.belief_state = "REFUTED"
                            _h_row.posterior_probability = h.posterior_probability
                return {
                    "attack_status": attack_res.attack_status, "is_falsified": attack_res.is_falsified,
                    "epistemic_impact": attack_res.epistemic_impact,
                    "simpsons_paradox": attack_res.simpsons_paradox_detected,
                    "outlier_sensitive": attack_res.outlier_sensitivity_high,
                    "simpsons_reversal": (attack_res.details or {}).get("simpsons_reversal") if attack_res.is_falsified else None,
                }

            self.checkpointer.execute_step_transactionally(
                investigation_id=investigation_id, execution_id=exec_id, step_index=step_counter,
                step_type="ADVERSARIAL_CHALLENGE", step_code="ADV-ATTACK", step_fn=step_adversarial,
            )
            step_counter += 1
            # Session 8: the attack outcome used to live only in memory (the step
            # record stores just a hash), so the analyst/verifier could not see
            # what the system tried to break. /investigations/{id} already reads
            # this event prefix; nothing ever wrote it.
            self.checkpointer.record_event(
                investigation_id=investigation_id, execution_id=exec_id,
                event_type="investigation.adversarial_challenge.completed",
                payload={
                    "leading_hypothesis": getattr(attack_res, "leading_hypothesis_code", None),
                    "counter_hypothesis": getattr(attack_res, "counter_hypothesis_code", None),
                    "attack_mechanism": getattr(attack_res, "attack_mechanism", None),
                    "discriminating_test_sql": getattr(attack_res, "discriminating_test_sql", None),
                    "attack_status": attack_res.attack_status,
                    "is_falsified": bool(attack_res.is_falsified),
                    "epistemic_impact": attack_res.epistemic_impact,
                    "simpsons_paradox_detected": bool(getattr(attack_res, "simpsons_paradox_detected", False)),
                    "outlier_sensitivity_high": bool(getattr(attack_res, "outlier_sensitivity_high", False)),
                    "details": dict(getattr(attack_res, "details", None) or {}),
                },
            )
        else:
            from types import SimpleNamespace
            attack_res = SimpleNamespace(
                attack_status="NOT_APPLICABLE", is_falsified=False, epistemic_impact=0.0,
                simpsons_paradox_detected=False, outlier_sensitivity_high=False, details={},
            )
            self.checkpointer.record_event(
                investigation_id=investigation_id, execution_id=exec_id,
                event_type="investigation.adversarial_challenge.skipped",
                payload={"task": analysis_plan.task, "reason": "Not required by the universal analysis evidence contract."},
            )

        # DEFECT-021: a genuine adversarial refutation (currently a gated
        # Simpson's reversal) is carried to the verdict and to the analyst-facing
        # answer so it is reported as a finding, not lost inside "inconclusive".
        adversarial_refutation = (
            dict((getattr(attack_res, "details", None) or {}).get("simpsons_reversal") or {}) or None
            if getattr(attack_res, "is_falsified", False) else None
        )

        # 7. Multiverse specification-curve analysis. It is a sensitivity
        # analysis, not a mandatory step for every question class.
        multiverse_required = analysis_plan.task in {"CAUSAL", "DIAGNOSTIC", "COMPARISON", "GENERAL_EXPLORATION", "PRESCRIPTIVE"}
        if cancellation_check and cancellation_check():
            return False

        if multiverse_required and canonical_group_dimension_col and canonical_group_dimension_col in primary_df.columns:
            multiverse_report = MultiverseEngine.evaluate_specification_curve(
                df=primary_df,
                dimension_col=canonical_group_dimension_col,
                target_metric_col=canonical_target_col,
            )
            # Session 8: expose the specification curve to the analyst/verifier
            # (/investigations/{id} reads the `calculation.multiverse` prefix;
            # previously nothing wrote it, so `multiverse` was always null).
            self.checkpointer.record_event(
                investigation_id=investigation_id, execution_id=exec_id,
                event_type="calculation.multiverse",
                payload={
                    "applicable": bool(multiverse_report.applicable),
                    "dimension": canonical_group_dimension_col,
                    "metric": canonical_target_col,
                    "total_specifications": multiverse_report.total_specifications,
                    "concordant_specifications": multiverse_report.concordant_specifications,
                    "robustness_pct": multiverse_report.robustness_pct,
                    "epistemic_summary": multiverse_report.epistemic_summary,
                    "specification_curve": list(multiverse_report.specification_curve or []),
                },
            )
        else:
            from packages.analytics_core.src.intelligence.multiverse_engine import MultiverseRobustnessReport
            multiverse_report = MultiverseRobustnessReport(
                total_specifications=0, concordant_specifications=0, robustness_pct=None,
                specification_curve=[],
                epistemic_summary=(
                    "Multiverse specification curve was not run because no unambiguous "
                    "grouping dimension was resolved for this claim."
                ),
                applicable=False,
            )
            if multiverse_required:
                skip_reason = "No unambiguous grouping dimension resolved."
            else:
                skip_reason = "Not required by the universal analysis evidence contract."
                multiverse_report = MultiverseRobustnessReport(
                    total_specifications=0, concordant_specifications=0, robustness_pct=None,
                    specification_curve=[],
                    epistemic_summary=f"Multiverse specification curve not applicable to task {analysis_plan.task}.",
                    applicable=False,
                )
            # Exactly one skip event, carrying the true reason.
            self.checkpointer.record_event(
                investigation_id=investigation_id, execution_id=exec_id,
                event_type="investigation.multiverse.skipped",
                payload={"task": analysis_plan.task, "reason": skip_reason},
            )

        # 8. Epistemic Calibration & Final Verdict Formulation
        final_probs = [h.posterior_probability for h in current_hyps]
        final_entropy = compute_shannon_entropy(final_probs)
        top_prob = max(final_probs)

        variance_explained_pct = None
        if canonical_group_dimension_col:
            variance_explained_pct = BeliefEngine.compute_variance_explained(
                primary_df, canonical_group_dimension_col, canonical_target_col
            )

        # DEFECT-007: positive-claim verification is scoped to the LEADING
        # hypothesis's own evidence lineage, not to the global experiment run.
        #
        # `all_verifications_passed` is deliberately retained for diagnostics
        # and epistemic calibration, because an investigation may contain an
        # unrelated failed/blocked auxiliary experiment. It must NOT erase a
        # directly targeted, independently VERIFIED result that supports the
        # proposition we are actually about to claim.
        #
        # A hypothesis can carry evidence IDs from both supporting and
        # contradicting classifications. Only evidence with an actual
        # VERIFIED proof is admissible here. This prevents a failed evidence
        # node from being treated as verified merely because it was linked to
        # the leading hypothesis.
        leading_evidence_ids = set(
            getattr(leading_hyp, "supporting_evidence_ids", None) or []
        ) | set(
            getattr(leading_hyp, "contradicting_evidence_ids", None) or []
        )
        leading_verified_evidence = False
        if leading_evidence_ids:
            with self.session_factory() as _verification_session:
                leading_verified_evidence = (
                    _verification_session.query(EvidenceVerification.id)
                    .filter(
                        EvidenceVerification.evidence_id.in_(leading_evidence_ids),
                        EvidenceVerification.status == VerificationStatus.VERIFIED,
                    )
                    .first()
                    is not None
                )
        has_verified_evidence = leading_verified_evidence

        # The assumption ledger was built before any experiment ran, so
        # INDEPENDENT_VERIFICATION (like FORECAST_GENERALIZATION above) was
        # necessarily created as unvalidated -- there was no evidence yet to
        # check it against. `has_verified_evidence` is now that evidence:
        # it means an independently re-executed proof agreed with the
        # leading hypothesis's result within its declared tolerance, which
        # is exactly what INDEPENDENT_VERIFICATION's statement requires.
        # Patch the ledger (and the derived quality card) to reflect this
        # before it is re-persisted below, instead of leaving the
        # human-facing ledger permanently claiming "unvalidated" while
        # downstream consumers each grow their own bespoke bypass for it.
        _ledger_updates: dict = {}
        if has_verified_evidence:
            _ledger_updates["INDEPENDENT_VERIFICATION"] = (
                True,
                "An independently re-executed verification proof agreed with the leading "
                "hypothesis's result within its declared tolerance.",
            )
        if forecast_generalization_validated:
            _ledger_updates["FORECAST_GENERALIZATION"] = (
                True,
                "A rolling-origin backtest showed the fitted trend beating a naive baseline "
                "out-of-sample.",
            )
        if _ledger_updates:
            assumption_items = AssumptionLedgerEngine.patch_validation(assumption_items, _ledger_updates)
            quality_card = AssumptionLedgerEngine.score(assumption_items)
            self.checkpointer.record_event(
                investigation_id=investigation_id,
                execution_id=exec_id,
                event_type="investigation.assumption_ledger.patched",
                payload={
                    "quality_card": quality_card.to_dict(),
                    "assumptions": [item.to_dict() for item in assumption_items],
                    "reason": "post_execution_evidence",
                },
            )

            def _repersist_assumption_ledger(session: Session):
                from apps.api.src.models.entities import Assumption
                for item in assumption_items:
                    if item.code not in _ledger_updates:
                        continue
                    row = (
                        session.query(Assumption)
                        .filter(
                            Assumption.investigation_id == investigation_id,
                            Assumption.statement.like(f"[{item.code}]%"),
                        )
                        .first()
                    )
                    if row is not None:
                        row.is_validated = item.is_validated

            self.checkpointer.execute_step_transactionally(
                investigation_id=investigation_id,
                execution_id=exec_id,
                step_index=2,
                step_type="ASSUMPTION_LEDGER_PATCH",
                step_code="ASSUMPTION-LEDGER-PATCH",
                step_fn=_repersist_assumption_ledger,
            )

        # Phase 12: final missingness sensitivity check against the terminal
        # leading hypothesis (re-run once more in case it changed on the last
        # iteration after the in-loop check above).
        counter_hyp_final_candidates = _stable_hypothesis_order([h for h in current_hyps if h.hypothesis_code != leading_hyp.hypothesis_code])
        if canonical_target_col in primary_df.columns:
            leader_val_final = getattr(leading_hyp, "target_value", None)
            counter_val_final = getattr(counter_hyp_final_candidates[0], "target_value", None) if counter_hyp_final_candidates else None
            if (
                canonical_group_dimension_col and leader_val_final and counter_val_final and leader_val_final != counter_val_final
                and canonical_group_dimension_col in primary_df.columns
                and str(leader_val_final) in set(primary_df[canonical_group_dimension_col].astype(str))
                and str(counter_val_final) in set(primary_df[canonical_group_dimension_col].astype(str))
            ):
                latest_missingness_result = MissingnessSensitivityEngine.analyze_group_ranking(
                    primary_df.assign(**{canonical_group_dimension_col: primary_df[canonical_group_dimension_col].astype(str)}),
                    semantic.metric_definition, canonical_target_col,
                    canonical_group_dimension_col, str(leader_val_final), str(counter_val_final),
                )
            else:
                latest_missingness_result = MissingnessSensitivityEngine.analyze(
                    primary_df, semantic.metric_definition, canonical_target_col,
                    group_col=canonical_group_dimension_col,
                )

        # DEFECT-007: a hypothesis is only "directly tested" if a verified
        # experiment actually targeted it (recorded in its own
        # supporting_evidence_ids or contradicting_evidence_ids -- either
        # counts, since a contradicting result is still direct evidence
        # about that specific hypothesis, unlike untouched posterior share
        # inherited purely from Bayesian normalization).
        leading_directly_tested = bool(
            getattr(leading_hyp, "supporting_evidence_ids", None)
            or getattr(leading_hyp, "contradicting_evidence_ids", None)
        )

        # DEFECT-015: extract churn identifiability parameters for verdict
        # formulation. IntentEngine's explicit CHURN classification is the
        # authoritative lexical signal here. Method selection may compile a
        # churn question into PREDICTION/ASSOCIATION for a particular execution
        # path, but that must not bypass the churn-specific identifiability
        # safeguards or the tenure/exposure confounding analysis.
        is_churn_q = (
            str(getattr(intent, "intent_type", "") or "").upper() == "CHURN"
            or method_decision.problem_class == ProblemClass.SURVIVAL_CHURN
        )
        candidate_groups = [c for c in primary_df.select_dtypes(include=['object', 'category']).columns if not any(id_k in c.lower() for id_k in ["_id", "id", "key", "pk", "fk", "uuid", "code", "zip"])] if primary_df is not None else []
        c_grp = canonical_group_dimension_col or (candidate_groups[0] if len(candidate_groups) == 1 else None)
        is_churn_unidentifiable = is_churn_q and (
            not getattr(semantic, "churn_outcome_available", True)
            or getattr(semantic, "churn_event_col", None) is None
            or (c_grp is None and len(candidate_groups) > 1)
        )
        churn_verdict_val = None
        if is_churn_q and not is_churn_unidentifiable and primary_df is not None:
            try:
                if c_grp and c_grp in primary_df.columns:
                    df_churn = primary_df.copy()
                    if "churn_event" not in df_churn.columns and getattr(semantic, "churn_event_col", None) in df_churn.columns:
                        df_churn["churn_event"] = df_churn[semantic.churn_event_col]
                    c_res = analyze_churn_identifiability(
                        df_churn,
                        group_col=c_grp,
                        known_confounders=getattr(semantic, "churn_confounder_cols", []),
                        exposure_col=getattr(semantic, "churn_exposure_col", None),
                        censored_col=getattr(semantic, "churn_censored_col", None),
                    )
                    churn_verdict_val = c_res.verdict
            except Exception:
                pass

        loaded_calibration_profile = load_calibration_profile(os.getenv("AAOS_CALIBRATION_PROFILE"))
        calibration_profile = None
        calibrated_probability = None
        calibration_status = "UNAVAILABLE"
        if loaded_calibration_profile is not None:
            expected_model = os.getenv("AAOS_CALIBRATION_MODEL", "AAOS_BELIEF_V1")
            expected_scope = os.getenv("AAOS_CALIBRATION_SCOPE")
            # Runtime population scope is an authority-bearing fact from the
            # active contract path: calibration applies to the primary dataset
            # population used by this investigation. A profile for another
            # population must never silently calibrate the current posterior.
            runtime_population_scope = "primary_dataset"
            if not loaded_calibration_profile.is_usable:
                calibration_status = "PROFILE_PROVENANCE_INVALID"
            elif expected_scope is None:
                calibration_status = "PROFILE_SCOPE_UNSPECIFIED"
            elif loaded_calibration_profile.model_id != expected_model:
                calibration_status = "PROFILE_MODEL_MISMATCH"
            elif loaded_calibration_profile.scope != expected_scope:
                calibration_status = "PROFILE_SCOPE_MISMATCH"
            elif loaded_calibration_profile.population_scope != runtime_population_scope:
                calibration_status = "PROFILE_POPULATION_MISMATCH"
            else:
                calibration_profile = loaded_calibration_profile
                calibrated_probability = calibration_profile.calibrate(top_prob)
                calibration_status = "CALIBRATED"

        # INDEPENDENT_VERIFICATION and FORECAST_GENERALIZATION no longer need
        # special-casing here: they were patched above (in place, on
        # assumption_items) with their real post-execution validation state,
        # so a plain is_validated check now already reflects has_verified_evidence
        # / forecast_generalization_validated.
        material_unvalidated_high_risk = sum(
            1 for item in assumption_items
            if getattr(item, "sensitivity_risk", "").lower() == "high"
            and not bool(getattr(item, "is_validated", False))
        )
        # Final downstream data-quality decision: contextual findings must now
        # change what the analyst is allowed to claim, not merely what it reports.
        # Existing missingness sensitivity and analyst robustness outputs are used
        # to clear the corresponding review when they actually resolve it.
        analyst_result = None
        _analyst_robust = None
        if analyst_result is not None:
            _analyst_robust = getattr(analyst_result, "numbers", {}).get("robust")
            if _analyst_robust is not True:
                _analyst_robust = None
        final_quality_decision = assess_downstream_impact(
            quality_assessment,
            analysis_plan.task,
            missingness_classification=(latest_missingness_result.classification if latest_missingness_result else None),
            outlier_robust=_analyst_robust if _analyst_robust is not None else (False if getattr(attack_res, "outlier_sensitivity_high", False) else None),
            selection_bias_resolved=False,
            temporal_issues_resolved=not bool(getattr(quality_assessment, "temporal_issues", []) or []),
            include_unresolved_without_runtime_evidence=False,
        )
        self.checkpointer.record_event(
            investigation_id=investigation_id,
            execution_id=exec_id,
            event_type="investigation.data_quality.downstream_decision",
            payload=final_quality_decision.to_dict(),
        )
        # NOTE: the quality-qualification downgrade that used to live here was
        # moved below (see the block immediately before the provenance manifest
        # is computed). `verdict_eval` is not assigned until VerdictEngine.
        # evaluate_verdict() runs further down, and `direct_ans`/`main_find` are
        # not assigned until ProvenanceEngine.formulate_direct_answer() runs
        # after that -- referencing any of them here raised UnboundLocalError
        # any time final_quality_decision.claim_qualification_required was True
        # (reproduced with a dataset containing a real partition dimension and
        # a genuine data-quality flag; the bug was silent on datasets where the
        # flag never fired, e.g. a single-series dataset with no partition
        # column at all, which is why it went unnoticed).

        # Release Claim Gate: deterministic three-state decision. The legacy
        # boolean gate below remains intact for verdict compatibility; this
        # structured gate is the public epistemic outcome and is persisted.
        _claim_raw = str(getattr(analysis_plan, "claim_type", "OBSERVATION")).upper()
        _claim_map = {
            "OBSERVATION": GateClaimType.OBSERVATION,
            "ASSOCIATION": GateClaimType.ASSOCIATION,
            "PREDICTION": GateClaimType.PREDICTION,
            "CAUSAL": GateClaimType.CAUSAL,
            "CAUSAL_INFERENCE": GateClaimType.CAUSAL,
            "RECOMMENDATION": GateClaimType.RECOMMENDATION,
            "SIMULATION": GateClaimType.RECOMMENDATION,
        }
        _requested_claim = _claim_map.get(_claim_raw, GateClaimType.OBSERVATION)
        _prediction_text = " ".join(
            str(getattr(e, "test_code", "")) + " " + str(getattr(e, "rationale", "")) + " " + str(getattr(e, "selection_rationale", ""))
            for e in state_mgr.get_executed_experiments()
        ).lower()
        _prediction_oos = _requested_claim != GateClaimType.PREDICTION or any(
            t in _prediction_text for t in ("out-of-sample", "out_of_sample", "holdout", "rolling-origin", "backtest")
        )
        if _requested_claim == GateClaimType.CAUSAL:
            if causal_gate_res.is_identifiable and causal_effect_estimated and has_verified_evidence:
                _gate_evidence_level = GateEvidenceLevel.CAUSAL_ESTIMATION
            elif causal_gate_res.is_identifiable:
                _gate_evidence_level = GateEvidenceLevel.CAUSAL_IDENTIFICATION
            else:
                _gate_evidence_level = GateEvidenceLevel.ASSOCIATION if has_verified_evidence else GateEvidenceLevel.VALIDATED_COMPUTATION
        elif _requested_claim == GateClaimType.PREDICTION:
            _gate_evidence_level = GateEvidenceLevel.PREDICTION if has_verified_evidence and _prediction_oos else GateEvidenceLevel.VALIDATED_COMPUTATION
        elif _requested_claim == GateClaimType.ASSOCIATION:
            _gate_evidence_level = GateEvidenceLevel.ASSOCIATION if has_verified_evidence else GateEvidenceLevel.VALIDATED_COMPUTATION
        elif _requested_claim == GateClaimType.RECOMMENDATION:
            _gate_evidence_level = GateEvidenceLevel.ASSOCIATION if has_verified_evidence else GateEvidenceLevel.VALIDATED_COMPUTATION
        else:
            _gate_evidence_level = GateEvidenceLevel.VALIDATED_COMPUTATION if has_verified_evidence else GateEvidenceLevel.RAW_OBSERVATION
        _design_status = (
            GateDesignStatus.KNOWN
            if getattr(analysis_plan, "decision_status", "") == "RESOLVED" and not list(getattr(analysis_plan, "unresolved_questions", []) or [])
            else GateDesignStatus.ASSUMED
            if analysis_plan is not None
            else GateDesignStatus.UNKNOWN
        )
        _contract_assumptions = list(getattr(analysis_plan, "limitations", []) or [])
        _causal_strategy = getattr(causal_gate_res.status, "value", str(causal_gate_res.status)) if causal_gate_res.is_identifiable else None
        _structured_gate = evaluate_gate(
            claim_type=_requested_claim,
            evidence_level=_gate_evidence_level,
            design_status=_design_status,
            assumptions=_contract_assumptions,
            identification_strategy=_causal_strategy,
            computed_evidence={
                "verified_evidence": bool(has_verified_evidence),
                "leading_hypothesis_directly_tested": bool(leading_directly_tested),
                "causal_identifiable": bool(causal_gate_res.is_identifiable),
                "causal_effect_estimated": bool(causal_effect_estimated),
                "data_quality_downstream_decision": final_quality_decision.to_dict(),
            },
            assumption_risk=bool(material_unvalidated_high_risk or final_quality_decision.claim_qualification_required),
            prediction_out_of_sample=_prediction_oos,
            selection_bias=bool(getattr(quality_assessment, "selection_bias_indicators", []) or []),
            verification_passed=bool(has_verified_evidence),
        )
        _gate_id = f"RG-{gen_uuid()[:12]}"
        self.checkpointer.record_event(
            investigation_id=investigation_id,
            execution_id=exec_id,
            event_type="investigation.claim_gate.evaluated",
            payload={"decision_id": _gate_id, **_structured_gate.to_dict()},
        )
        with self.session_factory() as _gate_session:
            _gate_session.add(ClaimGateDecision(
                id=_gate_id,
                investigation_id=investigation_id,
                question_id=investigation_id,
                requested_claim=_structured_gate.requested_claim.name,
                requested_level=int(_structured_gate.requested_level or _structured_gate.requested_claim.value),
                evidence_level=int(_structured_gate.evidence_level.value),
                max_supported_level=int(_structured_gate.max_supported_level or _structured_gate.evidence_level.value),
                outcome=_structured_gate.outcome,
                design_status=_structured_gate.design_status.value,
                identification_strategy=_structured_gate.identification_strategy,
                assumptions_json=_structured_gate.assumptions,
                blocking_conditions_json=_structured_gate.blocking_conditions,
                missing_evidence_json=_structured_gate.missing_evidence,
                recovery_actions_json=_structured_gate.recovery_actions,
                allowed_claim=_structured_gate.allowed_claim,
                blocked_claim=_structured_gate.blocked_claim,
                reason=_structured_gate.reason,
                computed_evidence_json=_structured_gate.computed_evidence,
            ))
            _gate_session.commit()

        claim_admission = admit_positive_claim(
            verdict_type="DIAGNOSED" if top_prob >= 0.70 else "INCONCLUSIVE",
            directly_tested=leading_directly_tested,
            verified_evidence=has_verified_evidence,
            missingness_classification=(latest_missingness_result.classification if latest_missingness_result else None),
            selection_bias_indicators=list(getattr(quality_assessment, "selection_bias_indicators", []) or []),
            causal_identifiable=causal_gate_res.is_identifiable,
            requested_causal=(analysis_plan.task == "CAUSAL" or method_decision.causal_intent is not None),
            causal_effect_estimated=causal_effect_estimated,
            method_claim_ceiling=method_decision.max_verdict_tier,
            required_assumptions_satisfied=(not bool(getattr(method_decision, "blocked_reasons", [])) and material_unvalidated_high_risk == 0),
            unvalidated_high_risk_assumptions=material_unvalidated_high_risk,
            probability_calibrated=calibrated_probability is not None,
            full_scope=all(bool(item.get("full_scope", True)) for item in experiment_scope_records) if experiment_scope_records else True,
            sampling_policy=("NONE" if all(str(item.get("sampling_policy", "NONE")).upper() == "NONE" for item in experiment_scope_records) else "MIXED"),
            sampling_reason=next((str(item.get("sampling_reason")) for item in experiment_scope_records if item.get("sampling_reason")), None),
        )

        # The structured three-state gate is authoritative for positive claims.
        # The legacy boolean remains a compatibility diagnostic, but REFUSE and
        # QUALIFIED_ANSWER may never be upgraded into a decisive positive
        # verdict by the legacy path.
        _structured_gate_allows_decisive = _structured_gate.outcome == "ANSWER"
        if not _structured_gate_allows_decisive:
            claim_admission = ClaimAdmission(
                allowed=False,
                ceiling=claim_admission.ceiling,
                reasons=tuple(dict.fromkeys((
                    *claim_admission.reasons,
                    f"structured_claim_gate_{_structured_gate.outcome.lower()}",
                ))),
            )

        # Check empirical refutation of counter-hypotheses
        counter_hyps = [h for h in current_hyps if getattr(h, "is_counter_hypothesis", False)]
        if not counter_hyps and len(current_hyps) > 1:
            counter_hyps = [h for h in current_hyps if h.hypothesis_code != leading_hyp.hypothesis_code]

        counter_empirically_refuted = False
        if counter_hyps:
            for ch in counter_hyps:
                ch_contradicting = getattr(ch, "contradicting_evidence_ids", None) or []
                if ch_contradicting and has_verified_evidence:
                    counter_empirically_refuted = True
                    break
                if str(getattr(ch, "belief_state", "")).lower() in ("falsified", "refuted"):
                    counter_empirically_refuted = True
                    break
                if int(getattr(ch, "refuted_prediction_count", 0) or 0) > 0:
                    counter_empirically_refuted = True
                    break

        verdict_eval = VerdictEngine.evaluate_verdict(
            leading_hypothesis_code=leading_hyp.hypothesis_code,
            leading_hypothesis_posterior=top_prob,
            all_verifications_passed=has_verified_evidence,
            adversarial_attack_survived=attack_res.attack_status in ["SURVIVED", "INCONCLUSIVE", "ROBUST"],
            variance_explained_pct=variance_explained_pct,
            causal_identifiable=causal_gate_res.is_identifiable,
            missingness_classification=(latest_missingness_result.classification if latest_missingness_result else None),
            missingness_rationale=(latest_missingness_result.rationale if latest_missingness_result else ""),
            leading_hypothesis_directly_tested=leading_directly_tested,
            leading_hypothesis_is_counter=bool(getattr(leading_hyp, "is_counter_hypothesis", False)),
            churn_verdict=churn_verdict_val,
            is_churn_unidentifiable=is_churn_unidentifiable,
            max_verdict_tier=method_decision.max_verdict_tier,
            method_selection_note=method_decision.rationale,
            confidence_semantics=("EMPIRICALLY_CALIBRATED" if calibrated_probability is not None else "MODEL_BASED_BELIEF"),
            calibrated_probability=calibrated_probability,
            calibration_status=calibration_status,
            authoritative_stopping_met=(authoritative_stop_decision.should_stop if authoritative_stop_decision is not None else False),
            authoritative_stopping_reason=(authoritative_stop_decision.reason if authoritative_stop_decision is not None else stopping_reason),
            unvalidated_high_risk_assumptions=material_unvalidated_high_risk,
            adversarial_refutation=adversarial_refutation,
            counter_hypothesis_empirically_refuted=counter_empirically_refuted,
        )
        if verdict_eval.verdict_type in {"DIAGNOSED", "STATISTICALLY_SIGNIFICANT", "OBSERVED"} and not claim_admission.allowed:
            original = verdict_eval.verdict_type
            verdict_eval.verdict_type = "INCONCLUSIVE"
            verdict_eval.confidence_score = 0.0
            verdict_eval.direct_answer = (
                f"Investigation inconclusive: the strongest observed result reached the analytical threshold, "
                f"but the universal claim gate withheld a positive verdict because: {', '.join(claim_admission.reasons)}."
            )
            verdict_eval.justification = f"Positive-claim gate blocked {original}: {', '.join(claim_admission.reasons)}."
            verdict_eval.main_finding = f"Positive claim withheld by universal scientific admissibility gate. Ceiling={claim_admission.ceiling}."

        self.checkpointer.record_event(
            investigation_id=investigation_id,
            execution_id=exec_id,
            event_type="calculation.verdict_determination",
            payload={
                "trace_id": f"TRACE-{investigation_id}:VERDICT",
                "formula": "Verdict = deterministic decision gate over posterior evidence, independent verification, adversarial survival, variance explained, causal identifiability, missingness, and method-selection ceiling.",
                "inputs": {
                    "leading_hypothesis": leading_hyp.hypothesis_code,
                    "posterior": top_prob,
                    "selection_bias_indicators": list(getattr(quality_assessment, "selection_bias_indicators", []) or []),
                    "leading_hypothesis_verified_evidence": has_verified_evidence,
                    "adversarial_survived": attack_res.attack_status in ["SURVIVED", "INCONCLUSIVE", "ROBUST"],
                    "variance_explained_pct": variance_explained_pct,
                    "causal_identifiable": causal_gate_res.is_identifiable,
                    "causal_effect_estimated": causal_effect_estimated,
                    "missingness_classification": latest_missingness_result.classification if latest_missingness_result else None,
                    "max_verdict_tier": method_decision.max_verdict_tier,
                    "claim_gate_allowed": claim_admission.allowed,
                    "claim_gate_reasons": list(claim_admission.reasons),
                    "claim_gate_ceiling": claim_admission.ceiling,
                    "experiment_scope_records": list(experiment_scope_records),
                },
                "output": {
                    "verdict_type": verdict_eval.verdict_type,
                    "justification": verdict_eval.justification,
                },
            },
        )

        # Phase 12 (Section 20): persist the complete, structured sensitivity
        # calculation as an immutable provenance event -- distinguishing
        # OBSERVED fact from ASSUMPTION from DERIVED sensitivity result -- so
        # the result can be reproduced by another process from this record
        # alone, independent of narrative text in the verdict.
        if latest_missingness_result is not None:
            self.checkpointer.record_event(
                investigation_id=investigation_id,
                execution_id=exec_id,
                event_type="investigation.missingness_sensitivity",
                payload=latest_missingness_result.to_provenance_dict(),
            )

        # 8 Disentangled Epistemic Vectors
        epistemic_vector = EpistemicCalibrationEngine.calibrate_epistemic_state(
            posterior_probability=top_prob,
            evidence_count=len(state_mgr.get_executed_experiments()),
            all_verifications_passed=has_verified_evidence,
            multiverse_robustness=(multiverse_report.robustness_score if getattr(multiverse_report, "applicable", True) else None),
            adversarial_survived=attack_res.attack_status in ["SURVIVED", "INCONCLUSIVE", "ROBUST"],
            sample_size=len(primary_df),
            causal_identifiable=causal_gate_res.is_identifiable,
            calibration_profile=calibration_profile,
            runtime_model_id=os.getenv("AAOS_CALIBRATION_MODEL", "AAOS_BELIEF_V1"),
            runtime_scope=os.getenv("AAOS_CALIBRATION_SCOPE"),
            runtime_population_scope="primary_dataset",
        )

        self.checkpointer.record_event(
            investigation_id=investigation_id,
            execution_id=exec_id,
            event_type="calculation.epistemic_calibration",
            payload={
                "trace_id": f"TRACE-{investigation_id}:EPISTEMIC",
                "formula": "Epistemic state is calibrated deterministically from posterior probability, verified evidence, robustness, adversarial survival, sample size, data quality, and causal status.",
                "inputs": {
                    "posterior_probability": top_prob,
                    "evidence_count": len(state_mgr.get_executed_experiments()),
                    "leading_hypothesis_verified_evidence": has_verified_evidence,
                    "multiverse_robustness": multiverse_report.robustness_score,
                    "sample_size": len(primary_df),
                    "causal_identifiable": causal_gate_res.is_identifiable,
                    "calibration_status": calibration_status,
                },
                "output": epistemic_vector.to_dict() if hasattr(epistemic_vector, "to_dict") else getattr(epistemic_vector, "__dict__", {}),
            },
        )

        experiments_summary = [
            {
                "code": getattr(obs, "experiment_code", ""),
                "hypothesis_code": getattr(obs, "hypothesis_code", ""),
                "structured_result": getattr(obs, "structured_result", {}),
            }
            for obs in state_mgr.state.raw_observations
        ]
        executed_codes = state_mgr.get_executed_fingerprints() or [getattr(obs, "experiment_code", "") for obs in state_mgr.state.raw_observations]

        direct_ans, main_find = ProvenanceEngine.formulate_direct_answer(
            intent=intent,
            semantic=semantic,
            leading_hypothesis_code=leading_hyp.hypothesis_code,
            leading_claim=leading_hyp.claim,
            posterior=top_prob,
            verdict_type=verdict_eval.verdict_type,
            experiments_summary=experiments_summary,
            multiverse_report=multiverse_report,
            adversarial_refutation=adversarial_refutation,
        )
        # Session 10: the numbers-first answer a human analyst would give.  The scientific
        # loop above decides how much evidence there is; it does not by itself report the
        # quantity that was asked for (group means and their difference with an interval,
        # the ranking, the correlation, the trend, where a period-over-period change came
        # from).  Computed deterministically from the resolved contract columns; never
        # raises and never changes a verdict except in the two guarded cases below.
        analyst_result = None
        try:
            _agg_default = None
            if semantic.metric_definition is not None:
                _agg_default = getattr(getattr(semantic.metric_definition, "aggregation_type", None), "value", None)
            _analyst_predictors = list(getattr(method_decision.estimand, "predictor_columns", None) or []) or list(
                getattr(analysis_plan.semantics, "explanatory_columns", None) or []
            )
            analyst_df = primary_df
            if getattr(semantic, "relational_access", None) is not None:
                try:
                    ra = semantic.relational_access
                    join_sql = f"SELECT {ra.base_table}.*, {ra.joined_table}.{ra.dim_column} AS {ra.dim_column}"
                    for flt in getattr(ra, "filters", []) or []:
                        if flt["table"] != ra.base_table and flt["table"] != ra.joined_table:
                            join_sql += f", {flt['table']}.{flt['column']} AS {flt['column']}"
                    join_sql += f"\nFROM {ra.base_table}"
                    for hop in ra.join_hops:
                        join_sql += f"\nINNER JOIN {hop['right_table']} ON {hop['left_table']}.{hop['left_key']} = {hop['right_table']}.{hop['right_key']}"
                    if getattr(ra, "filters", None):
                        from packages.analytics_core.src.sql.relational_compiler import _sql_literal
                        where_parts = [
                            f"{f['table']}.{f['column']} = {_sql_literal(f['value'])}"
                            for f in ra.filters
                        ]
                        join_sql += "\nWHERE " + " AND ".join(where_parts)

                    import duckdb
                    con = duckdb.connect(":memory:")
                    try:
                        for tbl_name, tbl_df in (data_context.datasets_map or {}).items():
                            con.register(tbl_name, tbl_df)
                        joined_df = con.execute(join_sql).fetchdf()
                        if joined_df is not None and not joined_df.empty:
                            analyst_df = joined_df
                    finally:
                        con.close()
                except Exception:
                    analyst_df = primary_df

            analyst_result = build_analyst_result(
                question,
                analyst_df,
                target=canonical_target_col,
                group=canonical_group_dimension_col,
                explanatory=_analyst_predictors,
                time_col=canonical_time_col,
                default_aggregation=_agg_default,
            )
        except Exception as analyst_exc:  # noqa: BLE001 -- the answer layer must never fail an investigation
            analyst_result = None
            self.checkpointer.record_event(
                investigation_id=investigation_id, execution_id=exec_id,
                event_type="investigation.analyst_result_failed", payload={"error": str(analyst_exc)},
            )
        analyst_verdict_note = ""
        # Churn/confounding questions carry identifiability logic (cohort traps, Simpson's) that the
        # loop owns; a marginal recomputation must not overwrite or up-grade those outcomes.
        _loop_owns_answer = bool(
            churn_verdict_val
            or is_churn_unidentifiable
            or "churn" in (question or "").lower()
            or "confound" in str(direct_ans or "").lower()
        )
        if analyst_result is not None and not _loop_owns_answer and verdict_eval.verdict_type != "REFUTED":
            _loop_answer = direct_ans
            _loop_verdict = verdict_eval.verdict_type
            _gates_clear = (
                claim_admission.allowed
                and not is_churn_unidentifiable
                and (latest_missingness_result is None
                     or latest_missingness_result.classification not in ("SENSITIVE", "UNIDENTIFIABLE", "INSUFFICIENT_EVIDENCE"))
            )
            if analyst_result.descriptive and _gates_clear and (
                verdict_eval.verdict_type == "INCONCLUSIVE" or analyst_result.supersedes_loop
            ):
                # A ranking/total is a recomputed fact, not a hypothesis: "inconclusive" was wrong.
                verdict_eval.verdict_type = "OBSERVED"
                verdict_eval.direct_answer = analyst_result.to_text()
                analyst_verdict_note = (
                    "Verdict OBSERVED: the question asks for a recomputed aggregate or a located change, not a tested hypothesis"
                    + (f" (the hypothesis loop's own verdict {_loop_verdict} tested a different claim and is not the answer)." if analyst_result.supersedes_loop and _loop_verdict != "INCONCLUSIVE" else ".")
                )
                direct_ans = analyst_result.to_text()
                main_find = f"{analyst_result.headline} (deterministic recomputation from the resolved contract columns)"
            elif analyst_result.finding == "none" and _gates_clear and analyst_result.numbers.get("adequate_power") and verdict_eval.verdict_type == "INCONCLUSIVE":
                # DEFECT-026 Limit 1: A powered null result is NO_DETECTABLE_EFFECT, not INCONCLUSIVE!
                verdict_eval.verdict_type = "NO_DETECTABLE_EFFECT"
                calc_pwr = float(analyst_result.numbers.get("power", 0.95))
                # Calibrated epistemic certainty: 95% CI bound and power, bounded between 0.80 and 0.95 (never 1.0)
                verdict_eval.confidence_score = round(min(0.95, max(0.80, calc_pwr)), 2)
                verdict_eval.direct_answer = analyst_result.to_text()
                analyst_verdict_note = (
                    "Verdict NO_DETECTABLE_EFFECT: adequately powered statistical test found no detectable difference; "
                    "bound established by 95% confidence interval."
                )
                direct_ans = analyst_result.to_text()
                main_find = f"{analyst_result.headline} (deterministic recomputation from the resolved contract columns)"
            else:
                if (
                    verdict_eval.verdict_type in ("DIAGNOSED", "STATISTICALLY_SIGNIFICANT")
                    and analyst_result.numbers.get("robust") is False
                ):
                    analyst_verdict_note = (
                        f"Downgraded from {verdict_eval.verdict_type}: the independent analyst recomputation "
                        "did not survive its robustness check (rank-based test / trimmed means)."
                    )
                    verdict_eval.verdict_type = "INCONCLUSIVE"
                    verdict_eval.confidence_score = 0.0
                direct_ans = analyst_result.to_text()
                if _loop_answer and not analyst_result.supersedes_loop:
                    main_find = f"{main_find} Scientific-loop statement: {_loop_answer}"
            if analyst_verdict_note:
                main_find = f"{main_find} {analyst_verdict_note}"
            self.checkpointer.record_event(
                investigation_id=investigation_id, execution_id=exec_id,
                event_type="investigation.analyst_result",
                payload={**analyst_result.to_dict(), "verdict_note": analyst_verdict_note, "final_verdict": verdict_eval.verdict_type},
            )
        # Phase 11 Part 15: attach the analyst-readable metric-semantics
        # rationale to the human-facing finding, so the verdict explains HOW
        # the target metric was aggregated and why (not merely the number).
        # A refutation states its own basis (per-row group means), so the
        # aggregation note -- which describes a different computation -- is
        # omitted there rather than contradicting it.
        if semantic.metric_definition is not None and verdict_eval.verdict_type != "REFUTED":
            main_find = f"{main_find} {semantic.metric_definition.narrative_explanation()}"

        # Phase 12 (Section 21): disclosure of a material missingness-sensitivity
        # finding must reach the human-facing verdict text -- it must never be
        # computed but silently dropped by a downstream text-formulation step
        # (this was exactly the Phase 10 finding in test_scenario_06_mnar_missingness:
        # the check existed but never reached the verdict). Appended here, after
        # ProvenanceEngine's own formulation, so no future change to that
        # formulation function can silently swallow it again.
        if latest_missingness_result is not None and latest_missingness_result.classification in (
            "SENSITIVE", "UNIDENTIFIABLE", "INSUFFICIENT_EVIDENCE",
        ):
            direct_ans = (
                f"{direct_ans} [Missingness sensitivity: {latest_missingness_result.classification} -- "
                f"{latest_missingness_result.rationale}]"
            )

        # Data-quality qualification gate (moved here from earlier in the
        # function -- see the NOTE above `assess_downstream_impact`'s call
        # site): applied after verdict_eval, direct_ans, and main_find have
        # all reached their final values, so this has the final say on the
        # verdict and its text reflects the actually-reported answer rather
        # than an intermediate one that later logic (e.g. the analyst
        # numbers-first reconciliation above) could still overwrite.
        if (
            final_quality_decision.claim_qualification_required
            and verdict_eval.verdict_type in {"DIAGNOSED", "STATISTICALLY_SIGNIFICANT"}
        ):
            verdict_eval.verdict_type = "INCONCLUSIVE"
            verdict_eval.confidence_score = 0.0
            quality_note = "Data-quality safeguards prevented a high-confidence conclusion because: " + "; ".join(final_quality_decision.blocking_reasons)
            direct_ans = f"{direct_ans} [{quality_note}]"
            main_find = f"{main_find} {quality_note}"
            verdict_eval.direct_answer = direct_ans
            verdict_eval.main_finding = main_find

        manifest_hash = ProvenanceEngine.compute_reproducible_manifest_hash(
            investigation_id=investigation_id,
            question=question,
            primary_dataset=semantic.primary_dataset_name,
            executed_experiments=executed_codes,
            final_verdict=verdict_eval.verdict_type,
            epistemic_vector=epistemic_vector,
        )
        self.checkpointer.record_event(
            investigation_id=investigation_id,
            execution_id=exec_id,
            event_type="calculation.provenance_manifest",
            payload={
                "trace_id": f"TRACE-{investigation_id}:MANIFEST",
                "formula": "SHA-256(canonical ordered investigation provenance manifest)",
                "inputs": {
                    "investigation_id": investigation_id,
                    "question": question,
                    "primary_dataset": semantic.primary_dataset_name,
                    "executed_experiments": executed_codes,
                    "final_verdict": verdict_eval.verdict_type,
                },
                "output": {"reproducible_manifest_hash": manifest_hash},
            },
        )
        # Grounded Business Decision Recommendation
        # P0 epistemic-integrity rule: recommendation creation is itself a
        # claim-admission boundary.  Evidence IDs alone are not sufficient.
        # The leading hypothesis must have explicitly linked supporting
        # evidence, every supporting item must currently be independently
        # VERIFIED, and a verified contradiction blocks the recommendation.
        decision_recs: List[DecisionRecommendation] = []
        recommendation_grounding = None
        _proposed_recommendation_id = None
        if (
            verdict_eval.verdict_type in ("DIAGNOSED", "STATISTICALLY_SIGNIFICANT")
            and top_prob >= 0.70
            and final_quality_decision.recommendation_allowed
        ):
            _proposed_recommendation_id = f"REC-{gen_uuid()[:8]}"
            _supporting_ids = tuple(dict.fromkeys(getattr(leading_hyp, "supporting_evidence_ids", None) or []))
            _contradicting_ids = tuple(dict.fromkeys(getattr(leading_hyp, "contradicting_evidence_ids", None) or []))
            _grounding_ids = tuple(dict.fromkeys(_supporting_ids + _contradicting_ids))
            _validation_map = {}
            _latest_verification_map = {}
            if _grounding_ids:
                with self.session_factory() as _grounding_session:
                    _evidence_rows = (
                        _grounding_session.query(Evidence)
                        .filter(Evidence.id.in_(_grounding_ids))
                        .all()
                    )
                    _validation_map = {str(e.id): str(getattr(e, "validation_status", "")) for e in _evidence_rows}
                    _verification_rows = (
                        _grounding_session.query(EvidenceVerification)
                        .filter(EvidenceVerification.evidence_id.in_(_grounding_ids))
                        .order_by(EvidenceVerification.created_at.asc())
                        .all()
                    )
                    for _vr in _verification_rows:
                        # Last verification is authoritative for current
                        # admissibility. A prior PASS cannot survive a newer FAIL.
                        _latest_verification_map[str(_vr.evidence_id)] = str(getattr(_vr, "status", ""))

            recommendation_grounding = evaluate_recommendation_grounding(
                supporting_evidence_ids=_supporting_ids,
                contradicting_evidence_ids=_contradicting_ids,
                evidence_validation_status_by_id=_validation_map,
                latest_verification_status_by_id=_latest_verification_map,
            )
            self.checkpointer.record_event(
                investigation_id=investigation_id,
                execution_id=exec_id,
                event_type="decision.recommendation.grounding_evaluated",
                payload={
                    "recommendation_id": _proposed_recommendation_id,
                    "recommendation_admissibility": recommendation_grounding.status,
                    "grounded_hypothesis_id": f"{investigation_id}_{leading_hyp.hypothesis_code}",
                    "supporting_evidence_ids": list(recommendation_grounding.supporting_evidence_ids),
                    "verified_supporting_evidence_ids": list(recommendation_grounding.verified_supporting_evidence_ids),
                    "unverified_supporting_evidence_ids": list(recommendation_grounding.unverified_supporting_evidence_ids),
                    "verified_contradicting_evidence_ids": list(recommendation_grounding.verified_contradicting_evidence_ids),
                    "reason": recommendation_grounding.reason,
                },
            )

        if recommendation_grounding is not None and recommendation_grounding.status == "GROUNDED":
            target_seg = getattr(leading_hyp, "target_value", None) or canonical_group_dimension_col
            # Audit P1 fix: the gain/risk/net math is no longer computed
            # inline here -- it's delegated to DecisionUtilityEngine, the
            # single authoritative implementation of expected-utility
            # calculation (Evidence -> DecisionUtilityEngine ->
            # ExpectedUtilityCalculation -> DecisionRecommendationRecord).
            # The controller's job is only to decide WHETHER a
            # recommendation is warranted (the verdict_type / top_prob
            # gate above) and how to describe the action; see
            # DecisionUtilityEngine for the grounded-utility rationale
            # (no fabricated dollar values, expressed in eta-squared
            # variance-explained points weighted by verified posterior).
            _utility = DecisionUtilityEngine.compute(
                variance_explained_pct=variance_explained_pct,
                posterior_probability=top_prob,
            )
            self.checkpointer.record_event(
                investigation_id=investigation_id,
                execution_id=exec_id,
                event_type="calculation.decision_utility",
                payload={
                    "trace_id": f"TRACE-{investigation_id}:DECISION-UTILITY",
                    "formula": getattr(_utility, "utility_function_description", "authoritative decision utility calculation"),
                    "inputs": {
                        "variance_explained_pct": variance_explained_pct,
                        "posterior_probability": top_prob,
                    },
                    "outputs": {
                        "expected_gain_metric": _utility.expected_gain_metric,
                        "downside_risk_metric": _utility.downside_risk_metric,
                        "net_expected_utility": _utility.net_expected_utility,
                    },
                },
            )
            decision_recs.append(
                DecisionRecommendation(
                    recommendation_id=_proposed_recommendation_id or f"REC-{gen_uuid()[:8]}",
                    action_title=f"Targeted mitigation focused on {target_seg}",
                    action_description=f"Implement targeted mitigation strategy focused on {target_seg}. Empirical evidence establishes concentration ({top_prob*100:.1f}% confidence).",
                    grounded_hypothesis_id=f"{investigation_id}_{leading_hyp.hypothesis_code}",
                    target_metric=canonical_target_col,
                    expected_utility=ExpectedUtilityCalculation(
                        action_name=f"Focus on {target_seg}",
                        expected_gain_metric=_utility.expected_gain_metric,
                        downside_risk_metric=_utility.downside_risk_metric,
                        probability_of_success=float(top_prob),
                        net_expected_utility=_utility.net_expected_utility,
                        utility_function_description=_utility.utility_function_description,
                    ),
                    policy_compliance_passed=True,
                    required_preconditions=["Data distribution stability"],
                )
            )

        def step_verdict(session: Session):
            v_entity = InvestigationVerdict(
                id=gen_uuid(),
                investigation_id=investigation_id,
                verdict_type=verdict_eval.verdict_type,
                confidence_score=(
                    1.0 if (verdict_eval.verdict_type == "OBSERVED" and analyst_result is not None and analyst_result.descriptive)
                    else round(min(0.95, max(0.80, float(analyst_result.numbers.get("power", 0.95)))), 2) if (verdict_eval.verdict_type == "NO_DETECTABLE_EFFECT" and analyst_result is not None)
                    else 0.90 if verdict_eval.verdict_type == "NO_DETECTABLE_EFFECT"
                    else top_prob
                ),
                justification=(
                    f"{verdict_eval.justification} Stopping reason: {stopping_reason}."
                    + (f" {temporal_provenance_note}" if temporal_provenance_note else "")
                ),
                net_variance_explained_pct=variance_explained_pct or 0.0,
                counter_hypothesis_refuted=(
                    verdict_eval.counter_hypothesis_refuted
                    if hasattr(verdict_eval, "counter_hypothesis_refuted")
                    else False
                ),
                epistemic_grade=epistemic_vector.overall_epistemic_grade,
            )
            session.merge(v_entity)

            # Persist grounded decision recommendations (DEFECT-009 fix):
            # previously `decision_recs` was built above and discarded --
            # never reaching the DB or the API. Persist each one now, in
            # the same transaction as the verdict itself.
            for rec in decision_recs:
                session.merge(
                    DecisionRecommendationRecord(
                        id=gen_uuid(),
                        investigation_id=investigation_id,
                        recommendation_id=rec.recommendation_id,
                        action_title=rec.action_title,
                        action_description=rec.action_description,
                        grounded_hypothesis_id=rec.grounded_hypothesis_id,
                        target_metric=rec.target_metric,
                        expected_gain_metric=rec.expected_utility.expected_gain_metric,
                        downside_risk_metric=rec.expected_utility.downside_risk_metric,
                        probability_of_success=rec.expected_utility.probability_of_success,
                        net_expected_utility=rec.expected_utility.net_expected_utility,
                        utility_function_description=rec.expected_utility.utility_function_description,
                        policy_compliance_passed=rec.policy_compliance_passed,
                        required_preconditions_json=list(rec.required_preconditions),
                    )
                )

            inv_rec = session.query(Investigation).filter(Investigation.id == investigation_id).first()
            if inv_rec:
                inv_rec.status = InvestigationState.COMPLETED
                inv_rec.verdict_type = verdict_eval.verdict_type
                inv_rec.direct_answer = direct_ans
                inv_rec.main_finding = main_find
                inv_rec.entropy_current = final_entropy
                inv_rec.confidence_score = 0.0 if (verdict_eval.verdict_type in ("INCONCLUSIVE", "REFUTED") or is_churn_unidentifiable) else (
                    # An OBSERVED descriptive answer is an exact recomputation of a sample statistic, not a belief
                    # about the world; the leading hypothesis's posterior is irrelevant to it.
                    1.0 if (verdict_eval.verdict_type == "OBSERVED" and analyst_result is not None and analyst_result.descriptive)
                    else round(min(0.95, max(0.80, float(analyst_result.numbers.get("power", 0.95)))), 2) if (verdict_eval.verdict_type == "NO_DETECTABLE_EFFECT" and analyst_result is not None)
                    else 0.90 if verdict_eval.verdict_type == "NO_DETECTABLE_EFFECT"
                    else top_prob
                )

                inv_rec.reproducible_manifest_hash = manifest_hash
                # Session 8: these two columns were only ever written for compound
                # objectives, so the API reported stopping_criteria_met=False for
                # every ordinary investigation. Persist the real outcome so a
                # human verifier can tell "converged" from "ran out of budget".
                inv_rec.stopping_criteria_met = bool(
                    authoritative_stop_decision.should_stop if authoritative_stop_decision is not None else False
                )
                inv_rec.stopping_rationale = stopping_reason

            # NOTE: this must reuse the already-open `session` (the same
            # transactional session `step_verdict` runs under, via
            # execute_step_transactionally) rather than opening a second,
            # independent session here. `inv_rec` above already has
            # uncommitted writes staged against the *same* on-disk SQLite
            # file; opening a second self.session_factory() session and
            # calling commit() on it while that outer transaction is still
            # open contends for the same single-writer SQLite lock and
            # reliably raises "database is locked" -- which the except
            # branch below then tried to log via checkpointer.record_event,
            # itself doing another db.commit() against the same locked
            # file, so the *logging of the failure* failed too and the
            # original error propagated unhandled instead of being caught.
            try:
                inv_contract = inv_rec
                if inv_contract and inv_contract.active_contract_id:
                    active_contract = session.query(InvestigationContract).filter(InvestigationContract.id == inv_contract.active_contract_id).first()
                    if active_contract:
                        active_contract.status = "COMPLETED"
                        active_contract.current_phase = AnalyticalPhase.FINALIZING
                        state = dict(active_contract.execution_state_json or {})
                        state.update({"final_verdict": verdict_eval.verdict_type, "stopping_reason": stopping_reason, "provenance_hash": manifest_hash})
                        active_contract.execution_state_json = state
                        inv_contract.current_phase = AnalyticalPhase.FINALIZING
            except Exception as contract_exc:
                self.checkpointer.record_event(investigation_id=investigation_id, execution_id=exec_id, event_type="investigation.contract_finalize_persistence_failed", payload={"error": str(contract_exc)})

            return {
                "verdict": verdict_eval.verdict_type,
                "confidence": top_prob,
                "manifest_hash": manifest_hash,
                "stopping_reason": stopping_reason,
            }

        self.checkpointer.execute_step_transactionally(
            investigation_id=investigation_id,
            execution_id=exec_id,
            step_index=step_counter,
            step_type="VERDICT_FORMULATION",
            step_code="VERDICT-01",
            step_fn=step_verdict,
        )

        state_mgr.finalize_verdict(verdict_eval.verdict_type, manifest_hash)

        # Persist a deterministic snapshot of the scientific state so recovery can
        # prove that hypotheses/evidence/beliefs/verdict are reconstructible from
        # durable state rather than transient Python objects.
        try:
            with self.session_factory() as snapshot_session:
                snapshot = build_scientific_state_snapshot(snapshot_session, investigation_id, runtime_state=state_mgr.get_state())
            self.checkpointer.record_event(
                investigation_id=investigation_id, execution_id=exec_id,
                event_type="investigation.scientific_state_snapshot",
                payload={"snapshot_hash": snapshot["snapshot_hash"], "snapshot": snapshot},
            )
        except Exception as snapshot_exc:
            self.checkpointer.record_event(
                investigation_id=investigation_id, execution_id=exec_id,
                event_type="investigation.scientific_state_snapshot_failed",
                payload={"error": str(snapshot_exc)},
            )

        # Transition to COMPLETED
        self.checkpointer.complete_execution(
            investigation_id=investigation_id,
            execution_id=exec_id,
            manifest_hash=manifest_hash,
        )

        return True