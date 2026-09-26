"""Deterministic analytical problem-classification and method-selection policy for AA-OS.

This module is a policy layer only. It does not execute SQL/statistics, infer metric
semantics independently, or perform causal identification. It consumes existing
IntentEngine/SemanticEngine outputs and produces one auditable decision that downstream
engines can consume.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Iterable
import re
import pandas as pd

from packages.schemas.src.analysis import (
    AggregationType,
    CausalIntent,
    EpistemicClaimType,
    GrainRef,
    MetricRef,
    ObjectiveType,
    TemporalScope,
    VariableRef,
)
from packages.analytics_core.src.semantic.metric_semantics import MetricDefinition
from packages.analytics_core.src.statistics.universal_preflight import inspect_design
from packages.analytics_core.src.statistics.preflight import (
    binary_target, forecast_target, independent_groups, numeric_pair,
)
# v20-C4: CanonicalSemanticResolution is now the authoritative semantic-role
# source for method selection (see docs/AAOS_V20C4_CANONICAL_SEMANTIC_AUTHORITY.md).
# decide() builds this once, at the same entry boundary predictive_hypothesis.py
# already builds it at for hypothesis synthesis, and threads it through every
# _select_* branch so EstimandSpec's role fields (target_column,
# comparison_dimension, time_column, predictor_columns, churn_bindings) are
# derived from canonical accessors instead of independently re-reading
# semantic.target_metric_col / semantic.group_dimension_col / semantic.time_col
# / semantic.secondary_metric_col / semantic.churn_*_col a second time.
from packages.analytics_core.src.intelligence.semantic_resolution_builder import (
    build_canonical_semantic_resolution,
)
from packages.schemas.src.semantic_resolution_contract import CanonicalSemanticResolution, QuestionRoleProposal


class ProblemClass(str, Enum):
    DESCRIPTIVE = "DESCRIPTIVE"
    DIAGNOSTIC = "DIAGNOSTIC"
    COMPARATIVE = "COMPARATIVE"
    CORRELATIONAL = "CORRELATIONAL"
    FORECASTING = "FORECASTING"
    SURVIVAL_CHURN = "SURVIVAL_CHURN"
    CAUSAL = "CAUSAL"
    FALLBACK = "FALLBACK"
    # v19: added so every canonical UniversalQuestionCompiler task has a
    # 1:1 ProblemClass counterpart (see CANONICAL_TASK_AUTHORITY below).
    # Existing members above are reused where the concept already matched
    # (ASSOCIATION->CORRELATIONAL, FORECAST->FORECASTING, COMPARISON->
    # COMPARATIVE, DIAGNOSTIC->DIAGNOSTIC, DESCRIPTIVE->DESCRIPTIVE,
    # CAUSAL->CAUSAL) rather than being duplicated here.
    PREDICTION = "PREDICTION"
    SEGMENTATION = "SEGMENTATION"
    RECONCILIATION = "RECONCILIATION"
    DATA_QUALITY = "DATA_QUALITY"
    GOVERNANCE = "GOVERNANCE"
    PRESCRIPTIVE = "PRESCRIPTIVE"
    GENERAL_EXPLORATION = "GENERAL_EXPLORATION"


class TaskAuthorityStatus(str, Enum):
    """v19: explicit per-task registry status (AAOS_V19 section 2).

    Replaces the previous silent ``if not candidates: return decision``
    fallthrough in ``select_for_plan`` with an auditable classification for
    every task UniversalQuestionCompiler can emit.
    """
    IMPLEMENTED = "IMPLEMENTED"
    SPECIALIST_BATTERY = "SPECIALIST_BATTERY"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True)
class TaskAuthorityEntry:
    status: TaskAuthorityStatus
    problem_class: "ProblemClass"
    # False when a MethodCapability/battery exists but the runtime
    # SPECIALIST AUTHORITY GATE in controller.py still blocks execution
    # because the canonical transition chain (computation -> independent
    # verification -> evidence -> belief -> stopping -> claim gate ->
    # verdict -> provenance) is not fully wired for this task yet. This is
    # the single source both select_for_plan and the controller's gate
    # read, replacing the controller's previously-separate hardcoded set.
    canonical_transition_wired: bool = True


# v19 section 2 + section 6: one explicit, auditable mapping from every
# canonical UniversalQuestionCompiler task to (a) whether AA-OS has a real
# analytical method for it today, and (b) which ProblemClass it corresponds
# to. This is the single place that answers "is this task implemented?" --
# nothing else in the codebase should independently decide that question.
CANONICAL_TASK_AUTHORITY: Dict[str, TaskAuthorityEntry] = {
    "ASSOCIATION": TaskAuthorityEntry(TaskAuthorityStatus.IMPLEMENTED, ProblemClass.CORRELATIONAL),
    "FORECAST": TaskAuthorityEntry(TaskAuthorityStatus.IMPLEMENTED, ProblemClass.FORECASTING),
    "PREDICTION": TaskAuthorityEntry(TaskAuthorityStatus.IMPLEMENTED, ProblemClass.PREDICTION),
    "SEGMENTATION": TaskAuthorityEntry(TaskAuthorityStatus.IMPLEMENTED, ProblemClass.SEGMENTATION),
    "COMPARISON": TaskAuthorityEntry(TaskAuthorityStatus.IMPLEMENTED, ProblemClass.COMPARATIVE),
    # row_reconciliation exists in METHOD_REGISTRY and is scored like any
    # other IMPLEMENTED method, but the controller's SPECIALIST AUTHORITY
    # GATE still blocks it pre-execution (no canonical transition executor
    # wired yet) -- tracked honestly via canonical_transition_wired=False
    # rather than pretending it either fully works or doesn't exist.
    "RECONCILIATION": TaskAuthorityEntry(TaskAuthorityStatus.IMPLEMENTED, ProblemClass.RECONCILIATION, canonical_transition_wired=False),
    "DIAGNOSTIC": TaskAuthorityEntry(TaskAuthorityStatus.SPECIALIST_BATTERY, ProblemClass.DIAGNOSTIC),
    "DESCRIPTIVE": TaskAuthorityEntry(TaskAuthorityStatus.SPECIALIST_BATTERY, ProblemClass.DESCRIPTIVE),
    "GENERAL_EXPLORATION": TaskAuthorityEntry(TaskAuthorityStatus.SPECIALIST_BATTERY, ProblemClass.GENERAL_EXPLORATION),
    # CAUSAL is a SPECIALIST_BATTERY (routed to the diagnostic battery with
    # a capped verdict tier) rather than IMPLEMENTED: there is no dedicated
    # causal-estimation method, only the conservative observational-only
    # gate in _select_causal_or_diagnostic.
    "CAUSAL": TaskAuthorityEntry(TaskAuthorityStatus.SPECIALIST_BATTERY, ProblemClass.CAUSAL),
    # No MethodCapability exists for these and the controller's SPECIALIST
    # AUTHORITY GATE already refuses to complete them -- explicit
    # UNSUPPORTED_ANALYTICAL_TASK rather than a silent fallback.
    "DATA_QUALITY": TaskAuthorityEntry(TaskAuthorityStatus.UNSUPPORTED, ProblemClass.DATA_QUALITY, canonical_transition_wired=False),
    "GOVERNANCE": TaskAuthorityEntry(TaskAuthorityStatus.UNSUPPORTED, ProblemClass.GOVERNANCE, canonical_transition_wired=False),
    "PRESCRIPTIVE": TaskAuthorityEntry(TaskAuthorityStatus.UNSUPPORTED, ProblemClass.PRESCRIPTIVE, canonical_transition_wired=False),
}

# v19 section 1 + section 11: two ProblemClass values represent a genuine
# methodological *specialization* discovered by MethodSelectionEngine.decide()
# that the compiler's lexical task classifier cannot see, and in both cases
# decide()'s check is deliberately the more conservative/precise one:
#   - SURVIVAL_CHURN: fires only once a churn outcome column is positively
#     semantically resolved (semantic.churn_event_col); the compiler task
#     vocabulary has no CHURN task at all (churn questions compile to
#     PREDICTION/DIAGNOSTIC/COMPARISON depending on phrasing).
#   - CAUSAL: decide()'s _has_explicit_causal_effect_language regex is
#     intentionally *narrower* than the compiler's CAUSAL classifier (which
#     matches on the bare stem "caus(e|ed|es|al)" and would misclassify
#     plain "what is the root cause of X" as CAUSAL). Overwriting decide()'s
#     DIAGNOSTIC-with-no-verdict-cap determination with a blind CAUSAL label
#     from the compiler would silently remove the overclaiming guard section
#     11 requires.
# For these two, the canonical task is still recorded (decision.canonical_task)
# for audit/conflict purposes, but problem_class is NOT overwritten from the
# compiler task -- decide()'s own determination wins because it is strictly
# more information, not less. Every other task's problem_class is fully
# authoritative from the compiler per section 1.
_METHOD_SELECTION_SPECIALIZATIONS = frozenset({ProblemClass.SURVIVAL_CHURN, ProblemClass.CAUSAL})


class MethodFamily(str, Enum):
    DIAGNOSTIC_BATTERY = "GENERIC_BATTERY"
    CORRELATION = "CORRELATION"
    FORECAST = "FORECAST"
    CHURN = "CHURN"


VERDICT_TIER_STATISTICALLY_SIGNIFICANT = "STATISTICALLY_SIGNIFICANT"

# Do not treat ordinary "cause" / "root cause" wording as a causal-effect claim.
# Those questions are diagnostic unless the question explicitly asks about a causal
# effect, intervention, treatment effect, or causal relationship.
_CAUSAL_EFFECT_PATTERNS = (
    re.compile(r"\bcausal(?:ly)?\b"),
    re.compile(r"\beffect\s+of\b"),
    re.compile(r"\btreatment\s+effect\b"),
    re.compile(r"\bintervention\b"),
    re.compile(r"\bcause\s+(?:[^?]*?)\s+on\b"),
    re.compile(r"\b(?:did|does|do|would|will|can|could)\s+[^?]*\bcause\b"),
)


@dataclass
class EstimandSpec:
    metric_ref: Optional[MetricRef] = None
    metric_definition: Optional[MetricDefinition] = None
    unit_of_analysis: Optional[GrainRef] = None
    temporal_scope: Optional[TemporalScope] = None
    population_filters: List[Any] = field(default_factory=list)
    numerator_column: Optional[str] = None
    denominator_column: Optional[str] = None
    weight_column: Optional[str] = None
    is_additive: Optional[bool] = None
    is_compositional: bool = False
    aggregation: Optional[AggregationType] = None
    comparison_dimension: Optional[str] = None
    time_column: Optional[str] = None
    treatment_variable: Optional[VariableRef] = None
    outcome_variable: Optional[VariableRef] = None
    churn_bindings: Optional[Dict[str, Any]] = None
    # v20-C3: explicit, single-source analytical-role bindings for the
    # bivariate/multivariate case (currently CORRELATIONAL). target_column
    # is the resolved outcome/response column; predictor_columns is the
    # resolved exposure/explanatory column(s). These are set exactly once,
    # by the MethodSelectionEngine._select_* method that produces this
    # estimand, from the same semantic resolution the rest of the estimand
    # already uses -- so ExperimentSynthesizer can consume them instead of
    # independently re-reading semantic.target_metric_col /
    # semantic.secondary_metric_col (see AAOS_V20C3_CANONICAL_EXPERIMENT_CONTRACT.md).
    # Left unset (None / []) for problem classes where no such binding has
    # been resolved yet; consumers must treat that as "not yet authoritative"
    # rather than silently falling back to a stale semantic field.
    target_column: Optional[str] = None
    predictor_columns: List[str] = field(default_factory=list)
    # Explicit user wording such as "controlling for", "holding constant", or
    # "after accounting for" requests a joint conditional model, not several
    # independent pairwise correlations. This flag is set from the canonical
    # question at method-selection time and is persisted in estimand provenance.
    joint_predictors: bool = False
    # v20-C4.2.1: variables SemanticEngine discovered (e.g. its
    # secondary_metric_col) that the question did NOT ask about. Recorded
    # for provenance and as exploratory/confounder candidates; never the
    # answer to the question -- predictor_columns is the requested role.
    discovered_columns: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    def deterministic_predictor_columns(self) -> List[str]:
        return sorted(dict.fromkeys(self.predictor_columns))

    @staticmethod
    def _serialize_temporal_scope(scope: Any) -> Optional[Dict[str, Any]]:
        """Serialize either the Pydantic TemporalScope or the resolver dataclass.

        Method-selection provenance can be constructed before the typed IR is
        compiled, so it must tolerate both representations without weakening
        the provenance contract.
        """
        if scope is None:
            return None
        if hasattr(scope, "model_dump"):
            value = scope.model_dump()
        else:
            value = {
                "column": getattr(scope, "column", None),
                "start": getattr(scope, "period_start", None),
                "end": getattr(scope, "period_end", None),
                "comparison_start": getattr(scope, "baseline_start", None),
                "comparison_end": getattr(scope, "baseline_end", None),
            }
        def _safe(v: Any) -> Any:
            if hasattr(v, "model_dump"):
                return v.model_dump()
            if hasattr(v, "isoformat"):
                return v.isoformat()
            return v
        return {str(k): _safe(v) for k, v in value.items()}

    def to_provenance_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric_ref.model_dump() if self.metric_ref is not None else None,
            "unit_of_analysis": self.unit_of_analysis.model_dump() if self.unit_of_analysis is not None else None,
            "temporal_scope": self._serialize_temporal_scope(self.temporal_scope),
            "population_filters": [f.model_dump() if hasattr(f, "model_dump") else str(f) for f in self.population_filters],
            "numerator_column": self.numerator_column,
            "denominator_column": self.denominator_column,
            "weight_column": self.weight_column,
            "is_additive": self.is_additive,
            "is_compositional": self.is_compositional,
            "aggregation": self.aggregation.value if isinstance(self.aggregation, AggregationType) else self.aggregation,
            "comparison_dimension": self.comparison_dimension,
            "time_column": self.time_column,
            "treatment_variable": self.treatment_variable.model_dump() if self.treatment_variable else None,
            "outcome_variable": self.outcome_variable.model_dump() if self.outcome_variable else None,
            "churn_bindings": self.churn_bindings,
            "target_column": self.target_column,
            "predictor_columns": list(self.predictor_columns),
            "joint_predictors": bool(self.joint_predictors),
            "discovered_columns": list(self.discovered_columns),
            "limitations": list(self.limitations),
        }




@dataclass(frozen=True)
class MethodCapability:
    code: str
    supported_problem_classes: frozenset[str]
    required_dtypes: Dict[str, Any] = field(default_factory=dict)
    required_roles: Tuple[str, ...] = ()
    assumptions: Tuple[str, ...] = ()
    min_sample: int = 0
    estimand: str = ""
    uncertainty_method: str = ""
    verification_method: str = ""
    failure_conditions: Tuple[str, ...] = ()
    executor_id: str = ""


METHOD_REGISTRY: Dict[str, MethodCapability] = {
    "association_categorical_binary": MethodCapability(
        code="association_categorical_binary", supported_problem_classes=frozenset({"ASSOCIATION"}),
        required_roles=("exposure", "binary_outcome"), assumptions=("independent observations",), min_sample=20,
        estimand="group-specific risk/rate contrast", uncertainty_method="exact_or_asymptotic_interval",
        verification_method="independent_contingency_recomputation", failure_conditions=("target not binary", "no usable exposure categories"), executor_id="scientific_loop:association_categorical_binary"),
    "forecast_rolling_origin": MethodCapability(
        code="forecast_rolling_origin", supported_problem_classes=frozenset({"FORECAST"}),
        required_roles=("time", "target"), assumptions=("ordered temporal observations",), min_sample=12,
        estimand="future trajectory", uncertainty_method="out_of_sample_residual_interval", verification_method="independent_forecast_recompute",
        failure_conditions=("no time column", "insufficient history"), executor_id="specialized:forecast_rolling_origin"),
    "binary_risk_prediction": MethodCapability(
        code="binary_risk_prediction", supported_problem_classes=frozenset({"PREDICTION"}),
        required_roles=("binary_target",), min_sample=40, estimand="conditional event probability", uncertainty_method="holdout_metrics",
        verification_method="independent_scoring_recompute", failure_conditions=("non-binary target", "temporal leakage"), executor_id="specialized:binary_risk_prediction"),
    "stable_segmentation": MethodCapability(
        code="stable_segmentation", supported_problem_classes=frozenset({"SEGMENTATION"}),
        required_roles=("entity_rows", "features"), min_sample=30, estimand="descriptive latent partition", uncertainty_method="stability_analysis",
        verification_method="seed_stability_recompute", failure_conditions=("too_few_rows", "no_variation"), executor_id="specialized:stable_segmentation"),
    "row_reconciliation": MethodCapability(
        code="row_reconciliation", supported_problem_classes=frozenset({"RECONCILIATION"}),
        required_roles=("numeric_fields",), min_sample=2, estimand="row-level identity residual", uncertainty_method="residual_distribution",
        verification_method="independent_formula_recompute", failure_conditions=("insufficient_bindings",), executor_id="specialized:row_reconciliation"),
    "association_numeric": MethodCapability(
        code="association_numeric", supported_problem_classes=frozenset({"ASSOCIATION"}),
        required_roles=("numeric_exposure", "numeric_outcome"), assumptions=("paired rows represent comparable observational units",), min_sample=20,
        estimand="association coefficient with uncertainty", uncertainty_method="bootstrap_or_asymptotic_interval",
        verification_method="independent_correlation_recompute", failure_conditions=("non_numeric exposure/outcome",), executor_id="scientific_loop:association_numeric"),
    "association_multivariate_ols": MethodCapability(
        code="association_multivariate_ols", supported_problem_classes=frozenset({"ASSOCIATION"}),
        required_roles=("all_numeric_exposure", "numeric_outcome"), assumptions=(
            "numeric outcome and all requested predictors",
            "paired rows represent comparable observational units",
            "conditional association is observational unless a causal identification strategy is supplied",
        ), min_sample=30,
        estimand="conditional association of each predictor holding the other included predictors constant",
        uncertainty_method="OLS_HC3_or_clustered_inference",
        verification_method="independent_multivariate_regression_recompute",
        failure_conditions=("non_numeric predictor/outcome", "insufficient degrees of freedom",),
        executor_id="scientific_loop:association_multivariate_ols"),
    "comparison_group_effect": MethodCapability(
        code="comparison_group_effect", supported_problem_classes=frozenset({"COMPARISON"}),
        required_roles=("grouping", "outcome"), assumptions=("group observations are independently interpretable at the selected grain",), min_sample=20,
        estimand="between-group outcome contrast", uncertainty_method="effect_size_and_interval",
        verification_method="independent_group_recompute", failure_conditions=("missing grouping or outcome",), executor_id="scientific_loop:comparison_group_effect"),
    "generic_diagnostic_battery": MethodCapability(
        code="generic_diagnostic_battery",
        supported_problem_classes=frozenset({"DIAGNOSTIC", "DESCRIPTIVE", "FALLBACK"}),
        min_sample=1,
        estimand="declared descriptive/diagnostic estimand",
        uncertainty_method="method-specific_or_non-inferential",
        verification_method="independent_recomputation_or_structural_check",
        failure_conditions=("unresolved_estimand",),
        executor_id="scientific_loop:generic_diagnostic_battery",
    ),
}


def _is_finite_entity_population(semantic: Any, df: Any) -> bool:
    """Return True only when semantic grain proves an observed entity/catalog table.

    Entity/catalog tables are descriptive finite populations: a row enumerates an
    observed entity rather than a sampled observation.  This permits descriptive
    between-group contrasts with singleton groups, while ordinary sampled data keep
    their inferential minimums.  The check is deliberately based on the already
    resolved semantic grain; it never guesses completeness from column names.
    """
    grain = str(getattr(semantic, "table_grain", "") or "").strip().lower()
    return grain.startswith(("entity_level (", "candidate_key (", "declared_key ("))


class MethodRegistry:
    """Queryable registry used by planners and specialist engines."""
    @classmethod
    def get(cls, code: str) -> MethodCapability:
        return METHOD_REGISTRY[code]

    @classmethod
    def executor_id(cls, code: str) -> str:
        return cls.get(code).executor_id

    @classmethod
    def preflight(cls, code: str, df: Any, plan: Any = None) -> Dict[str, Any]:
        """Run method-specific statistical/data-shape preflight before execution."""
        task = str(getattr(plan, "task", "")).upper() if plan is not None else ""
        semantic = getattr(plan, "semantics", None) if plan is not None else None
        target = getattr(semantic, "target_column", None) if semantic is not None else None
        reasons: List[str] = []
        result = None
        universal = inspect_design(
            df,
            target=target,
            grouping=(getattr(semantic, "group_dimension_col", None) if semantic is not None else None) or ((getattr(semantic, "grouping_columns", []) or [None])[0] if semantic is not None else None),
            # Only require a genuine multi-point time dimension for tasks
            # whose estimand actually depends on time (FORECAST). Passing
            # time_col unconditionally meant that any dataset with an
            # incidental, constant date field (e.g. a fixed signup_date/
            # observation_end_date recorded identically for every row --
            # common in churn datasets that aren't time-series data) tripped
            # "insufficient_time_points" and blocked admissibility for
            # COMPARISON/SEGMENTATION/ASSOCIATION questions that never
            # needed a time dimension in the first place.
            time_col=(getattr(semantic, "time_column", None) if semantic is not None and task == "FORECAST" else None),
            weights=(getattr(getattr(semantic, "metric_definition", None), "weight_column", None) if semantic is not None else None),
        )
        reasons.extend(universal.reasons)
        if code == "association_numeric":
            expl = list(getattr(semantic, "explanatory_columns", []) or [])
            if target and expl and all(c in df.columns for c in [expl[0], target]):
                result = numeric_pair(df[expl[0]], df[target], method=code, min_n=20)
            else:
                reasons.append("resolved_numeric_exposure_outcome_missing")
        elif code == "association_multivariate_ols":
            requested = list(getattr(semantic, "explanatory_columns", []) or [])
            if not target or target not in df.columns or not requested:
                reasons.append("resolved_multivariate_predictors_or_outcome_missing")
            else:
                missing = [c for c in requested if c not in df.columns]
                non_numeric = [c for c in requested + [target] if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])]
                if missing:
                    reasons.append(f"requested_predictors_missing:{missing}")
                if non_numeric:
                    reasons.append(f"non_numeric_predictor_or_outcome:{non_numeric}")
                usable_n = int(df[requested + [target]].dropna().shape[0]) if not missing else 0
                if usable_n < 30:
                    reasons.append(f"insufficient_rows_for_multivariate_regression:{usable_n}<30")
        elif code == "association_categorical_binary":
            if not target or target not in df.columns:
                reasons.append("binary_outcome_column_missing")
            else:
                result = binary_target(df[target], method=code, min_n=20)
                # A categorical exposure with one observed level is not an association problem.
                expl = list(getattr(semantic, "explanatory_columns", []) or [])
                group = list(getattr(semantic, "grouping_columns", []) or [])
                exposure_cols = expl or group
                if exposure_cols and exposure_cols[0] in df.columns and int(df[exposure_cols[0]].dropna().nunique()) < 2:
                    reasons.append("exposure_has_fewer_than_two_categories")
        elif code == "forecast_rolling_origin":
            if not target or target not in df.columns:
                reasons.append("forecast_target_missing")
            else:
                result = forecast_target(df[target], method=code, min_n=12)
        elif code == "binary_risk_prediction":
            if not target or target not in df.columns:
                reasons.append("binary_target_missing")
            else:
                result = binary_target(df[target], method=code, min_n=40)
        elif code == "comparison_group_effect":
            grouping = list(getattr(semantic, "grouping_columns", []) or [])
            grouping = grouping or ([getattr(semantic, "group_dimension_col", None)] if getattr(semantic, "group_dimension_col", None) else [])
            if not grouping or not target or grouping[0] not in df.columns or target not in df.columns:
                reasons.append("grouping_or_outcome_missing")
            else:
                groups = {k: g[target].tolist() for k, g in df[[grouping[0], target]].dropna().groupby(grouping[0])}
                # A proven entity/catalog grain is a finite observed population.
                # Do not apply sampled-data subgroup minima to it: singleton
                # categories can still be described exactly, but the result must
                # remain finite-population/descriptive (no population p-value).
                finite_population = _is_finite_entity_population(semantic, df)
                result = independent_groups(
                    groups, method=code, min_group_n=(1 if finite_population else 2), min_groups=2
                )
        elif code == "stable_segmentation":
            if len(df) < 30:
                reasons.append("too_few_rows:30")
            numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) > 1]
            if not numeric:
                reasons.append("no_varying_numeric_features")
        elif code == "row_reconciliation":
            numeric = list(df.select_dtypes(include=["number"]).columns) if hasattr(df, "select_dtypes") else []
            if len(numeric) < 2:
                reasons.append("fewer_than_two_numeric_fields")
        elif code == "generic_diagnostic_battery":
            if len(df) < 1:
                reasons.append("empty_dataset")
            if not target and not getattr(semantic, "metric_definition", None):
                reasons.append("unresolved_estimand")
        if result is not None:
            reasons.extend(result.reasons)
            return {**result.to_dict(), "reasons": list(dict.fromkeys(reasons)), "status": "BLOCKED" if reasons else result.status}
        return {"method": code, "status": "BLOCKED" if reasons else "ALLOWED", "reasons": list(dict.fromkeys(reasons)), "diagnostics": {"task": task}}

    @classmethod
    def for_problem_class(cls, problem_class: str) -> List[MethodCapability]:
        pc = str(problem_class).upper()
        return [m for m in METHOD_REGISTRY.values() if pc in m.supported_problem_classes]

    @classmethod
    def validate(cls, code: str, *, row_count: int, roles: Iterable[str], blocked_reasons: Iterable[str] = ()) -> Tuple[bool, List[str]]:
        cap = cls.get(code)
        errors: List[str] = []
        if row_count < cap.min_sample:
            errors.append(f"sample_size_below_minimum:{cap.min_sample}")
        missing_roles = set(cap.required_roles) - set(roles)
        errors.extend(f"missing_role:{r}" for r in sorted(missing_roles))
        errors.extend(f"blocked:{r}" for r in blocked_reasons)
        return (not errors, errors)

    @classmethod
    def admissibility_for_plan(cls, plan: Any, semantic: Any, quality: Any, df: Any, method_code: Optional[str] = None) -> Tuple[bool, Dict[str, Any]]:
        """Validate that the compiled analyst plan is methodologically executable.

        This is a policy gate, not an executor. It prevents specialist paths from
        running when the question's resolved roles, sample, or data-readiness state
        contradict the declared method capability. Generic diagnostic questions are
        intentionally not blocked here because their downstream engine performs its
        own assumption discovery.

        v20-C4.2.2c: `method_code` is optional and defaults to None for full
        backward compatibility with every existing caller (none of which
        passed it before this change) -- with method_code=None, this
        derives and validates its own best-fit code exactly as before.

        When `method_code` IS supplied (as select_for_plan()'s scoring loop
        now does, passing the specific candidate it is scoring), this
        function validates THAT method specifically. Previously it always
        derived its own `code` internally and returned that code's
        admissibility -- if the caller was actually asking about a
        *different* method (which happened whenever gate_method_for_plan()'s
        own, separately-computed suggestion disagreed with what this
        function's stricter internal derivation would have chosen), the
        caller would misapply the returned `ok` to the wrong method
        entirely. That is exactly how an invalid, non-numeric predictor
        (e.g. a string ID column) could get "association_numeric" scored
        as admissible: this function's own derivation correctly fell back
        to "generic_diagnostic_battery" (almost always admissible), but the
        caller was scoring "association_numeric" and applied that
        unrelated `ok=True` to it. See
        AAOS_V20C4_2_2B_COMPILER_SEMANTIC_AUTHORITY_AUDIT.md, Finding 3.

        Now, if `method_code` disagrees with this function's own derived
        code, the requested method is reported BLOCKED for that specific
        mismatch reason -- it is never silently answered using a different
        method's gate result.
        """
        task = str(getattr(plan, "task", "")).upper()
        mapping = {
            "PREDICTION": "binary_risk_prediction",
            "FORECAST": "forecast_rolling_origin",
            "SEGMENTATION": "stable_segmentation",
            "RECONCILIATION": "row_reconciliation",
            "DIAGNOSTIC": "generic_diagnostic_battery",
            "DESCRIPTIVE": "generic_diagnostic_battery",
            "FALLBACK": "generic_diagnostic_battery",
        }
        target = getattr(getattr(plan, "semantics", None), "target_column", None)
        expl = list(getattr(getattr(plan, "semantics", None), "explanatory_columns", []) or [])
        group = list(getattr(getattr(plan, "semantics", None), "grouping_columns", []) or [])
        code = None
        if task == "ASSOCIATION":
            if target and target in getattr(df, "columns", []):
                try:
                    target_is_binary = int(df[target].dropna().nunique()) == 2
                except Exception:
                    target_is_binary = False
                if target_is_binary and (expl or group):
                    code = "association_categorical_binary"
                elif expl and any(c in df.columns for c in expl):
                    try:
                        numeric_expl = [
                            c for c in expl
                            if c in df.columns and str(df[c].dtype).lower().startswith(("int", "float"))
                        ]
                        outcome_numeric = str(df[target].dtype).lower().startswith(("int", "float"))
                        has_valid_numeric = bool(numeric_expl) and outcome_numeric
                    except Exception:
                        has_valid_numeric = False
                    if has_valid_numeric:
                        code = "association_numeric"
        elif task == "COMPARISON":
            code = "comparison_group_effect"
        else:
            code = mapping.get(task)
        if not code:
            # Fail closed: an executable plan is never outside method policy.
            code = "generic_diagnostic_battery"

        # v20-C4.2.2c: if the caller asked specifically about a method that
        # this function's own derivation did NOT arrive at, that requested
        # method is not admissible -- full stop. Do not proceed to compute
        # and return `derived_code`'s own (possibly unrelated) gate result
        # under the requested method's name.
        if method_code is not None and method_code != code:
            requested_cap = cls.get(method_code)
            return False, {
                "status": "BLOCKED",
                "method": method_code,
                "task": task,
                "roles": sorted(roles_for_plan(plan, semantic, df)),
                "errors": [f"requested_method_mismatch: plan derives '{code}' for this task/role combination, not '{method_code}'"],
                "assumptions": list(requested_cap.assumptions) if requested_cap else [],
                "estimand": requested_cap.estimand if requested_cap else "",
                "uncertainty_method": requested_cap.uncertainty_method if requested_cap else "",
                "verification_method": requested_cap.verification_method if requested_cap else "",
                "executor_id": requested_cap.executor_id if requested_cap else "",
                "preflight": {"method": method_code, "status": "BLOCKED", "reasons": ["requested_method_mismatch"], "diagnostics": {"task": task, "derived_method": code}},
            }

        roles = set()
        if expl:
            roles.add("exposure")
            if all(c in df.columns for c in expl):
                roles.add("numeric_exposure") if all(str(df[c].dtype).lower().startswith(("int", "float")) for c in expl) else None
        if group or getattr(getattr(plan, "semantics", None), "grouping_columns", []):
            roles.add("grouping")
        if target and target in df.columns:
            roles.add("outcome")
            try:
                if int(df[target].dropna().nunique()) == 2:
                    roles.add("binary_outcome")
                    roles.add("binary_target")
                    if expl or group:
                        roles.add("exposure")
            except Exception:
                pass
            if str(df[target].dtype).lower().startswith(("int", "float")):
                roles.add("numeric_outcome")
        if task == "FORECAST":
            if getattr(getattr(plan, "semantics", None), "time_column", None) in df.columns:
                roles.add("time")
            roles.add("target")
        if task == "PREDICTION":
            roles.add("binary_target") if target and target in df.columns else None
        if task == "SEGMENTATION":
            roles.add("entity_rows")
            if len(df.columns) >= 2:
                roles.add("features")
        if task == "RECONCILIATION":
            try:
                if len(df.select_dtypes(include=["number"]).columns) >= 2:
                    roles.add("numeric_fields")
            except Exception:
                pass
        blocked: List[str] = []
        if quality is not None:
            blocked.extend(str(x) for x in (getattr(quality, "critical_issues", []) or []))
            if task == "PREDICTION" and getattr(quality, "leakage_indicators", None):
                blocked.append("prediction_leakage_indicators_present")
        ok = False
        # `comparison_group_effect` normally requires 20 rows for inferential
        # sampled-data analysis.  A proven finite entity/catalog population is a
        # different estimand: describe the observed entities exactly rather than
        # pretending the catalog is a random sample.  The downstream transition
        # and belief layers separately suppress inferential p-values for this case.
        if code == "comparison_group_effect" and _is_finite_entity_population(semantic, df):
            cap = cls.get(code)
            errors = list(blocked)
            missing_roles = set(cap.required_roles) - set(roles)
            errors.extend(f"missing_role:{r}" for r in sorted(missing_roles))
        else:
            ok, errors = cls.validate(code, row_count=len(df), roles=roles, blocked_reasons=blocked)
        preflight = cls.preflight(code, df, plan)
        preflight_reasons = list(preflight.get("reasons", []))
        if code == "comparison_group_effect" and _is_finite_entity_population(semantic, df):
            # Singleton categories are valid observations in a complete finite
            # catalog. The contrast remains descriptive over the observed
            # entities; suppress only the sampled-data singleton gate.
            preflight_reasons = [
                r for r in preflight_reasons
                if r not in {"group_contains_fewer_than_two_observations", "insufficient_groups:2"}
            ]
        errors.extend(f"statistical_preflight:{r}" for r in preflight_reasons)
        if code == "comparison_group_effect" and _is_finite_entity_population(semantic, df):
            ok = not errors
        else:
            ok = ok and not preflight_reasons
        return ok, {
            "status": "ALLOWED" if ok else "BLOCKED",
            "method": code,
            "task": task,
            "roles": sorted(roles),
            "errors": errors,
            "assumptions": list(cls.get(code).assumptions),
            "estimand": cls.get(code).estimand,
            "uncertainty_method": cls.get(code).uncertainty_method,
            "verification_method": cls.get(code).verification_method,
            "executor_id": cls.get(code).executor_id,
            "preflight": preflight,
        }


@dataclass
class MethodSelectionDecision:
    problem_class: ProblemClass
    objective: ObjectiveType
    estimand: EstimandSpec
    admissible_families: Set[MethodFamily]
    max_verdict_tier: Optional[str]
    causal_eligible: bool = False
    causal_intent: Optional[CausalIntent] = None
    fallback_used: bool = False
    rationale: str = ""
    selected_method_code: Optional[str] = None
    method_selection_scores: Dict[str, float] = field(default_factory=dict)
    method_selection_reasons: Dict[str, List[str]] = field(default_factory=dict)
    # v19: the raw UniversalQuestionCompiler task this decision was reconciled
    # against, and the CANONICAL_TASK_AUTHORITY status that reconciliation
    # produced. canonical_task is set by select_for_plan() unconditionally
    # (even on exception paths that never reach candidate scoring) so any
    # consumer can assert it against analysis_plan.task and detect drift.
    canonical_task: Optional[str] = None
    analytical_task_status: str = "UNKNOWN"

    @property
    def method_family(self) -> MethodFamily:
        """Backward-compatible single-family accessor for current synthesizers."""
        if len(self.admissible_families) == 1:
            return next(iter(self.admissible_families))
        return MethodFamily.DIAGNOSTIC_BATTERY

    @property
    def claim_type(self) -> EpistemicClaimType:
        """Default ledger claim type for the investigation class."""
        if self.problem_class == ProblemClass.CORRELATIONAL:
            return EpistemicClaimType.ASSOCIATION
        if self.problem_class == ProblemClass.FORECASTING:
            return EpistemicClaimType.PREDICTION
        if self.problem_class == ProblemClass.DESCRIPTIVE:
            return EpistemicClaimType.OBSERVATION
        return EpistemicClaimType.ASSOCIATION

    def claim_type_for(self, is_grouped: bool, aggregation_type: str = "") -> EpistemicClaimType:
        if self.problem_class == ProblemClass.CORRELATIONAL:
            return EpistemicClaimType.ASSOCIATION
        if self.problem_class == ProblemClass.FORECASTING or aggregation_type.upper() == "REGRESSION":
            return EpistemicClaimType.PREDICTION
        if self.problem_class == ProblemClass.SURVIVAL_CHURN:
            return EpistemicClaimType.ASSOCIATION
        return EpistemicClaimType.ASSOCIATION if is_grouped else EpistemicClaimType.OBSERVATION

    def to_provenance_dict(self) -> Dict[str, Any]:
        return {
            "engine": "MethodSelectionEngine",
            "problem_class": self.problem_class.value,
            "objective": self.objective.value,
            "admissible_families": sorted(f.value for f in self.admissible_families),
            "method_family": self.method_family.value,
            "max_verdict_tier": self.max_verdict_tier,
            "claim_type": self.claim_type.value,
            "causal_eligible": self.causal_eligible,
            "causal_intent_present": self.causal_intent is not None,
            "fallback_used": self.fallback_used,
            "rationale": self.rationale,
            "selected_method_code": self.selected_method_code,
            "method_selection_scores": dict(self.method_selection_scores),
            "method_selection_reasons": {k: list(v) for k, v in self.method_selection_reasons.items()},
            "estimand": self.estimand.to_provenance_dict(),
            "canonical_task": self.canonical_task,
            "analytical_task_status": self.analytical_task_status,
        }


def gate_method_for_plan(plan: Any, semantic: Any, df: Any) -> Optional[str]:
    task = str(getattr(plan, "task", "") or "").upper()
    if task == "ASSOCIATION":
        target = getattr(getattr(plan, "semantics", None), "target_column", None)
        expl = list(getattr(getattr(plan, "semantics", None), "explanatory_columns", []) or [])
        if target and target in df.columns:
            try:
                if int(df[target].dropna().nunique()) == 2 and expl:
                    return "association_categorical_binary"
                # v20-C4.2.2c: this branch used to check only
                # df[target].dtype (the outcome), never df[c].dtype for
                # c in expl (the exposure/predictor columns) -- so any
                # non-empty `expl` with a numeric target returned
                # "association_numeric" regardless of whether the named
                # predictor was actually numeric (e.g. a string ID
                # column). admissibility_for_plan()'s own equivalent
                # branch already correctly required BOTH sides to be
                # numeric; this now matches it, so the two functions
                # cannot disagree about what "numeric" requires. See
                # AAOS_V20C4_2_2B_COMPILER_SEMANTIC_AUTHORITY_AUDIT.md,
                # Finding 3.
                numeric_expl = [
                    c for c in expl
                    if c in df.columns and str(df[c].dtype).lower().startswith(("int", "float"))
                ]
                if (
                    numeric_expl
                    and str(df[target].dtype).lower().startswith(("int", "float"))
                ):
                    return "association_numeric"
            except Exception:
                pass
    mapping = {"FORECAST": "forecast_rolling_origin", "PREDICTION": "binary_risk_prediction", "SEGMENTATION": "stable_segmentation", "RECONCILIATION": "row_reconciliation", "COMPARISON": "comparison_group_effect"}
    return mapping.get(task)

def roles_for_plan(plan: Any, semantic: Any, df: Any) -> Set[str]:
    roles: Set[str] = set()
    sem = getattr(plan, "semantics", None)
    target = getattr(sem, "target_column", None)
    expl = list(getattr(sem, "explanatory_columns", []) or [])
    groups = list(getattr(sem, "grouping_columns", []) or [])
    if expl: roles.add("exposure")
    if expl and all(
        c in getattr(df, "columns", []) and str(df[c].dtype).lower().startswith(("int", "float"))
        for c in expl
    ):
        # v20-audit fix (DEFECT-022): association_multivariate_ols's own
        # required_roles previously accepted the generic "exposure" role
        # (granted whenever explanatory_columns is non-empty, regardless
        # of dtype) as sufficient -- so a string predictor (e.g. a
        # customer_id explicitly named in the question) reached this
        # candidate's generic MethodRegistry.validate() path undetected,
        # since gate_method_for_plan() never returns
        # "association_multivariate_ols" and so never routes it through
        # admissibility_for_plan()'s own dtype check either. Confirmed via
        # a live regression: "Is customer_id associated with annual_sales?"
        # selected association_multivariate_ols despite customer_id being
        # a string column. This role is granted only when every requested
        # explanatory column is numeric, matching association_numeric's
        # existing "numeric_exposure" convention and this capability's own
        # declared failure_conditions ("non_numeric predictor/outcome").
        roles.add("all_numeric_exposure")
    if groups: roles.add("grouping")
    if target and target in df.columns:
        roles.add("outcome")
        try:
            if int(df[target].dropna().nunique()) == 2:
                roles.update({"binary_outcome", "binary_target"})
        except Exception:
            pass
        if str(df[target].dtype).lower().startswith(("int", "float")):
            roles.update({"numeric_outcome", "outcome"})
    if getattr(sem, "time_column", None) in getattr(df, "columns", []): roles.add("time")
    task = str(getattr(plan, "task", "") or "").upper()
    if task == "FORECAST": roles.add("target")
    if task == "SEGMENTATION": roles.update({"entity_rows", "features"})
    if task == "RECONCILIATION": roles.add("numeric_fields")
    return roles

def blocked_reasons_for_quality(quality: Any, task: str) -> List[str]:
    if quality is None: return []
    out = [str(x) for x in (getattr(quality, "critical_issues", []) or [])]
    if task == "PREDICTION" and getattr(quality, "leakage_indicators", None):
        out.append("prediction_leakage_indicators_present")
    return out


class MethodSelectionEngine:
    """Deterministic policy engine: question semantics -> analytical strategy."""
    @classmethod
    def select_for_plan(cls, decision: MethodSelectionDecision, plan: Any, semantic: Any, quality: Any, df: Any) -> MethodSelectionDecision:
        """Make the registry the executable method-selection authority.

        The legacy decision supplies problem/objective context; this method evaluates
        all registry capabilities compatible with the compiled plan and selects the
        highest-scoring *admissible* method using fit to the estimand, resolved roles,
        data readiness, uncertainty requirements and verification availability.
        """
        task = str(getattr(plan, "task", "") or "").upper()

        # v19 section 1: the compiled canonical analysis plan is the single
        # authoritative analytical classification. decision.canonical_task is
        # always set here, on every path (including the UNSUPPORTED/
        # SPECIALIST_BATTERY early-returns below), so a caller can assert it
        # against analysis_plan.task and catch any path that reaches
        # execution without going through this reconciliation.
        decision.canonical_task = task
        entry = CANONICAL_TASK_AUTHORITY.get(task)

        # v19 section 1 + section 11: two specializations must not be
        # silently overwritten in EITHER direction by the canonical task's
        # 1:1 ProblemClass mapping:
        #   - SURVIVAL_CHURN: decide() resolved a real churn outcome column
        #     that no compiler task vocabulary word ("CHURN") even exists
        #     for; the compiler task is necessarily "about" something else
        #     (PREDICTION/DIAGNOSTIC/COMPARISON) and must not clobber it.
        #   - CAUSAL: decide()'s _has_explicit_causal_effect_language check
        #     is deliberately narrower than the compiler's cruder
        #     "caus(e|ed|es|al)" stem match (which also fires on plain
        #     "root cause" wording). Guard BOTH directions: the compiler
        #     must not promote a non-causal decide() decision up to CAUSAL
        #     (would silently remove the overclaiming cap), and it must not
        #     demote a decide()-resolved CAUSAL decision away from it either.
        churn_specialization = decision.problem_class == ProblemClass.SURVIVAL_CHURN
        causal_disagreement = (
            (decision.problem_class == ProblemClass.CAUSAL) !=
            (entry is not None and entry.problem_class == ProblemClass.CAUSAL)
        )
        preserve_legacy_problem_class = churn_specialization or causal_disagreement

        if not preserve_legacy_problem_class and entry is not None:
            decision.problem_class = entry.problem_class

        if entry is None:
            # Outside the 13-task closed vocabulary UniversalQuestionCompiler
            # is defined to emit. Do not guess; surface it explicitly rather
            # than silently keeping whatever decide() produced.
            decision.analytical_task_status = "UNRECOGNIZED_TASK"
            decision.rationale += (
                f" UNRECOGNIZED_TASK: '{task}' is outside the canonical task vocabulary; "
                "no method-selection authority decision was made for it."
            )
            return decision

        if preserve_legacy_problem_class:
            decision.analytical_task_status = "SPECIALIST_BATTERY"
            decision.rationale += (
                f" Canonical task={task}; problem_class retained as "
                f"{decision.problem_class.value} because MethodSelectionEngine's own "
                "churn-outcome/causal-effect-language resolution is a strictly more precise "
                "refinement than the compiler's lexical task label for this concern "
                "(see AAOS_V19_CANONICAL_AUTHORITY.md)."
            )
            return decision

        if entry.status == TaskAuthorityStatus.UNSUPPORTED:
            decision.analytical_task_status = "UNSUPPORTED_ANALYTICAL_TASK"
            decision.admissible_families = set()
            decision.selected_method_code = None
            decision.fallback_used = True
            decision.rationale += (
                f" UNSUPPORTED_ANALYTICAL_TASK: '{task}' has no registered analytical method "
                "in this release; execution must not proceed on the legacy decision."
            )
            return decision

        if entry.status == TaskAuthorityStatus.SPECIALIST_BATTERY:
            decision.analytical_task_status = "SPECIALIST_BATTERY"
            decision.rationale += (
                f" Canonical task={task} routes to the specialist/diagnostic battery; "
                "no single registry method capability applies."
            )
            return decision

        # entry.status == IMPLEMENTED from here on.
        candidates = MethodRegistry.for_problem_class(task)
        if not candidates:
            # Registry/authority drift: CANONICAL_TASK_AUTHORITY promises a
            # registry method for this task but METHOD_REGISTRY has none.
            # Fail closed and say so explicitly rather than silently
            # trusting whatever decide() produced.
            decision.analytical_task_status = "UNSUPPORTED_ANALYTICAL_TASK"
            decision.admissible_families = set()
            decision.selected_method_code = None
            decision.fallback_used = True
            decision.rationale += (
                f" UNSUPPORTED_ANALYTICAL_TASK: '{task}' is declared IMPLEMENTED in "
                "CANONICAL_TASK_AUTHORITY but METHOD_REGISTRY has no matching capability "
                "(registry/authority drift; treat as unsupported rather than silently falling back)."
            )
            return decision
        decision.analytical_task_status = "IMPLEMENTED"

        scored: Dict[str, float] = {}
        reasons: Dict[str, List[str]] = {}
        for cap in candidates:
            # v20-C4.2.2c: gate_method_for_plan() is now dtype-symmetric
            # (checks exposure AND outcome numeric-ness, matching
            # admissibility_for_plan()'s own logic -- see that function's
            # branch below), so it can no longer suggest "association_numeric"
            # for a candidate whose predictor isn't actually numeric. As a
            # second, independent layer of defense against exactly this
            # class of bug recurring, admissibility_for_plan() is now also
            # method-code-aware: even if it is ever invoked for a
            # mismatched candidate again in the future, it reports that
            # specific method as BLOCKED rather than silently substituting
            # a different method's gate result. See
            # AAOS_V20C4_2_2B_COMPILER_SEMANTIC_AUTHORITY_AUDIT.md, Finding 3.
            gated_code = gate_method_for_plan(plan, semantic, df)
            if cap.code == gated_code:
                ok, gate = MethodRegistry.admissibility_for_plan(plan, semantic, quality, df, method_code=cap.code)
            else:
                ok, gate = MethodRegistry.validate(cap.code, row_count=len(df), roles=roles_for_plan(plan, semantic, df), blocked_reasons=blocked_reasons_for_quality(quality, task))
            score = 0.0
            why: List[str] = []
            if ok:
                score += 100.0
                why.append("admissible")
            else:
                score -= 100.0
                why.append("blocked:" + ";".join(gate.get("errors", []) if isinstance(gate, dict) else []))
            # Prefer direct estimand alignment over generic fallbacks.
            est = cap.estimand.lower()
            plan_claim = str(getattr(plan, "claim_type", "")).upper()
            if task == "ASSOCIATION" and "association" in est:
                score += 25.0; why.append("estimand_matches_association")
            if task == "ASSOCIATION" and getattr(decision.estimand, "joint_predictors", False):
                if cap.code == "association_multivariate_ols":
                    score += 60.0; why.append("explicit_joint_predictor_request")
                elif cap.code == "association_numeric":
                    score -= 60.0; why.append("pairwise_method_disfavored_for_joint_request")
            if task in {"FORECAST", "PREDICTION"} and plan_claim == "PREDICTION":
                score += 20.0; why.append("claim_matches_prediction")
            # Favor lower-cost, more independently verifiable methods.
            if cap.verification_method:
                score += 5.0; why.append("has_independent_verification")
            if cap.uncertainty_method:
                score += 5.0; why.append("has_declared_uncertainty")
            score -= min(float(cap.min_sample) / 1000.0, 5.0)
            scored[cap.code] = score
            reasons[cap.code] = why

        admissible = [(code, score) for code, score in scored.items() if score >= 0]
        if not admissible:
            decision.selected_method_code = None
            decision.method_selection_scores = scored
            decision.method_selection_reasons = reasons
            return decision

        selected = max(admissible, key=lambda item: (item[1], item[0]))[0]
        decision.selected_method_code = selected
        decision.method_selection_scores = scored
        decision.method_selection_reasons = reasons
        decision.rationale += f" Registry-selected method={selected}; selection is authoritative after admissibility and fit scoring and binds execution to {MethodRegistry.executor_id(selected)}."
        return decision

    @classmethod
    def canonical_plan_view(cls, decision: MethodSelectionDecision, plan: Any) -> Any:
        """v20-C4.2.3 reconciliation step: plan PROPOSAL -> canonical validation -> reconciled view.

        select_for_plan()/admissibility_for_plan() read ``plan.semantics``.  The plan
        is only a proposal: canonical semantic resolution has already validated the
        requested predictors and may have DROPPED some (e.g. a categorical column is
        not a valid Pearson predictor -- recorded in estimand.limitations).  Without
        this step the dropped column still sits in ``plan.semantics`` and silently
        blocks a method the canonical estimand admits, so an invalid predictor would
        suppress the valid one.

        Canonical resolution may only NARROW a proposal, never introduce one.  This
        returns a COPY of the plan whose explanatory/grouping columns exclude the
        predictors canonical resolution rejected.  It is a no-op whenever the plan's
        predictors already agree with the estimand (or the estimand names none), and
        the original plan is left untouched so the proposal stays available for
        conflict detection and diagnostics.
        """
        est = getattr(decision, "estimand", None)
        sem = getattr(plan, "semantics", None)
        final_preds = list(getattr(est, "predictor_columns", None) or [])
        plan_preds = list(getattr(sem, "explanatory_columns", None) or [])
        if sem is None or not final_preds or not plan_preds:
            return plan
        if not set(final_preds).issubset(plan_preds):
            return plan  # canonical layer must not introduce predictors: leave the conflict visible
        dropped = [c for c in plan_preds if c not in set(final_preds)]
        if not dropped:
            return plan
        import copy as _copy
        view = _copy.deepcopy(plan)
        vs = view.semantics
        object.__setattr__(vs, "explanatory_columns", [c for c in plan_preds if c in set(final_preds)])
        grouping = list(getattr(vs, "grouping_columns", None) or [])
        if grouping:
            object.__setattr__(vs, "grouping_columns", [c for c in grouping if c not in set(dropped)])
        return view

    @classmethod
    def detect_canonical_plan_role_disagreement(cls, decision: MethodSelectionDecision, plan: Any) -> List[str]:
        """v20-C4.2 (detection increment only -- see AAOS_V20C4_2_CONTRACT_AUTHORITY_AUDIT.md).

        select_for_plan()'s own admissibility/scoring helpers
        (admissibility_for_plan/gate_method_for_plan/roles_for_plan, all
        module-level above) derive target/explanatory/grouping/time roles
        from `plan.semantics` -- the UniversalQuestionCompiler's own
        binding object -- independently of `decision.estimand`, which is
        the canonical-resolution-derived object C3/C4/C4.1 hardened.
        These two objects currently agree in every case exercised by this
        repository's test suite (both ultimately trace back to the same
        underlying SemanticResolution, and there is still no independent
        second resolver capable of making them disagree -- see the "known
        limitation" already documented in AAOS_V20C3_CANONICAL_EXPERIMENT_CONTRACT.md
        and AAOS_V20C4_1_METHOD_SELECTION_CANONICAL_AUTHORITY.md), but
        nothing today would notice if that stopped being true.

        This function is intentionally detection-only: it does not change
        which method gets selected, does not block execution, and does not
        prefer one object over the other. It exists so that real production
        traffic (not just hand-built adversarial tests) can surface the
        first genuine disagreement, if and when one occurs, before the
        deeper reconciliation work (unifying plan.semantics and
        decision.estimand into one authority feeding admissibility/scoring)
        is attempted. Rushing that reconciliation without first knowing
        whether -- and how often -- real disagreement occurs would be
        exactly the kind of change this project's own practice (see the
        C3/C3.1/C4/C4.1 fixture-bug and routing-guard incidents) has
        repeatedly shown is risky to do blind.
        """
        conflicts: List[str] = []
        sem = getattr(plan, "semantics", None)
        est = decision.estimand
        plan_target = getattr(sem, "target_column", None)
        if plan_target and est.target_column and plan_target != est.target_column:
            conflicts.append(f"target_column: plan.semantics={plan_target!r} vs decision.estimand={est.target_column!r}")
        plan_expl = list(getattr(sem, "explanatory_columns", []) or [])
        if plan_expl and est.predictor_columns and set(plan_expl) != set(est.predictor_columns):
            conflicts.append(f"predictor_columns: plan.semantics={plan_expl!r} vs decision.estimand={est.predictor_columns!r}")
        plan_group = list(getattr(sem, "grouping_columns", []) or [])
        if plan_group and est.comparison_dimension and est.comparison_dimension not in plan_group:
            conflicts.append(f"comparison_dimension: plan.semantics={plan_group!r} vs decision.estimand={est.comparison_dimension!r}")
        plan_time = getattr(sem, "time_column", None)
        if plan_time and est.time_column and plan_time != est.time_column:
            conflicts.append(f"time_column: plan.semantics={plan_time!r} vs decision.estimand={est.time_column!r}")
        return conflicts

    @classmethod
    def bind_candidates(cls, candidates: List[Any], method_code: Optional[str]) -> List[Any]:
        """Attach the authoritative method code to every executable candidate."""
        if not method_code:
            return candidates
        for candidate in candidates:
            candidate.method_code = method_code
            candidate.executable_spec = dict(getattr(candidate, "executable_spec", {}) or {})
            candidate.executable_spec["method_code"] = method_code
            candidate.executable_spec["executor_id"] = MethodRegistry.executor_id(method_code)
            candidate.fingerprint = candidate.compute_fingerprint()
        return candidates

    @classmethod
    def assert_candidate_binding(cls, candidate: Any, method_code: Optional[str]) -> Tuple[bool, str]:
        if not method_code:
            return True, "no_registry_method_selected"
        declared = getattr(candidate, "method_code", None) or getattr(candidate, "executable_spec", {}).get("method_code")
        if declared != method_code:
            return False, f"candidate_method_mismatch:{declared!r}!={method_code!r}"
        expected_executor = MethodRegistry.executor_id(method_code)
        actual_executor = getattr(candidate, "executable_spec", {}).get("executor_id")
        if actual_executor != expected_executor:
            return False, f"candidate_executor_mismatch:{actual_executor!r}!={expected_executor!r}"
        return True, expected_executor



    @staticmethod
    def decide(
        intent: Any,
        semantic: Any,
        question: str,
        temporal_scope: Optional[TemporalScope] = None,
        investigation_id: Optional[str] = None,
        primary_df: Any = None,
        question_roles: Optional[QuestionRoleProposal] = None,
    ) -> MethodSelectionDecision:
        try:
            if semantic is None:
                return MethodSelectionEngine.fallback("semantic_resolution_missing")

            # v20-C4: build the canonical semantic resolution once, here, at
            # the method-selection entry boundary -- before any routing
            # decision is made -- and thread it through every branch below.
            # This mirrors predictive_hypothesis._build_canonical_semantics'
            # decision=None path exactly (same task-string derivation from
            # intent.intent_type), since no MethodSelectionDecision exists
            # yet at this point for this call to key off of.
            #
            # Guard: canonical is the authority from here on. If it cannot
            # be constructed, this does NOT fall back to reading
            # semantic.target_metric_col etc. directly and continuing --
            # that would silently reintroduce the second, ungoverned
            # resolution path C4 exists to remove. Instead this fails
            # closed via the module's existing fallback() path, exactly as
            # a missing `semantic` already does above.
            task_for_canonical = str(getattr(intent, "intent_type", "") or "")
            try:
                canonical = build_canonical_semantic_resolution(semantic, task_for_canonical, question_roles=question_roles)
            except Exception as canonical_exc:
                return MethodSelectionEngine.fallback(
                    f"canonical_construction_failed:{type(canonical_exc).__name__}:{canonical_exc}"
                )

            intent_type = str(getattr(intent, "intent_type", "GENERAL") or "GENERAL").upper()
            has_temporal = temporal_scope if getattr(temporal_scope, "found", False) else None

            # Explicit causal-effect language is stronger than the lexical
            # CORRELATION label emitted by IntentEngine for words such as
            # "effect". The policy layer is the authoritative post-semantic
            # classifier, so it must recognize explicit causal intent before
            # routing ordinary correlation/forecast families.
            if MethodSelectionEngine._has_explicit_causal_effect_language(question):
                return MethodSelectionEngine._select_causal_or_diagnostic(intent, semantic, has_temporal, question, canonical)

            if intent_type == "CHURN":
                return MethodSelectionEngine._select_churn(intent, semantic, has_temporal, question, primary_df, canonical)
            if intent_type == "CORRELATION":
                # v20-C4.2.1: a question that names a CATEGORICAL explanatory
                # variable ("does plan tier affect cancellation rate?") is not
                # a Pearson/Spearman question -- that test is undefined for a
                # categorical predictor. Previously this branch bound the
                # predictor to whatever numeric secondary metric SemanticEngine
                # happened to discover (e.g. tenure_months) and ran a
                # correlation on THAT, silently answering a different question.
                # Route it to the same family the equivalent CHURN/segment
                # phrasing already reaches, mirroring the DEFECT-018
                # precedent below. Deliberately narrow: only when the single
                # requested predictor is categorical AND is the resolved
                # semantic dimension, so question, dimension and estimand all
                # name the same column end to end.
                requested = canonical.requested_explanatory_columns()
                if (
                    len(requested) == 1
                    and canonical.requested_categorical_columns() == requested
                    and requested[0] == canonical.primary_dimension()
                ):
                    if canonical.is_churn_relevant() and canonical.outcome_column():
                        routed = MethodSelectionEngine._select_churn(intent, semantic, has_temporal, question, primary_df, canonical)
                    else:
                        routed = MethodSelectionEngine._select_comparative(intent, semantic, has_temporal, canonical)
                    return MethodSelectionEngine._bind_requested_predictor(routed, canonical)
                # v20-C4.2.1c: fail closed on any OTHER single explicit
                # requested predictor that is not valid for a numeric
                # correlation, rather than silently substituting a
                # different DISCOVERED numeric column for it (the gap left
                # open by C4.2.1b's own invalid-role test: "is customer_name
                # associated with annual_sales?" must not quietly become a
                # correlation between annual_sales and whatever numeric
                # column SemanticEngine happened to discover -- that answers
                # a different question than the one asked). Covers both (a)
                # categorical but not the resolved semantic dimension (the
                # branch above only reroutes when it IS the dimension), and
                # (b) not a recognized numeric column at all. Reuses the
                # existing diagnostic-reroute policy this function already
                # applies two lines below for "no resolved secondary metric"
                # -- no new policy invented.
                if requested:
                    numeric_pool = set(getattr(semantic, "available_numeric_cols", None) or [])
                    categorical_pool = set(canonical.available_categorical_candidates)
                    valid_numeric_candidates = [
                        c for c in requested
                        if c in numeric_pool and c not in categorical_pool and c != canonical.outcome_column()
                    ]
                    if not valid_numeric_candidates:
                        candidate_str = requested[0] if len(requested) == 1 else ", ".join(requested)
                        return MethodSelectionEngine._select_diagnostic(
                            intent, semantic, has_temporal,
                            f"requested explanatory column '{candidate_str}' is not a valid predictor for "
                            f"a numeric correlation and is not the resolved semantic dimension",
                            canonical,
                        )
                # v20-C4.1: gate on the canonical secondary-metric binding,
                # not semantic.secondary_metric_col directly -- otherwise a
                # stale/cleared semantic field could suppress a perfectly
                # valid canonical decision even though _select_correlational
                # itself already reads canonical.secondary_metric_column().
                if canonical.secondary_metric_column():
                    routed = MethodSelectionEngine._select_correlational(intent, semantic, has_temporal, canonical)
                    routed.estimand.joint_predictors = bool(
                        len(routed.estimand.predictor_columns) > 1
                        and re.search(r"\b(?:controlling for|after accounting for|holding\s+[^?]*\s+constant|adjusted for|net of)\b", question, re.I)
                    )
                    if routed.estimand.joint_predictors:
                        routed.rationale += " Explicit control wording requests one joint conditional association model rather than independent pairwise correlations."
                    return routed
                return MethodSelectionEngine._select_diagnostic(intent, semantic, has_temporal, "correlation intent lacks a resolved secondary metric", canonical)
            if intent_type == "FORECAST":
                # v20-C4.1: same rationale as CORRELATION above, for time_column.
                if canonical.time_variable():
                    return MethodSelectionEngine._select_forecast(intent, semantic, has_temporal, canonical)
                return MethodSelectionEngine._select_diagnostic(intent, semantic, has_temporal, "forecast intent lacks a resolved time column", canonical)
            if intent_type in ("SEGMENTATION", "PERFORMANCE"):
                return MethodSelectionEngine._select_comparative(intent, semantic, has_temporal, canonical)
            if intent_type == "ROOT_CAUSE":
                # DEFECT-018: "why"/root-cause questions were always routed to
                # the generic diagnostic battery, purely by intent_type,
                # regardless of whether the semantic layer had positively
                # resolved a binary churn outcome as the target metric. A
                # question like "Why are customers churning?" is, in
                # substance, exactly the segment-difference-in-churn-rate
                # question the churn identifiability workflow already
                # answers -- but it was never reached, because CHURN-outcome
                # routing here was keyed solely off intent_type == "CHURN"
                # (which "why"-phrased questions never classify as; "why"
                # takes ROOT_CAUSE precedence in IntentEngine by design).
                # Mirrors the same already-resolved-secondary-metric-takes-
                # priority pattern used for CORRELATION above.
                #
                # v20-C4.1: gated on canonical.is_churn_relevant() AND a
                # resolved outcome_column(), not
                # `getattr(semantic, "churn_event_col", None)` directly, and
                # deliberately NOT on `canonical.outcome_column()` alone --
                # `outcome` is the same overloaded field an ordinary
                # resolved-target ROOT_CAUSE question also populates (e.g.
                # "why did revenue fall?"), so checking only
                # `outcome_column()` would have misrouted every ordinary
                # root-cause question with a resolved target straight into
                # the churn workflow. is_churn_relevant() checks the
                # field's provenance instead of guessing from its shape.
                if canonical.is_churn_relevant() and canonical.outcome_column():
                    return MethodSelectionEngine._select_churn(intent, semantic, has_temporal, question, primary_df, canonical)
                return MethodSelectionEngine._select_diagnostic(intent, semantic, has_temporal, "root-cause wording is diagnostic, not automatically causal", canonical)
            return MethodSelectionEngine._select_descriptive(intent, semantic, has_temporal, canonical)
        except Exception as exc:
            return MethodSelectionEngine.fallback(f"exception:{type(exc).__name__}:{exc}")

    @staticmethod
    def fallback(reason: str) -> MethodSelectionDecision:
        est = EstimandSpec(limitations=[f"fallback:{reason}"])
        return MethodSelectionDecision(
            problem_class=ProblemClass.FALLBACK,
            objective=ObjectiveType.DESCRIBE,
            estimand=est,
            admissible_families={
                MethodFamily.DIAGNOSTIC_BATTERY,
                MethodFamily.CORRELATION,
                MethodFamily.FORECAST,
                MethodFamily.CHURN,
            },
            max_verdict_tier=None,
            fallback_used=True,
            rationale=f"FALLBACK: {reason}; preserve legacy routing without inventing analytical semantics.",
        )

    @staticmethod
    def _metric_ref(semantic: Any, canonical: Optional[CanonicalSemanticResolution] = None) -> MetricRef:
        # v20-C4.1: previously read semantic.target_metric_col directly and
        # unconditionally, independent of whatever canonical.outcome_column()
        # said -- a second, ungoverned read of the exact same role identity
        # already threaded through _estimand as target_column. That made it
        # possible for EstimandSpec.target_column and
        # EstimandSpec.metric_ref.column to disagree whenever a canonical
        # resolution overrode the raw semantic field (see the same class of
        # bug _select_correlational/_select_forecast/_select_churn were
        # already fixed for). Now sourced from the same
        # canonical.outcome_column() call _estimand uses for target_column,
        # falling back to raw semantic only when no canonical is supplied
        # (the same pre-C4 direct-call compatibility path _estimand itself
        # documents).
        col = (
            canonical.outcome_column() if canonical is not None
            else getattr(semantic, "target_metric_col", None)
        )
        md = getattr(semantic, "metric_definition", None)
        agg = getattr(md, "aggregation_type", AggregationType.SUM)
        return MetricRef(name=col, table=semantic.primary_dataset_name, column=col, aggregation=agg)

    @staticmethod
    def _unit(semantic: Any, canonical: Optional[CanonicalSemanticResolution] = None) -> GrainRef:
        # v20-C4.1: same rationale as _metric_ref above -- the default unit
        # key must reference the same canonical column identity, not a
        # second, independent read of semantic.target_metric_col that could
        # disagree with it. world_model.verified_grains (an explicit,
        # independently-sourced grain declaration) still takes precedence
        # over either when present, unchanged from before.
        default_key = (
            canonical.outcome_column() if canonical is not None
            else getattr(semantic, "target_metric_col", None)
        )
        keys = [default_key]
        wm = getattr(semantic, "world_model", None)
        if wm is not None and hasattr(wm, "verified_grains"):
            try:
                keys = list(wm.verified_grains.get(semantic.primary_dataset_name, keys) or keys)
            except Exception:
                pass
        return GrainRef(table=semantic.primary_dataset_name, keys=keys)

    @staticmethod
    def _estimand(
        semantic: Any,
        temporal_scope: Optional[TemporalScope],
        limitations: Optional[List[str]] = None,
        canonical: Optional[CanonicalSemanticResolution] = None,
    ) -> EstimandSpec:
        md = getattr(semantic, "metric_definition", None)
        # v20-C4: when a CanonicalSemanticResolution is supplied (decide()
        # always supplies one -- see the guard there), it is the sole
        # source for these three role fields. This is not read alongside
        # semantic.target_metric_col / semantic.group_dimension_col /
        # semantic.time_col as a fallback; those raw fields are only
        # consulted when this method is called directly with no canonical
        # (existing unit tests exercising _estimand's other fields in
        # isolation), which is a distinct, pre-C4 code path, not a
        # fallback within the authoritative one.
        if canonical is not None:
            target_column = canonical.outcome_column()
            comparison_dimension = canonical.primary_dimension()
            time_column = canonical.time_variable()
        else:
            target_column = getattr(semantic, "target_metric_col", None)
            comparison_dimension = getattr(semantic, "group_dimension_col", None)
            time_column = getattr(semantic, "time_col", None)
        est = EstimandSpec(
            metric_ref=MethodSelectionEngine._metric_ref(semantic, canonical=canonical),
            metric_definition=md,
            unit_of_analysis=MethodSelectionEngine._unit(semantic, canonical=canonical),
            temporal_scope=temporal_scope,
            numerator_column=getattr(md, "numerator_column", None) if md else None,
            denominator_column=getattr(md, "denominator_column", None) if md else None,
            weight_column=getattr(md, "weight_column", None) if md else None,
            is_additive=getattr(md, "is_additive", None) if md else None,
            is_compositional=bool(getattr(md, "is_compositional", False)) if md else False,
            aggregation=getattr(md, "aggregation_type", None) if md else AggregationType.SUM,
            comparison_dimension=comparison_dimension,
            time_column=time_column,
            # v20-C3: bind the generic target role here too, so every
            # problem class (not just CORRELATIONAL/FORECASTING) exposes a
            # canonical target_column consumers can read instead of
            # semantic.target_metric_col directly.
            # v20-C4: now sourced from canonical.outcome_column() (see
            # above) rather than semantic.target_metric_col whenever a
            # canonical resolution is available.
            target_column=target_column,
            limitations=list(limitations or []),
        )
        if not est.limitations:
            est.limitations.append("population_filters_unavailable: no canonical filter parser is wired in this path")
        else:
            est.limitations.append("population_filters_unavailable: no canonical filter parser is wired in this path")
        return est

    @staticmethod
    def _select_descriptive(intent, semantic, temporal_scope, canonical=None):
        return MethodSelectionEngine._finalize(MethodSelectionDecision(
            problem_class=ProblemClass.DESCRIPTIVE,
            objective=ObjectiveType.DESCRIBE,
            estimand=MethodSelectionEngine._estimand(semantic, temporal_scope, canonical=canonical),
            admissible_families={MethodFamily.DIAGNOSTIC_BATTERY},
            max_verdict_tier=None,
            rationale="General analytical question routed to the descriptive/diagnostic battery.",
        ), intent, semantic)

    @staticmethod
    def _select_diagnostic(intent, semantic, temporal_scope, reason, canonical=None):
        return MethodSelectionEngine._finalize(MethodSelectionDecision(
            problem_class=ProblemClass.DIAGNOSTIC,
            objective=ObjectiveType.ROOT_CAUSE,
            estimand=MethodSelectionEngine._estimand(semantic, temporal_scope, [reason], canonical=canonical),
            admissible_families={MethodFamily.DIAGNOSTIC_BATTERY},
            max_verdict_tier=None,
            rationale=f"Diagnostic routing: {reason}.",
        ), intent, semantic)

    @staticmethod
    def _has_explicit_causal_effect_language(question: str) -> bool:
        q = (question or "").lower()
        return any(p.search(q) for p in _CAUSAL_EFFECT_PATTERNS)

    @staticmethod
    def _select_causal_or_diagnostic(intent, semantic, temporal_scope, question, canonical=None):
        table = getattr(semantic, "primary_dataset_name", None)
        # v20-C4.1: same role-identity fix as the correlation/forecast/churn
        # branches -- treatment/outcome for the causal gate must come from
        # the same canonical.primary_dimension()/outcome_column() calls
        # _estimand() below already uses, not a second, independent read of
        # semantic.group_dimension_col/target_metric_col that could
        # disagree with it. This does not touch the causal gate's own
        # logic or statistical treatment (still observational-only, still
        # NO_DAG_ASSUMED) -- only which object supplies the column names.
        treatment = (
            canonical.primary_dimension() if canonical is not None
            else getattr(semantic, "group_dimension_col", None)
        )
        outcome = (
            canonical.outcome_column() if canonical is not None
            else getattr(semantic, "target_metric_col", None)
        )
        if not table or not treatment or not outcome or treatment == outcome:
            return MethodSelectionEngine._select_diagnostic(
                intent, semantic, temporal_scope,
                "explicit causal-effect language but treatment/outcome could not be resolved",
                canonical,
            )

        causal_intent = CausalIntent(
            treatment=VariableRef(table=table, column=treatment),
            outcome=VariableRef(table=table, column=outcome),
            assumed_dag_id="NO_DAG_ASSUMED_OBSERVATIONAL_ONLY",
        )
        est = MethodSelectionEngine._estimand(
            semantic,
            temporal_scope,
            ["causal_dag_absent: controller has no identified causal graph; causal gate must remain observational-only"],
            canonical=canonical,
        )
        est.treatment_variable = causal_intent.treatment
        est.outcome_variable = causal_intent.outcome
        return MethodSelectionEngine._finalize(MethodSelectionDecision(
            problem_class=ProblemClass.CAUSAL,
            objective=ObjectiveType.ESTIMATE_CAUSAL_EFFECT,
            estimand=est,
            admissible_families={MethodFamily.DIAGNOSTIC_BATTERY},
            max_verdict_tier=VERDICT_TIER_STATISTICALLY_SIGNIFICANT,
            causal_eligible=True,
            causal_intent=causal_intent,
            rationale=("Explicit causal-effect request. Schema-valid causal intent constructed; "
                       "no DAG is invented, so the gate remains observational-only and the verdict is capped."),
        ), intent, semantic)

    @staticmethod
    def _bind_requested_predictor(decision: "MethodSelectionDecision", canonical: CanonicalSemanticResolution) -> "MethodSelectionDecision":
        """v20-C4.2.1: record the question's own explanatory variable as the
        estimand's predictor role, and keep the discovered secondary metric
        visible but non-authoritative (discovered_columns). Only sets fields
        on the estimand; never changes the selected family or claim ceiling."""
        est = decision.estimand
        requested = canonical.requested_explanatory_columns()
        est.predictor_columns = list(requested)
        discovered = canonical.secondary_metric_column()
        if discovered and discovered not in requested:
            est.discovered_columns = [discovered]
        return decision

    @staticmethod
    def _select_correlational(intent, semantic, temporal_scope, canonical=None):
        # v20-C3: this is the single place that binds the CORRELATIONAL
        # target/predictor roles. v20-C4: both roles are now sourced from
        # the canonical semantic resolution (canonical.outcome_column() via
        # _estimand, canonical.secondary_metric_column() here) rather than
        # re-reading semantic.secondary_metric_col directly -- canonical's
        # secondary_metric field is built with candidate_pool=None, so its
        # RESOLVED/UNRESOLVED truthiness is guaranteed identical to
        # semantic.secondary_metric_col's; decide() only reaches this
        # method once that's already confirmed non-empty, so both roles are
        # guaranteed populated here either way.
        # ExperimentSynthesizer consumes est.target_column/predictor_columns
        # rather than re-deriving them from semantic itself.
        est = MethodSelectionEngine._estimand(semantic, temporal_scope, canonical=canonical)
        secondary = (
            canonical.secondary_metric_column() if canonical is not None
            else getattr(semantic, "secondary_metric_col", None)
        )
        est.predictor_columns = [secondary] if secondary else []
        # v20-C4.2.1b: a question-explicit NUMERIC predictor is authoritative
        # over SemanticEngine's discovered secondary metric, mirroring the
        # categorical case _bind_requested_predictor already handles for the
        # rerouted branch above. Gated narrowly (explicit role > discovered
        # role ONLY AFTER semantic validity is established -- see
        # test_c4_2_1b): exactly one requested explanatory column, present
        # in the dataset's numeric pool, not the categorical pool, and not
        # the target/outcome column itself. An invalid explicit role (e.g. a
        # non-numeric column) is NOT trusted here -- it simply falls through
        # to the discovered-secondary-metric behavior above, unchanged; no
        # new fail-closed/reroute policy is invented by this branch.
        if canonical is not None:
            requested = canonical.requested_explanatory_columns()
            if len(requested) == 1:
                candidate = requested[0]
                numeric_pool = set(getattr(semantic, "available_numeric_cols", None) or [])
                categorical_pool = set(canonical.available_categorical_candidates)
                is_valid_numeric = (
                    candidate in numeric_pool
                    and candidate not in categorical_pool
                    and candidate != est.target_column
                )
                if is_valid_numeric and candidate != secondary:
                    est.predictor_columns = [candidate]
                    est.discovered_columns = [secondary] if secondary else []
            elif len(requested) > 1:
                numeric_pool = set(getattr(semantic, "available_numeric_cols", None) or [])
                categorical_pool = set(canonical.available_categorical_candidates)
                valid_numeric = []
                invalid = []
                for candidate in requested:
                    is_valid_cand = (
                        candidate in numeric_pool
                        and candidate not in categorical_pool
                        and candidate != est.target_column
                    )
                    if is_valid_cand:
                        if candidate not in valid_numeric:
                            valid_numeric.append(candidate)
                    else:
                        invalid.append(candidate)
                if valid_numeric:
                    est.predictor_columns = valid_numeric
                    if secondary and secondary not in valid_numeric:
                        est.discovered_columns = [secondary]
                else:
                    est.predictor_columns = []
                if invalid:
                    est.limitations.append(
                        f"Requested explanatory column(s) not valid for numeric association: {', '.join(invalid)}"
                    )
        return MethodSelectionEngine._finalize(MethodSelectionDecision(
            problem_class=ProblemClass.CORRELATIONAL,
            objective=ObjectiveType.COMPARE,
            estimand=est,
            admissible_families={MethodFamily.CORRELATION},
            max_verdict_tier=VERDICT_TIER_STATISTICALLY_SIGNIFICANT,
            rationale="Association analysis; claim ceiling prevents diagnostic/causal overclaiming. Joint control wording selects a multivariable conditional association estimand when requested.",
        ), intent, semantic)

    @staticmethod
    def _select_forecast(intent, semantic, temporal_scope, canonical=None):
        est = MethodSelectionEngine._estimand(semantic, temporal_scope, canonical=canonical)
        return MethodSelectionEngine._finalize(MethodSelectionDecision(
            problem_class=ProblemClass.FORECASTING,
            objective=ObjectiveType.FORECAST,
            estimand=est,
            admissible_families={MethodFamily.FORECAST},
            max_verdict_tier=VERDICT_TIER_STATISTICALLY_SIGNIFICANT,
            rationale="Temporal forecast; claim is predictive rather than diagnostic or causal.",
        ), intent, semantic)

    @staticmethod
    def _select_comparative(intent, semantic, temporal_scope, canonical=None):
        return MethodSelectionEngine._finalize(MethodSelectionDecision(
            problem_class=ProblemClass.COMPARATIVE,
            objective=ObjectiveType.COMPARE,
            estimand=MethodSelectionEngine._estimand(semantic, temporal_scope, canonical=canonical),
            admissible_families={MethodFamily.DIAGNOSTIC_BATTERY},
            max_verdict_tier=None,
            rationale="Segment/performance contrast routed to the comparison battery.",
        ), intent, semantic)

    @staticmethod
    def _select_churn(intent, semantic, temporal_scope, question, primary_df, canonical=None):
        # v20-C4.1: prefer the canonical outcome binding for the churn
        # event column itself, not just for exposure/censored/confounder
        # (which v20-C4 already migrated) -- but only when canonical
        # actually resolved this field via the churn-event path
        # (is_churn_relevant()). A canonical built for an ordinary
        # resolved-target question would have outcome_column() pointing at
        # target_metric_col, not a churn event, so blindly preferring it
        # here regardless of relevance would misroute the continuous-
        # metric-named / churn-absent branches below. When canonical is
        # not churn-relevant (or not supplied at all -- legacy no-canonical
        # callers), this falls back to semantic.churn_event_col exactly as
        # before.
        event_col = (
            canonical.outcome_column() if canonical is not None and canonical.is_churn_relevant()
            else getattr(semantic, "churn_event_col", None)
        )
        md = getattr(semantic, "metric_definition", None)
        target_metric = getattr(semantic, "target_metric_col", "") or ""
        continuous_metric_named = bool(
            target_metric
            and re.search(r"\b" + re.escape(target_metric.lower()) + r"\b", (question or "").lower())
            and md is not None
            and getattr(md, "is_additive", False)
            and getattr(md, "aggregation_type", None) == AggregationType.SUM
        )
        if event_col:
            est = MethodSelectionEngine._estimand(
                semantic, temporal_scope,
                [f"churn_event_column:{event_col}"],
                canonical=canonical,
            )
            # v20-C4: churn_bindings now sourced from the canonical
            # exposure/censored/confounder fields rather than re-reading
            # semantic.churn_*_col directly. event_col itself is left as
            # the routing variable above (not re-read from canonical)
            # because the `if event_col:` branch condition above already
            # is the routing decision -- canonical.outcome_column() is
            # guaranteed equal to it here (semantic_resolution_builder
            # resolves the canonical outcome field from this exact column
            # whenever churn_event_col is set), so re-reading it from
            # canonical would be the same value under a different name,
            # not a second authority.
            est.churn_bindings = {
                "event_col": event_col,
                "exposure_col": (
                    canonical.exposure_column() if canonical is not None
                    else getattr(semantic, "churn_exposure_col", None)
                ),
                "censored_col": (
                    canonical.censored_column() if canonical is not None
                    else getattr(semantic, "churn_censored_col", None)
                ),
                "confounder_cols": (
                    list(canonical.confounder_set().columns) if canonical is not None
                    else list(getattr(semantic, "churn_confounder_cols", []) or [])
                ),
                "outcome_available": getattr(semantic, "churn_outcome_available", False),
            }
            return MethodSelectionEngine._finalize(MethodSelectionDecision(
                problem_class=ProblemClass.SURVIVAL_CHURN,
                objective=ObjectiveType.PREDICT,
                estimand=est,
                admissible_families={MethodFamily.CHURN},
                max_verdict_tier=None,
                rationale="Resolved binary churn outcome; route to the existing churn identifiability workflow.",
            ), intent, semantic)

        # DEFECT-016: a metric explicitly named in the question and semantically resolved
        # as an additive magnitude is not silently trapped in the binary churn path.
        if continuous_metric_named:
            return MethodSelectionEngine._finalize(MethodSelectionDecision(
                problem_class=ProblemClass.DIAGNOSTIC,
                objective=ObjectiveType.ROOT_CAUSE,
                estimand=MethodSelectionEngine._estimand(semantic, temporal_scope, ["defect_016_guard"], canonical=canonical),
                admissible_families={MethodFamily.DIAGNOSTIC_BATTERY},
                max_verdict_tier=None,
                rationale="Churn vocabulary present, but the resolved target is a continuous additive metric; route diagnostically.",
            ), intent, semantic)

        return MethodSelectionEngine._finalize(MethodSelectionDecision(
            problem_class=ProblemClass.SURVIVAL_CHURN,
            objective=ObjectiveType.PREDICT,
            estimand=MethodSelectionEngine._estimand(semantic, temporal_scope, ["churn_outcome_absent"], canonical=canonical),
            admissible_families={MethodFamily.CHURN},
            max_verdict_tier=None,
            rationale="Churn question without resolved event column; retain churn workflow so existing unidentifiable path fails closed.",
        ), intent, semantic)

    @staticmethod
    def _finalize(decision, intent, semantic):
        if not decision.rationale:
            decision.rationale = "Method selected deterministically from structured intent and semantic context."
        decision.rationale += " Method capability registry is the authoritative declaration of prerequisites, assumptions, uncertainty, verification, and failure conditions."
        return decision
