"""ExperimentSynthesizer: Dynamic candidate experiment generation from predictive hypotheses,
unresolved predictions, and discovered evidence.
"""
from dataclasses import dataclass, field, asdict
import hashlib
from typing import Any, Dict, List, Optional
import pandas as pd

from packages.analytics_core.src.intelligence.predictive_hypothesis import PredictiveHypothesis
from packages.analytics_core.src.engines.semantic import SemanticResolution
from packages.analytics_core.src.intelligence.evidence_patterns import ACTIVE_DETECTORS, CandidateExplanation
from packages.analytics_core.src.semantic.metric_semantics import MetricDefinition, MetricSemanticsResolver
from packages.analytics_core.src.intelligence.likelihood_model import concentration_share_likelihoods, anova_eta_squared_likelihoods
from packages.schemas.src.analysis import AggregationType
from packages.schemas.src.semantic_role import ExperimentRole
from packages.analytics_core.src.relational.plan_serialization import plan_to_dict

# Named thresholds shared between hypothesis narrative text and the EIG
# likelihood model below, so the number a hypothesis claims and the number
# the experiment actually tests against can never silently drift apart.
CONCENTRATION_SHARE_THRESHOLD = 0.45
CONCENTRATION_EFFECT_IF_TRUE = 0.60
ISOLATION_SHARE_THRESHOLD = 0.50
ANOVA_ETA_SQUARED_THRESHOLD = 0.15


@dataclass
class UncertaintyState:
    """Explicit structural representation of what the investigation does and does not yet know."""
    unresolved_hypotheses: List[str] = field(default_factory=list)
    discriminating_variables: List[str] = field(default_factory=list)
    missing_evidence: List[str] = field(default_factory=list)
    conflicting_evidence: List[str] = field(default_factory=list)
    failed_predictions: List[str] = field(default_factory=list)
    assumption_failures: List[str] = field(default_factory=list)
    high_value_unknowns: List[str] = field(default_factory=list)
    decision_critical_unknowns: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unresolved_hypotheses": self.unresolved_hypotheses,
            "discriminating_variables": self.discriminating_variables,
            "missing_evidence": self.missing_evidence,
            "conflicting_evidence": self.conflicting_evidence,
            "failed_predictions": self.failed_predictions,
            "assumption_failures": self.assumption_failures,
            "high_value_unknowns": self.high_value_unknowns,
            "decision_critical_unknowns": self.decision_critical_unknowns,
        }


@dataclass
class CandidateExperiment:
    """Executable analytical experiment with explicit prediction linkages, utility priors, and multi-objective values."""
    code: str
    target_hypothesis_code: str
    tool_name: str
    query_sql: str
    description: str
    aggregation_type: str  # SUM, AVG, VARIANCE, ANOVA, CORRELATION
    discriminating_power: float = 0.0  # legacy ranking hint only; never converted into scientific likelihoods
    likelihood_if_true: Optional[float] = None
    likelihood_if_false: Optional[float] = None
    estimated_cost: float = 1.0        # 1.0 = standard
    reliability_weight: float = 0.95   # 0.0 .. 1.0
    decision_relevance: float = 1.0    # 0.0 .. 1.0
    target_dimension: Optional[str] = None
    target_metric: Optional[str] = None
    target_prediction_id: Optional[str] = None
    prediction_ids: List[str] = field(default_factory=list)
    hypothesis_ids: List[str] = field(default_factory=list)
    target_uncertainty: Optional[str] = None
    is_exploratory: bool = False
    metrics: List[str] = field(default_factory=list)
    dimensions: List[str] = field(default_factory=list)
    executable_spec: Dict[str, Any] = field(default_factory=dict)
    expected_outcomes: List[str] = field(default_factory=list)
    discrimination_value: float = 0.5
    adversarial_value: float = 0.0
    robustness_value: float = 0.0
    causal_value: float = 0.0
    selection_rationale: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)
    fingerprint: Optional[str] = None
    replication_of: Optional[str] = None
    refinement_of: Optional[str] = None
    # Explicit relational execution metadata. Empty for single-table experiments.
    # Any JOIN-bearing experiment must declare every hop so the controller can
    # independently verify cardinality before execution.
    join_hops: List[Dict[str, str]] = field(default_factory=list)
    # Canonical typed relational execution plan.
    # This is authoritative for relational experiments; query_sql is
    # a compiled representation and remains only for compatibility.
    relational_plan: Optional[Any] = None
    # Exact output column used as the experiment scalar primary value.
    primary_result_column: Optional[str] = None
    # Bound by MethodSelectionEngine before execution; None is permitted only for legacy/unregistered paths.
    method_code: Optional[str] = None
    # Scientific scope metadata: every experiment must make population coverage explicit.
    analysis_row_count: int = 0
    full_scope: bool = True
    sampling_policy: str = "NONE"
    sampling_reason: Optional[str] = None
    sampling_method: Optional[str] = None
    # v20-A: explicit contract binding + role tagging (see
    # packages/analytics_core/src/intelligence/experiment_contract_validation.py).
    # experiment_role defaults to UNASSIGNED (not PRIMARY) -- an experiment
    # that has not gone through explicit role tagging must not be silently
    # treated as the contract's primary experiment. The controller's hard
    # gate backfills a default role for legacy/untagged candidates
    # (see InvestigationController._infer_experiment_role) before validating,
    # rather than executing an UNASSIGNED experiment.
    experiment_role: str = "UNASSIGNED"
    contract_id: Optional[str] = None
    contract_version: Optional[int] = None
    problem_class: Optional[str] = None
    # Explicit outcome/predictor declaration for contract validation. When
    # left unset, validate_experiment_against_contract() derives these from
    # target_metric/metrics/dimensions for backward compatibility.
    outcome: Optional[str] = None
    predictors: List[str] = field(default_factory=list)
    # v20-C4.2.3: deterministic identity of the analytical pair this experiment
    # tests, and of the hypothesis it targets.  Both are stamped from the FINAL
    # contract (never derived from labels such as EXP-CORR-01 / HYP-01) and
    # deliberately excluded from compute_fingerprint() so existing experiment
    # de-duplication is unchanged.
    analytical_identity: str = ""
    hypothesis_analytical_identity: str = ""
    population: Optional[str] = None
    grain: Optional[str] = None
    time_scope: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.target_hypothesis_code and self.target_hypothesis_code not in self.hypothesis_ids:
            self.hypothesis_ids.append(self.target_hypothesis_code)
        if self.target_prediction_id and self.target_prediction_id not in self.prediction_ids:
            self.prediction_ids.append(self.target_prediction_id)
        if not self.prediction_ids and self.target_prediction_id:
            self.prediction_ids = [self.target_prediction_id]
        if self.prediction_ids and not self.target_prediction_id:
            self.target_prediction_id = self.prediction_ids[0]
        if self.target_metric and self.target_metric not in self.metrics:
            self.metrics.append(self.target_metric)
        if self.target_dimension and self.target_dimension not in self.dimensions:
            self.dimensions.append(self.target_dimension)
        if not self.executable_spec:
            self.executable_spec = {"sql": self.query_sql, "tool": self.tool_name, "aggregation": self.aggregation_type}
        if self.join_hops:
            self.executable_spec = dict(self.executable_spec)
            self.executable_spec["join_hops"] = [dict(h) for h in self.join_hops]
        if self.relational_plan is not None:
            self.executable_spec = dict(self.executable_spec)
            self.executable_spec["relational_plan"] = plan_to_dict(self.relational_plan)
        if not self.primary_result_column:
            agg = str(self.aggregation_type or "").upper()
            preferred = {
                "RATE": ["stratum_rate", "crude_rate", "persontime_rate", "rate", "total_metric", "mean_metric"],
                "COUNT": ["count_val", "count", "events", "eligible_n"],
                "VARIANCE": [self.target_metric or "", "variance", "variance_val", "mean_metric", "total_metric"],
                "CORRELATION": [self.target_metric or ""],
                "REGRESSION": [self.target_metric or ""],
                "MULTIVARIATE_REGRESSION": [self.target_metric or ""],
                "SUM": ["total_metric", self.target_metric or ""],
                "MEAN": ["mean_metric", "total_metric", self.target_metric or ""],
                "AVG": ["mean_metric", "total_metric", self.target_metric or ""],
            }.get(agg, ["total_metric", "mean_metric", self.target_metric or "", "metric", "value", "count_val"])
            sql_lower = f" {self.query_sql.lower()} "
            for alias in preferred:
                if alias and (f" as {alias.lower()}" in sql_lower or f" {alias.lower()} " in sql_lower):
                    self.primary_result_column = alias
                    break
        if self.primary_result_column:
            self.executable_spec = dict(self.executable_spec)
            self.executable_spec["primary_result_column"] = self.primary_result_column
        if not self.fingerprint:
            self.fingerprint = self.compute_fingerprint()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full executable contract, including method/executor binding."""
        return {
            "code": self.code,
            "target_hypothesis_code": self.target_hypothesis_code,
            "tool_name": self.tool_name,
            "query_sql": self.query_sql,
            "description": self.description,
            "aggregation_type": self.aggregation_type,
            "discriminating_power": self.discriminating_power,
            "likelihood_if_true": self.likelihood_if_true,
            "likelihood_if_false": self.likelihood_if_false,
            "estimated_cost": self.estimated_cost,
            "reliability_weight": self.reliability_weight,
            "decision_relevance": self.decision_relevance,
            "target_dimension": self.target_dimension,
            "target_metric": self.target_metric,
            "target_prediction_id": self.target_prediction_id,
            "prediction_ids": list(self.prediction_ids),
            "hypothesis_ids": list(self.hypothesis_ids),
            "target_uncertainty": self.target_uncertainty,
            "is_exploratory": self.is_exploratory,
            "metrics": list(self.metrics),
            "dimensions": list(self.dimensions),
            "executable_spec": dict(self.executable_spec),
            "expected_outcomes": list(self.expected_outcomes),
            "discrimination_value": self.discrimination_value,
            "adversarial_value": self.adversarial_value,
            "robustness_value": self.robustness_value,
            "causal_value": self.causal_value,
            "selection_rationale": self.selection_rationale,
            "provenance": dict(self.provenance),
            "fingerprint": self.fingerprint,
            "replication_of": self.replication_of,
            "refinement_of": self.refinement_of,
            "join_hops": [dict(h) for h in self.join_hops],
            "relational_plan": (
                plan_to_dict(self.relational_plan)
                if self.relational_plan is not None
                else None
            ),
            "primary_result_column": self.primary_result_column,
            "method_code": self.method_code,
            "analysis_row_count": int(self.analysis_row_count),
            "full_scope": bool(self.full_scope),
            "sampling_policy": self.sampling_policy,
            "sampling_reason": self.sampling_reason,
            "sampling_method": self.sampling_method,
            "experiment_role": self.experiment_role,
            "contract_id": self.contract_id,
            "contract_version": self.contract_version,
            "problem_class": self.problem_class,
            "outcome": self.outcome,
            "predictors": list(self.predictors),
            "analytical_identity": self.analytical_identity,
            "hypothesis_analytical_identity": self.hypothesis_analytical_identity,
            "population": self.population,
            "grain": self.grain,
            "time_scope": dict(self.time_scope) if self.time_scope else None,
        }

    def compute_fingerprint(self) -> str:
        pred_part = ",".join(sorted(self.prediction_ids or ([self.target_prediction_id] if self.target_prediction_id else [])))
        dim_part = ",".join(sorted(self.dimensions or ([self.target_dimension] if self.target_dimension else [])))
        metric_part = ",".join(sorted(self.metrics or ([self.target_metric] if self.target_metric else [])))
        join_part = repr(
            sorted(
                self.join_hops,
                key=lambda h: repr(sorted(h.items())),
            )
        )
        relational_plan_part = (
            repr(asdict(self.relational_plan))
            if self.relational_plan is not None
            else ""
        )
        payload = (
            f"{metric_part}:"
            f"{dim_part}:"
            f"{self.tool_name}:"
            f"{self.aggregation_type}:"
            f"{self.query_sql.strip()}:"
            f"{pred_part}:"
            f"{join_part}:"
            f"{relational_plan_part}:"
            f"{self.primary_result_column or ''}:"
            f"{self.method_code or ''}"
        )
        return hashlib.md5(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class DynamicReplanResult:
    """Result of adaptive evidence-driven candidate replanning."""
    candidates: List[CandidateExperiment] = field(default_factory=list)
    new_hypotheses: List[PredictiveHypothesis] = field(default_factory=list)
    new_predictions: List[Any] = field(default_factory=list)
    trigger_reason: str = ""

    def __iter__(self):
        return iter(self.candidates)

    def __getitem__(self, idx):
        return self.candidates[idx]

    def __len__(self):
        return len(self.candidates)


def _inject_time_filter(sql: str, time_filter_sql: Optional[str]) -> str:
    """Combine a resolved temporal scope into a generated `data_table` query.

    Closes DEFECT-002: every experiment generated here previously queried
    `data_table` unconditionally across its entire history, so a question
    like "why did revenue fall in March" was answered from an aggregate
    spanning every month in the dataset, diluting a real, sharp, single-
    month anomaly into noise. When the question's temporal language has
    been resolved (see TemporalResolver), every experiment testing that
    question is scoped to the same period -- not just the primary
    concentration test -- since a confounding or uniformity check run over
    the wrong time window is just as misleading as an unscoped one.
    """
    if not time_filter_sql:
        return sql
    if " WHERE " in sql:
        return sql.replace(" WHERE ", f" WHERE ({time_filter_sql}) AND ", 1)
    if " GROUP BY " in sql:
        return sql.replace(" GROUP BY ", f" WHERE ({time_filter_sql}) GROUP BY ", 1)
    return f"{sql} WHERE ({time_filter_sql})"


class ExperimentSynthesizer:
    """Generates an exhaustive candidate pool of discriminating analytical experiments."""

    @staticmethod
    def _synthesize_relational_experiments(hypotheses, semantic, predictions_by_hyp=None, predictions_map=None):
        """Build the sole authoritative experiment from a proven semantic join path."""
        access = getattr(semantic, "relational_access", None)
        if access is None or not hypotheses:
            return []
        from packages.analytics_core.src.relational.compiler import JoinHop, RelationalPlan
        from packages.analytics_core.src.relational.expressions import AggExpr, ColumnRef, Comparison, Literal

        hops = [JoinHop(**hop) for hop in access.join_hops]
        typed_filters = [
            Comparison(
                ColumnRef(binding["column"], binding["table"]),
                "=",
                Literal(binding["value"]),
            )
            for binding in (getattr(access, "filters", None) or [])
        ]
        # The relational experiment must inherit the already-resolved metric
        # semantics. Hard-coding SUM here would silently answer a different
        # estimand for rates/averages/counts. Only aggregations supported by the
        # typed relational expression layer are admitted; unsupported semantics
        # fail closed and remain available to a future specialized executor.
        metric_definition = getattr(semantic, "metric_definition", None)
        _aggregation_value = getattr(metric_definition, "aggregation_type", "SUM") or "SUM"
        aggregation_name = str(getattr(_aggregation_value, "value", _aggregation_value)).upper()
        if aggregation_name == "MEAN":
            aggregation_name = "AVG"
        result_alias = {
            "SUM": "total_metric",
            "AVG": "mean_metric",
            "COUNT": "count_val",
            "MIN": "min_metric",
            "MAX": "max_metric",
        }.get(aggregation_name)
        if result_alias is None:
            return []
        plan = RelationalPlan(
            base_table=access.base_table,
            hops=hops,
            select_columns=[f"{access.joined_table}.{access.dim_column} AS {access.dim_column}"],
            group_by=[f"{access.joined_table}.{access.dim_column}"],
            aggregations={result_alias: AggExpr(aggregation_name, ColumnRef(access.metric_column, access.base_table))},
            filters=typed_filters or None,
            order_by=f"{result_alias} DESC",
        )
        if plan.validate():
            return []
        predictions_by_hyp = predictions_by_hyp or {}
        predictions_map = predictions_map or {}
        candidates = []
        for hypothesis in hypotheses:
            prediction_id = predictions_by_hyp.get(hypothesis.hypothesis_code)
            candidates.append(CandidateExperiment(
                code=f"EXP-RELATIONAL-{hypothesis.hypothesis_code}",
                target_hypothesis_code=hypothesis.hypothesis_code,
                hypothesis_ids=[hypothesis.hypothesis_code],
                experiment_role=ExperimentRole.PRIMARY.value,
                target_prediction_id=prediction_id,
                prediction_ids=list(predictions_map.get(hypothesis.hypothesis_code, [prediction_id] if prediction_id else [])),
                tool_name="duckdb_sql",
                query_sql=plan.to_sql(),
                description=(
                    f"Aggregate {access.metric_column} by {access.dim_column} through proven-safe relational access"
                    + (f" with {len(typed_filters)} question-bound filter(s)." if typed_filters else ".")
                ),
                aggregation_type=aggregation_name,
                target_dimension=access.dim_column,
                target_metric=access.metric_column,
                primary_result_column=result_alias,
                join_hops=[dict(hop) for hop in access.join_hops],
                relational_plan=plan,
                analysis_row_count=0,
                full_scope=True,
            ))
        return candidates

    @staticmethod
    def synthesize_candidate_experiments(
        hypotheses: List[PredictiveHypothesis],
        semantic: SemanticResolution,
        unresolved_adversarial_issues: Optional[List[Dict[str, Any]]] = None,
        predictions_by_hyp: Optional[Dict[str, str]] = None,
        predictions_map: Optional[Dict[str, List[str]]] = None,
        time_filter_sql: Optional[str] = None,
        decision: Any = None,
        sample_stats: Optional[Dict[str, Any]] = None,
    ) -> List[CandidateExperiment]:
        """
        sample_stats (optional): {"row_count": int, "category_cardinalities": {dim: k}}
        computed by the caller from the actual primary_df. When supplied, it lets
        the concentration/isolation candidates below compute a real, data-dependent
        EIG likelihood instead of leaving it unavailable. Omitting it is always
        safe -- those candidates just fall back to EIG=0.0, exactly as before.
        """
        relational_candidates = ExperimentSynthesizer._synthesize_relational_experiments(hypotheses, semantic, predictions_by_hyp, predictions_map)
        if relational_candidates:
            return relational_candidates
        if decision is not None:
            from packages.analytics_core.src.engines.method_selection import MethodFamily
            family = getattr(decision, "method_family", None)
            # C3.1: dispatch routing must key off the MethodFamily the
            # canonical decision itself carries, not off a raw semantic.*
            # role field. semantic.secondary_metric_col / semantic.time_col
            # can be stale relative to decision.estimand (e.g. a
            # differently-phrased question resolved the roles through a
            # different path), and gating dispatch on them meant a valid
            # canonical decision could be silently routed to the generic
            # battery instead of the specialized synthesizer it selected.
            # Each specialized synthesizer already fails closed (returns
            # []) when the decision it was given carries an incomplete
            # canonical binding, so no semantic pre-check is needed here --
            # trust the decision.method_family and let the callee enforce
            # its own required bindings.
            if family == MethodFamily.CORRELATION:
                candidates = ExperimentSynthesizer._synthesize_correlation_experiments(hypotheses, semantic, predictions_by_hyp, predictions_map, decision=decision)
                for c in candidates:
                    c.fingerprint = c.compute_fingerprint()
                if time_filter_sql:
                    for c in candidates:
                        c.query_sql = _inject_time_filter(c.query_sql, time_filter_sql)
                        c.description = f"[Scoped to resolved question period] {c.description}"
                return candidates
            if family == MethodFamily.FORECAST:
                candidates = ExperimentSynthesizer._synthesize_forecast_experiments(hypotheses, semantic, predictions_by_hyp, predictions_map, decision=decision)
                for c in candidates:
                    c.fingerprint = c.compute_fingerprint()
                return candidates
            if family == MethodFamily.CHURN:
                candidates = ExperimentSynthesizer._synthesize_churn_experiments(hypotheses, semantic, predictions_by_hyp, predictions_map, decision=decision)
                for c in candidates:
                    c.fingerprint = c.compute_fingerprint()
                if time_filter_sql:
                    for c in candidates:
                        c.query_sql = _inject_time_filter(c.query_sql, time_filter_sql)
                        c.description = f"[Scoped to resolved question period] {c.description}"
                return candidates
            # Diagnostic/generic decision falls through to the existing generic battery.

        # DEFECT-005 (sections 4/5/8): CORRELATION and FORECAST hypotheses
        # need experiments that can actually discriminate them (a real
        # Pearson/Spearman association test; a real chronological
        # trend+backtest), not the root-cause-style concentration/ANOVA/
        # uniformity battery below, which was previously generated for
        # every hypothesis regardless of intent and produces a statistic
        # (share-of-total, categorical eta^2) that says nothing about
        # bivariate association or temporal predictability. Detected via
        # the hypothesis fields HypothesisSynthesizer now sets for these
        # two intents specifically (secondary_metric / claim_type
        # "PREDICTION"), so this still works for any caller that
        # constructs hypotheses directly without going through intent
        # classification (defensive fallback: falls through to the
        # existing generic battery below).
        # C3.1a: these three defensive hypothesis-based fallbacks omit
        # decision=decision, so calling them at all lets the helper enter
        # its legacy/no-decision branch and re-derive its role from raw
        # semantic.* fields -- even when a canonical decision was supplied
        # and already selected a *different* method family (e.g.
        # DIAGNOSTIC_BATTERY). That silently reinterprets the decision's
        # own hypothesis using semantic roles the decision explicitly did
        # not choose to use for this dispatch, breaking the "no analytical
        # role may fall back to raw semantic.* once a decision is supplied"
        # invariant one level below the family dispatch above. These
        # remain defensive fallbacks for callers that construct hypotheses
        # directly without a canonical decision at all (decision is None);
        # once a decision exists, family selection is already final and a
        # non-matching family (e.g. diagnostic) must fall through to the
        # generic decision path below rather than re-entering a
        # specialized helper via this legacy route.
        if decision is None and hypotheses and any(getattr(h, "secondary_metric", "") for h in hypotheses):
            candidates = ExperimentSynthesizer._synthesize_correlation_experiments(
                hypotheses, semantic, predictions_by_hyp, predictions_map,
            )
            for c in candidates:
                c.fingerprint = c.compute_fingerprint()
            if time_filter_sql:
                for c in candidates:
                    c.query_sql = _inject_time_filter(c.query_sql, time_filter_sql)
                    c.description = f"[Scoped to resolved question period] {c.description}"
            return candidates
        if decision is None and hypotheses and any(getattr(h, "claim_type", "") == "PREDICTION" for h in hypotheses):
            candidates = ExperimentSynthesizer._synthesize_forecast_experiments(
                hypotheses, semantic, predictions_by_hyp, predictions_map,
            )
            for c in candidates:
                c.fingerprint = c.compute_fingerprint()
            # Deliberately NOT time-filtered: a forecast experiment needs
            # the full chronological history to fit a trend and hold out a
            # backtest window; scoping it to a single resolved period (the
            # concentration-style filter meant for root-cause/segmentation
            # questions) would leave too few points to fit or backtest.
            return candidates
        if decision is None and hypotheses and getattr(semantic, "churn_event_col", None) is not None:
            candidates = ExperimentSynthesizer._synthesize_churn_experiments(
                hypotheses, semantic, predictions_by_hyp, predictions_map,
            )
            for c in candidates:
                c.fingerprint = c.compute_fingerprint()
            if time_filter_sql:
                for c in candidates:
                    c.query_sql = _inject_time_filter(c.query_sql, time_filter_sql)
                    c.description = f"[Scoped to resolved question period] {c.description}"
            return candidates

        # v20-C3.1: this generic (diagnostic-battery) path is reached for
        # problem classes with no dedicated MethodFamily specialization.
        # When a decision is supplied, decision.estimand is the sole
        # authority for the target-metric role: MethodSelectionEngine._estimand
        # always mirrors semantic.target_metric_col into estimand.target_column,
        # so in the ordinary case this is a no-op in value terms -- but it
        # must not silently fall back to semantic.target_metric_col when the
        # canonical role is genuinely absent, or a stale/incomplete semantic
        # field could no longer suppress the decision (see C3) but could
        # instead silently substitute for it here, one level down. Missing
        # required binding = fail closed. The comparison dimension is not a
        # required role for this generic path (many diagnostic questions have
        # none), so it is preferred from the decision when present but is not
        # itself a fail-closed condition.
        if decision is not None:
            _decision_estimand = getattr(decision, "estimand", None)
            metric = getattr(_decision_estimand, "target_column", None) if _decision_estimand is not None else None
            if not metric:
                # Canonical decision supplied but its target-column role is
                # unresolved -- fail closed rather than substituting
                # semantic.target_metric_col.
                return []
            resolved_group_dim = getattr(_decision_estimand, "comparison_dimension", None) if _decision_estimand is not None else None
        else:
            # Legacy/no-decision compatibility path only.
            metric = semantic.target_metric_col
            resolved_group_dim = semantic.group_dimension_col
        available_dims = semantic.available_categorical_cols or []
        if resolved_group_dim:
            primary_dim = resolved_group_dim
        else:
            # Never fall back to raw column order.  Prefer categorical fields
            # that behave like actual dimensions rather than entity identifiers
            # or near-unique labels.
            primary_dim = None
            if getattr(semantic, "primary_dataset_name", None) and getattr(semantic, "world_model", None):
                # The synthesizer intentionally avoids owning another schema
                # inference system.  When no explicit grouping was resolved,
                # select only from the semantic engine's categorical candidates
                # and let ambiguity remain unresolved rather than inventing a
                # group-by key.
                dim_candidates = [
                    d for d in available_dims
                    if not any(k in str(d).lower() for k in ("_id", "key", "pk", "fk", "uuid"))
                ]
                # `available_categorical_cols` is schema metadata, not raw data,
                # so near-uniqueness cannot be recomputed here.  Prefer the first
                # non-ID candidate only when it is the sole candidate.
                primary_dim = dim_candidates[0] if len(dim_candidates) == 1 else None
        predictions_by_hyp = predictions_by_hyp or {}
        predictions_map = predictions_map or {}
        candidates: List[CandidateExperiment] = []

        # Phase 11 (Metric Semantics Engine): consume the resolved MetricDefinition
        # instead of assuming every metric is SUM-based. Falls back to a SUM
        # MetricDefinition only if an older/manually-constructed SemanticResolution
        # has no metric_definition populated (defensive backward compatibility).
        metric_def: MetricDefinition = getattr(semantic, "metric_definition", None) or MetricDefinition(
            name=metric, table_name=semantic.primary_dataset_name, source_columns=[metric],
            semantic_type="sum_measure", aggregation_type=AggregationType.SUM,
            grain=semantic.table_grain, is_additive=True,
            valid_aggregations=[AggregationType.SUM], rationale="Default fallback (no MetricDefinition supplied).",
            semantic_resolution_status="LEGACY_FALLBACK",
        )
        metric_agg_str = metric_def.aggregation_type.value.upper()
        # A CandidateExperiment's own aggregation_type describes the aggregation
        # of ITS OWN query, which may legitimately differ from the primary
        # metric's semantics (e.g. the ANOVA test aggregates raw row-level
        # values, not the group total). Only the concentration/isolation/
        # confounding tests directly aggregate the target metric's total and
        # must therefore respect metric_def's semantics.
        metric_agg_expr = MetricSemanticsResolver.sql_aggregation_expression(metric_def, "total_metric")

        for idx, h in enumerate(hypotheses):
            h_dim = h.target_dimension or primary_dim
            target_val = getattr(h, "target_value", None)
            target_pid = predictions_by_hyp.get(h.hypothesis_code)
            all_pids = predictions_map.get(h.hypothesis_code, [target_pid] if target_pid else list(getattr(h, "prediction_ids", []) or []))
            all_pids = [p for p in all_pids if p]

            if target_val is not None and h_dim:
                # Emergent isolated test
                iso_n_categories = None
                if sample_stats:
                    iso_n_categories = (sample_stats.get("category_cardinalities") or {}).get(h_dim)
                iso_likelihoods = None
                if metric_def.is_additive and sample_stats:
                    iso_likelihoods = concentration_share_likelihoods(
                        n_rows=sample_stats.get("row_count") or 0,
                        n_categories=iso_n_categories,
                        share_threshold=ISOLATION_SHARE_THRESHOLD,
                        effect_size_if_true=CONCENTRATION_EFFECT_IF_TRUE,
                    )
                target_val_escaped = str(target_val).replace("'", "''")
                candidates.append(
                    CandidateExperiment(
                        code=f"EXP-ISOLATE-{h.hypothesis_code}",
                        target_hypothesis_code=h.hypothesis_code,
                        hypothesis_ids=[h.hypothesis_code],
                        target_prediction_id=target_pid or (all_pids[0] if all_pids else None),
                        prediction_ids=all_pids or ([target_pid] if target_pid else []),
                        target_uncertainty=f"Isolate specific concentration of {metric} in {h_dim}='{target_val}'",
                        is_exploratory=len(all_pids) == 0 and target_pid is None,
                        experiment_role=(
                            ExperimentRole.SUPPORTING.value if (len(all_pids) == 0 and target_pid is None)
                            else ExperimentRole.PRIMARY.value
                        ),
                        tool_name="duckdb_sql",
                        query_sql=f"SELECT '{target_val_escaped}' AS {h_dim}, {metric_agg_expr} FROM data_table WHERE {h_dim} = '{target_val_escaped}'",
                        description=f"Isolated measurement of {metric} ({metric_def.semantic_type}) specifically within segment '{target_val}' of {h_dim}.",
                        aggregation_type=metric_agg_str,
                        discriminating_power=0.98,
                        discrimination_value=0.98,
                        estimated_cost=1.0,
                        reliability_weight=0.98,
                        decision_relevance=1.0,
                        target_dimension=h_dim,
                        target_metric=metric,
                        primary_result_column="total_metric",
                        metrics=[metric],
                        dimensions=[h_dim],
                        expected_outcomes=[f"Total {metric} in segment '{target_val}' accounts for dominant share (> {ISOLATION_SHARE_THRESHOLD:.0%})"],
                        likelihood_if_true=iso_likelihoods.likelihood_if_true if iso_likelihoods else None,
                        likelihood_if_false=iso_likelihoods.likelihood_if_false if iso_likelihoods else None,
                        provenance=({"eig_likelihood_basis": iso_likelihoods.basis} if iso_likelihoods else {}),
                    )
                )

            if not h.is_counter_hypothesis:
                # 1. Primary Concentration & Volume Disparity Test
                #
                # Phase 11: "concentration" (share-of-total) is only a
                # scientifically meaningful framing for an ADDITIVE metric --
                # summing a rate/ratio/average across categories and reporting
                # "this segment is 80% of the total" is meaningless (Phase 11
                # Part 12). For a non-additive metric this test still has
                # discriminating value, but it must rank groups by the
                # correctly-aggregated value (mean/rate/ratio), not a SUM share.
                conc_code = "EXP-CONC" if idx == 0 else f"EXP-CONC-{h.hypothesis_code}"
                conc_likelihoods = None
                if metric_def.is_additive:
                    conc_sql = (
                        f"SELECT {h_dim}, {metric_agg_expr}, COUNT(*) AS n_rows FROM data_table GROUP BY {h_dim} ORDER BY total_metric DESC"
                        if h_dim else
                        f"SELECT {metric_agg_expr}, COUNT(*) AS n_rows FROM data_table"
                    )
                    conc_desc = (
                        f"Dimensional volume breakdown of {metric} ({metric_def.semantic_type}) across {h_dim} to evaluate concentration for {h.hypothesis_code}."
                        if h_dim else f"Aggregate total and row count for {metric}."
                    )
                    conc_outcomes = [f"Top segment accounts for >= {CONCENTRATION_SHARE_THRESHOLD:.0%} of total variance"]
                    if sample_stats and h_dim:
                        conc_n_categories = (sample_stats.get("category_cardinalities") or {}).get(h_dim)
                        conc_likelihoods = concentration_share_likelihoods(
                            n_rows=sample_stats.get("row_count") or 0,
                            n_categories=conc_n_categories,
                            share_threshold=CONCENTRATION_SHARE_THRESHOLD,
                            effect_size_if_true=CONCENTRATION_EFFECT_IF_TRUE,
                        )
                else:
                    # Non-additive metrics must be aggregated and ranked under
                    # a "mean_metric" alias (matching the declared
                    # primary_result_column below), NOT the shared
                    # metric_agg_expr, which is always aliased "total_metric"
                    # for the additive branch/isolation query above. Reusing
                    # metric_agg_expr here previously produced a query that
                    # returns a column named "total_metric" while declaring
                    # primary_result_column="mean_metric" -- a name that never
                    # appears in the result, which execution_provider.py
                    # would reject with "Declared primary result column
                    # 'mean_metric' was not returned by the query."
                    conc_mean_expr = MetricSemanticsResolver.sql_aggregation_expression(metric_def, "mean_metric")
                    conc_sql = (
                        f"SELECT {h_dim}, {conc_mean_expr}, COUNT(*) AS n_rows FROM data_table GROUP BY {h_dim} ORDER BY mean_metric DESC"
                        if h_dim else
                        f"SELECT {conc_mean_expr}, COUNT(*) AS n_rows FROM data_table"
                    )
                    conc_desc = (
                        f"Dimensional breakdown of {metric} ({metric_def.semantic_type}, aggregated as "
                        f"{metric_agg_str} -- NOT summed, since it is non-additive) across {h_dim} to evaluate "
                        f"whether any segment's rate/average diverges from the rest for {h.hypothesis_code}."
                        if h_dim else f"{metric_agg_str} of non-additive metric {metric}."
                    )
                    conc_outcomes = ["A segment's rate/average diverges materially from the rest (no share-of-total claim, since the metric is non-additive)"]
                candidates.append(
                    CandidateExperiment(
                        code=conc_code,
                        target_hypothesis_code=h.hypothesis_code,
                        hypothesis_ids=[h.hypothesis_code],
                        target_prediction_id=target_pid or (all_pids[0] if all_pids else None),
                        prediction_ids=all_pids or ([target_pid] if target_pid else []),
                        target_uncertainty=f"Determine whether variance in {metric} is localized across {h_dim}",
                        is_exploratory=len(all_pids) == 0 and target_pid is None,
                        experiment_role=(
                            ExperimentRole.SUPPORTING.value if (len(all_pids) == 0 and target_pid is None)
                            else ExperimentRole.PRIMARY.value
                        ),
                        tool_name="duckdb_sql",
                        query_sql=conc_sql,
                        description=conc_desc,
                        aggregation_type=metric_agg_str,
                        discriminating_power=0.90,
                        discrimination_value=0.90,
                        estimated_cost=1.0,
                        reliability_weight=0.95,
                        decision_relevance=1.0,
                        target_dimension=h_dim,
                        target_metric=metric,
                        primary_result_column=("total_metric" if metric_def.is_additive else "mean_metric"),
                        metrics=[metric],
                        dimensions=[h_dim] if h_dim else [],
                        expected_outcomes=conc_outcomes,
                        likelihood_if_true=conc_likelihoods.likelihood_if_true if conc_likelihoods else None,
                        likelihood_if_false=conc_likelihoods.likelihood_if_false if conc_likelihoods else None,
                        provenance=({"eig_likelihood_basis": conc_likelihoods.basis} if conc_likelihoods else {}),
                    )
                )

                # 2. ANOVA Variance Decomposition Test
                anova_code = "EXP-ANOVA" if idx == 0 else f"EXP-ANOVA-{h.hypothesis_code}"
                anova_sql = (
                    f"SELECT {h_dim}, {metric} FROM data_table WHERE {metric} IS NOT NULL"
                    if h_dim else
                    f"SELECT {metric} FROM data_table WHERE {metric} IS NOT NULL"
                )
                anova_likelihoods = None
                if sample_stats and h_dim:
                    anova_n_groups = (sample_stats.get("category_cardinalities") or {}).get(h_dim)
                    anova_likelihoods = anova_eta_squared_likelihoods(
                        n_rows=sample_stats.get("row_count") or 0,
                        n_groups=anova_n_groups,
                        eta_squared_threshold=ANOVA_ETA_SQUARED_THRESHOLD,
                        effect_size_if_true=ANOVA_ETA_SQUARED_THRESHOLD,
                    )
                candidates.append(
                    CandidateExperiment(
                        code=anova_code,
                        target_hypothesis_code=h.hypothesis_code,
                        hypothesis_ids=[h.hypothesis_code],
                        target_prediction_id=target_pid or (all_pids[0] if all_pids else None),
                        prediction_ids=all_pids or ([target_pid] if target_pid else []),
                        target_uncertainty=f"Measure statistical effect size eta-squared of {h_dim} on {metric}",
                        is_exploratory=len(all_pids) == 0 and target_pid is None,
                        experiment_role=(
                            ExperimentRole.SUPPORTING.value if (len(all_pids) == 0 and target_pid is None)
                            else ExperimentRole.PRIMARY.value
                        ),
                        tool_name="scipy_anova",
                        query_sql=anova_sql,
                        description=f"One-way ANOVA eta-squared variance decomposition of {metric} by {h_dim}." if h_dim else f"Variance calculation of {metric}.",
                        aggregation_type="VARIANCE",
                        discriminating_power=0.95,
                        discrimination_value=0.95,
                        estimated_cost=1.2,
                        reliability_weight=0.98,
                        decision_relevance=0.95,
                        target_dimension=h_dim,
                        target_metric=metric,
                        metrics=[metric],
                        dimensions=[h_dim] if h_dim else [],
                        expected_outcomes=[f"Between-group sum of squares eta^2 >= {ANOVA_ETA_SQUARED_THRESHOLD:.0%}"],
                        likelihood_if_true=anova_likelihoods.likelihood_if_true if anova_likelihoods else None,
                        likelihood_if_false=anova_likelihoods.likelihood_if_false if anova_likelihoods else None,
                        provenance=({"eig_likelihood_basis": anova_likelihoods.basis} if anova_likelihoods else {}),
                    )
                )
            else:
                # 3. Distributional Uniformity Test (Counter Hypothesis)
                unif_code = "EXP-UNIF" if idx == 1 else f"EXP-UNIF-{h.hypothesis_code}"
                unif_metric_expr = MetricSemanticsResolver.sql_aggregation_expression(metric_def, "mean_metric")
                unif_sql = (
                    f"SELECT {h_dim}, {unif_metric_expr}, STDDEV({metric}) AS std_metric FROM data_table GROUP BY {h_dim} ORDER BY mean_metric DESC"
                    if h_dim else
                    f"SELECT {unif_metric_expr}, STDDEV({metric}) AS std_metric FROM data_table"
                )
                candidates.append(
                    CandidateExperiment(
                        code=unif_code,
                        target_hypothesis_code=h.hypothesis_code,
                        hypothesis_ids=[h.hypothesis_code],
                        target_prediction_id=target_pid or (all_pids[0] if all_pids else None),
                        prediction_ids=all_pids or ([target_pid] if target_pid else []),
                        target_uncertainty=f"Test for uniform systemic drift across {h_dim}",
                        is_exploratory=len(all_pids) == 0 and target_pid is None,
                        # Counter-hypothesis falsification test (see the "3.
                        # Distributional Uniformity Test (Counter Hypothesis)"
                        # comment above) -- always ADVERSARIAL in intent
                        # regardless of prediction linkage.
                        experiment_role=ExperimentRole.ADVERSARIAL.value,
                        tool_name="duckdb_sql",
                        query_sql=unif_sql,
                        description=f"Categorical {metric_def.semantic_type} ({metric_agg_str}) and standard deviation of {metric} across {h_dim} to test uniform drift for {h.hypothesis_code}." if h_dim else f"{metric_agg_str} and standard deviation of {metric}.",
                        aggregation_type=metric_agg_str,
                        discriminating_power=0.85,
                        discrimination_value=0.85,
                        estimated_cost=1.0,
                        reliability_weight=0.90,
                        decision_relevance=0.90,
                        target_dimension=h_dim,
                        target_metric=metric,
                        metrics=[metric],
                        dimensions=[h_dim] if h_dim else [],
                        expected_outcomes=["Uniform mean and spread across all categories (< 5% dispersion)"],
                    )
                )

        # 4. Cross-Dimensional Confounding Interaction Tests
        dimensions = list(dict.fromkeys([h.target_dimension for h in hypotheses if h.target_dimension]))
        if len(dimensions) >= 2:
            dim_a, dim_b = dimensions[0], dimensions[1]
            target_h_code = "HYP-03" if any(h.hypothesis_code == "HYP-03" for h in hypotheses) else hypotheses[-1].hypothesis_code
            confound_expr = MetricSemanticsResolver.sql_aggregation_expression(metric_def, "total_metric")
            candidates.append(
                CandidateExperiment(
                    code="EXP-CONFOUND",
                    target_hypothesis_code=target_h_code,
                    hypothesis_ids=[target_h_code],
                    experiment_role=ExperimentRole.ADVERSARIAL.value,
                    tool_name="duckdb_sql",
                    query_sql=f"SELECT {dim_a}, {dim_b}, {confound_expr} FROM data_table GROUP BY {dim_a}, {dim_b} ORDER BY total_metric DESC, {dim_a}, {dim_b}",
                    description=f"Two-way cross-tabulation across {dim_a} and {dim_b} to test for confounding and interaction ({metric} aggregated as {metric_agg_str}).",
                    aggregation_type=metric_agg_str,
                    discriminating_power=0.88,
                    discrimination_value=0.88,
                    estimated_cost=1.5,
                    reliability_weight=0.92,
                    decision_relevance=0.85,
                    target_dimension=f"{dim_a}_{dim_b}",
                    target_metric=metric,
                    metrics=[metric],
                    dimensions=[dim_a, dim_b],
                    is_exploratory=True,
                    causal_value=0.85,
                )
            )

        # 5. Targeted Adversarial Follow-Up Tests
        if unresolved_adversarial_issues:
            for issue in unresolved_adversarial_issues:
                confounding_dim = issue.get("confounding_dimension") or issue.get("secondary_dimension") or issue.get("dimension")
                if confounding_dim and confounding_dim != primary_dim:
                    cond_expr = MetricSemanticsResolver.sql_aggregation_expression(metric_def, "total_metric")
                    candidates.append(
                        CandidateExperiment(
                            code=f"EXP-COND-{confounding_dim.upper()}",
                            target_hypothesis_code=hypotheses[0].hypothesis_code,
                            hypothesis_ids=[hypotheses[0].hypothesis_code],
                            experiment_role=ExperimentRole.ADVERSARIAL.value,
                            tool_name="duckdb_sql",
                            query_sql=f"SELECT {primary_dim}, {confounding_dim}, {cond_expr} FROM data_table GROUP BY {primary_dim}, {confounding_dim} ORDER BY total_metric DESC, {primary_dim}, {confounding_dim}",
                            description=f"Conditional group analysis across {primary_dim} conditioned on {confounding_dim} to test Simpson's paradox ({metric} aggregated as {metric_agg_str}).",
                            aggregation_type=metric_agg_str,
                            discriminating_power=0.99,
                            discrimination_value=0.99,
                            estimated_cost=1.4,
                            reliability_weight=0.96,
                            decision_relevance=0.95,
                            target_dimension=f"{primary_dim}_{confounding_dim}",
                            target_metric=metric,
                            metrics=[metric],
                            dimensions=[primary_dim, confounding_dim],
                            adversarial_value=0.95,
                            target_uncertainty=f"Resolve Simpson's paradox across secondary partition '{confounding_dim}'",
                            provenance={"attack_type": "simpsons_paradox", "confounding_dimension": confounding_dim},
                        )
                    )

        # 6. Verification / Replication Candidate Experiments
        # Formally produce synthesis-time VERIFICATION experiments (Audit P1-12)
        # to close the conceptual gap in scientific provenance.
        primary_candidates = [c for c in candidates if c.experiment_role == ExperimentRole.PRIMARY.value]
        if primary_candidates:
            lead_primary = primary_candidates[0]
            v_code = f"EXP-VERIFY-{lead_primary.code.replace('EXP-', '')}"
            candidates.append(
                CandidateExperiment(
                    code=v_code,
                    target_hypothesis_code=lead_primary.target_hypothesis_code,
                    hypothesis_ids=list(lead_primary.hypothesis_ids),
                    experiment_role=ExperimentRole.VERIFICATION.value,
                    replication_of=lead_primary.code,
                    tool_name="duckdb_sql",
                    query_sql=lead_primary.query_sql,
                    description=f"Independent recomputation verification of {lead_primary.code} ({lead_primary.description}).",
                    aggregation_type=lead_primary.aggregation_type,
                    discriminating_power=lead_primary.discriminating_power,
                    discrimination_value=lead_primary.discrimination_value * 0.9,
                    robustness_value=0.99,
                    reliability_weight=1.0,
                    decision_relevance=lead_primary.decision_relevance,
                    target_dimension=lead_primary.target_dimension,
                    target_metric=lead_primary.target_metric,
                    metrics=list(lead_primary.metrics),
                    dimensions=list(lead_primary.dimensions),
                    provenance={
                        "verification_target": lead_primary.code,
                        "replication_of": lead_primary.code,
                        "experiment_role": ExperimentRole.VERIFICATION.value,
                    },
                )
            )

        # Assign fingerprints
        for c in candidates:
            c.fingerprint = c.compute_fingerprint()

        if time_filter_sql:
            for c in candidates:
                c.query_sql = _inject_time_filter(c.query_sql, time_filter_sql)
                c.description = f"[Scoped to resolved question period] {c.description}"

        return candidates

    @staticmethod
    def _synthesize_correlation_experiments(
        hypotheses: List[PredictiveHypothesis],
        semantic: SemanticResolution,
        predictions_by_hyp: Optional[Dict[str, str]] = None,
        predictions_map: Optional[Dict[str, List[str]]] = None,
        decision: Any = None,
    ) -> List[CandidateExperiment]:
        """DEFECT-005 (section 4): real, deterministic, re-verifiable
        experiment for a CORRELATION hypothesis pair -- returns the raw
        paired (metric, secondary_metric) observations so the actual
        Pearson r / p-value can be computed AND independently re-verified
        by the dual-engine verification layer (see ScientificTransitionService,
        which computes r/p from this same result_df), exactly like every
        other experiment's result is independently re-verified.

        v20-C3: when a MethodSelectionDecision is supplied, its
        estimand.target_column / estimand.predictor_columns are the sole
        authoritative source of which column is the outcome and which is
        the predictor -- this function no longer independently reads
        semantic.target_metric_col / semantic.secondary_metric_col in that
        case. This is what makes the symmetric-association case
        ("is price associated with sales?" vs "are price and sales
        associated?") stable: whichever role assignment the compiler/
        method-selection layer resolved is exactly what gets tested, with
        no second, independent re-derivation here that could silently
        disagree with it.

        Fail-closed: if a decision is supplied but its estimand does not
        carry a resolved target/predictor pair, this returns no candidates
        rather than silently falling back to semantic.target_metric_col /
        semantic.secondary_metric_col (which could be stale relative to the
        canonical contract).

        decision may legitimately be omitted -- some callers construct
        hypotheses directly without going through MethodSelectionEngine
        (see the comment at the CORRELATION dispatch site above). That
        compatibility path is intentionally preserved and reads semantic
        directly, exactly as before; it is exercised by
        test_c3_canonical_experiment_contract.py.
        """
        predictions_by_hyp = predictions_by_hyp or {}
        predictions_map = predictions_map or {}
        if decision is not None:
            estimand = getattr(decision, "estimand", None)
            metric = getattr(estimand, "target_column", None) if estimand is not None else None
            predictors = list(dict.fromkeys(getattr(estimand, "predictor_columns", None) or [])) if estimand is not None else []
            if not metric or not predictors:
                # Canonical contract present but incomplete -- fail closed
                # rather than substituting the stale semantic.* fields.
                return []
            # v20-C4.2.1: Pearson/Spearman is undefined for a categorical
            # predictor. Fail closed instead of emitting a correlation
            # experiment over a string column (which would either error at
            # execution or, worse, be silently mislabeled).
            cat_pool = set(getattr(semantic, "available_categorical_cols", None) or [])
            valid_predictors = [p for p in predictors if p not in cat_pool]
            if not valid_predictors:
                return []
        else:
            # Legacy/compatibility path: no canonical decision was computed
            # for this call (see synthesize_candidate_experiments' defensive
            # fallback branch). Deliberately marked and tested, not a silent
            # "canonical or stale" fallback chain.
            metric = semantic.target_metric_col
            valid_predictors = [semantic.secondary_metric_col] if getattr(semantic, "secondary_metric_col", None) else []
            if not metric or not valid_predictors:
                return []

        if len(valid_predictors) > 1:
            joint_requested = bool(getattr(getattr(decision, "estimand", None), "joint_predictors", False)) if decision is not None else False
            if joint_requested:
                target_hyp = next((h for h in hypotheses if not getattr(h, "is_counter_hypothesis", False)), hypotheses[0] if hypotheses else None)
                if target_hyp is None:
                    return []
                pred_sql = ", ".join([metric] + valid_predictors)
                predicates = " AND ".join([f"{c} IS NOT NULL" for c in [metric] + valid_predictors])
                regression_sql = f"SELECT {pred_sql} FROM data_table WHERE {predicates}"
                return [
                    CandidateExperiment(
                        code="EXP-MV-REG",
                        target_hypothesis_code=target_hyp.hypothesis_code,
                        hypothesis_ids=[h.hypothesis_code for h in hypotheses],
                        target_prediction_id=predictions_by_hyp.get(target_hyp.hypothesis_code),
                        prediction_ids=predictions_map.get(target_hyp.hypothesis_code, []),
                        target_uncertainty=f"Estimate the conditional association of {metric} with {', '.join(valid_predictors)}, holding the other included predictors constant.",
                        is_exploratory=False,
                        experiment_role=ExperimentRole.PRIMARY.value,
                        tool_name="statsmodels_ols",
                        query_sql=regression_sql,
                        description=f"Multivariate OLS conditional association of {metric} on {', '.join(valid_predictors)}; each coefficient is interpreted holding the other included predictors constant.",
                        aggregation_type="MULTIVARIATE_REGRESSION",
                        discriminating_power=0.98,
                        likelihood_if_true=0.75,
                        likelihood_if_false=0.25,
                        estimated_cost=1.5,
                        reliability_weight=0.94,
                        decision_relevance=1.0,
                        target_dimension=None,
                        target_metric=metric,
                        metrics=[metric] + valid_predictors,
                        predictors=list(valid_predictors),
                        primary_result_column=metric,
                        expected_outcomes=[
                            f"The fitted conditional model reports coefficient, uncertainty, and diagnostics for each predictor; conclusions remain associational unless a causal identification strategy exists.",
                        ],
                        provenance={
                            "analysis_type": "multivariate_conditional_association",
                            "joint_predictors": True,
                            "predictors": list(valid_predictors),
                            "target": metric,
                        },
                    )
                ]
            candidates: List[CandidateExperiment] = []
            for idx, other in enumerate(valid_predictors, start=1):
                exp_code = f"EXP-CORR-{idx:02d}"
                target_hyp = next((h for h in hypotheses if getattr(h, "secondary_metric", "") == other and not getattr(h, "is_counter_hypothesis", False)), None)
                if target_hyp is None:
                    target_hyp = next((h for h in hypotheses if getattr(h, "secondary_metric", "") == other), None)
                if target_hyp is None:
                    target_hyp = hypotheses[idx - 1] if idx - 1 < len(hypotheses) else (hypotheses[0] if hypotheses else None)
                target_hyp_code = target_hyp.hypothesis_code if target_hyp else f"HYP-{idx:02d}"
                target_pid = predictions_by_hyp.get(target_hyp_code) if target_hyp else None
                all_pids = predictions_map.get(target_hyp_code, [target_pid] if target_pid else []) if target_hyp else []
                all_pids = [p for p in all_pids if p]

                corr_sql = f"SELECT {metric}, {other} FROM data_table WHERE {metric} IS NOT NULL AND {other} IS NOT NULL"
                candidates.append(
                    CandidateExperiment(
                        code=exp_code,
                        target_hypothesis_code=target_hyp_code,
                        hypothesis_ids=[target_hyp_code],
                        target_prediction_id=target_pid or (all_pids[0] if all_pids else None),
                        prediction_ids=all_pids,
                        target_uncertainty=f"Determine whether {metric} and {other} are materially associated",
                        is_exploratory=len(all_pids) == 0,
                        experiment_role=ExperimentRole.PRIMARY.value,
                        tool_name="scipy_correlation",
                        query_sql=corr_sql,
                        description=f"Paired observations of {metric} and {other} for bivariate Pearson/Spearman correlation testing.",
                        aggregation_type="CORRELATION",
                        discriminating_power=0.95,
                        discrimination_value=0.95,
                        estimated_cost=1.0,
                        reliability_weight=0.95,
                        decision_relevance=1.0,
                        target_dimension=None,
                        target_metric=metric,
                        metrics=[metric, other],
                        dimensions=[],
                        expected_outcomes=[
                            f"|Pearson r| between {metric} and {other} is substantially different from zero with p < 0.05.",
                        ],
                    )
                )
            return candidates

        other = valid_predictors[0]
        h1 = next((h for h in hypotheses if not getattr(h, "is_counter_hypothesis", False)), hypotheses[0]) if hypotheses else None
        h1_code = h1.hypothesis_code if h1 else "HYP-01"
        target_pid = predictions_by_hyp.get(h1_code) if h1 else None
        all_pids = predictions_map.get(h1_code, [target_pid] if target_pid else []) if h1 else []
        all_pids = [p for p in all_pids if p]

        corr_sql = f"SELECT {metric}, {other} FROM data_table WHERE {metric} IS NOT NULL AND {other} IS NOT NULL"
        return [
            CandidateExperiment(
                code="EXP-CORR",
                target_hypothesis_code=h1_code,
                hypothesis_ids=[h.hypothesis_code for h in hypotheses] if hypotheses else [h1_code],
                target_prediction_id=target_pid or (all_pids[0] if all_pids else None),
                prediction_ids=all_pids,
                target_uncertainty=f"Determine whether {metric} and {other} are materially associated",
                is_exploratory=len(all_pids) == 0,
                experiment_role=ExperimentRole.PRIMARY.value,
                tool_name="scipy_correlation",
                query_sql=corr_sql,
                description=f"Paired observations of {metric} and {other} for bivariate Pearson/Spearman correlation testing.",
                aggregation_type="CORRELATION",
                discriminating_power=0.95,
                discrimination_value=0.95,
                estimated_cost=1.0,
                reliability_weight=0.95,
                decision_relevance=1.0,
                target_dimension=None,
                target_metric=metric,
                metrics=[metric, other],
                dimensions=[],
                expected_outcomes=[
                    f"|Pearson r| between {metric} and {other} is substantially different from zero with p < 0.05.",
                ],
            )
        ]

    @staticmethod
    def _synthesize_forecast_experiments(
        hypotheses: List[PredictiveHypothesis],
        semantic: SemanticResolution,
        predictions_by_hyp: Optional[Dict[str, str]] = None,
        predictions_map: Optional[Dict[str, List[str]]] = None,
        decision: Any = None,
    ) -> List[CandidateExperiment]:
        """DEFECT-005 (section 5): real, deterministic, re-verifiable
        experiment for a FORECAST hypothesis pair -- returns the metric
        aggregated per chronological time bucket (ordered), so a real
        trend-slope significance test AND a chronological train/test
        backtest can be computed (and independently re-verified) from this
        same result_df, with no future information able to leak into the
        "training" portion since the ordering itself defines the split.

        v20-C3: consumes decision.estimand.target_column / time_column when
        a canonical decision is supplied, instead of independently reading
        semantic.target_metric_col / semantic.time_col. Fails closed (no
        candidates) if the canonical contract is present but incomplete.
        See _synthesize_correlation_experiments for the same pattern and
        the legacy/no-decision compatibility rationale.
        """
        predictions_by_hyp = predictions_by_hyp or {}
        predictions_map = predictions_map or {}
        if decision is not None:
            estimand = getattr(decision, "estimand", None)
            metric = getattr(estimand, "target_column", None) if estimand is not None else None
            time_col = getattr(estimand, "time_column", None) if estimand is not None else None
            if not metric or not time_col:
                return []
        else:
            metric = semantic.target_metric_col
            time_col = semantic.time_col
        candidates: List[CandidateExperiment] = []

        h1 = next((h for h in hypotheses if not h.is_counter_hypothesis), hypotheses[0])
        target_pid = predictions_by_hyp.get(h1.hypothesis_code)
        all_pids = predictions_map.get(h1.hypothesis_code, [target_pid] if target_pid else [])
        all_pids = [p for p in all_pids if p]

        # Aggregate to a per-period series (daily bucket via DATE_TRUNC) so
        # the regression/backtest operates on a genuine time series rather
        # than raw transaction-level noise; ordered chronologically so the
        # train/test split downstream is a real chronological holdout.
        metric_def: MetricDefinition = getattr(semantic, "metric_definition", None) or MetricDefinition(
            name=metric, table_name=semantic.primary_dataset_name, source_columns=[metric],
            semantic_type="sum_measure", aggregation_type=AggregationType.SUM,
            grain=semantic.table_grain, is_additive=True,
            valid_aggregations=[AggregationType.SUM],
            rationale="Forecast fallback (no MetricDefinition supplied).",
            semantic_resolution_status="LEGACY_FALLBACK",
        )
        period_expr = MetricSemanticsResolver.sql_aggregation_expression(metric_def, metric)
        forecast_sql = (
            f"SELECT DATE_TRUNC('day', TRY_CAST({time_col} AS DATE)) AS period, {period_expr} "
            f"FROM data_table WHERE {metric} IS NOT NULL AND {time_col} IS NOT NULL "
            f"AND TRY_CAST({time_col} AS DATE) IS NOT NULL "
            f"GROUP BY period ORDER BY period ASC"
        )
        candidates.append(
            CandidateExperiment(
                code="EXP-FORECAST-TREND",
                target_hypothesis_code=h1.hypothesis_code,
                hypothesis_ids=[h.hypothesis_code for h in hypotheses],
                target_prediction_id=target_pid or (all_pids[0] if all_pids else None),
                prediction_ids=all_pids,
                target_uncertainty=f"Determine whether {metric} over {time_col} contains predictive temporal structure",
                is_exploratory=len(all_pids) == 0,
                experiment_role=ExperimentRole.PRIMARY.value,
                tool_name="statsmodels_ols",
                query_sql=forecast_sql,
                description=f"Chronological daily series of {metric} over {time_col} for trend-significance testing and a chronological holdout backtest.",
                aggregation_type="REGRESSION",
                discriminating_power=0.95,
                discrimination_value=0.95,
                estimated_cost=1.2,
                reliability_weight=0.95,
                decision_relevance=1.0,
                target_dimension=time_col,
                target_metric=metric,
                metrics=[metric],
                dimensions=[time_col],
                expected_outcomes=[
                    f"Chronological holdout backtest error for a fitted trend is lower than a naive baseline, with a statistically significant (p < 0.05) trend slope.",
                ],
            )
        )
        return candidates

    @staticmethod
    def _synthesize_churn_experiments(
        hypotheses: List[PredictiveHypothesis],
        semantic: SemanticResolution,
        predictions_by_hyp: Optional[Dict[str, str]] = None,
        predictions_map: Optional[Dict[str, List[str]]] = None,
        decision: Any = None,
    ) -> List[CandidateExperiment]:
        """DEFECT-015: Synthesize experiments evaluating churn identifiability:
        - Crude rate comparison (events / eligible population)
        - Baseline uniformity test (overall event distribution)
        - Stratified confounder check (Simpson's paradox test)
        - Exposure-adjusted person-time rate (if exposure column present)
        - Censoring diagnostic (if censoring column present)

        v20-C3: when a MethodSelectionDecision is supplied, its
        estimand.churn_bindings dict (event_col / exposure_col /
        censored_col / confounder_cols / outcome_available) is the
        authoritative source for these roles, set once in
        MethodSelectionEngine._select_churn. This function no longer
        independently re-reads semantic.churn_* in that case. Falls back
        to the semantic fields directly only when no decision is supplied
        (legacy/compatibility path, same rationale as the correlation and
        forecast synthesizers above).

        The grouping dimension (`dim`) remains outside this canonical
        binding: MethodSelectionEngine resolves a comparison_dimension only
        when semantic.group_dimension_col is itself already set, so the
        categorical-candidate fallback below is schema metadata, not a
        resolved analytical role that could diverge from a canonical
        contract -- there is no second, independent resolution of it to
        disagree with.
        """
        predictions_by_hyp = predictions_by_hyp or {}
        predictions_map = predictions_map or {}
        candidates: List[CandidateExperiment] = []

        if decision is not None:
            churn_bindings = getattr(getattr(decision, "estimand", None), "churn_bindings", None)
            if not churn_bindings:
                # Canonical contract present but never resolved churn
                # bindings for this decision -- fail closed.
                return candidates
            resolved_event_col = churn_bindings.get("event_col")
            outcome_available = bool(churn_bindings.get("outcome_available", False)) and resolved_event_col is not None
        else:
            resolved_event_col = getattr(semantic, "churn_event_col", None)
            outcome_available = getattr(semantic, "churn_outcome_available", True) and resolved_event_col is not None

        if not outcome_available:
            # Generate presence check
            #
            # C3.1a: this candidate is only a schema-presence scan (its
            # query is a bare row count, unconditioned on any dimension or
            # metric) -- the target_dimension/target_metric/metrics/
            # dimensions fields below are cosmetic labels only, never
            # analytical-role authority. But when a decision is supplied,
            # even a cosmetic label must not be sourced from raw
            # semantic.group_dimension_col / semantic.target_metric_col,
            # since those can be stale relative to the canonical decision.
            # Prefer decision.estimand's own resolved fields for the label;
            # if the canonical values are unavailable, use neutral labels
            # ("count" / no dimension) rather than falling back to the
            # semantic fields.
            h1 = next((h for h in hypotheses if h.hypothesis_code == "HYP-01"), hypotheses[0])
            if decision is not None:
                _estimand = getattr(decision, "estimand", None)
                label_dim = getattr(_estimand, "comparison_dimension", None) if _estimand is not None else None
                label_metric = getattr(_estimand, "target_column", None) if _estimand is not None else None
            else:
                label_dim = semantic.group_dimension_col
                label_metric = semantic.target_metric_col
            label_dim = label_dim or ""
            label_metric = label_metric or "count"
            candidates.append(
                CandidateExperiment(
                    code="EXP-CHURN-PRESENCE-CHECK",
                    target_hypothesis_code=h1.hypothesis_code,
                    hypothesis_ids=[h.hypothesis_code for h in hypotheses],
                    target_prediction_id=predictions_by_hyp.get(h1.hypothesis_code),
                    prediction_ids=predictions_map.get(h1.hypothesis_code, []),
                    target_uncertainty="Validate presence of churn outcome variable in dataset schema",
                    is_exploratory=True,
                    experiment_role=ExperimentRole.SUPPORTING.value,
                    tool_name="duckdb_sql",
                    query_sql="SELECT COUNT(*) AS total_rows FROM data_table",
                    description="Dataset presence scan confirming that no churn/cancellation outcome column is recorded in schema.",
                    aggregation_type="COUNT",
                    discriminating_power=0.99,
                    discrimination_value=0.99,
                    estimated_cost=1.0,
                    reliability_weight=1.0,
                    decision_relevance=1.0,
                    target_dimension=label_dim,
                    target_metric=label_metric,
                    metrics=[label_metric],
                    dimensions=[label_dim] if label_dim else [],
                    expected_outcomes=["No churn event column available in schema."],
                )
            )
            return candidates

        event_col = resolved_event_col
        comparison_dim = (
            getattr(getattr(decision, "estimand", None), "comparison_dimension", None)
            if decision is not None else semantic.group_dimension_col
        )
        if comparison_dim:
            dim = comparison_dim
        else:
            dim_candidates = [
                d for d in (semantic.available_categorical_cols or [])
                if not any(k in str(d).lower() for k in ("_id", "key", "pk", "fk", "uuid"))
            ]
            dim = dim_candidates[0] if len(dim_candidates) == 1 else None
        if not dim:
            return candidates
        if decision is not None:
            confounders = list(churn_bindings.get("confounder_cols") or [])
            exposure = churn_bindings.get("exposure_col")
            censored = churn_bindings.get("censored_col")
        else:
            confounders = getattr(semantic, "churn_confounder_cols", []) or []
            exposure = getattr(semantic, "churn_exposure_col", None)
            censored = getattr(semantic, "churn_censored_col", None)
        h1 = next((h for h in hypotheses if h.hypothesis_code == "HYP-01"), hypotheses[0])
        h2 = next((h for h in hypotheses if h.hypothesis_code == "HYP-02"), hypotheses[min(1, len(hypotheses)-1)])
        # predictive_hypothesis.py assigns codes to the confounding (H3) and
        # exposure (H4) hypotheses positionally -- HYP-0{len(hyps)+1} at the
        # time each is appended -- so when confounders is empty (H3 skipped),
        # the exposure hypothesis lands on "HYP-03" instead of "HYP-04". A
        # hardcoded code lookup silently returns h4=None in that case, which
        # then made EXP-CHURN-EXPOSURE's target_hypothesis_code fall back to
        # h1 -- mistargeting evidence, leaving the real exposure hypothesis
        # untracked as "targeted", and letting the counter-hypothesis
        # stopping gate wrongly waive it, so the investigation stopped
        # before EXP-CHURN-EXPOSURE ever ran. Match on target_dimension
        # (the actual confounder/exposure column) instead of the
        # positional code, which is stable regardless of which hypotheses
        # exist.
        selected_confounder = confounders[0] if len(confounders) == 1 else None
        h3 = next((h for h in hypotheses if selected_confounder and h.target_dimension == selected_confounder), None)
        h4 = next((h for h in hypotheses if exposure and h.target_dimension == exposure), None)

        # 1. Crude Rate by Segment
        crude_sql = (
            f"SELECT {dim}, SUM({event_col}) AS events, COUNT(*) AS eligible_n, "
            f"CAST(SUM({event_col}) AS DOUBLE) / COUNT(*) AS crude_rate "
            f"FROM data_table WHERE {event_col} IS NOT NULL "
            f"GROUP BY {dim} ORDER BY crude_rate DESC"
        )
        candidates.append(
            CandidateExperiment(
                code="EXP-CHURN-CRUDE",
                target_hypothesis_code=h1.hypothesis_code,
                hypothesis_ids=[h1.hypothesis_code],
                target_prediction_id=predictions_by_hyp.get(h1.hypothesis_code),
                prediction_ids=predictions_map.get(h1.hypothesis_code, []),
                target_uncertainty=f"Measure crude churn rate across categories of {dim}",
                is_exploratory=False,
                experiment_role=ExperimentRole.PRIMARY.value,
                tool_name="duckdb_sql",
                query_sql=crude_sql,
                description=f"Crude churn rate across {dim} (events / eligible population, excluding unconfirmed/censored rows).",
                aggregation_type="RATE",
                discriminating_power=0.95,
                discrimination_value=0.95,
                estimated_cost=1.0,
                reliability_weight=0.95,
                decision_relevance=1.0,
                target_dimension=dim,
                target_metric=event_col,
                metrics=[event_col],
                dimensions=[dim],
                expected_outcomes=[f"Statistically significant difference in crude churn rates across {dim}"],
            )
        )

        # 2. Overall baseline uniformity test
        unif_sql = f"SELECT {event_col}, COUNT(*) AS count_val FROM data_table WHERE {event_col} IS NOT NULL GROUP BY {event_col} ORDER BY count_val DESC"
        candidates.append(
            CandidateExperiment(
                code="EXP-CHURN-UNIF",
                target_hypothesis_code=h2.hypothesis_code,
                hypothesis_ids=[h2.hypothesis_code],
                target_prediction_id=predictions_by_hyp.get(h2.hypothesis_code),
                prediction_ids=predictions_map.get(h2.hypothesis_code, []),
                target_uncertainty="Evaluate population baseline churn hazard uniformity",
                is_exploratory=False,
                experiment_role=ExperimentRole.ADVERSARIAL.value,
                tool_name="duckdb_sql",
                query_sql=unif_sql,
                description=f"Baseline event distribution for {event_col} evaluating population-level uniformity.",
                aggregation_type="COUNT",
                discriminating_power=0.80,
                discrimination_value=0.80,
                estimated_cost=1.0,
                reliability_weight=0.95,
                decision_relevance=0.85,
                target_dimension=dim,
                target_metric=event_col,
                metrics=[event_col],
                dimensions=[dim],
                expected_outcomes=["Baseline churn count distribution."],
            )
        )

        # 3. Stratified rate checks (Simpson's paradox tests) for EVERY
        # resolved confounder.  A realistic churn table may contain cohort,
        # tenure, region, etc.; testing only when exactly one exists silently
        # drops the competing explanations in the multi-confounder case.
        for idx, conf_col in enumerate(confounders, start=1):
            strat_sql = (
                f"SELECT {dim}, {conf_col}, SUM({event_col}) AS events, COUNT(*) AS eligible_n, "
                f"CAST(SUM({event_col}) AS DOUBLE) / COUNT(*) AS stratum_rate "
                f"FROM data_table WHERE {event_col} IS NOT NULL AND {conf_col} IS NOT NULL "
                f"GROUP BY {dim}, {conf_col} ORDER BY eligible_n DESC, stratum_rate DESC"
            )
            conf_h = next((h for h in hypotheses if getattr(h, "target_dimension", None) == conf_col), None)
            strat_target = conf_h.hypothesis_code if conf_h else h1.hypothesis_code
            candidates.append(
                CandidateExperiment(
                    code=f"EXP-CHURN-STRAT-{idx}",
                    target_hypothesis_code=strat_target,
                    hypothesis_ids=[strat_target],
                    target_prediction_id=predictions_by_hyp.get(strat_target),
                    prediction_ids=predictions_map.get(strat_target, []),
                    target_uncertainty=f"Evaluate whether churn differences across {dim} are confounded by {conf_col}",
                    is_exploratory=False,
                    experiment_role=ExperimentRole.ADVERSARIAL.value,
                    tool_name="duckdb_sql",
                    query_sql=strat_sql,
                    description=f"Stratified churn rate across {dim} conditioned on {conf_col} (Simpson's paradox test).",
                    aggregation_type="RATE",
                    discriminating_power=0.98,
                    discrimination_value=0.98,
                    estimated_cost=1.2,
                    reliability_weight=0.98,
                    decision_relevance=1.0,
                    target_dimension=conf_col,
                    target_metric=event_col,
                    metrics=[event_col],
                    dimensions=[dim, conf_col],
                    expected_outcomes=[f"Within-stratum churn differences persist or collapse across {conf_col}"],
                )
            )

        # 4. Exposure-adjusted person-time rate query
        if exposure:
            exp_sql = (
                f"SELECT {dim}, SUM({event_col}) AS events, SUM({exposure}) AS person_time, "
                f"CAST(SUM({event_col}) AS DOUBLE) / SUM({exposure}) AS persontime_rate "
                f"FROM data_table WHERE {exposure} IS NOT NULL "
                f"GROUP BY {dim} ORDER BY persontime_rate DESC"
            )
            exp_target = h4.hypothesis_code if h4 else h1.hypothesis_code
            candidates.append(
                CandidateExperiment(
                    code="EXP-CHURN-EXPOSURE",
                    target_hypothesis_code=exp_target,
                    hypothesis_ids=[exp_target],
                    target_prediction_id=predictions_by_hyp.get(exp_target),
                    prediction_ids=predictions_map.get(exp_target, []),
                    target_uncertainty=f"Evaluate exposure-adjusted person-time hazard rate in {exposure}",
                    is_exploratory=False,
                    experiment_role=ExperimentRole.SUPPORTING.value,
                    tool_name="duckdb_sql",
                    query_sql=exp_sql,
                    description=f"Person-time exposure-adjusted churn rate by {dim} (events / person-time at risk in {exposure}).",
                    aggregation_type="RATE",
                    discriminating_power=0.95,
                    discrimination_value=0.95,
                    estimated_cost=1.1,
                    reliability_weight=0.95,
                    decision_relevance=0.95,
                    target_dimension=exposure,
                    target_metric=event_col,
                    metrics=[event_col, exposure],
                    dimensions=[dim],
                    expected_outcomes=[f"Person-time adjusted hazard rates converge or diverge across {dim}"],
                )
            )

        # 5. Censoring breakdown diagnostic
        if censored:
            censor_sql = f"SELECT {censored}, {event_col}, COUNT(*) AS count_val FROM data_table GROUP BY {censored}, {event_col} ORDER BY count_val DESC"
            candidates.append(
                CandidateExperiment(
                    code="EXP-CHURN-CENSOR",
                    target_hypothesis_code=h1.hypothesis_code,
                    hypothesis_ids=[h1.hypothesis_code],
                    target_prediction_id=predictions_by_hyp.get(h1.hypothesis_code),
                    prediction_ids=predictions_map.get(h1.hypothesis_code, []),
                    target_uncertainty="Evaluate right-censoring proportion and incomplete observation",
                    is_exploratory=False,
                    experiment_role=ExperimentRole.SUPPORTING.value,
                    tool_name="duckdb_sql",
                    query_sql=censor_sql,
                    description=f"Censoring distribution breakdown ({censored}): confirmed non-churn vs right-censored.",
                    aggregation_type="COUNT",
                    discriminating_power=0.85,
                    discrimination_value=0.85,
                    estimated_cost=1.0,
                    reliability_weight=0.95,
                    decision_relevance=0.90,
                    target_dimension=censored,
                    target_metric=event_col,
                    metrics=[event_col, censored],
                    dimensions=[censored],
                    expected_outcomes=["Count of confirmed events, confirmed non-events, and censored rows."],
                )
            )

        return candidates

    @staticmethod
    def dynamically_replan_candidates(
        hypotheses: List[PredictiveHypothesis],
        semantic: SemanticResolution,
        executed_codes: List[str],
        current_posteriors: List[float],
        last_result_df: Optional[pd.DataFrame] = None,
        last_experiment: Optional[CandidateExperiment] = None,
        unresolved_adversarial_issues: Optional[List[Dict[str, Any]]] = None,
        time_filter_sql: Optional[str] = None,
        decision: Any = None,
        sample_stats: Optional[Dict[str, Any]] = None,
    ) -> DynamicReplanResult:
        """Adapts candidate pool and synthesizes emergent hypotheses based on newly observed evidence."""
        replan = DynamicReplanResult()
        modeled_dims: Dict[str, Any] = {}
        for h in hypotheses:
            if h.target_dimension and getattr(h, "target_value", None) is not None:
                modeled_dims[h.target_dimension] = h.target_value

        # Run active evidence pattern detectors
        discovered_explanations: List[CandidateExplanation] = []
        metric_is_additive = getattr(semantic.metric_definition, "is_additive", True)
        if last_result_df is not None and not last_result_df.empty:
            for detector in ACTIVE_DETECTORS:
                expl = detector.detect(last_result_df, modeled_dims, is_additive=metric_is_additive)
                if expl is not None:
                    discovered_explanations.append(expl)

        # Formulate emergent hypotheses if patterns discovered
        if discovered_explanations:
            pat_dicts = [
                {
                    "type": expl.pattern_name,
                    "dimension": expl.dimension_col,
                    "metric": expl.value_col,
                    "top_value": expl.dominant_value,
                    "top_share_pct": (expl.dominant_share * 100) if expl.dominant_share else 0.0,
                    "source_experiment_id": last_experiment.code if last_experiment else "",
                }
                for expl in discovered_explanations
            ]
            from packages.analytics_core.src.intelligence.predictive_hypothesis import HypothesisSynthesizer
            new_hyps = HypothesisSynthesizer.synthesize_emergent_hypotheses(pat_dicts, semantic, hypotheses)
            replan.new_hypotheses = new_hyps
            replan.trigger_reason = f"Discovered {len(discovered_explanations)} evidence patterns from experiment {last_experiment.code if last_experiment else ''}."

        all_hyps = list(hypotheses) + replan.new_hypotheses
        replan.candidates = [
            c for c in ExperimentSynthesizer.synthesize_candidate_experiments(
                all_hyps, semantic, unresolved_adversarial_issues=unresolved_adversarial_issues,
                time_filter_sql=time_filter_sql, decision=decision, sample_stats=sample_stats,
            )
            if c.code not in executed_codes
        ]

        return replan
