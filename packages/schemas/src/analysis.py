"""Authoritative Pydantic Schemas for Typed Analytical IR, Semantic World Model, Epistemic State Vectors, and Verification Proofs."""
from datetime import datetime, timezone
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field

from packages.shared.src.enums import (
    AnalysisStatus,
    AnalyticalVerdict,
    CausalEvidenceLevel,
    ConfidenceLevel,
    ValidationStatus,
    ChartType,
)


# ============================================================================
# 1. CORE ENUMS & ANALYTICAL PRIMITIVES
# ============================================================================

class ObjectiveType(str, Enum):
    DESCRIBE = "describe"
    COMPARE = "compare"
    ROOT_CAUSE = "root_cause"
    FORECAST = "forecast"
    PREDICT = "predict"
    ESTIMATE_CAUSAL_EFFECT = "estimate_causal_effect"
    OPTIMIZE = "optimize"


class AggregationType(str, Enum):
    SUM = "sum"
    MEAN = "mean"
    COUNT = "count"
    COUNT_DISTINCT = "count_distinct"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    RATE = "rate"
    # Phase 11 (Metric Semantics Engine): denominator-dependent and
    # weighted metric types. RATIO and PROPORTION are structurally
    # identical (numerator/denominator, non-additive) but kept distinct
    # for narrative/unit clarity (PROPORTION is conventionally in [0,1]
    # or a percentage of a whole; RATIO is a general numerator/denominator
    # relationship, e.g. cost-to-revenue). WEIGHTED_MEAN is an arithmetic
    # mean where each row/group must be weighted by a volume/weight column
    # rather than averaged unweighted.
    RATIO = "ratio"
    PROPORTION = "proportion"
    WEIGHTED_MEAN = "weighted_mean"


class FilterOperator(str, Enum):
    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    NOT_IN = "not_in"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"


class CardinalityType(str, Enum):
    ONE_TO_ONE = "ONE_TO_ONE"
    ONE_TO_MANY = "ONE_TO_MANY"
    MANY_TO_ONE = "MANY_TO_ONE"
    MANY_TO_MANY = "MANY_TO_MANY"
    one_to_one = "one_to_one"
    one_to_many = "one_to_many"
    many_to_one = "many_to_one"
    many_to_many = "many_to_many"


class CausalIdentifiabilityStatus(str, Enum):
    IDENTIFIED_BACKDOOR = "IDENTIFIED_BACKDOOR"
    QUASI_EXPERIMENTAL = "QUASI_EXPERIMENTAL"  # DiD, IV, RDD
    OBSERVATIONAL_ONLY = "OBSERVATIONAL_ONLY"
    NOT_IDENTIFIABLE = "NOT_IDENTIFIABLE"      # Unmeasured confounding bounds exceeded


class GrainPreservationStatus(str, Enum):
    PRESERVED = "PRESERVED"
    AGGREGATED_SAFELY = "AGGREGATED_SAFELY"
    FANOUT_DETECTED = "FANOUT_DETECTED"
    UNVERIFIED = "UNVERIFIED"


class EvidenceLevel(IntEnum):
    RAW_OBSERVATION = 0
    VALIDATED_COMPUTATION = 1
    DESCRIPTIVE_CLAIM = 2
    ASSOCIATION = 3
    ROBUST_ASSOCIATION = 4
    PREDICTION = 5
    CAUSAL_IDENTIFICATION = 6
    CAUSAL_ESTIMATION = 7
    RECOMMENDATION = 8


class ClaimGateOutcome(str, Enum):
    ANSWER = "ANSWER"
    QUALIFIED_ANSWER = "QUALIFIED_ANSWER"
    REFUSE = "REFUSE"


class ClaimGateDesignStatus(str, Enum):
    KNOWN = "KNOWN"
    ASSUMED = "ASSUMED"
    UNKNOWN = "UNKNOWN"


class EpistemicClaimType(str, Enum):
    OBSERVATION = "OBSERVATION"
    ASSOCIATION = "ASSOCIATION"
    PREDICTION = "PREDICTION"
    SIMULATION = "SIMULATION"
    CAUSAL_INFERENCE = "CAUSAL_INFERENCE"


class PredictionStatus(str, Enum):
    """Lifecycle status of a falsifiable prediction deduced from a hypothesis."""
    PENDING = "PENDING"
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    INCONCLUSIVE = "INCONCLUSIVE"
    NOT_TESTABLE = "NOT_TESTABLE"


class HypothesisStatus(str, Enum):
    """Lifecycle status of a hypothesis, distinct from legacy free-text status."""
    ACTIVE = "ACTIVE"
    SUPPORTED = "SUPPORTED"
    WEAKENED = "WEAKENED"
    REFUTED = "REFUTED"
    INCONCLUSIVE = "INCONCLUSIVE"
    RETIRED = "RETIRED"


# ============================================================================
# 2. TYPED ANALYTICAL IR (INTERMEDIATE REPRESENTATION)
# ============================================================================

class VariableRef(BaseModel):
    table: str
    column: str


class MetricRef(BaseModel):
    name: str
    table: Optional[str] = None
    column: Optional[str] = None
    aggregation: AggregationType = AggregationType.SUM


class GrainRef(BaseModel):
    table: str
    keys: List[str] = Field(min_length=1)


class FilterAST(BaseModel):
    variable: VariableRef
    operator: FilterOperator
    value: Optional[Union[str, int, float, bool, List[str], List[int], List[float]]] = None


class TemporalScope(BaseModel):
    column: VariableRef
    start: Optional[str] = None
    end: Optional[str] = None
    comparison_start: Optional[str] = None
    comparison_end: Optional[str] = None


class ConstraintSpec(BaseModel):
    max_acceptable_uncertainty: Optional[float] = Field(default=None, ge=0.0)
    compute_budget_seconds: Optional[int] = Field(default=None, gt=0)


class CausalIntent(BaseModel):
    treatment: VariableRef
    outcome: VariableRef
    assumed_dag_id: str
    required_identification: bool = True


class TypedAnalyticalIntent(BaseModel):
    """
    Canonical Typed Analytical IR.
    LLMs propose this object; deterministic validators and compilers verify and execute it.
    """
    intent_id: str
    objective: ObjectiveType
    target_metric: MetricRef
    population: List[FilterAST] = Field(default_factory=list)
    unit_of_analysis: GrainRef
    dimensions: List[VariableRef] = Field(default_factory=list)
    temporal_scope: Optional[TemporalScope] = None
    hypotheses: List[str] = Field(default_factory=list)
    constraints: ConstraintSpec = Field(default_factory=ConstraintSpec)
    causal_intent: Optional[CausalIntent] = None
    source_question: Optional[str] = None


# Alias for backward compatibility
AnalyticalIntent = TypedAnalyticalIntent


# ============================================================================
# 2B. CALCULATION LINEAGE & AUDITABLE DERIVATIONS
# ============================================================================

class CalculationTraceStep(BaseModel):
    """One auditable calculation/verification step in an AA-OS lineage trace."""
    step_id: str
    kind: str
    title: str
    formula: Optional[str] = None
    executable_expression: Optional[str] = None
    inputs: Dict[str, Any] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    output: Dict[str, Any] = Field(default_factory=dict)
    source_columns: List[str] = Field(default_factory=list)
    source_row_scope: Optional[str] = None
    execution_engine: Optional[str] = None
    parent_step_ids: List[str] = Field(default_factory=list)
    verification_status: Optional[str] = None
    notes: Optional[str] = None


class CalculationTraceSchema(BaseModel):
    """Canonical immutable lineage from dataset snapshot to analytical result."""
    trace_id: str
    trace_version: str = "1.0"
    investigation_id: str
    experiment_id: str
    dataset_fingerprints: Dict[str, str] = Field(default_factory=dict)
    input_row_count: int = 0
    output_row_count: int = 0
    steps: List[CalculationTraceStep] = Field(default_factory=list)
    final_output: Dict[str, Any] = Field(default_factory=dict)
    reproducibility_statement: str = ""
    canonical_hash: str


# ============================================================================
# 3. SEMANTIC WORLD MODEL (FORMAL RELATIONAL SCHEMAS)
# ============================================================================

class ColumnSchema(BaseModel):
    name: str
    data_type: str
    is_nullable: bool = False
    distinct_values_count: Optional[int] = None
    null_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    semantic_type: Optional[str] = None


class CandidateKey(BaseModel):
    columns: List[str] = Field(min_length=1)
    is_verified_primary: bool = False
    uniqueness_ratio: float = Field(default=1.0, ge=0.0, le=1.0)


class GrainDefinition(BaseModel):
    table_name: str
    grain_keys: List[str] = Field(min_length=1)
    description: Optional[str] = None


class TableSchema(BaseModel):
    table_id: str
    name: str
    row_count: int = Field(ge=0)
    columns: List[ColumnSchema] = Field(default_factory=list)
    candidate_keys: List[CandidateKey] = Field(default_factory=list)
    grain: Optional[GrainDefinition] = None


class JoinRelationship(BaseModel):
    source_table: str
    target_table: str
    source_columns: List[str] = Field(min_length=1)
    target_columns: List[str] = Field(min_length=1)
    cardinality: CardinalityType = CardinalityType.MANY_TO_ONE
    is_verified: bool = True
    join_integrity_score: float = Field(default=1.0, ge=0.0, le=1.0)


class FunctionalDependency(BaseModel):
    determinant: List[str] = Field(min_length=1)
    dependent: List[str] = Field(min_length=1)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class MetricDefinition(BaseModel):
    name: str
    table_name: str
    target_column: Optional[str] = None
    aggregation: AggregationType = AggregationType.SUM
    sql_formula: Optional[str] = None
    unit_of_analysis: str


class SemanticWorldModel(BaseModel):
    """
    Deterministic Semantic World Model representing verifiable schema ground truth.
    Eliminates unstructured dictionary representations.
    """
    dataset_hash: str
    tables: List[TableSchema] = Field(default_factory=list)
    relationships: List[JoinRelationship] = Field(default_factory=list)
    functional_dependencies: List[FunctionalDependency] = Field(default_factory=list)
    metrics: List[MetricDefinition] = Field(default_factory=list)


# Alias for backward compatibility
SemanticWorldModelSchema = SemanticWorldModel


# ============================================================================
# 4. FORMAL HYPOTHESIS GRAPH
# ============================================================================

class FalsificationCriteria(BaseModel):
    statistical_test: str
    threshold: float
    direction: str = "less_than"
    falsification_statement: Optional[str] = None


class HypothesisNode(BaseModel):
    hypothesis_id: str
    proposition: str
    causal_mechanism: Optional[str] = None
    prior_belief: float = Field(default=0.5, ge=0.0, le=1.0)
    falsification_criteria: List[FalsificationCriteria] = Field(default_factory=list)
    competing_hypothesis_ids: List[str] = Field(default_factory=list)
    posterior_belief: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    status: str = "proposed"


class HypothesisEdge(BaseModel):
    source_id: str
    target_id: str
    relationship_type: str = "COMPETES_WITH"


class HypothesisGraph(BaseModel):
    nodes: List[HypothesisNode] = Field(default_factory=list)
    edges: List[HypothesisEdge] = Field(default_factory=list)
    shannon_entropy: float = Field(default=0.0, ge=0.0)


# ============================================================================
# 5. EPISTEMIC STATE VECTORS & VERIFICATION PROOFS
# ============================================================================

class EpistemicStateVector(BaseModel):
    """
    Strict multi-dimensional uncertainty vector.
    Disallows collapsing epistemics into a single subjective confidence score.
    """
    model_config = {"protected_namespaces": ()}

    data_quality_score: float = Field(ge=0.0, le=1.0)
    semantic_certainty: float = Field(ge=0.0, le=1.0)
    statistical_confidence: float = Field(ge=0.0, le=1.0)
    model_fidelity: float = Field(ge=0.0, le=1.0)
    causal_status: CausalIdentifiabilityStatus = CausalIdentifiabilityStatus.OBSERVATIONAL_ONLY
    causal_sensitivity_e_value: Optional[float] = Field(default=None, ge=1.0)
    multiple_testing_adjusted: bool = True
    multiverse_robustness_pct: Optional[float] = Field(default=None, ge=0.0, le=100.0)


class GrainVerificationProof(BaseModel):
    """Mathematical verification distinguishing requested vs observed grain and fanout factors."""
    requested_grain: str
    observed_grain: str
    pre_join_row_count: int
    post_join_row_count: int
    distinct_key_count: int
    join_cardinality: CardinalityType
    expected_cardinality: CardinalityType
    actual_cardinality: CardinalityType
    preservation_status: GrainPreservationStatus
    fanout_factor: float
    verification_evidence: str


class JoinVerificationProof(BaseModel):
    left_table: str
    right_table: str
    join_keys: List[str]
    cardinality_left: str
    cardinality_right: str
    is_many_to_many: bool
    many_to_many_explicitly_requested: bool = False
    is_join_safe: bool


class CausalIdentifiabilityProof(BaseModel):
    treatment: str
    outcome: str
    assumed_dag_hash: str
    backdoor_adjustment_set: List[str] = Field(default_factory=list)
    is_identifiable: bool
    unmeasured_confounding_risk: str


class VerificationResult(BaseModel):
    primary_engine: str = "duckdb_sql"
    secondary_engine: str = "polars_vectorized"
    primary_metric_val: float
    secondary_metric_val: Optional[float] = None
    observed_delta_pct: Optional[float] = None
    tolerance_threshold: float = 1e-4
    is_mathematically_identical: bool
    grain_proof: Optional[GrainVerificationProof] = None
    join_proof: Optional[JoinVerificationProof] = None
    causal_proof: Optional[CausalIdentifiabilityProof] = None


# ============================================================================
# 6. EVIDENCE GRAPH & DECISION MODELS
# ============================================================================

class EvidenceGraphNode(BaseModel):
    finding_id: str
    evidence_id: str
    tool_used: str = "duckdb_sql"
    query_executed: Optional[str] = None
    table_name: str = "data_table"
    dataset_version: int = 1
    validation_status: str = "PASSED"
    tolerance_verified: bool = True
    grain_proof: Optional[GrainVerificationProof] = None
    join_proof: Optional[JoinVerificationProof] = None
    causal_proof: Optional[CausalIdentifiabilityProof] = None


class EvidenceNode(BaseModel):
    evidence_id: str
    finding_id: Optional[str] = None
    statement: str
    calculation_summary: str
    dataset_version: str
    row_count_analyzed: int
    sql_executed: Optional[str] = None
    verification_result: Optional[VerificationResult] = None
    validation_status: ValidationStatus = ValidationStatus.PASSED
    verified_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EvidenceEdge(BaseModel):
    source_id: str
    target_id: str
    relationship_type: str = "SUPPORTS"  # SUPPORTS, REFUTES, WEAKENS, VERIFIES


class EvidenceGraph(BaseModel):
    nodes: List[EvidenceNode] = Field(default_factory=list)
    edges: List[EvidenceEdge] = Field(default_factory=list)


class MultiVectorConfidence(BaseModel):
    """Legacy multi-vector confidence wrapper for backward compatibility."""
    model_config = {"protected_namespaces": ()}

    evidence_quality: int = Field(default=100, ge=0, le=100)
    data_quality: int = Field(default=100, ge=0, le=100)
    statistical_strength: int = Field(default=100, ge=0, le=100)
    causal_evidence: str = "OBSERVATIONAL"
    model_reliability: int = Field(default=100, ge=0, le=100)
    overall_verdict: str = "CONFIRMED"


class ExpectedUtilityCalculation(BaseModel):
    action_name: str
    expected_gain_metric: float
    downside_risk_metric: float
    probability_of_success: float = Field(ge=0.0, le=1.0)
    net_expected_utility: float
    utility_function_description: str


class DecisionRecommendation(BaseModel):
    recommendation_id: str
    action_title: str
    action_description: str
    grounded_hypothesis_id: str
    target_metric: str
    expected_utility: ExpectedUtilityCalculation
    policy_compliance_passed: bool = True
    required_preconditions: List[str] = Field(default_factory=list)


# ============================================================================
# 7. RESPONSE & COMPATIBILITY MODELS
# ============================================================================

class ClaimGateResultSchema(BaseModel):
    outcome: ClaimGateOutcome
    requested_claim: str
    evidence_level: EvidenceLevel
    design_status: ClaimGateDesignStatus
    assumptions: List[str] = Field(default_factory=list)
    allowed_claim: str
    blocked_claim: Optional[str] = None
    reason: str
    recovery_actions: List[str] = Field(default_factory=list)
    computed_evidence: Dict[str, Any] = Field(default_factory=dict)
    refusal_id: Optional[str] = None
    requested_level: Optional[int] = None
    max_supported_level: Optional[int] = None
    identification_strategy: Optional[str] = None
    blocking_conditions: List[str] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)


class EvidenceSchema(BaseModel):
    id: str = Field(description="Evidence unique ID")
    finding_id: Optional[str] = None
    statement: str
    calculation_summary: str
    dataset_version: str
    row_count_analyzed: int
    time_window: Optional[str] = None
    sql_executed: Optional[str] = None
    python_executed: Optional[str] = None
    statistical_test: Optional[str] = None
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    # P0 epistemic-integrity fix: this used to default to
    # ValidationStatus.PASSED, meaning any construction site that forgot to
    # explicitly compute a validation_status silently claimed verification
    # had *succeeded* -- the most severe possible version of the fabricated-
    # confidence problem also fixed on Evidence.confidence_score (see
    # apps/api/src/models/entities.py). Both current construction sites
    # (apps/api/src/services/analysis_service.py,
    # apps/api/src/ai/runtime.py) already pass this explicitly via
    # to_public_validation_status(), so making it required breaks nothing
    # and prevents a future silent-PASSED regression.
    validation_status: ValidationStatus
    raw_metrics: Dict[str, Any] = Field(default_factory=dict)
    verified_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    verification_result: Optional[VerificationResult] = None
    calculation_trace: Optional[CalculationTraceSchema] = None


class FindingSchema(BaseModel):
    id: str
    title: str
    summary: str
    importance_score: float = Field(ge=0, le=1.0)
    impact_magnitude: Optional[str] = None
    epistemic_state: Optional[EpistemicStateVector] = None  # Replaced single confidence enum
    confidence: ConfidenceLevel = ConfidenceLevel.HIGH     # Deprecated compatibility field
    verdict: AnalyticalVerdict = AnalyticalVerdict.CONFIRMED
    evidence_items: List[EvidenceSchema] = Field(default_factory=list)
    suggested_chart_type: Optional[ChartType] = None
    chart_data: Optional[Dict[str, Any]] = None
    recommended_actions: List[str] = Field(default_factory=list)
    decision_recommendations: List[DecisionRecommendation] = Field(default_factory=list)


class HypothesisSchema(BaseModel):
    id: str
    statement: str
    rationale: str
    priority: float = Field(ge=0, le=1.0)
    status: str = "proposed"
    reason_for_rejection: Optional[str] = None
    investigation_steps: List[str] = Field(default_factory=list)
    confirmed_findings: List[str] = Field(default_factory=list)

    # --- Prediction-lifecycle / canonical-state extension fields ---
    lifecycle_status: HypothesisStatus = HypothesisStatus.ACTIVE
    prior_probability: Optional[float] = None
    posterior_probability: Optional[float] = None
    target_metric: Optional[str] = None
    target_dimension: Optional[str] = None
    target_value: Optional[Union[float, str]] = None
    mechanism: Optional[str] = None
    source_evidence: List[str] = Field(default_factory=list)
    parent_hypotheses: List[str] = Field(default_factory=list)
    generated_reason: Optional[str] = None
    prediction_ids: List[str] = Field(default_factory=list)
    supporting_prediction_count: int = 0
    refuted_prediction_count: int = 0
    unresolved_prediction_count: int = 0
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    contradicting_evidence_ids: List[str] = Field(default_factory=list)


class PredictionSchema(BaseModel):
    """First-class falsifiable prediction deduced from a hypothesis."""
    prediction_id: str
    hypothesis_id: str
    statement: str
    target_metric: Optional[str] = None
    target_dimension: Optional[str] = None
    expected_direction: Optional[str] = None  # increase, decrease, correlate, invariant, concentrated
    expected_value: Optional[Union[float, str]] = None
    expected_range: Optional[List[float]] = None
    threshold: Optional[float] = None
    expected_relationship: Optional[str] = None
    expected_effect: Optional[str] = None
    # Epistemic confidence is never synthesized from templates or priors --
    # it must match StructuredPrediction's own contract (see
    # intelligence/prediction_engine.py). A required float with a fabricated
    # 0.5 default silently rejected every real prediction (confidence=None
    # by design) with a pydantic ValidationError as soon as it reached
    # state_mgr.create_prediction(), breaking prediction registration for
    # any investigation that actually creates one.
    confidence: Optional[float] = None
    testability: bool = True
    status: PredictionStatus = PredictionStatus.PENDING
    target_experiment_id: Optional[str] = None
    actual_observed_result: Optional[Dict[str, Any]] = None
    evaluation_reason: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    evaluated_at: Optional[datetime] = None


class AnalysisStepSchema(BaseModel):
    step_number: int
    title: str
    action_type: str
    description: str
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    tool_output: Optional[Dict[str, Any]] = None
    duration_ms: int = 0
    status: str = "completed"
    error_message: Optional[str] = None


class DiscoverySummary(BaseModel):
    grain: str
    candidate_keys: List[str] = Field(default_factory=list)
    primary_metrics: List[str] = Field(default_factory=list)
    primary_dimensions: List[str] = Field(default_factory=list)
    time_span: Optional[str] = None
    worthy_inquiries: List[str] = Field(default_factory=list)


class AuditSummary(BaseModel):
    data_quality_score: float
    total_rows: int
    null_columns_count: int
    duplicate_rows_count: int
    anomalous_outliers_count: int
    temporal_drift_detected: bool = False


class AnalysisManifestSchema(BaseModel):
    analysis_id: str
    user_question: str
    project_id: str
    dataset_versions_used: List[Dict[str, Any]]
    ai_provider: str
    ai_model: str
    ai_mode: str = "DETERMINISTIC"  # DETERMINISTIC (Mode 1 / Free) or AI_AUGMENTED (Mode 2)
    ai_fallback_triggered: bool = False
    ai_status_message: Optional[str] = None
    prompt_versions: Dict[str, str]
    created_at: datetime
    execution_time_seconds: float
    total_steps: int
    replan_count: int = 0
    validation_summary: Dict[str, Any]
    reproducible_hash: str


class AnalysisCreate(BaseModel):
    question: str
    project_id: str
    dataset_ids: Optional[List[str]] = None
    conversation_id: Optional[str] = None
    context_notes: Optional[str] = None
    # Deterministic local analysis/explanation is the default. AI_AUGMENTED only
    # changes the presentation layer after verified deterministic analysis.
    analysis_mode: Literal["DETERMINISTIC", "AI_AUGMENTED"] = "DETERMINISTIC"


class AnalysisResponse(BaseModel):
    id: str
    project_id: str
    question: str
    status: AnalysisStatus
    verdict: AnalyticalVerdict = AnalyticalVerdict.CONFIRMED
    direct_answer: Optional[str] = None
    main_finding: Optional[str] = None
    confidence: ConfidenceLevel = ConfidenceLevel.HIGH
    confidence_breakdown: Optional[MultiVectorConfidence] = None
    epistemic_state: Optional[EpistemicStateVector] = None
    discovery: Optional[DiscoverySummary] = None
    audit: Optional[AuditSummary] = None
    hypotheses: List[HypothesisSchema] = Field(default_factory=list)
    steps: List[AnalysisStepSchema] = Field(default_factory=list)
    findings: List[FindingSchema] = Field(default_factory=list)
    evidence: List[EvidenceSchema] = Field(default_factory=list)
    evidence_graph: Optional[Union[EvidenceGraph, List[EvidenceGraphNode]]] = None
    suggested_followups: List[str] = Field(default_factory=list)
    # Deterministically discovered questions that the supplied dataset can support.
    # This is distinct from conversational follow-ups for the current question.
    suggested_questions: List[Dict[str, Any]] = Field(default_factory=list)
    # Deterministic exploratory reconnaissance. These are screening statistics only;
    # they do not independently authorize inferential or causal claims.
    exploratory_analysis: Optional[Dict[str, Any]] = None
    executive_bullets: List[str] = Field(default_factory=list)
    what_if_scenarios: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    semantic_world_model: Optional[Union[SemanticWorldModel, SemanticWorldModelSchema, Dict[str, Any]]] = None
    investigation_graph: Optional[Union[HypothesisGraph, Dict[str, Any]]] = None
    decision_recommendations: List[DecisionRecommendation] = Field(default_factory=list)
    adversarial_critique: Optional[Dict[str, Any]] = None
    epistemic_objective: Optional[str] = None
    ai_mode: str = "DETERMINISTIC"  # DETERMINISTIC or AI_AUGMENTED
    ai_fallback_triggered: bool = False
    ai_status_message: Optional[str] = None
    # Structured explanation generated from canonical local investigation state.
    # AI_AUGMENTED may add an AI rewrite, but this local structure remains the source of truth.
    explanation: Optional[Dict[str, Any]] = None
    # Universal question -> analysis contract generated deterministically before
    # execution. This lets the client inspect exactly how AA-OS interpreted the
    # question, what claim was permitted, which evidence was required, and why it stopped.
    analysis_plan: Optional[Dict[str, Any]] = None
    claim_gate: Optional[ClaimGateResultSchema] = None
    manifest: Optional[AnalysisManifestSchema] = None
    created_at: datetime
    updated_at: datetime


# ============================================================================
# 8. CANONICAL INVESTIGATION KERNEL & STATE MODELS
# ============================================================================

class FirstClassExperiment(BaseModel):
    """First-class auditable analytical experiment specification."""
    experiment_id: str
    target_hypothesis_id: str
    purpose: str
    expected_evidence: str
    required_data: List[str] = Field(default_factory=list)
    analytical_method: str = "duckdb_sql"  # duckdb_sql, polars_sql, scipy_anova, statsmodels_ols
    executable_plan: Dict[str, Any] = Field(default_factory=dict)
    assumptions: List[str] = Field(default_factory=list)
    expected_information_gain: Optional[float] = None
    expected_cost: float = 1.0
    reliability_weight: float = 0.95
    decision_relevance: float = 1.0
    adversarial_value: float = 0.0
    robustness_value: float = 0.0
    causal_value: float = 0.0
    selection_rationale: Optional[str] = None
    fingerprint: Optional[str] = None
    replication_of_experiment_id: Optional[str] = None
    refinement_of_experiment_id: Optional[str] = None
    refinement_reason: Optional[str] = None
    validity_conditions: List[str] = Field(default_factory=list)
    target_prediction_ids: List[str] = Field(default_factory=list)
    execution_result: Optional[Dict[str, Any]] = None
    verification_result: Optional[Dict[str, Any]] = None
    provenance_hash: Optional[str] = None
    effect_estimate: Optional[float] = None
    uncertainty_bounds: Optional[List[float]] = None
    interpretation: Optional[str] = None
    conclusion: Optional[str] = None


class RawObservationRecord(BaseModel):
    """Raw empirical observation produced by executing an experiment, kept SEPARATE
    from hypothesis interpretations."""
    observation_id: str
    experiment_id: str
    primary_value: Optional[float] = None
    row_count: int = 0
    execution_time_ms: float = 0.0
    result_summary: Dict[str, Any] = Field(default_factory=dict)
    structured_result: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    calculation_trace: Optional[CalculationTraceSchema] = None
    # Authoritative verification outcome of the original execution. This is
    # persisted on the raw observation so idempotent replay cannot fabricate
    # VERIFIED status for evidence that originally failed or was partial.
    verification_status: str = "UNVERIFIED"


class PredictionEvidenceRecord(BaseModel):
    """Structured evidence produced by a prediction evaluation."""
    evidence_id: str
    source_prediction_id: str
    source_experiment_id: str
    target_hypothesis_id: str
    evidence_type: str = "prediction_evaluation"
    direction: str = "inconclusive"
    strength: float = 0.0
    prediction_status: str = "PENDING"
    actual_observed_result: Dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ConsistencyError(BaseModel):
    """Structured single inconsistency found by validate_state."""
    error_code: str
    message: str
    offending_ids: List[str] = Field(default_factory=list)
    severity: str = "error"  # error | warning


class StateConsistencyReport(BaseModel):
    """Aggregated result of a canonical-state consistency check."""
    is_consistent: bool
    errors: List[ConsistencyError] = Field(default_factory=list)
    checked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class UncertaintyStateSchema(BaseModel):
    """Canonical mirror of UncertaintyState."""
    unresolved_hypotheses: List[str] = Field(default_factory=list)
    discriminating_variables: List[str] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)
    conflicting_evidence: List[str] = Field(default_factory=list)
    failed_predictions: List[str] = Field(default_factory=list)
    assumption_failures: List[str] = Field(default_factory=list)
    high_value_unknowns: List[str] = Field(default_factory=list)
    decision_critical_unknowns: List[str] = Field(default_factory=list)

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


class EvidenceRecord(BaseModel):
    """Canonical immutable evidence ledger record linking claims directly to verified computation."""
    evidence_id: str
    evidence_identity: Optional[str] = None
    claim_statement: str
    claim_type: EpistemicClaimType = EpistemicClaimType.OBSERVATION
    source_experiment_id: str
    source_datasets: List[str] = Field(default_factory=list)
    source_columns: List[str] = Field(default_factory=list)
    row_count_evaluated: int = 0
    computation_proof: Dict[str, Any] = Field(default_factory=dict)
    assumptions: List[str] = Field(default_factory=list)
    verification_status: str = "UNVERIFIED"  # VERIFIED, PARTIALLY_VERIFIED, FAILED, UNVERIFIED
    statistical_p_value: Optional[float] = None
    effect_size: Optional[float] = None
    uncertainty_range: Optional[List[float]] = None
    provenance_hash: str
    calculation_trace: Optional[CalculationTraceSchema] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EvidenceLedgerSchema(BaseModel):
    """Authoritative ledger containing all verified empirical claims underpinning an investigation."""
    investigation_id: str
    records: List[EvidenceRecord] = Field(default_factory=list)
    total_claims: int = 0
    verified_claims_count: int = 0
    rejected_claims_count: int = 0


class CanonicalInvestigationState(BaseModel):
    """Authoritative scientific state container tracking the complete investigation lifecycle."""
    investigation_id: str
    original_question: str
    interpreted_objective: ObjectiveType = ObjectiveType.DESCRIBE
    semantic_world_model: Optional[Dict[str, Any]] = None
    known_facts: List[str] = Field(default_factory=list)
    unknowns: List[str] = Field(default_factory=list)
    hypotheses: List[HypothesisSchema] = Field(default_factory=list)
    priors: Dict[str, float] = Field(default_factory=dict)
    posterior_beliefs: Dict[str, float] = Field(default_factory=dict)
    evidence_ledger: EvidenceLedgerSchema
    predictions: List[PredictionSchema] = Field(default_factory=list)
    candidate_experiments: List[FirstClassExperiment] = Field(default_factory=list)
    completed_experiments: List[FirstClassExperiment] = Field(default_factory=list)
    failed_experiments: List[FirstClassExperiment] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list)
    verification_proofs: List[Dict[str, Any]] = Field(default_factory=list)
    adversarial_attacks: List[Dict[str, Any]] = Field(default_factory=list)
    uncertainty_score: float = 1.0
    information_gain_accumulated: float = 0.0
    analytical_assumptions: List[str] = Field(default_factory=list)
    data_quality_limitations: List[str] = Field(default_factory=list)
    provenance_hash: Optional[str] = None
    current_best_explanation: Optional[str] = None
    rejected_explanations: List[str] = Field(default_factory=list)
    stopping_state: Optional[Dict[str, Any]] = None
    final_verdict: Optional[AnalyticalVerdict] = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Uncertainty extensions
    unresolved_predictions: List[str] = Field(default_factory=list)
    refuted_predictions: List[str] = Field(default_factory=list)
    unsupported_assumptions: List[str] = Field(default_factory=list)
    hypotheses_requiring_discrimination: List[str] = Field(default_factory=list)
    experiments_that_could_reduce_uncertainty: List[str] = Field(default_factory=list)

    # Raw observations & prediction evidence
    raw_observations: List[RawObservationRecord] = Field(default_factory=list)
    prediction_evidence: List[PredictionEvidenceRecord] = Field(default_factory=list)
    calculation_traces: List[CalculationTraceSchema] = Field(default_factory=list)
    executed_experiment_ids: List[str] = Field(default_factory=list)
    failed_experiment_ids: List[str] = Field(default_factory=list)
    candidate_experiment_ids: List[str] = Field(default_factory=list)
    uncertainty_state: Optional[UncertaintyStateSchema] = None