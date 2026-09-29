"""SQLAlchemy ORM Canonical Domain Models for AA-OS Platform."""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

def utc_now():
    return datetime.now(timezone.utc)

import uuid
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Index,
    String,
    Text,
    JSON,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import event

Base = declarative_base()


def gen_uuid() -> str:
    return str(uuid.uuid4())


# ============================================================================
# 1. CORE TENANCY & IDENTITY DOMAIN
# ============================================================================

class User(Base):
    __tablename__ = "users"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    role = Column(String(50), default="analyst")
    organization_id = Column(String(36), ForeignKey("organizations.id"), nullable=True, default="default-org")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=utc_now)


class Organization(Base):
    __tablename__ = "organizations"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    name = Column(String(255), nullable=False)
    slug = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime, default=utc_now)


class OrganizationMember(Base):
    __tablename__ = "organization_members"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    org_id = Column(String(36), ForeignKey("organizations.id"), nullable=False, index=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    role = Column(String(50), default="analyst")
    created_at = Column(DateTime, default=utc_now)


class Project(Base):
    __tablename__ = "projects"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    org_id = Column(String(36), ForeignKey("organizations.id"), nullable=False, default="default-org", index=True)
    owner_id = Column(String(36), nullable=False, default="default-user", index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)


# ============================================================================
# 2. DATASETS & DATASET VERSIONING DOMAIN
# ============================================================================

class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    format = Column(String(50), default="csv")
    current_version = Column(Integer, default=1)
    row_count = Column(Integer, default=0)
    column_count = Column(Integer, default=0)
    data_quality_score = Column(Float, nullable=True)
    profile_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)


class DatasetTransformation(Base):
    """Immutable, queryable transformation edge between dataset versions.

    The edge is scientific lineage metadata: input/output content identities,
    deterministic operation parameters, and an integrity hash of the record.
    """
    __tablename__ = "dataset_transformations"
    __table_args__ = (
        UniqueConstraint("input_version_id", "output_version_id", "record_hash", name="uq_dataset_transformation_record"),
    )
    id = Column(String(36), primary_key=True, default=gen_uuid)
    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=False, index=True)
    input_version_id = Column(String(36), ForeignKey("dataset_versions.id"), nullable=True, index=True)
    output_version_id = Column(String(36), ForeignKey("dataset_versions.id"), nullable=False, index=True)
    operation = Column(String(100), nullable=False)
    input_content_hash = Column(String(64), nullable=True, index=True)
    output_content_hash = Column(String(64), nullable=True, index=True)
    parameters_json = Column(JSON, default=dict)
    affected_rows = Column(Integer, default=0)
    affected_columns_json = Column(JSON, default=list)
    reason = Column(Text, nullable=True)
    source_record_json = Column(JSON, default=dict)
    record_hash = Column(String(64), nullable=False, index=True)
    created_at = Column(DateTime, default=utc_now)


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=False, index=True)
    version_number = Column(Integer, nullable=False)
    file_path = Column(String(512), nullable=False)
    file_size_bytes = Column(Integer, default=0)
    row_count = Column(Integer, default=0)
    column_count = Column(Integer, default=0)
    # Content-addressed identity of the canonical logical dataset snapshot.
    # Unlike legacy fingerprints based on IDs/shape, this hashes schema + all cells.
    content_hash = Column(String(64), nullable=True, index=True)
    transformation_applied = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=utc_now)


# ============================================================================
# 3. CANONICAL SEMANTIC WORLD MODEL DOMAIN (PERSISTENT & VERSIONED)
# ============================================================================

class SemanticModel(Base):
    """Root versioned semantic model attached to a project and specific dataset versions."""
    __tablename__ = "semantic_models"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False, index=True)
    version_number = Column(Integer, default=1, nullable=False)
    status = Column(String(50), default="active")  # active, draft, deprecated
    summary_description = Column(Text, nullable=True)
    table_grains_json = Column(JSON, default=dict)
    dataset_version_ids_json = Column(JSON, default=list)  # list of DatasetVersion IDs
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)


class SemanticEntity(Base):
    """Named business entity identified from datasets (e.g. Customer, Product, Order)."""
    __tablename__ = "semantic_entities"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    semantic_model_id = Column(String(36), ForeignKey("semantic_models.id"), nullable=False, index=True)
    entity_name = Column(String(255), nullable=False)
    primary_key = Column(String(255), nullable=False)
    table_name = Column(String(255), nullable=False)
    natural_keys_json = Column(JSON, default=list)
    attributes_json = Column(JSON, default=list)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now)


class SemanticMetric(Base):
    """Quantitative business metric column or derived measure."""
    __tablename__ = "semantic_metrics"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    semantic_model_id = Column(String(36), ForeignKey("semantic_models.id"), nullable=False, index=True)
    metric_name = Column(String(255), nullable=False)
    display_name = Column(String(255), nullable=False)
    column_name = Column(String(255), nullable=True)
    table_name = Column(String(255), nullable=False)
    additivity = Column(String(50), default="fully_additive")  # fully_additive, semi_additive, non_additive
    unit = Column(String(50), default="USD")
    target_direction = Column(String(50), default="maximize")  # maximize, minimize, stabilize
    sql_formula = Column(Text, nullable=True)
    derived_expression = Column(Text, nullable=True)
    dependent_metrics_json = Column(JSON, default=list)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now)


class SemanticDimension(Base):
    """Categorical dimension and slice-and-dice attribute."""
    __tablename__ = "semantic_dimensions"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    semantic_model_id = Column(String(36), ForeignKey("semantic_models.id"), nullable=False, index=True)
    dimension_name = Column(String(255), nullable=False)
    column_name = Column(String(255), nullable=False)
    table_name = Column(String(255), nullable=False)
    cardinality = Column(Integer, default=0)
    hierarchy_level = Column(Integer, default=1)
    is_hierarchical = Column(Boolean, default=False)
    created_at = Column(DateTime, default=utc_now)


class SemanticRelationship(Base):
    """Foreign-key and join relationship edge between tables."""
    __tablename__ = "semantic_relationships"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    semantic_model_id = Column(String(36), ForeignKey("semantic_models.id"), nullable=False, index=True)
    source_table = Column(String(255), nullable=False)
    source_column = Column(String(255), nullable=False)
    target_table = Column(String(255), nullable=False)
    target_column = Column(String(255), nullable=False)
    cardinality = Column(String(50), default="many-to-one")
    # These are measured properties, not optimistic assumptions. Unknown until
    # a join-safety computation establishes them.
    join_safety_score = Column(Float, nullable=True)
    fanout_risk = Column(Boolean, nullable=True)
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime, default=utc_now)


class SemanticTimeDefinition(Base):
    """Temporal analysis dimension and grain definitions."""
    __tablename__ = "semantic_time_definitions"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    semantic_model_id = Column(String(36), ForeignKey("semantic_models.id"), nullable=False, index=True)
    table_name = Column(String(255), nullable=False)
    column_name = Column(String(255), nullable=False)
    grain = Column(String(50), default="daily")
    min_date = Column(String(50), nullable=True)
    max_date = Column(String(50), nullable=True)
    is_continuous = Column(Boolean, default=True)
    gap_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=utc_now)


class SemanticBusinessRule(Base):
    """Business integrity rules and data constraints."""
    __tablename__ = "semantic_business_rules"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    semantic_model_id = Column(String(36), ForeignKey("semantic_models.id"), nullable=False, index=True)
    table_name = Column(String(255), nullable=False)
    column_name = Column(String(255), nullable=False)
    rule_type = Column(String(100), default="non_null")
    expression = Column(Text, nullable=False)
    severity = Column(String(50), default="high")
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now)


# Legacy Aliases for Backward Compatibility with earlier metric/glossary APIs
class BusinessMetric(Base):
    __tablename__ = "business_metrics"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    display_name = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    sql_formula = Column(Text, nullable=False)
    unit = Column(String(50), nullable=True)
    category = Column(String(100), default="financial")
    dimensions_json = Column(JSON, default=list)
    filters = Column(Text, nullable=True)
    synonyms_json = Column(JSON, default=list)
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)


class BusinessGlossary(Base):
    __tablename__ = "business_glossary"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False, index=True)
    term = Column(String(255), nullable=False)
    definition = Column(Text, nullable=False)
    category = Column(String(100), default="general")
    synonyms_json = Column(JSON, default=list)
    related_metrics_json = Column(JSON, default=list)
    related_columns_json = Column(JSON, default=list)
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)


# ============================================================================
# 4. CANONICAL INVESTIGATION & EPISTEMIC GRAPH DOMAIN
# ============================================================================

class Investigation(Base):
    """Canonical Investigation entity orchestrating the full epistemic reasoning DAG."""
    __tablename__ = "investigations"
    id = Column(String(64), primary_key=True)  # e.g. INV-20260822-001 or UUID
    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False, index=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True, index=True)
    semantic_model_id = Column(String(36), ForeignKey("semantic_models.id"), nullable=True, index=True)
    question = Column(Text, nullable=False)
    status = Column(String(50), default="PLANNED", index=True)  # PLANNED, RUNNING, COMPLETED, FAILED
    verdict_type = Column(String(50), default="INCONCLUSIVE")  # OBSERVED, STATISTICALLY_SIGNIFICANT, DIAGNOSED, PREDICTED, SIMULATED, CAUSALLY_SUPPORTED, INCONCLUSIVE
    confidence_score = Column(Float, default=0.0)
    direct_answer = Column(Text, nullable=True)
    main_finding = Column(Text, nullable=True)
    entropy_initial = Column(Float, default=1.0)
    entropy_current = Column(Float, default=1.0)
    stopping_criteria_met = Column(Boolean, default=False)
    stopping_rationale = Column(Text, nullable=True)
    reproducible_manifest_hash = Column(String(128), nullable=True)
    dataset_version_ids_json = Column(JSON, default=list)
    # DEFECT-003 fix: the actual request-scoped dataset selection, persisted
    # at Investigation-creation time so the controller can enforce it later
    # in the same process, after a worker restart, or when replayed. NULL/
    # empty means "no explicit selection was made" (all project datasets
    # remain in scope) -- distinct from an empty list meaning "selected zero
    # datasets", which the API layer is responsible for rejecting before an
    # Investigation row is ever created.
    requested_dataset_ids_json = Column(JSON, nullable=True)
    parent_investigation_id = Column(String(64), ForeignKey("investigations.id"), nullable=True, index=True)
    is_subinvestigation = Column(Boolean, nullable=False, default=False, index=True)
    active_contract_id = Column(String(36), ForeignKey("investigation_contracts.id"), nullable=True, index=True)
    contract_revision = Column(Integer, nullable=False, default=0)
    current_phase = Column(String(64), nullable=True)
    execution_time_seconds = Column(Float, default=0.0)
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)



class InvestigationContract(Base):
    """Versioned, durable analytical contract compiled from the user's question.

    This is the authoritative intent/estimand/claim contract consumed by the
    investigation runtime.  A new row is created whenever the analytical plan
    materially changes, preserving a complete re-planning history.
    """
    __tablename__ = "investigation_contracts"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    investigation_id = Column(String(64), ForeignKey("investigations.id"), nullable=False, index=True)
    version = Column(Integer, nullable=False, default=1)
    parent_contract_id = Column(String(36), ForeignKey("investigation_contracts.id"), nullable=True, index=True)
    status = Column(String(40), nullable=False, default="ACTIVE")
    original_question = Column(Text, nullable=False)
    normalized_question = Column(Text, nullable=True)
    problem_class = Column(String(50), nullable=False)
    claim_type = Column(String(50), nullable=False)
    target_json = Column(JSON, nullable=True)
    explanatory_variables_json = Column(JSON, default=list)
    population_json = Column(JSON, nullable=True)
    grain_json = Column(JSON, nullable=True)
    scope_json = Column(JSON, nullable=True)
    time_window_json = Column(JSON, nullable=True)
    estimand_json = Column(JSON, nullable=True)
    assumptions_json = Column(JSON, default=list)
    data_requirements_json = Column(JSON, default=list)
    candidate_methods_json = Column(JSON, default=list)
    # v20-C4.2.3 authority model.  ``proposed_method_json`` is the compiler's
    # PROPOSAL (diagnostic only, never executed against).  ``selected_method_json``
    # is written ONLY by contract_authority.finalize_contract(), as a projection of
    # ``final_contract_json`` -- it is NULL until the contract is finalized, so a
    # proposal can never masquerade as the selected method.
    proposed_method_json = Column(JSON, nullable=True)
    selected_method_json = Column(JSON, nullable=True)
    # THE durable analytical authority: a serialized FinalAnalyticalContract
    # (packages/analytics_core/src/intelligence/analytical_identity.py).
    final_contract_json = Column(JSON, nullable=True)
    analytical_identity = Column(String(64), nullable=True, index=True)
    finalized_at = Column(DateTime, nullable=True)
    evidence_requirements_json = Column(JSON, default=list)
    stopping_criteria_json = Column(JSON, default=dict)
    ambiguity_state_json = Column(JSON, nullable=True)
    semantic_interpretations_json = Column(JSON, default=list)
    unresolved_questions_json = Column(JSON, default=list)
    confidence = Column(Float, nullable=True)
    current_phase = Column(String(64), nullable=True)
    # v20-A: canonical SemanticBindingSet for this contract version, as
    # SemanticBinding.to_dict() entries (see
    # packages/schemas/src/semantic_binding.py). Populated at contract-
    # compile time; the persisted contract and the in-memory runtime
    # contract must carry identical bindings.
    semantic_bindings_json = Column(JSON, default=list)
    execution_state_json = Column(JSON, default=dict)
    last_replan_reason = Column(Text, nullable=True)
    last_replanned_at = Column(DateTime, nullable=True)
    superseded_at = Column(DateTime, nullable=True)
    compiled_at = Column(DateTime, default=utc_now)

    # v28 SemanticBindingSet projection authority:
    # All legacy target, explanatory, time, and grouping fields are compatibility
    # projections derived from semantic_bindings_json, rather than competing truths.
    @property
    def semantic_binding_set(self) -> Any:
        from packages.schemas.src.semantic_binding import SemanticBindingSet
        return SemanticBindingSet.from_dict_list(self.semantic_bindings_json or [])

    @property
    def target_column(self) -> Optional[str]:
        proj = self.semantic_binding_set.projected_target_column()
        if proj is not None:
            return proj
        if isinstance(self.target_json, dict):
            return self.target_json.get("target")
        return None

    @property
    def explanatory_columns(self) -> list[str]:
        proj = self.semantic_binding_set.projected_explanatory_columns()
        if proj:
            return proj
        return list(self.explanatory_variables_json or [])

    @property
    def time_column(self) -> Optional[str]:
        proj = self.semantic_binding_set.projected_time_column()
        if proj is not None:
            return proj
        if isinstance(self.time_window_json, dict):
            return self.time_window_json.get("time_column")
        return None

    @property
    def group_dimension(self) -> Optional[str]:
        return self.semantic_binding_set.projected_group_dimension()



class InvestigationDataReadiness(Base):
    """Durable pre-analysis data-readiness assessment tied to a contract version."""
    __tablename__ = "investigation_data_readiness"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    investigation_id = Column(String(64), ForeignKey("investigations.id"), nullable=False, index=True)
    contract_id = Column(String(36), ForeignKey("investigation_contracts.id"), nullable=True, index=True)
    dataset_name = Column(String(255), nullable=False)
    dataset_version_ids_json = Column(JSON, default=list)
    source_fingerprints_json = Column(JSON, default=dict)
    row_count = Column(Integer, default=0)
    column_count = Column(Integer, default=0)
    overall_quality_score = Column(Float, nullable=True)
    fitness_verdict = Column(String(30), nullable=False)
    can_proceed = Column(Boolean, default=False)
    checks_json = Column(JSON, default=dict)
    critical_issues_json = Column(JSON, default=list)
    warnings_json = Column(JSON, default=list)
    recommendations_json = Column(JSON, default=list)
    assessed_at = Column(DateTime, default=utc_now)

class InvestigationObjective(Base):
    """Decomposed analytical objective within an investigation."""
    __tablename__ = "investigation_objectives"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    investigation_id = Column(String(64), ForeignKey("investigations.id"), nullable=False, index=True)
    statement = Column(Text, nullable=False)
    target_variable = Column(String(255), nullable=True)
    priority_rank = Column(Integer, default=1)
    status = Column(String(50), default="active")
    created_at = Column(DateTime, default=utc_now)


class InvestigationUnknown(Base):
    """Unknown target variable or causal factor under active inquiry."""
    __tablename__ = "investigation_unknowns"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    investigation_id = Column(String(64), ForeignKey("investigations.id"), nullable=False, index=True)
    variable_name = Column(String(255), nullable=False)
    target_metric = Column(String(255), nullable=True)
    dimension_scope = Column(String(255), nullable=True)
    prior_estimate = Column(String(255), nullable=True)
    posterior_estimate = Column(String(255), nullable=True)
    uncertainty_range = Column(String(255), nullable=True)
    is_resolved = Column(Boolean, default=False)
    created_at = Column(DateTime, default=utc_now)


class Hypothesis(Base):
    """Falsifiable explanatory proposition in the investigation graph."""
    __tablename__ = "hypotheses"
    __table_args__ = (UniqueConstraint("investigation_id", "canonical_identity", name="uq_hypotheses_investigation_identity"),)
    id = Column(String(64), primary_key=True, default=gen_uuid)
    investigation_id = Column(String(64), ForeignKey("investigations.id"), nullable=False, index=True)
    canonical_identity = Column(String(64), nullable=False, index=True)
    # v20-C4.2.3: deterministic analytical-pair identity (FinalAnalyticalContract.
    # pair_identity) -- NOT the human label in hypothesis_code.
    analytical_identity = Column(String(64), nullable=True, index=True)
    hypothesis_code = Column(String(50), default="HYP-01")
    statement = Column(Text, nullable=False)
    rationale = Column(Text, nullable=True)
    prior_probability = Column(Float, default=0.5)
    posterior_probability = Column(Float, default=0.5)
    belief_state = Column(String(50), default="untested")  # untested, weakened, supported, highly_likely, refuted
    status = Column(String(50), default="proposed")  # proposed, active, supported, weakened, refuted, retired
    supporting_prediction_count = Column(Integer, default=0)
    refuted_prediction_count = Column(Integer, default=0)
    unresolved_prediction_count = Column(Integer, default=0)
    target_metric = Column(String(100), nullable=True)
    target_dimension = Column(String(100), nullable=True)
    target_value = Column(String(255), nullable=True)
    mechanism_detail = Column(Text, nullable=True)
    generated_reason = Column(Text, nullable=True)
    source_evidence_json = Column(JSON, default=list)
    parent_hypotheses_json = Column(JSON, default=list)
    is_counter_hypothesis = Column(Boolean, default=False)
    last_evidence_likelihood = Column(Float, nullable=True)
    reason_for_rejection = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)


@event.listens_for(Hypothesis, "before_insert")
def _populate_hypothesis_canonical_identity(mapper, connection, target):
    """Guarantee ORM-created hypotheses carry the same canonical identity used by the runtime."""
    if not getattr(target, "canonical_identity", None):
        from packages.analytics_core.src.intelligence.hypothesis_identity import compute_semantic_identity
        target.canonical_identity = compute_semantic_identity(target)


class Prediction(Base):
    """Falsifiable empirical prediction deduced from a hypothesis."""
    __tablename__ = "predictions"
    id = Column(String(64), primary_key=True, default=gen_uuid)
    investigation_id = Column(String(64), ForeignKey("investigations.id"), nullable=True, index=True)
    hypothesis_id = Column(String(64), ForeignKey("hypotheses.id"), nullable=False, index=True)
    prediction_code = Column(String(50), default="PRED-01")
    statement = Column(Text, nullable=False)
    target_metric = Column(String(100), nullable=True)
    target_dimension = Column(String(100), nullable=True)
    target_segment = Column(String(100), nullable=True)
    expected_direction = Column(String(50), default="decrease")  # increase, decrease, correlate, invariant, concentrated
    expected_value = Column(String(100), nullable=True)
    threshold = Column(Float, nullable=True)
    expected_magnitude = Column(String(100), nullable=True)
    expected_relationship = Column(String(255), nullable=True)
    expected_effect = Column(String(100), nullable=True)
    # No epistemic confidence is implied merely by creating a Prediction;
    # it is never synthesized from templates or priors (see
    # StructuredPrediction in intelligence/prediction_engine.py).
    confidence = Column(Float, nullable=True)
    testability = Column(Boolean, default=True)
    falsification_condition = Column(Text, nullable=True)
    status = Column(String(50), default="PENDING", index=True)  # PENDING, SUPPORTED, REFUTED, INCONCLUSIVE, NOT_TESTABLE
    actual_observed_result_json = Column(JSON, default=dict)
    evaluation_reason = Column(Text, nullable=True)
    target_experiment_id = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=utc_now)
    evaluated_at = Column(DateTime, nullable=True)


class Experiment(Base):
    """Planned or executed analytical test designed to test predictions and discriminate hypotheses."""
    __tablename__ = "experiments"
    __table_args__ = (
        Index(
            "uq_experiments_primary_identity",
            "investigation_id",
            "analytical_identity",
            unique=True,
            sqlite_where=text("experiment_role = 'PRIMARY'"),
            postgresql_where=text("experiment_role = 'PRIMARY'"),
        ),
    )
    id = Column(String(64), primary_key=True, default=gen_uuid)
    investigation_id = Column(String(64), ForeignKey("investigations.id"), nullable=False, index=True)
    hypothesis_id = Column(String(64), ForeignKey("hypotheses.id"), nullable=True, index=True)
    target_prediction_id = Column(String(64), nullable=True, index=True)
    test_code = Column(String(50), default="TEST-01")
    # v20-C4.2.3: durable identity/role.  test_code and hypothesis_id are labels;
    # these columns are the identity.
    experiment_role = Column(String(20), nullable=True, index=True)
    analytical_identity = Column(String(64), nullable=True, index=True)
    hypothesis_canonical_identity = Column(String(64), nullable=True, index=True)
    contract_id = Column(String(36), ForeignKey("investigation_contracts.id"), nullable=True, index=True)
    tool_name = Column(String(100), nullable=False)
    arguments_json = Column(JSON, default=dict)
    rationale = Column(Text, nullable=True)
    expected_information_gain = Column(Float, nullable=True, default=None)
    adversarial_value = Column(Float, default=0.0)
    robustness_value = Column(Float, default=0.0)
    causal_value = Column(Float, default=0.0)
    selection_rationale = Column(Text, nullable=True)
    fingerprint = Column(String(64), nullable=True, index=True)
    replication_of = Column(String(64), nullable=True)
    refinement_of = Column(String(64), nullable=True)
    test_cost = Column(Float, default=1.0)
    test_reliability = Column(Float, default=0.95)
    utility_score = Column(Float, default=0.0)
    # Explicit scientific coverage metadata. A non-full-scope experiment cannot
    # silently inherit full-population evidentiary strength.
    analysis_row_count = Column(Integer, nullable=False, default=0)
    full_scope = Column(Boolean, nullable=False, default=True)
    sampling_policy = Column(String(50), nullable=False, default="NONE")
    sampling_reason = Column(Text, nullable=True)
    sampling_method = Column(String(100), nullable=True)
    status = Column(String(50), default="PLANNED")  # PLANNED, RUNNING, EXECUTED, SKIPPED, FAILED
    created_at = Column(DateTime, default=utc_now)
    executed_at = Column(DateTime, nullable=True)


class Observation(Base):
    """Raw empirical observation produced by executing an analytical experiment."""
    __tablename__ = "observations"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    experiment_id = Column(String(64), ForeignKey("experiments.id"), nullable=False, index=True)
    result_json = Column(JSON, default=dict)
    row_count_analyzed = Column(Integer, default=0)
    execution_time_ms = Column(Float, default=0.0)
    sql_executed = Column(Text, nullable=True)
    raw_data_hash = Column(String(128), nullable=True)
    created_at = Column(DateTime, default=utc_now)


class Evidence(Base):
    """Verified empirical claim synthesized from observations.

    ``evidence_identity_hash`` is the canonical scientific computation identity.
    It is unique within an investigation so duplicate computations can never
    create a second Bayesian evidence contribution.
    """
    __tablename__ = "evidence"
    __table_args__ = (
        UniqueConstraint("investigation_id", "evidence_identity_hash", name="uq_evidence_investigation_identity"),
    )
    id = Column(String(64), primary_key=True, default=gen_uuid)
    investigation_id = Column(String(64), ForeignKey("investigations.id"), nullable=False, index=True)
    experiment_id = Column(String(64), ForeignKey("experiments.id"), nullable=True, index=True)
    hypothesis_id = Column(String(64), ForeignKey("hypotheses.id"), nullable=True, index=True)
    statement = Column(Text, nullable=False)
    evidence_type = Column(String(50), default="DIRECT_MEASUREMENT")
    # P0 epistemic-integrity fix: this used to default to 0.95, meaning any
    # code path that constructed an Evidence row without explicitly
    # computing a confidence score (e.g. scripts/migrate_analysis_runs_to_
    # investigations.py, which migrates legacy AnalysisRun evidence with no
    # calibrated confidence available) silently fabricated a high-confidence
    # value. The column is already nullable at the DB level (see
    # migrations/versions/001_canonical_aaos_domain_models.py), so None is
    # a safe, honest default: "no confidence has been calculated yet."
    confidence_score = Column(Float, nullable=True, default=None)
    calculation_summary = Column(Text, nullable=True)
    statistical_test_name = Column(String(100), nullable=True)
    p_value = Column(Float, nullable=True)
    effect_size = Column(Float, nullable=True)
    validation_status = Column(String(50), default="unverified")  # verified, partial, unverified, failed
    evidence_identity_hash = Column(String(64), nullable=True, index=True)
    # v20-C4.2.3: inherited from the producing experiment at write time.
    analytical_identity = Column(String(64), nullable=True, index=True)
    relative_tolerance_observed = Column(Float, nullable=True)
    created_at = Column(DateTime, default=utc_now)


class EvidenceVerification(Base):
    """Independent secondary validation audit record for an evidence claim."""
    __tablename__ = "evidence_verifications"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    evidence_id = Column(String(64), ForeignKey("evidence.id"), nullable=False, index=True)
    primary_tool = Column(String(100), nullable=False)
    secondary_tool = Column(String(100), nullable=False)
    tolerance_threshold = Column(Float, default=0.01)
    # A missing verification delta means verification did not produce a
    # comparable secondary value. Zero is a real measurement and must never
    # be used as an implicit default.
    observed_delta_pct = Column(Float, nullable=True)
    is_deterministic = Column(Boolean, default=True)
    validator_fingerprint = Column(String(128), nullable=True)
    status = Column(String(50), default="UNVERIFIED")  # PASSED, FAILED, WARNING, UNVERIFIED
    created_at = Column(DateTime, default=utc_now)


class BeliefUpdate(Base):
    """Step in Bayesian posterior probability update trajectory."""
    __tablename__ = "belief_updates"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    investigation_id = Column(String(64), ForeignKey("investigations.id"), nullable=False, index=True)
    hypothesis_id = Column(String(64), ForeignKey("hypotheses.id"), nullable=False, index=True)
    evidence_id = Column(String(64), ForeignKey("evidence.id"), nullable=True, index=True)
    prior_probability = Column(Float, nullable=False)
    # Legacy field retained only for old persisted rows. It is never used to
    # store a Bayes factor. New evidence uses bayes_factor below.
    likelihood_p = Column(Float, nullable=True)
    bayes_factor = Column(Float, nullable=True)
    posterior_probability = Column(Float, nullable=False)
    entropy_delta = Column(Float, default=0.0)
    update_step_index = Column(Integer, default=1)
    created_at = Column(DateTime, default=utc_now)


class CounterHypothesis(Base):
    """Adversarial counter-hypothesis link evaluating competing explanations."""
    __tablename__ = "counter_hypotheses"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    primary_hypothesis_id = Column(String(64), ForeignKey("hypotheses.id"), nullable=False, index=True)
    counter_hypothesis_id = Column(String(64), ForeignKey("hypotheses.id"), nullable=False, index=True)
    relation_type = Column(String(50), default="MUTUALLY_EXCLUSIVE")  # MUTUALLY_EXCLUSIVE, COMPETING, MEDIATING
    discrimination_test_id = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=utc_now)


class Assumption(Base):
    """Explicit domain and statistical assumptions underpinning an investigation."""
    __tablename__ = "assumptions"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    investigation_id = Column(String(64), ForeignKey("investigations.id"), nullable=False, index=True)
    statement = Column(Text, nullable=False)
    is_validated = Column(Boolean, default=False)
    sensitivity_risk = Column(String(50), default="medium")  # low, medium, high
    created_at = Column(DateTime, default=utc_now)


class ClaimGateDecision(Base):
    """Immutable release-facing claim-admissibility decision for an analysis."""
    __tablename__ = "claim_gate_decisions"
    id = Column(String(64), primary_key=True, default=gen_uuid)
    investigation_id = Column(String(64), ForeignKey("investigations.id"), nullable=False, index=True)
    question_id = Column(String(64), nullable=True, index=True)
    requested_claim = Column(String(50), nullable=False)
    requested_level = Column(Integer, nullable=False)
    evidence_level = Column(Integer, nullable=False)
    max_supported_level = Column(Integer, nullable=False)
    outcome = Column(String(30), nullable=False, index=True)
    design_status = Column(String(30), nullable=False)
    identification_strategy = Column(String(255), nullable=True)
    assumptions_json = Column(JSON, default=list)
    blocking_conditions_json = Column(JSON, default=list)
    missing_evidence_json = Column(JSON, default=list)
    recovery_actions_json = Column(JSON, default=list)
    allowed_claim = Column(Text, nullable=False)
    blocked_claim = Column(Text, nullable=True)
    reason = Column(Text, nullable=False)
    computed_evidence_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=utc_now, index=True)


class InvestigationVerdict(Base):
    """Final earned epistemic verdict with complete mathematical audit trail."""
    __tablename__ = "investigation_verdicts"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    investigation_id = Column(String(64), ForeignKey("investigations.id"), unique=True, nullable=False, index=True)
    verdict_type = Column(String(50), nullable=False)
    confidence_score = Column(Float, default=0.0)
    justification = Column(Text, nullable=False)
    net_variance_explained_pct = Column(Float, default=0.0)
    counter_hypothesis_refuted = Column(Boolean, default=False)
    epistemic_grade = Column(String(20), default="NOT_ASSESSED")
    created_at = Column(DateTime, default=utc_now)


class DecisionRecommendationRecord(Base):
    """Persisted grounded business decision recommendation with its quantified
    expected-utility calculation (see `_measurement_confidence` / the
    grounded-utility computation in `controller.py`). Named *Record* to avoid
    colliding with `packages.schemas.src.analysis.DecisionRecommendation`,
    the pydantic schema the controller builds before persisting here.

    Previously this was built by the controller and immediately discarded --
    never persisted, never reaching the API -- see DEFECT-009 in
    CURRENT_DEFECT_REGISTER.md. This table, plus the corresponding query in
    the `/investigations/{id}` route, is the fix."""
    __tablename__ = "decision_recommendations"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    investigation_id = Column(String(64), ForeignKey("investigations.id"), nullable=False, index=True)
    recommendation_id = Column(String(64), nullable=False)
    action_title = Column(Text, nullable=False)
    action_description = Column(Text, nullable=False)
    grounded_hypothesis_id = Column(String(128), nullable=False)
    target_metric = Column(String(255), nullable=False)
    expected_gain_metric = Column(Float, default=0.0)
    downside_risk_metric = Column(Float, default=0.0)
    probability_of_success = Column(Float, default=0.0)
    net_expected_utility = Column(Float, default=0.0)
    utility_function_description = Column(Text, nullable=True)
    policy_compliance_passed = Column(Boolean, default=True)
    required_preconditions_json = Column(JSON, default=list)
    created_at = Column(DateTime, default=utc_now)


class InvestigationGraphEdge(Base):
    """Explicit directed graph edge linking investigation entities for DAG reconstruction."""
    __tablename__ = "investigation_graph_edges"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    investigation_id = Column(String(64), ForeignKey("investigations.id"), nullable=False, index=True)
    source_node_type = Column(String(50), nullable=False)  # OBJECTIVE, UNKNOWN, HYPOTHESIS, PREDICTION, EXPERIMENT, OBSERVATION, EVIDENCE, VERDICT
    source_node_id = Column(String(64), nullable=False, index=True)
    target_node_type = Column(String(50), nullable=False)
    target_node_id = Column(String(64), nullable=False, index=True)
    relationship_type = Column(String(50), nullable=False)  # DECOMPOSES_TO, PROPOSES, PREDICTS, TESTS, PRODUCES, SUPPORTS, REFUTES, UPDATES, CONCLUDES
    created_at = Column(DateTime, default=utc_now)


# ============================================================================
# 4B. DURABLE EXECUTION, LEASING, & EVENT STREAMING DOMAIN
# ============================================================================

class InvestigationJob(Base):
    """Durable queue job record for background investigation workers."""
    __tablename__ = "investigation_jobs"
    id = Column(String(64), primary_key=True, default=lambda: f"JOB-{gen_uuid()[:8]}")
    investigation_id = Column(String(64), ForeignKey("investigations.id"), nullable=False, index=True)
    priority = Column(Integer, default=10, index=True)
    status = Column(String(50), default="QUEUED", index=True)  # QUEUED, CLAIMED, RUNNING, WAITING_FOR_USER, WAITING_FOR_RESOURCE, COMPLETED, FAILED, CANCELLED, CANCEL_REQUESTED
    worker_id = Column(String(64), nullable=True, index=True)
    lease_expires_at = Column(DateTime, nullable=True, index=True)
    heartbeat_at = Column(DateTime, nullable=True)
    attempt = Column(Integer, default=1)
    max_attempts = Column(Integer, default=3)
    error_code = Column(String(100), nullable=True)  # From FailureTaxonomy
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)


class InvestigationExecution(Base):
    """Runtime execution instance tracking active worker, lease, attempt, and compute resource."""
    __tablename__ = "investigation_executions"
    id = Column(String(64), primary_key=True, default=lambda: f"EXEC-{gen_uuid()[:8]}")
    investigation_id = Column(String(64), ForeignKey("investigations.id"), nullable=False, index=True)
    worker_id = Column(String(64), nullable=False, index=True)
    status = Column(String(50), default="RUNNING", index=True)  # CREATED, RUNNING, CHECKPOINTED, COMPLETED, FAILED, ABORTED
    attempt = Column(Integer, default=1)
    current_step_index = Column(Integer, default=0)
    resource_id = Column(String(64), default="local_cpu")
    started_at = Column(DateTime, default=utc_now)
    heartbeat_at = Column(DateTime, default=utc_now)
    lease_expires_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_code = Column(String(100), nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now)


class InvestigationStepExecution(Base):
    """Step-level execution record guaranteeing idempotency and restart resumption."""
    __tablename__ = "investigation_step_executions"
    id = Column(String(64), primary_key=True, default=lambda: f"STEP-{gen_uuid()[:8]}")
    execution_id = Column(String(64), ForeignKey("investigation_executions.id"), nullable=False, index=True)
    investigation_id = Column(String(64), ForeignKey("investigations.id"), nullable=False, index=True)
    step_index = Column(Integer, nullable=False)
    step_type = Column(String(50), nullable=False)  # INTENT, SEMANTIC_RESOLVE, HYPOTHESIS_GEN, EXPERIMENT_EXEC, EVIDENCE_VERIFY, BELIEF_UPDATE, VERDICT_FORM
    idempotency_key = Column(String(128), unique=True, nullable=False, index=True)
    status = Column(String(50), default="COMPLETED")  # RUNNING, COMPLETED, FAILED, SKIPPED
    input_hash = Column(String(128), nullable=True)
    output_hash = Column(String(128), nullable=True)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, default=utc_now)
    completed_at = Column(DateTime, default=utc_now)


class InvestigationDecisionRequest(Base):
    """Human-in-the-loop decision request suspending investigation until user responds."""
    __tablename__ = "investigation_decision_requests"
    id = Column(String(64), primary_key=True, default=lambda: f"DEC-{gen_uuid()[:8]}")
    investigation_id = Column(String(64), ForeignKey("investigations.id"), nullable=False, index=True)
    execution_id = Column(String(64), ForeignKey("investigation_executions.id"), nullable=True, index=True)
    question = Column(Text, nullable=False)
    options_json = Column(JSON, default=list)
    default_action = Column(String(100), nullable=True)
    status = Column(String(50), default="PENDING", index=True)  # PENDING, RESOLVED, TIMED_OUT, CANCELLED
    user_response = Column(Text, nullable=True)
    user_id = Column(String(36), nullable=True)
    created_at = Column(DateTime, default=utc_now)
    resolved_at = Column(DateTime, nullable=True)


class InvestigationResourceWait(Base):
    """Resource allocation wait record suspending investigation until required compute is attached."""
    __tablename__ = "investigation_resource_waits"
    id = Column(String(64), primary_key=True, default=lambda: f"RES-{gen_uuid()[:8]}")
    investigation_id = Column(String(64), ForeignKey("investigations.id"), nullable=False, index=True)
    execution_id = Column(String(64), ForeignKey("investigation_executions.id"), nullable=True, index=True)
    resource_id = Column(String(100), nullable=False)
    reason = Column(Text, nullable=False)
    required_capabilities_json = Column(JSON, default=dict)
    status = Column(String(50), default="PENDING", index=True)  # PENDING, SATISFIED, TIMED_OUT, REJECTED
    requested_at = Column(DateTime, default=utc_now)
    satisfied_at = Column(DateTime, nullable=True)


class InvestigationEvent(Base):
    """Immutable persistent audit and SSE event stream log with monotonic sequence cursor."""
    __tablename__ = "investigation_events"
    __table_args__ = (UniqueConstraint("investigation_id", "sequence", name="uq_investigation_event_seq"),)

    id = Column(String(64), primary_key=True, default=lambda: f"EVT-{gen_uuid()[:8]}")
    sequence = Column(Integer, primary_key=False, nullable=True, index=True)
    investigation_id = Column(String(64), ForeignKey("investigations.id"), nullable=False, index=True)
    execution_id = Column(String(64), nullable=True, index=True)
    event_type = Column(String(100), nullable=False, index=True)
    event_payload_json = Column(JSON, default=dict)
    timestamp = Column(DateTime, default=utc_now, index=True)


# Backward-compatible Projection Model for legacy AnalysisRun queries
class AnalysisRun(Base):
    __tablename__ = "analysis_runs"
    id = Column(String(64), primary_key=True)
    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False, index=True)
    question = Column(Text, nullable=False)
    status = Column(String(50), default="COMPLETED")
    direct_answer = Column(Text, nullable=True)
    main_finding = Column(Text, nullable=True)
    # No epistemic confidence is implied merely by creating an AnalysisRun.
    confidence = Column(String(50), nullable=True)
    hypotheses_json = Column(JSON, default=list)
    steps_json = Column(JSON, default=list)
    findings_json = Column(JSON, default=list)
    evidence_json = Column(JSON, default=list)
    manifest_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)


# ============================================================================
# 5. DASHBOARDS, REPORTS, ALERTS, & CHAT DOMAIN
# ============================================================================

class Dashboard(Base):
    __tablename__ = "dashboards"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    widgets_json = Column(JSON, default=list)
    filters_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)


class Report(Base):
    __tablename__ = "reports"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False, index=True)
    analysis_id = Column(String(64), nullable=True)
    title = Column(String(255), nullable=False)
    report_type = Column(String(50), default="executive")
    executive_summary_json = Column(JSON, nullable=True)
    technical_report_json = Column(JSON, nullable=True)
    markdown_content = Column(Text, nullable=False)
    pdf_artifact_path = Column(String(512), nullable=True)
    created_at = Column(DateTime, default=utc_now)


class AlertRule(Base):
    __tablename__ = "alert_rules"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    metric_name = Column(String(255), nullable=False)
    dataset_id = Column(String(36), nullable=False)
    threshold_pct_change = Column(Float, default=15.0)
    frequency = Column(String(50), default="daily")
    severity = Column(String(50), default="high")
    auto_investigate = Column(Boolean, default=True)
    is_active = Column(Boolean, default=True)
    last_evaluated_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=utc_now)


class AlertEvent(Base):
    __tablename__ = "alert_events"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    project_id = Column(String(36), nullable=False, index=True)
    rule_id = Column(String(36), ForeignKey("alert_rules.id"), nullable=False)
    title = Column(String(255), nullable=False)
    severity = Column(String(50), default="high")
    description = Column(Text, nullable=False)
    current_value = Column(Float, nullable=False)
    expected_value = Column(Float, nullable=False)
    deviation_percentage = Column(Float, nullable=False)
    affected_dimensions_json = Column(JSON, default=dict)
    investigation_analysis_id = Column(String(64), nullable=True)
    evidence_summary = Column(Text, nullable=True)
    is_resolved = Column(Boolean, default=False)
    triggered_at = Column(DateTime, default=utc_now)


class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)


class Message(Base):
    __tablename__ = "messages"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    conversation_id = Column(String(36), ForeignKey("conversations.id"), nullable=False, index=True)
    sender_role = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    analysis_id = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=utc_now)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    project_id = Column(String(36), nullable=True, index=True)
    user_id = Column(String(36), nullable=True)
    action = Column(String(255), nullable=False)
    details_json = Column(JSON, default=dict)
    ip_address = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=utc_now)