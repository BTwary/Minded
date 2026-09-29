"""Semantic Graph Schemas: Deterministic ground-truth models of relational structures and semantics."""
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SemanticRole(str, Enum):
    PRIMARY_KEY = "PRIMARY_KEY"
    FOREIGN_KEY = "FOREIGN_KEY"
    DIMENSION = "DIMENSION"
    MEASURE = "MEASURE"
    TIMESTAMP = "TIMESTAMP"
    ATTRIBUTE = "ATTRIBUTE"


class CardinalityType(str, Enum):
    ONE_TO_ONE = "ONE_TO_ONE"
    ONE_TO_MANY = "ONE_TO_MANY"
    MANY_TO_ONE = "MANY_TO_ONE"
    MANY_TO_MANY = "MANY_TO_MANY"
    one_to_one = "one_to_one"
    one_to_many = "one_to_many"
    many_to_one = "many_to_one"
    many_to_many = "many_to_many"


class MetricAdditivity(str, Enum):
    UNKNOWN = "UNKNOWN"
    FULLY_ADDITIVE = "FULLY_ADDITIVE"
    SEMI_ADDITIVE = "SEMI_ADDITIVE"
    NON_ADDITIVE = "NON_ADDITIVE"


class MetricTargetDirection(str, Enum):
    INCREASING = "INCREASING"
    DECREASING = "DECREASING"
    NEUTRAL = "NEUTRAL"
    MAXIMIZE = "MAXIMIZE"
    MINIMIZE = "MINIMIZE"


class FunctionalDependency(BaseModel):
    """Formal functional dependency X -> Y discovered or declared in dataset."""
    determinant: List[str] = Field(min_length=1)
    dependent: List[str] = Field(min_length=1)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class SemanticColumnSchema(BaseModel):
    """Column definition with physical and semantic bindings."""
    name: str
    data_type: str
    is_nullable: bool = False
    semantic_role: SemanticRole = SemanticRole.ATTRIBUTE
    distinct_values_count: Optional[int] = None
    null_percentage: float = Field(default=0.0, ge=0.0, le=100.0)


class SemanticEntitySchema(BaseModel):
    """Table/Entity definition with verified grain and candidate keys."""
    name: str
    table_name: str
    verified_grain_keys: List[str] = Field(default_factory=list)
    row_count: int = Field(default=0, ge=0)
    columns: List[SemanticColumnSchema] = Field(default_factory=list)
    candidate_keys: List[List[str]] = Field(default_factory=list)


class EpistemicSource(str, Enum):
    PHYSICAL_FACT = "PHYSICAL_FACT"
    INFERRED_SEMANTICS = "INFERRED_SEMANTICS"
    EXTERNAL_DECLARATION = "EXTERNAL_DECLARATION"
    USER_CONFIRMATION = "USER_CONFIRMATION"
    MODEL_HYPOTHESIS = "MODEL_HYPOTHESIS"


# EntityNode compatibility model for world_model builder
class EntityNode(BaseModel):
    entity_name: str
    primary_key: str
    table_name: str
    natural_keys: List[str] = Field(default_factory=list)
    attributes: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    epistemic_source: EpistemicSource = EpistemicSource.INFERRED_SEMANTICS
    evidence_statement: Optional[str] = None
    grain_proven: bool = False


class RelationshipEdge(BaseModel):
    source_table: str
    target_table: str
    source_column: str
    target_column: str
    cardinality: Optional[CardinalityType] = None
    join_safety_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    fanout_risk: Optional[bool] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    epistemic_source: EpistemicSource = EpistemicSource.INFERRED_SEMANTICS
    evidence_statement: Optional[str] = None
    overlap_count: int = 0
    overlap_ratio: float = 0.0


class MetricNode(BaseModel):
    metric_name: str
    table_name: str
    column_name: Optional[str] = None
    additivity: MetricAdditivity = MetricAdditivity.UNKNOWN
    unit: str = "UNKNOWN"
    target_direction: MetricTargetDirection = MetricTargetDirection.MAXIMIZE
    description: Optional[str] = None
    epistemic_source: EpistemicSource = EpistemicSource.INFERRED_SEMANTICS
    evidence_statement: Optional[str] = None
    aggregation_type: str = "UNKNOWN"
    numerator_column: Optional[str] = None
    denominator_column: Optional[str] = None
    weight_column: Optional[str] = None


class TimeDimensionNode(BaseModel):
    column_name: str
    table_name: str
    lowest_grain: str = "day"
    min_date: Optional[str] = None
    max_date: Optional[str] = None
    epistemic_source: EpistemicSource = EpistemicSource.PHYSICAL_FACT


class BusinessRuleConstraint(BaseModel):
    table_name: str
    column_name: Optional[str] = None
    rule_type: str = "non_negativity"
    expression: str = ""
    severity: str = "warning"
    description: Optional[str] = None
    epistemic_source: EpistemicSource = EpistemicSource.PHYSICAL_FACT


class SemanticRelationshipSchema(BaseModel):
    """Formal join edge between entities with strict cardinality constraints."""
    source_table: str
    target_table: str
    source_columns: List[str] = Field(default_factory=list)
    target_columns: List[str] = Field(default_factory=list)
    cardinality: CardinalityType = CardinalityType.MANY_TO_ONE
    is_verified: bool = True
    join_integrity_score: float = Field(default=1.0, ge=0.0, le=1.0)
    epistemic_source: EpistemicSource = EpistemicSource.INFERRED_SEMANTICS


class SemanticMetricDefinition(BaseModel):
    """Declarative metric specification with grain and aggregation."""
    name: str
    table_name: str
    target_column: Optional[str] = None
    aggregation_type: str = "SUM"
    sql_formula: Optional[str] = None
    unit_of_analysis: str
    epistemic_source: EpistemicSource = EpistemicSource.INFERRED_SEMANTICS


class SemanticWorldModelSchema(BaseModel):
    """
    Deterministic Semantic World Model representing verifiable schema ground truth.
    """
    dataset_hash: str = "unhashed"
    entities: List[EntityNode] = Field(default_factory=list)
    relationships: List[RelationshipEdge] = Field(default_factory=list)
    metrics: List[MetricNode] = Field(default_factory=list)
    time_dimensions: List[TimeDimensionNode] = Field(default_factory=list)
    business_rules: List[BusinessRuleConstraint] = Field(default_factory=list)
    functional_dependencies: List[FunctionalDependency] = Field(default_factory=list)
    verified_grains: Dict[str, List[str]] = Field(default_factory=dict)
    table_grains: Dict[str, str] = Field(default_factory=dict)
    summary_description: Optional[str] = None
    epistemic_manifest: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
