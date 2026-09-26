"""Hypothesis Graph Schemas: Formal predictive hypothesis DAG and falsification criteria."""
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class HypothesisStatus(str, Enum):
    PROPOSED = "PROPOSED"
    ACTIVE = "ACTIVE"
    SUPPORTED = "SUPPORTED"
    WEAKENED = "WEAKENED"
    REFUTED = "REFUTED"


class HypothesisRelationshipType(str, Enum):
    COMPETES_WITH = "COMPETES_WITH"
    SPECIALIZES = "SPECIALIZES"
    MEDIATES = "MEDIATES"


class FalsificationCriteria(BaseModel):
    """Explicit empirical condition and statistical test that refutes the hypothesis."""
    statistical_test: str
    threshold: float
    direction: str = "less_than"  # less_than, greater_than, two_sided
    falsification_statement: Optional[str] = None


class HypothesisNodeSchema(BaseModel):
    """A formal node in the Hypothesis Graph with explicit testable mechanisms and Bayesian priors."""
    hypothesis_id: str
    proposition: str
    causal_mechanism: str
    prior_belief: float = Field(ge=0.0, le=1.0)
    falsification_criteria: FalsificationCriteria
    competing_hypothesis_ids: List[str] = Field(default_factory=list)
    posterior_belief: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    status: HypothesisStatus = HypothesisStatus.PROPOSED


class HypothesisEdgeSchema(BaseModel):
    """Explicit causal/epistemic relationship between hypothesis nodes."""
    source_id: str
    target_id: str
    relationship_type: HypothesisRelationshipType = HypothesisRelationshipType.COMPETES_WITH


class HypothesisGraphSchema(BaseModel):
    """
    Formal Hypothesis Graph representing competing propositions, priors, and falsification paths.
    Replaces unstructured dictionaries.
    """
    nodes: List[HypothesisNodeSchema] = Field(default_factory=list)
    edges: List[HypothesisEdgeSchema] = Field(default_factory=list)
    shannon_entropy: float = Field(ge=0.0, description="Normalized epistemic Shannon entropy across hypotheses")
