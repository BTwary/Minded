"""Pydantic Schemas for AI Planner Structured Outputs."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CandidateActionSchema(BaseModel):
    """Specific analytical tool action proposed by the planner."""
    tool: str = Field(description="Name of the analytical tool to execute (e.g. query_sql, decompose_variance, calculate_statistics, run_forecast, rfm_segmentation, derive_metric)")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Dictionary of parameter arguments for the tool")
    hypothesis_id: Optional[str] = Field(default="HYP-01", description="ID of the hypothesis being tested")
    rationale: str = Field(description="Justification explaining why this action has high information gain for the investigation")


class HypothesisPlanItem(BaseModel):
    """Individual hypothesis formulated by the planner."""
    id: str = Field(description="Identifier (e.g. HYP-01, HYP-02)")
    statement: str = Field(description="Testable business hypothesis statement")
    rationale: str = Field(description="Theoretical or empirical reasoning supporting this hypothesis")
    priority: float = Field(default=0.8, description="Priority weight between 0.0 and 1.0")
    status: str = Field(default="proposed", description="Initial status: proposed")


class InitialPlanSchema(BaseModel):
    """Structured initial investigation plan emitted by LLM planner."""
    investigation_objective: str = Field(description="Decomposed business objective of the inquiry")
    hypotheses: List[HypothesisPlanItem] = Field(description="List of competing prioritized hypotheses")
    first_action: CandidateActionSchema = Field(description="Optimal first analytical tool action to execute")


class HypothesisUpdateItem(BaseModel):
    """Status update for an evaluated hypothesis based on observed data."""
    id: str = Field(description="ID of the hypothesis being updated (e.g. HYP-01)")
    status: str = Field(description="New status: supported, rejected, investigating, inconclusive")
    reason: Optional[str] = Field(default=None, description="Detailed explanation if rejected or inconclusive")
    finding: Optional[str] = Field(default=None, description="Specific confirmed finding if supported")


class NextActionSchema(BaseModel):
    """Next step decision emitted by the multi-turn planner."""
    is_complete: bool = Field(default=False, description="Set to true if sufficient validated evidence has been collected to definitively answer the question")
    tool: Optional[str] = Field(default=None, description="Name of next tool to execute, or null if is_complete is true")
    arguments: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Parameters for the tool")
    hypothesis_id: Optional[str] = Field(default=None, description="ID of hypothesis being tested")
    hypothesis_update: Optional[HypothesisUpdateItem] = Field(default=None, description="Hypothesis status update based on latest observation")
    rationale: Optional[str] = Field(default=None, description="Analytical justification for next action")
    stopping_reason: Optional[str] = Field(default=None, description="Explanation of why evidence is sufficient to conclude")


class AdversarialCritiqueSchema(BaseModel):
    """Adversarial stress-test challenging the primary finding."""
    counter_hypothesis: str = Field(description="Plausible alternative explanation that could invalidate the primary finding (e.g. seasonal anomaly, channel cannibalization, data artifact)")
    refutation_evidence: str = Field(description="Specific empirical data or test that refutes or confirms the counter-hypothesis")
    is_refuted: bool = Field(default=True, description="True if empirical evidence successfully refutes the counter-hypothesis")
