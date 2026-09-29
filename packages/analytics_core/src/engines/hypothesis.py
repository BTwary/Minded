"""HypothesisEngine: Synthesizes falsifiable hypotheses and competing counter-hypotheses."""
from dataclasses import dataclass, field
from typing import List
from packages.analytics_core.src.engines.semantic import SemanticResolution


@dataclass
class CandidateHypothesis:
    """Structured hypothesis node with falsifiable prediction and initial prior."""
    code: str
    statement: str
    rationale: str
    prior_probability: float
    is_counter_hypothesis: bool
    target_metric: str
    target_dimension: str
    prior_source: str = "UNIFORM_STRUCTURAL_PRIOR"


class HypothesisEngine:
    """Generates competing explanatory hypotheses based on semantic bindings and intent."""

    @staticmethod
    def generate_hypotheses(semantic: SemanticResolution) -> List[CandidateHypothesis]:
        metric = semantic.target_metric_col
        dim = semantic.group_dimension_col or "global segment"

        h1 = CandidateHypothesis(
            code="HYP-01",
            statement=f"Variance concentration in specific categories of {dim} drove the change in {metric}.",
            rationale=f"Localized category concentration in {dim} is the highest-entropy explanatory candidate.",
            prior_probability=0.50,
            is_counter_hypothesis=False,
            target_metric=metric,
            target_dimension=dim,
        )

        h2 = CandidateHypothesis(
            code="HYP-02",
            statement=f"Uniform macro distribution shift across all categories of {dim} for {metric}.",
            rationale=f"Macro uniform baseline hypothesis evaluating systemic shifts across {dim}.",
            prior_probability=0.50,
            is_counter_hypothesis=True,
            target_metric=metric,
            target_dimension=dim,
        )

        return [h1, h2]
