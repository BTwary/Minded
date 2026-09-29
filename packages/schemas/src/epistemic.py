"""Epistemic State Vector: Strict multi-dimensional uncertainty without single-score collapse."""
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class CausalIdentifiabilityStatus(str, Enum):
    """Formal causal identifiability classification."""
    IDENTIFIED_BACKDOOR = "IDENTIFIED_BACKDOOR"
    QUASI_EXPERIMENTAL = "QUASI_EXPERIMENTAL"  # DiD, IV, RDD
    OBSERVATIONAL_ONLY = "OBSERVATIONAL_ONLY"
    NOT_IDENTIFIABLE = "NOT_IDENTIFIABLE"      # Unmeasured confounding bounds exceeded


class EpistemicStateVector(BaseModel):
    """
    Strict multi-dimensional uncertainty vector.
    Disallows collapsing epistemics into a single subjective confidence score.
    """
    data_quality_score: float = Field(
        ge=0.0, le=1.0, description="Missingness, schema stability, and pipeline health"
    )
    semantic_certainty: float = Field(
        ge=0.0, le=1.0, description="Confidence in SWM grain, candidate key, and metric mapping"
    )
    statistical_confidence: float = Field(
        ge=0.0, le=1.0, description="1 - p_value, ANOVA eta^2 effect size, or posterior mass"
    )
    model_fidelity: float = Field(
        ge=0.0, le=1.0, description="Assumption checks passed (normality, homoscedasticity, dual-engine delta)"
    )
    causal_status: CausalIdentifiabilityStatus = Field(
        default=CausalIdentifiabilityStatus.OBSERVATIONAL_ONLY,
        description="Formal causal identifiability state"
    )
    causal_sensitivity_e_value: Optional[float] = Field(
        default=None, ge=1.0, description="E-value for robustness against unmeasured confounding"
    )
    multiple_testing_adjusted: bool = Field(
        default=True, description="Whether Benjamini-Hochberg FDR / Bonferroni correction was applied"
    )
    multiverse_robustness_pct: Optional[float] = Field(
        default=100.0, ge=0.0, le=100.0, description="Percentage of defensible specifications agreeing"
    )
