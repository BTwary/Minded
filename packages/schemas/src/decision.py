"""Decision Engine Schemas: Grounded business recommendations with explicit Expected Utility calculations."""
from typing import List, Optional
from pydantic import BaseModel, Field


class ExpectedUtilityCalculation(BaseModel):
    """Formal mathematical Expected Utility calculation grounded in a Business World Model."""
    action_name: str
    expected_gain_metric: float
    downside_risk_metric: float
    probability_of_success: float = Field(ge=0.0, le=1.0)
    net_expected_utility: float
    utility_function_description: str


class DecisionRecommendationSchema(BaseModel):
    """
    Formal decision recommendation tied to empirical evidence and expected utility.
    Separates evidence observation from action recommendations.
    """
    recommendation_id: str
    action_title: str
    action_description: str
    grounded_hypothesis_id: str
    target_metric: str
    expected_utility: ExpectedUtilityCalculation
    policy_compliance_passed: bool = True
    required_preconditions: List[str] = Field(default_factory=list)
