"""Deterministic Mock & Heuristic AI Provider for Offline Benchmark Testing."""
import json
import re
from typing import Any, Dict, Optional, Type, TypeVar
from pydantic import BaseModel
from apps.api.src.ai.providers.base import BaseAIProvider

T = TypeVar("T", bound=BaseModel)


class DeterministicMockAIProvider(BaseAIProvider):
    """Zero-cost, 100% offline deterministic reasoning engine for tests and benchmarks."""

    def __init__(self, model_name: str = "deterministic-heuristic-v1"):
        self.model_name = model_name

    @property
    def is_ai_enabled(self) -> bool:
        """Mode 1: Deterministic Free Mode does not use external AI APIs."""
        return False

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        lower_p = prompt.lower()

        # Check if structured schema was requested in prompt
        if "```json" in prompt and "properties" in prompt:
            # Let generate_structured handle parsing
            pass

        if "revenue" in lower_p and ("march" in lower_p or "decline" in lower_p or "fall" in lower_p):
            return "Revenue experienced a significant decline driven by reduced sales in Region B and lower volume of Product X."
        if "churn" in lower_p:
            return "Customer churn risk is heavily concentrated among single-purchase buyers with no activity in 90 days."
        if "forecast" in lower_p:
            return "Forecast indicates steady baseline growth with expected seasonal fluctuation."

        return "Analysis completed based on deterministic computational evidence."

    def generate_structured(
        self,
        prompt: str,
        schema_cls: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
    ) -> T:
        lower_p = prompt.lower()
        schema_name = schema_cls.__name__

        # Mock structured generator for different schemas
        mock_data: Dict[str, Any] = {}

        if "Hypothesis" in schema_name or "Plan" in schema_name:
            if "march" in lower_p or "revenue" in lower_p or "fall" in lower_p or "decline" in lower_p:
                mock_data = {
                    "intent": "revenue_decline_root_cause_analysis",
                    "target_metrics": ["revenue", "profit", "order_count"],
                    "dimensions": ["region", "product_id", "category"],
                    "hypotheses": [
                        {
                            "id": "H1",
                            "statement": "Regional sales decline in Region B drove the primary revenue drop.",
                            "rationale": "Historical data shows Region B accounts for substantial revenue share.",
                            "priority": 0.95,
                            "investigation_steps": [
                                "Compare revenue by region across months.",
                                "Analyze product category breakdown in Region B.",
                            ],
                        },
                        {
                            "id": "H2",
                            "statement": "Product X experienced steep volume drop or inventory shortage.",
                            "rationale": "High-margin SKU performance significantly impacts monthly revenue.",
                            "priority": 0.88,
                            "investigation_steps": [
                                "Calculate product-level revenue contribution.",
                            ],
                        },
                    ],
                    "execution_plan": [
                        {"step": 1, "tool": "run_sql", "description": "Aggregate monthly revenue by region", "expected_output": "table"},
                        {"step": 2, "tool": "calculate_statistics", "description": "Conduct two-sample t-test comparing Feb vs March", "expected_output": "stats"},
                        {"step": 3, "tool": "detect_anomalies", "description": "Scan daily revenue for anomalous drops", "expected_output": "anomalies"},
                    ],
                }
            else:
                mock_data = {
                    "intent": "general_exploratory_analysis",
                    "target_metrics": ["revenue", "quantity"],
                    "dimensions": ["category", "region"],
                    "hypotheses": [
                        {
                            "id": "H1",
                            "statement": "Key performance metrics exhibit significant variance across categories.",
                            "rationale": "Top categories generate the vast majority of volume.",
                            "priority": 0.90,
                            "investigation_steps": ["Examine category breakdown."],
                        }
                    ],
                    "execution_plan": [
                        {"step": 1, "tool": "run_sql", "description": "Summarize metrics by category", "expected_output": "table"},
                    ],
                }

        elif "ChartSpec" in schema_name:
            mock_data = {
                "chart_type": "bar",
                "title": "Revenue by Region Comparison",
                "x_column": "region",
                "y_column": "total_revenue",
                "color_column": "region",
                "explanation": "Bar chart effectively compares regional revenue contributions.",
            }

        try:
            return schema_cls.model_validate(mock_data)
        except Exception:
            # Fallback to base implementation if schema mismatch
            return super().generate_structured(prompt, schema_cls, system_prompt, temperature)


class NoneAIProvider(DeterministicMockAIProvider):
    """Explicit Zero-API / Free Mode provider."""

    def __init__(self, model_name: str = "zero-api-deterministic"):
        super().__init__(model_name=model_name)
