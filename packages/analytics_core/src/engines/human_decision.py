"""HumanDecisionController: Manages human-in-the-loop decision requests and autonomy policies."""
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class DecisionPolicy(str, Enum):
    CONTINUE_AUTONOMOUSLY = "CONTINUE_AUTONOMOUSLY"
    ASSUME_AND_CONTINUE = "ASSUME_AND_CONTINUE"
    ASK_USER = "ASK_USER"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"
    ABORT = "ABORT"


@dataclass
class DecisionEvaluation:
    """Evaluation result for determining whether human intervention is required."""
    policy: DecisionPolicy
    question: Optional[str] = None
    options: Optional[List[str]] = None
    default_assumption: Optional[str] = None
    reason: Optional[str] = None


class HumanDecisionController:
    """Evaluates autonomy boundaries and structures human decision requests."""

    @staticmethod
    def evaluate_ambiguity(
        available_numeric_cols: List[str],
        resolved_metric: str,
        user_specified_metric: bool,
    ) -> DecisionEvaluation:
        # If there are multiple ambiguous target metrics and user didn't specify one
        if not user_specified_metric and len(available_numeric_cols) > 3:
            return DecisionEvaluation(
                policy=DecisionPolicy.ASSUME_AND_CONTINUE,
                question=f"Discovered multiple candidate metrics. Proceeding with '{resolved_metric}'.",
                options=available_numeric_cols[:4],
                default_assumption=resolved_metric,
                reason="Automatic resolution selected leading numeric metric from semantic model.",
            )

        return DecisionEvaluation(
            policy=DecisionPolicy.CONTINUE_AUTONOMOUSLY,
            reason="Clear metric and semantic mapping resolved.",
        )
