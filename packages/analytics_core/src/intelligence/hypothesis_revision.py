"""HypothesisRevisionEngine: deterministic hypothesis status and prediction-count revision driven
by prediction evaluation outcomes.

Updates supporting/refuted/unresolved prediction counts and structured lifecycle status
(ACTIVE / SUPPORTED / WEAKENED / REFUTED / INCONCLUSIVE / RETIRED).
"""
from dataclasses import dataclass
from typing import List, Optional

from packages.analytics_core.src.intelligence.predictive_hypothesis import PredictiveHypothesis
from packages.analytics_core.src.intelligence.prediction_engine import PredictionEvaluationResult


@dataclass
class HypothesisRevisionOutcome:
    hypothesis_code: str
    previous_status: str
    new_status: str
    changed: bool
    reason: str


class HypothesisRevisionEngine:
    """Applies prediction evaluation outcomes to parent hypothesis structured lifecycle fields."""

    @staticmethod
    def apply_prediction_outcome(
        hypothesis: PredictiveHypothesis,
        evaluation: PredictionEvaluationResult,
    ) -> HypothesisRevisionOutcome:
        previous_status = hypothesis.belief_state
        status = evaluation.status

        if status == "SUPPORTED":
            hypothesis.supporting_prediction_count += 1
        elif status == "REFUTED":
            hypothesis.refuted_prediction_count += 1
        elif status in ("INCONCLUSIVE", "NOT_TESTABLE", "PENDING"):
            hypothesis.unresolved_prediction_count += 1

        new_status = HypothesisRevisionEngine._derive_lifecycle_status(hypothesis)
        hypothesis.belief_state = new_status

        reason = f"Prediction evaluation ({status}) for {hypothesis.hypothesis_code}: {evaluation.reason}"

        return HypothesisRevisionOutcome(
            hypothesis_code=hypothesis.hypothesis_code,
            previous_status=previous_status,
            new_status=new_status,
            changed=(previous_status != new_status),
            reason=reason,
        )

    @staticmethod
    def _derive_lifecycle_status(hypothesis: PredictiveHypothesis) -> str:
        """Derives status from prediction counts and posterior probability."""
        support = hypothesis.supporting_prediction_count
        refute = hypothesis.refuted_prediction_count

        if refute > 0 and support == 0:
            return "REFUTED" if hypothesis.posterior_probability < 0.20 else "WEAKENED"
        if refute > 0 and support > 0:
            return "INCONCLUSIVE"
        if support > 0 and refute == 0:
            return "SUPPORTED" if hypothesis.posterior_probability > 0.70 else "ACTIVE"
        return "ACTIVE"

    @staticmethod
    def find_hypothesis_by_code(hypotheses: List[PredictiveHypothesis], code: str) -> Optional[PredictiveHypothesis]:
        return next((h for h in hypotheses if h.hypothesis_code == code), None)
