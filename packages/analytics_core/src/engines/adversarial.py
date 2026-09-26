"""AdversarialEngine: Evaluates competing explanations and determines counter-hypothesis status."""
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class AdversarialEvaluation:
    """Adversarial counter-hypothesis assessment."""
    counter_hypothesis_code: str
    posterior_probability: float
    status: str  # SUPPORTED, WEAKENED, CONTRADICTED, REFUTED, INCONCLUSIVE
    is_falsified: bool
    epistemic_rationale: str


class AdversarialEngine:
    """Evaluates adversarial counter-hypotheses with nuanced epistemic calibration."""

    @staticmethod
    def evaluate_counter_hypothesis(
        counter_code: str,
        counter_posterior: float,
        discriminating_evidence_count: int,
    ) -> AdversarialEvaluation:
        if counter_posterior < 0.10 and discriminating_evidence_count >= 1:
            status = "REFUTED"
            is_falsified = True
            rationale = f"Counter-hypothesis {counter_code} empirically refuted by discriminating test evidence (posterior {counter_posterior:.2f})."
        elif counter_posterior < 0.30:
            status = "WEAKENED"
            is_falsified = False
            rationale = f"Counter-hypothesis {counter_code} significantly weakened relative to primary model (posterior {counter_posterior:.2f})."
        elif counter_posterior > 0.70:
            status = "SUPPORTED"
            is_falsified = False
            rationale = f"Counter-hypothesis {counter_code} strongly supported by empirical evidence (posterior {counter_posterior:.2f})."
        else:
            status = "INCONCLUSIVE"
            is_falsified = False
            rationale = f"Insufficient discrimination between primary and counter hypothesis {counter_code} (posterior {counter_posterior:.2f})."

        return AdversarialEvaluation(
            counter_hypothesis_code=counter_code,
            posterior_probability=counter_posterior,
            status=status,
            is_falsified=is_falsified,
            epistemic_rationale=rationale,
        )
