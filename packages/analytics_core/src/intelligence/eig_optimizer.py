"""EIGOptimizer: Expected Information Gain (EIG) and multi-objective experiment utility optimizer."""
from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Tuple
from packages.analytics_core.src.intelligence.predictive_hypothesis import PredictiveHypothesis
from packages.analytics_core.src.intelligence.experiment_synthesizer import CandidateExperiment


@dataclass
class ScoredExperiment:
    """Candidate experiment with computed EIG and utility scores."""
    experiment: CandidateExperiment
    expected_information_gain: float  # Shannon entropy drop expectation
    utility_score: float              # Overall multi-objective utility
    selection_rationale: str

    @property
    def robustness_value(self) -> float:
        return getattr(self.experiment, "robustness_value", 0.0)

    @property
    def adversarial_value(self) -> float:
        return getattr(self.experiment, "adversarial_value", 0.0)

    @property
    def causal_value(self) -> float:
        return getattr(self.experiment, "causal_value", 0.0)


class ComputeCostEstimator:
    """Estimates relational compute cost based on SQL AST / query complexity."""

    @staticmethod
    def estimate_compute_cost(sql_query: Optional[str]) -> float:
        """Estimates compute cost based on Relational AST / operator complexity."""
        if not sql_query:
            return 1.0

        q_upper = sql_query.upper()
        # Count expensive operations
        cross_joins = q_upper.count("CROSS JOIN") + (1 if "," in q_upper.split("FROM")[-1].split("WHERE")[0] and "JOIN" not in q_upper else 0)
        joins = q_upper.count(" JOIN ")
        windows = q_upper.count("OVER (") + q_upper.count("OVER(")
        subqueries = q_upper.count("SELECT ") - 1 if q_upper.count("SELECT ") > 1 else 0

        # Cost Proxy Formula: Base + (Joins * 2) + (Windows * 5) + (Subqueries * 3) + (CrossJoins * 50)
        cost_proxy = 1.0 + (joins * 2.0) + (windows * 5.0) + (subqueries * 3.0) + (cross_joins * 50.0)
        return float(cost_proxy)


class EIGOptimizer:
    """Optimizes next-best analytical experiment selection via Expected Information Gain (EIG) and Compute Cost Arbitrage."""


    @staticmethod
    def compute_entropy(probs: List[float]) -> float:
        """Compute Shannon entropy H(P) = - sum(p * log2(p))."""
        ent = 0.0
        for p in probs:
            if p > 1e-9:
                ent -= p * math.log2(p)
        return float(ent)

    @classmethod
    def compute_expected_entropy_reduction(
        cls,
        current_probs: List[float],
        target_hyp_index: int,
        discriminating_power: float,
        likelihood_if_true: Optional[float] = None,
        likelihood_if_false: Optional[float] = None,
    ) -> float:
        """
        Calculates exact expected Shannon entropy reduction E[Delta H]:
        EIG = H(P) - [ P(E_pos)*H(P|E_pos) + P(E_neg)*H(P|E_neg) ]
        """
        current_entropy = cls.compute_entropy(current_probs)
        if current_entropy < 1e-6:
            return 0.0

        p_target = current_probs[target_hyp_index] if target_hyp_index < len(current_probs) else 0.50
        
        # When explicit statistical likelihoods are absent, fall back to discriminating_power
        # if available on the candidate experiment.
        if likelihood_if_true is None or likelihood_if_false is None:
            if discriminating_power and discriminating_power > 0.0:
                l_pos = max(0.01, min(0.99, float(discriminating_power)))
                l_neg = max(0.01, min(0.99, 1.0 - float(discriminating_power)))
            else:
                return 0.0
        else:
            l_pos = float(likelihood_if_true)
            l_neg = float(likelihood_if_false)
        if not (0.0 <= l_pos <= 1.0 and 0.0 <= l_neg <= 1.0):
            return 0.0

        # P(E_pos) = p*l_pos + (1-p)*l_neg
        p_evidence_pos = p_target * l_pos + (1.0 - p_target) * l_neg
        p_evidence_neg = 1.0 - p_evidence_pos

        # Posterior if evidence is positive: P(H|E+) = P(H)*P(E+|H) / P(E+)
        post_target_pos = (p_target * l_pos) / max(p_evidence_pos, 1e-9)
        # Posterior if evidence is negative: P(H|E-) = P(H)*P(E-|H) / P(E-).
        # P(E-|H) = 1 - l_pos (NOT l_neg, which is P(E+|H_false) -- a
        # different conditional entirely). Using l_neg here was a live bug:
        # it silently computed P(H|E-) from the wrong likelihood, which
        # never surfaced while every candidate's l_pos/l_neg were None
        # (EIG=0.0 short-circuited before this line ever ran with real
        # numbers). Now that ExperimentSynthesizer populates real
        # likelihoods, this branch actually executes and must be correct.
        post_target_neg = (p_target * (1.0 - l_pos)) / max(p_evidence_neg, 1e-9)

        # Build simulated posterior distributions
        probs_pos = current_probs[:]
        probs_pos[target_hyp_index] = post_target_pos
        scale_pos = (1.0 - post_target_pos) / max(1.0 - p_target, 1e-9)
        for i in range(len(probs_pos)):
            if i != target_hyp_index:
                probs_pos[i] *= scale_pos

        probs_neg = current_probs[:]
        probs_neg[target_hyp_index] = post_target_neg
        scale_neg = (1.0 - post_target_neg) / max(1.0 - p_target, 1e-9)
        for i in range(len(probs_neg)):
            if i != target_hyp_index:
                probs_neg[i] *= scale_neg

        exp_entropy = (
            p_evidence_pos * cls.compute_entropy(probs_pos)
            + p_evidence_neg * cls.compute_entropy(probs_neg)
        )

        eig = max(0.0, current_entropy - exp_entropy)
        return float(eig)

    @classmethod
    def score_candidate_experiments(
        cls,
        candidates: List[CandidateExperiment],
        hypotheses: List[PredictiveHypothesis],
        executed_codes: Optional[List[str]] = None,
        multiverse_report: Optional[Any] = None,
        executed_fingerprints: Optional[List[str]] = None,
    ) -> List[ScoredExperiment]:
        executed_codes = executed_codes or []
        executed_fingerprints = executed_fingerprints or []
        # Deterministic scientific ordering: candidate arrival order must never
        # affect EIG or tie-breaking. Canonical identity is the strongest stable
        # scientific key; hypothesis_code is the deterministic fallback.
        ordered_hypotheses = sorted(
            hypotheses,
            key=lambda h: (
                str(getattr(h, "canonical_identity", "") or ""),
                str(getattr(h, "hypothesis_code", "") or ""),
            ),
        )
        current_probs = [h.posterior_probability for h in ordered_hypotheses]
        hyp_index_map = {h.hypothesis_code: idx for idx, h in enumerate(ordered_hypotheses)}
        scored: List[ScoredExperiment] = []

        # A global multiverse robustness score is an investigation-level
        # sensitivity result, not evidence specific to every candidate experiment.
        # Do not mass-copy it onto candidates: that would manufacture candidate
        # differentiation (or erase real differentiation) and contaminate planner
        # utility with a non-local statistic. Candidate robustness_value is used only
        # when the synthesizer/experiment itself supplied an explicit value.

        for exp in candidates:
            if exp.code in executed_codes and not (exp.replication_of or exp.refinement_of):
                continue
            if exp.fingerprint and exp.fingerprint in executed_fingerprints and not (exp.replication_of or exp.refinement_of):
                continue

            target_idx = hyp_index_map.get(exp.target_hypothesis_code, 0)
            eig = cls.compute_expected_entropy_reduction(
                current_probs=current_probs,
                target_hyp_index=target_idx,
                discriminating_power=exp.discriminating_power,
                likelihood_if_true=getattr(exp, "likelihood_if_true", None),
                likelihood_if_false=getattr(exp, "likelihood_if_false", None),
            )

            # Multi-objective utility including EIG, adversarial challenge, robustness, and causal value
            adv_val = getattr(exp, "adversarial_value", 0.0)
            rob_val = getattr(exp, "robustness_value", 0.0)
            causal_val = getattr(exp, "causal_value", 0.0)

            composite_info_value = eig + 0.35 * adv_val + 0.25 * rob_val + 0.25 * causal_val
            utility = (composite_info_value * exp.reliability_weight * exp.decision_relevance) / max(0.1, exp.estimated_cost)

            reasons = [f"EIG={eig:.3f} bits" if eig > 0 else "EIG=unavailable", f"Reliability={exp.reliability_weight:.2f}", f"Cost={exp.estimated_cost:.1f}x"]
            if adv_val > 0:
                reasons.append(f"Adversarial Value={adv_val:.2f}")
            if rob_val > 0:
                reasons.append(f"Robustness Value={rob_val:.2f}")
            if causal_val > 0:
                reasons.append(f"Causal Value={causal_val:.2f}")
            if getattr(exp, "target_prediction_id", None):
                reasons.append(f"Tests Prediction={exp.target_prediction_id}")

            rationale = f"{', '.join(reasons)} -> Utility={utility:.3f}"
            exp.selection_rationale = rationale

            scored.append(
                ScoredExperiment(
                    experiment=exp,
                    expected_information_gain=float(eig),
                    utility_score=float(utility),
                    selection_rationale=rationale,
                )
            )

        # Sort by utility, then by scientific tie-breakers.  Never rely on
        # Python list arrival order when utilities are equal (or numerically
        # indistinguishable), otherwise two valid evidence-arrival orders can
        # produce different next experiments.
        scored.sort(
            key=lambda s: (
                -round(float(s.utility_score), 12),
                str(getattr(s.experiment, "code", "") or ""),
                str(getattr(s.experiment, "target_hypothesis_code", "") or ""),
                str(getattr(s.experiment, "fingerprint", "") or ""),
            )
        )
        return scored

    @classmethod
    def select_next_experiment(
        cls,
        candidates: List[CandidateExperiment],
        hypotheses: List[PredictiveHypothesis],
        executed_codes: Optional[List[str]] = None,
        executed_fingerprints: Optional[List[str]] = None,
    ) -> Tuple[CandidateExperiment, str]:
        """Select the highest-utility unexecuted experiment."""
        scored = cls.score_candidate_experiments(candidates, hypotheses, executed_codes, executed_fingerprints=executed_fingerprints)
        if not scored:
            raise ValueError("No remaining candidate experiments to execute.")

        top_choice = scored[0]
        explanation = (
            f"Autonomous Selection: Selected {top_choice.experiment.code} ({top_choice.experiment.description}) "
            f"as optimal next-best test ({top_choice.selection_rationale})."
        )
        return top_choice.experiment, explanation

    @classmethod
    def rank_candidates(cls, candidates: List[Dict[str, Any]], priors: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        """Rank candidate dictionaries without treating supplied EIG labels as evidence.

        EIG is computed only when the candidate supplies an explicit predictive
        likelihood pair and a valid target hypothesis index. Legacy
        ``expected_information_gain`` fields are ignored.
        """
        scored = []
        normalized_priors = list(priors or [])
        for c in candidates:
            eig = 0.0
            target_idx = c.get("target_hypothesis_index")
            l_true = c.get("likelihood_if_true")
            l_false = c.get("likelihood_if_false")
            if normalized_priors and target_idx is not None and l_true is not None and l_false is not None:
                try:
                    eig = cls.compute_expected_entropy_reduction(
                        current_probs=normalized_priors,
                        target_hyp_index=int(target_idx),
                        discriminating_power=0.0,
                        likelihood_if_true=float(l_true),
                        likelihood_if_false=float(l_false),
                    )
                except (TypeError, ValueError, IndexError):
                    eig = 0.0
            adv = c.get("adversarial_value", 0.0)
            rob = c.get("robustness_value", 0.0)
            cost = c.get("estimated_cost", 0.1)
            utility = (eig + 0.35 * adv + 0.25 * rob) / max(0.1, cost)
            scored.append({**c, "computed_expected_information_gain": float(eig), "computed_utility": utility})
        # Deterministic ranking: equal (or numerically indistinguishable)
        # utilities must not inherit caller/list arrival order.  A candidate's
        # stable scientific identity is the tie-breaker, followed by its code
        # and hypothesis target.
        scored.sort(
            key=lambda x: (
                -round(float(x["computed_utility"]), 12),
                str(x.get("fingerprint", "") or ""),
                str(x.get("code", "") or ""),
                str(x.get("target_hypothesis_code", "") or ""),
            )
        )
        return scored


EIGExperimentPlanner = EIGOptimizer
