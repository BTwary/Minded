"""StoppingEngine: Multi-constraint autonomous investigation termination evaluator."""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class StoppingDecision:
    """Evaluation result determining if an investigation DAG should conclude."""
    should_stop: bool
    rationale: str
    criteria_breakdown: Dict[str, bool]
    reason: str = "SUFFICIENTLY_RESOLVED"

    @property
    def stopping_reason(self) -> str:
        return self.reason


class StoppingEngine:
    """Evaluates strict multi-constraint stopping criteria based on evidence sufficiency and uncertainty reduction."""

    @classmethod
    def evaluate_stopping_policy(cls, *args, **kwargs):
        return cls.evaluate_stopping(*args, **kwargs)

    @staticmethod
    def evaluate_stopping(
        leading_posterior: Optional[float] = None,
        entropy_delta: float = 0.0,
        experiments_completed: int = 0,
        all_verifications_passed: bool = True,
        counter_hypothesis_evaluated: bool = True,
        has_unresolved_adversarial_issue: bool = False,
        min_experiments: int = 1,
        current_entropy: Optional[float] = None,
        initial_entropy: Optional[float] = None,
        iteration_count: Optional[int] = None,
        max_iterations: Optional[int] = None,
        current_posteriors: Optional[List[float]] = None,
        unresolved_adversarial_issues: Optional[List[Any]] = None,
        all_verified: Optional[bool] = None,
        adversarial_survived: Optional[bool] = None,
        missingness_classification: Optional[str] = None,
        entropy_history: Optional[List[float]] = None,
        **kwargs,
    ) -> StoppingDecision:
        if all_verified is not None:
            all_verifications_passed = all_verified
        if adversarial_survived is not None:
            has_unresolved_adversarial_issue = not adversarial_survived
        if iteration_count is not None:
            experiments_completed = iteration_count
        if current_posteriors:
            leading_posterior = max(current_posteriors)
        if leading_posterior is None:
            leading_posterior = 0.50
        if current_entropy is not None and initial_entropy is not None:
            entropy_delta = current_entropy - initial_entropy
        if unresolved_adversarial_issues is not None:
            has_unresolved_adversarial_issue = len(unresolved_adversarial_issues) > 0

        # VISION (Recursive Uncertainty Gate): stagnation means the LAST rounds
        # stopped reducing uncertainty -- not that uncertainty never moved from
        # round 0. `entropy_delta` above is cumulative (current - initial), so a
        # run that learned a lot early and then plateaued has a large cumulative
        # delta and could never be recognised as stagnated: it burned the whole
        # safety budget. When the caller supplies the per-round entropy trace
        # (initial entropy first), stagnation is judged on the most recent
        # rounds. Without a trace the legacy cumulative behaviour is unchanged.
        recent_entropy_delta: Optional[float] = None
        if entropy_history is not None and len(entropy_history) >= 3:
            _tail = [float(x) for x in entropy_history[-3:]]
            recent_entropy_delta = max(abs(_tail[1] - _tail[0]), abs(_tail[2] - _tail[1]))
        _stagnation_signal = recent_entropy_delta if recent_entropy_delta is not None else abs(entropy_delta)

        # Phase 12 (Section 22): a conclusion that is materially unstable under
        # the required missingness sensitivity analysis (SENSITIVE or
        # UNIDENTIFIABLE) must not be allowed to reach a decisive/standard
        # stop -- it is gated exactly like an unresolved adversarial issue.
        # ROBUST and INSUFFICIENT_EVIDENCE do NOT gate stopping: a robust
        # conclusion should stop normally, and "we couldn't run the analysis"
        # is not itself proof of instability (do not force every missingness
        # issue to prevent stopping -- spec Section 22).
        missingness_gates_stopping = missingness_classification in ("SENSITIVE", "UNIDENTIFIABLE")
        if missingness_gates_stopping:
            has_unresolved_adversarial_issue = True

        # Safety boundary: max iterations can terminate execution, but it must
        # never overwrite a stronger unresolved safety gate. The caller needs a
        # truthful reason showing whether the investigation exhausted its budget
        # while missingness/adversarial issues remained unresolved.
        if max_iterations is not None and experiments_completed >= max_iterations:
            if missingness_gates_stopping or has_unresolved_adversarial_issue:
                unresolved = []
                if missingness_gates_stopping:
                    unresolved.append(f"missingness={missingness_classification}")
                if has_unresolved_adversarial_issue:
                    unresolved.append("adversarial_challenge")
                return StoppingDecision(
                    should_stop=True,
                    rationale=(
                        f"Safety iteration limit ({max_iterations}) reached while unresolved scientific safety "
                        f"gates remained: {', '.join(unresolved)}. Stop is inconclusive, not a positive resolution."
                    ),
                    criteria_breakdown={
                        "max_iterations_reached": True,
                        "missingness_resolved": not missingness_gates_stopping,
                        "adversarial_resolved": not has_unresolved_adversarial_issue,
                    },
                    reason="SAFETY_LIMIT_WITH_UNRESOLVED_GATES",
                )
            return StoppingDecision(
                should_stop=True,
                rationale=f"Max safety iterations ({max_iterations}) reached.",
                criteria_breakdown={"max_iterations_reached": True},
                reason="MAX_ITERATIONS_REACHED",
            )

        # Normal scientific stopping requires an explicit competing-hypothesis evaluation.
        # A high posterior alone is not sufficient: without a counter-hypothesis challenge
        # the apparent winner may only reflect an untested hypothesis space.
        competing_hypothesis_gate = counter_hypothesis_evaluated

        # Decisive one-step resolution: overwhelming evidence (>= 0.85), verified,
        # competing hypothesis evaluated, and no unresolved adversarial issues.
        decisive_one_step = (
            experiments_completed >= min_experiments
            and leading_posterior >= 0.85
            and all_verifications_passed
            and competing_hypothesis_gate
            and not has_unresolved_adversarial_issue
        )

        # Standard multi-step resolution: at least 2 experiments, sufficient uncertainty
        # reduction, verified, and competing hypothesis evaluated.
        standard_resolution = (
            experiments_completed >= 2
            and all_verifications_passed
            and competing_hypothesis_gate
            and (leading_posterior >= 0.70 or (leading_posterior >= 0.60 and entropy_delta <= -0.30))
            and not has_unresolved_adversarial_issue
        )

        # Inconclusive / Stagnation resolution: >= 3 experiments completed, no further
        # entropy gain, verified, and competing hypothesis evaluated.
        stagnation_resolution = (
            experiments_completed >= 3
            and all_verifications_passed
            and competing_hypothesis_gate
            and _stagnation_signal < 0.05
            and not has_unresolved_adversarial_issue
        )

        can_stop = decisive_one_step or standard_resolution or stagnation_resolution

        criteria = {
            "evidence_sufficiency": experiments_completed >= 1,
            "verification_passed": all_verifications_passed,
            "counter_hypothesis_evaluated": counter_hypothesis_evaluated,
            "adversarial_resolved": not has_unresolved_adversarial_issue,
            "missingness_resolved": not missingness_gates_stopping,
            "acceptable_uncertainty": leading_posterior >= 0.70 or decisive_one_step or stagnation_resolution,
        }

        reason = "INSUFFICIENT_DISCRIMINATION"
        if not counter_hypothesis_evaluated and not has_unresolved_adversarial_issue:
            rationale = (
                "Investigation continuing: no competing hypothesis has yet been explicitly evaluated; "
                "a positive posterior cannot trigger normal stopping without counter-hypothesis testing."
            )
            reason = "COUNTER_HYPOTHESIS_NOT_EVALUATED"
        elif missingness_gates_stopping and not (unresolved_adversarial_issues):
            rationale = (
                f"Investigation continuing: the leading conclusion is materially unstable under missingness "
                f"sensitivity analysis (classification={missingness_classification}) -- a high posterior over "
                f"an unstable/selection-biased claim must not automatically produce a decisive stop "
                f"(experiments completed: {experiments_completed})."
            )
            reason = "MISSINGNESS_SENSITIVITY_UNRESOLVED"
        elif has_unresolved_adversarial_issue:
            rationale = (
                f"Investigation continuing: unresolved adversarial vulnerability (Simpson's paradox or confounding detected). "
                f"Spawning targeted follow-up test (experiments completed: {experiments_completed})."
            )
            reason = "ADVERSARIAL_CHALLENGE_UNRESOLVED"
        elif can_stop:
            if decisive_one_step:
                rationale = (
                    f"Decisive one-step resolution achieved: leading posterior {leading_posterior:.2f} "
                    f"with verified dual-engine execution and no adversarial vulnerabilities."
                )
                reason = "DECISIVE_SIGNAL_RESOLVED"
            elif stagnation_resolution:
                rationale = (
                    f"Investigation terminated: empirical data exhausted with residual uncertainty "
                    f"(leading posterior {leading_posterior:.2f}, entropy delta {entropy_delta:.2f})."
                )
                reason = "UNCERTAINTY_STAGNATED"
            else:
                rationale = (
                    f"Multi-constraint stopping criteria satisfied: leading posterior {leading_posterior:.2f}, "
                    f"entropy reduction {entropy_delta:.2f}, dual-engine verifications passed, and competing hypotheses evaluated."
                )
                reason = "SUFFICIENTLY_RESOLVED"
        else:
            rationale = (
                f"Investigation continuing: insufficient evidence discrimination (posterior {leading_posterior:.2f}, "
                f"verified: {all_verifications_passed}, experiments: {experiments_completed})."
            )

        return StoppingDecision(
            should_stop=can_stop,
            rationale=rationale,
            criteria_breakdown=criteria,
            reason=reason,
        )
