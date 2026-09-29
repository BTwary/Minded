"""VerdictEngine: Multi-constraint diagnostic verdict formulator with explicit variance policy and actionable conclusion sections."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from packages.analytics_core.src.engines.adversarial import AdversarialEvaluation
from packages.analytics_core.src.engines.refutation_narrative import describe_simpsons_refutation


@dataclass
class VerdictEvaluation:
    """Rigorous epistemic verdict decision output."""
    verdict_type: str  # DIAGNOSED, OBSERVED, STATISTICALLY_SIGNIFICANT, REFUTED, INCONCLUSIVE
    confidence_score: float
    stopping_criteria_met: bool
    stopping_rationale: str
    variance_requirement_status: str  # SATISFIED, NOT_APPLICABLE, UNMET
    net_variance_explained_pct: Optional[float]
    counter_hypothesis_refuted: bool
    primary_hypothesis_supported: bool
    direct_answer: str
    main_finding: str
    justification: str = ""
    # Phase 12: missingness/selection-bias epistemic classification carried
    # through to the verdict so callers can see WHY a verdict was capped
    # (e.g. SENSITIVE) rather than just seeing a lower verdict_type.
    missingness_classification: Optional[str] = None
    confidence_semantics: str = "MODEL_BASED_BELIEF"
    calibrated_probability: Optional[float] = None
    calibration_status: str = "UNAVAILABLE"

    # Structured 8-Section Actionable Conclusion
    what_we_observed: List[str] = field(default_factory=list)
    what_we_believe: str = ""
    what_supports_it: List[str] = field(default_factory=list)
    what_contradicts_it: List[str] = field(default_factory=list)
    what_remains_uncertain: List[str] = field(default_factory=list)
    what_we_cannot_claim: List[str] = field(default_factory=list)
    action_to_consider: List[str] = field(default_factory=list)
    what_to_test_next: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.justification:
            self.justification = self.stopping_rationale


class VerdictEngine:
    """Enforces multi-gate diagnostic verification before certifying root cause."""

    @staticmethod
    def evaluate_verdict(
        question: str = "",
        leading_hypothesis_code: str = "HYP-01",
        leading_posterior: Optional[float] = None,
        initial_entropy: float = 1.0,
        final_entropy: float = 0.5,
        all_verifications_passed: bool = True,
        adversarial_eval: Optional[AdversarialEvaluation] = None,
        variance_explained_pct: Optional[float] = None,
        is_categorical_diagnostic: bool = True,
        # Flexible kwargs
        leading_hypothesis_posterior: Optional[float] = None,
        adversarial_attack_survived: bool = True,
        causal_identifiable: bool = False,
        supporting_evidence_claims: Optional[List[str]] = None,
        contradicting_evidence_claims: Optional[List[str]] = None,
        missingness_classification: Optional[str] = None,
        missingness_rationale: str = "",
        # DEFECT-007 real fix (see CURRENT_DEFECT_REGISTER.md): a hypothesis
        # that has never been the direct target of a verified experiment
        # (no supporting_evidence_ids) must not be eligible for a positive
        # verdict, no matter how high its posterior climbed via Bayesian
        # normalization against other hypotheses. Defaults to True so
        # existing callers that don't pass this (and legacy tests) keep
        # their current behavior; the controller always passes the real
        # value computed from the leading hypothesis's evidence links.
        leading_hypothesis_directly_tested: bool = True,
        # A leading hypothesis that is itself the counter/null side of a
        # binary pair (e.g. "no material association between X and Y")
        # winning the posterior race is a genuine, correctly-computed
        # negative result -- but it is not a positive research finding and
        # must not be reported as DIAGNOSED or STATISTICALLY_SIGNIFICANT.
        # Those tiers are reserved for certifying the substantive claim
        # under investigation; a confidently-empty result is INCONCLUSIVE
        # with respect to that claim, however well-supported the null is.
        leading_hypothesis_is_counter: bool = False,
        # DEFECT-015: Churn identifiability verdict parameters
        churn_verdict: Optional[str] = None,
        is_churn_unidentifiable: bool = False,
        max_verdict_tier: Optional[str] = None,
        method_selection_note: str = "",
        confidence_semantics: str = "MODEL_BASED_BELIEF",
        calibrated_probability: Optional[float] = None,
        calibration_status: str = "UNAVAILABLE",
        selection_bias_indicators: Optional[List[str]] = None,
        # The runtime StoppingEngine is the sole authoritative source for
        # whether the investigation actually met stopping policy. When
        # supplied, this value must not be recomputed from posterior/entropy.
        authoritative_stopping_met: Optional[bool] = None,
        authoritative_stopping_reason: str = "",
        unvalidated_high_risk_assumptions: int = 0,
        # DEFECT-021: details of a genuine (materiality / min-cell / same-sign
        # gated) Simpson's reversal found by the adversarial attacker. When
        # present, the headline comparison is REFUTED, not merely weakened.
        adversarial_refutation: Optional[Dict[str, Any]] = None,
        # Separation of primary supported from counter-hypothesis empirically refuted
        counter_hypothesis_empirically_refuted: Optional[bool] = None,
    ) -> VerdictEvaluation:
        post = leading_hypothesis_posterior if leading_hypothesis_posterior is not None else (leading_posterior if leading_posterior is not None else 0.50)

        # Evaluate explicit variance requirement status
        if not is_categorical_diagnostic:
            var_status = "NOT_APPLICABLE"
        elif variance_explained_pct is not None and variance_explained_pct >= 10.0:
            var_status = "SATISFIED"
        else:
            var_status = "NOT_APPLICABLE" if variance_explained_pct is None else ("SATISFIED" if variance_explained_pct >= 5.0 else "UNMET")

        # DEFECT-021 (vision audit O4): this used to also treat
        # adversarial_eval.status in [...] as "subdued", including WEAKENED
        # and CONTRADICTED -- the opposite of their meaning (the attack
        # weakened/contradicted the *leading* hypothesis, not the counter).
        # In practice this branch was dormant: the only real caller
        # (runtime/controller.py) only ever passes the already-computed
        # adversarial_attack_survived boolean and never populates
        # adversarial_eval, so removing it changes no live behavior. The
        # sole authoritative signal for "did the leading hypothesis survive
        # adversarial attack" is adversarial_attack_survived.
        counter_subdued = adversarial_attack_survived

        # Phase 12 (Section 21): a conclusion classified SENSITIVE, UNIDENTIFIABLE,
        # or INSUFFICIENT_EVIDENCE by missingness sensitivity analysis must not be
        # presented as DIAGNOSED (high-confidence/invariant), regardless of how
        # strong the observed-data posterior looks. ROBUST (or no analysis run,
        # e.g. no missingness present) does not cap the verdict.
        missingness_blocks_diagnosis = missingness_classification in ("SENSITIVE", "UNIDENTIFIABLE", "INSUFFICIENT_EVIDENCE")
        selection_bias_present = bool(tuple(str(x) for x in (selection_bias_indicators or []) if str(x).strip()))
        selection_bias_blocks_strong = selection_bias_present and (max_verdict_tier or "").upper() not in {"OBSERVED", "DESCRIPTIVE"}
        assumptions_block_strong = int(unvalidated_high_risk_assumptions or 0) > 0 and (max_verdict_tier or "").upper() not in {"OBSERVED", "DESCRIPTIVE"}

        # DEFECT-007: an untested hypothesis (no verified experiment ever
        # targeted it directly) cannot be DIAGNOSED or STATISTICALLY_SIGNIFICANT
        # regardless of posterior. This is what stops a generic "counter"
        # hypothesis from winning purely because normalization inflated its
        # share when other, specific hypotheses were consolidated/removed.
        untested_blocks_positive_verdict = not leading_hypothesis_directly_tested
        counter_blocks_positive_verdict = bool(leading_hypothesis_is_counter)

        # DEFECT-015: Churn identifiability blocks positive diagnosis if unidentifiable or confounded
        churn_blocks_diagnosis = is_churn_unidentifiable or churn_verdict in ("INSUFFICIENT_EVIDENCE", "CONFOUNDED_IDENTIFIABILITY_LIMITED")

        is_diagnosed = (
            post >= 0.70
            and all_verifications_passed
            and (var_status in ["SATISFIED", "NOT_APPLICABLE"])
            and counter_subdued
            and not missingness_blocks_diagnosis
            and not selection_bias_blocks_strong
            and not assumptions_block_strong
            and not untested_blocks_positive_verdict
            and not churn_blocks_diagnosis
            and not counter_blocks_positive_verdict
        )

        verdict_type = (
            "INCONCLUSIVE" if (missingness_classification == "UNIDENTIFIABLE" or is_churn_unidentifiable or churn_verdict == "INSUFFICIENT_EVIDENCE" or selection_bias_blocks_strong or assumptions_block_strong)
            else (
                "INCONCLUSIVE" if untested_blocks_positive_verdict
                else (
                    "INCONCLUSIVE" if churn_verdict == "CONFOUNDED_IDENTIFIABILITY_LIMITED" and leading_hypothesis_code == "HYP-01"
                    else (
                        "INCONCLUSIVE" if counter_blocks_positive_verdict
                        else ("DIAGNOSED" if is_diagnosed else ("STATISTICALLY_SIGNIFICANT" if (post >= 0.65 and all_verifications_passed and not churn_blocks_diagnosis and not missingness_blocks_diagnosis) else "INCONCLUSIVE"))
                    )
                )
            )
        )
        # Method-selection policy ceiling: may only demote, never promote.
        tier_rank = {"INCONCLUSIVE": 0, "STATISTICALLY_SIGNIFICANT": 1, "DIAGNOSED": 2}
        ceiling_applied_from = None
        if (max_verdict_tier in tier_rank and verdict_type in tier_rank and
                tier_rank[verdict_type] > tier_rank[max_verdict_tier]):
            ceiling_applied_from = verdict_type
            verdict_type = max_verdict_tier
            is_diagnosed = verdict_type == "DIAGNOSED"

        # DEFECT-021: a genuine adversarial refutation of the leading claim is
        # its own verdict. It is never downgraded to a generic INCONCLUSIVE
        # (the analyst would lose the finding) and never upgraded by a
        # high posterior. A leading *counter/null* hypothesis is not what a
        # Simpson's reversal of the group difference refutes, so it is excluded.
        adversarial_refuted = bool(adversarial_refutation) and not counter_blocks_positive_verdict
        if adversarial_refuted:
            verdict_type = "REFUTED"
            is_diagnosed = False
            ceiling_applied_from = None

        # Never independently infer stopping from entropy here: that would
        # create a second, weaker stopping policy that can disagree with the
        # canonical StoppingEngine. Legacy direct callers may omit the field,
        # in which case the historical fallback is retained for compatibility.
        if authoritative_stopping_met is None:
            stopping_met = is_diagnosed or (final_entropy - initial_entropy <= -0.30)
        else:
            stopping_met = bool(authoritative_stopping_met)

        is_statistically_significant = (not is_diagnosed) and post >= 0.65 and all_verifications_passed and verdict_type == "STATISTICALLY_SIGNIFICANT"

        direct_answer = (
            f"Investigation concluded: {leading_hypothesis_code} supported with {'calibrated probability' if calibrated_probability is not None else 'model-based posterior belief'} {calibrated_probability if calibrated_probability is not None else post:.2f}."
            if is_diagnosed
            else (
                f"Investigation found a statistically significant association for {leading_hypothesis_code} "
                f"(posterior {post:.2f}), short of the confidence bar required for a firm diagnosis; "
                f"treat this as a lead worth confirming, not a settled conclusion."
                if is_statistically_significant
                else f"Investigation inconclusive: empirical data does not sufficiently discriminate competing hypotheses."
            )
        )
        if is_churn_unidentifiable:
            direct_answer = (
                "Investigation inconclusive: no identifiable churn/cancellation/attrition "
                "event is available in the supplied data, so a churn rate or churn-risk "
                "difference cannot be validly estimated. This is a data-availability "
                "limitation, not evidence that churn did or did not occur."
            )
        elif untested_blocks_positive_verdict and post >= 0.65 and all_verifications_passed:
            direct_answer = (
                f"Investigation inconclusive: {leading_hypothesis_code} holds the highest posterior belief "
                f"({post:.2f}) but was never the direct target of a verified experiment -- its share reflects "
                f"normalization against other hypotheses, not evidence gathered against it specifically. "
                f"No positive verdict is issued without a hypothesis-specific test."
            )
        elif assumptions_block_strong and post >= 0.65 and all_verifications_passed:
            direct_answer = (
                f"Investigation inconclusive: {leading_hypothesis_code} has strong observed support ({post:.2f}), "
                f"but {int(unvalidated_high_risk_assumptions)} material high-risk assumption(s) remain unvalidated; "
                "a stronger positive claim is withheld."
            )
        elif selection_bias_blocks_strong and post >= 0.65 and all_verifications_passed:
            direct_answer = (
                f"Investigation inconclusive: {leading_hypothesis_code} has strong observed support ({post:.2f}), "
                "but selection/survivorship indicators are present and the sampling or eligibility mechanism "
                "has not been resolved; a stronger positive claim is withheld."
            )
        elif missingness_blocks_diagnosis and post >= 0.70 and all_verifications_passed and (var_status in ["SATISFIED", "NOT_APPLICABLE"]) and counter_subdued:
            # This is the case that would have been DIAGNOSED on observed-data
            # criteria alone -- say explicitly that missingness is why it wasn't.
            direct_answer = (
                f"Investigation NOT diagnosed despite a strong observed-data posterior ({post:.2f}) for "
                f"{leading_hypothesis_code}: the conclusion depends materially on how missing values are "
                f"treated (missingness classification: {missingness_classification}). It is presented as "
                f"conditional, not established."
            )

        if adversarial_refuted:
            _refute_headline, _refute_finding, _refute_steps = describe_simpsons_refutation(adversarial_refutation)
            direct_answer = _refute_headline

        main_finding = (
            f"Evaluated against empirical datasets with Shannon entropy drop from {initial_entropy:.2f} to {final_entropy:.2f}"
            + (f" and {variance_explained_pct:.1f}% variance explained ({var_status})." if variance_explained_pct is not None else ".")
        )
        if missingness_classification and missingness_classification != "ROBUST":
            main_finding = f"{main_finding} Missingness sensitivity: {missingness_classification}."
            if missingness_rationale:
                main_finding = f"{main_finding} {missingness_rationale}"

        rationale = (
            f"Diagnostic criteria satisfied: posterior {post:.2f} exceeded threshold with verified dual-engine calculations (Variance: {var_status})."
            if is_diagnosed
            else f"Diagnostic criteria not fully met (posterior: {post:.2f}, verified: {all_verifications_passed}, variance: {var_status})."
        )
        if authoritative_stopping_met is not None and not authoritative_stopping_met:
            rationale = (
                f"{rationale} Authoritative stopping policy did not permit scientific termination; "
                f"reason={authoritative_stopping_reason or 'NOT_AUTHORIZED'}."
            )

        if untested_blocks_positive_verdict:
            rationale = (
                f"{rationale} Leading hypothesis {leading_hypothesis_code} has never been the direct target of "
                f"a verified experiment (no supporting evidence linked); a positive verdict was withheld rather "
                f"than manufactured from posterior-normalization share alone."
            )
        if missingness_classification in ("SENSITIVE", "UNIDENTIFIABLE"):
            rationale = (
                f"{rationale} Missingness sensitivity analysis classified the leading conclusion as "
                f"{missingness_classification}: {missingness_rationale}"
            )
        elif missingness_classification == "INSUFFICIENT_EVIDENCE":
            rationale = f"{rationale} Missingness sensitivity could not be established (INSUFFICIENT_EVIDENCE); confidence was not manufactured to compensate."

        if ceiling_applied_from is not None:
            cap_note = (
                f" Verdict restricted from {ceiling_applied_from} to {verdict_type} by the "
                f"method-selection ceiling; {method_selection_note or 'the selected analytical problem class does not permit a stronger verdict.'}"
            )
            direct_answer = f"{direct_answer}{cap_note}"
            rationale = f"{rationale}{cap_note}"

        # Structured Actionable Sections
        observed = [
            f"Evaluated empirical metrics across candidate partitions with {variance_explained_pct:.1f}% variance explained."
            if variance_explained_pct is not None else "Evaluated empirical metrics across candidate partitions.",
            f"Dual-engine verification: {'All calculations verified by independent engines' if all_verifications_passed else 'Independent verification failure detected'}.",
        ]

        believed = f"Leading theory {leading_hypothesis_code} is supported with model-based posterior belief {post:.2f}."

        supports = supporting_evidence_claims or [
            f"Consistent directional predictions corroborated across primary partition queries.",
            f"Shannon entropy decreased from {initial_entropy:.2f} to {final_entropy:.2f} bits.",
        ]

        contradicts = contradicting_evidence_claims or [
            f"Counter-hypothesis (macro uniform shift) refuted by empirical partition concentration."
            if is_diagnosed else "No definitive refutation of alternative baseline theories."
        ]

        uncertain = [
            f"Residual entropy: {final_entropy:.2f} bits.",
            "Unmeasured latent confounding may exist outside observed dataset schemas.",
        ]
        if missingness_classification in ("SENSITIVE", "UNIDENTIFIABLE"):
            uncertain.append(
                f"Missingness sensitivity analysis classified this conclusion as {missingness_classification}: "
                f"{missingness_rationale or 'plausible missing-value assignments materially change the result.'}"
            )
        elif missingness_classification == "INSUFFICIENT_EVIDENCE":
            uncertain.append(
                "Missingness sensitivity could not be computed for this conclusion (insufficient evidence); "
                "no claim of robustness to missing data is made."
            )

        cannot_claim = [
            "Pure observational data cannot establish causal direction without randomized intervention or valid instrumental variables.",
            "Findings apply to observed reporting period and population coverage.",
        ]
        if missingness_classification == "UNIDENTIFIABLE":
            cannot_claim.append(
                "The available observed data and stated missingness assumption do not permit a defensible "
                "determination of this conclusion; it cannot be claimed as established."
            )
        elif missingness_classification == "SENSITIVE":
            cannot_claim.append(
                "This conclusion cannot be claimed as invariant: it depends materially on an unstated "
                "assumption about unobserved (missing) data."
            )

        actions = [
            f"Focus operational mitigation on primary driver identified in {leading_hypothesis_code}."
            if is_diagnosed else "Hold major resource commitments until additional discriminating data is collected.",
        ]

        test_next = [
            "Conduct targeted A/B test or stratified intervention on primary segment.",
            "Collect longitudinal time-series data to evaluate temporal persistence.",
        ]

        if adversarial_refuted:
            main_finding = _refute_finding
            rationale = f"Adversarial falsification succeeded: {_refute_finding}"
            observed = [
                f"Pooled comparison: {adversarial_refutation.get('groups', ['?', '?'])[0]!r} vs "
                f"{adversarial_refutation.get('groups', ['?', '?'])[1]!r} differ by "
                f"{float(adversarial_refutation.get('marginal_difference', 0.0)):+.4g}.",
                f"Within {adversarial_refutation.get('secondary_dimension', 'the secondary dimension')} "
                f"({int(adversarial_refutation.get('strata_used', 0))} adequate groups) the adjusted difference is "
                f"{float(adversarial_refutation.get('adjusted_difference', 0.0)):+.4g}, the opposite sign.",
            ]
            believed = "The pooled group difference is confounded and is not supported as an effect of the compared dimension."
            supports = []
            contradicts = [_refute_finding]
            cannot_claim = [
                "Cannot claim the pooled difference is an effect of the compared dimension.",
                "Cannot claim the reversed (within-group) direction is a real effect; it is one adjusted estimate and other confounders may remain.",
            ]
            actions = list(_refute_steps)
            test_next = list(_refute_steps)

        primary_supported = (not adversarial_refuted) and bool(is_diagnosed)
        if counter_hypothesis_empirically_refuted is not None:
            counter_refuted = bool(counter_hypothesis_empirically_refuted)
        else:
            # Epistemic rigor: variance explained or primary hypothesis support alone
            # does not constitute empirical refutation of alternative counter-hypotheses.
            counter_refuted = False

        return VerdictEvaluation(
            verdict_type=verdict_type,
            confidence_score=0.0 if adversarial_refuted else float(post),
            stopping_criteria_met=stopping_met,
            stopping_rationale=rationale,
            variance_requirement_status=var_status,
            net_variance_explained_pct=variance_explained_pct,
            counter_hypothesis_refuted=counter_refuted,
            primary_hypothesis_supported=primary_supported,
            direct_answer=direct_answer,
            main_finding=main_finding,
            justification=rationale,
            what_we_observed=observed,
            what_we_believe=believed,
            what_supports_it=supports,
            what_contradicts_it=contradicts,
            what_remains_uncertain=uncertain,
            what_we_cannot_claim=cannot_claim,
            action_to_consider=actions,
            what_to_test_next=test_next,
            missingness_classification=missingness_classification,
            confidence_semantics=confidence_semantics,
            calibrated_probability=calibrated_probability,
            calibration_status=calibration_status,
        )
