"""ProvenanceEngine: Generates canonical SHA-256 cryptographic provenance manifests."""
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple


class ProvenanceEngine:
    """Computes cryptographic content-addressed SHA-256 manifests for AA-OS investigations."""

    @staticmethod
    def generate_manifest_hash(
        investigation_id: str,
        project_id: str,
        question: str,
        dataset_fingerprints: Dict[str, str],
        experiments_log: List[Dict[str, Any]],
        verdict_summary: Dict[str, Any],
    ) -> str:
        """Construct a canonically ordered JSON manifest and compute its SHA-256 hex digest."""
        canonical_payload = {
            "investigation_id": investigation_id,
            "project_id": project_id,
            "question": question,
            "datasets": sorted([{"name": k, "hash": v} for k, v in dataset_fingerprints.items()], key=lambda x: x["name"]),
            "experiments": sorted(experiments_log, key=lambda x: x.get("step_code", "")),
            "verdict": verdict_summary,
        }
        serialized = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    @staticmethod
    def formulate_direct_answer(
        intent: Any,
        semantic: Any,
        leading_hypothesis_code: str,
        leading_claim: str,
        posterior: float,
        verdict_type: str,
        experiments_summary: List[Dict[str, Any]],
        multiverse_report: Optional[Any] = None,
        adversarial_refutation: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str]:
        """Formulates direct answer and main finding based on evidence."""
        # DEFECT-021: a genuine adversarial refutation is reported to the
        # analyst as what it is -- with the numbers and next steps -- instead
        # of the generic "unable to find sufficient evidence" message.
        if verdict_type == "REFUTED" and adversarial_refutation:
            from packages.analytics_core.src.engines.refutation_narrative import describe_simpsons_refutation
            _headline, _finding, _steps = describe_simpsons_refutation(adversarial_refutation)
            return _headline, f"{_finding} Suggested next steps: " + " ".join(_steps)
        is_churn = getattr(intent, "intent_type", None) == "CHURN" or any(w in getattr(intent, "raw_question", "").lower() for w in ["churn", "cancellation", "attrition", "dropoff"])
        outcome_available = getattr(semantic, "churn_outcome_available", True) and getattr(semantic, "churn_event_col", None) is not None

        if is_churn and not outcome_available:
            direct_ans = "The dataset contains no identifiable churn outcome, so AA-OS cannot estimate or predict churn from this data."
            main_find = f"Data-availability limitation: No churn or cancellation event variable was found in dataset '{getattr(semantic, 'primary_dataset_name', 'primary')}'. AA-OS will not fabricate churn predictions or probabilities from data lacking an outcome variable."
            return direct_ans, main_find

        if verdict_type == "DIAGNOSED" and posterior >= 0.70:
            direct_ans = f"{leading_claim} (Posterior belief: {posterior*100:.1f}%)"
            if (
                multiverse_report
                and getattr(multiverse_report, "applicable", True)
                and getattr(multiverse_report, "robustness_pct", None) is not None
            ):
                direct_ans += f" [Multiverse Robustness: {multiverse_report.robustness_pct:.1f}% across specifications]"
            main_find = f"Empirical evaluation confirmed {leading_hypothesis_code} with {len(experiments_summary)} verified experiments."
        elif verdict_type == "STATISTICALLY_SIGNIFICANT" and posterior >= 0.65:
            direct_ans = (
                f"{leading_claim} (Posterior belief: {posterior*100:.1f}%) -- statistically significant but "
                f"short of the confidence bar required for a firm diagnosis; treat as a lead worth confirming."
            )
            if (
                multiverse_report
                and getattr(multiverse_report, "applicable", True)
                and getattr(multiverse_report, "robustness_pct", None) is not None
            ):
                direct_ans += f" [Multiverse Robustness: {multiverse_report.robustness_pct:.1f}% across specifications]"
            main_find = f"Empirical evaluation found statistically significant support for {leading_hypothesis_code} with {len(experiments_summary)} verified experiments, below the diagnosis confidence bar."
        else:
            # DEFECT-019: this branch previously fired whenever the
            # "leading" hypothesis's claim TEXT happened to contain the word
            # "confound" -- regardless of whether any evidence was actually
            # gathered. When no dimension could be autonomously resolved
            # (e.g. an undirected "why are customers churning?" question
            # against a dataset with multiple, equally-plausible candidate
            # dimensions and none named -- experiment_synthesizer.py
            # deliberately declines to guess among them, by design, rather
            # than invent a group-by key), zero experiments ever ran, yet
            # the confounding-specific hypothesis (tied at the uniform
            # prior along with the others, purely by insertion order) was
            # still picked as "leading" and its unresolved claim text --
            # containing "confound" -- produced a specific, false empirical
            # assertion ("aggregate churn differences ... are confounded ...
            # (Simpson's paradox)") as if stratification had actually been
            # tested and had failed. Nothing was tested. Requiring at least
            # one executed experiment before asserting a specific
            # confounding finding keeps this claim honest; with zero
            # experiments it now falls through to the generic
            # insufficient-evidence message below, consistent with this
            # codebase's existing non-fabrication principle for other
            # data-availability gaps (see the churn_outcome_available check
            # above).
            if is_churn and "confound" in leading_claim.lower() and experiments_summary:
                direct_ans = f"Inconclusive: aggregate churn differences across segments are confounded or mediated by other variables (Simpson's paradox)."
                main_find = f"Identifiability limitation: segment effect does not survive stratification (leading posterior: {posterior*100:.1f}%)."
            elif not experiments_summary:
                # No experiment ever ran against this question (e.g. the
                # question left a required dimension unnamed/ambiguous and
                # the synthesizer correctly declined to guess one -- see
                # experiment_synthesizer.py). This is a different situation
                # from "we gathered evidence and it didn't discriminate";
                # say so plainly rather than implying an ambiguous empirical
                # result was reached.
                direct_ans = "Unable to find a testable analysis for this question against the supplied data."
                main_find = "No experiment was executed: the question did not resolve to a specific, analyzable dimension or metric in this dataset."
            else:
                direct_ans = f"Inconclusive: unable to find sufficient evidence to distinguish competing explanations."
                main_find = f"Data exhibits high residual uncertainty across tested hypotheses (leading posterior: {posterior*100:.1f}%)."
        return direct_ans, main_find

    @staticmethod
    def compute_reproducible_manifest_hash(
        investigation_id: str,
        question: str,
        primary_dataset: str,
        executed_experiments: List[str],
        final_verdict: str,
        epistemic_vector: Optional[Any] = None,
    ) -> str:
        canonical_payload = {
            "question": question.strip().lower(),
            "primary_dataset": primary_dataset,
            "executed_experiments": executed_experiments,
            "final_verdict": final_verdict,
            "epistemic_vector": epistemic_vector.to_dict() if epistemic_vector and hasattr(epistemic_vector, "to_dict") else {},
        }
        serialized = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

