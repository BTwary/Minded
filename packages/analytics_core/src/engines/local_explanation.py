"""Deterministic local explanation layer for AA-OS.

This engine explains only evidence that already exists in the canonical
investigation state. It never performs new analysis, calls an LLM, or invents
statistics. AI providers may optionally rewrite/augment this explanation at a
higher layer, but the local explanation remains the source of truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional


@dataclass(frozen=True)
class LocalExplanation:
    mode: str
    summary: str
    what_was_analyzed: str
    evidence: List[str]
    conclusion: str
    limitations: List[str]
    verification: str
    calculation_trace: List[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "summary": self.summary,
            "what_was_analyzed": self.what_was_analyzed,
            "evidence": list(self.evidence),
            "conclusion": self.conclusion,
            "limitations": list(self.limitations),
            "verification": self.verification,
            "calculation_trace": list(self.calculation_trace),
        }


class LocalExplanationEngine:
    """Build a human-readable explanation from persisted AA-OS facts."""

    @staticmethod
    def build(
        *,
        question: str,
        verdict_type: str,
        direct_answer: Optional[str],
        justification: Optional[str],
        hypotheses: Iterable[Any] = (),
        experiments: Iterable[Any] = (),
        observations: Iterable[Any] = (),
        evidence_rows: Iterable[Any] = (),
        method_selection: Optional[dict[str, Any]] = None,
        assumption_items: Optional[Iterable[Any]] = None,
    ) -> LocalExplanation:
        hyps = list(hypotheses)
        exps = list(experiments)
        obs = list(observations)
        evid = list(evidence_rows)
        assumptions = list(assumption_items or [])
        traces = []
        for e in evid:
            trace = getattr(e, "calculation_trace", None)
            if trace:
                traces.append(trace if isinstance(trace, dict) else getattr(trace, "model_dump", lambda: {})())

        supported = [h for h in hyps if str(getattr(h, "status", "")).lower() == "supported"]
        refuted = [h for h in hyps if str(getattr(h, "status", "")).lower() in {"refuted", "weakened"}]
        verified = [e for e in evid if str(getattr(e, "validation_status", "")).upper() in {"VERIFIED", "PASSED"}]
        failed_verification = [e for e in evid if str(getattr(e, "validation_status", "")).upper() in {"FAILED", "UNVERIFIED"}]

        class_name = (method_selection or {}).get("problem_class") or "analysis"
        objective = (method_selection or {}).get("objective") or ""
        method_family = (method_selection or {}).get("method_family") or ""

        what_was_analyzed = f"The question was handled as a {class_name.lower()} investigation."
        if objective:
            what_was_analyzed += f" Analytical objective: {objective}."
        if method_family:
            what_was_analyzed += f" Method family: {method_family}."
        what_was_analyzed += f" AA-OS executed {len(exps)} analytical experiment(s) and recorded {len(obs)} observation(s)."

        evidence_text: List[str] = []
        for row in evid[:5]:
            statement = str(getattr(row, "statement", "")).strip()
            if statement:
                evidence_text.append(statement)
        if not evidence_text:
            evidence_text.append("No persisted evidence statement was available for the final explanation.")

        if supported:
            top = max(supported, key=lambda h: float(getattr(h, "posterior_probability", 0.0) or 0.0))
            conclusion = str(getattr(top, "statement", "A leading hypothesis was supported."))
        elif direct_answer:
            conclusion = str(direct_answer)
        elif justification:
            conclusion = str(justification)
        else:
            conclusion = f"AA-OS finished with verdict {verdict_type}."

        limitations: List[str] = []
        if not verified:
            limitations.append("No evidence was recorded as independently verified in the final state.")
        if failed_verification:
            limitations.append(f"{len(failed_verification)} evidence item(s) were not independently verified or failed verification.")
        if not hyps:
            limitations.append("No persisted hypotheses were available for a structured explanation.")
        if verdict_type in {"INCONCLUSIVE", "INSUFFICIENT_DATA", "CONFLICTING_EVIDENCE"}:
            limitations.append("The final verdict is inconclusive; this explanation does not convert uncertainty into a positive finding.")

        rationale = str(justification or "").strip()
        if rationale:
            limitations.append(f"Verdict rationale: {rationale}")
        if refuted:
            limitations.append(f"{len(refuted)} competing hypothesis(es) were weakened or refuted during the investigation.")
        unvalidated = [a for a in assumptions if not bool(getattr(a, "is_validated", False))]
        high_risk = [a for a in unvalidated if str(getattr(a, "sensitivity_risk", "medium")).lower() == "high"]
        if high_risk:
            limitations.append(f"{len(high_risk)} high-risk assumption(s) remain unvalidated; review them before treating the conclusion as decision-grade.")
        elif unvalidated:
            limitations.append(f"{len(unvalidated)} assumption/limitation item(s) remain unvalidated.")

        verification = (
            f"{len(verified)} of {len(evid)} persisted evidence item(s) are marked independently verified."
            if evid
            else "No persisted evidence verification record was available."
        )

        summary = f"AA-OS completed the local analytical investigation with verdict {verdict_type}."
        return LocalExplanation(
            mode="DETERMINISTIC",
            summary=summary,
            what_was_analyzed=what_was_analyzed,
            evidence=evidence_text,
            conclusion=conclusion,
            limitations=limitations,
            verification=verification,
            calculation_trace=traces,
        )
