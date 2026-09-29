"""Deterministic scientific assumption and limitation ledger for AA-OS.

The ledger makes assumptions explicit before a conclusion is presented.  It is
not a statistical engine and never upgrades evidence.  It converts facts that
already exist in the canonical investigation state into a structured human-
review surface: what the analysis assumes, why the assumption matters, and
whether AA-OS has evidence that validates it.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class AssumptionItem:
    code: str
    category: str
    statement: str
    sensitivity_risk: str = "medium"
    is_validated: bool = False
    validation_evidence: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnalyticalQualityCard:
    overall_status: str
    quality_score: float
    assumption_count: int
    high_risk_unvalidated: int
    validated_count: int
    limitation_count: int
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AssumptionLedgerEngine:
    """Builds an explicit assumption/limitation ledger from canonical facts."""

    @staticmethod
    def score(items: List[AssumptionItem]) -> AnalyticalQualityCard:
        """Recompute the quality card from a (possibly patched) item list.

        Factored out of build() so callers that patch items after execution
        (see patch_validation()) can recompute a card that matches reality,
        instead of the card staying frozen at its pre-execution snapshot.
        """
        limitations = [x for x in items if not x.is_validated]
        high_risk = [x for x in limitations if x.sensitivity_risk == "high"]
        validated = [x for x in items if x.is_validated]

        penalty = min(0.55, len(high_risk) * 0.08 + max(0, len(limitations) - len(high_risk)) * 0.02)
        quality_score = round(max(0.0, min(100.0, 100.0 - penalty * 100.0)), 1)
        status = "GREEN" if not high_risk else ("AMBER" if len(high_risk) <= 2 else "RED")
        notes: List[str] = []
        if high_risk:
            notes.append(f"{len(high_risk)} high-risk assumption(s) remain unvalidated.")
        if limitations:
            notes.append(f"{len(limitations)} assumption/limitation item(s) are not independently validated.")
        if not notes:
            notes.append("No material unvalidated assumptions were detected by the current rule set.")

        return AnalyticalQualityCard(
            overall_status=status,
            quality_score=quality_score,
            assumption_count=len(items),
            high_risk_unvalidated=len(high_risk),
            validated_count=len(validated),
            limitation_count=len(limitations),
            notes=notes,
        )

    @staticmethod
    def patch_validation(
        items: List[AssumptionItem],
        updates: Dict[str, tuple[bool, Optional[str]]],
    ) -> List[AssumptionItem]:
        """Return a new item list with named codes' validation state updated.

        AssumptionItem is frozen by design (a ledger entry must not be
        silently rewritten mid-review), but some assumptions -- independent
        verification chief among them -- can only be checked against real
        evidence *after* the experiment that build() ran ahead of has
        executed. This applies that later-known truth explicitly, once, by
        constructing replacement items rather than mutating in place, so the
        ledger the human verifier sees reflects what actually happened
        instead of a permanent pre-execution guess.
        """
        patched: List[AssumptionItem] = []
        for item in items:
            if item.code in updates:
                validated, evidence = updates[item.code]
                patched.append(
                    AssumptionItem(
                        code=item.code,
                        category=item.category,
                        statement=item.statement,
                        sensitivity_risk=item.sensitivity_risk,
                        is_validated=validated,
                        validation_evidence=evidence or item.validation_evidence,
                    )
                )
            else:
                patched.append(item)
        return patched

    @staticmethod
    def build(
        *,
        question: str,
        method_decision: Any,
        semantic: Any,
        quality_assessment: Any,
        temporal_scope: Any = None,
        causal_gate_result: Any = None,
        missingness_result: Any = None,
        join_reports: Optional[Iterable[Any]] = None,
    ) -> tuple[List[AssumptionItem], AnalyticalQualityCard]:
        items: List[AssumptionItem] = []

        def add(code: str, category: str, statement: str, risk: str = "medium", validated: bool = False, evidence: str | None = None):
            if any(x.code == code for x in items):
                return
            items.append(AssumptionItem(code, category, statement, risk, validated, evidence))

        md = getattr(semantic, "metric_definition", None)
        metric = getattr(semantic, "target_metric_col", None) or getattr(getattr(method_decision, "estimand", None), "metric_ref", None)
        aggregation = getattr(getattr(md, "aggregation_type", None), "value", None) or getattr(md, "aggregation_type", None)
        grain = getattr(getattr(method_decision, "estimand", None), "unit_of_analysis", None)

        add(
            "DATA_SCOPE_STABILITY",
            "DATA",
            "The analysis is conditioned on the supplied dataset snapshot and its observed records; material unseen or future data may change the result.",
            "medium",
            validated=True,
            evidence="Dataset was successfully acquired and passed the pre-investigation quality gate.",
        )

        if metric:
            resolution_status = getattr(md, "semantic_resolution_status", "RESOLVED")
            metric_text = f"Target metric '{metric}'"
            if aggregation:
                metric_text += f" is interpreted using {str(aggregation).upper()} aggregation"
            metric_text += "."
            if resolution_status == "UNRESOLVED_DEFAULT_SUM":
                # The metric semantics resolver could not positively confirm
                # additivity for this column from its name or observed
                # values; SUM was used only as an unconfirmed fallback so the
                # investigation would not stall. Marking this "validated"
                # (as before) hid that the aggregation was guessed, not
                # resolved -- a human reviewer needs to see this one.
                # "medium", not "high": UNRESOLVED_DEFAULT_SUM is a by-design
                # fallback specifically meant not to block the investigation
                # (see metric_semantics.py's own rationale text) -- it must
                # surface honestly to the human reviewer as unvalidated, but
                # must not silently start gating verdicts the way a "high"
                # entry does via material_unvalidated_high_risk.
                add(
                    "METRIC_SEMANTICS",
                    "ESTIMAND",
                    metric_text + " This aggregation is an unconfirmed default (UNRESOLVED_DEFAULT_SUM): "
                    "no rate/ratio/average/additive-magnitude signal was found for this column, so SUM "
                    "was assumed rather than positively resolved.",
                    "medium",
                    validated=False,
                )
            else:
                add(
                    "METRIC_SEMANTICS",
                    "ESTIMAND",
                    metric_text,
                    "low",
                    validated=True,
                    evidence=f"Resolved by the canonical semantic metric definition (status: {resolution_status}).",
                )
        else:
            add("METRIC_RESOLUTION", "ESTIMAND", "A target metric could not be fully resolved from the available semantic model; downstream claims must remain conservative.", "high")

        if grain is not None:
            keys = getattr(grain, "keys", None) or []
            add(
                "UNIT_OF_ANALYSIS",
                "ESTIMAND",
                f"The result is interpreted at the resolved unit of analysis{(' with keys ' + ', '.join(map(str, keys))) if keys else ''}.",
                "medium",
                validated=True,
                evidence="Unit of analysis was bound in the canonical estimand specification.",
            )

        q_score = float(getattr(quality_assessment, "overall_quality_score", 0.0) or 0.0)
        if q_score >= 85:
            add("DATA_QUALITY_BASELINE", "DATA", f"The dataset quality gate score is {q_score:.1f}/100 and is treated as sufficiently fit for the selected analysis.", "low", True, "DataQualityGate FIT/healthy score")
        elif q_score >= 60:
            add("DATA_QUALITY_CAUTION", "DATA", f"The dataset quality score is {q_score:.1f}/100; warnings or moderate defects may affect the result.", "medium", False)
        else:
            add("DATA_QUALITY_RISK", "DATA", f"The dataset quality score is {q_score:.1f}/100; analytical conclusions are materially exposed to data-quality limitations.", "high", False)

        critical_issues = list(getattr(quality_assessment, "critical_issues", []) or [])
        warnings = list(getattr(quality_assessment, "warnings", []) or [])
        for idx, warning in enumerate(warnings[:5], 1):
            add(f"DQ_WARNING_{idx}", "DATA", str(warning), "medium", False)
        for idx, issue in enumerate(critical_issues[:5], 1):
            add(f"DQ_ISSUE_{idx}", "DATA", str(issue), "high", False)

        if getattr(temporal_scope, "found", False):
            ambiguity = getattr(temporal_scope, "ambiguous", False)
            note = getattr(temporal_scope, "ambiguity_note", None)
            add(
                "TEMPORAL_SCOPE",
                "TIME",
                "The requested time reference is interpreted using the dataset's resolved time column rather than wall-clock time." + (f" Ambiguity: {note}" if ambiguity else ""),
                "high" if ambiguity else "low",
                validated=not ambiguity,
                evidence="TemporalResolver matched the request to the observed dataset time range." if not ambiguity else None,
            )
        elif getattr(semantic, "time_col", None):
            add("TEMPORAL_SCOPE_ABSENT", "TIME", "No explicit temporal restriction was identified in the question; the investigation may use the available historical time scope as defined by the analytical plan.", "medium")

        if getattr(method_decision, "problem_class", None) and str(getattr(method_decision.problem_class, "value", method_decision.problem_class)) == "FORECASTING":
            add("FORECAST_GENERALIZATION", "FORECAST", "Historical patterns are assumed to provide useful information about the forecast horizon; unexpected regime changes may invalidate the forecast.", "high")
            add("FORECAST_BACKTESTING", "FORECAST", "Forecast method choice is assumed to be defensible only to the extent supported by out-of-sample/backtest performance available for the dataset.", "medium")

        if getattr(method_decision, "problem_class", None) and str(getattr(method_decision.problem_class, "value", method_decision.problem_class)) == "CAUSAL":
            causal_status = getattr(causal_gate_result, "status", None)
            causal_status_value = getattr(causal_status, "value", causal_status)
            if causal_status_value in ("OBSERVATIONAL_ONLY", "NOT_IDENTIFIABLE", None):
                add("CAUSAL_IDENTIFICATION", "CAUSAL", "The available evidence does not establish a formally identified causal effect; associations must not be presented as causal effects.", "high", False)
            else:
                add("CAUSAL_IDENTIFICATION", "CAUSAL", "The stated causal identification assumptions are treated as necessary conditions for the reported effect.", "high", validated=True, evidence=str(getattr(causal_gate_result, "diagnostic_message", "Causal identifiability gate passed.")))

        if missingness_result is not None:
            classification = str(getattr(missingness_result, "classification", "UNKNOWN"))
            if classification not in {"ROBUST", "RESISTANT"}:
                add("MISSINGNESS_SENSITIVITY", "SELECTION", f"Missingness sensitivity is classified as {classification}: {getattr(missingness_result, 'rationale', '')}", "high")
            else:
                add("MISSINGNESS_SENSITIVITY", "SELECTION", "The latest missingness-sensitivity analysis did not identify material instability under the tested missingness assumptions.", "low", True, getattr(missingness_result, "rationale", None))

        reports = list(join_reports or [])
        if reports:
            unsafe = [r for r in reports if str(getattr(getattr(r, "status", None), "value", getattr(r, "status", None))) != "SAFE"]
            if unsafe:
                add("RELATIONAL_JOIN_SAFETY", "RELATIONAL", "At least one declared join has unresolved safety risk; dependent claims must not treat joined records as unbiased evidence.", "high")
            else:
                add("RELATIONAL_JOIN_SAFETY", "RELATIONAL", "Declared joins passed the key/cardinality/fanout safety checks used for this investigation.", "low", True, "Relational join-safety gate")

        add("INDEPENDENT_VERIFICATION", "VERIFICATION", "Numerical claims are treated as evidentially stronger only when an independent verification path agrees within its declared tolerance.", "high", False)
        add("NO_SILENT_SAMPLING", "EXECUTION", "AA-OS must not silently sample, truncate, or simplify the supplied dataset when doing so could change the requested estimand.", "high", True, "AA-OS execution contract")

        return items, AssumptionLedgerEngine.score(items)
