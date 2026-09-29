"""Universal question -> analysis compiler for AA-OS.

This is the policy brain above the existing statistical/causal engines.  It does
not perform numerical analysis itself.  It determines what the user is asking,
what can legitimately be claimed, which evidence is necessary, which checks are
conditional, and how much experimentation is justified.

The design is deliberately deterministic for the local/offline core.  An LLM
may enrich the interpretation later, but it cannot override this contract or
invent evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Sequence

from packages.schemas.src.analysis import EpistemicClaimType


@dataclass
class AnalysisPlan:
    """Executable scientific contract derived from the user's question."""
    plan_version: str
    question: str
    problem_class: str
    objective: str
    claim_type: str
    target_column: Optional[str]
    explanatory_columns: List[str]
    comparison_dimension: Optional[str]
    time_column: Optional[str]
    estimand: str
    primary_evidence: List[str]
    conditional_checks: List[str]
    competing_hypotheses: List[str]
    allowed_experiment_families: List[str]
    preferred_experiment_codes: List[str]
    max_experiments: int
    require_independent_verification: bool = True
    require_adversarial_check: bool = False
    require_multiverse_check: bool = False
    stop_when: List[str] = field(default_factory=list)
    missing_context: List[str] = field(default_factory=list)
    ambiguity_notes: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_version": self.plan_version,
            "question": self.question,
            "problem_class": self.problem_class,
            "objective": self.objective,
            "claim_type": self.claim_type,
            "target_column": self.target_column,
            "explanatory_columns": list(self.explanatory_columns),
            "comparison_dimension": self.comparison_dimension,
            "time_column": self.time_column,
            "estimand": self.estimand,
            "primary_evidence": list(self.primary_evidence),
            "conditional_checks": list(self.conditional_checks),
            "competing_hypotheses": list(self.competing_hypotheses),
            "allowed_experiment_families": list(self.allowed_experiment_families),
            "preferred_experiment_codes": list(self.preferred_experiment_codes),
            "max_experiments": self.max_experiments,
            "require_independent_verification": self.require_independent_verification,
            "require_adversarial_check": self.require_adversarial_check,
            "require_multiverse_check": self.require_multiverse_check,
            "stop_when": list(self.stop_when),
            "missing_context": list(self.missing_context),
            "ambiguity_notes": list(self.ambiguity_notes),
            "confidence": self.confidence,
        }


class UniversalAnalyst:
    """Compiles arbitrary analyst questions into a bounded scientific plan."""

    _RECON = re.compile(
        r"\b(reconcil\w*|discrepanc\w*|mismatch\w*|inconsisten\w*|mathemat\w*|formula\w*|calculation\w*|match(?:es)?|agree)\b"
        r"|\b(?:does|should)\s+[^?]*\bequal\b",
        re.IGNORECASE,
    )
    _GOV = re.compile(r"\b(gdpr|ccpa|privacy|pii|personal data|ethical|ethics|harm|fair|fairness|bias|discrimination|legal|compliance)\b", re.IGNORECASE)
    # _SEG: expanded to catch "which customer segments exist", "what segments", "customer groups", personas, etc.
    _SEG = re.compile(
        r"\b(cluster(?:ing)?|segmentation|persona\w*|group\s+similar|hidden\s+segment\w*"
        r"|which\s+\w*\s*segment|customer\s+segment\w*|what\s+segment|find\s+segment|identify\s+segment"
        r"|segment\w*\s+exist|segment\w*\s+are\s+there|customer\s+group|user\s+group)\b",
        re.IGNORECASE,
    )
    # _PRED: individual-level risk/probability at the customer/record grain (NOT time-series)
    _PRED = re.compile(r"\b(likely to churn|likely to convert|probability|probabilities|risk score|who will|which .* will|predict .* customer|predict .* user)\b", re.IGNORECASE)
    # _FORECAST: time-series projection into the future — checked BEFORE _PRED to prevent
    # "Will sales likely increase next quarter?" being swallowed by "likely" in _PRED
    _FORECAST = re.compile(
        r"\b(forecast|next\s+(month|quarter|year|week)|future|will\s+\w+\s+increase|will\s+\w+\s+decrease"
        r"|sales.*next|revenue.*next|trend.*next|predict.*next\s+(month|quarter|year))\b",
        re.IGNORECASE,
    )
    _CAUSAL = re.compile(r"\b(causal|causally|intervention|treatment effect|effect of|did .* cause|does .* cause|would .* if|impact of)\b", re.IGNORECASE)
    _ASSOC = re.compile(
        r"\b(affect|associated|association|relationship|correlat\w*|depend|linked|difference between"
        r"|different\s+rates?|differ\s+by|churning\s+at\s+different)\b",
        re.IGNORECASE,
    )
    _COMPARE = re.compile(r"\b(highest|lowest|higher|lower|largest|smallest|best|worst|compare|difference|which .* top|which .* highest|rank)\b", re.IGNORECASE)
    _DIAG = re.compile(r"\b(why|driver\w*|root cause|reason|explain|declin\w*|drop|fell|fall|spike|surge|change|what caused)\b", re.IGNORECASE)
    # DATA_QUALITY: cleaning, fixing, remediation, or showing exactly what was changed in a dataset
    _DATA_QUALITY = re.compile(
        r"\b(clean(?:ing|se)?|fix\s+data|repair\s+data|remediat\w*|scrub\s+data"
        r"|show\s+.*\s+changed|what\s+.*\s+changed|report\s+what\s+changed|data\s+issue\w*|bad\s+data|invalid\s+data"
        r"|malformed|remove\s+duplicate|deduplic|impute|fill\s+missing|correct\s+data|data\s+quality)\b",
        re.IGNORECASE,
    )

    @classmethod
    def compile(cls, question: str, intent: Any, semantic: Any, df: Any = None) -> AnalysisPlan:
        q = (question or "").strip()
        ql = q.lower()
        cols = cls._columns(df, semantic)
        categorical = cls._categorical(df, semantic)
        numeric = cls._numeric(df, semantic)

        target = cls._resolve_target(ql, semantic, cols, numeric)
        explainers = cls._resolve_explainers(ql, semantic, cols, target)
        comparison = getattr(semantic, "group_dimension_col", None) or (explainers[0] if explainers else None)
        time_col = getattr(semantic, "time_col", None)

        # Highest-specificity classification first: specialized question types
        # must never be swallowed by broad words such as "change" or "predict".
        if cls._GOV.search(ql):
            problem, objective, claim = "GOVERNANCE", "governance_review", EpistemicClaimType.OBSERVATION.value
            evidence = ["PII/sensitive-field detection", "data-use and harm risks", "scope/legal-context disclosure"]
            families, preferred = ["GOVERNANCE"], ["EXP-GOVERNANCE-SCAN"]
            max_exp = 1
            conditional = ["jurisdiction/legal-basis review requires human/organizational context"]
            competitors = ["legitimate_use", "potential_harm", "unknown_legal_basis"]
            stop = ["governance scan completed and unresolved context is explicitly disclosed"]
        elif cls._RECON.search(ql):
            problem, objective, claim = "RECONCILIATION", "reconcile_metrics", EpistemicClaimType.OBSERVATION.value
            evidence = ["explicit mathematical identity", "row-level discrepancies", "magnitude and rate of disagreement"]
            families, preferred = ["RECONCILIATION"], ["EXP-RECONCILIATION"]
            max_exp = 2
            conditional = ["rounding tolerance", "legitimate business exceptions", "unit/currency consistency"]
            competitors = ["data_error", "rounding_or_precision", "legitimate_exception", "unknown_rule"]
            stop = ["identity is tested on the intended grain", "remaining discrepancies are classified or explicitly unknown"]
        elif cls._SEG.search(ql):
            problem, objective, claim = "SEGMENTATION", "discover_segments", EpistemicClaimType.OBSERVATION.value
            evidence = ["feature schema", "cluster quality", "cluster stability", "segment interpretability"]
            families, preferred = ["SEGMENTATION"], ["EXP-SEGMENTATION"]
            max_exp = 4
            conditional = ["feature scaling", "mixed-type handling", "cluster stability"]
            competitors = ["no_stable_segments", "few_stable_segments", "multiple_plausible_segmentations"]
            stop = ["selected segmentation is stable and interpretable", "additional segmentation methods do not materially improve stability"]
        elif cls._PRED.search(ql):
            problem, objective, claim = "PREDICTION", "predict_individual_risk", EpistemicClaimType.PREDICTION.value
            evidence = ["time/label definition", "leakage-safe features", "out-of-sample discrimination", "calibration"]
            families, preferred = ["PREDICTION"], ["EXP-PREDICTION-BASELINE", "EXP-PREDICTION-CALIBRATION"]
            max_exp = 4
            conditional = ["temporal validation", "class imbalance", "probability calibration"]
            competitors = ["baseline_model", "nonlinear_model", "no_predictive_signal"]
            stop = ["out-of-sample performance is established and calibrated", "no material unresolved leakage or stability issue remains"]
        elif cls._FORECAST.search(ql):
            problem, objective, claim = "FORECASTING", "forecast_future", EpistemicClaimType.PREDICTION.value
            evidence = ["chronological history", "rolling-origin backtest", "forecast interval", "model residual diagnostics"]
            families, preferred = ["FORECAST"], ["EXP-FORECAST-BACKTEST"]
            max_exp = 3
            conditional = ["seasonality", "structural breaks", "minimum history"]
            competitors = ["naive", "drift", "seasonal", "holt_family"]
            stop = ["a validated forecast model and uncertainty interval are available"]
        elif cls._CAUSAL.search(ql):
            problem, objective, claim = "CAUSAL", "estimate_causal_effect", EpistemicClaimType.CAUSAL_INFERENCE.value
            evidence = ["causal identification", "assumption audit", "effect estimate", "uncertainty", "sensitivity"]
            families, preferred = ["CAUSAL"], ["EXP-CAUSAL-IDENTIFICATION", "EXP-CAUSAL-EFFECT"]
            max_exp = 4
            conditional = ["DAG/identification", "positivity", "consistency", "no unmeasured confounding under the stated design"]
            competitors = ["causal_effect", "confounding", "selection_or_measurement_bias"]
            stop = ["identification is established", "effect is estimated and independently checked", "sensitivity is either reassuring or explicitly unresolved"]
        elif cls._ASSOC.search(ql):
            problem, objective, claim = "ASSOCIATION", "estimate_association", EpistemicClaimType.ASSOCIATION.value
            evidence = ["group/variable relationship", "appropriate inferential test", "effect size/interval", "independent verification"]
            families, preferred = ["CORRELATION", "CHURN"], ["EXP-ASSOCIATION-PRIMARY"]
            max_exp = 3
            conditional = ["confounding", "group imbalance", "missingness sensitivity"]
            competitors = ["association", "no_association", "confounding_or_selection"]
            stop = ["primary association is estimated", "verification passes", "a material alternative explanation is addressed or reported"]
        elif cls._COMPARE.search(ql):
            problem, objective, claim = "COMPARISON", "compare_groups", EpistemicClaimType.ASSOCIATION.value
            evidence = ["group definitions", "sample sizes", "effect size", "uncertainty", "multiple-comparison control where needed"]
            families, preferred = ["COMPARISON"], ["EXP-COMPARISON-PRIMARY"]
            max_exp = 3
            conditional = ["variance/skew", "multiple comparisons", "paired vs independent design"]
            competitors = ["group_difference", "no_material_difference", "design_or_measurement_artifact"]
            stop = ["the requested comparison is estimated with uncertainty and relevant multiplicity control"]
        elif cls._DIAG.search(ql):
            problem, objective, claim = "DIAGNOSTIC", "explain_observed_change", EpistemicClaimType.ASSOCIATION.value
            evidence = ["baseline/change decomposition", "candidate drivers", "confounder checks", "adversarial challenge"]
            families, preferred = ["DIAGNOSTIC"], ["EXP-DIAGNOSTIC-PRIMARY"]
            max_exp = 5
            conditional = ["temporal leakage", "selection bias", "Simpson's paradox", "post-treatment variables"]
            competitors = ["volume_mix", "rate_effect", "composition_change", "data_artifact", "other_unobserved_driver"]
            stop = ["leading explanation is directly supported", "major alternatives are tested or explicitly unresolved"]
        elif cls._DATA_QUALITY.search(ql):
            problem, objective, claim = "DATA_QUALITY", "clean_and_remediate_data", EpistemicClaimType.OBSERVATION.value
            evidence = ["data hygiene audit", "remediation change log", "imputation variance trace", "unresolved schema errors"]
            families, preferred = ["DATA_QUALITY"], ["EXP-DATA-CLEANING"]
            max_exp = 2
            conditional = ["remediation rules", "missingness mechanism", "audit log completeness"]
            competitors = ["clean_dataset", "remediated_with_uncertainty", "irremediable_data_errors"]
            stop = ["data quality checks completed and remediation changelog generated"]
        else:
            problem, objective, claim = "DESCRIPTIVE", "describe_data", EpistemicClaimType.OBSERVATION.value
            evidence = ["distribution", "baseline", "scope", "data-quality caveats"]
            families, preferred = ["DESCRIPTIVE"], ["EXP-DESCRIPTIVE-PROFILE"]
            max_exp = 2
            conditional = ["missingness", "outliers"]
            competitors = ["observed_structure", "sampling_or_quality_artifact"]
            stop = ["requested descriptive summary is complete and traceable"]

        missing_context: List[str] = []
        if not getattr(intent, "business_objective", ""):
            missing_context.append("business_objective")
        # Audience/success are never inferred from the dataset. They are context,
        # not numerical facts; the UI can request them only when decision guidance
        # is requested.
        if problem in {"DIAGNOSTIC", "CAUSAL", "PREDICTION", "COMPARISON"}:
            missing_context.extend(["decision_context", "success_criterion"])

        ambiguity = []
        if not target:
            ambiguity.append("target variable could not be confidently resolved from question/schema")
        if cls._ASSOC.search(ql) and not explainers:
            ambiguity.append("association wording detected but explanatory variable was not resolved")

        score = 0.92
        if not target:
            score -= 0.25
        if not explainers and problem in {"ASSOCIATION", "CAUSAL", "SEGMENTATION", "COMPARISON"}:
            score -= 0.20
        if ambiguity:
            score -= 0.10
        score = max(0.05, min(0.98, score))

        estimand = cls._estimand(problem, target, explainers, comparison, time_col, semantic)

        # Adversarial analysis is a conditional scientific requirement, not a
        # default ceremony.  Multiverse analysis is reserved for specification-
        # sensitive questions where several reasonable analysis choices exist.
        require_adv = problem in {"CAUSAL", "DIAGNOSTIC", "ASSOCIATION", "PREDICTION"}
        require_multi = problem in {"CAUSAL", "DIAGNOSTIC", "PREDICTION", "SEGMENTATION"}
        if problem in {"DESCRIPTIVE", "RECONCILIATION", "GOVERNANCE", "FORECASTING"}:
            require_adv = False
            require_multi = False

        return AnalysisPlan(
            plan_version="UAP-1.0",
            question=q,
            problem_class=problem,
            objective=objective,
            claim_type=claim,
            target_column=target,
            explanatory_columns=explainers,
            comparison_dimension=comparison,
            time_column=time_col,
            estimand=estimand,
            primary_evidence=evidence,
            conditional_checks=conditional,
            competing_hypotheses=competitors,
            allowed_experiment_families=families,
            preferred_experiment_codes=preferred,
            max_experiments=max_exp,
            require_independent_verification=True,
            require_adversarial_check=require_adv,
            require_multiverse_check=require_multi,
            stop_when=stop,
            missing_context=missing_context,
            ambiguity_notes=ambiguity,
            confidence=score,
        )

    @staticmethod
    def _columns(df: Any, semantic: Any) -> List[str]:
        vals: List[str] = []
        if df is not None and hasattr(df, "columns"):
            vals.extend([str(c) for c in df.columns])
        for attr in ("available_numeric_cols", "available_categorical_cols"):
            vals.extend([str(c) for c in (getattr(semantic, attr, []) or [])])
        return list(dict.fromkeys(vals))

    @staticmethod
    def _numeric(df: Any, semantic: Any) -> List[str]:
        vals = [str(c) for c in (getattr(semantic, "available_numeric_cols", []) or [])]
        if df is not None and hasattr(df, "select_dtypes"):
            try:
                vals.extend([str(c) for c in df.select_dtypes(include="number").columns])
            except Exception:
                pass
        return list(dict.fromkeys(vals))

    @staticmethod
    def _categorical(df: Any, semantic: Any) -> List[str]:
        vals = [str(c) for c in (getattr(semantic, "available_categorical_cols", []) or [])]
        if df is not None and hasattr(df, "select_dtypes"):
            try:
                vals.extend([str(c) for c in df.select_dtypes(include=["object", "category", "bool"]).columns])
            except Exception:
                pass
        return list(dict.fromkeys(vals))

    @staticmethod
    def _resolve_target(question: str, semantic: Any, cols: Sequence[str], numeric: Sequence[str]) -> Optional[str]:
        # Prefer explicitly resolved semantic targets.
        t = getattr(semantic, "target_metric_col", None)
        if t and t in cols:
            # For churn questions prefer actual event binding when present.
            if getattr(semantic, "churn_event_col", None) and any(k in question for k in ("churn", "attrition", "retention", "cancel")):
                ev = getattr(semantic, "churn_event_col")
                if ev in cols:
                    return ev
            return t
        qtokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", question))
        for c in cols:
            if c.lower() in qtokens:
                return c
        for c in numeric:
            root = c.lower().replace("_", " ")
            if root in question:
                return c
        return numeric[0] if numeric else (cols[0] if cols else None)

    @staticmethod
    def _resolve_explainers(question: str, semantic: Any, cols: Sequence[str], target: Optional[str]) -> List[str]:
        out: List[str] = []
        for c in (
            getattr(semantic, "group_dimension_col", None),
            getattr(semantic, "secondary_metric_col", None),
        ):
            if c and c in cols and c != target and c not in out:
                out.append(c)
        qtokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", question.lower()))
        for c in cols:
            if c == target:
                continue
            if c.lower() in qtokens and c not in out:
                out.append(c)
        return out[:6]

    @staticmethod
    def _estimand(problem: str, target: Optional[str], explainers: List[str], comparison: Optional[str], time_col: Optional[str], semantic: Any) -> str:
        if problem == "ASSOCIATION":
            return f"association between {explainers[0] if explainers else 'explanatory variable'} and {target or 'outcome'} at the resolved unit of analysis"
        if problem == "CAUSAL":
            return f"causal effect of {explainers[0] if explainers else 'treatment'} on {target or 'outcome'}, conditional on identification assumptions"
        if problem == "PREDICTION":
            return f"out-of-sample probability/risk for {target or 'target outcome'} at the current-record grain"
        if problem == "FORECASTING":
            return f"future value of {target or 'target metric'} over the requested horizon using chronological validation"
        if problem == "SEGMENTATION":
            return "stable partition of observations using the features requested or justified by the schema"
        if problem == "RECONCILIATION":
            return "row/group-level agreement with the explicitly stated mathematical identity"
        if problem == "GOVERNANCE":
            return "observed governance/privacy risks in the supplied data and stated use context"
        if problem == "COMPARISON":
            return f"difference in {target or 'target metric'} across {comparison or 'the requested groups'}"
        return f"descriptive properties of {target or 'the available data'}"
