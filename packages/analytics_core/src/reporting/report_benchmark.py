"""Machine-checkable quality gate for stakeholder-facing analytical reports.

The validator checks a normalized report contract, not arbitrary rendered PDF text.
Callers should build this contract from authoritative execution/evidence records before
publishing. Human judgement and dataset-level golden-benchmark checks remain separate.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

SCHEMA_VERSION = "1.0.0"


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ReportMetadata(ContractModel):
    title: str = Field(min_length=1)
    author_or_system: str = Field(min_length=1)
    generated_at: str = Field(min_length=1)
    version: str = Field(min_length=1)
    data_as_of: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    stakeholder: Optional[str] = None
    project_id: Optional[str] = None


class ExecutiveSummary(ContractModel):
    text: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    # Each item must include literal, fact_id, and semantic_role.
    key_numbers: List[Dict[str, str]] = Field(min_length=1)
    non_metric_numeric_tokens: List[str] = Field(default_factory=list)
    recommended_action: str = Field(min_length=1)


class AnalysisQuestion(ContractModel):
    question_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    decision_to_inform: str = Field(min_length=1)
    hypotheses: List[str] = Field(default_factory=list)


class MetricDefinition(ContractModel):
    name: str = Field(min_length=1)
    definition: str = Field(min_length=1)
    unit: Optional[str] = None
    numerator: Optional[str] = None
    denominator: Optional[str] = None


class GlossaryEntry(ContractModel):
    term: str = Field(min_length=1)
    definition: str = Field(min_length=1)


class Scope(ContractModel):
    time_range: str = Field(min_length=1)
    population: str = Field(min_length=1)
    exclusions: List[str] = Field(default_factory=list)
    metric_definitions: List[MetricDefinition] = Field(min_length=1)
    glossary: List[GlossaryEntry] = Field(default_factory=list)


class DataSource(ContractModel):
    name: str = Field(min_length=1)
    extraction_time: str = Field(min_length=1)
    row_count: int = Field(ge=0)
    known_limitations: List[str]
    owner: str = Field(min_length=1)


class CleaningStep(ContractModel):
    operation: str = Field(min_length=1)
    rows_affected: int = Field(ge=0)
    reason: str = Field(min_length=1)
    disclosed_in_report: bool


class ConfidenceInterval(ContractModel):
    lower: float
    upper: float
    level: float = Field(gt=0, lt=1)


class StatisticalMethod(ContractModel):
    name: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    inference_type: Literal["descriptive", "inferential", "forecast"]
    assumptions_checked: List[str] = Field(default_factory=list)
    fallback_used: Optional[str] = None
    sample_size: Optional[int] = Field(default=None, ge=0)
    p_value: Optional[float] = Field(default=None, ge=0, le=1)
    effect_size: Optional[str] = None
    confidence_interval: Optional[ConfidenceInterval] = None


class Methodology(ContractModel):
    tools: List[str] = Field(min_length=1)
    cleaning_steps: List[CleaningStep]
    statistical_methods: List[StatisticalMethod] = Field(default_factory=list)
    assumptions: List[str] = Field(min_length=1)
    metrics_computation: str = Field(min_length=1)


class ReconciliationCheck(ContractModel):
    name: str = Field(min_length=1)
    observed: Any
    expected: Any
    tolerance: Optional[float] = Field(default=None, ge=0)
    passed: bool
    exception_explanation: Optional[str] = None


class DataQuality(ContractModel):
    missing_value_percentages: Dict[str, float]
    duplicate_count: int = Field(ge=0)
    duplicate_handling: str = Field(min_length=1)
    outlier_handling: str = Field(min_length=1)
    reconciliation_checks: List[ReconciliationCheck] = Field(min_length=1)


class EvidenceRecord(ContractModel):
    evidence_id: str = Field(min_length=1)
    status: Literal["VERIFIED", "PASSED", "UNVERIFIED", "FAILED"]
    source: str = Field(min_length=1)
    calculation_id: str = Field(min_length=1)
    description: str = Field(min_length=1)


class ComputedFact(ContractModel):
    fact_id: str = Field(min_length=1)
    value: Any
    unit: str = Field(min_length=1)
    lineage_id: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    calculation_id: str = Field(min_length=1)
    evidence_refs: List[str] = Field(min_length=1)
    verified: bool


class NumericReference(ContractModel):
    literal: str = Field(min_length=1)
    fact_id: str = Field(min_length=1)
    semantic_role: str = Field(min_length=1)


class CausalSupport(ContractModel):
    design: Literal["randomized_experiment", "credible_quasi_experiment", "controlled_analysis", "observational_only", "none"]
    supports_causal_claim: bool
    evidence_refs: List[str] = Field(default_factory=list)
    assumption_checks: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(min_length=1)


class ReportClaim(ContractModel):
    claim_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    claim_type: Literal["descriptive", "associational", "causal", "recommendation", "methodological"]
    fact_ids: List[str] = Field(default_factory=list)
    evidence_refs: List[str] = Field(default_factory=list)
    numeric_refs: List[NumericReference] = Field(default_factory=list)
    non_metric_numeric_tokens: List[str] = Field(default_factory=list)
    causal_support: Optional[CausalSupport] = None


class DataPoint(ContractModel):
    label: str = Field(min_length=1)
    value: Any
    fact_id: str = Field(min_length=1)
    n: Optional[int] = Field(default=None, ge=0)


class ResultTable(ContractModel):
    table_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    rows: List[DataPoint] = Field(min_length=1)
    source_note: str = Field(min_length=1)


class AxisDefinition(ContractModel):
    label: str = Field(min_length=1)
    unit: str = Field(min_length=1)


class Visualization(ContractModel):
    chart_id: str = Field(min_length=1)
    chart_type: Literal["bar", "line", "scatter", "histogram", "box", "waterfall", "heatmap", "forest", "none"]
    title: str = Field(min_length=1)
    x_axis: AxisDefinition
    y_axis: AxisDefinition
    source_note: str = Field(min_length=1)
    data: List[DataPoint] = Field(default_factory=list)


class PredictiveOutput(ContractModel):
    point_forecast: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    interval_level: Optional[float] = Field(default=None, gt=0, lt=1)
    uncertainty_explanation: str = Field(min_length=1)


class Finding(ContractModel):
    finding_id: str = Field(min_length=1)
    headline: str = Field(min_length=1)
    analysis_type: Literal["descriptive", "comparative", "diagnostic", "predictive", "exploratory", "data_quality", "experimental", "causal"]
    interpretation: str = Field(min_length=1)
    confidence: Literal["high", "moderate", "low", "unknown"]
    confidence_explanation: str = Field(min_length=1)
    metric_names: List[str] = Field(min_length=1)
    fact_ids: List[str] = Field(min_length=1)
    evidence_refs: List[str] = Field(min_length=1)
    chart_refs: List[str] = Field(default_factory=list)
    table_refs: List[str] = Field(default_factory=list)
    sample_size: Optional[int] = Field(default=None, ge=0)
    predictive_output: Optional[PredictiveOutput] = None


class Conclusion(ContractModel):
    question_id: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    fact_ids: List[str] = Field(min_length=1)
    evidence_refs: List[str] = Field(min_length=1)


class Recommendation(ContractModel):
    action: str = Field(min_length=1)
    priority: Literal["critical", "high", "medium", "low", "none"]
    expected_impact: str = Field(min_length=1)
    finding_refs: List[str] = Field(min_length=1)
    numeric_refs: List[NumericReference] = Field(default_factory=list)
    non_metric_numeric_tokens: List[str] = Field(default_factory=list)
    owner: Optional[str] = None


class DataDictionaryEntry(ContractModel):
    field: str = Field(min_length=1)
    definition: str = Field(min_length=1)
    data_type: str = Field(min_length=1)
    units_or_values: str = Field(min_length=1)


class Appendix(ContractModel):
    queries_or_code: List[str] = Field(default_factory=list)
    data_dictionary: List[DataDictionaryEntry] = Field(min_length=1)
    full_tables: List[Dict[str, Any]] = Field(default_factory=list)
    version_history: List[str] = Field(min_length=1)


class ReportBenchmarkContract(ContractModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    metadata: ReportMetadata
    executive_summary: ExecutiveSummary
    objective: List[AnalysisQuestion] = Field(min_length=1)
    scope: Scope
    data_sources: List[DataSource] = Field(min_length=1)
    methodology: Methodology
    data_quality: DataQuality
    evidence: List[EvidenceRecord] = Field(min_length=1)
    facts: List[ComputedFact] = Field(min_length=1)
    claims: List[ReportClaim] = Field(min_length=1)
    findings: List[Finding] = Field(min_length=1)
    result_tables: List[ResultTable] = Field(default_factory=list)
    visualizations: List[Visualization] = Field(default_factory=list)
    conclusions: List[Conclusion] = Field(min_length=1)
    recommendations: List[Recommendation] = Field(min_length=1)
    limitations: List[str] = Field(min_length=1)
    causation_caveat: str = Field(min_length=1)
    unanswered_questions: List[str] = Field(min_length=1)
    appendix: Appendix


class BenchmarkRunMetrics(ContractModel):
    """Golden-suite metrics; these cannot be inferred from one report alone."""
    rerun_exact_match_rate: float = Field(ge=0, le=1)
    key_figure_traceability_rate: float = Field(ge=0, le=1)
    planted_issue_detection_rate: float = Field(ge=0, le=1)
    false_insight_rate_on_no_signal: float = Field(ge=0, le=1)
    quality_rubric_average: float = Field(ge=0, le=2)
    reconciliation_pass_rate: float = Field(ge=0, le=1)
    statistical_validity_passed: bool
    golden_dataset_count: int = Field(ge=0)
    human_reviewers: int = Field(ge=0)
    inter_rater_agreement: Optional[float] = Field(default=None, ge=0, le=1)


_CAUSAL_HARD = re.compile(r"\b(caused by|caused|causes|because of|due to|driven by|led to|resulted in|as a result of|because it|because the)\b", re.I)
_CAUSAL_HEDGE = re.compile(r"\b(may|might|could|possible|possibly|potential|potentially|associated with|correlated with|consistent with|suggests|suggestive of|appears to|cannot establish)\b", re.I)
_NUMERIC_TOKEN = re.compile(r"(?<![A-Za-z])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")


class ReportValidationResult(BaseModel):
    valid: bool
    publishable: bool
    grade: Literal["FAIL", "BRONZE", "SILVER", "GOLD"]
    completeness_rate: float
    numeric_traceability_rate: float
    quality_rubric: Dict[str, float]
    quality_average: float
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


def _norm_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(str(value).replace(",", "").replace("%", ""))
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _same_numeric(literal: Any, fact_value: Any) -> bool:
    lv, fv = _norm_number(literal), _norm_number(fact_value)
    if lv is None or fv is None:
        return str(literal).strip() == str(fact_value).strip()
    if isinstance(literal, str) and literal.strip().endswith("%"):
        return math.isclose(lv, fv * 100 if abs(fv) <= 1 else fv, rel_tol=1e-4, abs_tol=1e-6)
    return math.isclose(lv, fv, rel_tol=1e-6, abs_tol=1e-8)


def _numeric_text_issues(text: str, fact_ids: Sequence[str], facts: Dict[str, ComputedFact], context: str, ignored: Sequence[str] = ()) -> List[str]:
    """Ensure numeric tokens in visible narrative map to supplied computed facts."""
    errors: List[str] = []
    allowed_facts = [facts[fid] for fid in fact_ids if fid in facts]
    ignored_norms = {_norm_number(value) for value in ignored}
    for token in _NUMERIC_TOKEN.findall(text or ""):
        numeric = _norm_number(token)
        if token in ignored or (numeric is not None and numeric in ignored_norms):
            continue
        if any(_same_numeric(token, fact.value) for fact in allowed_facts):
            continue
        errors.append(f"{context} contains untraceable numeric token {token!r}.")
    return errors


def _reference_issues(report: ReportBenchmarkContract) -> List[str]:
    errors: List[str] = []
    facts = {f.fact_id: f for f in report.facts}
    evidence = {e.evidence_id: e for e in report.evidence}
    findings = {f.finding_id: f for f in report.findings}
    charts = {c.chart_id: c for c in report.visualizations}
    tables = {t.table_id: t for t in report.result_tables}
    question_ids = [q.question_id for q in report.objective]
    conclusion_ids = [c.question_id for c in report.conclusions]
    if len(question_ids) != len(set(question_ids)):
        errors.append("Objective contains duplicate question_id values.")
    if len(conclusion_ids) != len(set(conclusion_ids)):
        errors.append("Conclusions contain duplicate question_id values.")
    if set(question_ids) != set(conclusion_ids) or len(question_ids) != len(conclusion_ids):
        errors.append(f"Question-answer mismatch: {len(question_ids)} questions but {len(conclusion_ids)} unique conclusions; IDs must match exactly.")

    metric_names = {m.name for m in report.scope.metric_definitions}
    summary_refs = report.executive_summary.key_numbers
    summary_registered: List[str] = []
    for ref in summary_refs:
        literal, fact_id = ref.get("literal"), ref.get("fact_id")
        if not literal or not fact_id:
            errors.append("Every executive-summary key number must contain literal and fact_id.")
            continue
        summary_registered.append(literal)
        fact = facts.get(fact_id)
        if fact is None:
            errors.append(f"Executive summary key number {literal!r} references missing fact {fact_id}.")
        elif not fact.verified:
            errors.append(f"Executive summary key number {literal!r} references unverified fact {fact_id}.")
        elif not _same_numeric(literal, fact.value):
            errors.append(f"Executive summary key number {literal!r} does not match fact {fact_id} value.")
    for token in _NUMERIC_TOKEN.findall(report.executive_summary.text):
        if token not in summary_registered and token not in report.executive_summary.non_metric_numeric_tokens:
            token_value = _norm_number(token)
            matched = any(token_value is not None and _norm_number(known) == token_value for known in summary_registered + report.executive_summary.non_metric_numeric_tokens)
            if not matched:
                errors.append(f"Executive summary contains untraceable numeric token {token!r}.")
    for finding in report.findings:
        missing_metrics = sorted(set(finding.metric_names) - metric_names)
        if missing_metrics:
            errors.append(f"Finding {finding.finding_id} uses undefined metrics: {missing_metrics}.")
        for fid in finding.fact_ids:
            if fid not in facts:
                errors.append(f"Finding {finding.finding_id} references missing fact {fid}.")
        for eid in finding.evidence_refs:
            if eid not in evidence:
                errors.append(f"Finding {finding.finding_id} references missing evidence {eid}.")
            elif evidence[eid].status not in {"VERIFIED", "PASSED"}:
                errors.append(f"Finding {finding.finding_id} references non-verified evidence {eid}.")
        for chart_id in finding.chart_refs:
            if chart_id not in charts:
                errors.append(f"Finding {finding.finding_id} references missing chart {chart_id}.")
        for table_id in finding.table_refs:
            if table_id not in tables:
                errors.append(f"Finding {finding.finding_id} references missing table {table_id}.")
        if not finding.chart_refs and not finding.table_refs:
            errors.append(f"Finding {finding.finding_id} must cite at least one chart or result table.")
        if finding.sample_size is None:
            errors.append(f"Finding {finding.finding_id} must report sample size (including zero when appropriate).")
        if finding.analysis_type == "predictive":
            pred = finding.predictive_output
            if pred is None or pred.lower_bound is None or pred.upper_bound is None or pred.interval_level is None:
                errors.append(f"Forecast finding {finding.finding_id} lacks an uncertainty interval and interval level.")
            elif pred.lower_bound > pred.upper_bound:
                errors.append(f"Forecast finding {finding.finding_id} has lower_bound greater than upper_bound.")
        errors.extend(_numeric_text_issues(finding.headline, finding.fact_ids, facts, f"Finding {finding.finding_id} headline", [str(finding.sample_size)] if finding.sample_size is not None else []))
        errors.extend(_numeric_text_issues(finding.interpretation, finding.fact_ids, facts, f"Finding {finding.finding_id} interpretation", [str(finding.sample_size)] if finding.sample_size is not None else []))

    for fact in report.facts:
        if fact.calculation_id not in {e.calculation_id for e in report.evidence}:
            errors.append(f"Fact {fact.fact_id} calculation_id has no matching evidence calculation_id.")
        for eid in fact.evidence_refs:
            if eid not in evidence:
                errors.append(f"Fact {fact.fact_id} references missing evidence {eid}.")
            elif evidence[eid].status not in {"VERIFIED", "PASSED"} and fact.verified:
                errors.append(f"Fact {fact.fact_id} is marked verified but references non-verified evidence {eid}.")
        if fact.verified and not fact.lineage_id.strip():
            errors.append(f"Verified fact {fact.fact_id} has no lineage ID.")

    for claim in report.claims:
        if not claim.fact_ids:
            errors.append(f"Claim {claim.claim_id} has no linked computed fact.")
        if not claim.evidence_refs:
            errors.append(f"Claim {claim.claim_id} has no evidence reference.")
        for fid in claim.fact_ids:
            if fid not in facts:
                errors.append(f"Claim {claim.claim_id} references missing fact {fid}.")
        for eid in claim.evidence_refs:
            if eid not in evidence:
                errors.append(f"Claim {claim.claim_id} references missing evidence {eid}.")
            elif evidence[eid].status not in {"VERIFIED", "PASSED"}:
                errors.append(f"Claim {claim.claim_id} relies on non-verified evidence {eid}.")
        for numref in claim.numeric_refs:
            fact = facts.get(numref.fact_id)
            if not fact:
                errors.append(f"Claim {claim.claim_id} numeric value {numref.literal!r} points to missing fact {numref.fact_id}.")
            elif numref.fact_id not in claim.fact_ids:
                errors.append(f"Claim {claim.claim_id} numeric fact {numref.fact_id} is not included in fact_ids.")
            elif not _same_numeric(numref.literal, fact.value):
                errors.append(f"Claim {claim.claim_id} numeric value {numref.literal!r} does not match fact {numref.fact_id} value {fact.value!r}.")
        numeric_tokens = _NUMERIC_TOKEN.findall(claim.text)
        registered = [n.literal for n in claim.numeric_refs] + claim.non_metric_numeric_tokens
        unregistered = []
        for token in numeric_tokens:
            if token not in registered:
                # Compare normalized representations too, e.g. 44106.69 and 44,106.69.
                token_val = _norm_number(token)
                if not any(token_val is not None and _norm_number(item) == token_val for item in registered):
                    unregistered.append(token)
        if unregistered:
            errors.append(f"Claim {claim.claim_id} contains untraceable numeric tokens: {unregistered}.")
        if claim.claim_type == "causal":
            support = claim.causal_support
            if support is None:
                errors.append(f"Causal claim {claim.claim_id} has no causal-support record.")
            elif not support.supports_causal_claim or support.design in {"observational_only", "none"}:
                errors.append(f"Causal claim {claim.claim_id} is not supported by an admissible causal design.")
            else:
                if not support.evidence_refs or any(eid not in evidence or evidence[eid].status not in {"VERIFIED", "PASSED"} for eid in support.evidence_refs):
                    errors.append(f"Causal claim {claim.claim_id} lacks verified support evidence.")
                if not support.assumption_checks:
                    errors.append(f"Causal claim {claim.claim_id} has no recorded assumption checks.")
        elif _CAUSAL_HARD.search(claim.text) and not _CAUSAL_HEDGE.search(claim.text):
            errors.append(f"Claim {claim.claim_id} uses unhedged causal language but is not certified as a supported causal claim.")

    for conclusion in report.conclusions:
        errors.extend(_numeric_text_issues(conclusion.answer, conclusion.fact_ids, facts, f"Conclusion for question {conclusion.question_id}"))
        for fid in conclusion.fact_ids:
            if fid not in facts:
                errors.append(f"Conclusion for question {conclusion.question_id} references missing fact {fid}.")
        for eid in conclusion.evidence_refs:
            if eid not in evidence or evidence[eid].status not in {"VERIFIED", "PASSED"}:
                errors.append(f"Conclusion for question {conclusion.question_id} references missing or unverified evidence {eid}.")
    for rec in report.recommendations:
        linked_fact_ids: set[str] = set()
        for fid in rec.finding_refs:
            if fid not in findings:
                errors.append(f"Recommendation references missing finding {fid}.")
            else:
                linked_fact_ids.update(findings[fid].fact_ids)
        rec_registered = [nr.literal for nr in rec.numeric_refs] + rec.non_metric_numeric_tokens
        for nr in rec.numeric_refs:
            fact = facts.get(nr.fact_id)
            if fact is None:
                errors.append(f"Recommendation numeric value {nr.literal!r} references missing fact {nr.fact_id}.")
            elif nr.fact_id not in linked_fact_ids:
                errors.append(f"Recommendation numeric fact {nr.fact_id} is not supported by its referenced findings.")
            elif not _same_numeric(nr.literal, fact.value):
                errors.append(f"Recommendation numeric value {nr.literal!r} does not match fact {nr.fact_id}.")
        for text_field, text_value in (("action", rec.action), ("expected impact", rec.expected_impact)):
            for token in _NUMERIC_TOKEN.findall(text_value):
                if token in rec_registered:
                    continue
                normalized = _norm_number(token)
                if any(normalized is not None and _norm_number(known) == normalized for known in rec_registered):
                    continue
                errors.append(f"Recommendation {text_field} contains untraceable numeric token {token!r}.")

    for table in report.result_tables:
        for row in table.rows:
            fact = facts.get(row.fact_id)
            if fact is None:
                errors.append(f"Table {table.table_id} row {row.label} references missing fact {row.fact_id}.")
            elif not _same_numeric(row.value, fact.value):
                errors.append(f"Table {table.table_id} row {row.label} does not match fact {row.fact_id} value.")
    for chart in report.visualizations:
        for point in chart.data:
            fact = facts.get(point.fact_id)
            if fact is None:
                errors.append(f"Chart {chart.chart_id} point {point.label} references missing fact {point.fact_id}.")
            elif not _same_numeric(point.value, fact.value):
                errors.append(f"Chart {chart.chart_id} point {point.label} does not match fact {point.fact_id} value.")
    return errors


def _quality_rubric(report: ReportBenchmarkContract) -> Dict[str, float]:
    summary = report.executive_summary
    lines = [line for line in summary.text.splitlines() if line.strip()]
    recs = report.recommendations
    finding_ids = {f.finding_id for f in report.findings}
    claims_have_trace = all((not c.numeric_refs and not _NUMERIC_TOKEN.search(c.text)) or all(n.fact_id in c.fact_ids for n in c.numeric_refs) for c in report.claims)
    visual_quality = 2.0 if not report.visualizations or all(c.title and c.x_axis.label and c.x_axis.unit and c.y_axis.label and c.y_axis.unit and c.source_note for c in report.visualizations) else 0.0
    audience_fit = 2.0 if report.metadata.stakeholder else 1.0
    return {
        "answer_first": 2.0 if summary.question and summary.answer and summary.key_numbers and summary.recommended_action and len(lines) <= 6 else 1.0,
        "insight_not_description": 2.0 if all(f.interpretation and f.confidence_explanation for f in report.findings) and any(f.analysis_type in {"diagnostic", "predictive", "causal"} for f in report.findings) else 1.0,
        "decision_linked": 2.0 if recs and all(r.finding_refs and set(r.finding_refs) <= finding_ids for r in recs) else 0.0,
        "prioritization": 2.0 if len(recs) > 0 and any(r.priority in {"critical", "high"} for r in recs) else 1.0,
        "audience_fit": audience_fit,
        "calibrated_honesty": 2.0 if report.limitations and report.causation_caveat and report.unanswered_questions and claims_have_trace else 1.0,
        "visual_quality": visual_quality,
        "readability": 2.0 if len(lines) <= 6 and len(summary.text) <= 1200 else 1.0,
    }


def validate_report_contract(payload: Dict[str, Any], benchmark_metrics: Optional[Dict[str, Any]] = None) -> ReportValidationResult:
    """Validate report structure, evidence lineage, hard-fail rules, and a quality rubric.

    A valid report may earn Bronze. Silver/Gold are additional quality/research benchmark
    tiers and require explicit BenchmarkRunMetrics from a golden-set run.
    """
    required_sections = ["metadata", "executive_summary", "objective", "scope", "data_sources", "methodology", "data_quality", "evidence", "facts", "claims", "findings", "result_tables", "conclusions", "recommendations", "limitations", "causation_caveat", "unanswered_questions", "appendix"]
    present = sum(1 for section in required_sections if section in payload and payload[section] not in (None, "", []))
    completeness_rate = present / len(required_sections)
    errors: List[str] = []
    warnings: List[str] = []
    try:
        report = ReportBenchmarkContract.model_validate(payload)
    except ValidationError as exc:
        for error in exc.errors(include_url=False):
            loc = ".".join(str(x) for x in error.get("loc", ()))
            errors.append(f"Schema validation at {loc or '<root>'}: {error.get('msg', 'invalid value')}.")
        return ReportValidationResult(valid=False, publishable=False, grade="FAIL", completeness_rate=completeness_rate, numeric_traceability_rate=0.0, quality_rubric={}, quality_average=0.0, errors=errors, warnings=warnings)

    if len([line for line in report.executive_summary.text.splitlines() if line.strip()]) > 6:
        errors.append("Executive summary exceeds the six-line maximum.")
    if not report.data_quality.missing_value_percentages:
        errors.append("Data quality section must report missing-value percentages for key fields.")
    if not report.result_tables:
        errors.append("Report must include at least one auditable result table, even when charts are also present.")
    for field, percentage in report.data_quality.missing_value_percentages.items():
        if not 0 <= percentage <= 100:
            errors.append(f"Missing-value percentage for {field} must be between 0 and 100.")
    for source in report.data_sources:
        if not source.known_limitations:
            errors.append(f"Data source {source.name!r} must explicitly state known limitations or ['None identified'].")
    for method in report.methodology.statistical_methods:
        if not method.assumptions_checked and not (method.fallback_used or "").strip():
            errors.append(f"Statistical method {method.name!r} has neither recorded assumption checks nor a documented fallback.")
        if method.inference_type == "inferential":
            if method.p_value is None or not method.effect_size or method.confidence_interval is None:
                errors.append(f"Inferential method {method.name!r} must report p-value, effect size, and confidence interval.")
            elif method.confidence_interval.lower > method.confidence_interval.upper:
                errors.append(f"Inferential method {method.name!r} has an invalid confidence interval: lower exceeds upper.")
        if method.inference_type == "forecast" and not (method.fallback_used or "").strip():
            warnings.append(f"Forecast method {method.name!r} should document model selection and validation design in its rationale or fallback field.")
    for step in report.methodology.cleaning_steps:
        if step.rows_affected > 0 and not step.disclosed_in_report and step.operation.casefold() not in {"none", "no-op", "no operation"}:
            errors.append(f"Cleaning step {step.operation!r} affects {step.rows_affected} rows but is not disclosed.")
    for check in report.data_quality.reconciliation_checks:
        observed_num, expected_num = _norm_number(check.observed), _norm_number(check.expected)
        mismatch = (not math.isclose(observed_num, expected_num, rel_tol=0, abs_tol=float(check.tolerance or 0)) if observed_num is not None and expected_num is not None else check.observed != check.expected)
        if check.passed and mismatch:
            errors.append(f"Reconciliation check {check.name!r} is marked passed but observed and expected values exceed tolerance.")
        if not check.passed and not (check.exception_explanation or "").strip():
            errors.append(f"Reconciliation check {check.name!r} failed without an explained exception.")
        elif not check.passed and check.exception_explanation:
            warnings.append(f"Reconciliation check {check.name!r} failed, with an exception documented: {check.exception_explanation}")
    if len(report.objective) != len(report.conclusions):
        errors.append(f"The report states {len(report.objective)} question(s) but has {len(report.conclusions)} conclusion(s).")
    if _NUMERIC_TOKEN.search(report.executive_summary.text) and not report.executive_summary.key_numbers:
        errors.append("Executive summary contains numerical claims but has no key-number trace list.")

    errors.extend(_reference_issues(report))
    # Check user-visible narrative independently of the claims registry so a producer
    # cannot bypass causal discipline by omitting a sentence from the structured claims.
    supported_causal_texts = [re.sub(r"\s+", " ", claim.text.strip()).casefold() for claim in report.claims if claim.claim_type == "causal" and claim.causal_support and claim.causal_support.supports_causal_claim and claim.causal_support.design not in {"observational_only", "none"}]
    visible_narratives: List[Tuple[str, str]] = [("executive summary", report.executive_summary.text)]
    visible_narratives += [(f"finding {f.finding_id} headline", f.headline) for f in report.findings]
    visible_narratives += [(f"finding {f.finding_id} interpretation", f.interpretation) for f in report.findings]
    visible_narratives += [(f"conclusion {c.question_id}", c.answer) for c in report.conclusions]
    for context, narrative in visible_narratives:
        if _CAUSAL_HARD.search(narrative) and not _CAUSAL_HEDGE.search(narrative):
            normalized = re.sub(r"\s+", " ", narrative.strip()).casefold()
            if not any(normalized == approved or normalized in approved or approved in normalized for approved in supported_causal_texts):
                errors.append(f"{context.capitalize()} contains unhedged causal language with no matching supported causal claim.")
    rubric = _quality_rubric(report)
    quality_average = sum(rubric.values()) / len(rubric) if rubric else 0.0
    facts_by_id = {fact.fact_id: fact for fact in report.facts}
    total_numeric_tokens = 0
    traced_numeric_tokens = 0

    def _count_tokens(text_value: str, fact_ids: Sequence[str], explicit: Sequence[str] = (), ignored: Sequence[str] = ()) -> None:
        nonlocal total_numeric_tokens, traced_numeric_tokens
        allowed_facts = [facts_by_id[fid] for fid in fact_ids if fid in facts_by_id]
        for token in _NUMERIC_TOKEN.findall(text_value or ""):
            total_numeric_tokens += 1
            if token in ignored or token in explicit:
                traced_numeric_tokens += 1
                continue
            token_num = _norm_number(token)
            if any(_same_numeric(token, fact.value) for fact in allowed_facts):
                traced_numeric_tokens += 1
                continue
            if token_num is not None and any(token_num == _norm_number(known) for known in list(explicit) + list(ignored)):
                traced_numeric_tokens += 1

    summary_fact_ids = [str(item.get("fact_id")) for item in report.executive_summary.key_numbers if item.get("fact_id")]
    _count_tokens(report.executive_summary.text, summary_fact_ids, [str(item.get("literal", "")) for item in report.executive_summary.key_numbers], report.executive_summary.non_metric_numeric_tokens)
    for claim in report.claims:
        _count_tokens(claim.text, claim.fact_ids, [nr.literal for nr in claim.numeric_refs], claim.non_metric_numeric_tokens)
    for finding in report.findings:
        ignored = [str(finding.sample_size)] if finding.sample_size is not None else []
        _count_tokens(finding.headline, finding.fact_ids, ignored=ignored)
        _count_tokens(finding.interpretation, finding.fact_ids, ignored=ignored)
    for conclusion in report.conclusions:
        _count_tokens(conclusion.answer, conclusion.fact_ids)
    findings_by_id = {finding.finding_id: finding for finding in report.findings}
    for rec in report.recommendations:
        rec_fact_ids = list({fid for finding_id in rec.finding_refs for fid in (findings_by_id[finding_id].fact_ids if finding_id in findings_by_id else [])})
        explicit = [nr.literal for nr in rec.numeric_refs]
        _count_tokens(rec.action, rec_fact_ids, explicit, rec.non_metric_numeric_tokens)
        _count_tokens(rec.expected_impact, rec_fact_ids, explicit, rec.non_metric_numeric_tokens)
    numeric_traceability_rate = traced_numeric_tokens / total_numeric_tokens if total_numeric_tokens else 1.0
    metrics: Optional[BenchmarkRunMetrics] = None
    if benchmark_metrics is not None:
        try:
            metrics = BenchmarkRunMetrics.model_validate(benchmark_metrics)
        except ValidationError as exc:
            errors.append("Golden-benchmark metrics were provided but are invalid: " + "; ".join(e.get("msg", "invalid") for e in exc.errors(include_url=False)))

    report_checks_passed = not errors
    grade: Literal["FAIL", "BRONZE", "SILVER", "GOLD"] = "BRONZE" if report_checks_passed else "FAIL"
    if report_checks_passed and metrics is not None:
        # Bronze is a release gate, not just a label: reproducibility and numeric
        # traceability must both be exact when golden-suite metrics are supplied.
        if metrics.rerun_exact_match_rate < 1.0:
            errors.append("Bronze gate failed: reruns must match exactly across the golden suite.")
        if metrics.key_figure_traceability_rate < 1.0:
            errors.append("Bronze gate failed: every key figure must be traceable.")
        if metrics.rerun_exact_match_rate < 1.0 or metrics.key_figure_traceability_rate < 1.0:
            grade = "FAIL"
        else:
            if metrics.reconciliation_pass_rate >= 1.0 and metrics.statistical_validity_passed and quality_average >= 1.3 and metrics.quality_rubric_average >= 1.3:
                grade = "SILVER"
            if (grade == "SILVER" and metrics.planted_issue_detection_rate > 0.90 and metrics.false_insight_rate_on_no_signal == 0 and quality_average >= 1.7 and metrics.quality_rubric_average >= 1.7 and metrics.golden_dataset_count >= 20 and metrics.human_reviewers >= 2 and metrics.inter_rater_agreement is not None):
                grade = "GOLD"
            if metrics.reconciliation_pass_rate < 1.0:
                warnings.append("Silver gate unmet: one or more golden-suite reconciliation checks failed.")
            if not metrics.statistical_validity_passed:
                warnings.append("Silver gate unmet: statistical validity checks did not pass.")
            if metrics.quality_rubric_average < 1.3:
                warnings.append("Silver gate unmet: independent quality-rubric average is below 1.3.")
            if metrics.planted_issue_detection_rate <= 0.90:
                warnings.append("Gold gate unmet: planted-issue detection rate must be greater than 90%.")
            if metrics.false_insight_rate_on_no_signal > 0:
                warnings.append("Gold gate unmet: false-insight rate on no-signal datasets must be zero.")
            if metrics.golden_dataset_count < 20:
                warnings.append("Gold gate unmet: benchmark set contains fewer than 20 datasets.")
            if metrics.human_reviewers < 2 or metrics.inter_rater_agreement is None:
                warnings.append("Gold gate unmet: at least two human reviewers and inter-rater agreement are required.")
    elif report_checks_passed:
        warnings.append("Only report-level Bronze validation was evaluated; provide golden-suite metrics to assess Silver/Gold readiness.")

    valid = not errors
    if not valid:
        grade = "FAIL"
    return ReportValidationResult(valid=valid, publishable=valid and grade != "FAIL", grade=grade, completeness_rate=round(completeness_rate, 4), numeric_traceability_rate=round(numeric_traceability_rate, 4), quality_rubric=rubric, quality_average=round(quality_average, 3), errors=errors, warnings=warnings)


def report_benchmark_json_schema() -> Dict[str, Any]:
    """Return the JSON Schema contract generated from the Pydantic models."""
    schema = ReportBenchmarkContract.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://minded.local/schemas/report-benchmark-1.0.0.json"
    schema["title"] = "MindEd Real-World Analysis Report Benchmark Contract"
    return schema
