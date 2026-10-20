"""Evidence-grounded, analysis-family narrative composition for MindEd reports.

The composer is deliberately hybrid: a small set of report blueprints controls section
shape, while the actual prose is composed from saved, verified facts. It does not call an
LLM, re-run analysis, infer missing values, or invent causes. A future language-model
paraphraser may rephrase only the claims returned here, preserving each claim's fact and
evidence references, and must pass the same validation contract.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")

REPORT_BLUEPRINTS: Dict[str, Dict[str, Any]] = {
    "comparison": {
        "title": "What the comparison shows",
        "section_plan": ["direct_answer", "narrative", "key_findings", "visualizations", "interpretation", "calculation", "limitations", "evidence"],
    },
    "trend": {
        "title": "How the measure changed over time",
        "section_plan": ["direct_answer", "narrative", "key_findings", "visualizations", "interpretation", "calculation", "limitations", "evidence"],
    },
    "trend_comparison": {
        "title": "How trends differ",
        "section_plan": ["direct_answer", "narrative", "key_findings", "visualizations", "interpretation", "calculation", "limitations", "evidence"],
    },
    "relationship": {
        "title": "What the relationship indicates",
        "section_plan": ["direct_answer", "narrative", "key_findings", "visualizations", "interpretation", "calculation", "limitations", "evidence"],
    },
    "distribution": {
        "title": "What the distribution shows",
        "section_plan": ["direct_answer", "narrative", "key_findings", "visualizations", "interpretation", "calculation", "limitations", "evidence"],
    },
    "forecast": {
        "title": "What the forecast indicates",
        "section_plan": ["direct_answer", "narrative", "key_findings", "visualizations", "interpretation", "calculation", "limitations", "evidence"],
    },
    "diagnostic": {
        "title": "What changed and what remains unexplained",
        "section_plan": ["direct_answer", "narrative", "key_findings", "visualizations", "interpretation", "calculation", "limitations", "evidence", "next_steps"],
    },
    "composition": {
        "title": "How the total is composed",
        "section_plan": ["direct_answer", "narrative", "key_findings", "visualizations", "interpretation", "calculation", "limitations", "evidence"],
    },
    "descriptive": {
        "title": "What the recorded results show",
        "section_plan": ["direct_answer", "narrative", "key_findings", "visualizations", "interpretation", "calculation", "limitations", "evidence"],
    },
}


def _d(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _finite(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
        return math.isfinite(number)
    except (TypeError, ValueError, OverflowError):
        return False


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if _finite(value):
        number = float(value)
        if number.is_integer():
            return f"{int(number):,}"
        return f"{number:,.5f}".rstrip("0").rstrip(".")
    return str(value)


def _fmt_pct(value: Any) -> str:
    return f"{float(value):,.1f}" if _finite(value) else str(value)


def _norm_num(token: str) -> str:
    raw = token.replace(",", "").replace("%", "")
    try:
        value = float(raw)
    except (ValueError, OverflowError):
        return raw
    if not math.isfinite(value):
        return raw
    if value == 0:
        return "0"
    return format(value, ".10g")


def _number_atoms(value: Any) -> set[str]:
    atoms: set[str] = set()
    if _finite(value):
        atoms.add(_norm_num(_fmt(value)))
        atoms.add(_norm_num(str(value)))
    elif isinstance(value, str):
        atoms.update(_norm_num(token) for token in _NUMBER_RE.findall(value))
    return atoms


def _numbers_in_text(value: str) -> List[str]:
    return [_norm_num(token) for token in _NUMBER_RE.findall(value)]


def _walk_scalar_paths(value: Any, path: str, out: Dict[str, Any], limit: int = 3000) -> None:
    if len(out) >= limit:
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _walk_scalar_paths(child, f"{path}.{key}" if path else str(key), out, limit)
            if len(out) >= limit:
                break
    elif isinstance(value, (list, tuple)):
        for idx, child in enumerate(value):
            _walk_scalar_paths(child, f"{path}[{idx}]", out, limit)
            if len(out) >= limit:
                break
    elif value is None or isinstance(value, (str, int, float, bool)):
        out[path] = value


def validate_narrative_claim(
    item: Dict[str, Any], facts: Dict[str, Dict[str, Any]],
    verified_evidence_ids: Iterable[str], scalar_paths: Iterable[str],
) -> Tuple[bool, List[str]]:
    """Validate claim provenance and ensure its printed numbers map to linked facts."""
    fact_ids = list(item.get("fact_ids") or [])
    missing_facts = [fid for fid in fact_ids if fid not in facts]
    source_paths = list(item.get("source_paths") or [])
    known_paths = set(scalar_paths)
    missing_sources = [path for path in source_paths if not (
        path in known_paths or path.startswith("derived from ")
        or path.startswith("analyst_result.numbers.table.")
        or path.startswith("analyst_result.numbers.correlations key:")
        or "(group label)" in path or path.startswith("derived:")
    )]
    allowed_evidence = set(verified_evidence_ids)
    invalid_refs = [ref for ref in (item.get("evidence_refs") or []) if ref not in allowed_evidence]
    linked_numeric_atoms: set[str] = set()
    for fid in fact_ids:
        if fid in facts:
            linked_numeric_atoms.update(_number_atoms(facts[fid].get("value")))
    numeric_literals = list(item.get("numeric_literals") or [])
    unsupported_numbers = [tok for tok in numeric_literals if tok not in linked_numeric_atoms]
    issues: List[str] = []
    if not fact_ids:
        issues.append("no linked facts")
    if missing_facts:
        issues.append(f"missing facts: {missing_facts}")
    if missing_sources:
        issues.append(f"missing source paths: {missing_sources}")
    if not item.get("evidence_refs") or invalid_refs:
        issues.append("no verified evidence reference")
    if unsupported_numbers:
        issues.append(f"unsupported numeric literals: {unsupported_numbers}")
    return not issues, issues


def _verified_evidence(evidence: Any) -> List[Dict[str, Any]]:
    if not isinstance(evidence, list):
        return []
    return [item for item in evidence if isinstance(item, dict)
            and str(item.get("validation_status") or item.get("status") or "").upper() in {"VERIFIED", "PASSED"}
            and str(item.get("id") or item.get("evidence_id") or "").strip()]


def _is_verified(status: str, execution: Dict[str, Any], evidence: List[Dict[str, Any]]) -> bool:
    if status.upper() != "COMPLETED":
        return False
    if execution:
        return (str(execution.get("contract_status") or "").upper() == "MATCHED"
                and str(execution.get("verification_status") or "").upper() == "VERIFIED"
                and bool(evidence))
    return bool(evidence)


def _family(question: str, result: Dict[str, Any], numbers: Dict[str, Any]) -> str:
    q = question.lower()
    kind = str(result.get("kind") or "").upper()
    if any(x in q for x in ("forecast", "predict", "projection", "projected")) or "FORECAST" in kind or _d(numbers.get("run_rate_projection")).get("projected_value") is not None:
        return "forecast"
    if any(x in q for x in ("why ", "why did", "cause", "caused", "root cause", "explain why")) or kind in {"ROOT_CAUSE", "DIAGNOSTIC", "DIAGNOSIS"}:
        return "diagnostic"
    if kind in {"TREND_COMPARISON", "GROUP_TREND_COMPARISON"} or ("trend" in q and any(x in q for x in ("compare", "by region", "by group", "across regions", "between groups"))):
        return "trend_comparison"
    if kind == "TREND" or any(k in numbers for k in ("slope_per_period", "sen_slope", "trend_slope")) or "trend_inference" in numbers:
        return "trend"
    if kind in {"ASSOCIATION", "CORRELATION", "REGRESSION", "RELATIONSHIP"} or any(k in numbers for k in ("pearson_r", "correlations", "scatter_data")):
        return "relationship"
    if "distribution" in q or any(k in numbers for k in ("distribution_profile", "histogram_bins")) or kind in {"DISTRIBUTION", "OUTLIER_ANALYSIS"}:
        return "distribution"
    if "composition" in q or "share of total" in q or "composition_over_time" in numbers:
        return "composition"
    if isinstance(numbers.get("table"), dict) and len(numbers["table"]) >= 2 or kind in {"RANKING", "GROUP_COMPARISON", "RATE_COMPARISON", "COMPARISON"}:
        return "comparison"
    return "descriptive"


def _secondary_families(question: str, result: Dict[str, Any], numbers: Dict[str, Any], primary: str) -> List[str]:
    """Return additional analysis families only when both question and saved values support them."""
    q = question.lower()
    found: List[str] = []
    table = _d(numbers.get("table"))
    has_slope = any(_finite(numbers.get(key)) for key in ("slope_per_period", "sen_slope", "trend_slope")) or any(
        _finite(_d(numbers.get("trend_inference")).get(key)) for key in ("slope", "sen_slope")
    )
    has_distribution = bool(_d(numbers.get("distribution_profile")) or numbers.get("histogram_bins"))
    has_relationship = bool(_d(numbers.get("correlations")) or _finite(numbers.get("pearson_r")))
    asks_compare = any(word in q for word in ("compare", "versus", " vs ", "by region", "by group", "across regions", "between groups", "which region", "which regions", "contributed most", "largest contribution", "top contributor"))
    asks_trend = any(word in q for word in ("trend", "over time", "growth", "decline", "changed", "change over", "year-over-year", "year over year", "over the past", "over the last"))
    asks_distribution = any(word in q for word in ("distribution", "variability", "spread", "outlier", "dispersion", "volatility"))
    asks_diagnostic = any(word in q for word in ("why ", "why did", "cause", "root cause", "explain why"))

    if primary != "comparison" and len(table) >= 2 and asks_compare:
        found.append("comparison")
    if primary not in {"trend", "trend_comparison"} and has_slope and asks_trend:
        found.append("trend")
    if primary != "diagnostic" and asks_diagnostic and (_finite(numbers.get("previous")) and _finite(numbers.get("current")) or numbers.get("top_contributor") is not None):
        found.append("diagnostic")
    if primary != "distribution" and asks_distribution and has_distribution:
        found.append("distribution")
    if primary != "relationship" and has_relationship and any(word in q for word in ("relationship", "correlat", "associated", "related")):
        found.append("relationship")
    if primary != "composition" and ("composition_over_time" in numbers or "composition" in q):
        found.append("composition")
    # Keep the family set small and deterministic. The first family owns the lead paragraph.
    return list(dict.fromkeys(found))[:3]


def compose_report_narrative(
    *,
    question: str,
    analyst_result: Dict[str, Any],
    execution_record: Dict[str, Any],
    evidence: Any,
    status: str,
    _family_override: Optional[str] = None,
    _include_secondary: bool = True,
) -> Dict[str, Any]:
    """Compose family-aware explanatory prose, validating each claim against facts.

    The input is a saved result object. The function never reads raw datasets and never
    performs an inferential analysis. Derived claims are arithmetic summaries of already
    saved, verified output values and retain their parent fact IDs.
    """
    result = _d(analyst_result)
    numbers = _d(result.get("numbers"))
    execution = _d(execution_record or numbers.get("execution_record"))
    verified_evidence = _verified_evidence(evidence)
    evidence_ids = [str(ev.get("id") or ev.get("evidence_id")) for ev in verified_evidence]
    is_verified = _is_verified(status, execution, verified_evidence)
    family = _family_override or _family(question, result, numbers)
    is_primary_composition = _family_override is None and _include_secondary
    blueprint = REPORT_BLUEPRINTS.get(family, REPORT_BLUEPRINTS["descriptive"])

    all_scalars: Dict[str, Any] = {}
    _walk_scalar_paths({"analyst_result": result, "execution_record": execution}, "", all_scalars)
    facts: Dict[str, Dict[str, Any]] = {}
    for path, value in all_scalars.items():
        if value is None:
            continue
        fact_id = path.replace("[", ".").replace("]", "").replace(" ", "_")
        facts[fact_id] = {
            "value": value,
            "source_paths": [path],
            "evidence_refs": list(evidence_ids),
            "derivation": None,
        }

    def derived_fact(fact_id: str, value: Any, parent_ids: Sequence[str], formula: str) -> str:
        parent_facts = [facts[fid] for fid in parent_ids if fid in facts]
        refs = sorted({ref for fact in parent_facts for ref in fact.get("evidence_refs", [])})
        paths = [path for fact in parent_facts for path in fact.get("source_paths", [])]
        facts[fact_id] = {
            "value": value,
            "source_paths": paths,
            "evidence_refs": refs,
            "derivation": {"formula": formula, "parent_fact_ids": list(parent_ids)},
        }
        return fact_id

    claims: List[Dict[str, Any]] = []

    def claim(text: str, fact_ids: Sequence[str], explanation: Optional[str] = None) -> None:
        ids = list(dict.fromkeys(fact_ids))
        refs = sorted({ref for fid in ids if fid in facts for ref in facts[fid].get("evidence_refs", [])})
        claims.append({
            "claim_id": f"claim_{len(claims) + 1}",
            "family": family,
            "text": text,
            "fact_ids": ids,
            "evidence_refs": refs,
            "source_paths": sorted({path for fid in ids if fid in facts for path in facts[fid].get("source_paths", [])}),
            "numeric_literals": _numbers_in_text(text),
            "explanation": explanation or "Derived from saved result values; no new analysis was fitted for the narrative.",
        })

    metric = str(numbers.get("target_column") or _d(numbers.get("canonical_contract")).get("target_column") or execution.get("target_column") or "the requested measure")
    operation = str(numbers.get("operation") or numbers.get("aggregation") or execution.get("aggregation") or "recorded value").upper()
    grain = str(numbers.get("grain") or numbers.get("period_grain") or "period")

    # Comparison and ranking blueprint. The aggregate table is the factual base; gaps and
    # shares are explicitly derived and retain their parent fact IDs.
    table = _d(numbers.get("table"))
    table_entries: List[Tuple[str, float, Optional[float], str, str]] = []
    for label, raw in table.items():
        row = _d(raw)
        val = row.get("value", raw)
        if _finite(val):
            value_id = f"group:{label}:value"
            label_id = f"group:{label}:label"
            facts[value_id] = {"value": float(val), "source_paths": [f"analyst_result.numbers.table.{label}.value"], "evidence_refs": list(evidence_ids), "derivation": None}
            facts[label_id] = {"value": str(label), "source_paths": [f"analyst_result.numbers.table.{label} (group label)"], "evidence_refs": list(evidence_ids), "derivation": None}
            n_val = row.get("n")
            n_id: Optional[str] = None
            if _finite(n_val):
                n_id = f"group:{label}:n"
                facts[n_id] = {"value": float(n_val), "source_paths": [f"analyst_result.numbers.table.{label}.n"], "evidence_refs": list(evidence_ids), "derivation": None}
            table_entries.append((str(label), float(val), float(n_val) if _finite(n_val) else None, value_id, label_id))

    if is_verified and evidence_ids:
        if family in {"comparison", "composition"} and len(table_entries) >= 2:
            ordered = sorted(table_entries, key=lambda entry: entry[1], reverse=True)
            high, low = ordered[0], ordered[-1]
            question_lower = str(question or "").casefold()
            asks_for_low = bool(re.search(r"\b(?:lowest|least|smallest|worst|bottom)\b", question_lower)) and not bool(
                re.search(r"\b(?:highest|most|largest|best|top)\b", question_lower)
            )
            leader = low if asks_for_low else high
            direction_word = "lowest" if asks_for_low else "highest"
            claim(f"{leader[0]} recorded the {direction_word} {operation.lower()} of {metric} among the returned groups ({_fmt(leader[1])}).", [leader[3], leader[4]])
            if high[1] != low[1]:
                gap = high[1] - low[1]
                gap_id = derived_fact("derived:top_to_bottom_gap", gap, [high[3], low[3]], "highest_group_value - lowest_group_value")
                claim(f"The gap between {high[0]} and {low[0]} was {_fmt(gap)} in the recorded metric.", [high[3], high[4], low[3], low[4], gap_id])
                if low[1] > 0:
                    pct = round((gap / low[1]) * 100.0, 1)
                    pct_id = derived_fact("derived:relative_gap_percent", pct, [high[3], low[3], gap_id], "round((highest_group_value - lowest_group_value) / lowest_group_value * 100, 1)")
                    claim(f"That gap equals {_fmt_pct(pct)}% of {low[0]}'s value; this percentage uses {low[0]} as the reference.", [high[3], low[3], high[4], low[4], gap_id, pct_id])
                additive = operation in {"SUM", "TOTAL", "COUNT", "ROW_COUNT"}
                total = sum(entry[1] for entry in ordered)
                if additive and total > 0 and all(entry[1] >= 0 for entry in ordered):
                    total_id = derived_fact("derived:returned_group_total", total, [entry[3] for entry in ordered], "sum of returned group-level values")
                    share = round((leader[1] / total) * 100.0, 1)
                    share_id = derived_fact(
                        "derived:lowest_group_share_percent" if asks_for_low else "derived:highest_group_share_percent",
                        share,
                        [leader[3], total_id],
                        "round(selected leader group value / sum of returned group values * 100, 1)",
                    )
                    claim(f"{leader[0]} accounts for {_fmt_pct(share)}% of the combined value across the returned groups.", [leader[3], leader[4], total_id, share_id])
            if len(table_entries) > 2:
                count_id = derived_fact("derived:returned_group_count", len(table_entries), [entry[3] for entry in table_entries], "count of groups in the saved result table")
                claim(f"The comparison includes {_fmt(len(table_entries))} returned groups.", [count_id])

        elif family in {"trend", "trend_comparison"}:
            inference = _d(numbers.get("trend_inference"))
            slope = next((v for v in (numbers.get("slope_per_period"), numbers.get("sen_slope"), numbers.get("trend_slope"), inference.get("sen_slope"), inference.get("slope")) if _finite(v)), None)
            slope_path = next((p for p in ("analyst_result.numbers.slope_per_period", "analyst_result.numbers.sen_slope", "analyst_result.numbers.trend_slope", "analyst_result.numbers.trend_inference.sen_slope", "analyst_result.numbers.trend_inference.slope") if p in all_scalars), None)
            if slope is not None:
                sid = "trend:slope"
                facts[sid] = {"value": float(slope), "source_paths": [slope_path or "analyst_result.numbers.trend_inference.slope"], "evidence_refs": list(evidence_ids), "derivation": None}
                direction = "upward" if float(slope) > 0 else "downward" if float(slope) < 0 else "flat"
                claim(f"The recorded trend in {metric} is {direction}, with an estimated slope of {_fmt(slope)} per {grain}.", [sid])
                start = next((v for v in (numbers.get("start_level"), inference.get("start_level")) if _finite(v)), None)
                end = next((v for v in (numbers.get("end_level"), inference.get("end_level")) if _finite(v)), None)
                if start is not None and end is not None:
                    start_id = "trend:start_level"
                    end_id = "trend:end_level"
                    start_path = "analyst_result.numbers.start_level" if _finite(numbers.get("start_level")) else "analyst_result.numbers.trend_inference.start_level"
                    end_path = "analyst_result.numbers.end_level" if _finite(numbers.get("end_level")) else "analyst_result.numbers.trend_inference.end_level"
                    facts[start_id] = {"value": float(start), "source_paths": [start_path], "evidence_refs": list(evidence_ids), "derivation": None}
                    facts[end_id] = {"value": float(end), "source_paths": [end_path], "evidence_refs": list(evidence_ids), "derivation": None}
                    change = float(end) - float(start)
                    change_id = derived_fact("trend:observed_endpoint_change", change, [start_id, end_id], "end_level - start_level")
                    claim(f"The saved endpoint values moved from {_fmt(start)} to {_fmt(end)}, a difference of {_fmt(change)}.", [start_id, end_id, change_id])
                lo = next((v for v in (numbers.get("ci_low"), inference.get("ci_low"), numbers.get("slope_ci_low")) if _finite(v)), None)
                hi = next((v for v in (numbers.get("ci_high"), inference.get("ci_high"), numbers.get("slope_ci_high")) if _finite(v)), None)
                if lo is not None and hi is not None and lo <= hi:
                    lo_id = "trend:interval_low"
                    hi_id = "trend:interval_high"
                    lo_path = "analyst_result.numbers.ci_low" if _finite(numbers.get("ci_low")) else "analyst_result.numbers.trend_inference.ci_low" if _finite(inference.get("ci_low")) else "analyst_result.numbers.slope_ci_low"
                    hi_path = "analyst_result.numbers.ci_high" if _finite(numbers.get("ci_high")) else "analyst_result.numbers.trend_inference.ci_high" if _finite(inference.get("ci_high")) else "analyst_result.numbers.slope_ci_high"
                    facts[lo_id] = {"value": float(lo), "source_paths": [lo_path], "evidence_refs": list(evidence_ids), "derivation": None}
                    facts[hi_id] = {"value": float(hi), "source_paths": [hi_path], "evidence_refs": list(evidence_ids), "derivation": None}
                    claim(f"The recorded slope interval extends from {_fmt(lo)} to {_fmt(hi)} per {grain}.", [lo_id, hi_id])
            else:
                claim("The result includes a trend-related analysis, but no structured slope estimate is available for a more specific explanation.", ["analyst_result.kind"] if "analyst_result.kind" in facts else [])

        elif family == "relationship":
            correlations = _d(numbers.get("correlations"))
            valid = [(str(k), float(v)) for k, v in correlations.items() if _finite(v) and abs(float(v)) <= 1]
            if valid:
                valid.sort(key=lambda pair: abs(pair[1]), reverse=True)
                variable, corr = valid[0]
                rid = "relationship:strongest_correlation"
                facts[rid] = {"value": corr, "source_paths": [f"analyst_result.numbers.correlations.{variable}"], "evidence_refs": list(evidence_ids), "derivation": None}
                vid = "relationship:variable_label"
                facts[vid] = {"value": variable, "source_paths": [f"analyst_result.numbers.correlations key: {variable}"], "evidence_refs": list(evidence_ids), "derivation": None}
                sign = "positive" if corr > 0 else "negative" if corr < 0 else "zero"
                claim(f"Among the returned correlations, {variable} has the largest absolute correlation with {metric} (r = {_fmt(corr)}), a {sign} linear association in this result.", [rid, vid])
                claim("Correlation describes association; it does not by itself establish a causal mechanism.", [rid, vid])
            elif _finite(numbers.get("pearson_r")) and abs(float(numbers["pearson_r"])) <= 1:
                corr = float(numbers["pearson_r"])
                rid = "relationship:pearson_r"
                facts[rid] = {"value": corr, "source_paths": ["analyst_result.numbers.pearson_r"], "evidence_refs": list(evidence_ids), "derivation": None}
                sign = "positive" if corr > 0 else "negative" if corr < 0 else "zero"
                claim(f"The recorded Pearson correlation is r = {_fmt(corr)}, indicating a {sign} linear association in the observed data.", [rid])
                claim("Correlation does not establish causation.", [rid])

        elif family == "distribution":
            profile = _d(numbers.get("distribution_profile"))
            profiles = [(str(k), _d(v)) for k, v in profile.items() if isinstance(v, dict)] if profile else [("observed", numbers)]
            added = False
            for label, values in profiles:
                q1 = values.get("p25", values.get("q1"))
                med = values.get("p50", values.get("median"))
                q3 = values.get("p75", values.get("q3"))
                if _finite(q1) and _finite(med) and _finite(q3) and float(q1) <= float(med) <= float(q3):
                    suffix = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "observed"
                    q1_id = f"distribution:{suffix}:q1"
                    med_id = f"distribution:{suffix}:median"
                    q3_id = f"distribution:{suffix}:q3"
                    iqr_id = f"distribution:{suffix}:iqr"
                    facts[q1_id] = {"value": float(q1), "source_paths": [f"analyst_result.numbers.distribution_profile.{label}.p25"], "evidence_refs": list(evidence_ids), "derivation": None}
                    facts[med_id] = {"value": float(med), "source_paths": [f"analyst_result.numbers.distribution_profile.{label}.p50"], "evidence_refs": list(evidence_ids), "derivation": None}
                    facts[q3_id] = {"value": float(q3), "source_paths": [f"analyst_result.numbers.distribution_profile.{label}.p75"], "evidence_refs": list(evidence_ids), "derivation": None}
                    derived_fact(iqr_id, float(q3) - float(q1), [q1_id, q3_id], "p75 - p25")
                    claim(f"For {label}, the middle half of {metric} spans {_fmt(q1)} to {_fmt(q3)}; the interquartile range is {_fmt(float(q3)-float(q1))}.", [q1_id, med_id, q3_id, iqr_id])
                    added = True
            if not added:
                profile = _d(numbers.get("distribution_profile"))
                valid = {k: v for k, v in profile.items() if _finite(v)}
                if valid:
                    top_key = next((k for k in ("median", "mean", "min", "max", "std", "std_dev", "iqr") if k in valid), None)
                    if top_key:
                        fid = f"distribution:{top_key}"
                        facts[fid] = {"value": valid[top_key], "source_paths": [f"analyst_result.numbers.distribution_profile.{top_key}"], "evidence_refs": list(evidence_ids), "derivation": None}
                        claim(f"The saved distribution summary reports {top_key.replace('_', ' ')} = {_fmt(valid[top_key])} for {metric}.", [fid])

        elif family == "forecast":
            projection = _d(numbers.get("run_rate_projection"))
            projected = projection.get("projected_value")
            if _finite(projected):
                pid = "forecast:projected_value"
                facts[pid] = {"value": float(projected), "source_paths": ["analyst_result.numbers.run_rate_projection.projected_value"], "evidence_refs": list(evidence_ids), "derivation": None}
                claim(f"The recorded projection for {metric} is {_fmt(projected)} for the next {grain}.", [pid])
                lo, hi = projection.get("pi_low"), projection.get("pi_high")
                if _finite(lo) and _finite(hi) and float(lo) <= float(projected) <= float(hi):
                    lid = "forecast:interval_low"; hid = "forecast:interval_high"
                    facts[lid] = {"value": float(lo), "source_paths": ["analyst_result.numbers.run_rate_projection.pi_low"], "evidence_refs": list(evidence_ids), "derivation": None}
                    facts[hid] = {"value": float(hi), "source_paths": ["analyst_result.numbers.run_rate_projection.pi_high"], "evidence_refs": list(evidence_ids), "derivation": None}
                    claim(f"The saved prediction range extends from {_fmt(lo)} to {_fmt(hi)}.", [pid, lid, hid])
                claim("A projection describes a modeled future value, not a guaranteed outcome.", [pid])
            else:
                claim("A forecast was requested, but the saved result contains no projected value. Historical observations are not substituted for a forecast.", ["analyst_result.kind"] if "analyst_result.kind" in facts else [])

        elif family == "diagnostic":
            previous = numbers.get("previous")
            current = numbers.get("current")
            if _finite(previous) and _finite(current):
                prior_id = "diagnostic:previous_value"; current_id = "diagnostic:current_value"
                facts[prior_id] = {"value": float(previous), "source_paths": ["analyst_result.numbers.previous"], "evidence_refs": list(evidence_ids), "derivation": None}
                facts[current_id] = {"value": float(current), "source_paths": ["analyst_result.numbers.current"], "evidence_refs": list(evidence_ids), "derivation": None}
                delta = float(current) - float(previous)
                delta_id = derived_fact("diagnostic:observed_change", delta, [prior_id, current_id], "current - previous")
                direction = "increased" if delta > 0 else "decreased" if delta < 0 else "was unchanged"
                claim(f"{metric.capitalize()} {direction} from {_fmt(previous)} to {_fmt(current)}, a recorded change of {_fmt(delta)}.", [prior_id, current_id, delta_id])
            contributor = numbers.get("top_contributor")
            share = numbers.get("top_contribution_share")
            if contributor is not None and _finite(share) and 0 <= float(share) <= 1:
                cid = "diagnostic:top_contributor"
                facts[cid] = {"value": str(contributor), "source_paths": ["analyst_result.numbers.top_contributor"], "evidence_refs": list(evidence_ids), "derivation": None}
                sid = "diagnostic:contribution_share_percent"
                derived_fact(sid, float(share) * 100.0, ["analyst_result.numbers.top_contribution_share"], "top_contribution_share * 100") if "analyst_result.numbers.top_contribution_share" in facts else None
                if sid not in facts:
                    facts[sid] = {"value": float(share) * 100.0, "source_paths": ["derived from analyst_result.numbers.top_contribution_share"], "evidence_refs": list(evidence_ids), "derivation": {"formula": "share * 100", "parent_fact_ids": []}}
                claim(f"{contributor} accounts for {_fmt_pct(round(float(share)*100, 1))}% of the recorded change contribution. This is an arithmetic attribution within the decomposition, not proof of an underlying cause.", [cid, sid])
            claim("The observed change and any arithmetic contributions identify where to investigate; they do not, by themselves, prove why the outcome changed.", ["analyst_result.kind"] if "analyst_result.kind" in facts else [])

        else:  # simple descriptives and other structured numeric answers
            value = numbers.get("value", numbers.get("top_value"))
            if _finite(value):
                fid = "descriptive:recorded_value"
                facts[fid] = {"value": float(value), "source_paths": ["analyst_result.numbers.value" if _finite(numbers.get("value")) else "analyst_result.numbers.top_value"], "evidence_refs": list(evidence_ids), "derivation": None}
                claim(f"The recorded {operation.lower()} for {metric} is {_fmt(value)}.", [fid])

    # Claim validation is structural and numeric: every source path and evidence reference
    # must exist, and every numeric literal in the prose must be derivable from its linked
    # facts. Unsupported claims remain in the audit trail but are not presented to readers.
    scalar_paths = set(all_scalars)
    accepted: List[Dict[str, Any]] = []
    audited: List[Dict[str, Any]] = []
    issues: List[str] = []
    for item in claims:
        ok, claim_issues = validate_narrative_claim(item, facts, evidence_ids, scalar_paths)
        item["validation_status"] = "SUPPORTED" if ok else "WITHHELD"
        item["validation_issues"] = claim_issues
        if ok:
            accepted.append(item)
        else:
            issues.append(f"{item['claim_id']}: " + "; ".join(claim_issues))
        audited.append(item)

    if not is_verified:
        narrative_status = "WITHHELD"
        summary = "A descriptive narrative is withheld because the saved analysis does not meet the report verification gate."
        paragraphs: List[str] = []
    elif accepted:
        narrative_status = "READY" if not issues else "PARTIAL"
        summary = accepted[0]["text"]
        paragraphs = [item["text"] for item in accepted[1:5]]
    else:
        narrative_status = "LIMITED"
        summary = "The saved result does not expose enough structured, verified facts to compose a more detailed explanation. The direct answer and evidence record remain the authoritative output."
        paragraphs = []

    caveats = [str(x).strip() for x in (result.get("caveats") or []) if str(x).strip()]
    if family == "relationship" and not any("caus" in x.lower() for x in caveats):
        caveats.append("An observed association does not establish causation.")
    if family == "trend" and not any("caus" in x.lower() for x in caveats):
        caveats.append("An observed historical trend describes change over the analysed window; it does not by itself explain the cause or guarantee future change.")
    if family == "diagnostic" and not any("caus" in x.lower() for x in caveats):
        caveats.append("Observed changes and arithmetic contributions do not establish the underlying cause without additional identifying evidence.")
    next_steps = [str(x).strip() for x in (result.get("next_steps") or []) if str(x).strip()]

    used_fact_ids: set[str] = set()
    def include_fact(fact_id: str) -> None:
        if fact_id in used_fact_ids or fact_id not in facts:
            return
        used_fact_ids.add(fact_id)
        derivation = facts[fact_id].get("derivation") or {}
        for parent in derivation.get("parent_fact_ids", []) if isinstance(derivation, dict) else []:
            include_fact(str(parent))
    for item in audited:
        for fact_id in item.get("fact_ids") or []:
            include_fact(str(fact_id))
    audit_facts = {key: facts[key] for key in sorted(used_fact_ids)}

    report = {
        "family": family,
        "families": [family],
        "title": blueprint["title"],
        "section_plan": list(blueprint["section_plan"]),
        "status": narrative_status,
        "summary": summary,
        "paragraphs": paragraphs,
        "claims": audited,
        "supported_claim_count": len(accepted),
        "withheld_claim_count": len(claims) - len(accepted),
        "validation": {"status": "PASSED" if is_verified and not issues else "PARTIAL" if is_verified and accepted else "WITHHELD", "issues": issues},
        "limitations": list(dict.fromkeys(caveats)),
        "next_steps": next_steps,
        "facts": audit_facts,
        "evidence_refs": evidence_ids,
    }

    if is_primary_composition:
        secondary_families = _secondary_families(question, result, numbers, family)
        for secondary in secondary_families:
            sub = compose_report_narrative(
                question=question, analyst_result=result, execution_record=execution,
                evidence=evidence, status=status, _family_override=secondary,
                _include_secondary=False,
            )
            report["families"].append(secondary)
            report["claims"].extend([
                {**item, "claim_id": f"{secondary}_{item.get('claim_id', 'claim')}"}
                for item in (sub.get("claims") or [])
            ])
            report["facts"].update(_d(sub.get("facts")))
            if sub.get("summary") and sub.get("status") in {"READY", "PARTIAL"}:
                if sub["summary"] != report["summary"] and sub["summary"] not in report["paragraphs"]:
                    report["paragraphs"].append(sub["summary"])
            for paragraph in sub.get("paragraphs") or []:
                if paragraph not in report["paragraphs"] and paragraph != report["summary"]:
                    report["paragraphs"].append(paragraph)
            report["paragraphs"] = report["paragraphs"][:7]
            report["supported_claim_count"] += int(sub.get("supported_claim_count") or 0)
            report["withheld_claim_count"] += int(sub.get("withheld_claim_count") or 0)
            report["limitations"] = list(dict.fromkeys(report["limitations"] + list(sub.get("limitations") or [])))
            report["next_steps"] = list(dict.fromkeys(report["next_steps"] + list(sub.get("next_steps") or [])))
            report["validation"]["issues"].extend(list(_d(sub.get("validation")).get("issues") or []))
        report["families"] = list(dict.fromkeys(report["families"]))
        if len(report["families"]) > 1:
            report["title"] = "What the combined results show"
            report["section_plan"] = list(dict.fromkeys(
                section for item in report["families"] for section in REPORT_BLUEPRINTS[item]["section_plan"]
            ))
        supported_claims = [item for item in report["claims"] if item.get("validation_status") == "SUPPORTED"]
        withheld_claims = [item for item in report["claims"] if item.get("validation_status") != "SUPPORTED"]
        report["supported_claim_count"] = len(supported_claims)
        report["withheld_claim_count"] = len(withheld_claims)
        if supported_claims:
            report["status"] = "PARTIAL" if withheld_claims else "READY"
            lead = next((item for item in supported_claims if item.get("family") == family), supported_claims[0])
            report["summary"] = lead["text"]
            # Preserve at least one supported finding from every selected family before
            # filling the remaining space with additional details from the primary family.
            chosen = [lead]
            for selected_family in report["families"]:
                match = next((item for item in supported_claims if item.get("family") == selected_family and item not in chosen), None)
                if match is not None:
                    chosen.append(match)
            for item in supported_claims:
                if item not in chosen and len(chosen) < 8:
                    chosen.append(item)
            report["paragraphs"] = [item["text"] for item in chosen if item is not lead][:7]
            report["validation"]["status"] = "PARTIAL" if withheld_claims or report["validation"]["issues"] else "PASSED"
        elif not is_verified:
            report["status"] = "WITHHELD"
            report["validation"]["status"] = "WITHHELD"
        else:
            report["status"] = "LIMITED"
            report["validation"]["status"] = "WITHHELD"
        used_ids: set[str] = set()
        def include(key: str) -> None:
            if key in used_ids or key not in report["facts"]:
                return
            used_ids.add(key)
            derivation = _d(report["facts"][key].get("derivation"))
            for parent in derivation.get("parent_fact_ids", []):
                include(str(parent))
        for item in report["claims"]:
            for key in item.get("fact_ids") or []:
                include(str(key))
        report["facts"] = {key: report["facts"][key] for key in sorted(used_ids)}

    return report
