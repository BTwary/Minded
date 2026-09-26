"""Schema-grounded natural-language semantic proposal layer.

An optional BYOM model can enrich interpretation, but it can never execute
analysis. Every returned reference is validated against the physical schema
before the proposal is handed to the deterministic compiler.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Optional

from packages.analytics_core.src.governance.ai_authority import validate_ai_proposal

@dataclass(frozen=True)
class NLPProposal:
    task: str = "GENERAL_EXPLORATION"
    target: Optional[str] = None
    group: Optional[str] = None
    time: Optional[str] = None
    secondary_target: Optional[str] = None
    clauses: tuple[str, ...] = ()
    target_phrases: tuple[str, ...] = ()
    grouping_phrases: tuple[str, ...] = ()
    temporal_phrases: tuple[str, ...] = ()
    confidence: float = 0.0
    source: str = "deterministic"
    limitations: tuple[str, ...] = ()


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def _resolve_column(candidate: Any, columns: list[str]) -> Optional[str]:
    if not isinstance(candidate, str) or not candidate.strip():
        return None
    c = candidate.strip()
    exact = {str(x).lower(): str(x) for x in columns}
    if c.lower() in exact:
        return exact[c.lower()]
    nc = _normalize(c)
    scored = sorted(((SequenceMatcher(None, nc, _normalize(col)).ratio(), col) for col in columns), reverse=True)
    if scored and scored[0][0] >= 0.82:
        return scored[0][1]
    return None


def _parse_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def interpret_with_schema(question: str, columns: list[str], ai_provider: Any = None) -> NLPProposal:
    """Use explicitly injected AI provider only as a validated proposal; deterministic fallback is always available."""
    q = " ".join((question or "").split())
    ql = q.lower()
    clauses = tuple(c.strip() for c in re.split(r"\b(?:and|while|whereas|or else)\b|[;]", q, flags=re.I) if c.strip())
    proposal: dict[str, Any] | None = None
    limitations: list[str] = []
    # Pure deterministic path if no AI provider explicitly injected, or if provider is None / "none".
    # The deterministic interpreter is authoritative and always available.
    if ai_provider is not None and getattr(ai_provider, "provider_type", None) not in ("none", None):
        try:
            prompt = (
                "Return JSON only. Interpret the user's analytical question against this physical schema. "
                "Never invent columns. Allowed task values: DATA_QUALITY, DESCRIPTIVE, COMPARISON, "
                "ASSOCIATION, DIAGNOSTIC, FORECAST, PREDICTION, SEGMENTATION, CAUSAL, "
                "RECONCILIATION, PRESCRIPTIVE, GOVERNANCE, GENERAL_EXPLORATION. "
                "Fields: task,target,group,time,secondary_target. Use null when unresolved. "
                f"\nSCHEMA: {columns}\nQUESTION: {q}"
            )
            raw = ai_provider.generate_response(prompt, system_prompt="You are a semantic parser. JSON only; no SQL; no invented schema.")
            proposal = _parse_json(raw)
            if proposal is None:
                limitations.append("llm_semantic_proposal_unparseable; deterministic interpretation retained")
            else:
                validated = validate_ai_proposal(proposal, columns)
                if not validated.get("allowed"):
                    limitations.append(str(validated.get("reason")))
                    proposal = None
        except Exception as exc:
            limitations.append(f"ai_proposal_generation_failed: {exc}; deterministic interpretation retained")
            proposal = None
    if proposal is None:
        task = "GENERAL_EXPLORATION"
        if re.search(r"\b(causal|treatment effect|intervention|effect of|impact of)\b", ql): task = "CAUSAL"
        elif re.search(r"\b(forecast|project|future|next|will|expected to)\b", ql): task = "FORECAST"
        elif re.search(r"\b(predict|likely|risk|probability|odds|chance)\b", ql): task = "PREDICTION"
        elif re.search(r"\b(why|driver|caused|root cause|what explains|what is behind)\b", ql): task = "DIAGNOSTIC"
        elif re.search(r"\b(compare|difference|higher|lower|highest|lowest|top|bottom|more|less|which .* most|which .* least|typical|usual|by [a-z_ ]+)\b", ql): task = "COMPARISON"
        elif re.search(r"\b(correlat\w*|relationship\w*|associat\w*|link\w*|relat\w*|connect\w*|move together)\b", ql): task = "ASSOCIATION"
        elif re.search(r"\b(typical|usual|average|median|mean|summarize|summary|how much|how many|what is the)\b", ql): task = "DESCRIPTIVE"
        elif re.search(r"\b(cluster|segment|persona|groups)\b", ql): task = "SEGMENTATION"
        elif re.search(r"\b(missing|duplicate|quality|outlier|invalid)\b", ql): task = "DATA_QUALITY"
        proposal = {"task": task}
        source = "deterministic"
        confidence = 0.55 if task != "GENERAL_EXPLORATION" else 0.25
    else:
        source = "byom_validated"
        confidence = 0.85

    # Deterministic linguistic cues are schema-grounded, never free-form guesses.
    target_phrases = tuple(re.findall(r"\b(?:average|mean|sum|total|revenue|sales|cost|profit|margin|rate|conversion|churn|orders?|customers?|retention|aov|price|quantity|units?|volume)\b", ql))
    grouping_phrases = tuple(re.findall(r"(?:by|per|for each|across|within)\s+([a-z][a-z0-9_ -]{1,40})", ql))
    grouping_phrases += tuple(re.findall(r"\b(?:which|what)\s+(regions?|countries?|categories?|segments?|tiers?|plans?|channels?|groups?)\b", ql))
    temporal_phrases = tuple(re.findall(r"\b(?:today|yesterday|this month|last month|this quarter|last quarter|this year|last year|next month|next quarter|next year|q[1-4]|trailing \d+ (?:days?|months?|years?))\b", ql))

    target = _resolve_column(proposal.get("target") or proposal.get("target_metric"), columns)
    group = _resolve_column(proposal.get("group") or proposal.get("group_dimension"), columns)
    time = _resolve_column(proposal.get("time") or proposal.get("time_column"), columns)
    secondary = _resolve_column(proposal.get("secondary_target") or proposal.get("secondary_metric"), columns)

    # Resolve common natural-language role aliases only when there is one
    # defensible physical candidate. Ambiguity is preserved as a limitation.
    def role_alias(phrase: str, *, prefer_group: bool = False) -> Optional[str]:
        p = _normalize(phrase).rstrip("s")
        ranked = []
        for c in columns:
            cn = _normalize(c)
            base = cn.rstrip("s")
            score = 0.0
            if base == p or base.startswith(p + " ") or base.endswith(" " + p):
                score = 0.95
            elif p in base.split():
                score = 0.88
            elif p in base:
                score = 0.76
            if prefer_group and c.lower().endswith(('_id','id','_key','key','code')):
                score -= 0.18
            if score > 0:
                ranked.append((score, c))
        ranked.sort(reverse=True)
        if not ranked:
            return None
        if len(ranked) == 1 or ranked[0][0] - ranked[1][0] >= 0.10:
            return ranked[0][1]
        return None

    if not group:
        for phrase in grouping_phrases:
            group = _resolve_column(phrase, columns) or role_alias(phrase, prefer_group=True)
            if group:
                break
    if not time and temporal_phrases:
        # A temporal phrase does not itself identify a time column, so only use
        # an obvious date/time schema candidate when it is unique.
        temporal_candidates = [c for c in columns if re.search(r"(^|_|\b)(date|time|timestamp|datetime|period|month|quarter|year)(_|$)", c.lower())]
        if len(temporal_candidates) == 1:
            time = temporal_candidates[0]

    unresolved = []
    if proposal.get("target") and target is None: unresolved.append("target_not_resolved_to_schema")
    if proposal.get("group") and group is None: unresolved.append("group_not_resolved_to_schema")
    if proposal.get("time") and time is None: unresolved.append("time_not_resolved_to_schema")
    return NLPProposal(
        task=str(proposal.get("task") or "GENERAL_EXPLORATION").upper(),
        target=target, group=group, time=time, secondary_target=secondary,
        clauses=clauses, target_phrases=target_phrases, grouping_phrases=grouping_phrases,
        temporal_phrases=temporal_phrases, confidence=confidence, source=source,
        limitations=tuple(limitations + unresolved),
    )
