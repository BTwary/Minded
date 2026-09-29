"""Universal natural-language interpretation before deterministic compilation.

This layer broadens language coverage without becoming an analytical executor.
It proposes structured intent; the canonical Question Compiler remains the final
authority and the schema remains the authority for physical columns.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import re
from typing import Any


@dataclass(frozen=True)
class NaturalLanguageInterpretation:
    task_hint: str
    target_phrases: list[str] = field(default_factory=list)
    grouping_phrases: list[str] = field(default_factory=list)
    temporal_phrases: list[str] = field(default_factory=list)
    compound_clauses: list[str] = field(default_factory=list)
    asks_for_action: bool = False
    asks_for_cause: bool = False
    asks_for_prediction: bool = False
    notes: list[str] = field(default_factory=list)


_BIVARIATE_TRIGGER_RE = re.compile(
    r"\b(?:between|correlation of|relationship of|association of|associated with|correlated with|related to|relate to|compare|comparing)\b",
    re.I,
)
_INVERTED_RELATION_RE = re.compile(
    r"\b(?:is|are|do|does|can|whether)\s+(.+?)\s+\b(?:associated with|correlated with|related to|relate to|depend on|depends on|influence|influences|affect|affects)\b",
    re.I,
)
_RELATION_OBJECT_RE = re.compile(
    r"\b(?:associated with|correlated with|related to|relate to|depend on|depends on|influence|influences|affect|affects)\s+(.+?)(?=(?:,|;|\bcontrolling for\b|\bgiven\b|\bafter accounting for\b|\bholding\b|\bwhile\b|$))",
    re.I,
)


def _protect_bivariate_and(text: str) -> str:
    """Protect the "and" inside bivariate and multi-predictor argument lists
    (e.g. "correlation between revenue and marketing_spend", "compare revenue and cost",
    "Are price and marketing_spend associated with annual_sales?") from
    being treated as a compound-clause connective.
    """
    # 1. Protect 'and' in inverted relation questions: "Are price and marketing_spend associated with..."
    def _protect_inverted(m: re.Match) -> str:
        subject = m.group(1)
        protected_sub = re.sub(r"\band\b", "\x00BIVAND\x00", subject, flags=re.I)
        return m.group(0).replace(subject, protected_sub, 1)

    text = _INVERTED_RELATION_RE.sub(_protect_inverted, text)

    # 2. Protect 'and' following bivariate / relation triggers: "between X and Y",
    # "associated with X and Y", and "depend on X and Y".
    parts = _BIVARIATE_TRIGGER_RE.split(text)
    triggers = _BIVARIATE_TRIGGER_RE.findall(text)
    if triggers:
        out = [parts[0]]
        for trigger, after in zip(triggers, parts[1:]):
            after = re.sub(r"\band\b", "\x00BIVAND\x00", after, count=1, flags=re.I)
            out.append(trigger)
            out.append(after)
        text = "".join(out)

    def _protect_relation_object(m: re.Match) -> str:
        body = m.group(1)
        body = re.sub(r"\band\b", "\x00BIVAND\x00", body, count=1, flags=re.I)
        return m.group(0).replace(m.group(1), body, 1)

    return _RELATION_OBJECT_RE.sub(_protect_relation_object, text)


def interpret_question(question: str) -> NaturalLanguageInterpretation:
    q = " ".join((question or "").split())
    ql = q.lower()
    protected_q = _protect_bivariate_and(q)
    clauses = [
        c.strip().replace("\x00BIVAND\x00", "and")
        for c in re.split(r"\b(?:and|while|whereas|or else)\b|[;]", protected_q, flags=re.I)
        if c.strip()
    ]
    asks_for_cause = bool(re.search(r"\bwhy\b|\bwhat (?:caused|drove|driving|led)\b|\bdriver", ql))
    asks_for_prediction = bool(re.search(r"\bwill\b|\bforecast\b|\bpredict\b|\blikely\b|\brisk\b|\bnext\b", ql))
    asks_for_action = bool(re.search(r"\bwhat should\b|\brecommend\b|\baction\b|\boptimi[sz]e?\b", ql))
    if asks_for_action:
        task = "PRESCRIPTIVE"
    elif asks_for_cause:
        task = "DIAGNOSTIC"
    elif asks_for_prediction:
        task = "PREDICTION"
    elif re.search(r"\b(compare|difference|higher|lower|best|worst|versus|vs)\b", ql):
        task = "COMPARISON"
    elif re.search(r"\b(correlat\w*|relationship\w*|associat\w*|link\w*)\b", ql):
        task = "ASSOCIATION"
    elif re.search(r"\b(check|quality|missing|duplicate|outlier|invalid)\b", ql):
        task = "DATA_QUALITY"
    else:
        task = "DESCRIPTIVE"

    target_phrases = re.findall(r"(?:average|mean|sum|total|revenue|sales|cost|profit|margin|rate|conversion|churn|orders?|customers?)\b", ql)
    grouping = re.findall(r"(?:by|per|for each|across)\s+([a-z][a-z0-9_ -]{1,40})", ql)
    temporal = re.findall(r"\b(?:today|yesterday|this month|last month|this quarter|last quarter|this year|last year|q[1-4]|trailing \d+ (?:days?|months?|years?))\b", ql)
    notes = []
    if len(clauses) > 1:
        notes.append("compound_question_detected; deterministic planner should preserve each clause rather than answer only the first.")
    if not target_phrases:
        notes.append("target_not_explicit; schema/metric semantics must resolve the target or return insufficient evidence.")
    return NaturalLanguageInterpretation(task, target_phrases, grouping, temporal, clauses, asks_for_action, asks_for_cause, asks_for_prediction, notes)
