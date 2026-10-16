"""Deterministic decomposition for genuinely compound analyst questions.

The planner expands clearly separable analytical clauses into objective statements.
It deliberately does not invent dependencies: clauses that explicitly refer to
prior findings are marked dependent and are not executed as independent analyses.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List


@dataclass(frozen=True)
class ObjectiveSpec:
    ordinal: int
    statement: str
    depends_on_prior_objective: bool
    dependency_reason: str | None = None
    # Exact predecessor references prevent a later task from being authorized
    # merely because some unrelated earlier objective succeeded.
    depends_on_ordinals: tuple[int, ...] = ()


@dataclass(frozen=True)
class CompoundObjectivePlan:
    is_compound: bool
    objectives: List[ObjectiveSpec]
    dependency_blocked: bool = False
    dependency_reason: str | None = None
    source_question: str = ""
    validation_issues: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "is_compound": self.is_compound,
            "source_question": self.source_question,
            "dependency_blocked": self.dependency_blocked,
            "dependency_reason": self.dependency_reason,
            "validation_issues": list(self.validation_issues),
            "objectives": [
                {"ordinal": o.ordinal, "statement": o.statement,
                 "depends_on_prior_objective": o.depends_on_prior_objective,
                 "depends_on_ordinals": list(o.depends_on_ordinals),
                 "dependency_reason": o.dependency_reason}
                for o in self.objectives
            ],
        }


_DEPENDENCY_RE = re.compile(
    r"\b(?:those|these|such|that|the same|affected|above|previous|earlier|mentioned|it|its)\b"
    r"|\b(?:among|for|of|within)\s+(?:those|these)\b",
    re.I,
)
# Task-lead words that legitimately start a second analytical clause after
# "and" / a comma. "how", "when", "should" and "could" cover follow-up
# clauses like "... and how has it changed" or "... and should we act on it"
# that were previously left unsplit and fell through as one confused intent.
_TASK_LEAD_WORDS = (
    r"forecast|predict|compare|why|what|which|how|when|did|does|is|are|"
    r"will|would|should|could|can|identify|find|determine|rank|check"
)
_TASK_AFTER_AND_RE = re.compile(
    rf"\s+\band\s+(?=(?:{_TASK_LEAD_WORDS})\b)",
    re.I,
)
_COMMA_CLAUSE_RE = re.compile(
    rf",\s*(?=(?:and\s+)?(?:{_TASK_LEAD_WORDS})\b)",
    re.I,
)
# Metric vocabulary is intentionally broad, but this is a decomposition lexicon,
# not a schema resolver. Every expanded metric must still be grounded downstream
# against actual columns/aliases before execution.
_METRIC_TOKEN = r"(?:gross\s+profit|net\s+profit|gross\s+margin|net\s+margin|average\s+order\s+value|customer\s+acquisition\s+cost|acquisition\s+cost|cost\s+per\s+lead|cost\s+per\s+click|click\s+through\s+rate|conversion\s+rate|response\s+time|resolution\s+time|lifetime\s+value|customer\s+lifetime\s+value|customer\s+count|active\s+users|new\s+customers|repeat\s+customers|unique\s+customers|return\s+rate|refund\s+rate|defect\s+rate|error\s+rate|on\s+time\s+delivery|delivery\s+time|inventory\s+turnover|stockout\s+rate|employee\s+turnover|engagement\s+rate|bounce\s+rate|market\s+share|win\s+rate|close\s+rate|lead\s+volume|average\s+price|unit\s+economics|customer\s+satisfaction|satisfaction\s+score|support\s+tickets|support\s+volume|marketing\s+spend|ad\s+spend|operating\s+income|operating\s+costs|operating\s+expenses|total\s+expenses|total\s+costs|revenue|sales|profit|cost|costs|margin|orders|customers|churn|retention|conversion|quantity|units|price|discount|returns|refunds|expenses|AOV|CAC|CLV|LTV|ROI|ROAS|CTR|CPA|CPC|CPM|DAU|MAU|ARPU|sessions|visits|tickets|satisfaction|growth|growth\s+rate|impressions|clicks|leads|signups|registrations|downloads|subscriptions|cancellations|renewals|transactions|inventory|stock|defects|errors|complaints|attendance|absenteeism|productivity|utilization|cycle\s+time|throughput|backlog|quality|rating|ratings|reviews|refunds|tax|shipping|freight|commission|discounts|units\s+sold|net\s+revenue\s+retention|gross\s+revenue\s+retention|cohort\s+retention|revenue\s+per\s+(?:user|customer|employee)|profit\s+per\s+(?:user|customer|employee)|average\s+revenue\s+per\s+user|average\s+revenue\s+per\s+account|ARPA|ARPPU|paid\s+conversion|trial\s+conversion|lead\s+conversion|cart\s+abandonment|checkout\s+completion|repeat\s+purchase\s+rate|purchase\s+frequency|basket\s+size|average\s+basket\s+value|fulfillment\s+rate|order\s+accuracy|on\s+time\s+in\s+full|SLA\s+breach\s+rate|first\s+response\s+time|first\s+contact\s+resolution|resolution\s+rate|escalation\s+rate|reopen\s+rate|cost\s+per\s+acquisition|cost\s+per\s+conversion|customer\s+effort\s+score|net\s+promoter\s+score|NPS|CSAT|employee\s+engagement|time\s+to\s+hire|time\s+to\s+fill|offer\s+acceptance\s+rate|revenue\s+concentration|customer\s+concentration|supplier\s+concentration|forecast\s+error|forecast\s+bias|inventory\s+days\s+on\s+hand|days\s+sales\s+outstanding|days\s+payable\s+outstanding|cash\s+conversion\s+cycle|working\s+capital|breakage|writeoffs|write\s+offs|bad\s+debt|contribution\s+margin|EBITDA|earnings|gross\s+merchandise\s+value|GMV|take\s+rate|commission\s+rate|impression\s+share|cost\s+per\s+thousand|CPM|organic\s+traffic|paid\s+traffic|engaged\s+sessions|session\s+duration|pages\s+per\s+session|unsubscribe\s+rate|email\s+open\s+rate|email\s+click\s+rate|delivery\s+success\s+rate|payment\s+failure\s+rate|fraud\s+rate|chargeback\s+rate|claim\s+rate|loss\s+ratio|utilization\s+rate|capacity\s+utilization|downtime|uptime|mean\s+time\s+to\s+repair|mean\s+time\s+between\s+failures|incident\s+rate|safety\s+incident\s+rate|scrap\s+rate|rework\s+rate|yield\s+rate|first\s+pass\s+yield|supplier\s+defect\s+rate|purchase\s+order\s+cycle\s+time|lead\s+time|shipping\s+cost|return\s+on\s+investment|payback\s+period|break-even|break\s+even|incremental\s+lift|uplift|baseline|variance|standard\s+deviation|percentile|median|mode|missingness|completeness|duplicate\s+count|duplicate\s+rate|invalid\s+rate|outlier\s+rate|data\s+freshness|data\s+latency)"
# Accept two or more listed metrics, including lists of four or more items.
_METRIC_LIST = rf"(?P<metrics>{_METRIC_TOKEN}(?:\s*,\s*{_METRIC_TOKEN})*(?:\s*,?\s+and\s+{_METRIC_TOKEN})+)"
_METRIC_ENUM_RE = re.compile(
    rf"(?P<prefix>.*?){_METRIC_LIST}(?P<suffix>\s+(?:by|across|for each|within|over|during)\b.*)$",
    re.I,
)
_METRIC_ENUM_AFTER_BY_RE = re.compile(
    rf"^(?P<prefix>.*\bby\s+){_METRIC_LIST}(?P<suffix>.*)$",
    re.I,
)


def _expand_metric_list(prefix: str, metric_text: str, suffix: str) -> list[str] | None:
    # Split the matched metric enumeration only on list separators; preserve
    # multi-word metric names such as "gross profit" and "return rate".
    metrics = re.split(r"\s*,\s*|\s*,?\s+and\s+", metric_text, flags=re.I)
    metrics = [m.strip() for m in metrics if m.strip()]
    out: list[str] = []
    seen: set[str] = set()
    for metric in metrics:
        key = metric.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(f"{prefix}{metric}{suffix}".strip())
    return out if len(out) > 1 else None


def _metric_enumeration_objectives(question: str) -> list[str] | None:
    """Expand a conjunction of metrics before or after a grouping phrase."""
    q = question.strip()
    m = _METRIC_ENUM_RE.match(q)
    if m:
        expanded = _expand_metric_list(m.group("prefix"), m.group("metrics"), m.group("suffix"))
        if expanded:
            return expanded
    m = _METRIC_ENUM_AFTER_BY_RE.match(q)
    if m:
        expanded = _expand_metric_list(m.group("prefix"), m.group("metrics"), m.group("suffix"))
        if expanded:
            return expanded
    return None


def _split_clauses(question: str) -> list[str]:
    q = " ".join((question or "").split()).strip()
    # Joint relationship/correlation requests are a single estimand. This check
    # must precede metric-list expansion: "relationship between price and profit
    # by region" must not become two unrelated descriptive objectives.
    relation_cues = re.search(
        r"\b(?:relationship|correlation|correlat\w*|association|associat\w*|"
        r"jointly\s+predictive|multivariate|covary|moves?\s+together|depend\s+on)\b",
        q, re.I,
    )
    relation_pair = re.search(
        r"\b(?:relationship|correlation|correlat\w*|association|associat\w*|connection)\s+between\s+(.+?)\s+and\s+(.+?)"
        r"(?=\s+(?:by|across|within|among|over|during|for\s+each)\b|[?.!]*$)",
        q, re.I,
    )
    relation_of_with = re.search(
        r"\b(?:correlation|relationship|association)\s+of\s+(.+?)\s+with\s+(.+?)"
        r"(?=\s+(?:by|across|within|among|over|during|for\s+each)\b|[?.!]*$)",
        q, re.I,
    )
    if relation_cues and (relation_pair or relation_of_with):
        return [q]

    expanded = _metric_enumeration_objectives(q)
    if expanded:
        return expanded

    # Split only at conjunctions that start a recognizable analytical clause.
    pieces = _TASK_AFTER_AND_RE.split(q)
    out: list[str] = []
    for piece in pieces:
        out.extend([p.strip(" ,") for p in _COMMA_CLAUSE_RE.split(piece) if p.strip(" ,")])
    return out if out else [q]


def plan_compound_question(question: str) -> CompoundObjectivePlan:
    clauses = _split_clauses(question)
    # Fail closed on pathological/over-segmented prompts rather than creating an
    # unbounded execution plan. The controller can ask the user to split the task.
    if len(clauses) > 12:
        return CompoundObjectivePlan(
            is_compound=True, objectives=[ObjectiveSpec(1, question.strip(), True, "question decomposed into more than 12 objectives")],
            dependency_blocked=True, dependency_reason="compound question exceeds the safe 12-objective planning limit",
            source_question=question.strip(),
        )
    # If the custom splitter found multiple clauses, honor them; otherwise there
    # is no compound orchestration to perform.
    if len(clauses) <= 1:
        return CompoundObjectivePlan(False, [])

    objectives: list[ObjectiveSpec] = []
    for i, clause in enumerate(clauses, start=1):
        dependency_match = _DEPENDENCY_RE.search(clause)
        objectives.append(
            ObjectiveSpec(
                ordinal=i,
                statement=clause,
                depends_on_prior_objective=dependency_match is not None,
                dependency_reason=(
                    f"clause references prior context via {dependency_match.group(0)!r}"
                    if dependency_match else None
                ),
                depends_on_ordinals=((i - 1,) if dependency_match and i > 1 else ()),
            )
        )

    dependent = next((o for o in objectives if o.depends_on_prior_objective), None)
    if dependent:
        return CompoundObjectivePlan(
            is_compound=True,
            objectives=objectives,
            dependency_blocked=True,
            dependency_reason=dependent.dependency_reason,
        )
    known = {o.ordinal for o in objectives}
    issues = tuple(
        f"objective {o.ordinal} has invalid prerequisite ordinal {dep}"
        for o in objectives for dep in o.depends_on_ordinals
        if dep not in known or dep >= o.ordinal
    )
    return CompoundObjectivePlan(
        is_compound=True, objectives=objectives, source_question=question.strip(),
        validation_issues=issues,
    )
