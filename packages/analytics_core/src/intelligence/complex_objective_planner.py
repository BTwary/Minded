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


@dataclass(frozen=True)
class CompoundObjectivePlan:
    is_compound: bool
    objectives: List[ObjectiveSpec]
    dependency_blocked: bool = False
    dependency_reason: str | None = None


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
_METRIC_TOKEN = r"(?:revenue|sales|profit|cost|costs|margin|orders|customers|churn|conversion|quantity|units|price|discount)"
_METRIC_ENUM_RE = re.compile(
    rf"(?P<prefix>.*?)(?P<metrics>{_METRIC_TOKEN})(?:\s*,\s*(?P<more>{_METRIC_TOKEN}(?:\s*,\s*{_METRIC_TOKEN})*))?\s+and\s+(?P<last>{_METRIC_TOKEN})(?P<suffix>\s+(?:by|per|across|for each)\b.*)$",
    re.I,
)
# Symmetric case: the metric list trails the grouping phrase instead of
# leading it, e.g. "rank the top 5 products BY revenue and profit" -- the
# original regex only handled "revenue and profit BY region", so a metric
# enumeration placed after "by" fell through untouched and lost every
# metric but whichever the single-target resolver happened to guess.
_METRIC_ENUM_AFTER_BY_RE = re.compile(
    rf"^(?P<prefix>.*\bby\s+)(?P<metrics>{_METRIC_TOKEN})(?:\s*,\s*(?P<more>{_METRIC_TOKEN}(?:\s*,\s*{_METRIC_TOKEN})*))?\s+and\s+(?P<last>{_METRIC_TOKEN})(?P<suffix>[^a-zA-Z]*)$",
    re.I,
)


def _expand_metric_list(prefix: str, metrics_head: str, more: str | None, last: str, suffix: str) -> list[str] | None:
    metrics = [metrics_head]
    if more:
        metrics.extend([x.strip() for x in more.split(",")])
    metrics.append(last)
    # Preserve the user's wording while making every target independently executable.
    out = []
    seen = set()
    for metric in metrics:
        key = metric.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(f"{prefix}{metric}{suffix}".strip())
    return out if len(out) > 1 else None


def _metric_enumeration_objectives(question: str) -> list[str] | None:
    """Expand a conjunction of metrics into independently executable objectives.

    Handles both "revenue and profit by region" (list before the grouping
    clause) and "top 5 products by revenue and profit" (list after it).
    """
    q = question.strip()
    m = _METRIC_ENUM_RE.match(q)
    if m:
        expanded = _expand_metric_list(m.group("prefix"), m.group("metrics"), m.group("more"), m.group("last"), m.group("suffix"))
        if expanded:
            return expanded
    m = _METRIC_ENUM_AFTER_BY_RE.match(q)
    if m:
        return _expand_metric_list(m.group("prefix"), m.group("metrics"), m.group("more"), m.group("last"), m.group("suffix"))
    return None


def _split_clauses(question: str) -> list[str]:
    q = " ".join((question or "").split()).strip()
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
    return CompoundObjectivePlan(is_compound=True, objectives=objectives)
