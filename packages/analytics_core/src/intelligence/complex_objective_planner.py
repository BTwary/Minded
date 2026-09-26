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
    r"\b(?:those|these|such|that|the same|affected|above|previous|earlier|mentioned)\b"
    r"|\b(?:among|for|of|within)\s+(?:those|these)\b",
    re.I,
)
_TASK_AFTER_AND_RE = re.compile(
    r"\s+\band\s+(?=(?:forecast|predict|compare|why|what|which|did|does|is|are|will|identify|find|determine|rank|check)\b)",
    re.I,
)
_COMMA_CLAUSE_RE = re.compile(
    r",\s*(?=(?:and\s+)?(?:why|what|which|did|does|is|are|can|will|identify|find|determine|rank|check)\b)",
    re.I,
)
_METRIC_ENUM_RE = re.compile(
    r"(?P<prefix>.*?)(?P<metrics>revenue|sales|profit|cost|margin|orders|customers)(?:\s*,\s*(?P<more>(?:revenue|sales|profit|cost|margin|orders|customers)(?:\s*,\s*(?:revenue|sales|profit|cost|margin|orders|customers))*))?\s+and\s+(?P<last>revenue|sales|profit|cost|margin|orders|customers)(?P<suffix>\s+(?:by|per|across|for each)\b.*)$",
    re.I,
)


def _metric_enumeration_objectives(question: str) -> list[str] | None:
    """Expand "revenue and profit by region" into two explicit objectives."""
    m = _METRIC_ENUM_RE.match(question.strip())
    if not m:
        return None
    prefix = m.group("prefix")
    metrics = [m.group("metrics")]
    if m.group("more"):
        metrics.extend([x.strip() for x in m.group("more").split(",")])
    metrics.append(m.group("last"))
    suffix = m.group("suffix")
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
