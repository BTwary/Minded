"""Deterministic relational planning for natural-language analytical questions.

The planner discovers relational candidates only from observed schemas/data, then
proves each candidate with the canonical join-safety authority. It never treats
column-name similarity as proof of a safe relationship and never executes SQL.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from packages.analytics_core.src.relational.join_safety import (
    SafetyStatus,
    assess_join_safety,
    assess_metric_aggregation_safety,
)


@dataclass(frozen=True)
class JoinPlan:
    left_dataset: str
    right_dataset: str
    left_key: str
    right_key: str
    join_hops: List[Dict[str, str]]
    status: str
    rationale: str
    overlap_score: float = 0.0


def _norm_key(name: str) -> str:
    """Normalize common relational-key suffixes for candidate generation."""
    s = str(name).lower().strip()
    for suffix in ("_id", "_key", "_code"):
        if s.endswith(suffix):
            return s[: -len(suffix)]
    return s


def _looks_like_key(name: str) -> bool:
    low = str(name).lower().strip()
    return (
        low in {"id", "key", "code"}
        or low.endswith(("_id", "_key", "_code"))
        or low.endswith(("id", "key", "code"))
    )


def _overlap_score(left: pd.Series, right: pd.Series) -> float:
    l_values = set(left.dropna().astype(str))
    r_values = set(right.dropna().astype(str))
    if not l_values or not r_values:
        return 0.0
    overlap = len(l_values & r_values)
    # Jaccard is conservative for asymmetric parent/child tables; containment
    # captures whether the smaller key domain is represented in the larger one.
    jaccard = overlap / max(1, len(l_values | r_values))
    containment = overlap / max(1, min(len(l_values), len(r_values)))
    return max(jaccard, containment * 0.9)


def discover_join_candidates(datasets: Dict[str, pd.DataFrame]) -> List[JoinPlan]:
    """Discover deterministic, data-backed two-table join candidates.

    Candidate generation requires a key-like name or normalized key-name match,
    non-empty overlap, and compatible physical types. Safety is then established
    by the canonical ``assess_join_safety`` authority.
    """
    names = sorted(datasets)
    plans: List[JoinPlan] = []

    for i, left_name in enumerate(names):
        left = datasets[left_name]
        for right_name in names[i + 1 :]:
            right = datasets[right_name]
            candidates: List[Tuple[str, str, float]] = []

            for lk in map(str, left.columns):
                for rk in map(str, right.columns):
                    if not (_looks_like_key(lk) or _looks_like_key(rk)):
                        continue
                    same_normalized_name = _norm_key(lk) == _norm_key(rk)
                    if not same_normalized_name:
                        # Avoid joining arbitrary columns solely because values overlap.
                        continue
                    if not (lk in left.columns and rk in right.columns):
                        continue
                    l_series, r_series = left[lk], right[rk]
                    if l_series.dropna().empty or r_series.dropna().empty:
                        continue
                    score = _overlap_score(l_series, r_series)
                    if score <= 0.0:
                        continue
                    safety = assess_join_safety(
                        left, right, lk, rk,
                        left_table=left_name,
                        right_table=right_name,
                    )
                    # UNKNOWN/UNSAFE candidates remain visible to the planner as
                    # explainable rejected candidates, but SAFE is the only status
                    # eligible for execution by downstream callers.
                    candidates.append((lk, rk, score))

            ranked = sorted(candidates, key=lambda x: (-x[2], x[0], x[1]))[:5]
            for lk, rk, score in ranked:
                safety = assess_join_safety(
                    left, right, lk, rk,
                    left_table=left_name,
                    right_table=right_name,
                )
                status = safety.status.value if hasattr(safety.status, "value") else str(safety.status)

                # Canonical orientation: for a one-to-many relationship, keep
                # the repeated/fact side on the left and the unique/dimension
                # side on the right. This makes downstream metric/group planning
                # deterministic and avoids losing useful fields merely because
                # table names sort alphabetically.
                out_left, out_right = left_name, right_name
                out_lk, out_rk = lk, rk
                out_safety = safety
                if safety.cardinality.value == "one_to_many":
                    out_left, out_right = right_name, left_name
                    out_lk, out_rk = rk, lk
                    out_safety = assess_join_safety(
                        right, left, rk, lk,
                        left_table=right_name,
                        right_table=left_name,
                    )

                out_status = out_safety.status.value if hasattr(out_safety.status, "value") else str(out_safety.status)
                plans.append(
                    JoinPlan(
                        left_dataset=out_left,
                        right_dataset=out_right,
                        left_key=out_lk,
                        right_key=out_rk,
                        join_hops=[{
                            "left_table": out_left,
                            "right_table": out_right,
                            "left_key": out_lk,
                            "right_key": out_rk,
                        }],
                        status=out_status,
                        rationale=(
                            f"Observed normalized key match {out_lk!r}↔{out_rk!r}; "
                            f"value-overlap score={score:.3f}; "
                            f"cardinality={out_safety.cardinality.value}; "
                            f"join safety={out_status}."
                        ),
                        overlap_score=score,
                    )
                )
    return plans


def find_safe_two_table_plan(
    datasets: Dict[str, pd.DataFrame], *, target_column: str, grouping_column: str,
) -> Optional[JoinPlan]:
    """Return the highest-overlap safe plan carrying the requested columns."""
    target_column = str(target_column)
    grouping_column = str(grouping_column)
    safe: List[JoinPlan] = []
    for plan in discover_join_candidates(datasets):
        left_cols = set(map(str, datasets[plan.left_dataset].columns))
        right_cols = set(map(str, datasets[plan.right_dataset].columns))
        if not {target_column, grouping_column}.issubset(left_cols | right_cols):
            continue
        if plan.status.upper() == SafetyStatus.SAFE.value.upper():
            safe.append(plan)
    safe.sort(key=lambda p: (-p.overlap_score, p.left_dataset, p.right_dataset, p.left_key, p.right_key))
    return safe[0] if safe else None


def find_safe_three_table_plan(
    datasets: Dict[str, pd.DataFrame], *, target_column: str, grouping_column: str,
) -> Optional[JoinPlan]:
    """Return one unambiguous, grain-safe, connected two-hop plan.

    The target metric must originate in the base table and the requested
    grouping column in the terminal table.  M2-A has no pre-aggregation, so
    every raw metric aggregate path containing a fanout-risk hop is rejected.
    """
    candidates: List[Tuple[float, JoinPlan]] = []
    pairwise = discover_join_candidates(datasets)
    for first in pairwise:
        if first.status.upper() != SafetyStatus.SAFE.value.upper():
            continue
        base_columns = set(map(str, datasets[first.left_dataset].columns))
        if str(target_column) not in base_columns:
            continue
        for second in pairwise:
            if second.status.upper() != SafetyStatus.SAFE.value.upper():
                continue
            # Pairwise discovery canonically keeps a repeated/fact side on
            # the left. A one-to-one edge has no fact-side orientation, so it
            # may be reversed solely to connect the deterministic path.
            if second.left_dataset == first.right_dataset:
                second_hop = dict(second.join_hops[0])
                terminal = second.right_dataset
            elif second.right_dataset == first.right_dataset:
                reverse_report = assess_join_safety(
                    datasets[second.left_dataset], datasets[second.right_dataset],
                    second.left_key, second.right_key,
                    left_table=second.left_dataset, right_table=second.right_dataset,
                )
                if reverse_report.cardinality.value != "one_to_one":
                    continue
                second_hop = {
                    "left_table": second.right_dataset,
                    "right_table": second.left_dataset,
                    "left_key": second.right_key,
                    "right_key": second.left_key,
                }
                terminal = second.left_dataset
            else:
                continue
            if terminal in {first.left_dataset, first.right_dataset}:
                continue
            if str(grouping_column) not in set(map(str, datasets[terminal].columns)):
                continue
            hops = [dict(first.join_hops[0]), second_hop]
            reports = [
                assess_join_safety(
                    datasets[hop["left_table"]], datasets[hop["right_table"]],
                    hop["left_key"], hop["right_key"],
                    left_table=hop["left_table"], right_table=hop["right_table"],
                )
                for hop in hops
            ]
            aggregate_safety = assess_metric_aggregation_safety(first.left_dataset, reports)
            if not aggregate_safety.safe:
                continue
            candidates.append((
                first.overlap_score + second.overlap_score,
                JoinPlan(
                    left_dataset=first.left_dataset,
                    right_dataset=terminal,
                    left_key=first.left_key,
                    right_key=first.right_key,
                    join_hops=hops,
                    status=SafetyStatus.SAFE.value,
                    rationale=(
                        f"Two-hop path {first.left_dataset}->{first.right_dataset}->{terminal}. "
                        f"{first.rationale} {second.rationale} {aggregate_safety.reason}"
                    ),
                    overlap_score=first.overlap_score + second.overlap_score,
                ),
            ))
    candidates.sort(key=lambda item: (-item[0], item[1].left_dataset, item[1].right_dataset,
                                       item[1].left_key, item[1].right_key))
    if not candidates:
        return None
    # Equal scores are ambiguous rather than arbitrarily resolved.
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        return None
    return candidates[0][1]

def find_safe_n_table_plan(
    datasets: Dict[str, pd.DataFrame],
    *,
    target_column: str,
    grouping_column: str,
    max_hops: int = 5,
) -> Optional[JoinPlan]:
    """Find a deterministic grain-safe path across up to ``max_hops + 1`` tables.

    This generalizes the original two-/three-table discovery.  Only SAFE
    pairwise edges are traversed, the metric must originate in the base table,
    and raw aggregation must not cross a fanout-risk hop.  Ties at the best
    score are rejected rather than guessed.
    """
    target_column = str(target_column)
    grouping_column = str(grouping_column)
    if max_hops < 1:
        raise ValueError("max_hops must be >= 1")

    edges = [p for p in discover_join_candidates(datasets) if p.status.upper() == SafetyStatus.SAFE.value]
    adjacency: Dict[str, List[Tuple[str, Dict[str, str], float]]] = {}
    for edge in edges:
        hop = dict(edge.join_hops[0])
        adjacency.setdefault(edge.left_dataset, []).append((edge.right_dataset, hop, edge.overlap_score))
        # Safe one-to-one relationships can be traversed in either direction.
        report = assess_join_safety(
            datasets[edge.left_dataset], datasets[edge.right_dataset],
            edge.left_key, edge.right_key,
            left_table=edge.left_dataset, right_table=edge.right_dataset,
        )
        if report.cardinality.value == "one_to_one":
            reverse_hop = {
                "left_table": edge.right_dataset,
                "right_table": edge.left_dataset,
                "left_key": edge.right_key,
                "right_key": edge.left_key,
            }
            adjacency.setdefault(edge.right_dataset, []).append((edge.left_dataset, reverse_hop, edge.overlap_score))

    candidates: List[Tuple[float, JoinPlan]] = []
    names = sorted(datasets)

    def dfs(base: str, current: str, visited: set[str], hops: List[Dict[str, str]], score: float) -> None:
        if len(hops) > max_hops:
            return
        if current != base and grouping_column in set(map(str, datasets[current].columns)):
            reports = [
                assess_join_safety(
                    datasets[h["left_table"]], datasets[h["right_table"]],
                    h["left_key"], h["right_key"],
                    left_table=h["left_table"], right_table=h["right_table"],
                )
                for h in hops
            ]
            if all(r.status == SafetyStatus.SAFE for r in reports):
                safety = assess_metric_aggregation_safety(base, reports)
                if safety.safe:
                    candidates.append((
                        score,
                        JoinPlan(
                            left_dataset=base,
                            right_dataset=current,
                            left_key=hops[0]["left_key"],
                            right_key=hops[-1]["right_key"],
                            join_hops=[dict(h) for h in hops],
                            status=SafetyStatus.SAFE.value,
                            rationale=(
                                f"Safe {len(hops)}-hop path {base}->{current}. "
                                + " ".join(
                                    f"{h['left_table']}.{h['left_key']}→{h['right_table']}.{h['right_key']}"
                                    for h in hops
                                )
                                + f". {safety.reason}"
                            ),
                            overlap_score=score,
                        ),
                    ))

        if len(hops) == max_hops:
            return
        neighbors = sorted(adjacency.get(current, []), key=lambda x: (-x[2], x[0], x[1]["left_key"], x[1]["right_key"]))
        for nxt, hop, edge_score in neighbors:
            if nxt in visited:
                continue
            dfs(base, nxt, visited | {nxt}, hops + [dict(hop)], score + edge_score)

    for base in names:
        if target_column not in set(map(str, datasets[base].columns)):
            continue
        dfs(base, base, {base}, [], 0.0)

    if not candidates:
        return None
    candidates.sort(key=lambda item: (-(item[0] / max(1, len(item[1].join_hops))), -item[0], len(item[1].join_hops), item[1].left_dataset, item[1].right_dataset, repr(item[1].join_hops)))
    best_rank = candidates[0][0] / max(1, len(candidates[0][1].join_hops))
    tied = [c for c in candidates if (c[0] / max(1, len(c[1].join_hops))) == best_rank]
    if len(tied) > 1:
        return None
    return candidates[0][1]

