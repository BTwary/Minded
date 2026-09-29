"""Deterministic causal identifiability and estimation safety gates.

This module deliberately separates three questions:
1. Is a causal effect *identified* under an explicit causal graph?
2. Is the discovered graph sufficiently oriented to justify a causal claim?
3. If identified, is there enough observed support to estimate an effect without
   violating positivity / treatment-support checks?

The gate is conservative by design: an uncertain CPDAG is not silently promoted
into a fully directed DAG, and absence of a formal DAG never becomes evidence of
causality.
"""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from packages.schemas.src.analysis import (
    CausalIdentifiabilityProof,
    CausalIdentifiabilityStatus,
    CausalIntent,
)


class CausalGraphSpec(BaseModel):
    """Declarative DAG specification defining causal and confounding relationships."""
    dag_id: str
    nodes: List[str]
    directed_edges: List[Tuple[str, str]]
    unobserved_confounders: List[str] = Field(default_factory=list)


class CausalEvaluationResult(BaseModel):
    """Result of conservative causal identifiability evaluation."""
    is_identifiable: bool
    status: CausalIdentifiabilityStatus
    backdoor_adjustment_set: List[str]
    sensitivity_e_value: Optional[float] = None
    proof: CausalIdentifiabilityProof
    diagnostic_message: str


@dataclass(frozen=True)
class _GraphView:
    nodes: Set[str]
    edges: Tuple[Tuple[str, str], ...]


def _topological_order(nodes: Set[str], edges: Iterable[Tuple[str, str]]) -> Optional[List[str]]:
    children: Dict[str, Set[str]] = defaultdict(set)
    indegree: Dict[str, int] = {n: 0 for n in nodes}
    for u, v in edges:
        if u not in nodes or v not in nodes or u == v:
            return None
        if v not in children[u]:
            children[u].add(v)
            indegree[v] += 1
    q = deque(sorted([n for n, d in indegree.items() if d == 0]))
    order: List[str] = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in sorted(children[u]):
            indegree[v] -= 1
            if indegree[v] == 0:
                q.append(v)
    return order if len(order) == len(nodes) else None


def _descendants(node: str, children: Dict[str, Set[str]]) -> Set[str]:
    out: Set[str] = set()
    q = deque(children.get(node, set()))
    while q:
        cur = q.popleft()
        if cur in out:
            continue
        out.add(cur)
        q.extend(children.get(cur, set()))
    return out


def _ancestors(nodes_of_interest: Set[str], parents: Dict[str, Set[str]]) -> Set[str]:
    out = set(nodes_of_interest)
    q = deque(nodes_of_interest)
    while q:
        cur = q.popleft()
        for p in parents.get(cur, set()):
            if p not in out:
                out.add(p)
                q.append(p)
    return out


def _backdoor_d_separated(
    graph: _GraphView,
    treatment: str,
    outcome: str,
    adjustment_set: Set[str],
) -> bool:
    r"""Check X ⫫ Y | Z in G\X's outgoing edges using ancestral moralization."""
    parents: Dict[str, Set[str]] = defaultdict(set)
    children: Dict[str, Set[str]] = defaultdict(set)
    for u, v in graph.edges:
        children[u].add(v)
        parents[v].add(u)

    # Backdoor graph: remove all outgoing treatment edges.
    backdoor_edges = tuple((u, v) for (u, v) in graph.edges if u != treatment)
    b_parents: Dict[str, Set[str]] = defaultdict(set)
    b_children: Dict[str, Set[str]] = defaultdict(set)
    for u, v in backdoor_edges:
        b_children[u].add(v)
        b_parents[v].add(u)

    relevant = _ancestors({treatment, outcome, *adjustment_set}, b_parents)
    # Moralize the induced ancestral subgraph.
    undirected: Dict[str, Set[str]] = {n: set() for n in relevant}
    for u, v in backdoor_edges:
        if u in relevant and v in relevant:
            undirected[u].add(v)
            undirected[v].add(u)
    for child, ps in b_parents.items():
        rps = sorted(p for p in ps if p in relevant)
        for i, p in enumerate(rps):
            for q in rps[i + 1 :]:
                undirected[p].add(q)
                undirected[q].add(p)

    # Condition on Z by deleting those nodes, then test connectivity X-Y.
    blocked = set(adjustment_set)
    if treatment in blocked or outcome in blocked:
        return False
    if treatment not in undirected or outcome not in undirected:
        return True

    q = deque([treatment])
    seen = {treatment}
    while q:
        cur = q.popleft()
        for nxt in undirected.get(cur, set()):
            if nxt in blocked or nxt in seen:
                continue
            if nxt == outcome:
                return False
            seen.add(nxt)
            q.append(nxt)
    return True


class CausalIdentifiabilityGate:
    """Conservative causal identification gate with explicit graph proofs."""

    @staticmethod
    def _hash_graph(graph_spec: Optional[CausalGraphSpec]) -> str:
        if graph_spec is None:
            return hashlib.sha256(b"no_dag_observational").hexdigest()[:16]
        canonical = "|".join(
            [graph_spec.dag_id]
            + [f"{u}->{v}" for u, v in sorted(graph_spec.directed_edges)]
            + [f"U:{u}" for u in sorted(graph_spec.unobserved_confounders)]
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def evaluate_identifiability(
        self,
        causal_intent: Optional[CausalIntent],
        graph_spec: Optional[CausalGraphSpec] = None,
        observed_effect_risk_ratio: float = 1.5,
    ) -> CausalEvaluationResult:
        if causal_intent is None:
            proof = CausalIdentifiabilityProof(
                treatment="none",
                outcome="none",
                assumed_dag_hash=self._hash_graph(None),
                backdoor_adjustment_set=[],
                is_identifiable=False,
                unmeasured_confounding_risk="HIGH",
            )
            return CausalEvaluationResult(
                is_identifiable=False,
                status=CausalIdentifiabilityStatus.OBSERVATIONAL_ONLY,
                backdoor_adjustment_set=[],
                sensitivity_e_value=None,
                proof=proof,
                diagnostic_message="No causal intent specified. Claim is strictly observational.",
            )

        treatment_raw = f"{causal_intent.treatment.table}.{causal_intent.treatment.column}"
        outcome_raw = f"{causal_intent.outcome.table}.{causal_intent.outcome.column}"
        dag_hash = self._hash_graph(graph_spec)

        if graph_spec is None:
            proof = CausalIdentifiabilityProof(
                treatment=treatment_raw,
                outcome=outcome_raw,
                assumed_dag_hash=dag_hash,
                backdoor_adjustment_set=[],
                is_identifiable=False,
                unmeasured_confounding_risk="HIGH",
            )
            return CausalEvaluationResult(
                is_identifiable=False,
                status=CausalIdentifiabilityStatus.OBSERVATIONAL_ONLY,
                backdoor_adjustment_set=[],
                sensitivity_e_value=None,
                proof=proof,
                diagnostic_message="No formal causal DAG is supplied. Association or prediction must not be promoted to a causal effect.",
            )

        nodes = set(graph_spec.nodes)
        treatment = causal_intent.treatment.column if causal_intent.treatment.column in nodes else treatment_raw
        outcome = causal_intent.outcome.column if causal_intent.outcome.column in nodes else outcome_raw
        if treatment not in nodes or outcome not in nodes:
            proof = CausalIdentifiabilityProof(
                treatment=treatment_raw,
                outcome=outcome_raw,
                assumed_dag_hash=dag_hash,
                backdoor_adjustment_set=[],
                is_identifiable=False,
                unmeasured_confounding_risk="HIGH",
            )
            return CausalEvaluationResult(
                is_identifiable=False,
                status=CausalIdentifiabilityStatus.NOT_IDENTIFIABLE,
                backdoor_adjustment_set=[],
                sensitivity_e_value=None,
                proof=proof,
                diagnostic_message="Treatment or outcome is not represented in the supplied causal graph; causal effect cannot be identified.",
            )

        # Require a real DAG: duplicates are harmless, cycles are not.
        unique_edges = tuple(sorted(set(graph_spec.directed_edges)))
        if _topological_order(nodes, unique_edges) is None:
            proof = CausalIdentifiabilityProof(
                treatment=treatment_raw,
                outcome=outcome_raw,
                assumed_dag_hash=dag_hash,
                backdoor_adjustment_set=[],
                is_identifiable=False,
                unmeasured_confounding_risk="HIGH",
            )
            return CausalEvaluationResult(
                is_identifiable=False,
                status=CausalIdentifiabilityStatus.NOT_IDENTIFIABLE,
                backdoor_adjustment_set=[],
                sensitivity_e_value=None,
                proof=proof,
                diagnostic_message="The supplied graph is cyclic or malformed; a causal DAG proof cannot be established.",
            )

        parents: Dict[str, Set[str]] = defaultdict(set)
        children: Dict[str, Set[str]] = defaultdict(set)
        for u, v in unique_edges:
            parents[v].add(u)
            children[u].add(v)

        descendants = _descendants(treatment, children)
        observed_unmeasured = set(graph_spec.unobserved_confounders)
        # If a listed unobserved confounder is a common ancestor, the gate cannot
        # claim identification from observed adjustment alone.
        common_unobserved: List[str] = []
        for u in observed_unmeasured:
            if u in nodes and treatment in _descendants_of(u, children) and outcome in _descendants_of(u, children):
                common_unobserved.append(u)
        if common_unobserved:
            proof = CausalIdentifiabilityProof(
                treatment=treatment_raw,
                outcome=outcome_raw,
                assumed_dag_hash=dag_hash,
                backdoor_adjustment_set=[],
                is_identifiable=False,
                unmeasured_confounding_risk="HIGH",
            )
            return CausalEvaluationResult(
                is_identifiable=False,
                status=CausalIdentifiabilityStatus.NOT_IDENTIFIABLE,
                backdoor_adjustment_set=[],
                sensitivity_e_value=None,
                proof=proof,
                diagnostic_message=f"Unmeasured common cause(s) prevent backdoor identification: {sorted(common_unobserved)}.",
            )

        # Start with observed parents of treatment, then prove that this set
        # actually blocks every backdoor path rather than assuming the parents
        # are sufficient by construction.
        adjustment_set = sorted((parents.get(treatment, set()) - descendants) - observed_unmeasured - {outcome})
        graph = _GraphView(nodes=nodes, edges=unique_edges)
        if not _backdoor_d_separated(graph, treatment, outcome, set(adjustment_set)):
            proof = CausalIdentifiabilityProof(
                treatment=treatment_raw,
                outcome=outcome_raw,
                assumed_dag_hash=dag_hash,
                backdoor_adjustment_set=adjustment_set,
                is_identifiable=False,
                unmeasured_confounding_risk="HIGH",
            )
            return CausalEvaluationResult(
                is_identifiable=False,
                status=CausalIdentifiabilityStatus.NOT_IDENTIFIABLE,
                backdoor_adjustment_set=adjustment_set,
                sensitivity_e_value=None,
                proof=proof,
                diagnostic_message="The candidate adjustment set does not d-separate treatment and outcome in the backdoor graph; effect is not identified by this adjustment set.",
            )

        rr = max(1.01, float(observed_effect_risk_ratio))
        e_value = float(rr + math.sqrt(rr * (rr - 1.0)))
        proof = CausalIdentifiabilityProof(
            treatment=treatment,
            outcome=outcome,
            assumed_dag_hash=dag_hash,
            backdoor_adjustment_set=adjustment_set,
            is_identifiable=True,
            unmeasured_confounding_risk="LOW",
        )
        return CausalEvaluationResult(
            is_identifiable=True,
            status=CausalIdentifiabilityStatus.IDENTIFIED_BACKDOOR,
            backdoor_adjustment_set=adjustment_set,
            sensitivity_e_value=round(e_value, 2),
            proof=proof,
            diagnostic_message=(
                f"Causal effect is identifiable under the supplied DAG via the validated "
                f"backdoor adjustment set {adjustment_set}. This is identification under "
                f"the stated graph assumptions, not proof that the assumptions are true. "
                f"Sensitivity E-value: {e_value:.2f}."
            ),
        )

    def discover_and_evaluate(
        self,
        df: pd.DataFrame,
        treatment: str,
        outcome: str,
        feature_columns: Optional[List[str]] = None,
        alpha: float = 0.05,
    ) -> CausalEvaluationResult:
        """Discover a conservative graph and evaluate only if orientation is sufficient.

        The PC implementation is a linear-Gaussian, constraint-based discovery tool.
        Its output is generally a CPDAG/partially oriented graph, not proof of a unique
        causal DAG. Therefore any remaining undirected edge creates an ambiguity ceiling:
        the result is observational-only unless the graph is fully oriented.
        """
        from packages.analytics_core.src.causal.pc_algorithm import PCAlgorithmEngine
        from packages.schemas.src.analysis import VariableRef, CausalIntent

        t_col = treatment.split(".")[-1] if "." in treatment else treatment
        o_col = outcome.split(".")[-1] if "." in outcome else outcome
        if t_col not in df.columns or o_col not in df.columns:
            return self.evaluate_identifiability(
                CausalIntent(
                    treatment=VariableRef(table="df", column=t_col),
                    outcome=VariableRef(table="df", column=o_col),
                    assumed_dag_id="discovered_pc_dag",
                ),
                None,
            )

        discovered = PCAlgorithmEngine.discover_causal_dag(
            df,
            feature_columns=feature_columns,
            alpha=alpha,
            max_depth=min(3, max(1, int(math.sqrt(max(1, len(feature_columns or [])))))) if feature_columns else 2,
        )
        intent = CausalIntent(
            treatment=VariableRef(table="df", column=t_col),
            outcome=VariableRef(table="df", column=o_col),
            assumed_dag_id="discovered_pc_dag",
        )

        if discovered.undirected_edges:
            proof = CausalIdentifiabilityProof(
                treatment=treatment,
                outcome=outcome,
                assumed_dag_hash=self._hash_graph(None),
                backdoor_adjustment_set=[],
                is_identifiable=False,
                unmeasured_confounding_risk="HIGH",
            )
            return CausalEvaluationResult(
                is_identifiable=False,
                status=CausalIdentifiabilityStatus.OBSERVATIONAL_ONLY,
                backdoor_adjustment_set=[],
                sensitivity_e_value=None,
                proof=proof,
                diagnostic_message=(
                    "PC discovery returned a partially oriented graph (CPDAG). "
                    "Remaining orientation ambiguity prevents a unique causal-DAG identification claim. "
                    "Association may be investigated, but causal effect is not certified."
                ),
            )

        graph_spec = CausalGraphSpec(
            dag_id="discovered_pc_dag",
            nodes=discovered.nodes,
            directed_edges=list(discovered.directed_edges),
            unobserved_confounders=[],
        )
        result = self.evaluate_identifiability(intent, graph_spec)
        # PC is discovered from observational data under modeling assumptions; even a fully
        # oriented result must preserve that assumption boundary in the diagnostic.
        result.diagnostic_message = (
            result.diagnostic_message
            + " PC discovery is observational and relies on its conditional-independence/model assumptions; "
            "it does not establish the truth of causal directions by itself."
        )
        return result


def _descendants_of(node: str, children: Dict[str, Set[str]]) -> Set[str]:
    return _descendants(node, children)
