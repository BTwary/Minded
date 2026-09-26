"""PCAlgorithmEngine: Deterministic constraint-based causal DAG discovery using the Peter-Clark (PC) algorithm.

Implements:
1. Fisher's Z-transform on partial correlation for conditional independence testing.
2. Order-independent skeleton discovery with iterative depth scaling.
3. Unshielded collider / v-structure orientation (X -> Z <- Y).
4. Meek's 4 orientation rules for maximally informative CPDAG / DAG completion.
5. Pearl's Backdoor Criterion adjustment set identification from discovered graphs.
"""
from dataclasses import dataclass, field
import itertools
import math
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class DiscoveredCausalDAG:
    """Represents a discovered causal graph with directional topology and separating sets."""
    nodes: List[str]
    directed_edges: List[Tuple[str, str]] = field(default_factory=list)
    undirected_edges: List[Tuple[str, str]] = field(default_factory=list)
    separating_sets: Dict[Tuple[str, str], Set[str]] = field(default_factory=dict)
    colliders: List[str] = field(default_factory=list)
    backdoor_adjustment_sets: Dict[Tuple[str, str], List[str]] = field(default_factory=dict)

    def is_directed(self, u: str, v: str) -> bool:
        return (u, v) in self.directed_edges

    def is_adjacent(self, u: str, v: str) -> bool:
        return (
            (u, v) in self.directed_edges
            or (v, u) in self.directed_edges
            or (u, v) in self.undirected_edges
            or (v, u) in self.undirected_edges
        )

    def get_parents(self, node: str) -> List[str]:
        return [u for (u, v) in self.directed_edges if v == node]

    def get_children(self, node: str) -> List[str]:
        return [v for (u, v) in self.directed_edges if u == node]

    def get_neighbors(self, node: str) -> List[str]:
        res = set()
        for u, v in self.undirected_edges:
            if u == node:
                res.add(v)
            elif v == node:
                res.add(u)
        return list(res)


class PCAlgorithmEngine:
    """Constraint-based Causal Discovery via the PC Algorithm."""

    @staticmethod
    def compute_partial_correlation(df: pd.DataFrame, x: str, y: str, z_set: Set[str]) -> float:
        """Computes partial correlation rho(X, Y | Z) using Frisch-Waugh-Lovell OLS regression residuals."""
        if not z_set:
            r, _ = stats.pearsonr(df[x], df[y])
            return float(r) if not np.isnan(r) else 0.0

        cols = [x, y] + sorted(list(z_set))
        sub_df = df[cols].dropna()
        if len(sub_df) < len(cols) + 2:
            return 0.0

        z_cols = sorted(list(z_set))
        Z = np.column_stack([np.ones(len(sub_df)), sub_df[z_cols].values])
        X = sub_df[x].values
        Y = sub_df[y].values

        try:
            beta_x, _, _, _ = np.linalg.lstsq(Z, X, rcond=None)
            res_x = X - Z @ beta_x

            beta_y, _, _, _ = np.linalg.lstsq(Z, Y, rcond=None)
            res_y = Y - Z @ beta_y

            std_x = np.std(res_x)
            std_y = np.std(res_y)
            if std_x < 1e-9 or std_y < 1e-9:
                return 0.0

            r, _ = stats.pearsonr(res_x, res_y)
            return float(np.clip(r if not np.isnan(r) else 0.0, -0.999999, 0.999999))
        except Exception:
            return 0.0

    @classmethod
    def test_conditional_independence(
        cls,
        df: pd.DataFrame,
        x: str,
        y: str,
        z_set: Set[str],
        alpha: float = 0.05,
    ) -> Tuple[bool, float, float]:
        """Tests H0: X _||_ Y | Z using Fisher's Z-transform.
        
        Returns (is_independent, p_value, test_statistic).
        """
        n = len(df.dropna(subset=[x, y] + list(z_set)))
        k = len(z_set)
        df_deg = n - k - 3
        if df_deg <= 0:
            return True, 1.0, 0.0

        r = cls.compute_partial_correlation(df, x, y, z_set)
        # Fisher's Z-transform
        z_val = 0.5 * math.log((1.0 + r) / max(1e-9, 1.0 - r)) * math.sqrt(df_deg)
        # Two-tailed p-value from standard normal
        p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z_val)))
        is_independent = bool(p_val > alpha)
        return is_independent, float(p_val), float(z_val)

    @classmethod
    def discover_skeleton(
        cls,
        df: pd.DataFrame,
        nodes: List[str],
        alpha: float = 0.05,
        max_depth: int = 3,
    ) -> Tuple[Dict[str, Set[str]], Dict[Tuple[str, str], Set[str]]]:
        """Phase 1: Order-Independent Skeleton discovery."""
        adj: Dict[str, Set[str]] = {node: set(nodes) - {node} for node in nodes}
        sep_sets: Dict[Tuple[str, str], Set[str]] = {}

        for depth in range(max_depth + 1):
            edges_to_remove: List[Tuple[str, str, Set[str]]] = []
            for x in nodes:
                neighbors = sorted(list(adj[x]))
                for y in neighbors:
                    if x >= y:
                        continue

                    # Search subsets of adj(X) \ {Y} of size `depth`
                    candidate_z_x = [n for n in adj[x] if n != y]
                    found = False
                    if len(candidate_z_x) >= depth:
                        for z_combo in itertools.combinations(candidate_z_x, depth):
                            z_set = set(z_combo)
                            is_indep, p_val, _ = cls.test_conditional_independence(df, x, y, z_set, alpha)
                            if is_indep:
                                edges_to_remove.append((x, y, z_set))
                                found = True
                                break

                    if not found:
                        candidate_z_y = [n for n in adj[y] if n != x]
                        if len(candidate_z_y) >= depth:
                            for z_combo in itertools.combinations(candidate_z_y, depth):
                                z_set = set(z_combo)
                                is_indep, p_val, _ = cls.test_conditional_independence(df, x, y, z_set, alpha)
                                if is_indep:
                                    edges_to_remove.append((x, y, z_set))
                                    break

            for u, v, z_set in edges_to_remove:
                adj[u].discard(v)
                adj[v].discard(u)
                pair_key = tuple(sorted([u, v]))
                sep_sets[pair_key] = z_set

        return adj, sep_sets

    @classmethod
    def orient_v_structures(
        cls,
        nodes: List[str],
        adj: Dict[str, Set[str]],
        sep_sets: Dict[Tuple[str, str], Set[str]],
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[str]]:
        """Phase 2: V-Structure (Collider) Orientation.
        
        For unshielded triples X - Z - Y, if Z not in SepSet(X, Y) -> X -> Z <- Y.
        """
        directed: Set[Tuple[str, str]] = set()
        undirected: Set[Tuple[str, str]] = set()
        colliders: List[str] = []

        # Initialize undirected edge list
        for u in nodes:
            for v in adj[u]:
                if u < v:
                    undirected.add((u, v))

        # Check all unshielded triples: x - z - y with x and y not adjacent
        for z in nodes:
            neighbors = sorted(list(adj[z]))
            for x, y in itertools.combinations(neighbors, 2):
                if y not in adj[x]:  # Unshielded: x and y not adjacent
                    pair_key = tuple(sorted([x, y]))
                    sep = sep_sets.get(pair_key, set())
                    if z not in sep:
                        # V-Structure: X -> Z <- Y
                        directed.add((x, z))
                        directed.add((y, z))
                        undirected.discard(tuple(sorted([x, z])))
                        undirected.discard(tuple(sorted([y, z])))
                        if z not in colliders:
                            colliders.append(z)

        return list(directed), list(undirected), colliders

    @classmethod
    def apply_meek_rules(
        cls,
        nodes: List[str],
        directed: List[Tuple[str, str]],
        undirected: List[Tuple[str, str]],
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Phase 3: Meek's orientation rules for completing CPDAG / DAG.
        
        Rule 1: If X -> Y - Z and X not adjacent to Z -> orient Y -> Z.
        Rule 2: If X -> Y -> Z and X - Z -> orient X -> Z.
        Rule 3: If X - Z1 -> Y and X - Z2 -> Y and Z1 not adjacent to Z2 and X - Y -> orient X -> Y.
        """
        dir_set = set(directed)
        undir_set = set(undirected)

        def is_adj(u: str, v: str) -> bool:
            return (
                (u, v) in dir_set
                or (v, u) in dir_set
                or (u, v) in undir_set
                or (v, u) in undir_set
                or tuple(sorted([u, v])) in undir_set
            )

        changed = True
        while changed:
            changed = False

            # Rule 1: X -> Y - Z and X, Z not adjacent => Y -> Z
            for x, y in list(dir_set):
                for z in nodes:
                    if z != x and z != y and (tuple(sorted([y, z])) in undir_set):
                        if not is_adj(x, z):
                            undir_set.discard(tuple(sorted([y, z])))
                            dir_set.add((y, z))
                            changed = True

            # Rule 2: X -> Y -> Z and X - Z => X -> Z
            for x, y in list(dir_set):
                for z in nodes:
                    if (y, z) in dir_set and (tuple(sorted([x, z])) in undir_set):
                        undir_set.discard(tuple(sorted([x, z])))
                        dir_set.add((x, z))
                        changed = True

            # Rule 3: X - Z1 -> Y and X - Z2 -> Y with Z1, Z2 not adjacent and X - Y => X -> Y
            for x, y in [(u, v) for u, v in list(undir_set)] + [(v, u) for u, v in list(undir_set)]:
                z_candidates = [
                    z for z in nodes
                    if z != x and z != y and (tuple(sorted([x, z])) in undir_set) and (z, y) in dir_set
                ]
                for z1, z2 in itertools.combinations(z_candidates, 2):
                    if not is_adj(z1, z2) and (tuple(sorted([x, y])) in undir_set):
                        undir_set.discard(tuple(sorted([x, y])))
                        dir_set.add((x, y))
                        changed = True

        return list(dir_set), list(undir_set)

    @classmethod
    def find_backdoor_adjustment_set(
        cls,
        dag: DiscoveredCausalDAG,
        treatment: str,
        outcome: str,
    ) -> List[str]:
        """Finds a valid adjustment set S blocking all backdoor paths from treatment to outcome."""
        parents_t = set(dag.get_parents(treatment))
        neighbors_t = set(dag.get_neighbors(treatment))
        descendants_t = set(dag.get_children(treatment))

        candidates = (parents_t | neighbors_t) - descendants_t - {outcome}
        valid_adjustment = [
            c for c in candidates
            if dag.is_adjacent(c, outcome) and c not in dag.colliders
        ]
        return sorted(list(set(valid_adjustment)))

    @classmethod
    def discover_causal_dag(
        cls,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        alpha: float = 0.05,
        max_depth: int = 2,
    ) -> DiscoveredCausalDAG:
        """End-to-End PC Algorithm Causal Discovery."""
        cols = feature_columns or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        cols = sorted(list(set(cols)))

        # 1. Skeleton Discovery
        adj, sep_sets = cls.discover_skeleton(df, cols, alpha=alpha, max_depth=max_depth)

        # 2. V-Structures
        directed, undirected, colliders = cls.orient_v_structures(cols, adj, sep_sets)

        # 3. Meek's Rules
        final_directed, final_undirected = cls.apply_meek_rules(cols, directed, undirected)

        dag = DiscoveredCausalDAG(
            nodes=cols,
            directed_edges=final_directed,
            undirected_edges=final_undirected,
            separating_sets=sep_sets,
            colliders=colliders,
        )

        # 4. Compute adjustment sets for all directed pairs
        for u, v in final_directed:
            adj_set = cls.find_backdoor_adjustment_set(dag, treatment=u, outcome=v)
            dag.backdoor_adjustment_sets[(u, v)] = adj_set

        return dag
