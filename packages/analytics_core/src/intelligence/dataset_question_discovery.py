"""Deterministic dataset-driven question discovery for MindEd AA-OS.

The discovery layer never invents facts or numbers. It inspects the actual
loaded datasets and emits only questions whose required structural evidence
is present in the local semantic world model/dataframes. External AI is not
required; providers may later enrich wording, but may not manufacture a
capability that the local data cannot support.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from packages.analytics_core.src.semantic.metric_semantics import MetricSemanticsResolver
from packages.analytics_core.src.semantic.world_model import SemanticWorldModelBuilder
from packages.schemas.src.semantic_graph import EpistemicSource


@dataclass(frozen=True)
class SuggestedQuestion:
    question: str
    rationale: str
    evidence_scope: List[str] = field(default_factory=list)
    answerability: str = "SUPPORTED_BY_SCHEMA"
    utility_score: float = 0.5
    epistemic_tier: str = "INFERRED_SEMANTICS"
    analytical_problem_class: str = "DESCRIPTIVE_AGGREGATION"
    expected_information_gain: Optional[float] = None
    feasibility_score: float = 1.0
    redundancy_penalty: float = 0.0


EPISTEMIC_WEIGHTS: Dict[str, float] = {
    EpistemicSource.PHYSICAL_FACT.value: 1.0,
    EpistemicSource.USER_CONFIRMATION.value: 0.95,
    EpistemicSource.EXTERNAL_DECLARATION.value: 0.90,
    EpistemicSource.INFERRED_SEMANTICS.value: 0.75,
    EpistemicSource.MODEL_HYPOTHESIS.value: 0.50,
}

INFORMATION_VALUE_PRIOR_BY_PROBLEM_CLASS: Dict[str, float] = {
    "DESCRIPTIVE_RECORD_COUNT": 0.35,
    "DESCRIPTIVE_AGGREGATION": 0.55,
    "SEGMENT_COMPARISON": 0.75,
    "TEMPORAL_TREND": 0.80,
    "LONGITUDINAL_COMPARISON": 0.88,
    "RELATIONAL_CROSS_TABLE": 0.85,
    "IDENTITY_RECONCILIATION": 0.90,
}


class DatasetQuestionDiscovery:
    """Discover useful, locally answerable questions directly from datasets with multi-dimensional ranking."""

    MAX_QUESTIONS = 12

    @classmethod
    def discover(
        cls,
        datasets: Dict[str, pd.DataFrame],
        *,
        world_model=None,
        max_questions: Optional[int] = None,
    ) -> List[SuggestedQuestion]:
        if not datasets:
            return []

        model = world_model or SemanticWorldModelBuilder().build_world_model(datasets)
        limit = max(1, int(max_questions or cls.MAX_QUESTIONS))
        raw_candidates: List[SuggestedQuestion] = []

        metrics_by_table: Dict[str, List[str]] = {}
        dims_by_table: Dict[str, List[str]] = {}
        times_by_table: Dict[str, List[str]] = {}

        # Build lookup tables from model for epistemic tiers
        entity_by_table = {e.table_name: e for e in getattr(model, "entities", []) or []}
        metric_by_table_col = {
            (m.table_name, m.column_name): m
            for m in getattr(model, "metrics", []) or []
            if m.column_name
        }

        for table_name, df in datasets.items():
            row_count = len(df)
            entity_node = entity_by_table.get(table_name)
            is_empty = row_count == 0

            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            categorical_cols = [
                c for c in df.columns
                if pd.api.types.is_string_dtype(df[c]) or isinstance(df[c].dtype, pd.CategoricalDtype)
            ]
            date_cols = [
                c for c in df.columns
                if pd.api.types.is_datetime64_any_dtype(df[c])
            ]

            # Add columns explicitly represented by the semantic world model.
            model_metrics = [m.column_name for m in getattr(model, "metrics", []) if m.table_name == table_name and m.column_name]
            metrics: List[str] = []
            for c in model_metrics + numeric_cols:
                if c not in metrics and not cls._is_identifier(c):
                    metrics.append(c)
            metrics_by_table[table_name] = metrics[:6]

            dims = []
            for c in categorical_cols:
                if cls._is_identifier(c):
                    continue
                nunique = int(df[c].nunique(dropna=True)) if not is_empty else 0
                if 2 <= nunique <= 50:
                    dims.append(c)
            dims_by_table[table_name] = dims[:6]

            model_times = [t.column_name for t in getattr(model, "time_dimensions", []) if t.table_name == table_name]
            times = []
            for c in model_times + date_cols:
                if c not in times:
                    times.append(c)
            times_by_table[table_name] = times[:3]

            entity_label = cls._display(table_name)

            # 1. Record Count Candidate
            count_feasibility = 1.0 if row_count >= 5 else (0.5 if row_count > 0 else 0.0)
            count_answerability = "SUPPORTED_BY_SCHEMA" if row_count > 0 else "EMPTY_DATASET"
            count_epistemic = (
                EpistemicSource.PHYSICAL_FACT.value
                if (entity_node and entity_node.grain_proven)
                else EpistemicSource.PHYSICAL_FACT.value
            )
            count_prior = INFORMATION_VALUE_PRIOR_BY_PROBLEM_CLASS["DESCRIPTIVE_RECORD_COUNT"]
            count_utility = cls._calculate_utility(
                feasibility=count_feasibility,
                epistemic_tier=count_epistemic,
                information_value_prior=count_prior,
                relational_safety=1.0,
            )
            raw_candidates.append(SuggestedQuestion(
                question=f"How many {entity_label} records are in the dataset?",
                rationale="The dataset contains rows that can be counted directly from verifiable physical storage.",
                evidence_scope=[table_name],
                answerability=count_answerability,
                utility_score=count_utility,
                epistemic_tier=count_epistemic,
                analytical_problem_class="DESCRIPTIVE_RECORD_COUNT",
                feasibility_score=count_feasibility,
            ))

            for metric in metrics[:3]:
                metric_label = cls._display(metric)
                m_node = metric_by_table_col.get((table_name, metric))
                metric_epistemic = (
                    getattr(m_node.epistemic_source, "value", str(m_node.epistemic_source))
                    if m_node
                    else EpistemicSource.INFERRED_SEMANTICS.value
                )
                missing_ratio = float(df[metric].isna().mean()) if row_count > 0 else 1.0
                metric_feasibility = 1.0
                metric_answerability = "SUPPORTED_BY_SCHEMA"
                if missing_ratio > 0.5:
                    metric_feasibility *= 0.6
                    metric_answerability = "HIGH_MISSINGNESS_RISK"
                elif is_empty:
                    metric_feasibility = 0.0
                    metric_answerability = "EMPTY_DATASET"

                # 2. Overall Metric Distribution / Aggregation
                if m_node is None:
                    # Do not generate a metric suggestion from an unresolved
                    # semantic node. Unknown aggregation is not evidence for SUM.
                    continue
                agg_type = getattr(m_node, "aggregation_type", "UNKNOWN")
                agg_prior = INFORMATION_VALUE_PRIOR_BY_PROBLEM_CLASS["DESCRIPTIVE_AGGREGATION"]
                agg_utility = cls._calculate_utility(
                    feasibility=metric_feasibility,
                    epistemic_tier=metric_epistemic,
                    information_value_prior=agg_prior,
                    relational_safety=1.0,
                )
                raw_candidates.append(SuggestedQuestion(
                    question=f"What is the overall {metric_label} in {entity_label}?",
                    rationale=f"A numeric measure is present in the table and can be summarized locally with resolved {agg_type} semantics.",
                    evidence_scope=[table_name, metric],
                    answerability=metric_answerability,
                    utility_score=agg_utility,
                    epistemic_tier=metric_epistemic,
                    analytical_problem_class="DESCRIPTIVE_AGGREGATION",
                    feasibility_score=metric_feasibility,
                ))

                # 3. Categorical Segment Comparison
                # Do not let column order choose a grouping dimension. Generate the
                # autonomous suggestion only when semantic discovery has exactly one
                # admissible dimension for this table. Ambiguity must be resolved by
                # the semantic planner rather than guessed here.
                if len(dims_by_table[table_name]) == 1:
                    dim = dims_by_table[table_name][0]
                    dim_prior = INFORMATION_VALUE_PRIOR_BY_PROBLEM_CLASS["SEGMENT_COMPARISON"]
                    dim_utility = cls._calculate_utility(
                        feasibility=metric_feasibility,
                        epistemic_tier=metric_epistemic,
                        information_value_prior=dim_prior,
                        relational_safety=1.0,
                    )
                    raw_candidates.append(SuggestedQuestion(
                        question=f"How does {metric_label} differ across {cls._display(dim)}?",
                        rationale="A numeric measure and a low-cardinality categorical dimension coexist in the same table.",
                        evidence_scope=[table_name, metric, dim],
                        answerability=metric_answerability,
                        utility_score=dim_utility,
                        epistemic_tier=metric_epistemic,
                        analytical_problem_class="SEGMENT_COMPARISON",
                        feasibility_score=metric_feasibility,
                    ))

                # 4. Temporal Trend
                # A table with several plausible time columns cannot safely receive
                # a proactive trend question without a temporal semantic binding.
                if len(times_by_table[table_name]) == 1:
                    time_col = times_by_table[table_name][0]
                    trend_prior = INFORMATION_VALUE_PRIOR_BY_PROBLEM_CLASS["TEMPORAL_TREND"]
                    trend_utility = cls._calculate_utility(
                        feasibility=metric_feasibility,
                        epistemic_tier=metric_epistemic,
                        information_value_prior=trend_prior,
                        relational_safety=1.0,
                    )
                    raw_candidates.append(SuggestedQuestion(
                        question=f"What trend does {metric_label} show over {cls._display(time_col)}?",
                        rationale="The table contains both a numeric measure and a detected time dimension.",
                        evidence_scope=[table_name, metric, time_col],
                        answerability=metric_answerability,
                        utility_score=trend_utility,
                        epistemic_tier=metric_epistemic,
                        analytical_problem_class="TEMPORAL_TREND",
                        feasibility_score=metric_feasibility,
                    ))

                # 5. Longitudinal Group Comparison
                if len(dims_by_table[table_name]) == 1 and len(times_by_table[table_name]) == 1:
                    dim = dims_by_table[table_name][0]
                    long_prior = INFORMATION_VALUE_PRIOR_BY_PROBLEM_CLASS["LONGITUDINAL_COMPARISON"]
                    long_utility = cls._calculate_utility(
                        feasibility=metric_feasibility,
                        epistemic_tier=metric_epistemic,
                        information_value_prior=long_prior,
                        relational_safety=1.0,
                    )
                    raw_candidates.append(SuggestedQuestion(
                        question=f"Which {cls._display(dim)} groups are changing most over time for {metric_label}?",
                        rationale="The same table contains a measure, grouping dimension, and time dimension, enabling a longitudinal comparison.",
                        evidence_scope=[table_name, metric, dim, times_by_table[table_name][0]],
                        answerability=metric_answerability,
                        utility_score=long_utility,
                        epistemic_tier=metric_epistemic,
                        analytical_problem_class="LONGITUDINAL_COMPARISON",
                        feasibility_score=metric_feasibility,
                    ))

            # 6. Accounting / Reconciliation Identity Candidates
            try:
                from packages.analytics_core.src.intelligence.reconciliation_engine import SEMANTIC_ROLES
                table_roles: Dict[str, List[str]] = {}
                for col in df.columns:
                    col_clean = str(col).lower().replace("-", "_")
                    for role, keywords in SEMANTIC_ROLES.items():
                        if any(kw in col_clean for kw in keywords):
                            table_roles.setdefault(role, []).append(col)
                            break

                qty_cols = table_roles.get("QUANTITY", [])
                price_cols = table_roles.get("UNIT_PRICE", [])
                gross_cols = (
                    table_roles.get("GROSS_AMOUNT", [])
                    or table_roles.get("TOTAL", [])
                    or table_roles.get("REVENUE", [])
                )
                # Reconciliation is safe to propose only when each semantic role has
                # exactly one candidate. Multiple quantity/price/gross columns require
                # explicit semantic binding; choosing the first would be arbitrary.
                if len(qty_cols) == 1 and len(price_cols) == 1 and len(gross_cols) == 1:
                    q_col, p_col, g_col = qty_cols[0], price_cols[0], gross_cols[0]
                    recon_prior = INFORMATION_VALUE_PRIOR_BY_PROBLEM_CLASS["IDENTITY_RECONCILIATION"]
                    recon_utility = cls._calculate_utility(
                        feasibility=1.0 if row_count > 0 else 0.0,
                        epistemic_tier=EpistemicSource.INFERRED_SEMANTICS.value,
                        information_value_prior=recon_prior,
                        relational_safety=1.0,
                    )
                    raw_candidates.append(SuggestedQuestion(
                        question=f"Do {cls._display(g_col)} reconcile with {cls._display(q_col)} and {cls._display(p_col)} as an accounting identity?",
                        rationale="The table contains identifiable quantity, unit price, and gross/total measures that admit an algebraic integrity audit.",
                        evidence_scope=[table_name, g_col, q_col, p_col],
                        answerability="SUPPORTED_BY_SCHEMA",
                        utility_score=recon_utility,
                        epistemic_tier=EpistemicSource.INFERRED_SEMANTICS.value,
                        analytical_problem_class="IDENTITY_RECONCILIATION",
                        feasibility_score=1.0,
                    ))
            except Exception:
                pass

        # 7. Cross-table suggestions from verified relationships
        for rel in getattr(model, "relationships", []) or []:
            left_metrics = metrics_by_table.get(rel.source_table, [])
            right_dims = dims_by_table.get(rel.target_table, [])
            # A relationship can be genuine while still leaving several metrics or
            # dimensions available. Do not turn that schema ambiguity into an
            # arbitrary cross-table question.
            if len(left_metrics) != 1 or len(right_dims) != 1:
                continue
            metric = left_metrics[0]
            dim = right_dims[0]

            rel_epistemic = getattr(rel.epistemic_source, "value", str(rel.epistemic_source))
            rel_safety = float(getattr(rel, "join_safety_score", 1.0))
            fanout = bool(getattr(rel, "fanout_risk", False))
            rel_feasibility = rel_safety * (0.65 if fanout else 1.0)
            rel_answerability = "FANOUT_RISK_WARNING" if fanout else "SUPPORTED_BY_SCHEMA"
            rel_prior = INFORMATION_VALUE_PRIOR_BY_PROBLEM_CLASS["RELATIONAL_CROSS_TABLE"]

            rel_utility = cls._calculate_utility(
                feasibility=rel_feasibility,
                epistemic_tier=rel_epistemic,
                information_value_prior=rel_prior,
                relational_safety=rel_safety,
            )

            raw_candidates.append(SuggestedQuestion(
                question=(
                    f"How does {cls._display(metric)} vary across "
                    f"{cls._display(dim)} using the relationship between "
                    f"{cls._display(rel.source_table)} and {cls._display(rel.target_table)}?"
                ),
                rationale=(
                    f"A discovered {getattr(rel.cardinality, 'value', rel.cardinality)} relationship "
                    f"connects {rel.source_table}.{rel.source_column} to "
                    f"{rel.target_table}.{rel.target_column} (safety score: {rel_safety:.2f})."
                ),
                evidence_scope=[
                    rel.source_table, rel.source_column,
                    rel.target_table, rel.target_column,
                    metric, dim,
                ],
                answerability=rel_answerability,
                utility_score=rel_utility,
                epistemic_tier=rel_epistemic,
                analytical_problem_class="RELATIONAL_CROSS_TABLE",
                feasibility_score=rel_feasibility,
            ))

        # Greedy selection with diversity / redundancy penalty
        return cls._rank_and_select(raw_candidates, limit=limit)

    @classmethod
    def _calculate_utility(
        cls,
        *,
        feasibility: float,
        epistemic_tier: str,
        information_value_prior: float,
        relational_safety: float,
    ) -> float:
        w_feasibility = 0.30
        w_epistemic = 0.25
        w_information_value_prior = 0.35
        w_relational = 0.10

        epistemic_score = EPISTEMIC_WEIGHTS.get(epistemic_tier, 0.75)
        raw_score = (
            w_feasibility * max(0.0, min(1.0, feasibility))
            + w_epistemic * epistemic_score
            + w_information_value_prior * max(0.0, min(1.0, information_value_prior))
            + w_relational * max(0.0, min(1.0, relational_safety))
        )
        return round(float(raw_score), 3)

    @classmethod
    def _rank_and_select(
        cls,
        candidates: List[SuggestedQuestion],
        *,
        limit: int,
    ) -> List[SuggestedQuestion]:
        remaining = list(candidates)
        selected: List[SuggestedQuestion] = []
        seen_questions: Set[str] = set()
        feature_frequency: Dict[str, int] = {}

        while len(selected) < limit and remaining:
            best_cand: Optional[SuggestedQuestion] = None
            best_net_score = -1.0
            best_penalty = 0.0
            best_idx = -1

            for idx, cand in enumerate(remaining):
                q_key = " ".join(cand.question.lower().split())
                if q_key in seen_questions:
                    continue

                # Compute redundancy penalty based on already selected features
                overlap_count = sum(feature_frequency.get(feat, 0) for feat in cand.evidence_scope)
                penalty = min(0.35, 0.07 * overlap_count)
                net_score = max(0.0, cand.utility_score - penalty)

                if net_score > best_net_score:
                    best_net_score = net_score
                    best_penalty = penalty
                    best_cand = cand
                    best_idx = idx

            if best_cand is None or best_idx == -1:
                break

            final_question = SuggestedQuestion(
                question=best_cand.question,
                rationale=best_cand.rationale,
                evidence_scope=list(best_cand.evidence_scope),
                answerability=best_cand.answerability,
                utility_score=round(best_net_score, 3),
                epistemic_tier=best_cand.epistemic_tier,
                analytical_problem_class=best_cand.analytical_problem_class,
                expected_information_gain=best_cand.expected_information_gain,
                feasibility_score=best_cand.feasibility_score,
                redundancy_penalty=round(best_penalty, 3),
            )
            selected.append(final_question)
            seen_questions.add(" ".join(best_cand.question.lower().split()))

            for feat in best_cand.evidence_scope:
                feature_frequency[feat] = feature_frequency.get(feat, 0) + 1

            remaining.pop(best_idx)

        return selected

    @staticmethod
    def _is_identifier(column: str) -> bool:
        low = str(column).lower()
        return (
            low in {"id", "key", "code"}
            or any(token in low for token in ("_id", "_key", "_code", "uuid", "zip", "pk", "fk"))
        )

    @staticmethod
    def _display(value: str) -> str:
        return str(value).replace("_", " ").strip()

