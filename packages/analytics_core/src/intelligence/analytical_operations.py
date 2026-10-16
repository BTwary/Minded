"""Typed deterministic analytical primitives and validated operation graphs.

This module is deliberately independent of natural-language parsing. A parser or a human
clarification flow must supply explicit, schema-grounded operation arguments; this module
validates and executes those arguments without reinterpreting the original question.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import pandas as pd


# Registry metadata is the single source of truth for the descriptive primitives supported
# by this module. SQL/other backends should implement the same semantics before being enabled.
AGGREGATION_REGISTRY: dict[str, dict[str, Any]] = {
    "ROW_COUNT": {"needs_column": False, "numeric": False},
    "DUPLICATE_ROW_COUNT": {"needs_column": False, "numeric": False},
    "COUNT": {"needs_column": True, "numeric": False},
    "COUNT_DISTINCT": {"needs_column": True, "numeric": False},
    "SUM": {"needs_column": True, "numeric": True},
    "MEAN": {"needs_column": True, "numeric": True},
    "MEDIAN": {"needs_column": True, "numeric": True},
    "MIN": {"needs_column": True, "numeric": False},
    "MAX": {"needs_column": True, "numeric": False},
    "VARIANCE": {"needs_column": True, "numeric": True},
    "STDDEV": {"needs_column": True, "numeric": True},
    "MODE": {"needs_column": True, "numeric": False},
    "RANGE": {"needs_column": True, "numeric": True},
    "Q1": {"needs_column": True, "numeric": True},
    "Q3": {"needs_column": True, "numeric": True},
    "IQR": {"needs_column": True, "numeric": True},
    "MISSING_COUNT": {"needs_column": True, "numeric": False},
    "MISSING_PERCENT": {"needs_column": True, "numeric": False},
    "PERCENTILE": {"needs_column": True, "numeric": True, "needs_parameter": True},
    "ABS_SUM": {"needs_column": True, "numeric": True},
    "POSITIVE_COUNT": {"needs_column": True, "numeric": True},
    "NEGATIVE_COUNT": {"needs_column": True, "numeric": True},
    "ZERO_COUNT": {"needs_column": True, "numeric": True},
    "NONZERO_COUNT": {"needs_column": True, "numeric": True},
    "MAD": {"needs_column": True, "numeric": True},
    "SKEWNESS": {"needs_column": True, "numeric": True},
    "KURTOSIS": {"needs_column": True, "numeric": True},
    "COEFFICIENT_OF_VARIATION": {"needs_column": True, "numeric": True},
}


class AnalyticalOperationError(ValueError):
    """The requested operation is unsupported or cannot be executed faithfully."""


class _MissingGroupSentinel:
    pass


_MISSING_GROUP = _MissingGroupSentinel()


def _finite_or_none(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def aggregate_frame(
    frame: pd.DataFrame,
    *,
    operation: str,
    target_column: str | None = None,
    grouping_columns: Sequence[str] = (),
    percentile: float | None = None,
    filters: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Execute a typed aggregation over a DataFrame with explicit null/group semantics.

    Null group keys are retained as ``"<missing>"``. Null metric values are excluded from
    all metric calculations except ROW_COUNT. Filters use the project's canonical
    ``filter_condition_mask`` implementation; unknown operators/columns fail closed.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise AnalyticalOperationError("Dataset must be a non-empty DataFrame.")
    op = str(operation or "").strip().upper()
    spec = AGGREGATION_REGISTRY.get(op)
    if spec is None:
        raise AnalyticalOperationError(f"Unsupported aggregation: {op or '<empty>'}.")
    groups = tuple(str(x) for x in grouping_columns)
    if len(set(groups)) != len(groups):
        raise AnalyticalOperationError("Grouping columns must be unique.")
    missing_groups = [c for c in groups if c not in frame.columns]
    if missing_groups:
        raise AnalyticalOperationError(f"Unknown grouping column(s): {missing_groups}.")
    if spec["needs_column"] and (not target_column or target_column not in frame.columns):
        raise AnalyticalOperationError("The selected aggregation requires a valid target column.")
    if not spec["needs_column"] and target_column is not None:
        # Whole-row operations intentionally have no target column; ignoring one would be
        # a likely misunderstanding and is rejected rather than silently discarded.
        raise AnalyticalOperationError(f"{op} does not accept a target column.")
    if spec.get("numeric") and not pd.api.types.is_numeric_dtype(frame[target_column]):
        raise AnalyticalOperationError(f"{op} requires a numeric column: {target_column}.")
    if op == "PERCENTILE" and (percentile is None or not 0 <= float(percentile) <= 100):
        raise AnalyticalOperationError("PERCENTILE requires an explicit percentile from 0 to 100.")
    if op != "PERCENTILE" and percentile is not None:
        raise AnalyticalOperationError("A percentile parameter is only valid for PERCENTILE.")

    population = frame.copy()
    applied_filters: list[dict[str, Any]] = []
    for condition in filters:
        column = condition.get("column")
        operator = str(condition.get("operator", "")).upper()
        if column not in population.columns:
            raise AnalyticalOperationError(f"Unknown filter column: {column!r}.")
        try:
            from packages.analytics_core.src.intelligence.intent_state import filter_condition_mask
            mask = filter_condition_mask(population[column], operator, condition.get("value"))
        except (ImportError, TypeError, ValueError, KeyError) as exc:
            raise AnalyticalOperationError(f"Invalid filter for {column!r}: {exc}") from exc
        if len(mask) != len(population):
            raise AnalyticalOperationError("Filter mask length does not match the dataset.")
        population = population.loc[mask].copy()
        applied_filters.append(dict(condition))

    def calculate(part: pd.DataFrame) -> Any:
        if op == "ROW_COUNT":
            return int(len(part))
        if op == "DUPLICATE_ROW_COUNT":
            return int(part.duplicated(keep="first").sum())
        series = part[target_column]
        if op == "COUNT": value = series.count()
        elif op == "COUNT_DISTINCT": value = series.nunique(dropna=True)
        elif op == "SUM": value = series.sum(skipna=True)
        elif op == "MEAN": value = series.mean(skipna=True)
        elif op == "MEDIAN": value = series.median(skipna=True)
        elif op == "MIN": value = series.min(skipna=True)
        elif op == "MAX": value = series.max(skipna=True)
        elif op == "VARIANCE": value = series.var(skipna=True, ddof=1)
        elif op == "STDDEV": value = series.std(skipna=True, ddof=1)
        elif op == "MODE":
            counts = series.dropna().value_counts(dropna=True)
            if counts.empty:
                value = None
            else:
                top_count = counts.max()
                tied = [v for v, count in counts.items() if count == top_count]
                value = sorted(tied, key=lambda item: (str(type(item)), str(item)))[0]
        elif op == "RANGE": value = series.max(skipna=True) - series.min(skipna=True) if series.notna().any() else None
        elif op == "Q1": value = series.quantile(0.25, interpolation="linear")
        elif op == "Q3": value = series.quantile(0.75, interpolation="linear")
        elif op == "IQR": value = series.quantile(0.75, interpolation="linear") - series.quantile(0.25, interpolation="linear") if series.notna().any() else None
        elif op == "MISSING_COUNT": value = int(series.isna().sum())
        elif op == "MISSING_PERCENT": value = (float(series.isna().sum()) / len(part) * 100.0) if len(part) else None
        elif op == "ABS_SUM": value = series.abs().sum(skipna=True)
        elif op == "POSITIVE_COUNT": value = int((series.dropna() > 0).sum())
        elif op == "NEGATIVE_COUNT": value = int((series.dropna() < 0).sum())
        elif op == "ZERO_COUNT": value = int((series.dropna() == 0).sum())
        elif op == "NONZERO_COUNT": value = int((series.dropna() != 0).sum())
        elif op == "MAD": value = (series - series.median()).abs().median() if series.notna().any() else None
        elif op == "SKEWNESS": value = series.skew(skipna=True) if series.count() >= 3 else None
        elif op == "KURTOSIS": value = series.kurt(skipna=True) if series.count() >= 4 else None
        elif op == "COEFFICIENT_OF_VARIATION":
            mean = series.mean(skipna=True)
            value = (series.std(skipna=True, ddof=1) / abs(mean)) if series.count() >= 2 and mean != 0 else None
        else: value = series.quantile(float(percentile) / 100.0, interpolation="linear")
        return _finite_or_none(value)

    if groups:
        work = population.copy()
        # Use collision-safe temporary names rather than overwriting a user's real column.
        group_keys = []
        for idx, col in enumerate(groups):
            key = f"__aaos_group_{idx}__"
            while key in work.columns:
                key = "_" + key
            work[key] = work[col].astype(object).where(work[col].notna(), _MISSING_GROUP)
            group_keys.append(key)
        records: list[dict[str, Any]] = []
        grouped = work.groupby(group_keys, dropna=False, sort=False, observed=False)
        for key_values, part in grouped:
            if not isinstance(key_values, tuple):
                key_values = (key_values,)
            group_value = {col: ("<missing>" if val is _MISSING_GROUP else _finite_or_none(val)) for col, val in zip(groups, key_values)}
            value = calculate(part)
            # Independent check uses an independently constructed boolean mask against the
            # filtered population, not the grouped partition iterator.
            mask = pd.Series(True, index=work.index)
            for key, expected in zip(group_keys, key_values):
                mask &= work[key].eq(expected).fillna(False)
            checked = calculate(population.loc[mask])
            if value != checked and not (
                isinstance(value, (int, float)) and isinstance(checked, (int, float))
                and math.isclose(value, checked, rel_tol=1e-12, abs_tol=1e-12)
            ):
                raise AnalyticalOperationError("Independent group recomputation did not match.")
            records.append({"group": group_value, "value": value, "row_count": int(len(part))})
        if len(population) == 0:
            return {
                "operation": op, "target_column": target_column,
                "grouping_columns": list(groups), "percentile": percentile,
                "population_row_count": 0, "filters": applied_filters,
                "null_group_policy": "include_as_<missing>", "records": [],
                "verification": {
                    "status": "unverified_zero_matches",
                    "method": "independent_group_recomputation",
                    "matches": False,
                    "zero_matching_rows": True,
                    "reason": "Filtered population contains zero rows; no groups were evaluated.",
                },
            }
        return {
            "operation": op, "target_column": target_column,
            "grouping_columns": list(groups), "percentile": percentile,
            "population_row_count": int(len(population)), "filters": applied_filters,
            "null_group_policy": "include_as_<missing>", "records": records,
            "verification": {"status": "verified", "method": "independent_group_recomputation", "matches": True, "zero_matching_rows": False},
        }
    if len(population) == 0:
        return {
            "operation": op, "target_column": target_column, "grouping_columns": [],
            "percentile": percentile, "population_row_count": 0,
            "filters": applied_filters, "value": None,
            "verification": {
                "status": "unverified_zero_matches",
                "method": "independent_recomputation",
                "matches": False,
                "zero_matching_rows": True,
                "reason": "Filtered population contains zero rows.",
            },
        }
    value = calculate(population)
    checked = calculate(population.copy())
    if value != checked and not (
        isinstance(value, (int, float)) and isinstance(checked, (int, float))
        and math.isclose(value, checked, rel_tol=1e-12, abs_tol=1e-12)
    ):
        raise AnalyticalOperationError("Independent recomputation did not match.")
    return {
        "operation": op, "target_column": target_column, "grouping_columns": [],
        "percentile": percentile, "population_row_count": int(len(population)),
        "filters": applied_filters, "value": value,
        "verification": {"status": "verified", "method": "independent_recomputation", "matches": True, "zero_matching_rows": False},
    }


@dataclass(frozen=True)
class OperationNode:
    """One persisted operation. Inputs reference dataset columns or prior node outputs."""
    node_id: str
    operation: str
    params: Mapping[str, Any]
    depends_on: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"node_id": self.node_id, "operation": self.operation,
                "params": dict(self.params), "depends_on": list(self.depends_on)}


@dataclass(frozen=True)
class OperationGraph:
    version: int
    nodes: tuple[OperationNode, ...]
    requested_outputs: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "nodes": [n.to_dict() for n in self.nodes],
                "requested_outputs": list(self.requested_outputs)}

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def graph_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def validate_operation_graph(graph: OperationGraph, columns: Sequence[str]) -> tuple[str, ...]:
    """Validate graph identity, supported nodes, references, and acyclic dependencies."""
    if graph.version != 1:
        raise AnalyticalOperationError(f"Unsupported operation graph version: {graph.version}.")
    if not graph.nodes:
        raise AnalyticalOperationError("Operation graph must contain at least one node.")
    ids = [node.node_id for node in graph.nodes]
    if any(not x or not isinstance(x, str) for x in ids) or len(set(ids)) != len(ids):
        raise AnalyticalOperationError("Operation node IDs must be non-empty and unique.")
    by_id = {node.node_id: node for node in graph.nodes}
    for node in graph.nodes:
        if node.operation not in {"AGGREGATE", "RANK"}:
            raise AnalyticalOperationError(f"Unsupported graph operation: {node.operation}.")
        if any(dep not in by_id for dep in node.depends_on):
            raise AnalyticalOperationError(f"Node {node.node_id!r} references a missing dependency.")
        if node.node_id in node.depends_on:
            raise AnalyticalOperationError("Operation graph contains a self-cycle.")
        if node.operation == "AGGREGATE":
            op = str(node.params.get("aggregation", "")).upper()
            if op not in AGGREGATION_REGISTRY:
                raise AnalyticalOperationError(f"Unsupported aggregation in node {node.node_id!r}: {op}.")
            target = node.params.get("target_column")
            if target is not None and target not in columns:
                raise AnalyticalOperationError(f"Unknown target column in node {node.node_id!r}: {target!r}.")
            groups = node.params.get("grouping_columns", [])
            if not isinstance(groups, (list, tuple)) or any(col not in columns for col in groups):
                raise AnalyticalOperationError(f"Invalid grouping columns in node {node.node_id!r}.")
        elif node.operation == "RANK":
            if len(node.depends_on) != 1:
                raise AnalyticalOperationError("RANK must depend on exactly one prior operation.")
            if str(node.params.get("direction", "DESC")).upper() not in {"ASC", "DESC"}:
                raise AnalyticalOperationError("RANK direction must be ASC or DESC.")
    if any(output not in by_id for output in graph.requested_outputs):
        raise AnalyticalOperationError("Every requested output must reference a graph node.")
    # Deterministic topological sort, preserving authored order among available nodes.
    pending = {node.node_id: set(node.depends_on) for node in graph.nodes}
    order: list[str] = []
    while pending:
        ready = [node.node_id for node in graph.nodes if node.node_id in pending and not pending[node.node_id]]
        if not ready:
            raise AnalyticalOperationError("Operation graph contains a dependency cycle.")
        for node_id in ready:
            order.append(node_id)
            del pending[node_id]
            for deps in pending.values():
                deps.discard(node_id)
    return tuple(order)


def execute_operation_graph(frame: pd.DataFrame, graph: OperationGraph) -> dict[str, Any]:
    """Execute a validated graph. The graph itself must already be approved by the caller."""
    order = validate_operation_graph(graph, [str(c) for c in frame.columns])
    by_id = {node.node_id: node for node in graph.nodes}
    results: dict[str, Any] = {}
    for node_id in order:
        node = by_id[node_id]
        if node.operation == "AGGREGATE":
            params = dict(node.params)
            results[node_id] = aggregate_frame(
                frame, operation=params.pop("aggregation"),
                target_column=params.pop("target_column", None),
                grouping_columns=params.pop("grouping_columns", ()),
                percentile=params.pop("percentile", None), filters=params.pop("filters", ()),
            )
        else:
            source = results[node.depends_on[0]]
            records = source.get("records") if isinstance(source, dict) else None
            if records is None:
                raise AnalyticalOperationError("RANK input must be a grouped aggregate result.")
            direction = str(node.params.get("direction", "DESC")).upper()
            records = list(records)
            # Keep undefined values last in either direction; reverse=True on a tuple
            # would incorrectly move null results to the front for descending ranks.
            if direction == "DESC":
                records.sort(key=lambda row: (row["value"] is None, -(row["value"]) if isinstance(row["value"], (int, float)) else 0))
            else:
                records.sort(key=lambda row: (row["value"] is None, row["value"] if isinstance(row["value"], (int, float)) else 0))
            results[node_id] = {"operation": "RANK", "direction": direction,
                                "records": [{**record, "rank": i} for i, record in enumerate(records, 1)],
                                "source_node": node.depends_on[0]}
    has_zero_match = any(
        isinstance(res, dict) and (
            res.get("population_row_count") == 0
            or bool((res.get("verification") or {}).get("zero_matching_rows"))
        )
        for res in results.values()
    )
    all_analytical_verified = all(
        (not isinstance(res, dict)) or (res.get("verification") or {}).get("status") == "verified"
        for res in results.values() if isinstance(res, dict) and "verification" in res
    ) and not has_zero_match

    graph_verification = {
        "status": "verified" if all_analytical_verified else ("unverified_zero_matches" if has_zero_match else "unverified"),
        "graph_validated": True,
        "analytical_verified": all_analytical_verified,
        "zero_matching_rows": has_zero_match,
        **({"reason": "One or more operation nodes matched zero rows."} if has_zero_match else {}),
    }
    return {"graph_hash": graph.graph_hash(), "execution_order": list(order),
            "outputs": {node_id: results[node_id] for node_id in graph.requested_outputs},
            "verification": graph_verification}


def build_supported_aggregate_then_rank_graph(question: str, contract: Any, columns: Sequence[str]) -> OperationGraph | None:
    """Compile one narrowly supported compound pattern into a typed graph.

    Supported shape: ``Calculate total revenue by region and rank regions by revenue``.
    Both clauses must refer to the same resolved measure and single resolved grouping column.
    Anything else returns None so the normal unresolved-compound gate continues to refuse it.
    """
    import re
    from packages.analytics_core.src.intelligence.canonical_question_contract import resolve_phrase_to_column

    text = " ".join(str(question or "").split()).strip()
    split = re.search(r"\s+and\s+(?:then\s+)?(rank|order|sort)\s+(.+)$", text, re.I)
    if not split:
        return None
    first_clause = text[:split.start()].strip(" ,")
    rank_clause = split.group(2).strip().rstrip("?.!")
    groups = tuple(getattr(contract, "grouping_columns", ()) or ())
    target = getattr(contract, "target_column", None)
    aggregation = str(getattr(contract, "requested_aggregation", None) or "").upper()
    if len(groups) != 1 or not target or target not in columns or groups[0] not in columns:
        return None
    if aggregation not in AGGREGATION_REGISTRY or aggregation in {"PERCENTILE", "RATE"}:
        return None
    if not re.search(r"\bby\b", first_clause, re.I):
        return None
    # The ranked entity must resolve to the exact group dimension. A second measure, a
    # different dimension, or an ambiguous phrase cannot be merged into this graph.
    rank_match = re.match(r"(?:the\s+)?(.+?)(?:\s+by\s+(.+))?$", rank_clause, re.I)
    if not rank_match:
        return None
    entity_phrase = rank_match.group(1).strip()
    metric_phrase = (rank_match.group(2) or "").strip()
    entity_phrase = re.sub(r"\b(?:highest|lowest|ascending|descending|first|last)\b", "", entity_phrase, flags=re.I).strip()
    entity_col = resolve_phrase_to_column(entity_phrase, columns)
    if entity_col != groups[0]:
        return None
    if metric_phrase:
        metric_col = resolve_phrase_to_column(re.sub(r"\b(?:total|sum|average|avg|mean|median)\b", "", metric_phrase, flags=re.I).strip(), columns)
        if metric_col != target:
            return None
    direction = "ASC" if re.search(r"\b(?:lowest|ascending|least|smallest)\b", rank_clause, re.I) else "DESC"
    params: dict[str, Any] = {
        "aggregation": aggregation,
        "target_column": target,
        "grouping_columns": list(groups),
        "filters": [dict(item) for item in (getattr(contract, "filter_conditions", ()) or ())],
    }
    if aggregation == "PERCENTILE":
        percentile = (getattr(contract, "resolution_evidence", {}) or {}).get("percentile")
        if percentile is None:
            return None
        params["percentile"] = percentile
    return OperationGraph(
        version=1,
        nodes=(
            OperationNode("aggregate", "AGGREGATE", params),
            OperationNode("rank", "RANK", {"direction": direction}, ("aggregate",)),
        ),
        requested_outputs=("aggregate", "rank"),
    )
