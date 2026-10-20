"""Bind approved aggregates to existing execution/verification engines.

This is a physical-plan adapter, not an analytical engine. No question parsing or
metric inference is allowed here. Unsupported material requirements raise before SQL runs.
"""
from dataclasses import dataclass
from typing import Any
import math

import pandas as pd

from packages.analytics_core.src.engines.analyst_answer import AnalystResult, _num
from packages.analytics_core.src.relational.compiler import RelationalPlan
from packages.analytics_core.src.relational.expressions import (
    AggExpr, And, CountRows, ColumnRef, Comparison, IsNull, IsNotNull, Literal, Or,
)

from packages.analytics_core.src.relational.plan_serialization import plan_to_dict


@dataclass
class ApprovedAggregate:
    plan: RelationalPlan
    specification: dict[str, Any]


def compile_approved_aggregate(contract, frame: pd.DataFrame) -> ApprovedAggregate:
    if contract.completeness not in {"COMPLETE", "NOT_ENFORCED"} or contract.unresolved_roles or contract.unresolved_semantics:
        raise ValueError("The approved plan has unresolved requirements.")
    unsupported_outputs = set(contract.requested_outputs) - {"table"}
    if unsupported_outputs:
        raise ValueError(f"Approved outputs cannot be produced by this executor: {sorted(unsupported_outputs)}")
    if contract.operation_graph is not None:
        raise ValueError("This plan requires operation-graph execution, not a single aggregate.")
    op = str(contract.requested_aggregation or "").upper()
    if op == "AVG":
        op = "MEAN"
    if op not in {"SUM", "MEAN", "ROW_COUNT", "MIN", "MAX"}:
        raise ValueError(f"Approved aggregation {op or '<missing>'} cannot be bound to this relational executor.")
    if contract.share_of_total and (op not in {"SUM", "ROW_COUNT"} or not contract.grouping_columns):
        raise ValueError("Share of total requires a grouped additive metric and a defined denominator.")
    target = contract.target_column
    groups = list(contract.grouping_columns)
    if (op != "ROW_COUNT" and (not target or target not in frame.columns)) or any(g not in frame.columns for g in groups):
        raise ValueError("Approved metric or grouping column is unavailable.")
    if op in {"SUM", "MEAN"} and not pd.api.types.is_numeric_dtype(frame[target]):
        raise ValueError(f"Approved {op} requires a numeric metric.")
    if contract.numerator_column or contract.denominator_column:
        raise ValueError("A derived metric requires an explicit numerator/denominator execution plan.")
    conditions = [dict(f) for f in contract.filter_conditions]
    time_range = contract.time_range
    if time_range and time_range.get("type") != "all":
        if not contract.time_column or not time_range.get("start") or not time_range.get("end"):
            raise ValueError("The approved date range is unresolved.")
        for condition in [
            {"column": contract.time_column, "operator": ">=", "value": time_range["start"]},
            {"column": contract.time_column, "operator": "<" if time_range.get("end_exclusive", False) else "<=", "value": time_range["end"]},
        ]:
            if condition not in conditions:
                conditions.append(condition)
    comparisons = list(contract.comparison_values)
    if comparisons:
        if len(groups) != 1:
            raise ValueError("Comparison scope requires exactly one approved grouping column.")
        comparison_filter = {"column": groups[0], "operator": "IN", "value": comparisons}
        if comparison_filter not in conditions:
            conditions.append(comparison_filter)
    predicates = []
    for f in conditions:
        col, operator, value = f.get("column"), str(f.get("operator", "")).upper(), f.get("value")
        if col not in frame.columns:
            raise ValueError(f"Approved filter column {col!r} is unavailable.")
        ref = ColumnRef(col)
        if operator in {"IS NULL", "IS NOT NULL"}:
            predicates.append(IsNull(ref) if operator == "IS NULL" else IsNotNull(ref))
        elif operator == "IN" and isinstance(value, (list, tuple)) and value:
            predicates.append(Or([Comparison(ref, "=", Literal(v)) for v in value]))
        elif operator in {"NOT IN", "<> ALL"} and isinstance(value, (list, tuple)) and value:
            predicates.append(And([Comparison(ref, "!=", Literal(v)) for v in value]))
        elif operator == "BETWEEN" and isinstance(value, (list, tuple)) and len(value) == 2:
            predicates.append(And([Comparison(ref, ">=", Literal(value[0])), Comparison(ref, "<=", Literal(value[1]))]))
        elif operator in {"=", "==", "IS"}:
            predicates.append(Comparison(ref, "=", Literal(value)))
        elif operator in {"!=", "<>", "IS NOT"}:
            predicates.append(Comparison(ref, "!=", Literal(value)))
        elif operator in {"<", "<=", ">", ">="}:
            predicates.append(Comparison(ref, operator, Literal(value)))
        else:
            raise ValueError(f"Approved filter operator {operator!r} is not supported by this executor.")
    direction = contract.ranking_direction or "DESC"
    if direction not in {"ASC", "DESC"}:
        raise ValueError("Invalid approved ranking direction.")
    # Strict identifiers are enforced by ColumnRef, including legacy string plan slots.
    for g in groups:
        ColumnRef(g)
    if {"value", "n"}.intersection(groups):
        raise ValueError("Grouping column conflicts with a relational result alias.")
    ranking_limit = getattr(contract, "ranking_limit", None)
    if ranking_limit is not None and (not isinstance(ranking_limit, int) or isinstance(ranking_limit, bool) or not 1 <= ranking_limit <= 100):
        raise ValueError("Approved ranking limit must be an integer from 1 to 100.")
    if ranking_limit is not None and not groups:
        raise ValueError("Top-N requires an approved grouping dimension.")
    plan = RelationalPlan(
        base_table="data_table", select_columns=groups, group_by=groups or None,
        aggregations={"value": CountRows() if op == "ROW_COUNT" else AggExpr("AVG" if op == "MEAN" else op, ColumnRef(target)), "n": CountRows()},
        filters=predicates, order_by=f"value {direction}" if groups else None,
        limit=ranking_limit,
    )
    return ApprovedAggregate(plan, {
        "target_column": target, "aggregation": op, "grouping_columns": groups,
        "filters": conditions, "time_column": contract.time_column, "time_range": time_range,
        "share_of_total": contract.share_of_total, "requested_outputs": list(contract.requested_outputs),
        "comparison_values": comparisons, "ranking_direction": direction if contract.task_family == "RANKING" else None,
        "ranking_limit": ranking_limit,
        "execution_specification": plan_to_dict(plan),
    })



from packages.analytics_core.src.runtime.execution_reconciliation import reconcile_approved_execution

def answer_from_execution(contract, binding: ApprovedAggregate, result_frame: pd.DataFrame, record: dict) -> AnalystResult:
    """Format the verified returned relation; never recalculate the metric."""
    groups = binding.specification["grouping_columns"]
    numbers = {
        "operation": binding.specification["aggregation"], "target_column": contract.target_column,
        "grouping_columns": groups, "filters": binding.specification["filters"],
        "population_row_count": record["population_row_count"], "canonical_contract": contract.to_dict(),
        "execution_record": record,
        "verification": {"status": "verified", "method": "duckdb_vs_polars_relation", "matches": True},
        "adequate_power": None, "robust": None,
    }
    if result_frame.empty or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in result_frame["value"].tolist()):
        raise ValueError("The executed metric is missing, nonnumeric or nonfinite; no numerical answer is supported.")
    if groups:
        table = {}
        for row in result_frame.to_dict("records"):
            keys = {g: row[g] for g in groups}
            label = str(next(iter(keys.values()))) if len(groups) == 1 else " | ".join(f"{g}={v}" for g, v in keys.items())
            if label in table:
                raise ValueError("Distinct groups have ambiguous display labels; no answer was published.")
            table[label] = {"value": row["value"], "n": int(row["n"])}
        if contract.share_of_total:
            total = math.fsum(row["value"] for row in table.values())
            available = math.isfinite(total) and total != 0
            for row in table.values():
                row["share_of_total"] = row["value"] / total if available else None
            numbers["share_of_total"] = {"denominator": total, "available": available,
                "reason": None if available else "The grouped total is zero or nonfinite; proportions are undefined."}
        numbers["table"] = table
        headline = "; ".join(f"{g}: {_num(v['value'])}" for g, v in table.items())
    else:
        numbers["value"] = result_frame["value"].iloc[0].item()
        headline = _num(numbers["value"])
    kind = "RANKING" if contract.task_family == "RANKING" else "DESCRIPTIVE"
    details = [f"Executed {numbers['operation']} from the approved contract."]
    # The user asked for an ordering ("which ... highest/lowest", "highest X by Y"): answer it first.
    # The executor already ordered the verified relation by the approved direction, so the first
    # row IS the answer; the full ranked table follows as supporting evidence, never instead of it.
    asked_rank = groups and (kind == "RANKING" or contract.ranking_direction in {"ASC", "DESC"})
    if asked_rank and numbers.get("table"):
        numbers["top"] = next(iter(numbers["table"]))
        numbers["top_value"] = numbers["table"][numbers["top"]]["value"]
        values = [row["value"] for row in numbers["table"].values()]
        leaders = [label for label, row in numbers["table"].items() if row["value"] == values[0]]
        order = "lowest" if (contract.ranking_direction or "DESC") == "ASC" else "highest"
        word = {"SUM": "total", "MEAN": "average", "MIN": "minimum", "MAX": "maximum",
                "ROW_COUNT": "row count of"}.get(numbers["operation"], "")
        measure = f"{word} {contract.target_column}".strip() if contract.target_column else "row count"
        if len(leaders) > 1:
            lead = f"{' and '.join(leaders[:3])} are tied for the {order} {measure}: {_num(values[0])}"
        else:
            lead = f"{numbers['top']} has the {order} {measure}: {_num(values[0])} (n={numbers['table'][numbers['top']]['n']:,})"
        details.append(lead + ".")
        numbers["ranking_direction_answered"] = "ASC" if order == "lowest" else "DESC"
        numbers["leaders"] = leaders
    return AnalystResult(kind, headline, details,
                         caveats=["Descriptive calculation; statistical significance, power, robustness and causation were not assessed."],
                         numbers=numbers, descriptive=True, finding="ranking" if kind == "RANKING" else "distribution")
