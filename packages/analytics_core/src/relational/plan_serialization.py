"""JSON-safe persistence for the typed relational-plan contract."""
from typing import Any, Dict

from .compiler import JoinHop, RelationalPlan
from .expressions import (
    AggExpr, And, ColumnRef, Comparison, DistinctCount, DivExpr, IsNotNull,
    IsNull, Literal, Or,
)


def expr_to_dict(expr: Any) -> Dict[str, Any]:
    if isinstance(expr, ColumnRef):
        return {"type": "ColumnRef", "column": expr.column, "qualifier": expr.qualifier}
    if isinstance(expr, Literal):
        return {"type": "Literal", "value": expr.value}
    if isinstance(expr, AggExpr):
        return {"type": "AggExpr", "fn": expr.fn, "column": expr_to_dict(expr.column)}
    if isinstance(expr, DistinctCount):
        return {"type": "DistinctCount", "column": expr_to_dict(expr.column)}
    if isinstance(expr, Comparison):
        return {"type": "Comparison", "left": expr_to_dict(expr.left), "op": expr.op, "right": expr_to_dict(expr.right)}
    if isinstance(expr, (And, Or)):
        return {"type": type(expr).__name__, "operands": [expr_to_dict(item) for item in expr.operands]}
    if isinstance(expr, (IsNull, IsNotNull)):
        return {"type": type(expr).__name__, "expr": expr_to_dict(expr.expr)}
    if isinstance(expr, DivExpr):
        return {"type": "DivExpr", "numerator": expr_to_dict(expr.numerator), "denominator": expr_to_dict(expr.denominator), "zero_denominator": expr.zero_denominator}
    raise TypeError(f"Unsupported PlanExpr type: {type(expr).__name__}")


def expr_from_dict(payload: Dict[str, Any]) -> Any:
    typ = payload["type"]
    if typ == "ColumnRef": return ColumnRef(payload["column"], payload.get("qualifier"))
    if typ == "Literal": return Literal(payload.get("value"))
    if typ == "AggExpr": return AggExpr(payload["fn"], expr_from_dict(payload["column"]))
    if typ == "DistinctCount": return DistinctCount(expr_from_dict(payload["column"]))
    if typ == "Comparison": return Comparison(expr_from_dict(payload["left"]), payload["op"], expr_from_dict(payload["right"]))
    if typ == "And": return And([expr_from_dict(item) for item in payload["operands"]])
    if typ == "Or": return Or([expr_from_dict(item) for item in payload["operands"]])
    if typ == "IsNull": return IsNull(expr_from_dict(payload["expr"]))
    if typ == "IsNotNull": return IsNotNull(expr_from_dict(payload["expr"]))
    if typ == "DivExpr": return DivExpr(expr_from_dict(payload["numerator"]), expr_from_dict(payload["denominator"]), payload.get("zero_denominator"))
    raise ValueError(f"Unsupported serialized PlanExpr type: {typ!r}")


def plan_to_dict(plan: RelationalPlan) -> Dict[str, Any]:
    return {
        "base_table": plan.base_table,
        "hops": [hop.__dict__.copy() for hop in plan.hops],
        "select_columns": list(plan.select_columns),
        "group_by": list(plan.group_by) if plan.group_by is not None else None,
        "aggregations": {alias: expr_to_dict(expr) for alias, expr in (plan.aggregations or {}).items()},
        "filters": [expr_to_dict(expr) for expr in (plan.filters or [])],
        "order_by": plan.order_by,
        "limit": plan.limit,
    }


def plan_from_dict(payload: Dict[str, Any]) -> RelationalPlan:
    return RelationalPlan(
        base_table=payload["base_table"],
        hops=[JoinHop(**hop) for hop in payload.get("hops", [])],
        select_columns=list(payload.get("select_columns") or []),
        group_by=list(payload["group_by"]) if payload.get("group_by") is not None else None,
        aggregations={alias: expr_from_dict(expr) for alias, expr in (payload.get("aggregations") or {}).items()} or None,
        filters=[expr_from_dict(expr) for expr in (payload.get("filters") or [])] or None,
        order_by=payload.get("order_by"), limit=payload.get("limit"),
    )
