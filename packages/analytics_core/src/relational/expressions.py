"""Typed expression AST for RelationalPlan.

Layer 1 scope. In this layer, TWO RelationalPlan fields are typed and
free-form-SQL-free:

    aggregations: Dict[output_alias, PlanExpr]
    filters:      List[PlanExpr]   (row-level predicates only)

The following plan fields REMAIN STRINGS and are NOT YET covered by the
 typed-plan invariant. They are accepted as trusted literals today and are
outstanding architecture/security debt of this layer:

    RelationalPlan.base_table
    RelationalPlan.select_columns
    RelationalPlan.group_by
    RelationalPlan.order_by
    JoinHop.left_table / right_table / left_key / right_key

A caller with write access to a RelationalPlan can therefore still place
free-form SQL into these fields. Removing that capability is a follow-up
layer; until then, only callers already inside the trusted compiler surface
may construct plans. External input must never flow directly into these
fields.

The accurate Layer 1 claim is therefore:

    aggregations and filters no longer accept free-form SQL.

NOT:

    no free-form SQL anywhere in the plan.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Optional, Union

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_COMPARISON_OPS = ("=", "!=", "<>", "<", "<=", ">", ">=")
_AGGREGATE_FNS = ("SUM", "AVG", "MEAN", "COUNT", "MIN", "MAX")


def _validate_identifier(value: str, *, field: str) -> None:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(
            f"{field} must be a strict identifier matching "
            f"[A-Za-z_][A-Za-z0-9_]*; got {value!r}"
        )


def _validate_literal_value(value: Any, *, field: str) -> None:
    if value is None or isinstance(value, (bool, int, float, str)):
        return
    raise TypeError(
        f"{field} must be None, bool, int, float, or str; got "
        f"{type(value).__name__}"
    )


@dataclass(frozen=True)
class ColumnRef:
    column: str
    qualifier: Optional[str] = None

    def __post_init__(self) -> None:
        _validate_identifier(self.column, field="ColumnRef.column")
        if self.qualifier is not None:
            _validate_identifier(self.qualifier, field="ColumnRef.qualifier")


@dataclass(frozen=True)
class Literal:
    value: Any

    def __post_init__(self) -> None:
        _validate_literal_value(self.value, field="Literal.value")


@dataclass(frozen=True)
class Comparison:
    left: "PlanExpr"
    op: str
    right: "PlanExpr"

    def __post_init__(self) -> None:
        if self.op not in _COMPARISON_OPS:
            raise ValueError(f"unsupported comparison op: {self.op!r}")
        if not _is_value_expr(self.left):
            raise TypeError(
                "Comparison.left must be a value expression "
                f"(ColumnRef/Literal/AggExpr/DistinctCount/DivExpr); "
                f"got {type(self.left).__name__}"
            )
        if not _is_value_expr(self.right):
            raise TypeError(
                "Comparison.right must be a value expression; "
                f"got {type(self.right).__name__}"
            )


@dataclass(frozen=True)
class And:
    operands: List["PlanExpr"]

    def __post_init__(self) -> None:
        if not self.operands:
            raise ValueError("And requires at least one operand")
        for i, operand in enumerate(self.operands):
            if not _is_predicate_expr(operand):
                raise TypeError(
                    f"And.operands[{i}] must be a predicate "
                    f"(Comparison/And/Or/IsNull/IsNotNull); "
                    f"got {type(operand).__name__}"
                )


@dataclass(frozen=True)
class Or:
    operands: List["PlanExpr"]

    def __post_init__(self) -> None:
        if not self.operands:
            raise ValueError("Or requires at least one operand")
        for i, operand in enumerate(self.operands):
            if not _is_predicate_expr(operand):
                raise TypeError(
                    f"Or.operands[{i}] must be a predicate; "
                    f"got {type(operand).__name__}"
                )


@dataclass(frozen=True)
class IsNull:
    expr: "PlanExpr"

    def __post_init__(self) -> None:
        if not _is_value_expr(self.expr):
            raise TypeError(
                "IsNull.expr must be a value expression; "
                f"got {type(self.expr).__name__}"
            )


@dataclass(frozen=True)
class IsNotNull:
    expr: "PlanExpr"

    def __post_init__(self) -> None:
        if not _is_value_expr(self.expr):
            raise TypeError(
                "IsNotNull.expr must be a value expression; "
                f"got {type(self.expr).__name__}"
            )


@dataclass(frozen=True)
class AggExpr:
    fn: str
    column: ColumnRef

    def __post_init__(self) -> None:
        if not isinstance(self.column, ColumnRef):
            raise TypeError(
                "AggExpr.column must be ColumnRef; "
                f"got {type(self.column).__name__}"
            )
        fn_upper = str(self.fn).upper()
        if fn_upper not in _AGGREGATE_FNS:
            raise ValueError(f"unsupported aggregate: {self.fn!r}")
        object.__setattr__(self, "fn", fn_upper)


@dataclass(frozen=True)
class DistinctCount:
    column: ColumnRef

    def __post_init__(self) -> None:
        if not isinstance(self.column, ColumnRef):
            raise TypeError(
                "DistinctCount.column must be ColumnRef; "
                f"got {type(self.column).__name__}"
            )


@dataclass(frozen=True)
class DivExpr:
    numerator: "PlanExpr"
    denominator: "PlanExpr"
    zero_denominator: Optional[Any] = None

    def __post_init__(self) -> None:
        if not _is_value_expr(self.numerator):
            raise TypeError(
                "DivExpr.numerator must be a value expression; "
                f"got {type(self.numerator).__name__}"
            )
        if not _is_value_expr(self.denominator):
            raise TypeError(
                "DivExpr.denominator must be a value expression; "
                f"got {type(self.denominator).__name__}"
            )
        if self.zero_denominator is not None:
            _validate_literal_value(
                self.zero_denominator, field="DivExpr.zero_denominator"
            )


PlanExpr = Union[
    ColumnRef,
    Literal,
    Comparison,
    And,
    Or,
    IsNull,
    IsNotNull,
    AggExpr,
    DistinctCount,
    DivExpr,
]

_PREDICATE_TYPES = (Comparison, And, Or, IsNull, IsNotNull)
_VALUE_TYPES = (ColumnRef, Literal, AggExpr, DistinctCount, DivExpr)


def _is_predicate_expr(expr: Any) -> bool:
    return isinstance(expr, _PREDICATE_TYPES)


def _is_value_expr(expr: Any) -> bool:
    return isinstance(expr, _VALUE_TYPES)


def is_row_level(expr: PlanExpr) -> bool:
    """Return True iff expr is safe to emit in a SQL WHERE clause.

    Aggregate nodes are not row-level. A DivExpr is row-level only when
    both operands are row-level. This permits future HAVING predicates to
    contain aggregates without weakening the WHERE boundary.
    """
    if isinstance(expr, (ColumnRef, Literal)):
        return True
    if isinstance(expr, (AggExpr, DistinctCount)):
        return False
    if isinstance(expr, Comparison):
        return is_row_level(expr.left) and is_row_level(expr.right)
    if isinstance(expr, (And, Or)):
        return all(is_row_level(op) for op in expr.operands)
    if isinstance(expr, (IsNull, IsNotNull)):
        return is_row_level(expr.expr)
    if isinstance(expr, DivExpr):
        return is_row_level(expr.numerator) and is_row_level(expr.denominator)
    return False


_SQL_AGG_FN = {
    "SUM": "SUM",
    "AVG": "AVG",
    "MEAN": "AVG",
    "COUNT": "COUNT",
    "MIN": "MIN",
    "MAX": "MAX",
}


def _sql_quote(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    raise TypeError(f"unsupported_literal_type:{type(value).__name__}")


def to_sql(expr: PlanExpr) -> str:
    """Compile one PlanExpr to DuckDB-compatible SQL without parsing SQL."""
    if isinstance(expr, ColumnRef):
        return f"{expr.qualifier}.{expr.column}" if expr.qualifier else expr.column
    if isinstance(expr, Literal):
        return _sql_quote(expr.value)
    if isinstance(expr, Comparison):
        return f"({to_sql(expr.left)} {expr.op} {to_sql(expr.right)})"
    if isinstance(expr, And):
        return "(" + " AND ".join(to_sql(o) for o in expr.operands) + ")"
    if isinstance(expr, Or):
        return "(" + " OR ".join(to_sql(o) for o in expr.operands) + ")"
    if isinstance(expr, IsNull):
        return f"({to_sql(expr.expr)} IS NULL)"
    if isinstance(expr, IsNotNull):
        return f"({to_sql(expr.expr)} IS NOT NULL)"
    if isinstance(expr, AggExpr):
        return f"{_SQL_AGG_FN[expr.fn]}({to_sql(expr.column)})"
    if isinstance(expr, DistinctCount):
        return f"COUNT(DISTINCT {to_sql(expr.column)})"
    if isinstance(expr, DivExpr):
        num = to_sql(expr.numerator)
        den = to_sql(expr.denominator)
        if expr.zero_denominator is None:
            return f"({num} / NULLIF({den}, 0))"
        return (
            f"(CASE WHEN {den} = 0 THEN {_sql_quote(expr.zero_denominator)} "
            f"ELSE {num} / {den} END)"
        )
    raise TypeError(f"unknown_plan_expr:{type(expr).__name__}")


def to_polars(expr: PlanExpr, resolve_column=None):
    """Compile one PlanExpr to a native Polars expression.

    ``resolve_column`` is optional for backwards compatibility. When omitted,
    qualified references use the same ``<table>__aaos__<column>`` physical
    namespace employed by the relational plan compiler.
    """
    import polars as pl

    def _resolve(ref: ColumnRef) -> str:
        if resolve_column is not None:
            return resolve_column(ref)
        if ref.qualifier:
            return f"{ref.qualifier}__aaos__{ref.column}"
        return ref.column

    if isinstance(expr, ColumnRef):
        return pl.col(_resolve(expr))
    if isinstance(expr, Literal):
        return pl.lit(expr.value)
    if isinstance(expr, Comparison):
        left, right = to_polars(expr.left, resolve_column), to_polars(expr.right, resolve_column)
        return {
            "=": left == right,
            "!=": left != right,
            "<>": left != right,
            "<": left < right,
            "<=": left <= right,
            ">": left > right,
            ">=": left >= right,
        }[expr.op]
    if isinstance(expr, And):
        result = to_polars(expr.operands[0], resolve_column)
        for operand in expr.operands[1:]:
            result = result & to_polars(operand, resolve_column)
        return result
    if isinstance(expr, Or):
        result = to_polars(expr.operands[0], resolve_column)
        for operand in expr.operands[1:]:
            result = result | to_polars(operand, resolve_column)
        return result
    if isinstance(expr, IsNull):
        return to_polars(expr.expr, resolve_column).is_null()
    if isinstance(expr, IsNotNull):
        return to_polars(expr.expr, resolve_column).is_not_null()
    if isinstance(expr, AggExpr):
        col = pl.col(_resolve(expr.column))
        if expr.fn == "SUM":
            return col.sum()
        if expr.fn in ("AVG", "MEAN"):
            return col.mean()
        if expr.fn == "COUNT":
            return col.count()
        if expr.fn == "MIN":
            return col.min()
        if expr.fn == "MAX":
            return col.max()
        raise ValueError(f"unsupported_aggregate:{expr.fn}")
    if isinstance(expr, DistinctCount):
        return pl.col(_resolve(expr.column)).n_unique()
    if isinstance(expr, DivExpr):
        num = to_polars(expr.numerator, resolve_column)
        den = to_polars(expr.denominator, resolve_column)
        if expr.zero_denominator is None:
            return (
                pl.when(den.is_null()).then(pl.lit(None))
                .when(den != 0).then(num / den)
                .otherwise(pl.lit(None))
            )
        return (
            pl.when(den.is_null()).then(pl.lit(None))
            .when(den != 0).then(num / den)
            .otherwise(pl.lit(expr.zero_denominator))
        )
    raise TypeError(f"unknown_plan_expr:{type(expr).__name__}")
