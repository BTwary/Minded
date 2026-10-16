
"""Typed expression AST for RelationalPlan.

Layer 1 scope
-------------
The aggregations and filters fields are represented by typed PlanExpr nodes,
not arbitrary SQL strings.

The following RelationalPlan fields remain strings and are NOT covered by
the typed-plan invariant:

    RelationalPlan.base_table
    RelationalPlan.select_columns
    RelationalPlan.group_by
    RelationalPlan.order_by
    JoinHop.left_table / right_table / left_key / right_key

These remain trusted-compiler-only fields until a subsequent migration
makes them typed. External input must never flow directly into them.

SQL identifiers emitted by this module are quoted. Identifier validation
remains strict for ColumnRef, while quote_sql_identifier can safely quote
physical column names containing spaces, punctuation, or reserved words.

This module does not claim that every RelationalPlan field is free of
arbitrary SQL. Only aggregations and filters are typed expression trees.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, List, Optional, Union


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_COMPARISON_OPS = ("=", "!=", "<>", "<", "<=", ">", ">=")
_AGGREGATE_FNS = ("SUM", "AVG", "MEAN", "COUNT", "MIN", "MAX")


def _validate_identifier(value: str, *, field: str) -> None:
    """Validate identifiers accepted by the typed ColumnRef AST."""
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(
            f"{field} must be a strict identifier matching "
            f"[A-Za-z_][A-Za-z0-9_]*; got {value!r}"
        )


def quote_sql_identifier(
    value: str, *, field: str = "identifier"
) -> str:
    """Quote one complete DuckDB identifier.

    This helper is suitable for physical column or table names obtained from
    trusted schema metadata. It treats the entire value as one identifier:
    a period in the value is not interpreted as a qualifier separator.

    Double quotes inside the identifier are escaped by doubling them.
    """
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"Invalid SQL {field}: {value!r}")
    return '"' + value.replace('"', '""') + '"'


def _validate_literal_value(value: Any, *, field: str) -> None:
    """Reject values that cannot be represented by the supported literal AST."""
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{field} must not be NaN or infinite")
        return
    raise TypeError(
        f"{field} must be None, bool, int, finite float, or str; got "
        f"{type(value).__name__}"
    )


@dataclass(frozen=True)
class ColumnRef:
    """Reference a validated column, optionally qualified by a table alias."""

    column: str
    qualifier: Optional[str] = None

    def __post_init__(self) -> None:
        _validate_identifier(self.column, field="ColumnRef.column")
        if self.qualifier is not None:
            _validate_identifier(self.qualifier, field="ColumnRef.qualifier")


@dataclass(frozen=True)
class Literal:
    """A scalar literal represented as data, never as raw SQL."""

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
                "(ColumnRef/Literal/AggExpr/DistinctCount/DivExpr); "
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
                    "(Comparison/And/Or/IsNull/IsNotNull); "
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
class CountRows:
    """COUNT(*) includes rows whose metric is null."""


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
                self.zero_denominator,
                field="DivExpr.zero_denominator",
            )


PlanExpr = Union[
    CountRows,
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
_VALUE_TYPES = (ColumnRef, Literal, AggExpr, DistinctCount, DivExpr, CountRows)


def _is_predicate_expr(expr: Any) -> bool:
    return isinstance(expr, _PREDICATE_TYPES)


def _is_value_expr(expr: Any) -> bool:
    return isinstance(expr, _VALUE_TYPES)


def is_row_level(expr: PlanExpr) -> bool:
    """Return True iff expr is safe to emit in a SQL WHERE clause.

    Aggregate nodes are not row-level. DivExpr is row-level only if both
    operands are row-level, preserving the WHERE/HAVING boundary.
    """
    if isinstance(expr, (ColumnRef, Literal)):
        return True
    if isinstance(expr, (AggExpr, DistinctCount, CountRows)):
        return False
    if isinstance(expr, Comparison):
        return is_row_level(expr.left) and is_row_level(expr.right)
    if isinstance(expr, (And, Or)):
        return all(is_row_level(op) for op in expr.operands)
    if isinstance(expr, (IsNull, IsNotNull)):
        return is_row_level(expr.expr)
    if isinstance(expr, DivExpr):
        return (
            is_row_level(expr.numerator)
            and is_row_level(expr.denominator)
        )
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
    """Encode a supported scalar as a SQL literal."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("SQL float literals must be finite")
        return repr(value)
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    raise TypeError(f"unsupported_literal_type:{type(value).__name__}")


def to_sql(expr: PlanExpr) -> str:
    """Compile one typed PlanExpr to DuckDB-compatible SQL."""
    if isinstance(expr, ColumnRef):
        column = quote_sql_identifier(expr.column, field="column")
        if expr.qualifier is not None:
            qualifier = quote_sql_identifier(
                expr.qualifier, field="qualifier"
            )
            return f"{qualifier}.{column}"
        return column

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

    if isinstance(expr, CountRows):
        return "COUNT(*)"

    if isinstance(expr, AggExpr):
        if expr.fn == "COUNT":
            return "COUNT(*)"
        return f"{_SQL_AGG_FN[expr.fn]}({to_sql(expr.column)})"

    if isinstance(expr, DistinctCount):
        return f"COUNT(DISTINCT {to_sql(expr.column)})"

    if isinstance(expr, DivExpr):
        numerator = to_sql(expr.numerator)
        denominator = to_sql(expr.denominator)
        if expr.zero_denominator is None:
            return f"({numerator} / NULLIF({denominator}, 0))"
        return (
            f"(CASE WHEN {denominator} = 0 "
            f"THEN {_sql_quote(expr.zero_denominator)} "
            f"ELSE {numerator} / {denominator} END)"
        )

    raise TypeError(f"unknown_plan_expr:{type(expr).__name__}")


def to_polars(expr: PlanExpr, resolve_column=None):
    """Compile one PlanExpr to a native Polars expression.

    ``resolve_column`` is optional for backwards compatibility. When omitted,
    qualified references use the ``<table>__aaos__<column>`` physical
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
        left = to_polars(expr.left, resolve_column)
        right = to_polars(expr.right, resolve_column)
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

    if isinstance(expr, CountRows):
        return pl.len()

    if isinstance(expr, AggExpr):
        if expr.fn == "COUNT":
            return pl.len()
        col = pl.col(_resolve(expr.column))
        if expr.fn == "SUM":
            return col.sum()
        if expr.fn in ("AVG", "MEAN"):
            return col.mean()
        if expr.fn == "MIN":
            return col.min()
        if expr.fn == "MAX":
            return col.max()
        raise ValueError(f"unsupported_aggregate:{expr.fn}")

    if isinstance(expr, DistinctCount):
        return pl.col(_resolve(expr.column)).n_unique()

    if isinstance(expr, DivExpr):
        numerator = to_polars(expr.numerator, resolve_column)
        denominator = to_polars(expr.denominator, resolve_column)
        if expr.zero_denominator is None:
            return (
                pl.when(denominator.is_null())
                .then(pl.lit(None))
                .when(denominator != 0)
                .then(numerator / denominator)
                .otherwise(pl.lit(None))
            )
        return (
            pl.when(denominator.is_null())
            .then(pl.lit(None))
            .when(denominator != 0)
            .then(numerator / denominator)
            .otherwise(pl.lit(expr.zero_denominator))
        )

    raise TypeError(f"unknown_plan_expr:{type(expr).__name__}")
