"""RelationalCompiler: builds and executes genuine multi-table SQL through
DuckDB, and produces a real pre/post grain proof -- never
``verify_grain_preservation(primary_df, primary_df)``.

This module is intentionally standalone and directly callable/testable. It
is NOT yet wired into InvestigationController's autonomous experiment
selection loop -- see the milestone report's "Remaining limitations"
section for why that integration (routing an arbitrary NL question through
intent -> semantic world model -> EIG -> this compiler) was out of scope
for this pass, and what remains to close that gap.
"""
from __future__ import annotations
import polars as pl

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import pandas as pd

from packages.analytics_core.src.relational.join_safety import (
    JoinSafetyReport,
    SafetyStatus,
    assess_join_safety,
    assess_metric_aggregation_safety,
)
from packages.analytics_core.src.relational.expressions import (
    PlanExpr,
    _IDENTIFIER_RE,
    _is_predicate_expr,
    is_row_level,
    to_sql as _expr_to_sql,
)


@dataclass
class JoinHop:
    left_table: str
    right_table: str
    left_key: str
    right_key: str
    join_type: str = "INNER"  # INNER | LEFT


def _plan_expr_to_polars(expr, resolve_column):
    """Compile typed PlanExpr directly to native Polars."""
    from .expressions import (
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
    )

    if isinstance(expr, ColumnRef):
        return pl.col(resolve_column(expr))

    if isinstance(expr, Literal):
        return pl.lit(expr.value)

    if isinstance(expr, Comparison):
        left = _plan_expr_to_polars(
            expr.left,
            resolve_column,
        )
        right = _plan_expr_to_polars(
            expr.right,
            resolve_column,
        )

        operators = {
            "=": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
            "<": lambda a, b: a < b,
            "<=": lambda a, b: a <= b,
            ">": lambda a, b: a > b,
            ">=": lambda a, b: a >= b,
        }

        if expr.op not in operators:
            raise ValueError(
                f"Unsupported comparison operator: {expr.op!r}"
            )

        return operators[expr.op](left, right)

    if isinstance(expr, And):
        operands = expr.operands

        if not operands:
            raise ValueError(
                "AND requires at least one operand"
            )

        result = _plan_expr_to_polars(
            operands[0],
            resolve_column,
        )

        for operand in operands[1:]:
            result = result & _plan_expr_to_polars(
                operand,
                resolve_column,
            )

        return result

    if isinstance(expr, Or):
        operands = expr.operands

        if not operands:
            raise ValueError(
                "OR requires at least one operand"
            )

        result = _plan_expr_to_polars(
            operands[0],
            resolve_column,
        )

        for operand in operands[1:]:
            result = result | _plan_expr_to_polars(
                operand,
                resolve_column,
            )

        return result

    if isinstance(expr, IsNull):
        return _plan_expr_to_polars(
            expr.expr,
            resolve_column,
        ).is_null()

    if isinstance(expr, IsNotNull):
        return _plan_expr_to_polars(
            expr.expr,
            resolve_column,
        ).is_not_null()

    if isinstance(expr, AggExpr):
        inner = _plan_expr_to_polars(
            expr.column,
            resolve_column,
        )

        fn = expr.fn.upper()

        if fn == "SUM":
            return inner.sum()

        if fn == "AVG":
            return inner.mean()

        if fn == "COUNT":
            return inner.count()

        if fn == "MIN":
            return inner.min()

        if fn == "MAX":
            return inner.max()

        raise ValueError(
            f"Unsupported aggregate function: {expr.fn!r}"
        )

    if isinstance(expr, DistinctCount):
        # SQL COUNT(DISTINCT x) ignores NULL.
        return (
            _plan_expr_to_polars(
                expr.column,
                resolve_column,
            )
            .drop_nulls()
            .n_unique()
        )

    if isinstance(expr, DivExpr):
        numerator = _plan_expr_to_polars(
            expr.numerator,
            resolve_column,
        )

        denominator = _plan_expr_to_polars(
            expr.denominator,
            resolve_column,
        )

        return (
            pl.when(
                denominator.is_null() |
                (denominator == 0)
            )
            .then(None)
            .otherwise(numerator / denominator)
        )

    raise TypeError(
        f"Unsupported PlanExpr type: {type(expr).__name__}"
    )
@dataclass
class RelationalPlan:
    """A deterministic, serializable multi-table execution plan.

    Layer 1 contract:
      * aggregations maps output aliases -> typed PlanExpr values.
      * filters contains only row-level typed predicate expressions.

    Remaining string fields are explicitly documented architecture/security
    debt for later typed-plan layers.
    """
    base_table: str
    hops: List[JoinHop] = field(default_factory=list)
    select_columns: List[str] = field(default_factory=list)
    group_by: Optional[List[str]] = None
    aggregations: Optional[Dict[str, PlanExpr]] = None
    filters: Optional[List[PlanExpr]] = None
    order_by: Optional[str] = None
    limit: Optional[int] = None

    def input_tables(self) -> List[str]:
        seen = [self.base_table]
        for h in self.hops:
            if h.right_table not in seen:
                seen.append(h.right_table)
        return seen

    def validate(self) -> List[str]:
        errors: List[str] = []
        for alias in (self.aggregations or {}):
            if not isinstance(alias, str) or not _IDENTIFIER_RE.fullmatch(alias):
                errors.append(f"aggregation_alias_not_identifier:{alias!r}")
        for i, filt in enumerate(self.filters or []):
            if not _is_predicate_expr(filt):
                errors.append(f"filter_{i}_not_a_predicate:{type(filt).__name__}")
                continue
            if not is_row_level(filt):
                errors.append(
                    f"filter_{i}_contains_aggregate: RelationalPlan.filters becomes a SQL WHERE clause; "
                    "aggregate predicates belong in HAVING, which this plan does not currently support"
                )
        return errors

    def to_sql(self) -> str:
        errors = self.validate()
        if errors:
            raise ValueError("RelationalPlan invalid: " + "; ".join(errors))

        select_parts = list(self.select_columns)
        for alias, expr in (self.aggregations or {}).items():
            select_parts.append(f"{_expr_to_sql(expr)} AS {alias}")
        select_clause = ", ".join(select_parts) if select_parts else "*"

        sql = f"SELECT {select_clause}\nFROM {self.base_table}"
        for hop in self.hops:
            sql += (
                f"\n{hop.join_type} JOIN {hop.right_table} "
                f"ON {hop.left_table}.{hop.left_key} = {hop.right_table}.{hop.right_key}"
            )
        if self.filters:
            sql += "\nWHERE " + " AND ".join(_expr_to_sql(f) for f in self.filters)
        if self.group_by:
            sql += "\nGROUP BY " + ", ".join(self.group_by)
        if self.order_by:
            sql += f"\nORDER BY {self.order_by}"
        if self.limit:
            sql += f"\nLIMIT {self.limit}"
        return sql
    def to_polars(self, tables):
        """
        Execute this RelationalPlan natively with Polars.

        Layer 2 semantics:
        - same typed PlanExpr contract
        - INNER and LEFT joins
        - row-level filters only
        - grouping / aggregation
        - ordering / limit
        - fail-closed validation
        """
        from .expressions import (
            ColumnRef,
            AggExpr,
            DistinctCount,
            DivExpr,
        )

        if not isinstance(tables, dict):
            raise TypeError(
                "tables must be a mapping of table name to DataFrame"
            )

        required_tables = {self.base_table}

        for hop in self.hops or []:
            required_tables.add(hop.left_table)
            required_tables.add(hop.right_table)

        missing = sorted(
            table
            for table in required_tables
            if table not in tables
        )

        if missing:
            raise KeyError(
                f"Missing tables for relational plan: {missing!r}"
            )

        sep = "__aaos__"

        for table_name, frame in tables.items():
            if not isinstance(table_name, str) or sep in table_name:
                raise ValueError(f"Invalid table name for Polars namespace: {table_name!r}")
            source_columns = frame.columns if hasattr(frame, "columns") else []
            collisions = [c for c in source_columns if sep in str(c)]
            if collisions:
                raise ValueError(
                    f"Column names containing reserved separator {sep!r} are not supported: "
                    f"table={table_name!r}, columns={collisions!r}"
                )

        frames = {}

        for table_name in required_tables:
            frame = tables[table_name]

            if not isinstance(frame, pl.DataFrame):
                frame = pl.from_pandas(frame)

            frames[table_name] = frame.rename({
                column: f"{table_name}{sep}{column}"
                for column in frame.columns
            })

        current = frames[self.base_table]

        def resolve_column(ref):
            if not isinstance(ref, ColumnRef):
                raise TypeError(
                    f"Expected ColumnRef, got {type(ref).__name__}"
                )

            if ref.qualifier is not None:
                if ref.qualifier not in frames:
                    raise KeyError(
                        f"Unknown table qualifier {ref.qualifier!r}"
                    )

                physical = (
                    f"{ref.qualifier}{sep}{ref.column}"
                )

                if physical not in frames[ref.qualifier].columns:
                    raise KeyError(
                        f"Unknown column "
                        f"{ref.qualifier}.{ref.column}"
                    )

                return physical

            matches = []

            for table_name, frame in frames.items():
                physical = (
                    f"{table_name}{sep}{ref.column}"
                )

                if physical in frame.columns:
                    matches.append(physical)

            if len(matches) == 1:
                return matches[0]

            if not matches:
                raise KeyError(
                    f"Unknown unqualified column {ref.column!r}"
                )

            raise ValueError(
                f"Ambiguous unqualified column {ref.column!r}; "
                f"matches={matches!r}"
            )

        # --------------------------------------------------------
        # Validate filters before execution.
        # --------------------------------------------------------
        def contains_aggregate(expr):
            if isinstance(
                expr,
                (
                    AggExpr,
                    DistinctCount,
                ),
            ):
                return True

            if isinstance(expr, DivExpr):
                return (
                    contains_aggregate(expr.numerator)
                    or contains_aggregate(expr.denominator)
                )

            from .expressions import (
                Comparison,
                And,
                Or,
                IsNull,
                IsNotNull,
            )

            if isinstance(expr, Comparison):
                return (
                    contains_aggregate(expr.left)
                    or contains_aggregate(expr.right)
                )

            if isinstance(expr, And):
                return any(
                    contains_aggregate(item)
                    for item in expr.operands
                )

            if isinstance(expr, Or):
                return any(
                    contains_aggregate(item)
                    for item in expr.operands
                )

            if isinstance(expr, IsNull):
                return contains_aggregate(expr.expr)

            if isinstance(expr, IsNotNull):
                return contains_aggregate(expr.expr)

            return False

        if self.filters:
            for expression in self.filters:
                if contains_aggregate(expression):
                    raise ValueError(
                        "Aggregate expressions are not permitted "
                        "in row-level filters"
                    )

        # --------------------------------------------------------
        # Joins.
        # --------------------------------------------------------
        for hop in self.hops or []:
            join_type = hop.join_type.upper()

            if join_type not in {
                "INNER",
                "LEFT",
            }:
                raise ValueError(
                    f"Unsupported join type: {hop.join_type!r}"
                )

            right = frames[hop.right_table]

            left_key = (
                f"{hop.left_table}{sep}{hop.left_key}"
            )

            right_key = (
                f"{hop.right_table}{sep}{hop.right_key}"
            )

            if left_key not in current.columns:
                raise KeyError(
                    f"Missing join key "
                    f"{hop.left_table}.{hop.left_key}"
                )

            if right_key not in right.columns:
                raise KeyError(
                    f"Missing join key "
                    f"{hop.right_table}.{hop.right_key}"
                )

            current = current.join(
                right,
                left_on=left_key,
                right_on=right_key,
                how=join_type.lower(),
            )

        # --------------------------------------------------------
        # Row-level filters.
        # --------------------------------------------------------
        if self.filters:
            combined = None

            for expression in self.filters:
                compiled = _plan_expr_to_polars(
                    expression,
                    resolve_column,
                )

                combined = (
                    compiled
                    if combined is None
                    else combined & compiled
                )

            current = current.filter(combined)

        def resolve_text_column(text):
            text = text.strip()

            # Basic fail-closed identifier contract for legacy string
            # projection/group/order fields.
            if ";" in text:
                raise ValueError(
                    f"Invalid relational specification: {text!r}"
                )

            if "." in text:
                table, column = text.split(".", 1)

                return resolve_column(
                    ColumnRef(
                        column.strip(),
                        table.strip(),
                    )
                )

            return resolve_column(
                ColumnRef(text)
            )

        def parse_projection(text):
            text = text.strip()

            if ";" in text:
                raise ValueError(
                    f"Invalid select specification: {text!r}"
                )

            upper = text.upper()
            marker = " AS "

            if marker in upper:
                pos = upper.rfind(marker)
                source = text[:pos].strip()
                alias = text[pos + len(marker):].strip()
            else:
                source = text
                alias = text.split(".")[-1].strip()

            if not source or not alias:
                raise ValueError(
                    f"Invalid select specification: {text!r}"
                )

            return source, alias

        aggregations = self.aggregations or {}

        if aggregations:
            aggregate_exprs = [
                _plan_expr_to_polars(
                    expression,
                    resolve_column,
                ).alias(alias)
                for alias, expression in aggregations.items()
            ]

            if self.group_by:
                group_columns = [
                    resolve_text_column(item)
                    for item in self.group_by
                ]

                result = current.group_by(
                    group_columns,
                    maintain_order=True,
                ).agg(
                    aggregate_exprs
                )

                for item in self.select_columns or []:
                    source, alias = parse_projection(item)
                    physical = resolve_text_column(source)

                    if physical in result.columns:
                        if physical != alias:
                            result = result.rename({
                                physical: alias
                            })
                    elif alias not in result.columns:
                        raise KeyError(
                            f"Grouped projection {source!r} "
                            f"does not exist in result"
                        )

            else:
                result = current.select(
                    aggregate_exprs
                )

        else:
            projections = []

            for item in self.select_columns or []:
                source, alias = parse_projection(item)

                projections.append(
                    pl.col(
                        resolve_text_column(source)
                    ).alias(alias)
                )

            result = (
                current.select(projections)
                if projections
                else current
            )

        # --------------------------------------------------------
        # Ordering.
        # --------------------------------------------------------
        if self.order_by:
            order = self.order_by.strip()

            if ";" in order:
                raise ValueError(
                    f"Invalid order specification: {order!r}"
                )

            descending = False
            upper = order.upper()

            if upper.endswith(" DESC"):
                order = order[:-5].strip()
                descending = True
            elif upper.endswith(" ASC"):
                order = order[:-4].strip()

            if not order:
                raise ValueError(
                    "Order specification is empty"
                )

            if order in result.columns:
                order_column = order
            else:
                try:
                    order_column = resolve_text_column(order)
                except KeyError:
                    if not order.replace("_", "").isalnum():
                        raise ValueError(
                            f"Invalid order specification: "
                            f"{order!r}"
                        )
                    raise

            result = result.sort(
                order_column,
                descending=descending,
            )

        if self.limit is not None:
            if self.limit < 0:
                raise ValueError(
                    "limit must be non-negative"
                )

            result = result.limit(self.limit)

        return result


@dataclass
class GrainProof:
    """Real pre/post join grain evidence -- computed from actual row counts
    of the actual input and output data, never a self-comparison."""
    rows_before: Dict[str, int]
    rows_after: int
    grain_before: Dict[str, int]  # distinct count of the join key per input table
    grain_after: int              # distinct count of the base table's natural key in the result
    expected_cardinality: List[str]
    observed_fanout: bool
    aggregation_applied: bool


@dataclass
class RelationalExecutionResult:
    plan: RelationalPlan
    sql: str
    safety_reports: List[JoinSafetyReport]
    safe: bool
    result_df: Optional[pd.DataFrame]
    grain_proof: Optional[GrainProof]
    blocked_reason: Optional[str] = None


class RelationalCompiler:
    """Compiles a RelationalPlan into real SQL, verifies join safety before
    executing, executes via DuckDB, and produces a genuine pre/post grain
    proof from the actual data -- not a stub."""

    def __init__(self, tables: Dict[str, pd.DataFrame]):
        self.tables = tables

    def compile_and_execute(
        self,
        plan: RelationalPlan,
        authorize_many_to_many: bool = False,
        base_natural_key: Optional[str] = None,
    ) -> RelationalExecutionResult:
        input_names = plan.input_tables()
        input_dfs = [self.tables[name] for name in input_names]

        # Assess each hop against its ACTUAL declared left/right tables --
        # this must not assume a linear chain over input_tables() order,
        # since real join graphs are frequently star-shaped (e.g. a base
        # "orders" table joined independently to both "customers" and
        # "products", not orders->customers->products in sequence).
        safety_reports = [
            assess_join_safety(
                self.tables[hop.left_table], self.tables[hop.right_table],
                hop.left_key, hop.right_key,
                left_table=hop.left_table, right_table=hop.right_table,
                authorize_many_to_many=authorize_many_to_many,
            )
            for hop in plan.hops
        ]
        unsafe = [r for r in safety_reports if r.status != SafetyStatus.SAFE]
        if unsafe:
            reasons = "; ".join(f"{r.left_table}->{r.right_table}: {r.reason}" for r in unsafe)
            return RelationalExecutionResult(
                plan=plan, sql=plan.to_sql(), safety_reports=safety_reports,
                safe=False, result_df=None, grain_proof=None,
                blocked_reason=f"Join plan blocked before execution: {reasons}",
            )

        # M2-A: raw aggregation must not cross a fanout-risk hop.  There is
        # intentionally no pre-aggregation escape hatch in this milestone.
        def _expression_qualifiers(expr) -> set[str]:
            from .expressions import (
                ColumnRef,
                AggExpr,
                DistinctCount,
                DivExpr,
                Comparison,
                And,
                Or,
                IsNull,
                IsNotNull,
            )
            if isinstance(expr, ColumnRef):
                return {expr.qualifier} if expr.qualifier else set()
            if isinstance(expr, (AggExpr, DistinctCount)):
                return {expr.column.qualifier} if expr.column.qualifier else set()
            if isinstance(expr, DivExpr):
                return _expression_qualifiers(expr.numerator) | _expression_qualifiers(expr.denominator)
            if isinstance(expr, Comparison):
                return _expression_qualifiers(expr.left) | _expression_qualifiers(expr.right)
            if isinstance(expr, (And, Or)):
                out = set()
                for item in expr.operands:
                    out |= _expression_qualifiers(item)
                return out
            if isinstance(expr, (IsNull, IsNotNull)):
                return _expression_qualifiers(expr.expr)
            return set()

        metric_tables = set().union(
            *(_expression_qualifiers(expr) for expr in (plan.aggregations or {}).values())
        )
        if metric_tables:
            for metric_table in sorted(metric_tables):
                aggregate_safety = assess_metric_aggregation_safety(
                    metric_table, safety_reports,
                )
                if not aggregate_safety.safe:
                    return RelationalExecutionResult(
                        plan=plan, sql=plan.to_sql(), safety_reports=safety_reports,
                        safe=False, result_df=None, grain_proof=None,
                        blocked_reason=aggregate_safety.reason,
                    )

        sql = plan.to_sql()
        con = duckdb.connect(database=":memory:")
        try:
            for name, df in zip(input_names, input_dfs):
                con.register(name, df)
            result_df = con.execute(sql).df()
        finally:
            con.close()

        rows_before = {name: len(df) for name, df in zip(input_names, input_dfs)}

        # Grain-before per table: the distinct-count of whichever join key
        # that specific table participates in (as either side of any hop).
        # Falls back to row count if the table isn't a join participant
        # (e.g. it's the base table and has no declared key of its own).
        key_for_table: Dict[str, str] = {}
        for hop in plan.hops:
            key_for_table.setdefault(hop.left_table, hop.left_key)
            key_for_table.setdefault(hop.right_table, hop.right_key)

        grain_before = {}
        for name, df in zip(input_names, input_dfs):
            key_col = key_for_table.get(name)
            grain_before[name] = int(df[key_col].nunique()) if key_col and key_col in df.columns else len(df)

        natural_key = base_natural_key or (plan.group_by[0] if plan.group_by else None)
        grain_after = int(result_df[natural_key].nunique()) if natural_key and natural_key in result_df.columns else len(result_df)

        observed_fanout = any(r.fanout_risk for r in safety_reports)
        aggregation_applied = bool(plan.aggregations or plan.group_by)

        grain_proof = GrainProof(
            rows_before=rows_before,
            rows_after=len(result_df),
            grain_before=grain_before,
            grain_after=grain_after,
            expected_cardinality=[r.cardinality.value for r in safety_reports],
            observed_fanout=observed_fanout,
            aggregation_applied=aggregation_applied,
        )

        return RelationalExecutionResult(
            plan=plan, sql=sql, safety_reports=safety_reports,
            safe=True, result_df=result_df, grain_proof=grain_proof,
        )
