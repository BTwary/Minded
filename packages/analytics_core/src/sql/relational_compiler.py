"""Deterministic Relational Analytical Compiler translating Typed Analytical IR into verifiable physical execution plans."""
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from packages.schemas.src.analysis import (
    AggregationType,
    FilterAST,
    FilterOperator,
    ObjectiveType,
    TypedAnalyticalIntent,
)
from packages.schemas.src.semantic_graph import SemanticWorldModelSchema
import re as _re
from typing import Any, Dict, List, Optional

_IDENTIFIER_RE = _re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


def _sql_literal(v: Any) -> str:
    """Produce a SQL-safe literal for embedding in a WHERE clause.

    All string values are enclosed in single quotes with internal single-quotes
    doubled (standard SQL escaping). Numeric values are validated to contain
    only digits, signs, and decimal points before bare interpolation.
    Other types fall back to the string representation with quoting.

    This is the single canonical literal-production authority for the
    relational compiler. Never bypass it with bare f-string interpolation.
    """
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, int):
        return str(int(v))
    if isinstance(v, float):
        import math
        if math.isnan(v) or math.isinf(v):
            raise ValueError(f"Non-finite float value cannot be embedded as a SQL literal: {v!r}")
        return repr(v)
    if isinstance(v, str):
        escaped = v.replace("'", "''")
        return f"'{escaped}'"
    return _sql_literal(str(v))



class RelationalOperator(str, Enum):
    SCAN = "SCAN"
    FILTER = "FILTER"
    JOIN = "JOIN"
    AGGREGATE = "AGGREGATE"
    SORT = "SORT"
    PROJECT = "PROJECT"


class PlanNode(BaseModel):
    """A node in the logical relational algebra plan."""
    node_id: str
    operator: RelationalOperator
    description: str
    target_table: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    children: List[str] = Field(default_factory=list)


class PhysicalCompiledPlan(BaseModel):
    """Deterministic physical query execution plan ready for DuckDB / Polars execution."""
    intent_id: str
    sql_query: str
    target_metric_col: str
    aggregation_type: str
    group_by_columns: List[str]
    filter_clauses: List[str]
    unit_of_analysis_keys: List[str]
    logical_nodes: List[PlanNode]
    requires_grain_verification: bool = True


class RelationalCompiler:
    """
    Translates validated TypedAnalyticalIntent into physical Relational Plans
    without relying on LLM SQL generation.
    """

    def compile(
        self,
        intent: TypedAnalyticalIntent,
        world_model: Optional[SemanticWorldModelSchema] = None,
    ) -> PhysicalCompiledPlan:
        logical_nodes: List[PlanNode] = []
        table_name = intent.target_metric.table or intent.unit_of_analysis.table
        if not table_name:
            raise ValueError(
                "Relational compilation requires an explicit source table; "
                "refusing arbitrary data_table fallback."
            )
        if not _IDENTIFIER_RE.fullmatch(str(table_name)):
            raise ValueError(f"Invalid source table identifier: {table_name!r}")

        # 1. SCAN Node
        scan_id = "node_scan_01"
        logical_nodes.append(PlanNode(
            node_id=scan_id,
            operator=RelationalOperator.SCAN,
            description=f"Scan source table '{table_name}'",
            target_table=table_name,
        ))

        # 2. FILTER Nodes (Population Filters)
        filter_clauses: List[str] = []
        last_node_id = scan_id

        for idx, f in enumerate(intent.population):
            f_node_id = f"node_filter_{idx+1}"
            op_sql = self._map_operator(f.operator, f.value)
            clause = f"{f.variable.column} {op_sql}"
            filter_clauses.append(clause)

            logical_nodes.append(PlanNode(
                node_id=f_node_id,
                operator=RelationalOperator.FILTER,
                description=f"Apply population filter: {clause}",
                parameters={"column": f.variable.column, "operator": f.operator.value, "value": f.value},
                children=[last_node_id],
            ))
            last_node_id = f_node_id

        # Temporal Scope Filter
        if intent.temporal_scope:
            ts = intent.temporal_scope
            if ts.start and ts.end:
                t_clause = f"{ts.column.column} >= {_sql_literal(str(ts.start))} AND {ts.column.column} <= {_sql_literal(str(ts.end))}"
                filter_clauses.append(t_clause)
                t_node_id = "node_temporal_filter"
                logical_nodes.append(PlanNode(
                    node_id=t_node_id,
                    operator=RelationalOperator.FILTER,
                    description=f"Apply temporal scope: {t_clause}",
                    parameters={"start": ts.start, "end": ts.end},
                    children=[last_node_id],
                ))
                last_node_id = t_node_id

        # 3. AGGREGATE Node
        metric_col = intent.target_metric.column or intent.target_metric.name
        agg_func = intent.target_metric.aggregation.value.upper()
        if agg_func == "MEAN":
            agg_func = "AVG"

        group_cols = [d.column for d in intent.dimensions]

        agg_node_id = "node_aggregate_01"
        logical_nodes.append(PlanNode(
            node_id=agg_node_id,
            operator=RelationalOperator.AGGREGATE,
            description=f"Aggregate {agg_func}({metric_col}) GROUP BY {group_cols}",
            parameters={"metric": metric_col, "aggregation": agg_func, "group_by": group_cols},
            children=[last_node_id],
        ))

        # 4. Generate Physical SQL
        select_parts = []
        for gc in group_cols:
            select_parts.append(f"{gc}")

        metric_alias = intent.target_metric.name or "metric_val"
        select_parts.append(f"{agg_func}({metric_col}) AS {metric_alias}")

        where_clause = ""
        if filter_clauses:
            where_clause = " WHERE " + " AND ".join(filter_clauses)

        group_by_clause = ""
        order_by_clause = ""
        if group_cols:
            group_by_clause = " GROUP BY " + ", ".join(group_cols)
            order_by_clause = f" ORDER BY {metric_alias} DESC"

        sql_query = f"SELECT {', '.join(select_parts)} FROM {table_name}{where_clause}{group_by_clause}{order_by_clause}"

        return PhysicalCompiledPlan(
            intent_id=intent.intent_id,
            sql_query=sql_query,
            target_metric_col=metric_col,
            aggregation_type=agg_func,
            group_by_columns=group_cols,
            filter_clauses=filter_clauses,
            unit_of_analysis_keys=intent.unit_of_analysis.keys,
            logical_nodes=logical_nodes,
            requires_grain_verification=True,
        )

    def _map_operator(self, op: FilterOperator, val: Any) -> str:
        """Translate a FilterOperator and its value to a SQL predicate fragment.

        All value embedding goes through _sql_literal() — the single canonical
        literal-production authority for this compiler. Never add bare
        f-string literal interpolation here.
        """
        if op == FilterOperator.EQ:
            return f"= {_sql_literal(val)}"
        if op == FilterOperator.NE:
            return f"!= {_sql_literal(val)}"
        if op == FilterOperator.GT:
            return f"> {_sql_literal(val)}"
        if op == FilterOperator.GTE:
            return f">= {_sql_literal(val)}"
        if op == FilterOperator.LT:
            return f"< {_sql_literal(val)}"
        if op == FilterOperator.LTE:
            return f"<= {_sql_literal(val)}"
        if op == FilterOperator.IN:
            if isinstance(val, list):
                items = ", ".join(_sql_literal(v) for v in val)
                return f"IN ({items})"
            return f"= {_sql_literal(val)}"
        if op == FilterOperator.NOT_IN:
            if isinstance(val, list):
                items = ", ".join(_sql_literal(v) for v in val)
                return f"NOT IN ({items})"
            return f"!= {_sql_literal(val)}"
        if op == FilterOperator.IS_NULL:
            return "IS NULL"
        if op == FilterOperator.IS_NOT_NULL:
            return "IS NOT NULL"
        return f"= {_sql_literal(val)}"
