"""LakehousePushdownCompiler: Transpiles MindEd DuckDB SQL queries to remote lakehouses (Snowflake, BigQuery, ClickHouse)."""
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

try:
    import sqlglot
    from sqlglot import exp
    _HAS_SQLGLOT = True
except ImportError:
    _HAS_SQLGLOT = False


class LakehouseDialect(Enum):
    DUCKDB = "duckdb"
    SNOWFLAKE = "snowflake"
    BIGQUERY = "bigquery"
    CLICKHOUSE = "clickhouse"
    DATABRICKS = "databricks"


@dataclass
class PushdownResult:
    """Result of SQL pushdown transpilation and cost estimation."""
    transpiled_sql: str
    target_dialect: LakehouseDialect
    estimated_bytes_scanned: Optional[int]
    estimated_cost_usd: Optional[float]


class LakehousePushdownCompiler:
    """Transpiles in-memory DuckDB queries to remote enterprise lakehouse SQL dialects."""

    WAREHOUSE_RATES_PER_TB = {
        LakehouseDialect.SNOWFLAKE: 2.0,
        LakehouseDialect.BIGQUERY: 6.25,
        LakehouseDialect.CLICKHOUSE: 0.5,
        LakehouseDialect.DATABRICKS: 3.0,
    }

    @staticmethod
    def transpile_to_target(
        sql: str,
        source: LakehouseDialect = LakehouseDialect.DUCKDB,
        target: LakehouseDialect = LakehouseDialect.SNOWFLAKE,
    ) -> str:
        """Transpiles generated SQL from DuckDB dialect to target warehouse dialect."""
        if source == target:
            return sql

        if _HAS_SQLGLOT:
            try:
                transpiled = sqlglot.transpile(
                    sql,
                    read=source.value,
                    write=target.value,
                    pretty=True,
                )
                return transpiled[0] if transpiled else sql
            except Exception as e:
                raise ValueError(f"Lakehouse transpilation failed from {source.value} to {target.value}: {e}")

        # Fallback if sqlglot unavailable
        return sql

    @staticmethod
    def estimate_cost_from_bytes(
        estimated_bytes: int,
        target: LakehouseDialect,
    ) -> float:
        """Converts estimated bytes scanned into estimated dollar cost for EIG optimizer."""
        rate_per_tb = LakehousePushdownCompiler.WAREHOUSE_RATES_PER_TB.get(target, 2.0)
        estimated_tb = estimated_bytes / 1e12
        return float(estimated_tb * rate_per_tb)

    @staticmethod
    def apply_column_pruning(sql: str, required_columns: List[str]) -> str:
        """Optimizes pushdown queries by pruning unused columns from SELECT *."""
        if not _HAS_SQLGLOT:
            return sql

        try:
            parsed = sqlglot.parse_one(sql)
            for select in parsed.find_all(exp.Select):
                if any(isinstance(col, exp.Star) for col in select.expressions):
                    select.set(
                        "expressions",
                        [exp.column(c) for c in required_columns],
                    )
            return parsed.sql()
        except Exception:
            return sql
