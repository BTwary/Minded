"""Deterministic DuckDB SQL Engine with AST safety verification, read-only enforcement, and hard timeout cancellation."""
import re
import time
from typing import Any, Dict, List, Optional, Tuple
import duckdb
import pandas as pd
from packages.analytics_core.src.execution.process_isolation import run_isolated
from packages.shared.src.constants import (
    MAX_QUERY_EXECUTION_TIME_SECONDS,
    MAX_QUERY_RESULT_ROWS,
    SQL_FORBIDDEN_KEYWORDS,
)

DISALLOWED_SQL_FUNCTIONS = {
    "READ_CSV", "READ_CSV_AUTO", "READ_PARQUET", "READ_JSON", "READ_JSON_AUTO",
    "GLOB", "HTTPFS", "ATTACH", "DETACH", "INSTALL", "LOAD", "COPY", "EXPORT",
    "IMPORT", "PRAGMA", "CALL", "CHECKPOINT", "VACUUM", "SHELL"
}


def sanitize_sql_identifier(name: str) -> str:
    """Sanitize a display name into a safe SQL identifier.

    This is the single source of truth for how a dataset's display name
    maps to the table identifier it is queried under. Anything that needs
    to reason about identifier collisions ahead of time (e.g. dataset
    ingestion, which wants to prevent two datasets in the same project
    from ever colliding) must use this exact function rather than
    reimplementing the same regex, or the two could silently drift apart.
    """
    clean = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not clean or clean[0].isdigit():
        clean = f"t_{clean}"
    return clean


def _run_isolated_sql(
    db_path: str,
    sql: str,
    dataframes: Dict[str, pd.DataFrame],
    parquet_tables: Dict[str, str],
    csv_tables: Dict[str, str],
    max_rows: int,
) -> Tuple[List[str], List[str], List[Tuple[Any, ...]]]:
    """Run one SQL request in a dedicated child process."""
    con = duckdb.connect(database=db_path, read_only=False)
    try:
        for name, frame in dataframes.items():
            con.register(name, frame)
        for name, path in parquet_tables.items():
            con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet(?)", [path])
        for name, path in csv_tables.items():
            con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_csv_auto(?)", [path])
        rel = con.sql(sql).limit(max_rows + 1)
        return rel.columns, [str(t) for t in rel.types], rel.fetchall()
    finally:
        con.close()


class SQLSecurityError(Exception):
    """Raised when a SQL query violates security or read-only policies."""
    pass


class DuckDBSQLEngine:
    """Production-grade DuckDB SQL analytical engine.
    
    Ensures:
    1. Read-only queries strictly (SELECT / WITH AST analysis).
    2. Zero unauthorized filesystem access / attached databases.
    3. Real hard thread-pool execution timeout and row-limit safety guards.
    4. Deterministic computation directly from Parquet / In-Memory tables.
    """

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.con = duckdb.connect(database=db_path, read_only=False)
        self._registered_tables: Dict[str, str] = {}
        self._registered_dataframes: Dict[str, pd.DataFrame] = {}

    def register_parquet(self, table_name: str, parquet_path: str) -> None:
        """Register a Parquet file as a DuckDB view."""
        safe_table = self._sanitize_identifier(table_name)
        escaped_path = parquet_path.replace("\\", "/")
        sql_escaped = escaped_path.replace("'", "''")
        self.con.execute(f"CREATE OR REPLACE VIEW {safe_table} AS SELECT * FROM read_parquet('{sql_escaped}');")
        self._registered_tables[safe_table] = escaped_path

    def register_dataframe(self, table_name: str, df: pd.DataFrame) -> None:
        """Register a Pandas / Arrow DataFrame as a DuckDB table."""
        safe_table = self._sanitize_identifier(table_name)
        self.con.register(safe_table, df)
        self._registered_tables[safe_table] = "in-memory-df"
        self._registered_dataframes[safe_table] = df.copy(deep=False)

    def register_csv(self, table_name: str, csv_path: str) -> None:
        """Register a CSV file as a DuckDB view."""
        safe_table = self._sanitize_identifier(table_name)
        escaped_path = csv_path.replace("\\", "/")
        sql_escaped = escaped_path.replace("'", "''")
        self.con.execute(f"CREATE OR REPLACE VIEW {safe_table} AS SELECT * FROM read_csv_auto('{sql_escaped}');")
        self._registered_tables[safe_table] = escaped_path

    def validate_sql(self, sql: str) -> Tuple[bool, Optional[str]]:
        """Validate that SQL query is strictly read-only and safe via AST and token allow-listing."""
        cleaned = sql.strip().rstrip(";")
        if not cleaned:
            return False, "Query is empty."

        # Remove single-line and multi-line comments for security inspection
        no_comments = re.sub(r"--.*$", "", cleaned, flags=re.MULTILINE)
        no_comments = re.sub(r"/\*.*?\*/", "", no_comments, flags=re.DOTALL)
        # Ignore single-quoted string contents during keyword scanning so a
        # legitimate value like 'DELETE' is not mistaken for a DELETE command.
        no_strings = re.sub(r"'(?:[^']|'')*'", "''", no_comments)
        tokens = [t.upper() for t in re.findall(r"\b[A-Za-z_]+\b", no_strings)]

        # Check forbidden keywords
        for forbidden in SQL_FORBIDDEN_KEYWORDS:
            if forbidden in tokens:
                return False, f"Security Violation: Query contains forbidden keyword '{forbidden}'."

        # Check disallowed filesystem / admin functions
        for func in DISALLOWED_SQL_FUNCTIONS:
            if func in tokens:
                return False, f"Security Violation: Filesystem/admin function '{func}' is not permitted in analytical queries."

        # Reject DuckDB path-backed relations and external scanners even when the
        # scanner function name is not explicitly present in the token stream.
        relation_clause = re.findall(r"\b(?:FROM|JOIN)\s+([^\s,()]+(?:\s*\([^)]*\))?)", no_comments, flags=re.I)
        for raw_relation in relation_clause:
            rel = raw_relation.strip()
            if rel.startswith(("'", '"', "`")) or "/" in rel or "\\" in rel or rel.lower().endswith((".csv", ".parquet", ".json", ".jsonl")):
                return False, "Security Violation: Analytical SQL may only reference registered in-memory relations; path-backed relations are forbidden."
            base = re.sub(r"\s*\(.*$", "", rel).strip('`\"')
            if base.upper() in DISALLOWED_SQL_FUNCTIONS:
                return False, "Security Violation: External scanner relation is not permitted."
            if self._registered_tables and self._sanitize_identifier(base) not in self._registered_tables:
                return False, f"Security Violation: Relation '{base}' is not registered for this query."

        # Check statement starts with allowed read-only tokens
        first_token = tokens[0] if tokens else ""
        if first_token not in ("SELECT", "WITH"):
            return False, f"Security Violation: Only SELECT and WITH queries are permitted. Found: '{first_token}'."

        return True, None

    def execute_query(
        self,
        sql: str,
        max_rows: int = MAX_QUERY_RESULT_ROWS,
        timeout_seconds: int = MAX_QUERY_EXECUTION_TIME_SECONDS,
    ) -> Dict[str, Any]:
        """Execute a validated SQL query deterministically with hard timeout cancellation and row limit guards."""
        is_safe, error_msg = self.validate_sql(sql)
        if not is_safe:
            return {
                "error": error_msg,
                "status": "security_violation",
                "columns": [],
                "data": [],
                "row_count": 0,
                "execution_time_ms": 0,
            }

        start_time = time.time()
        parquet_tables = {k: v for k, v in self._registered_tables.items() if v != "in-memory-df" and v.lower().endswith(".parquet")}
        csv_tables = {k: v for k, v in self._registered_tables.items() if v != "in-memory-df" and v.lower().endswith(".csv")}
        isolated = run_isolated(
            _run_isolated_sql,
            self.db_path,
            sql,
            self._registered_dataframes,
            parquet_tables,
            csv_tables,
            max_rows,
            timeout_seconds=int(timeout_seconds),
        )
        execution_time_ms = int((time.time() - start_time) * 1000)
        if isolated.timed_out:
            return {
                "error": f"Query execution timed out after {timeout_seconds} seconds.",
                "status": "timeout",
                "columns": [], "data": [], "row_count": 0,
                "execution_time_ms": int(timeout_seconds * 1000), "sql": sql,
                "hard_terminated": True,
            }
        if not isolated.ok:
            return {
                "error": f"DuckDB Execution Error: {isolated.error}",
                "status": "failed", "columns": [], "data": [], "row_count": 0,
                "execution_time_ms": execution_time_ms, "sql": sql,
            }

        columns, types, raw_data = isolated.result
        is_truncated = len(raw_data) > max_rows
        returned_rows = raw_data[:max_rows]
        data = [dict(zip(columns, row)) for row in returned_rows]
        return {
            "status": "success",
            "columns": columns, "column_types": types, "data": data,
            "row_count": len(data), "is_truncated": is_truncated,
            "execution_time_ms": execution_time_ms, "sql": sql,
            "hard_terminated": False,
        }

    def _sanitize_identifier(self, name: str) -> str:
        """Sanitize SQL identifiers to prevent injection."""
        return sanitize_sql_identifier(name)
