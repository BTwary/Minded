"""ExecutionProvider: Abstract interface and physical implementations for analytical compute engines."""
from abc import ABC, abstractmethod
import re
import time
from typing import Any, Dict, Optional, Tuple
import duckdb
import pandas as pd
from packages.shared.src.constants import MAX_QUERY_EXECUTION_TIME_SECONDS, MAX_QUERY_RESULT_ROWS
from packages.analytics_core.src.execution.process_isolation import run_isolated


class QueryExecutionTimeoutError(TimeoutError):
    """Raised when an analytical query exceeds the configured hard wall-clock budget."""


class BaseExecutionProvider(ABC):
    """Abstract execution provider for running analytical queries across engines."""

    @abstractmethod
    def execute_query(
        self,
        df: pd.DataFrame,
        table_name: str,
        query_sql: str,
    ) -> Tuple[float, int, float, pd.DataFrame]:
        """Execute query on dataframe. Returns (primary_metric, row_count, duration_ms, result_df)."""
        pass


def _derive_primary_metric(
    result: pd.DataFrame,
    *,
    aggregation_type: Optional[str],
    primary_result_column: Optional[str],
    sql: str,
) -> Tuple[float, Optional[str]]:
    """Derive an observed primary metric without arbitrary column-order guessing."""
    if result.empty or result.shape[1] == 0:
        raise ValueError("Analytical query returned no values for primary metric extraction.")

    numeric_cols = [c for c in result.columns if pd.api.types.is_numeric_dtype(result[c])]
    agg = str(aggregation_type or "").upper()

    if result.shape[0] == 1:
        if primary_result_column:
            if primary_result_column not in result.columns:
                raise ValueError(f"Declared primary result column {primary_result_column!r} was not returned by the query.")
            if not pd.api.types.is_numeric_dtype(result[primary_result_column]):
                raise ValueError(f"Declared primary result column {primary_result_column!r} is not numeric.")
            value = result[primary_result_column].iloc[0]
            if pd.isna(value):
                raise ValueError(f"Declared primary result column {primary_result_column!r} is NULL.")
            return float(value), primary_result_column
        if len(numeric_cols) == 1:
            return float(result[numeric_cols[0]].iloc[0]), numeric_cols[0]
        raise ValueError("Scalar analytical result has multiple numeric columns but no declared primary result column.")

    if agg == "VARIANCE":
        if not primary_result_column or primary_result_column not in result.columns:
            raise ValueError("Variance analysis requires an explicit numeric result column.")
        if not pd.api.types.is_numeric_dtype(result[primary_result_column]):
            raise ValueError(f"Variance result column {primary_result_column!r} is not numeric.")
        values = pd.to_numeric(result[primary_result_column], errors="coerce").dropna()
        if len(values) == 0:
            raise ValueError("Variance is not computable because the observed numeric result is empty.")
        import numpy as np
        return float(np.var(values.to_numpy(dtype=float), ddof=0)), primary_result_column

    if agg == "CORRELATION":
        # Raw paired observations: primary_value is the actual observed Pearson r.
        cols = []
        if primary_result_column in result.columns and pd.api.types.is_numeric_dtype(result[primary_result_column]):
            others = [c for c in numeric_cols if c != primary_result_column]
            if len(others) == 1:
                cols = [primary_result_column, others[0]]
        elif len(numeric_cols) == 2:
            cols = numeric_cols
        if len(cols) < 2:
            raise ValueError("Correlation requires two numeric result columns.")
        pair = result[cols].dropna()
        if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
            raise ValueError("Correlation is not identifiable from the observed result rows.")
        return float(pair.iloc[:, 0].corr(pair.iloc[:, 1])), "correlation"

    if agg == "MULTIVARIATE_REGRESSION":
        if not primary_result_column or primary_result_column not in result.columns:
            raise ValueError("Multivariate regression requires an explicit numeric primary result column.")
        if not pd.api.types.is_numeric_dtype(result[primary_result_column]):
            raise ValueError("Multivariate regression requires a numeric target column.")
        predictor_cols = [c for c in result.columns if c != primary_result_column and pd.api.types.is_numeric_dtype(result[c])]
        if len(predictor_cols) < 2:
            raise ValueError("Multivariate regression requires at least two numeric predictors.")
        pair = result[[primary_result_column] + predictor_cols].apply(pd.to_numeric, errors="coerce").dropna()
        if len(pair) <= len(predictor_cols) + 2 or pair[primary_result_column].nunique() < 2:
            raise ValueError("Multivariate regression is not identifiable from the observed rows.")
        import numpy as np
        y = pair[primary_result_column].to_numpy(dtype=float)
        x = pair[predictor_cols].to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(x)), x])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        yhat = X @ beta
        sse = float(np.sum((y - yhat) ** 2))
        sst = float(np.sum((y - np.mean(y)) ** 2))
        if sst <= 0.0:
            raise ValueError("Multivariate regression target has zero variance.")
        return float(1.0 - sse / sst), "r_squared"

    if agg == "REGRESSION":
        # Time-bucketed trend requires an explicit result column; the first two
        # numeric columns are not a semantic contract. Downstream trend/
        # significance testing (intelligence/transition.py) regresses the
        # metric against chronological row order (np.arange), not against
        # any specific column's values -- it relies on the query's own
        # ORDER BY to define the x-axis. The grouping key that produces that
        # order (e.g. a DATE_TRUNC('day', ...) period column) is a DATE/
        # DATETIME dtype, not numeric, so requiring a *second numeric*
        # column here was checking for something nothing downstream reads:
        # every chronological forecast query would fail this even though
        # the outcome column alone is sufficient. Require deterministic
        # ordering instead of a numeric predictor column.
        if primary_result_column not in result.columns or not pd.api.types.is_numeric_dtype(result[primary_result_column]):
            raise ValueError("Regression requires an explicit numeric primary result column.")
        if not re.search(r"\bORDER\s+BY\s+", sql, re.IGNORECASE):
            raise ValueError("Regression trend requires the query to be deterministically ordered chronologically (ORDER BY).")
        y = pd.to_numeric(result[primary_result_column], errors="coerce").dropna()
        if len(y) < 3 or y.nunique() < 2:
            raise ValueError("Regression trend is not identifiable from the observed result rows.")
        import numpy as np
        x = np.arange(len(y), dtype=float)
        slope = float(np.polyfit(x, y.to_numpy(dtype=float), 1)[0])
        return slope, "regression_slope"

    if not primary_result_column or primary_result_column not in result.columns:
        raise ValueError("Multi-row analytical result requires an explicit primary result column.")
    if not pd.api.types.is_numeric_dtype(result[primary_result_column]):
        raise ValueError(f"Declared primary result column {primary_result_column!r} is not numeric.")
    # The result must be deterministically ordered before we can treat its
    # first row as authoritative. This does not require the ORDER BY clause
    # to reference primary_result_column by name -- a legitimate case is a
    # time-series query ordered chronologically (ORDER BY period ASC) whose
    # primary metric is a different column (e.g. revenue). What matters is
    # that SOME deterministic ORDER BY is present at all, not which column
    # it names.
    if not re.search(r"\bORDER\s+BY\s+", sql, re.IGNORECASE):
        raise ValueError(f"Multi-row primary metric {primary_result_column!r} is not deterministically ordered by the query.")
    value = result[primary_result_column].iloc[0]
    if pd.isna(value):
        raise ValueError(f"Declared primary result column {primary_result_column!r} has NULL first value.")
    return float(value), primary_result_column


def _execute_duckdb_worker(
    tables: Dict[str, pd.DataFrame],
    sql: str,
    max_result_rows: int,
    aggregation_type: Optional[str],
    primary_result_column: Optional[str],
) -> Dict[str, Any]:
    """Execute one analytical query inside a dedicated killable child process."""
    con = duckdb.connect(":memory:")
    try:
        for name, frame in tables.items():
            con.register(name, frame)
        result = con.execute(sql).fetchdf()
        truncated = len(result) > max_result_rows
        if truncated:
            result = result.head(max_result_rows).copy()
        primary_metric, resolved_primary_column = _derive_primary_metric(
            result,
            aggregation_type=aggregation_type,
            primary_result_column=primary_result_column,
            sql=sql,
        )
        return {"primary_metric": primary_metric, "primary_result_column": resolved_primary_column, "result_df": result, "is_truncated": truncated}
    finally:
        con.close()


class DuckDBExecutionProvider(BaseExecutionProvider):
    """Local DuckDB execution with a real wall-clock interrupt guard.

    A fresh in-memory connection is used per experiment. A watchdog calls
    DuckDB's interrupt hook when the configured timeout is reached, so a slow
    query cannot leave the HTTP investigation request blocked indefinitely.
    """

    def __init__(self, timeout_seconds: Optional[int] = None, max_result_rows: int = MAX_QUERY_RESULT_ROWS):
        self.timeout_seconds = int(timeout_seconds or MAX_QUERY_EXECUTION_TIME_SECONDS)
        self.max_result_rows = int(max_result_rows)

    def execute_query(
        self,
        df: Optional[pd.DataFrame] = None,
        table_name: str = "data_table",
        query_sql: str = "",
        query: Optional[str] = None,
        data_context: Optional[Any] = None,
        primary_table: Optional[str] = None,
        aggregation_type: Optional[str] = None,
        primary_result_column: Optional[str] = None,
    ) -> Any:
        sql = query or query_sql
        tables: Dict[str, pd.DataFrame] = {}
        if data_context is not None and hasattr(data_context, "datasets_map"):
            tables.update(dict(data_context.datasets_map))
            if "data_table" not in tables and primary_table in tables:
                tables["data_table"] = tables[primary_table]
            elif "data_table" not in tables:
                raise ValueError(
                    f"Primary table {primary_table!r} is not present in the execution context; "
                    "refusing arbitrary-table fallback."
                )
        elif df is not None:
            tables[table_name] = df
            if table_name != "data_table":
                tables["data_table"] = df

        started = time.monotonic()
        isolated = run_isolated(
            _execute_duckdb_worker,
            tables,
            sql,
            self.max_result_rows,
            aggregation_type,
            primary_result_column,
            timeout_seconds=self.timeout_seconds,
        )
        duration_ms = max(0.1, (time.monotonic() - started) * 1000.0)
        if isolated.timed_out:
            raise QueryExecutionTimeoutError(isolated.error or "Analytical query exceeded its execution budget.")
        if not isolated.ok:
            raise RuntimeError(isolated.error or "Isolated DuckDB worker failed without a result.")

        payload = isolated.result
        result_df = payload["result_df"]
        truncated = bool(payload.get("is_truncated", False))
        n_rows = len(df) if df is not None else (
            len(next(iter(data_context.datasets_map.values())))
            if data_context and getattr(data_context, "datasets_map", None) else len(result_df)
        )

        class ExecutionResult:
            def __init__(self, val, rows, dur, result_frame, input_rows=None, truncated=False):
                self.primary_value = val
                # Backward-compatible contract: row_count is the input population
                # size used to compute the result. Result cardinality is explicit.
                self.row_count = rows
                self.input_row_count = rows if input_rows is None else input_rows
                self.result_row_count = len(result_frame)
                self.execution_time_ms = dur
                self.result_df = result_frame
                self.status = "TRUNCATED" if truncated else "SUCCESS"
                self.is_truncated = truncated
                self.primary_result_column = None
            def __iter__(self):
                return iter((self.primary_value, self.row_count, self.execution_time_ms, self.result_df))
            def __getitem__(self, idx):
                return (self.primary_value, self.row_count, self.execution_time_ms, self.result_df)[idx]

        execution = ExecutionResult(
            payload["primary_metric"], n_rows, duration_ms, result_df, input_rows=n_rows, truncated=truncated
        )
        execution.primary_result_column = payload.get("primary_result_column")
        return execution

