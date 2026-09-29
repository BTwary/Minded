import os
import time
from pathlib import Path
import pandas as pd
import numpy as np

from packages.analytics_core.src.execution.process_isolation import run_isolated
from packages.analytics_core.src.engines.execution_provider import DuckDBExecutionProvider


def _sleeper(seconds):
    time.sleep(seconds)
    return 'finished'


def _compute_dataframe_worker(df: pd.DataFrame, multiplier: float) -> pd.DataFrame:
    out = df.copy()
    out["doubled"] = out["val"] * multiplier
    return out


def test_process_isolation_kills_timed_out_worker():
    started = time.monotonic()
    result = run_isolated(_sleeper, 2, timeout_seconds=0.1)
    elapsed = time.monotonic() - started
    assert result.timed_out is True
    assert result.ok is False
    assert 'terminated' in (result.error or '')
    assert elapsed < 1.0


def test_process_isolation_roundtrip_dataframe_arrow():
    input_df = pd.DataFrame({"val": [1.5, 2.5, 3.5], "category": ["alpha", "beta", "gamma"]})
    result = run_isolated(_compute_dataframe_worker, input_df, 2.0, timeout_seconds=10)
    assert result.ok is True
    assert result.result is not None
    assert isinstance(result.result, pd.DataFrame)
    assert list(result.result["doubled"]) == [3.0, 5.0, 7.0]
    assert list(result.result["category"]) == ["alpha", "beta", "gamma"]


def test_duckdb_execution_provider_zero_pickle():
    provider = DuckDBExecutionProvider(timeout_seconds=10)
    df = pd.DataFrame({"metric": [10.0, 20.0, 30.0], "group": ["A", "B", "A"]})
    primary_metric, row_count, duration_ms, res_df = provider.execute_query(
        df=df,
        table_name="data_table",
        query_sql="SELECT AVG(metric) as avg_metric FROM data_table",
        primary_result_column="avg_metric",
    )
    assert primary_metric == 20.0
    assert row_count == 3
    assert len(res_df) == 1
    assert isinstance(res_df, pd.DataFrame)


def test_duckdb_sql_engine_isolated_zero_pickle():
    from packages.analytics_core.src.sql.engine import DuckDBSQLEngine
    engine = DuckDBSQLEngine()
    test_df = pd.DataFrame({"id": [1, 2, 3], "score": [85.0, 92.5, 78.0]})
    engine.register_dataframe("test_scores", test_df)
    res = engine.execute_query("SELECT id, score FROM test_scores WHERE score > 80.0 ORDER BY score DESC")
    assert res["status"] == "success"
    assert res["row_count"] == 2
    assert len(res["data"]) == 2
    assert res["data"][0]["score"] == 92.5


