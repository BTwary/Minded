import pandas as pd
import duckdb
import polars as pl

from packages.analytics_core.src.relational.compiler import (
    RelationalPlan,
    JoinHop,
)
from packages.analytics_core.src.relational.expressions import (
    ColumnRef,
    AggExpr,
)
from packages.analytics_core.src.engines.verification import (
    VerificationEngine,
    VerificationStatus,
)


def test_relational_plan_native_duckdb_polars_verification():
    tables = {
        "orders": pd.DataFrame(
            {
                "customer_id": ["C1", "C2", "C1"],
                "amount": [100.0, 200.0, 150.0],
            }
        ),
        "customers": pd.DataFrame(
            {
                "customer_id": ["C1", "C2"],
                "segment": ["A", "B"],
            }
        ),
    }

    plan = RelationalPlan(
        base_table="orders",
        hops=[
            JoinHop(
                "orders",
                "customers",
                "customer_id",
                "customer_id",
                join_type="LEFT",
            )
        ],
        select_columns=[
            "customers.segment AS segment",
        ],
        aggregations={
            "total_amount": AggExpr(
                "SUM",
                ColumnRef("amount", "orders"),
            )
        },
        group_by=[
            "customers.segment",
        ],
        order_by="segment ASC",
    )

    # Primary: DuckDB from the canonical plan.
    duck = duckdb.connect(database=":memory:")

    try:
        for name, frame in tables.items():
            duck.register(name, frame)

        primary_sql = plan.to_sql()
        primary_df = duck.execute(primary_sql).fetchdf()
    finally:
        duck.close()

    # The system's multi-row primary-metric convention (see
    # execution_provider._derive_primary_metric) treats the first row of a
    # deterministically-ordered result as authoritative, not a grand total
    # across the grouped rows -- summing "total_amount" across segments here
    # would not be an independently-verifiable single quantity in the same
    # sense the dual-engine check computes. Order is "segment ASC", so this
    # is segment A's total.
    primary_value = float(primary_df["total_amount"].iloc[0])

    # Secondary: native Polars from the SAME typed plan.
    secondary_df = plan.to_polars(tables).to_pandas()

    assert list(primary_df.columns) == list(
        secondary_df.columns
    )

    assert len(primary_df) == len(secondary_df)

    # The verification engine must execute the typed plan itself.
    verification = VerificationEngine.verify_secondary(
        primary_df=tables["orders"],
        target_metric_col="amount",
        aggregation_type="SUM",
        primary_metric=primary_value,
        query_sql=primary_sql,
        group_dimension_col="segment",
        primary_result_df=primary_df,
        primary_result_truncated=False,
        relational_plan=plan,
        relation_tables=tables,
        primary_result_column="total_amount",
    )

    assert verification.status == VerificationStatus.VERIFIED
    assert verification.secondary_method == (
        "relational_plan_polars_native"
    )
    assert verification.failure_reason is None
