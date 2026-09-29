import pandas as pd
import pytest


def test_rbac_injection_is_rejected():
    from packages.analytics_core.src.security.governance import GovernanceEngine
    with pytest.raises(ValueError, match="Unsafe RBAC value"):
        GovernanceEngine.inject_rbac_filters(
            "SELECT * FROM orders", {"region": "x' OR '1'='1"}
        )


def test_sql_string_literal_named_delete_is_not_a_keyword():
    from packages.analytics_core.src.sql.engine import DuckDBSQLEngine
    engine = DuckDBSQLEngine()
    ok, error = engine.validate_sql("SELECT * FROM t WHERE status = 'DELETE'")
    assert ok is True, error
    ok, error = engine.validate_sql("DELETE FROM t")
    assert ok is False


def test_uplift_all_treated_fails_closed():
    pytest.importorskip("polars")
    from packages.analytics_core.src.ml.uplift_engine import UpliftEngine
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "t": [1, 1, 1], "y": [0, 1, 1]})
    with pytest.raises(ValueError, match="both treatment and control"):
        UpliftEngine.compute_t_learner_uplift(df, "t", "y", ["x"])


def test_uplift_nan_feature_fails_closed():
    pytest.importorskip("polars")
    from packages.analytics_core.src.ml.uplift_engine import UpliftEngine
    df = pd.DataFrame({"x": [1.0, None, 3.0, 4.0], "t": [0, 1, 0, 1], "y": [0, 1, 0, 1]})
    with pytest.raises(ValueError, match="NaN detected"):
        UpliftEngine.compute_t_learner_uplift(df, "t", "y", ["x"])


def test_uplift_requires_binary_outcome():
    pytest.importorskip("polars")
    from packages.analytics_core.src.ml.uplift_engine import UpliftEngine
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "t": [0, 1, 0, 1], "y": [0.1, 0.2, 0.3, 0.4]})
    with pytest.raises(ValueError, match="binary outcomes 0/1"):
        UpliftEngine.compute_t_learner_uplift(df, "t", "y", ["x"])


def test_polars_qualified_aggregation_uses_physical_namespace():
    pl = pytest.importorskip("polars")
    from packages.analytics_core.src.relational.expressions import AggExpr, ColumnRef, DistinctCount, to_polars
    frame = pl.DataFrame({
        "orders__aaos__amount": [10, 20],
        "customers__aaos__amount": [100, 200],
    })
    got = frame.select([
        to_polars(AggExpr("SUM", ColumnRef("amount", "orders"))).alias("orders_sum"),
        to_polars(DistinctCount(ColumnRef("amount", "customers"))).alias("customer_distinct"),
    ])
    assert got["orders_sum"].item() == 30
    assert got["customer_distinct"].item() == 2


def test_polars_divexpr_distinguishes_null_and_zero_denominators():
    pl = pytest.importorskip("polars")
    from packages.analytics_core.src.relational.expressions import ColumnRef, DivExpr, to_polars
    frame = pl.DataFrame({"num": [10.0, 10.0], "den": [None, 0.0]})
    expr = to_polars(DivExpr(ColumnRef("num"), ColumnRef("den"), zero_denominator=0.0))
    out = frame.select(expr.alias("ratio"))
    assert out["ratio"][0] is None
    assert out["ratio"][1] == 0.0


def test_polars_equivalence_reports_absolute_delta():
    pytest.importorskip("polars")
    from packages.analytics_core.src.engines.polars_query_recompiler import result_frames_equivalent
    left = pd.DataFrame({"x": [1000.0]})
    right = pd.DataFrame({"x": [1000.002]})
    ok, delta = result_frames_equivalent(left, right, tolerance=0.001)
    assert ok is False
    assert abs(delta - 0.002) < 1e-9


def test_empty_result_is_unverified_not_verified():
    pytest.importorskip("polars")
    from packages.analytics_core.src.engines.verification import VerificationEngine
    from packages.analytics_core.src.execution.state_machine import VerificationStatus
    df = pd.DataFrame({"region": [], "order_id": []})
    result = VerificationEngine.verify_secondary(
        primary_df=df,
        target_metric_col="order_id",
        aggregation_type="COUNT",
        primary_metric=0.0,
        group_dimension_col="region",
    )
    assert result.status == VerificationStatus.UNVERIFIED
    assert result.secondary_value is None
    assert result.failure_reason == "no_observations_to_verify"
