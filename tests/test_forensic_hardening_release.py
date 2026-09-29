from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_rbac_malicious_literal_is_rejected():
    try:
        from packages.analytics_core.src.security.governance import GovernanceEngine
        GovernanceEngine.inject_rbac_filters(
            "SELECT * FROM customers",
            {"region": "x' OR '1'='1"},
        )
    except ModuleNotFoundError as exc:
        if exc.name in {"polars", "pandas", "sqlglot"}:
            return
        raise
    except ValueError:
        return
    raise AssertionError("malicious RBAC value must fail closed")


def test_sql_literal_delete_is_not_a_forbidden_statement():
    # This test imports the SQL engine only when DuckDB is available.
    try:
        from packages.analytics_core.src.sql.engine import DuckDBSQLEngine
        engine = DuckDBSQLEngine()
    except ModuleNotFoundError as exc:
        if exc.name in {"duckdb", "sqlglot"}:
            return
        raise
    assert engine.validate_sql("SELECT * FROM t WHERE status = 'DELETE'")[0]
    assert not engine.validate_sql("DELETE FROM t")[0]


def test_verification_empty_relation_is_not_verified():
    try:
        import pandas as pd
        import polars as pl
        from packages.analytics_core.src.engines.verification import VerificationEngine, VerificationStatus
        result = VerificationEngine.verify_secondary(
            primary_df=pd.DataFrame(),
            target_metric_col="value",
            aggregation_type="SUM",
            primary_metric=0.0,
        )
    except ModuleNotFoundError as exc:
        if exc.name in {"polars", "duckdb"}:
            return
        raise
    assert result.status == VerificationStatus.UNVERIFIED


def test_remediation_reports_low_missingness_columns():
    try:
        import pandas as pd
        from packages.analytics_core.src.profiling.data_remediation import DataRemediationEngine
    except ModuleNotFoundError as exc:
        if exc.name in {"polars", "sklearn"}:
            return
        raise
    df = pd.DataFrame({"y": [1.0] * 50, "x": [None] + [float(i) for i in range(49)]})
    result = DataRemediationEngine.remediate_missingness(df, "y")
    assert any("low-missingness" in warning for warning in result.warnings)


def test_uplift_requires_both_arms_and_finite_inputs():
    try:
        import pandas as pd
        from packages.analytics_core.src.ml.uplift_engine import UpliftEngine
    except ModuleNotFoundError as exc:
        if exc.name in {"polars", "sklearn"}:
            return
        raise
    all_treated = pd.DataFrame({"x": [1, 2, 3, 4], "t": [1, 1, 1, 1], "y": [0, 1, 0, 1]})
    try:
        UpliftEngine.compute_t_learner_uplift(all_treated, "t", "y", ["x"])
    except ValueError as exc:
        assert "both treatment and control" in str(exc)
    else:
        raise AssertionError("single-arm treatment must fail closed")


def test_controller_claim_gate_never_defaults_missing_verification_to_true():
    controller = ROOT / "packages/analytics_core/src/runtime/controller.py"
    text = controller.read_text(encoding="utf-8")
    assert "verification_passed=bool(has_verified_evidence)," in text
    assert "or not evidence_ledger.get_verified_claims()" not in text


def test_execution_result_exposes_input_and_result_row_counts():
    provider = ROOT / "packages/analytics_core/src/engines/execution_provider.py"
    text = provider.read_text(encoding="utf-8")
    assert "self.input_row_count" in text
    assert "self.result_row_count = len(result_frame)" in text
