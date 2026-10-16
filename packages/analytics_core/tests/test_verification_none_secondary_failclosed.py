"""Regression tests: a fail-closed independent verification must never crash the
controller and must never be reported as VERIFIED or carry a fabricated value.

Root cause (forensic): ``VerificationEngine.verify_secondary`` correctly returns
``secondary_value=None, observed_delta_pct=None, status=FAILED`` when the
independent engine yields no scalar. The controller then passed that ``None``
delta into ``EpistemicCalibrationEngine.calibrate_vectors``, which executed
``verification_delta_pct / 100.0`` -> ``TypeError: unsupported operand type(s)
for /: 'NoneType' and 'float'``. The step was rolled back and the experiment
stayed registered as executed with no failure record.
"""
import hashlib
import os
import uuid

import numpy as np
import pandas as pd
import pytest

from packages.analytics_core.src.engines import verification as V
from packages.analytics_core.src.engines.verification import VerificationEngine
from packages.analytics_core.src.execution.state_machine import VerificationStatus
from packages.analytics_core.src.intelligence.epistemic_calibration import EpistemicCalibrationEngine

PRIMARY_METRIC = 324728.2826551341  # value from the recorded failing run

DF = pd.DataFrame({"category": list("AABBCC"), "revenue": [1.0, 2, 3, 4, 5, 6]})
PRIM = pd.DataFrame({"category": ["C", "B", "A"], "total_metric": [11.0, 7, 3]})
SQL_OK = "SELECT category, SUM(revenue) AS total_metric FROM data_table GROUP BY category ORDER BY total_metric DESC"
SQL_NO_ORDER = "SELECT category, SUM(revenue) AS total_metric FROM data_table GROUP BY category"
SQL_UNSUPPORTED = (
    "SELECT category, SUM(revenue) AS total_metric FROM data_table "
    "GROUP BY category USING SAMPLE 50% ORDER BY total_metric DESC"
)


def _assert_fail_closed(result, *, primary=None):
    """Shared invariant for every 'no independent scalar' outcome."""
    assert result.status != VerificationStatus.VERIFIED
    assert result.status in (VerificationStatus.FAILED, VerificationStatus.UNVERIFIED)
    assert result.secondary_value is None
    assert result.observed_delta_pct is None
    assert result.failure_reason  # actual cause is named
    assert result.failure_reason != "unexpected_exception_in_verification_engine", (
        f"verifier swallowed an unexpected exception: {result.exception_type}: {result.exception_message}"
    )
    if primary is not None:
        # nothing fabricated from the primary metric or a sentinel
        assert result.secondary_value != primary
        assert result.secondary_value != 0.0


# --------------------------------------------------------------------------
# A/B. Root cause: calibrate_vectors with the exact bad state (delta None)
# --------------------------------------------------------------------------
class TestCalibrateVectorsNoneDelta:
    def _vec(self, delta):
        return EpistemicCalibrationEngine.calibrate_vectors(
            df=PRIM,
            posterior_probability=0.9,
            verification_delta_pct=delta,
            variance_explained_pct=40.0,
            multiverse_robustness_pct=0.0,
            causal_identifiable=False,
        )

    def test_none_delta_does_not_raise_and_is_maximal_uncertainty(self):
        vec = self._vec(None)  # raised TypeError before the fix
        assert vec.model_uncertainty == 1.0
        assert vec.overall_epistemic_grade != "A_CALIBRATED"

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_delta_is_maximal_uncertainty(self, bad):
        assert self._vec(bad).model_uncertainty == 1.0

    def test_numeric_delta_unchanged(self):
        assert self._vec(0.0).model_uncertainty == 0.0
        assert self._vec(2.0).model_uncertainty == pytest.approx(0.02)
        assert self._vec(250.0).model_uncertainty == 1.0  # clipped, as before

    def test_none_delta_never_produces_confident_evidence(self):
        vec = self._vec(None)
        confidence = float(np.clip(vec.evidence_strength * (1.0 - vec.model_uncertainty), 0.0, 1.0))
        assert confidence == 0.0


# --------------------------------------------------------------------------
# C. Successful numeric verification still VERIFIED
# --------------------------------------------------------------------------
class TestSuccessfulVerificationUnchanged:
    def test_relation_authoritative_path_verified(self):
        r = VerificationEngine.verify_secondary(
            primary_df=DF, target_metric_col="revenue", aggregation_type="SUM",
            primary_metric=11.0, query_sql=SQL_OK, group_dimension_col="category",
            primary_result_df=PRIM, primary_result_column="total_metric",
        )
        assert r.status == VerificationStatus.VERIFIED
        assert r.secondary_value == pytest.approx(11.0)
        assert r.observed_delta_pct == 0.0

    def test_legacy_scalar_path_verified(self):
        r = VerificationEngine.verify_secondary(
            primary_df=DF, target_metric_col="revenue", aggregation_type="SUM", primary_metric=21.0,
        )
        assert r.status == VerificationStatus.VERIFIED
        assert r.secondary_value == pytest.approx(21.0)

    def test_numeric_disagreement_fails_with_real_numbers(self):
        r = VerificationEngine.verify_secondary(
            primary_df=DF, target_metric_col="revenue", aggregation_type="SUM", primary_metric=999.0,
        )
        assert r.status == VerificationStatus.FAILED
        assert r.secondary_value == pytest.approx(21.0)
        assert r.observed_delta_pct is not None
        assert r.failure_reason == "numerical_disagreement_exceeds_tolerance"


# --------------------------------------------------------------------------
# D/E/F/G. Verifier fail-closed paths: numeric primary, no secondary scalar
# --------------------------------------------------------------------------
class TestVerifierFailsClosedWithoutSecondaryScalar:
    def test_secondary_engine_cannot_run_sql(self):
        r = VerificationEngine.verify_secondary(
            primary_df=DF, target_metric_col="revenue", aggregation_type="SUM",
            primary_metric=PRIMARY_METRIC, query_sql=SQL_UNSUPPORTED, group_dimension_col="category",
            primary_result_df=PRIM, primary_result_column="total_metric",
        )
        _assert_fail_closed(r, primary=PRIMARY_METRIC)
        assert r.failure_reason == "no_independent_result_relation_available"
        assert r.exception_type  # original exception is preserved, not hidden

    def test_legacy_path_no_computation_for_aggregation_type(self):
        r = VerificationEngine.verify_secondary(
            primary_df=DF, target_metric_col="revenue", aggregation_type="NOT_A_REAL_AGG",
            primary_metric=PRIMARY_METRIC, query_sql=None, group_dimension_col="category",
        )
        _assert_fail_closed(r, primary=PRIMARY_METRIC)
        assert r.failure_reason == "no_computation_path_for_aggregation_type"

    def test_empty_primary_frame_is_unverified(self):
        r = VerificationEngine.verify_secondary(
            primary_df=DF.iloc[0:0], target_metric_col="revenue", aggregation_type="SUM",
            primary_metric=PRIMARY_METRIC,
        )
        _assert_fail_closed(r, primary=PRIMARY_METRIC)

    def test_relation_disagreement_without_scalar_does_not_fabricate_zero(self):
        # Secondary relation exists but disagrees, and no scalar can be reduced
        # (no ORDER BY) -> secondary_value must stay None, not 0.0.
        wrong = PRIM.assign(total_metric=[99.0, 98, 97])
        r = VerificationEngine.verify_secondary(
            primary_df=DF, target_metric_col="revenue", aggregation_type="SUM",
            primary_metric=99.0, query_sql=SQL_NO_ORDER, group_dimension_col="category",
            primary_result_df=wrong,
        )
        assert r.status == VerificationStatus.FAILED
        assert r.failure_reason == "result_relation_disagreement"
        assert r.secondary_value is None  # was fabricated 0.0 before the fix
        assert r.secondary_value != 0.0
        assert r.observed_delta_pct is not None  # a real measured disagreement

    def test_truncated_primary_never_verified(self):
        r = VerificationEngine.verify_secondary(
            primary_df=DF, target_metric_col="revenue", aggregation_type="SUM",
            primary_metric=11.0, query_sql=SQL_OK, group_dimension_col="category",
            primary_result_df=PRIM, primary_result_column="total_metric",
            primary_result_truncated=True,
        )
        assert r.status == VerificationStatus.FAILED
        assert r.status != VerificationStatus.VERIFIED
        assert r.failure_reason == "primary_result_truncated_cannot_verify_full_relation"


class _Plan:
    """Minimal RelationalPlan stand-in exposing only ``to_polars``."""

    def __init__(self, behavior):
        self._behavior = behavior

    def to_polars(self, tables):
        return self._behavior(tables)


def _plan_verify(plan, sql=SQL_OK, primary_result_df=PRIM):
    return VerificationEngine.verify_secondary(
        primary_df=DF, target_metric_col="revenue", aggregation_type="SUM",
        primary_metric=PRIMARY_METRIC, query_sql=sql, group_dimension_col="category",
        primary_result_df=primary_result_df, primary_result_column="total_metric",
        relational_plan=plan, relation_tables={"t": DF},
    )


class TestRelationalPlanFailClosed:
    def test_genuine_secondary_engine_failure(self):
        def boom(_):
            raise RuntimeError("polars engine exploded")
        r = _plan_verify(_Plan(boom))
        _assert_fail_closed(r, primary=PRIMARY_METRIC)
        assert r.failure_reason == "relational_plan_secondary_execution_failed"
        assert r.exception_type == "RuntimeError" and "exploded" in r.exception_message

    def test_none_result_from_secondary_engine(self):
        r = _plan_verify(_Plan(lambda _: None))
        _assert_fail_closed(r, primary=PRIMARY_METRIC)
        assert r.failure_reason == "relational_plan_returned_no_relation"
        assert r.exception_type == "NoneResult"

    def test_scalar_reduction_failure(self):
        import polars as pl
        # Relation equals primary, but reduction fails (no ORDER BY on multi-row result).
        r = _plan_verify(_Plan(lambda _: pl.from_pandas(PRIM)), sql=SQL_NO_ORDER)
        _assert_fail_closed(r, primary=PRIMARY_METRIC)
        assert r.failure_reason == "secondary_relation_scalar_reduction_failed"
        assert r.exception_type == "ValueError"


# --------------------------------------------------------------------------
# Controller level (real InvestigationController)
# --------------------------------------------------------------------------
def _synthetic_sales(n=900, seed=7):
    rng = np.random.RandomState(seed)
    cats = rng.choice(["Electronics", "Furniture", "Clothing", "Books"], size=n, p=[0.4, 0.3, 0.2, 0.1])
    base = {"Electronics": 900.0, "Furniture": 600.0, "Clothing": 200.0, "Books": 40.0}
    rev = np.array([max(1.0, rng.normal(base[c], base[c] * 0.2)) for c in cats])
    return pd.DataFrame({
        "order_id": [f"O{i:06d}" for i in range(n)],
        "category": cats,
        "region": rng.choice(["East", "West"], size=n),
        "revenue": np.round(rev, 2),
    })


QUESTION = "Why does Electronics generate so much more revenue than other product categories?"


def _run_controller(monkeypatch, verify_impl):
    from apps.api.src.core.database import SessionLocal
    from apps.api.src.models.entities import Evidence, Investigation
    from packages.analytics_core.src.engines.dataset_provider import BaseDatasetProvider, InvestigationDataContext
    from packages.analytics_core.src.runtime import state as state_mod
    from packages.analytics_core.src.runtime.controller import InvestigationController

    df = _synthetic_sales()

    class Provider(BaseDatasetProvider):
        def acquire_context(self, project_id, dataset_ids=None):
            return InvestigationDataContext(
                project_id=project_id, datasets_map={"sales": df},
                dataset_fingerprints={"sales": hashlib.sha256(df.to_json().encode()).hexdigest()},
                requested_dataset_ids=dataset_ids,
            )

    calls = []

    def patched(*args, **kwargs):
        calls.append(kwargs.get("primary_metric"))
        return verify_impl(**kwargs)

    monkeypatch.setattr(V.VerificationEngine, "verify_secondary", staticmethod(patched))

    managers = []
    sm_cls = next(c for c in vars(state_mod).values() if isinstance(c, type) and hasattr(c, "record_experiment_failed"))
    orig_init = sm_cls.__init__

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        managers.append(self)

    monkeypatch.setattr(sm_cls, "__init__", init)

    db = SessionLocal()
    iid = f"INV-NONE-{uuid.uuid4().hex[:8]}"
    db.add(Investigation(id=iid, project_id=f"p-{uuid.uuid4().hex[:6]}", question=QUESTION, status="PLANNED"))
    db.commit()
    db.close()

    InvestigationController(session_factory=SessionLocal, dataset_provider=Provider()).execute_investigation(
        investigation_id=iid, worker_id="w-none-delta")

    db = SessionLocal()
    try:
        from sqlalchemy import text
        events = [(r[0], r[1] or "") for r in db.execute(
            text("select event_type, event_payload_json from investigation_events where investigation_id=:i"),
            {"i": iid}).fetchall()]
        evidence = [(e.validation_status, e.confidence_score) for e in
                    db.query(Evidence).filter(Evidence.investigation_id == iid).all()]
        inv = db.query(Investigation).filter(Investigation.id == iid).first()
        rationale = str(inv.stopping_rationale)
    finally:
        db.close()
    return calls, events, evidence, rationale, managers


def _failed_closed_result(**kw):
    return V.VerificationResult(
        primary_tool="duckdb_sql", secondary_tool="polars_vectorized",
        primary_value=kw.get("primary_metric"), secondary_value=None, observed_delta_pct=None,
        status=VerificationStatus.FAILED, secondary_method="none_computed",
        exception_type="InjectedSecondaryFailure", exception_message="independent engine produced no scalar",
        sql_attempted=kw.get("query_sql"), failure_reason="no_independent_result_relation_available",
    )


def test_controller_survives_failed_independent_verification(monkeypatch):
    calls, events, evidence, rationale, _ = _run_controller(monkeypatch, _failed_closed_result)
    assert calls, "verifier was never reached; test would be vacuous"
    blob = " ".join(p for _, p in events)
    assert "NoneType" not in blob, "None delta crashed a downstream consumer"
    assert rationale != "EXPERIMENT_EXECUTION_FAILED"
    # The fail-closed evidence is recorded as FAILED with zero confidence, never VERIFIED.
    assert evidence, "failed-verification evidence must be persisted, not rolled back"
    for status, confidence in evidence:
        assert str(status).upper().endswith("FAILED")
        assert float(confidence) == 0.0


def test_step_exception_is_recorded_as_failed_experiment(monkeypatch):
    def raises(**kw):
        raise RuntimeError("synthetic step failure")

    calls, events, _, rationale, managers = _run_controller(monkeypatch, raises)
    assert calls and managers
    assert rationale == "EXPERIMENT_EXECUTION_FAILED"
    sm = managers[-1]
    executed = list(sm.state.executed_experiment_ids)
    failed = list(sm.state.failed_experiment_ids)
    assert executed, "expected the experiment to have been registered as executed"
    assert set(executed) <= set(failed), (
        f"stopping_reason=EXPERIMENT_EXECUTION_FAILED but failed_experiment_ids={failed} "
        f"does not include executed {executed}"
    )


# --------------------------------------------------------------------------
# Remaining fabricated-value sites: VERIFIED results must not carry a copied
# primary_metric as an "independent" secondary value.
# --------------------------------------------------------------------------
class TestNoPrimaryMetricSubstitutionOnVerified:
    def test_relation_equivalent_but_no_scalar_keeps_secondary_none(self):
        # Secondary relation matches the primary relation (so VERIFIED is justified),
        # but without ORDER BY no scalar can be reduced. Before the fix this copied
        # primary_metric into secondary_value.
        r = VerificationEngine.verify_secondary(
            primary_df=DF, target_metric_col="revenue", aggregation_type="SUM",
            primary_metric=123.0, query_sql=SQL_NO_ORDER, group_dimension_col="category",
            primary_result_df=PRIM, primary_result_column="total_metric",
        )
        assert r.status == VerificationStatus.VERIFIED
        assert r.secondary_value is None
        assert r.secondary_value != 123.0
        assert r.observed_delta_pct == 0.0

    def test_no_fabricated_secondary_sentinels_remain_in_source(self):
        # The native-recompute VERIFIED branch only runs after a scalar was reduced
        # (secondary_method == "sql_sqlcontext"), so its former primary_metric fallback
        # is unreachable by behavior; guard the source so it cannot return.
        import inspect
        text = inspect.getsource(V)
        assert "secondary_val = primary_metric" not in text
        assert "else primary_metric" not in text
        assert "else 0.0" not in text


# --------------------------------------------------------------------------
# Truncation: the primary metric must come from the FULL result.
# --------------------------------------------------------------------------
class TestPrimaryMetricNotDerivedFromTruncatedRows:
    def _raw_frame(self):
        rng = np.random.RandomState(3)
        return pd.DataFrame({"category": rng.choice(list("ABC"), 40), "revenue": rng.normal(100, 30, 40).round(2)})

    def test_variance_uses_all_rows_when_result_is_truncated(self):
        from packages.analytics_core.src.engines.execution_provider import _execute_duckdb_worker
        frame = self._raw_frame()
        out = _execute_duckdb_worker(
            {"data_table": frame}, "SELECT category, revenue FROM data_table", 10, "VARIANCE", "revenue")
        assert out["is_truncated"] is True
        assert len(out["result_df"]) == 10  # transport frame still bounded
        assert out["primary_metric"] == pytest.approx(float(np.var(frame["revenue"].to_numpy(), ddof=0)))
        assert out["primary_metric"] != pytest.approx(float(np.var(frame["revenue"].head(10).to_numpy(), ddof=0)))

    def test_untruncated_behavior_unchanged(self):
        from packages.analytics_core.src.engines.execution_provider import _execute_duckdb_worker
        frame = self._raw_frame()
        out = _execute_duckdb_worker(
            {"data_table": frame}, "SELECT category, revenue FROM data_table", 1000, "VARIANCE", "revenue")
        assert out["is_truncated"] is False
        assert out["primary_metric"] == pytest.approx(float(np.var(frame["revenue"].to_numpy(), ddof=0)))

    def test_truncated_primary_still_never_verifies(self):
        # Correcting the primary value must not weaken the verifier: a truncated
        # relation still cannot be independently verified.
        r = VerificationEngine.verify_secondary(
            primary_df=DF, target_metric_col="revenue", aggregation_type="SUM",
            primary_metric=11.0, query_sql=SQL_OK, group_dimension_col="category",
            primary_result_df=PRIM, primary_result_column="total_metric",
            primary_result_truncated=True,
        )
        assert r.status == VerificationStatus.FAILED


# --------------------------------------------------------------------------
# Timeout handler: same executed-vs-failed consistency as EXPERIMENT_EXECUTION_FAILED
# --------------------------------------------------------------------------
def test_timeout_is_recorded_as_failed_experiment(monkeypatch):
    from packages.analytics_core.src.engines.execution_provider import QueryExecutionTimeoutError

    def times_out(**kw):
        raise QueryExecutionTimeoutError("synthetic timeout")

    calls, events, _, rationale, managers = _run_controller(monkeypatch, times_out)
    assert calls and managers
    assert rationale == "QUERY_EXECUTION_TIMEOUT"
    sm = managers[-1]
    executed = list(sm.state.executed_experiment_ids)
    failed = list(sm.state.failed_experiment_ids)
    assert executed and set(executed) <= set(failed), (executed, failed)
