"""Execution semantics through the production controller, DuckDB and independent verifier."""
from dataclasses import replace
import uuid

import pandas as pd
import pytest

from apps.api.src.core.database import SessionLocal
from apps.api.src.models.entities import Investigation, InvestigationEvent, Observation, Experiment, Evidence
from packages.analytics_core.src.engines.dataset_provider import InMemoryDatasetProvider
from packages.analytics_core.src.runtime.controller import InvestigationController
from packages.analytics_core.src.governance.guided_investigation import evaluate_guided_investigation


@pytest.fixture
def sales():
    return pd.DataFrame({"region": ["West", "West", "South", "South", "East"],
                         "revenue": [100., 300., 50., 150., 900.],
                         "profit": [10., 30., 5., 15., 90.],
                         "order_date": ["2025-01-01", "2025-02-01", "2025-01-02", "2025-02-02", "2024-01-01"]})


def run_plan(frame, question, *, change_contract=None, change_provider=None):
    from packages.analytics_core.src.data.content_identity import compute_content_hash
    provider = InMemoryDatasetProvider({"sales": frame})
    ctx = provider.acquire_context("proj-default", ["sales"])
    contract = evaluate_guided_investigation(question, df=frame).canonical_contract
    if change_contract:
        contract = change_contract(contract)
    iid = "INV-PLAN-" + uuid.uuid4().hex[:8]
    with SessionLocal() as db:
        db.add(Investigation(id=iid, project_id="proj-default", question=question, status="QUEUED", requested_dataset_ids_json=["sales"]))
        db.flush()
        db.add(InvestigationEvent(id=str(uuid.uuid4()), investigation_id=iid, event_type="investigation.canonical_contract_locked", event_payload_json={
            "contract": contract.to_dict(), "contract_hash": contract.contract_hash(),
            "dataset_name": "sales", "dataset_fingerprint": ctx.dataset_fingerprints["sales"],
            "dataset_metadata": ctx.dataset_versions["sales"],
            "dataset_content_fingerprint": compute_content_hash(frame),
        }))
        db.commit()
    if change_provider:
        provider = change_provider(provider)
    controller = InvestigationController(session_factory=SessionLocal, dataset_provider=provider)
    controller.execute_investigation(iid, worker_id="contract-test")
    with SessionLocal() as db:
        inv = db.get(Investigation, iid)
        events = {e.event_type: e.event_payload_json for e in db.query(InvestigationEvent).filter_by(investigation_id=iid).order_by(InvestigationEvent.sequence)}
        observations = db.query(Observation).join(Experiment).filter(Experiment.investigation_id == iid).all()
        return {"id": iid, "status": inv.status, "verdict": inv.verdict_type, "answer": inv.direct_answer,
                "events": events, "observations": [{"sql": o.sql_executed, "result": o.result_json} for o in observations],
                "evidence": [e.validation_status for e in db.query(Evidence).filter_by(investigation_id=iid)]}


@pytest.mark.parametrize("word,op,west", [("total", "SUM", 400.), ("average", "MEAN", 200.)])
def test_approved_aggregate_executes_the_approved_measure(sales, word, op, west):
    run = run_plan(sales, f"What is the {word} revenue by region?")
    assert "investigation.analyst_result" in run["events"], run
    answer = run["events"]["investigation.analyst_result"]
    assert run["status"] == "COMPLETED"
    assert answer["numbers"]["table"]["West"]["value"] == west
    record = answer["numbers"]["execution_record"]
    assert record["aggregation"] == op
    assert record["target_column"] == "revenue"
    assert record["grouping_columns"] == ["region"]
    assert record["contract_status"] == "MATCHED"
    assert record["verification_status"] == "VERIFIED"
    assert record["sql"] == run["observations"][0]["sql"]
    assert record["dataset"]["dataset_id"] == "sales"
    assert record["dataset"]["dataset_version"] == 1
    assert record["result_row_count"] == 3
    assert record["population_row_count"] == 5
    assert run["evidence"] == ["VERIFIED"]
    scope = run["events"]["investigation.dataset_scope_resolved"]
    assert scope["resolved_dataset_names"] == ["sales"]
    assert scope["resolved_dataset_versions"]["sales"]["dataset_version"] == 1


def test_ranking_preserves_average_and_lowest_direction(sales):
    run = run_plan(sales, "Which region has the lowest average revenue?")
    assert run["status"] == "COMPLETED"
    assert "investigation.analyst_result" in run["events"], run
    answer = run["events"]["investigation.analyst_result"]
    record = answer["numbers"]["execution_record"]
    assert record["aggregation"] == "MEAN"
    assert record["ranking_direction"] == "ASC"
    assert list(answer["numbers"]["table"])[0] == "South"
    assert "ASC" in record["sql"]


def test_approved_correction_owns_execution_not_original_text(sales):
    run = run_plan(sales, "What is the total revenue by region?", change_contract=lambda c: replace(c, requested_aggregation="MEAN", target_column="profit"))
    assert "investigation.analyst_result" in run["events"], run
    answer = run["events"]["investigation.analyst_result"]
    assert answer["numbers"]["table"]["West"]["value"] == 20.
    assert "execution_record" in answer["numbers"], run["observations"]
    assert answer["numbers"]["execution_record"]["aggregation"] == "MEAN"
    assert 'AVG' in run["observations"][0]["sql"]
    assert 'profit' in run["observations"][0]["sql"]


def test_comparison_applies_exact_named_scope(sales):
    run = run_plan(sales, "Compare average revenue in West and South")
    assert "investigation.analyst_result" in run["events"], run
    numbers = run["events"]["investigation.analyst_result"]["numbers"]
    assert numbers["table"] == {"West": {"value": 200., "n": 2}, "South": {"value": 100., "n": 2}}
    record = numbers["execution_record"]
    assert record["aggregation"] == "MEAN"
    assert record["comparison_values"] == ["West", "South"]
    assert record["filters"] == [{"column": "region", "operator": "IN", "value": ["West", "South"]}]
    assert record["population_row_count"] == 4
    assert "East" not in str(run["observations"][0]["result"]["result_rows"])


@pytest.mark.parametrize("question,expected", [
    ("What is the total revenue by region where revenue > 100?", {"East": 900., "West": 300., "South": 150.}),
    ("What is the total revenue by region in 2025?", {"West": 400., "South": 200.}),
])
def test_filter_and_calendar_scope_reach_sql(sales, question, expected):
    run = run_plan(sales, question)
    assert "investigation.analyst_result" in run["events"], run
    numbers = run["events"]["investigation.analyst_result"]["numbers"]
    assert {k: v["value"] for k, v in numbers["table"].items()} == expected
    assert "WHERE" in run["observations"][0]["sql"]
    assert numbers["execution_record"]["filters"]


def test_absolute_approved_date_range_overrides_question(sales):
    run = run_plan(sales, "What is the total revenue by region?", change_contract=lambda c: replace(
        c, time_column="order_date", time_range={"type": "absolute", "start": "2025-01-01", "end": "2025-02-01", "end_exclusive": True}))
    numbers = run["events"]["investigation.analyst_result"]["numbers"]
    assert {k: v["value"] for k, v in numbers["table"].items()} == {"West": 100., "South": 50.}
    assert numbers["execution_record"]["time_range"]["end_exclusive"] is True


@pytest.mark.parametrize("tampering", ["metric", "dataset", "version", "missing_receipt", "result", "changed_input"])
def test_execution_mismatch_or_failed_verification_never_answers(sales, monkeypatch, tampering):
    from packages.analytics_core.src.engines.execution_provider import DuckDBExecutionProvider
    original = DuckDBExecutionProvider.execute_query
    def execute(self, **kwargs):
        if tampering == "metric":
            kwargs["query"] = kwargs["query"].replace('"revenue"', '"profit"')
        if tampering == "changed_input":
            # Same revenue aggregates, different physical input. Scalar/relation agreement
            # alone cannot establish that the approved dataset actually ran.
            from dataclasses import replace as replace_context
            ctx = kwargs["data_context"]
            changed = ctx.datasets_map["sales"].copy()
            changed["profit"] *= 10
            kwargs["data_context"] = replace_context(ctx, datasets_map={"sales": changed})
        result = original(self, **kwargs)
        if tampering == "dataset":
            result.source_datasets = ["another-dataset"]
        elif tampering == "version":
            result.source_dataset_versions = {"sales": {"dataset_version": 99}}
        elif tampering == "missing_receipt":
            del result.executed_query
        elif tampering == "result":
            result.result_df.loc[0, "value"] += 1
        return result
    monkeypatch.setattr(DuckDBExecutionProvider, "execute_query", execute)
    run = run_plan(sales, "What is the total revenue by region?")
    assert run["status"] == "FAILED"
    assert run["verdict"] not in {"OBSERVED", "STATISTICALLY_SIGNIFICANT"}
    assert "investigation.analyst_result" not in run["events"]
    assert run["evidence"] == ["FAILED"]
    record = run["events"]["investigation.approved_execution"]
    assert record["contract_status"] == "MISMATCH" or record["verification_status"] != "VERIFIED"
    if tampering == "metric":
        assert '"profit"' in record["sql"]


@pytest.mark.parametrize("change", ["data", "physical_data", "version", "identity"])
def test_approved_snapshot_change_refuses_before_execution(sales, change):
    from dataclasses import replace as ctx_replace
    class Changed(InMemoryDatasetProvider):
        def acquire_context(self, project_id, dataset_ids=None):
            ctx = super().acquire_context(project_id, dataset_ids)
            if change == "physical_data":
                changed = ctx.datasets_map["sales"].copy()
                changed["profit"] *= 10
                return ctx_replace(ctx, datasets_map={"sales": changed})
            if change == "data":
                return ctx_replace(ctx, dataset_fingerprints={"sales": "changed"})
            metadata = dict(ctx.dataset_versions["sales"])
            metadata["dataset_version" if change == "version" else "dataset_id"] = 99 if change == "version" else "another"
            return ctx_replace(ctx, dataset_versions={"sales": metadata})
    run = run_plan(sales, "What is the total revenue by region?", change_provider=lambda p: Changed(p.datasets_map))
    assert run["status"] == "FAILED"
    assert not run["observations"]
    assert "investigation.analyst_result" not in run["events"]


@pytest.mark.parametrize("question", ["Recommend an optimal marketing budget", "Compare average revenue in West and Atlantis"])
def test_unsupported_question_never_gets_predefined_result(sales, question):
    run = run_plan(sales, question)
    assert "investigation.analyst_result" not in run["events"]
    assert not run["observations"]
    assert "unsupported or unresolved" in run["answer"]


def test_exclusion_is_preserved_without_contradictory_inclusion(sales):
    sales["status"] = ["active", "cancelled", "active", "cancelled", "active"]
    run = run_plan(sales, "Calculate total revenue by region, excluding cancelled subscriptions.")
    numbers = run["events"]["investigation.analyst_result"]["numbers"]
    assert numbers["filters"] == [{"column": "status", "operator": "!=", "value": "cancelled"}]
    assert {k: v["value"] for k, v in numbers["table"].items()} == {"East": 900., "West": 100., "South": 50.}


def test_desktop_packages_the_same_controller_and_isolated_provider():
    """Static wiring only; this does not claim Windows/native-window execution."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    entry = (root / "packaging/windows/minded_entry.py").read_text()
    assert 'from apps.api.src.main import app' in entry
    assert 'os.environ["AAOS_LOCAL_AUTOWORKER"] = "true"' in entry
    assert '"--aaos-isolated-worker"' in entry
    main = (root / "apps/api/src/main.py").read_text()
    assert 'from packages.analytics_core.src.execution.worker import WorkerSupervisor' in main
    worker = (root / "packages/analytics_core/src/execution/worker.py").read_text()
    assert 'self.controller = controller or InvestigationController(' in worker
    assert 'self.controller.execute_investigation(' in worker
    build = (root / "packaging/windows/build_exe_scope.ps1").read_text()
    assert '--collect-all packages' in build


def test_comparison_preview_and_contract_show_the_same_scope(sales):
    preview = evaluate_guided_investigation("Compare average revenue in West and South", df=sales)
    assert "West" in preview.planned_calculation and "South" in preview.planned_calculation
    assert "WHERE" in preview.planned_calculation
    assert preview.canonical_contract.filter_conditions == ({"column": "region", "operator": "IN", "value": ["West", "South"]},)


def test_scope_row_count_comes_from_sql_null_semantics(sales):
    sales.loc[0, "profit"] = None
    sales.loc[3, "revenue"] = None
    run = run_plan(sales, "What is the total revenue by region where profit != 0?")
    numbers = run["events"]["investigation.analyst_result"]["numbers"]
    # SQL excludes NULL profit. NULL revenue does not remove the supporting input row.
    assert numbers["population_row_count"] == 4
    assert numbers["table"]["West"]["n"] == 1
    assert run["observations"][0]["result"]["result_rows"][0]["n"] >= 1
    assert 'COUNT(*) AS n' in run["observations"][0]["sql"]


def test_typed_row_count_roundtrips_and_rejects_aggregate_in_where():
    from packages.analytics_core.src.relational.compiler import RelationalPlan
    from packages.analytics_core.src.relational.expressions import CountRows, Comparison, Literal, to_sql, to_polars
    from packages.analytics_core.src.relational.plan_serialization import plan_to_dict, plan_from_dict
    import polars as pl
    plan = RelationalPlan(base_table="data_table", aggregations={"value": CountRows()})
    restored = plan_from_dict(plan_to_dict(plan))
    frame = pd.DataFrame({"metric": [1., None, 3.]})
    assert to_sql(CountRows()) == "COUNT(*)"
    assert pl.from_pandas(frame).select(to_polars(CountRows())).item() == 3
    assert restored.to_polars({"data_table": frame})["value"][0] == 3
    bad = RelationalPlan(base_table="data_table", filters=[Comparison(CountRows(), ">", Literal(0))])
    assert bad.validate()


def test_grouped_row_count_keeps_rows_with_null_metrics(sales):
    sales.loc[0, "revenue"] = None
    run = run_plan(sales, "Count the records in each region.")
    numbers = run["events"]["investigation.analyst_result"]["numbers"]
    assert numbers["operation"] == "ROW_COUNT"
    assert numbers["table"]["West"]["value"] == 2
    assert numbers["population_row_count"] == 5


def test_undefined_group_metric_does_not_become_a_verified_number(sales):
    sales.loc[sales.region == "West", "revenue"] = None
    run = run_plan(sales, "What is the average revenue by region?")
    assert "investigation.analyst_result" not in run["events"]
    assert run["status"] == "FAILED"
    assert run["observations"]
    assert run["evidence"] == ["FAILED"]
    assert "investigation.approved_execution" in run["events"]


def test_zero_matching_scope_does_not_answer_unfiltered_question(sales):
    run = run_plan(sales, "What is the total revenue by region where revenue > 1000000?")
    assert run["status"] == "FAILED"
    assert "investigation.analyst_result" not in run["events"]


def test_expanded_primitive_uses_existing_canonical_executor(sales):
    run = run_plan(sales, "Calculate variance of revenue by region")
    assert run["status"] == "COMPLETED", run
    numbers = run["events"]["investigation.analyst_result"]["numbers"]
    assert numbers["table"]["region=West"]["value"] == pytest.approx(20000.)
    record = numbers["execution_record"]
    assert record["aggregation"] == "VARIANCE"
    assert record["sql"] is None
    assert record["contract_status"] == "MATCHED"
    assert record["verification_status"] == "VERIFIED"
    assert run["observations"][0]["sql"] is None


def test_approved_operation_graph_executes_and_persists_actual_outputs(sales):
    run = run_plan(sales, "Calculate total revenue by region and rank regions by revenue")
    assert run["status"] == "COMPLETED", run
    numbers = run["events"]["investigation.analyst_result"]["numbers"]
    ranked = numbers["operation_graph"]["outputs"]["rank"]["records"]
    assert [(r["group"]["region"], r["value"]) for r in ranked] == [("East", 900.), ("West", 400.), ("South", 200.)]
    record = numbers["execution_record"]
    assert record["operation_graph"] == record["approved_contract"]["operation_graph"]
    assert record["population_row_count"] == 5
    assert record["output_row_counts"] == {"aggregate": 3, "rank": 3}
    assert record["sql"] is None
    assert run["observations"][0]["result"]["result_rows"][1]["result"]["records"] == ranked


def test_graph_cannot_drop_an_outer_approved_filter(sales):
    run = run_plan(sales, "Calculate total revenue by region and rank regions by revenue",
                   change_contract=lambda c: replace(c, filter_conditions=({"column": "revenue", "operator": ">", "value": 200},)))
    assert run["status"] == "FAILED", run
    assert not run["observations"]
    assert "investigation.analyst_result" not in run["events"]


def test_specialist_without_contract_adapter_stops_before_any_calculation(sales, monkeypatch):
    from packages.analytics_core.src.engines.execution_provider import DuckDBExecutionProvider
    def unexpected(*args, **kwargs):
        pytest.fail("An unsupported approved contract must never execute a generic experiment")
    monkeypatch.setattr(DuckDBExecutionProvider, "execute_query", unexpected)
    run = run_plan(sales, "What is the total revenue by region?", change_contract=lambda c: replace(c, task_family="PREDICTION"))
    assert run["status"] == "FAILED"
    assert "no contract-bound execution adapter" in run["answer"]
    assert not run["observations"]


def test_share_of_total_is_derived_from_the_actual_verified_relation(sales):
    run = run_plan(sales, "What is the total revenue by region?", change_contract=lambda c: replace(c, share_of_total=True))
    numbers = run["events"]["investigation.analyst_result"]["numbers"]
    assert numbers["share_of_total"]["denominator"] == 1500.
    assert numbers["table"]["West"]["share_of_total"] == pytest.approx(400 / 1500)
    assert numbers["execution_record"]["share_of_total"] is True


@pytest.mark.parametrize("changes", [{"requested_outputs": ("chart",)}, {"share_of_total": True, "requested_aggregation": "MEAN"}])
def test_unbound_material_output_requirements_stop_before_execution(sales, changes):
    run = run_plan(sales, "What is the total revenue by region?", change_contract=lambda c: replace(c, **changes))
    assert run["status"] == "FAILED"
    assert not run["observations"]
    assert "investigation.analyst_result" not in run["events"]


def test_truncated_relation_is_not_presented_as_a_complete_answer(sales, monkeypatch):
    from packages.analytics_core.src.engines.execution_provider import DuckDBExecutionProvider
    original = DuckDBExecutionProvider.execute_query
    def execute(self, **kwargs):
        result = original(self, **kwargs)
        result.is_truncated = True
        return result
    monkeypatch.setattr(DuckDBExecutionProvider, "execute_query", execute)
    run = run_plan(sales, "What is the total revenue by region?")
    assert run["status"] == "FAILED"
    assert "investigation.analyst_result" not in run["events"]
    assert run["evidence"] == ["FAILED"]
    assert "truncated" in run["events"]["investigation.approved_execution"]["verification_reason"]


def test_unknown_graph_parameter_is_not_silently_ignored(sales):
    def change(c):
        from copy import deepcopy
        graph = deepcopy(c.operation_graph)
        graph["nodes"][0]["params"]["date_range"] = {"start": "2025-01-01"}
        return replace(c, operation_graph=graph)
    run = run_plan(sales, "Calculate total revenue by region and rank regions by revenue", change_contract=change)
    assert run["status"] == "FAILED"
    assert not run["observations"]
    assert "does not consume" in run["answer"]


def test_history_reopens_source_identity_from_actual_execution_receipt(sales):
    from apps.api.src.api.v1.investigations import list_analysis_history
    from apps.api.src.models.entities import User, Dataset
    run = run_plan(sales, "What is the total revenue by region?")
    with SessionLocal() as db:
        user = db.get(User, "default-user")
        if user is None:
            user = User(id="default-user", email="approved-history@example.test", full_name="Approved History", hashed_password="unused", is_active=True)
            db.add(user)
        db.merge(Dataset(id="sales", project_id="proj-default", name="sales", current_version=1, row_count=5, column_count=4))
        db.commit()
        result = list_analysis_history(limit=1000, current_user=user, db=db)
        record = next(r for r in result["records"] if r["id"] == run["id"])
        source = record["source_datasets"][0]
        assert source["name"] == "sales"
        assert source["dataset_id"] == "sales"
        assert source["dataset_version"] == 1
        assert source["available"] is True
        assert record["verification_status"] == "VERIFIED"
