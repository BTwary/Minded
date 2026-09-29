import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from apps.api.src.models.entities import (
    Base, BeliefUpdate, Evidence, EvidenceVerification, Experiment,
    Investigation, InvestigationVerdict, Observation,
)
from packages.analytics_core.src.engines.dataset_provider import InMemoryDatasetProvider
from packages.analytics_core.src.relational.plan_serialization import plan_from_dict
from packages.analytics_core.src.runtime.controller import InvestigationController


def test_controller_autonomously_executes_safe_two_table_plan():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False)
    tables = {
        "customers": pd.DataFrame({
            "customer_id": ["C1", "C2", "C3"],
            "segment": ["Enterprise", "SMB", "Enterprise"],
        }),
        "orders": pd.DataFrame({
            "order_id": ["O1", "O2", "O3", "O4", "O5"],
            "customer_id": ["C1", "C1", "C2", "C3", "C2"],
            "amount": [100, 150, 80, 250, 20],
        }),
    }
    investigation_id = "INV-M1-TWO-TABLE"
    with session_factory() as session:
        session.add(Investigation(
            id=investigation_id,
            project_id="PROJECT-M1",
            question="Which customer segment generates the highest total order amount?",
            status="PLANNED",
        ))
        session.commit()

    controller = InvestigationController(
        session_factory=session_factory,
        dataset_provider=InMemoryDatasetProvider(tables),
    )
    completed = controller.execute_investigation(investigation_id, worker_id="m1-test")
    if not completed:
        from apps.api.src.models.entities import InvestigationExecution
        with session_factory() as session:
            failure = session.query(InvestigationExecution).filter_by(investigation_id=investigation_id).one()
            raise AssertionError(failure.error_message)

    with session_factory() as session:
        investigation = session.query(Investigation).filter_by(id=investigation_id).one()
        experiments = session.query(Experiment).filter_by(investigation_id=investigation_id).all()
        assert investigation.status == "COMPLETED"
        assert experiments
        plan_experiment = next(exp for exp in experiments if (exp.arguments_json or {}).get("relational_plan"))
        persisted_plan = plan_experiment.arguments_json["relational_plan"]
        reconstructed_plan = plan_from_dict(persisted_plan)
        assert reconstructed_plan.base_table == "orders"
        assert {(hop.left_table, hop.right_table) for hop in reconstructed_plan.hops} == {("orders", "customers")}
        observation = session.query(Observation).filter_by(experiment_id=plan_experiment.id).one()
        assert reconstructed_plan.to_sql() == observation.sql_executed
        result = observation.result_json
        rows = result["result_rows"]
        assert any(row.get("segment") == "Enterprise" and row.get("total_metric") == 500 for row in rows)
        assert any(row.get("segment") == "SMB" and row.get("total_metric") == 100 for row in rows)
        assert rows[0]["segment"] == "Enterprise"
        assert result["structured_result"]["top_value"] == "Enterprise"
        evidence = session.query(Evidence).filter_by(experiment_id=plan_experiment.id).one()
        verification = session.query(EvidenceVerification).filter_by(evidence_id=evidence.id).one()
        assert verification.secondary_tool == "relational_plan_polars_native"
        assert verification.status == "VERIFIED"
        assert session.query(BeliefUpdate).filter_by(investigation_id=investigation_id).count() > 0
        assert session.query(InvestigationVerdict).filter_by(investigation_id=investigation_id).one()
        assert investigation.reproducible_manifest_hash
