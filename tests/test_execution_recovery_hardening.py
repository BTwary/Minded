import os, sys, tempfile
from datetime import datetime, timezone, timedelta
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from apps.api.src.models.entities import Base, Investigation, InvestigationExecution, InvestigationStepExecution, InvestigationJob
from packages.analytics_core.src.execution.checkpointer import InvestigationCheckpointer
from packages.analytics_core.src.execution.state_machine import ExecutionState, JobState, FailureTaxonomy


def make_session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine), engine


def test_worker_restart_aborts_previous_execution_and_creates_new_owner():
    SessionLocal, _ = make_session()
    db = SessionLocal()
    inv = Investigation(id="INV-1", project_id="P1", question="q")
    db.add(inv); db.commit(); db.close()
    cp = InvestigationCheckpointer(session_factory=SessionLocal)
    first = cp.get_or_create_execution("INV-1", "worker-A", attempt=1)
    second = cp.get_or_create_execution("INV-1", "worker-B", attempt=2)
    assert first != second
    db = SessionLocal()
    old = db.get(InvestigationExecution, first)
    new = db.get(InvestigationExecution, second)
    assert old.status == ExecutionState.ABORTED
    assert old.error_code == FailureTaxonomy.WORKER_CRASH
    assert new.worker_id == "worker-B"
    db.close()


def test_completed_step_is_still_idempotent_after_restart():
    SessionLocal, _ = make_session()
    db = SessionLocal(); db.add(Investigation(id="INV-2", project_id="P1", question="q")); db.commit(); db.close()
    cp = InvestigationCheckpointer(session_factory=SessionLocal)
    first = cp.get_or_create_execution("INV-2", "worker-A", attempt=1)
    calls=[]
    out = cp.execute_step_transactionally("INV-2", first, 1, "TEST", "SAME-STEP", lambda db: calls.append(1) or {"ok": True})
    assert out == {"ok": True}
    second = cp.get_or_create_execution("INV-2", "worker-B", attempt=2)
    out2 = cp.execute_step_transactionally("INV-2", second, 1, "TEST", "SAME-STEP", lambda db: calls.append(2) or {"ok": True})
    assert out2 is None
    assert calls == [1]


def test_stale_execution_recovery_is_auditable():
    SessionLocal, _ = make_session()
    db = SessionLocal(); db.add(Investigation(id="INV-3", project_id="P1", question="q")); db.commit()
    old = InvestigationExecution(id="EXEC-old", investigation_id="INV-3", worker_id="dead-worker", status=ExecutionState.RUNNING, heartbeat_at=datetime.now(timezone.utc)-timedelta(minutes=10))
    db.add(old); db.commit(); db.close()
    cp = InvestigationCheckpointer(session_factory=SessionLocal)
    assert cp.recover_stale_executions(stale_after_seconds=60) == 1
    db=SessionLocal(); rec=db.get(InvestigationExecution, "EXEC-old")
    assert rec.status == ExecutionState.ABORTED
    assert rec.error_code == FailureTaxonomy.WORKER_CRASH
    db.close()
