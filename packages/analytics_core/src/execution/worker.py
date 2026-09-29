"""Pure Autonomous Investigation Worker: Decoupled Durable Execution Runner."""
import os
import threading
import time
from typing import Optional
from sqlalchemy.orm import Session

from apps.api.src.core.database import SessionLocal
from apps.api.src.models.entities import Investigation, gen_uuid
from packages.analytics_core.src.execution.queue import (
    BaseQueueProvider,
    get_default_queue_provider,
)
from packages.analytics_core.src.execution.state_machine import (
    InvestigationState,
    JobState,
    FailureTaxonomy,
    transition_investigation,
    transition_job,
)
from packages.analytics_core.src.execution.checkpointer import (
    InvestigationCheckpointer,
    DatasetUnavailableError,
)
from packages.analytics_core.src.runtime.controller import InvestigationController


class InvestigationWorker:
    """Pure durable worker orchestrating async execution through InvestigationController."""

    def __init__(
        self,
        worker_id: Optional[str] = None,
        queue_provider: Optional[BaseQueueProvider] = None,
        session_factory=SessionLocal,
        heartbeat_interval_seconds: int = 15,
        lease_duration_seconds: int = 60,
        controller: Optional[InvestigationController] = None,
    ):
        self.worker_id = worker_id or f"worker-{os.getpid()}-{gen_uuid()[:6]}"
        self.queue = queue_provider or get_default_queue_provider()
        self.session_factory = session_factory
        self.heartbeat_interval = heartbeat_interval_seconds
        self.lease_duration = lease_duration_seconds
        self.checkpointer = InvestigationCheckpointer(session_factory=session_factory)
        self.controller = controller or InvestigationController(
            session_factory=session_factory,
            checkpointer=self.checkpointer,
        )
        self._running = False
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._current_job_id: Optional[str] = None
        self._lease_lost = False

    def _start_heartbeat(self, job_id: str):
        self._current_job_id = job_id
        self._lease_lost = False

        def _heartbeat_loop():
            while self._current_job_id == job_id and self._running:
                time.sleep(self.heartbeat_interval)
                if self._current_job_id == job_id:
                    ok = self.queue.renew_lease(
                        job_id=job_id,
                        worker_id=self.worker_id,
                        lease_duration_seconds=self.lease_duration,
                    )
                    if not ok:
                        self._lease_lost = True
                        break

        self._heartbeat_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    def _stop_heartbeat(self):
        self._current_job_id = None
        self._heartbeat_thread = None

    def _is_cancelled(self, investigation_id: str) -> bool:
        """Check if investigation was marked CANCEL_REQUESTED."""
        db: Session = self.session_factory()
        try:
            inv = db.query(Investigation).filter(Investigation.id == investigation_id).first()
            return inv is not None and inv.status == InvestigationState.CANCEL_REQUESTED
        finally:
            db.close()

    def process_next_job(self) -> bool:
        """Claim and execute one queued investigation job via InvestigationController."""
        job_claim = self.queue.claim_job(
            worker_id=self.worker_id,
            lease_duration_seconds=self.lease_duration,
        )
        if not job_claim:
            return False

        job_id = job_claim["job_id"]
        investigation_id = job_claim["investigation_id"]
        attempt = job_claim["attempt"]

        self._running = True
        self._start_heartbeat(job_id)

        # Reconcile any execution records left stale by an earlier crashed worker.
        self.checkpointer.recover_stale_executions(stale_after_seconds=max(self.lease_duration * 2, 30))

        # Transition Job from CLAIMED to RUNNING
        db = self.session_factory()
        try:
            transition_job(
                db=db,
                job_id=job_id,
                to_state=JobState.RUNNING,
                worker_id=self.worker_id,
            )
        finally:
            db.close()

        try:
            success = self.controller.execute_investigation(
                investigation_id=investigation_id,
                worker_id=self.worker_id,
                attempt=attempt,
                cancellation_check=lambda: self._is_cancelled(investigation_id) or self._lease_lost,
            )
            if success and not self._lease_lost:
                completed = self.queue.complete_job(job_id=job_id, worker_id=self.worker_id)
                if not completed:
                    raise RuntimeError(f"JobLifecycleError: Failed to complete job {job_id} for worker {self.worker_id}")
            elif self._lease_lost:
                # The queue lease has moved to another worker. Do not write a
                # terminal investigation result from the stale worker. The
                # replacement worker will resume from durable checkpoints.
                self.checkpointer.record_event(
                    investigation_id=investigation_id,
                    execution_id=None,
                    event_type="investigation.worker_lease_lost",
                    payload={"job_id": job_id, "worker_id": self.worker_id},
                )
                return False
            return success
        except DatasetUnavailableError as e:
            db = self.session_factory()
            try:
                transition_investigation(
                    db=db,
                    investigation_id=investigation_id,
                    to_state=InvestigationState.FAILED,
                    error_code=FailureTaxonomy.DATASET_UNAVAILABLE,
                    error_message=str(e),
                )
            finally:
                db.close()
            self.queue.fail_job(job_id=job_id, worker_id=self.worker_id, error_code=FailureTaxonomy.DATASET_UNAVAILABLE, error_message=str(e))
            return False
        except Exception as e:
            import traceback
            traceback.print_exc()
            db = self.session_factory()
            try:
                transition_investigation(
                    db=db,
                    investigation_id=investigation_id,
                    to_state=InvestigationState.FAILED,
                    error_code=FailureTaxonomy.INTERNAL_ERROR,
                    error_message=str(e),
                )
            finally:
                db.close()
            self.queue.fail_job(job_id=job_id, worker_id=self.worker_id, error_code=FailureTaxonomy.INTERNAL_ERROR, error_message=str(e))
            return False
        finally:
            self._stop_heartbeat()
            self._running = False


class WorkerSupervisor:
    """Supervises worker processes with graceful SIGINT/SIGTERM shutdown and queue polling."""

    def __init__(
        self,
        worker_id: Optional[str] = None,
        poll_interval_seconds: float = 1.0,
        session_factory=SessionLocal,
    ):
        self.worker = InvestigationWorker(worker_id=worker_id, session_factory=session_factory)
        self.poll_interval = poll_interval_seconds
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run_loop(self):
        """Continuously poll durable queue and execute investigations until stopped."""
        print(f"[WorkerSupervisor] Starting autonomous worker {self.worker.worker_id}...")
        while not self._stop_event.is_set():
            try:
                processed = self.worker.process_next_job()
                if not processed:
                    time.sleep(self.poll_interval)
            except Exception as e:
                print(f"[WorkerSupervisor Error] {e}")
                time.sleep(self.poll_interval)
        print(f"[WorkerSupervisor] Worker {self.worker.worker_id} stopped cleanly.")
