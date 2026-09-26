"""Transactional Step-Level Checkpointer, Idempotency Enforcer, and Event Logger for AA-OS."""
from datetime import datetime, timezone
import hashlib
import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import desc
from apps.api.src.core.database import SessionLocal
from apps.api.src.models.entities import (
    Investigation,
    InvestigationExecution,
    InvestigationStepExecution,
    InvestigationDecisionRequest,
    InvestigationResourceWait,
    InvestigationEvent,
    gen_uuid,
)
from packages.analytics_core.src.execution.state_machine import (
    InvestigationState,
    ExecutionState,
    ExecutionStateMachine,
    FailureTaxonomy,
    VerificationStatus,
    transition_investigation,
    transition_execution,
)


class DatasetUnavailableError(Exception):
    """Raised when an investigation cannot load requested datasets (NO synthetic fallbacks allowed)."""
    pass


class InvestigationCheckpointer:
    """Manages transactional step-level checkpointing, idempotency, and audit event logs."""

    def __init__(self, session_factory=SessionLocal):
        self.session_factory = session_factory

    def get_or_create_execution(
        self,
        investigation_id: str,
        worker_id: str,
        attempt: int = 1,
        resource_id: str = "local_cpu",
    ) -> str:
        """Get the current worker execution or create a restart-safe execution attempt.

        A restarted worker MUST NOT reuse an expired worker's execution record.
        The previous execution is retained as ABORTED for auditability and a new
        execution row becomes the durable owner of the attempt. Step-level
        idempotency keys remain investigation-scoped, so already committed steps
        are not duplicated when the controller replays from the durable state.
        """
        db: Session = self.session_factory()
        try:
            active_exec = (
                db.query(InvestigationExecution)
                .filter(
                    InvestigationExecution.investigation_id == investigation_id,
                    InvestigationExecution.worker_id == worker_id,
                    InvestigationExecution.status.in_([ExecutionState.RUNNING, ExecutionState.CHECKPOINTED]),
                )
                .order_by(desc(InvestigationExecution.created_at))
                .first()
            )
            if active_exec:
                active_exec.heartbeat_at = datetime.now(timezone.utc)
                db.commit()
                return active_exec.id

            prior_exec = (
                db.query(InvestigationExecution)
                .filter(
                    InvestigationExecution.investigation_id == investigation_id,
                    InvestigationExecution.status.in_([ExecutionState.RUNNING, ExecutionState.CHECKPOINTED]),
                )
                .order_by(desc(InvestigationExecution.created_at))
                .first()
            )
            if prior_exec is not None and prior_exec.worker_id != worker_id:
                prior_exec.status = ExecutionState.ABORTED
                prior_exec.error_code = FailureTaxonomy.WORKER_CRASH
                prior_exec.error_message = (
                    f"Execution ownership transferred from worker {prior_exec.worker_id} "
                    f"to restarted worker {worker_id}; prior attempt preserved for audit."
                )
                prior_exec.completed_at = datetime.now(timezone.utc)

            exec_id = f"EXEC-{gen_uuid()[:8]}"
            execution = InvestigationExecution(
                id=exec_id,
                investigation_id=investigation_id,
                worker_id=worker_id,
                status=ExecutionState.RUNNING,
                attempt=max(1, attempt),
                current_step_index=(prior_exec.current_step_index if prior_exec is not None else 0),
                resource_id=resource_id,
                started_at=datetime.now(timezone.utc),
                heartbeat_at=datetime.now(timezone.utc),
            )
            db.add(execution)
            db.commit()
            return exec_id
        finally:
            db.close()

    def recover_stale_executions(self, stale_after_seconds: int = 120) -> int:
        """Mark stale RUNNING/CHECKPOINTED executions as worker-crash recoveries.

        This is deliberately conservative: only executions with an old heartbeat
        are aborted. The queue lease is the concurrency authority; this method
        supplies durable execution-state truth for audits and restart recovery.
        """
        from datetime import timedelta
        db: Session = self.session_factory()
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(seconds=max(1, stale_after_seconds))
            stale = (
                db.query(InvestigationExecution)
                .filter(
                    InvestigationExecution.status.in_([ExecutionState.RUNNING, ExecutionState.CHECKPOINTED]),
                    InvestigationExecution.heartbeat_at < cutoff,
                )
                .all()
            )
            for execution in stale:
                execution.status = ExecutionState.ABORTED
                execution.error_code = FailureTaxonomy.WORKER_CRASH
                execution.error_message = f"No heartbeat received for {stale_after_seconds}s; execution reclaimed for restart."
                execution.completed_at = datetime.now(timezone.utc)
            db.commit()
            return len(stale)
        finally:
            db.close()

    def record_event(
        self,
        investigation_id: str,
        event_type: str,
        payload: Dict[str, Any],
        execution_id: Optional[str] = None,
    ) -> str:
        """Persist an immutable audit/SSE event to the database with concurrency-safe sequence allocation."""
        from sqlalchemy import func
        from sqlalchemy.exc import IntegrityError
        
        max_retries = 5
        import json
        safe_payload = json.loads(json.dumps(payload, default=str))

        for attempt in range(max_retries):
            db: Session = self.session_factory()
            try:
                max_seq = db.query(func.max(InvestigationEvent.sequence)).filter(InvestigationEvent.investigation_id == investigation_id).scalar()
                next_seq = (max_seq or 0) + 1

                event_id = f"EVT-{gen_uuid()[:8]}"
                event = InvestigationEvent(
                    id=event_id,
                    sequence=next_seq,
                    investigation_id=investigation_id,
                    execution_id=execution_id,
                    event_type=event_type,
                    event_payload_json=safe_payload,
                    timestamp=datetime.now(timezone.utc),
                )
                db.add(event)
                db.commit()
                return event_id
            except IntegrityError:
                db.rollback()
                if attempt == max_retries - 1:
                    raise
                time.sleep(0.02 * (attempt + 1))
            finally:
                db.close()

    def is_step_completed(self, investigation_id: str, idempotency_key: str) -> bool:
        """Check if an exact step idempotency key has already completed in the database."""
        db: Session = self.session_factory()
        try:
            step = (
                db.query(InvestigationStepExecution)
                .filter(
                    InvestigationStepExecution.investigation_id == investigation_id,
                    InvestigationStepExecution.idempotency_key == idempotency_key,
                    InvestigationStepExecution.status == "COMPLETED",
                )
                .first()
            )
            return step is not None
        finally:
            db.close()

    def execute_step_transactionally(
        self,
        investigation_id: str,
        execution_id: str,
        step_index: int,
        step_type: str,
        step_code: str,
        step_fn: Callable[[Session], Any],
        input_data: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Execute an analytical step with ACID transactional checkpointing.
        If the step was already completed in a prior attempt/worker, it is skipped safely.
        """
        idempotency_key = f"{investigation_id}:{step_type}:{step_code}"

        # 1. Idempotency check
        if self.is_step_completed(investigation_id, idempotency_key):
            return None

        # 2. Record step start
        input_hash = hashlib.sha256(json.dumps(input_data or {}, sort_keys=True).encode()).hexdigest()
        self.record_event(
            investigation_id=investigation_id,
            execution_id=execution_id,
            event_type="step.started",
            payload={"step_index": step_index, "step_type": step_type, "step_code": step_code, "idempotency_key": idempotency_key},
        )

        db: Session = self.session_factory()
        try:
            # 3. Execute domain step logic (which adds entities to db session)
            result = step_fn(db)

            # 4. Add StepExecution record within the SAME session
            step_exec = InvestigationStepExecution(
                id=f"STEP-{gen_uuid()[:8]}",
                execution_id=execution_id,
                investigation_id=investigation_id,
                step_index=step_index,
                step_type=step_type,
                idempotency_key=idempotency_key,
                status="COMPLETED",
                input_hash=input_hash,
                output_hash=hashlib.sha256(str(result).encode()).hexdigest() if result else None,
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
            )
            db.add(step_exec)

            # 5. Update execution attempt heartbeat and step index
            exec_rec = db.query(InvestigationExecution).filter(InvestigationExecution.id == execution_id).first()
            if exec_rec:
                exec_rec.current_step_index = step_index
                exec_rec.heartbeat_at = datetime.now(timezone.utc)

            # 6. Commit all domain entities and step execution in a single atomic transaction
            db.commit()

            # 7. Record step completed event
            self.record_event(
                investigation_id=investigation_id,
                execution_id=execution_id,
                event_type="step.completed",
                payload={"step_index": step_index, "step_type": step_type, "step_code": step_code},
            )
            return result
        except Exception as e:
            db.rollback()
            # Record step failure
            self.record_event(
                investigation_id=investigation_id,
                execution_id=execution_id,
                event_type="step.failed",
                payload={"step_index": step_index, "step_type": step_type, "step_code": step_code, "error": str(e)},
            )
            raise
        finally:
            db.close()

    def request_human_decision(
        self,
        investigation_id: str,
        execution_id: str,
        question: str,
        options: List[str],
        default_action: Optional[str] = None,
    ) -> str:
        """Suspend execution and persist a DecisionRequest waiting for user response."""
        db: Session = self.session_factory()
        try:
            req_id = f"DEC-{gen_uuid()[:8]}"
            dec_req = InvestigationDecisionRequest(
                id=req_id,
                investigation_id=investigation_id,
                execution_id=execution_id,
                question=question,
                options_json=options,
                default_action=default_action,
                status="PENDING",
                created_at=datetime.now(timezone.utc),
            )
            db.add(dec_req)
            db.commit()

            # State transition to WAITING_FOR_USER
            transition_investigation(
                db=db,
                investigation_id=investigation_id,
                to_state="WAITING_FOR_USER",
                actor="checkpointer",
            )

            # Record event
            self.record_event(
                investigation_id=investigation_id,
                execution_id=execution_id,
                event_type="waiting_for_user",
                payload={"decision_request_id": req_id, "question": question, "options": options},
            )
            return req_id
        finally:
            db.close()

    def request_resource_wait(
        self,
        investigation_id: str,
        execution_id: str,
        resource_id: str,
        reason: str,
        required_capabilities: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Suspend execution and persist a ResourceWait record waiting for compute attachment."""
        db: Session = self.session_factory()
        try:
            wait_id = f"RES-{gen_uuid()[:8]}"
            res_wait = InvestigationResourceWait(
                id=wait_id,
                investigation_id=investigation_id,
                execution_id=execution_id,
                resource_id=resource_id,
                reason=reason,
                required_capabilities_json=required_capabilities or {},
                status="PENDING",
                requested_at=datetime.now(timezone.utc),
            )
            db.add(res_wait)
            db.commit()

            # State transition to WAITING_FOR_RESOURCE
            transition_investigation(
                db=db,
                investigation_id=investigation_id,
                to_state="WAITING_FOR_RESOURCE",
                actor="checkpointer",
            )

            self.record_event(
                investigation_id=investigation_id,
                execution_id=execution_id,
                event_type="waiting_for_resource",
                payload={"resource_wait_id": wait_id, "resource_id": resource_id, "reason": reason},
            )
            return wait_id
        finally:
            db.close()

    def complete_execution(
        self,
        execution_id: str,
        investigation_id: Optional[str] = None,
        manifest_hash: Optional[str] = None,
        final_status: str = "COMPLETED",
        failure_taxonomy: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Mark an execution attempt completed or failed."""
        db: Session = self.session_factory()
        try:
            exec_rec = db.query(InvestigationExecution).filter(InvestigationExecution.id == execution_id).first()
            if exec_rec:
                exec_rec.status = final_status
                exec_rec.completed_at = datetime.now(timezone.utc)
                if failure_taxonomy:
                    # BUGFIX (Phase 10): `InvestigationExecution` has no
                    # `failure_taxonomy` column -- this used to set a
                    # transient, never-persisted attribute, silently
                    # discarding the structured failure classification on
                    # every failed run and leaving `error_code` always NULL.
                    exec_rec.error_code = failure_taxonomy
                if error_message:
                    exec_rec.error_message = error_message
                db.commit()
        finally:
            db.close()

    def fail_execution(
        self,
        investigation_id: str,
        execution_id: str,
        error_message: str,
        failure_class: str = FailureTaxonomy.ANALYTICAL_ERROR,
    ) -> None:
        """Fail execution and mark investigation as FAILED.

        Also closes the blank-terminal-state gap: a FAILED investigation
        with no direct_answer/main_finding is indistinguishable from a
        silent crash to any caller (API response, human-verifier UI). Every
        fail_execution call already carries a specific, human-written
        error_message describing exactly why the analysis could not
        proceed -- this surfaces that same text onto the investigation's
        own answer fields so it is never blank, without inventing wording
        that the caller (controller.py) did not itself provide. Does not
        overwrite an already-set direct_answer (e.g. a caller that already
        composed its own terminal message before calling this).

        For DATA_QUALITY_FAILURE specifically, also sets verdict_type to
        INSUFFICIENT_DATA: this is the one failure class that represents a
        genuine scientific outcome (the analysis was identified and
        attempted but the sample cannot support it), as opposed to an
        operational failure (RESOURCE_UNAVAILABLE, WORKER_CRASH, ...) which
        is not itself a scientific verdict and is left untouched.
        """
        self.complete_execution(
            execution_id=execution_id,
            investigation_id=investigation_id,
            final_status="FAILED",
            failure_taxonomy=failure_class,
            error_message=error_message,
        )
        db: Session = self.session_factory()
        try:
            inv = db.query(Investigation).filter(Investigation.id == investigation_id).first()
            if inv:
                if inv.status != InvestigationState.FAILED:
                    transition_investigation(
                        db=db,
                        investigation_id=investigation_id,
                        to_state=InvestigationState.FAILED,
                        actor="checkpointer",
                    )
                if error_message and not (inv.direct_answer or "").strip():
                    terminal_message = (
                        f"Could not complete this analysis. Reason: {error_message.rstrip('.')}. "
                        "No conclusion was produced."
                    )
                    inv.direct_answer = terminal_message
                    if not (inv.main_finding or "").strip():
                        inv.main_finding = terminal_message
                if failure_class == FailureTaxonomy.DATA_QUALITY_FAILURE:
                    inv.verdict_type = "INSUFFICIENT_DATA"
                db.commit()
        except Exception:
            pass
        finally:
            db.close()

