"""Durable State Machines & Canonical Transition Validators for AA-OS Investigations, Jobs, and Executions."""
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set
from sqlalchemy.orm import Session
from apps.api.src.models.entities import (
    Investigation,
    InvestigationJob,
    InvestigationExecution,
    InvestigationEvent,
    gen_uuid,
)


class AnalyticalPhase:
    """Canonical scientific phases exposed to the UI and durable event stream."""
    QUESTION_UNDERSTANDING = "QUESTION_UNDERSTANDING"
    SEMANTIC_MODEL = "SEMANTIC_MODEL"
    DATA_READINESS = "DATA_READINESS"
    CONTRACT_COMPILED = "CONTRACT_COMPILED"
    HYPOTHESIS_FORMATION = "HYPOTHESIS_FORMATION"
    METHOD_SELECTION = "METHOD_SELECTION"
    EXPERIMENT_SELECTION = "EXPERIMENT_SELECTION"
    EXPERIMENT_EXECUTION = "EXPERIMENT_EXECUTION"
    VERIFICATION = "VERIFICATION"
    BELIEF_UPDATE = "BELIEF_UPDATE"
    ADVERSARIAL_REVIEW = "ADVERSARIAL_REVIEW"
    REPLANNING = "REPLANNING"
    STOPPING = "STOPPING"
    VERDICT = "VERDICT"
    DECISION_GUIDANCE = "DECISION_GUIDANCE"
    FINALIZING = "FINALIZING"

class InvestigationState:
    PLANNED = "PLANNED"
    QUEUED = "QUEUED"
    CLAIMED = "CLAIMED"
    RUNNING = "RUNNING"
    WAITING_FOR_USER = "WAITING_FOR_USER"
    WAITING_FOR_RESOURCE = "WAITING_FOR_RESOURCE"
    VERIFYING = "VERIFYING"
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"


class JobState:
    QUEUED = "QUEUED"
    CLAIMED = "CLAIMED"
    RUNNING = "RUNNING"
    LEASE_EXPIRED = "LEASE_EXPIRED"
    WAITING_FOR_USER = "WAITING_FOR_USER"
    WAITING_FOR_RESOURCE = "WAITING_FOR_RESOURCE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"


class ExecutionState:
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    CHECKPOINTED = "CHECKPOINTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"


class FailureTaxonomy:
    DATASET_UNAVAILABLE = "DATASET_UNAVAILABLE"
    DATA_QUALITY_FAILURE = "DATA_QUALITY_FAILURE"
    ANALYTICAL_ERROR = "ANALYTICAL_ERROR"
    VERIFICATION_FAILURE = "VERIFICATION_FAILURE"
    RESOURCE_UNAVAILABLE = "RESOURCE_UNAVAILABLE"
    RESOURCE_TIMEOUT = "RESOURCE_TIMEOUT"
    USER_REQUIRED = "USER_REQUIRED"
    CANCELLED = "CANCELLED"
    WORKER_CRASH = "WORKER_CRASH"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    PROVENANCE_FAILURE = "PROVENANCE_FAILURE"


class VerificationStatus:
    VERIFIED = "VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED"
    FAILED = "FAILED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


# ---------------------------------------------------------------------------
# P0 epistemic-integrity fix (verification-status contract)
#
# This is the ONE authoritative translation between the internal
# VerificationStatus vocabulary above (written by TransitionEngine /
# InvestigationController to Evidence.validation_status) and the public
# ValidationStatus vocabulary (packages.shared.src.enums) exposed through
# AnalysisService / AnalysisResponse.
#
# Prior to this fix, apps/api/src/services/analysis_service.py re-implemented
# this mapping inline with a case-sensitive comparison against lowercase
# strings ("verified"/"failed"/"partial") that never matched the actual
# uppercase values the controller writes ("VERIFIED"/"FAILED"/
# "PARTIALLY_VERIFIED"), so EVERY evidence row -- verified or not -- fell
# through to ValidationStatus.SKIPPED. packages/analytics_core/src/runtime/
# state_reconstruction.py had the identical bug for hypothesis
# supporting/contradicting-evidence classification after a restart.
#
# Both call sites now import and use to_public_validation_status() below
# instead of maintaining their own string comparisons. Do not add another
# comparison against these status strings anywhere else in the codebase --
# route it through this function so there is exactly one place that can
# ever be wrong.
# ---------------------------------------------------------------------------
from packages.shared.src.enums import ValidationStatus  # noqa: E402

# Legacy lowercase vocabulary written by pre-canonical-VerificationStatus
# code paths (e.g. scripts/migrate_analysis_runs_to_investigations.py,
# and the Evidence.validation_status column's own "unverified" default).
# Mapped explicitly, rather than silently, so old and new data both
# resolve to a truthful public status.
_VERIFICATION_TO_VALIDATION_STATUS = {
    VerificationStatus.VERIFIED: ValidationStatus.PASSED,
    VerificationStatus.FAILED: ValidationStatus.FAILED,
    # Deliberate public-vocabulary decision, not an accidental renaming:
    # ValidationStatus has no PARTIAL member (its four values are PASSED /
    # FAILED / WARNING / SKIPPED / UNKNOWN), and WARNING is the intended
    # public representation of "the experiment ran and produced a result,
    # but dual-engine verification only partially confirmed it" -- i.e.
    # the same "proceed, but flag this as less certain" semantics WARNING
    # already carries elsewhere in the public contract. If a future
    # consumer needs to distinguish PARTIALLY_VERIFIED from other WARNING
    # causes, add a dedicated ValidationStatus.PARTIAL member explicitly
    # rather than overloading WARNING further -- do not silently repurpose
    # this mapping.
    VerificationStatus.PARTIALLY_VERIFIED: ValidationStatus.WARNING,
    # UNVERIFIED / NOT_APPLICABLE are deliberately-defined non-verified
    # states (verification was legitimately not performed / does not
    # apply here) -- this is the ONLY legitimate way to produce SKIPPED.
    # An unrecognized status is never given this treatment; see the
    # UNKNOWN fallback below.
    VerificationStatus.UNVERIFIED: ValidationStatus.SKIPPED,
    VerificationStatus.NOT_APPLICABLE: ValidationStatus.SKIPPED,
    "verified": ValidationStatus.PASSED,
    "failed": ValidationStatus.FAILED,
    "partial": ValidationStatus.WARNING,
    "unverified": ValidationStatus.SKIPPED,
}


def to_public_validation_status(internal_status: Optional[str]) -> ValidationStatus:
    """Translate an internal verification status into the public ValidationStatus.

    `internal_status` is whatever is currently stored in
    Evidence.validation_status: either a canonical VerificationStatus value
    (VERIFIED/FAILED/PARTIALLY_VERIFIED/UNVERIFIED/NOT_APPLICABLE) or a
    legacy lowercase equivalent. A value of None means no verification was
    ever recorded for this evidence, which is legitimately SKIPPED.

    A value that matches neither vocabulary is NOT a legitimate skip -- it
    means the internal status has drifted (a new status was introduced
    upstream without updating this mapping, or the data is corrupted) and
    must be surfaced as ValidationStatus.UNKNOWN rather than hidden as
    SKIPPED, so it is visible to operators, tests, and monitoring instead
    of silently disappearing.
    """
    if internal_status is None:
        return ValidationStatus.SKIPPED
    if internal_status in _VERIFICATION_TO_VALIDATION_STATUS:
        return _VERIFICATION_TO_VALIDATION_STATUS[internal_status]
    # Defense in depth against future casing drift -- try a
    # case-insensitive match before giving up, since a status that merely
    # differs in case is not really "unknown."
    upper = internal_status.strip().upper()
    for key, value in _VERIFICATION_TO_VALIDATION_STATUS.items():
        if key.upper() == upper:
            return value
    return ValidationStatus.UNKNOWN


class InvestigationStateMachine:
    """Strict lifecycle transition validator for the canonical Investigation entity."""

    _VALID_TRANSITIONS: Dict[str, Set[str]] = {
        InvestigationState.PLANNED: {
            InvestigationState.QUEUED,
            InvestigationState.CANCELLED,
        },
        InvestigationState.QUEUED: {
            InvestigationState.CLAIMED,
            InvestigationState.RUNNING,
            InvestigationState.CANCEL_REQUESTED,
            InvestigationState.CANCELLED,
            InvestigationState.FAILED,
        },
        InvestigationState.CLAIMED: {
            InvestigationState.RUNNING,
            InvestigationState.QUEUED,  # If claim lost / lease expired
            InvestigationState.CANCEL_REQUESTED,
            InvestigationState.CANCELLED,
            InvestigationState.FAILED,
        },
        InvestigationState.RUNNING: {
            InvestigationState.WAITING_FOR_USER,
            InvestigationState.WAITING_FOR_RESOURCE,
            InvestigationState.VERIFYING,
            InvestigationState.COMPLETED,
            InvestigationState.PARTIAL,
            InvestigationState.FAILED,
            InvestigationState.CANCEL_REQUESTED,
            InvestigationState.CANCELLED,
        },
        InvestigationState.WAITING_FOR_USER: {
            InvestigationState.QUEUED,
            InvestigationState.RUNNING,
            InvestigationState.CANCEL_REQUESTED,
            InvestigationState.CANCELLED,
            InvestigationState.FAILED,
        },
        InvestigationState.WAITING_FOR_RESOURCE: {
            InvestigationState.QUEUED,
            InvestigationState.RUNNING,
            InvestigationState.CANCEL_REQUESTED,
            InvestigationState.CANCELLED,
            InvestigationState.FAILED,
        },
        InvestigationState.VERIFYING: {
            InvestigationState.COMPLETED,
            InvestigationState.PARTIAL,
            InvestigationState.FAILED,
            InvestigationState.CANCEL_REQUESTED,
        },
        InvestigationState.CANCEL_REQUESTED: {
            InvestigationState.CANCELLED,
            InvestigationState.FAILED,
            InvestigationState.PARTIAL,
        },
        # Terminal states
        InvestigationState.COMPLETED: set(),
        InvestigationState.PARTIAL: set(),
        InvestigationState.FAILED: set(),
        InvestigationState.CANCELLED: set(),
    }

    @classmethod
    def can_transition(cls, from_state: str, to_state: str) -> bool:
        allowed = cls._VALID_TRANSITIONS.get(from_state, set())
        return to_state in allowed

    @classmethod
    def validate_transition(cls, from_state: str, to_state: str) -> None:
        if not cls.can_transition(from_state, to_state):
            raise ValueError(
                f"Invalid Investigation transition from '{from_state}' to '{to_state}'. "
                f"Allowed transitions: {cls._VALID_TRANSITIONS.get(from_state, set())}"
            )


class JobStateMachine:
    """Strict transition validator for durable queue InvestigationJob entities."""

    _VALID_TRANSITIONS: Dict[str, Set[str]] = {
        JobState.QUEUED: {JobState.CLAIMED, JobState.CANCELLED, JobState.FAILED},
        JobState.CLAIMED: {JobState.RUNNING, JobState.COMPLETED, JobState.QUEUED, JobState.FAILED, JobState.CANCELLED},
        JobState.RUNNING: {
            JobState.WAITING_FOR_USER,
            JobState.WAITING_FOR_RESOURCE,
            JobState.COMPLETED,
            JobState.FAILED,
            JobState.LEASE_EXPIRED,
            JobState.CANCEL_REQUESTED,
            JobState.CANCELLED,
        },
        JobState.LEASE_EXPIRED: {JobState.QUEUED, JobState.CLAIMED, JobState.FAILED},
        JobState.WAITING_FOR_USER: {JobState.QUEUED, JobState.RUNNING, JobState.CANCELLED, JobState.FAILED},
        JobState.WAITING_FOR_RESOURCE: {JobState.QUEUED, JobState.RUNNING, JobState.CANCELLED, JobState.FAILED},
        JobState.CANCEL_REQUESTED: {JobState.CANCELLED, JobState.FAILED},
        JobState.COMPLETED: set(),
        JobState.FAILED: set(),
        JobState.CANCELLED: set(),
    }

    @classmethod
    def can_transition(cls, from_state: str, to_state: str) -> bool:
        allowed = cls._VALID_TRANSITIONS.get(from_state, set())
        return to_state in allowed

    @classmethod
    def validate_transition(cls, from_state: str, to_state: str) -> None:
        if not cls.can_transition(from_state, to_state):
            raise ValueError(
                f"Invalid Job transition from '{from_state}' to '{to_state}'. "
                f"Allowed transitions: {cls._VALID_TRANSITIONS.get(from_state, set())}"
            )


class ExecutionStateMachine:
    """Strict transition validator for InvestigationExecution attempts."""

    _VALID_TRANSITIONS: Dict[str, Set[str]] = {
        ExecutionState.CREATED: {ExecutionState.RUNNING, ExecutionState.FAILED, ExecutionState.ABORTED},
        ExecutionState.RUNNING: {
            ExecutionState.CHECKPOINTED,
            ExecutionState.COMPLETED,
            ExecutionState.FAILED,
            ExecutionState.ABORTED,
        },
        ExecutionState.CHECKPOINTED: {
            ExecutionState.RUNNING,
            ExecutionState.COMPLETED,
            ExecutionState.FAILED,
            ExecutionState.ABORTED,
        },
        ExecutionState.COMPLETED: set(),
        ExecutionState.FAILED: set(),
        ExecutionState.ABORTED: set(),
    }

    @classmethod
    def can_transition(cls, from_state: str, to_state: str) -> bool:
        allowed = cls._VALID_TRANSITIONS.get(from_state, set())
        return to_state in allowed

    @classmethod
    def validate_transition(cls, from_state: str, to_state: str) -> None:
        if not cls.can_transition(from_state, to_state):
            raise ValueError(
                f"Invalid Execution transition from '{from_state}' to '{to_state}'. "
                f"Allowed transitions: {cls._VALID_TRANSITIONS.get(from_state, set())}"
            )


# ============================================================================
# CANONICAL TRANSITION ENFORCERS (Single Source of Truth)
# ============================================================================

def transition_investigation(
    db: Session,
    investigation_id: str,
    to_state: str,
    expected_from: Optional[str] = None,
    actor: str = "worker",
    error_code: Optional[str] = None,
    error_message: Optional[str] = None,
) -> Investigation:
    """Enforce state machine transition on an Investigation record and persist audit event."""
    inv = db.query(Investigation).filter(Investigation.id == investigation_id).first()
    if not inv:
        raise ValueError(f"Investigation {investigation_id} not found.")

    current_state = inv.status
    if expected_from and current_state != expected_from:
        raise ValueError(
            f"Precondition failed for Investigation {investigation_id}: expected state '{expected_from}', found '{current_state}'."
        )

    # Validate transition
    InvestigationStateMachine.validate_transition(current_state, to_state)

    inv.status = to_state
    inv.updated_at = datetime.now(timezone.utc)

    # Emit transition event with monotonic sequence cursor
    from sqlalchemy import func
    max_seq = db.query(func.max(InvestigationEvent.sequence)).filter(InvestigationEvent.investigation_id == investigation_id).scalar()
    next_seq = (max_seq or 0) + 1

    event = InvestigationEvent(
        id=f"EVT-{gen_uuid()[:8]}",
        sequence=next_seq,
        investigation_id=investigation_id,
        event_type="investigation.state_transition",
        event_payload_json={
            "from_state": current_state,
            "to_state": to_state,
            "actor": actor,
            "error_code": error_code,
            "error_message": error_message,
        },
        timestamp=datetime.now(timezone.utc),
    )
    db.add(event)
    db.commit()
    return inv


def transition_job(
    db: Session,
    job_id: str,
    to_state: str,
    expected_from: Optional[str] = None,
    worker_id: Optional[str] = None,
    error_code: Optional[str] = None,
    error_message: Optional[str] = None,
) -> InvestigationJob:
    """Enforce state machine transition on a durable InvestigationJob."""
    job = db.query(InvestigationJob).filter(InvestigationJob.id == job_id).first()
    if not job:
        raise ValueError(f"Job {job_id} not found.")

    current_state = job.status
    if expected_from and current_state != expected_from:
        raise ValueError(
            f"Precondition failed for Job {job_id}: expected state '{expected_from}', found '{current_state}'."
        )

    JobStateMachine.validate_transition(current_state, to_state)

    job.status = to_state
    if worker_id is not None:
        job.worker_id = worker_id
    if error_code:
        job.error_code = error_code
    if error_message:
        job.error_message = error_message
    job.updated_at = datetime.now(timezone.utc)
    db.commit()
    return job


def transition_execution(
    db: Session,
    execution_id: str,
    to_state: str,
    expected_from: Optional[str] = None,
    error_code: Optional[str] = None,
    error_message: Optional[str] = None,
) -> InvestigationExecution:
    """Enforce state machine transition on an InvestigationExecution attempt."""
    exec_rec = db.query(InvestigationExecution).filter(InvestigationExecution.id == execution_id).first()
    if not exec_rec:
        raise ValueError(f"Execution {execution_id} not found.")

    current_state = exec_rec.status
    if expected_from and current_state != expected_from:
        raise ValueError(
            f"Precondition failed for Execution {execution_id}: expected state '{expected_from}', found '{current_state}'."
        )

    ExecutionStateMachine.validate_transition(current_state, to_state)

    exec_rec.status = to_state
    if to_state in [ExecutionState.COMPLETED, ExecutionState.FAILED, ExecutionState.ABORTED]:
        exec_rec.completed_at = datetime.now(timezone.utc)
    if error_code:
        exec_rec.error_code = error_code
    if error_message:
        exec_rec.error_message = error_message

    db.commit()
    return exec_rec
