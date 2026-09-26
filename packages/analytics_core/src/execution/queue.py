"""Durable Queue Provider with True Atomic Distributed Claiming and Lease Renewal."""
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
import os
from typing import Any, Dict, List, Optional
from sqlalchemy import text, or_, and_
from sqlalchemy.orm import Session
from apps.api.src.core.database import SessionLocal
from apps.api.src.models.entities import InvestigationJob, Investigation, gen_uuid
from packages.analytics_core.src.execution.state_machine import (
    JobState,
    JobStateMachine,
    transition_job,
)


class BaseQueueProvider(ABC):
    """Abstract interface for durable job queues."""

    @abstractmethod
    def enqueue(self, investigation_id: str, priority: int = 10) -> str:
        pass

    @abstractmethod
    def claim_job(self, worker_id: str, lease_duration_seconds: int = 60) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def renew_lease(self, job_id: str, worker_id: str, lease_duration_seconds: int = 60) -> bool:
        pass

    @abstractmethod
    def complete_job(self, job_id: str, worker_id: str) -> bool:
        pass

    @abstractmethod
    def fail_job(self, job_id: str, worker_id: str, error_code: str, error_message: str) -> bool:
        pass

    @abstractmethod
    def suspend_job(self, job_id: str, worker_id: str, reason: str = "WAITING_FOR_USER") -> bool:
        pass

    @abstractmethod
    def resume_job(self, investigation_id: str) -> bool:
        pass

    @abstractmethod
    def reclaim_stale_jobs(self) -> int:
        pass


class DatabaseQueueProvider(BaseQueueProvider):
    """
    Durable Database Queue Provider guaranteeing atomic distributed claiming across workers
    using atomic conditional updates with rowcount verification.
    """

    def __init__(self, session_factory=SessionLocal):
        self.session_factory = session_factory

    def enqueue(self, investigation_id: str, priority: int = 10) -> str:
        db: Session = self.session_factory()
        try:
            job_id = f"JOB-{gen_uuid()[:8]}"
            job = InvestigationJob(
                id=job_id,
                investigation_id=investigation_id,
                priority=priority,
                status=JobState.QUEUED,
                attempt=1,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            db.add(job)
            db.commit()
            return job_id
        finally:
            db.close()

    def claim_job(self, worker_id: str, lease_duration_seconds: int = 60) -> Optional[Dict[str, Any]]:
        """
        Atomically claim the highest-priority eligible job.
        Uses atomic conditional UPDATE to prevent any race condition under multi-worker concurrency.
        """
        db: Session = self.session_factory()
        try:
            now = datetime.now(timezone.utc)
            lease_expires = now + timedelta(seconds=lease_duration_seconds)

            # 1. Fetch candidates ordered by priority and age
            candidates = (
                db.query(InvestigationJob.id, InvestigationJob.investigation_id, InvestigationJob.attempt)
                .filter(
                    or_(
                        InvestigationJob.status.in_([JobState.QUEUED, JobState.LEASE_EXPIRED]),
                        and_(
                            InvestigationJob.status.in_([JobState.CLAIMED, JobState.RUNNING]),
                            InvestigationJob.lease_expires_at < now,
                        ),
                    )
                )
                .order_by(InvestigationJob.priority.desc(), InvestigationJob.created_at.asc())
                .limit(10)
                .all()
            )

            for cand_id, inv_id, cand_attempt in candidates:
                # 2. Perform Atomic Conditional Update with rowcount check
                stmt = (
                    db.query(InvestigationJob)
                    .filter(
                        InvestigationJob.id == cand_id,
                        or_(
                            InvestigationJob.status == JobState.QUEUED,
                            InvestigationJob.status == JobState.LEASE_EXPIRED,
                            and_(
                                InvestigationJob.status.in_([JobState.CLAIMED, JobState.RUNNING]),
                                InvestigationJob.lease_expires_at < now,
                            ),
                        ),
                    )
                    .update(
                        {
                            "status": JobState.CLAIMED,
                            "worker_id": worker_id,
                            "lease_expires_at": lease_expires,
                            "heartbeat_at": now,
                            "updated_at": now,
                        },
                        synchronize_session=False,
                    )
                )
                db.commit()

                if stmt == 1:
                    # Exactly 1 row updated -> Exclusive lease won by worker_id!
                    return {
                        "job_id": cand_id,
                        "investigation_id": inv_id,
                        "attempt": cand_attempt,
                        "worker_id": worker_id,
                        "lease_expires_at": lease_expires.isoformat(),
                    }

            return None
        except Exception:
            db.rollback()
            return None
        finally:
            db.close()

    def renew_lease(self, job_id: str, worker_id: str, lease_duration_seconds: int = 60) -> bool:
        db: Session = self.session_factory()
        try:
            now = datetime.now(timezone.utc)
            lease_expires = now + timedelta(seconds=lease_duration_seconds)
            rows = (
                db.query(InvestigationJob)
                .filter(
                    InvestigationJob.id == job_id,
                    InvestigationJob.worker_id == worker_id,
                    InvestigationJob.status.in_([JobState.CLAIMED, JobState.RUNNING]),
                )
                .update(
                    {
                        "lease_expires_at": lease_expires,
                        "heartbeat_at": now,
                        "updated_at": now,
                    },
                    synchronize_session=False,
                )
            )
            db.commit()
            return rows == 1
        except Exception:
            db.rollback()
            return False
        finally:
            db.close()

    def complete_job(self, job_id: str, worker_id: str) -> bool:
        db: Session = self.session_factory()
        try:
            transition_job(
                db=db,
                job_id=job_id,
                to_state=JobState.COMPLETED,
                worker_id=worker_id,
            )
            return True
        except Exception:
            db.rollback()
            return False
        finally:
            db.close()

    def fail_job(self, job_id: str, worker_id: str, error_code: str, error_message: str) -> bool:
        db: Session = self.session_factory()
        try:
            job = db.query(InvestigationJob).filter(InvestigationJob.id == job_id).first()
            if not job:
                return False

            non_retryable_codes = ["DATASET_UNAVAILABLE", "CANCELLED", "PROVENANCE_FAILURE", "DATA_QUALITY_FAILURE"]
            target_state = JobState.FAILED if (error_code in non_retryable_codes or job.attempt >= job.max_attempts) else JobState.QUEUED

            transition_job(
                db=db,
                job_id=job_id,
                to_state=target_state,
                worker_id=worker_id,
                error_code=error_code,
                error_message=error_message,
            )
            return True
        except Exception:
            db.rollback()
            return False
        finally:
            db.close()

    def suspend_job(self, job_id: str, worker_id: str, reason: str = "WAITING_FOR_USER") -> bool:
        db: Session = self.session_factory()
        try:
            transition_job(
                db=db,
                job_id=job_id,
                to_state=reason,
                worker_id=worker_id,
            )
            return True
        except Exception:
            db.rollback()
            return False
        finally:
            db.close()

    def resume_job(self, investigation_id: str) -> bool:
        db: Session = self.session_factory()
        try:
            job = db.query(InvestigationJob).filter(
                InvestigationJob.investigation_id == investigation_id,
                InvestigationJob.status.in_([JobState.WAITING_FOR_USER, JobState.WAITING_FOR_RESOURCE]),
            ).first()
            if not job:
                return False

            transition_job(
                db=db,
                job_id=job.id,
                to_state=JobState.QUEUED,
            )
            return True
        except Exception:
            db.rollback()
            return False
        finally:
            db.close()

    def reclaim_stale_jobs(self) -> int:
        """Sweep expired leases and reset jobs to LEASE_EXPIRED / QUEUED for other workers."""
        db: Session = self.session_factory()
        try:
            now = datetime.now(timezone.utc)
            jobs = (
                db.query(InvestigationJob)
                .filter(
                    InvestigationJob.status.in_([JobState.CLAIMED, JobState.RUNNING]),
                    InvestigationJob.lease_expires_at < now,
                )
                .all()
            )
            count = 0
            for j in jobs:
                try:
                    transition_job(
                        db=db,
                        job_id=j.id,
                        to_state=JobState.LEASE_EXPIRED,
                    )
                    count += 1
                except Exception:
                    pass
            return count
        except Exception:
            db.rollback()
            return 0
        finally:
            db.close()


def get_default_queue_provider() -> BaseQueueProvider:
    return DatabaseQueueProvider()
