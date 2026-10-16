"""Durable Asynchronous Investigations API with Distributed Job Queue, Live SSE Replay, and Human Decision Support."""
import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Query, Header, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, model_validator
from sqlalchemy import delete, select, tuple_
from sqlalchemy.orm import Session

from apps.api.src.core.database import get_db, SessionLocal
from apps.api.src.core.security import get_current_user, verify_project_ownership, get_user_authorized_project_ids
from apps.api.src.models.entities import (
    User,
    AnalysisRun,
    AuditLog,
    Investigation,
    InvestigationJob,
    InvestigationExecution,
    InvestigationStepExecution,
    InvestigationObjective,
    Hypothesis,
    Experiment,
    Observation,
    Evidence,
    EvidenceVerification,
    BeliefUpdate,
    InvestigationVerdict,
    InvestigationGraphEdge,
    InvestigationDecisionRequest,
    InvestigationResourceWait,
    InvestigationEvent,
    DecisionRecommendationRecord,
    Dataset,
    DatasetVersion,
    AlertEvent,
    AlertRule,
    Project,
    ClaimGateDecision,
    InvestigationContract,
    InvestigationDataReadiness,
    Assumption,
    Prediction,
    gen_uuid,
)
from packages.analytics_core.src.execution.queue import get_default_queue_provider
from packages.analytics_core.src.engines.local_explanation import LocalExplanationEngine
from packages.analytics_core.src.runtime.cross_dataset_review import build_cross_dataset_review
from packages.analytics_core.src.governance.recommendation_grounding import evaluate_recommendation_grounding
from packages.analytics_core.src.governance.verification_packet import (
    build_verification_packet,
    current_signoff,
    evaluate_review,
    validate_decision,
    validate_signoff,
)
from packages.analytics_core.src.execution.state_machine import (
    InvestigationState,
    JobState,
    transition_investigation,
    transition_job,
)

router = APIRouter(prefix="/investigations", tags=["investigations"])
logger = logging.getLogger(__name__)


class ListFieldPatch(BaseModel):
    add: Optional[List[str]] = None
    remove: Optional[List[str]] = None
    replace: Optional[List[str]] = None

    @model_validator(mode="after")
    def validate_patch(self):
        if self.replace is not None and (self.add is not None or self.remove is not None):
            raise ValueError("'replace' cannot be combined with 'add' or 'remove'.")
        if self.add is None and self.remove is None and self.replace is None:
            raise ValueError("At least one of 'add', 'remove', or 'replace' is required.")
        return self


class InvestigationCreateRequest(BaseModel):
    project_id: str
    question: str
    target_metric: Optional[str] = "revenue"
    priority: Optional[int] = 10
    dataset_ids: Optional[List[str]] = None
    monitoring_event_id: Optional[str] = None
    selected_metric: Optional[str] = None
    selected_aggregation: Optional[str] = None
    selected_group_dimension: Optional[str] = None
    confirmed_assumptions: Optional[Union[List[str], ListFieldPatch]] = None
    scope_filter: Optional[str] = None
    selected_dataset: Optional[str] = None
    clarification_response: Optional[str] = None
    custom_instruction: Optional[str] = None
    # Typed patch against the canonical intent contract, e.g. {"set":{"time_column":"created_at"}}.
    intent_patch: Optional[Dict[str, Any]] = None
    pending_intent_slot: Optional[str] = None
    plan_hash: Optional[str] = None


_GUIDANCE_KEYS = {
    "selected_metric",
    "selected_aggregation",
    "selected_group_dimension",
    "selected_dataset",
    "confirmed_assumptions",
    "scope_filter",
    "clarification_response",
    "custom_instruction",
    "intent_patch",
    "pending_intent_slot",
}


class PreAnalysisInterpretRequest(BaseModel):
    project_id: str
    question: str
    dataset_ids: Optional[List[str]] = None
    # Optional human guidance. When present, the preview is produced by the same resolver that
    # investigation creation uses, so the plan hash returned here is the hash the API will
    # accept for exactly this guidance.
    selected_metric: Optional[str] = None
    selected_aggregation: Optional[str] = None
    selected_group_dimension: Optional[str] = None
    selected_dataset: Optional[str] = None
    confirmed_assumptions: Optional[Union[List[str], ListFieldPatch]] = None
    scope_filter: Optional[str] = None
    clarification_response: Optional[str] = None
    custom_instruction: Optional[str] = None
    intent_patch: Optional[Dict[str, Any]] = None
    pending_intent_slot: Optional[str] = None


class HumanReviewSubmissionRequest(BaseModel):
    action: str  # ACCEPT | REQUEST_CORRECTION | REJECT | REDO
    comments: Optional[str] = None
    modified_parameters: Optional[Dict[str, Any]] = None


class GuidanceRevisionRequest(BaseModel):
    """Patch one or more intent fields before the analytical contract is locked.

    Empty fields are deliberately omitted from the event so a correction to an
    aggregation, for example, cannot erase a previously chosen grouping.
    """
    selected_metric: Optional[str] = None
    selected_aggregation: Optional[str] = None
    selected_group_dimension: Optional[str] = None
    selected_dataset: Optional[str] = None
    confirmed_assumptions: Optional[Union[List[str], ListFieldPatch]] = None
    scope_filter: Optional[str] = None
    clarification_response: Optional[str] = None
    custom_instruction: Optional[str] = None
    intent_patch: Optional[Dict[str, Any]] = None
    pending_intent_slot: Optional[str] = None
    plan_hash: Optional[str] = None


class DecisionResponseRequest(BaseModel):
    user_response: str


class VerificationDecisionRequest(BaseModel):
    item_id: str
    decision: str  # CONFIRMED | REJECTED | NEEDS_REWORK
    comment: Optional[str] = None
    # Fingerprint of the item as the reviewer saw it; if the result changed since
    # they loaded the packet the decision is refused rather than misattributed.
    item_fingerprint: Optional[str] = None


class VerificationSignoffRequest(BaseModel):
    outcome: str  # VERIFIED | REJECTED | NEEDS_REWORK
    comment: Optional[str] = None
    packet_fingerprint: Optional[str] = None


def _dispatch_ready_investigation(
    *,
    db: Session,
    investigation_id: str,
    priority: int,
    actor: str,
    question: str,
    project_id: str,
    created_at: Optional[datetime] = None,
) -> Dict[str, Any]:
    inv = db.query(Investigation).filter(Investigation.id == investigation_id).first()
    if inv is None:
        raise HTTPException(status_code=404, detail="Investigation not found.")
    if inv.status != InvestigationState.READY:
        raise HTTPException(
            status_code=409,
            detail=f"Only READY investigations can be dispatched (current status: {inv.status}).",
        )
    locked_contract = (
        db.query(InvestigationEvent)
        .filter(
            InvestigationEvent.investigation_id == investigation_id,
            InvestigationEvent.event_type == "investigation.canonical_contract_locked",
        )
        .first()
    )
    if locked_contract is None:
        raise HTTPException(status_code=409, detail="Investigation has no locked canonical contract.")

    try:
        job_id = get_default_queue_provider().enqueue(
            investigation_id=investigation_id,
            priority=priority,
            db=db,
        )
        db.add(InvestigationEvent(
            id=f"EVT-{gen_uuid()[:8]}",
            investigation_id=investigation_id,
            event_type="investigation.dispatch_succeeded",
            event_payload_json={"job_id": job_id},
            timestamp=datetime.now(timezone.utc),
        ))
        transition_investigation(
            db=db,
            investigation_id=investigation_id,
            to_state=InvestigationState.QUEUED,
            expected_from=InvestigationState.READY,
            actor=actor,
        )
        return {
            "investigation_id": investigation_id,
            "job_id": job_id,
            "status": InvestigationState.QUEUED,
            "question": question,
            "project_id": project_id,
            "created_at": (created_at or inv.created_at).isoformat() if (created_at or inv.created_at) else None,
        }
    except Exception as exc:
        db.rollback()
        logger.exception("Failed to dispatch READY investigation %s", investigation_id)
        current_inv = (
            db.query(Investigation)
            .filter(Investigation.id == investigation_id)
            .first()
        )
        if current_inv and current_inv.status in {
            InvestigationState.QUEUED,
            InvestigationState.CLAIMED,
            InvestigationState.RUNNING,
        }:
            active_job = (
                db.query(InvestigationJob)
                .filter(
                    InvestigationJob.investigation_id == investigation_id,
                    InvestigationJob.status.in_(
                        [JobState.QUEUED, JobState.CLAIMED, JobState.RUNNING]
                    ),
                )
                .order_by(InvestigationJob.created_at.desc())
                .first()
            )
            return {
                "investigation_id": investigation_id,
                "job_id": active_job.id if active_job else None,
                "status": current_inv.status,
                "question": question,
                "project_id": project_id,
                "created_at": (created_at or current_inv.created_at).isoformat()
                if (created_at or current_inv.created_at)
                else None,
            }
        db.add(InvestigationEvent(
            id=f"EVT-{gen_uuid()[:8]}",
            investigation_id=investigation_id,
            event_type="investigation.dispatch_failed",
            event_payload_json={"error": f"{type(exc).__name__}: {exc}"},
            timestamp=datetime.now(timezone.utc),
        ))
        db.commit()
        return {
            "investigation_id": investigation_id,
            "job_id": None,
            "status": InvestigationState.READY,
            "question": question,
            "project_id": project_id,
            "created_at": (created_at or inv.created_at).isoformat() if (created_at or inv.created_at) else None,
            "dispatch_error": f"{type(exc).__name__}: {exc}",
            "retry_endpoint": f"/investigations/{investigation_id}/dispatch",
        }


@router.post("")
def create_investigation(
    payload: InvestigationCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Create a new durable investigation and enqueue it to the QueueProvider.
    Returns immediately with investigation_id, job_id, and status: 'QUEUED'.
    Worker processes claim and execute the job completely asynchronously.
    """
    verify_project_ownership(payload.project_id, current_user, db)

    # Explicit multi-dataset scope is fail-closed: every requested dataset must
    # exist in the selected project, and duplicate IDs are rejected. The
    # runtime must never silently drop a requested dataset and continue with a
    # narrower population than the analyst selected.
    requested_dataset_ids = list(payload.dataset_ids or [])
    monitoring_event = None
    monitoring_context = None
    analysis_question = payload.question
    pinned_dataset_version_id = None
    if payload.monitoring_event_id:
        monitoring_event = (
            db.query(AlertEvent)
            .join(AlertRule, AlertEvent.rule_id == AlertRule.id)
            .filter(
                AlertEvent.id == payload.monitoring_event_id,
                AlertEvent.project_id == payload.project_id,
                AlertRule.project_id == payload.project_id,
            )
            .first()
        )
        if monitoring_event is None:
            raise HTTPException(status_code=404, detail="Selected monitoring event is unavailable.")
        if not monitoring_event.dataset_id or not monitoring_event.dataset_version_id:
            raise HTTPException(status_code=409, detail="The selected alert has no recorded dataset-version scope.")
        source_dataset = db.query(Dataset).filter(
            Dataset.id == monitoring_event.dataset_id,
            Dataset.project_id == payload.project_id,
        ).first()
        source_version = db.query(DatasetVersion).filter(
            DatasetVersion.id == monitoring_event.dataset_version_id,
            DatasetVersion.dataset_id == monitoring_event.dataset_id,
            DatasetVersion.version_number == monitoring_event.dataset_version,
        ).first()
        if (
            source_dataset is None
            or source_version is None
            or source_dataset.current_version != monitoring_event.dataset_version
            or (
                monitoring_event.dataset_fingerprint
                and source_version.content_hash != monitoring_event.dataset_fingerprint
            )
        ):
            raise HTTPException(
                status_code=409,
                detail="The alert's exact source snapshot is unavailable or no longer current. Restore that snapshot before investigating; another version will not be substituted.",
            )
        event_dataset_ids = [monitoring_event.dataset_id]
        if requested_dataset_ids and requested_dataset_ids != event_dataset_ids:
            raise HTTPException(status_code=400, detail="Monitoring follow-up scope must remain the selected event's dataset.")
        requested_dataset_ids = event_dataset_ids
        pinned_dataset_version_id = source_version.id
        monitoring_context = {
            "event_id": monitoring_event.id,
            "dataset_id": source_dataset.id,
            "dataset_version": source_version.version_number,
            "dataset_version_id": source_version.id,
            "dataset_fingerprint": source_version.content_hash,
            "metric": monitoring_event.metric_name,
            "baseline_value": monitoring_event.expected_value,
            "observed_value": monitoring_event.current_value,
            "deviation_percentage": monitoring_event.deviation_percentage,
            "threshold_pct_change": db.query(AlertRule.threshold_pct_change)
                .filter(AlertRule.id == monitoring_event.rule_id).scalar(),
            "evidence_summary": monitoring_event.evidence_summary,
        }
        analysis_question = (
            f"{payload.question.strip()}\n\n"
            "Selected persisted monitoring event evidence (use only as recorded context; "
            "do not assert causation or infer unrecorded facts): "
            f"{json.dumps(monitoring_context, sort_keys=True, separators=(',', ':'))}"
        )
    if len(requested_dataset_ids) != len(set(requested_dataset_ids)):
        raise HTTPException(status_code=400, detail="Duplicate dataset IDs are not allowed in an investigation scope.")
    if requested_dataset_ids:
        found_rows = db.query(Dataset.id, Dataset.name).filter(
            Dataset.project_id == payload.project_id,
            Dataset.id.in_(requested_dataset_ids),
        ).all()
        found_ids = {row[0] for row in found_rows}
        missing = [dataset_id for dataset_id in requested_dataset_ids if dataset_id not in found_ids]
        if missing:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "DATASET_SCOPE_INVALID",
                    "message": "One or more requested datasets are unavailable in this project.",
                    "missing_dataset_ids": missing,
                },
            )
        names_to_ids: dict[str, list[str]] = {}
        for dataset_id, dataset_name in found_rows:
            names_to_ids.setdefault(str(dataset_name), []).append(str(dataset_id))
        duplicate_names = {name: ids for name, ids in names_to_ids.items() if len(ids) > 1}
        if duplicate_names:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "DATASET_SCOPE_AMBIGUOUS",
                    "message": "Selected datasets contain duplicate display names. Rename the datasets or select only one of each duplicate name; AA-OS refuses to collapse two physical datasets into one table.",
                    "duplicate_names": duplicate_names,
                },
            )

    # Resolve guidance before creating or queuing executable work.
    from packages.analytics_core.src.governance.guided_investigation import (
        ResolverOutcome,
        ResolverStatus,
        apply_human_guidance_to_contract,
        normalize_guidance_payload,
    )
    if payload.selected_aggregation:
        normalized_aggregation = payload.selected_aggregation.strip().upper()
        if normalized_aggregation == "AVG":
            normalized_aggregation = "MEAN"
        if normalized_aggregation not in {
            "SUM", "MEAN", "COUNT", "COUNT_DISTINCT", "MEDIAN", "MIN", "MAX",
            "ROW_COUNT", "AVERAGE",
        }:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported aggregation '{payload.selected_aggregation}'.",
            )

    from packages.analytics_core.src.engines.dataset_provider import DatabaseDatasetProvider
    ds_provider = DatabaseDatasetProvider(session_factory=SessionLocal)
    ds_context = ds_provider.acquire_context(payload.project_id, requested_dataset_ids)

    guidance = {
        key: value
        for key, value in payload.model_dump(exclude_unset=True).items()
        if key in {
            "selected_metric",
            "selected_aggregation",
            "selected_group_dimension",
            "selected_dataset",
            "confirmed_assumptions",
            "scope_filter",
            "clarification_response",
            "custom_instruction",
            "intent_patch",
            "pending_intent_slot",
        }
        and value is not None
    }
    try:
        guidance = normalize_guidance_payload(guidance)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid intent guidance: {exc}") from exc

    if payload.selected_dataset:
        selected_dataset = payload.selected_dataset.strip()
        dataset_name = next(
            (
                name
                for name, metadata in ds_context.dataset_versions.items()
                if str(metadata.get("dataset_id")) == selected_dataset
            ),
            selected_dataset,
        )
        guidance["selected_dataset"] = dataset_name

    if ds_context.load_errors or ds_context.unavailable_requested_ids or ds_context.scope_errors:
        resolution = ResolverOutcome(
            status=ResolverStatus.NEEDS_CLARIFICATION,
            unresolved_instruction=payload.question,
            reason=(
                "The complete dataset scope could not be loaded for pre-execution contract resolution. "
                "Restore the selected dataset snapshot and retry; no work was queued."
            ),
            schema_information={
                "load_errors": dict(ds_context.load_errors),
                "unavailable_requested_ids": list(ds_context.unavailable_requested_ids),
                "scope_errors": list(ds_context.scope_errors),
            },
        )
    else:
        resolution = apply_human_guidance_to_contract(
            question=payload.question,
            data_context=ds_context,
            guidance=guidance,
        )

    investigation_id = f"INV-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{gen_uuid()[:8]}"
    guidance_event_payload = {
        key: value
        for key, value in guidance.items()
        if key in {
            "selected_metric",
            "selected_aggregation",
            "selected_group_dimension",
            "selected_dataset",
            "confirmed_assumptions",
            "scope_filter",
            "clarification_response",
            "custom_instruction",
            "intent_patch",
            "pending_intent_slot",
        }
        and value not in (None, "", [])
    }
    inv = Investigation(
        id=investigation_id,
        project_id=payload.project_id,
        user_id=current_user.id,
        question=analysis_question,
        status=InvestigationState.PLANNED,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        requested_dataset_ids_json=requested_dataset_ids or None,
        dataset_version_ids_json=[pinned_dataset_version_id] if pinned_dataset_version_id else [],
        parent_investigation_id=None,
        is_subinvestigation=False,
    )
    db.add(inv)
    db.add(InvestigationEvent(
        id=f"EVT-{gen_uuid()[:8]}",
        investigation_id=investigation_id,
        event_type="investigation.created",
        event_payload_json={
            "question": payload.question,
            "project_id": payload.project_id,
            "monitoring_event_id": payload.monitoring_event_id,
            "monitoring_evidence": monitoring_context,
        },
        timestamp=datetime.now(timezone.utc),
    ))
    if guidance_event_payload:
        db.add(InvestigationEvent(
            id=f"EVT-{gen_uuid()[:8]}",
            investigation_id=investigation_id,
            event_type="investigation.human_guidance_intent",
            event_payload_json=guidance_event_payload,
            timestamp=datetime.now(timezone.utc),
        ))
    db.add(InvestigationEvent(
        id=f"EVT-{gen_uuid()[:8]}",
        investigation_id=investigation_id,
        event_type="investigation.guidance_resolution",
        event_payload_json=resolution.to_dict(),
        timestamp=datetime.now(timezone.utc),
    ))
    db.flush()

    if resolution.status == ResolverStatus.RESOLVED:
        if resolution.contract is None:
            raise HTTPException(status_code=500, detail="Resolved guidance did not produce a canonical contract.")
        selected_dataset_name = resolution.schema_information.get("dataset")
        dataset_fp = ds_context.dataset_fingerprints.get(selected_dataset_name)
        from packages.analytics_core.src.governance.guided_investigation import compute_resolved_plan_hash
        current_plan_hash = compute_resolved_plan_hash(
            contract=resolution.contract,
            dataset_fingerprint=dataset_fp,
            schema_version=1,
        )
        if not payload.plan_hash:
            db.rollback()
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "PLAN_APPROVAL_REQUIRED",
                    "message": "Explicit plan approval is required to execute an investigation. Inspect and approve the plan preview.",
                    "current_plan_hash": current_plan_hash,
                    "resolution": resolution.to_dict(),
                },
            )
        if payload.plan_hash != current_plan_hash:
            db.rollback()
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "STALE_PLAN_APPROVAL",
                    "message": "The approved plan hash does not match the current resolved plan. Refresh the plan preview before confirming execution.",
                    "submitted_plan_hash": payload.plan_hash,
                    "current_plan_hash": current_plan_hash,
                    "resolution": resolution.to_dict(),
                },
            )
        db.add(InvestigationEvent(
            id=f"EVT-{gen_uuid()[:8]}",
            investigation_id=investigation_id,
            event_type="investigation.canonical_contract_locked",
            event_payload_json={
                "version": 1,
                "contract": resolution.contract.to_dict(),
                "contract_hash": resolution.contract.contract_hash(),
                "plan_hash": current_plan_hash,
                "guidance": guidance,
                "dataset_name": selected_dataset_name,
                "dataset_metadata": ds_context.dataset_versions.get(selected_dataset_name, {}),
                "dataset_fingerprint": dataset_fp,
                "locked_at": datetime.now(timezone.utc).isoformat(),
            },
            timestamp=datetime.now(timezone.utc),
        ))
        transition_investigation(
            db=db,
            investigation_id=investigation_id,
            to_state=InvestigationState.READY,
            expected_from=InvestigationState.PLANNED,
            actor=current_user.email,
        )
        return _dispatch_ready_investigation(
            db=db,
            investigation_id=investigation_id,
            priority=payload.priority or 10,
            actor=current_user.email,
            question=payload.question,
            project_id=payload.project_id,
            created_at=inv.created_at,
        )

    transition_investigation(
        db=db,
        investigation_id=investigation_id,
        to_state=InvestigationState.NEEDS_CLARIFICATION,
        expected_from=InvestigationState.PLANNED,
        actor=current_user.email,
    )

    return {
        "investigation_id": investigation_id,
        "job_id": None,
        "status": InvestigationState.NEEDS_CLARIFICATION,
        "question": payload.question,
        "project_id": payload.project_id,
        "created_at": inv.created_at.isoformat(),
        "resolution": resolution.to_dict(),
    }


@router.post("/{investigation_id}/dispatch")
def retry_investigation_dispatch(
    investigation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    inv = db.query(Investigation).filter(Investigation.id == investigation_id).first()
    if inv is None:
        raise HTTPException(status_code=404, detail="Investigation not found.")
    verify_project_ownership(inv.project_id, current_user, db)
    return _dispatch_ready_investigation(
        db=db,
        investigation_id=investigation_id,
        priority=10,
        actor=current_user.email,
        question=inv.question,
        project_id=inv.project_id,
        created_at=inv.created_at,
    )


@router.get("")
def list_investigations(
    project_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List investigations isolated strictly to authorized user projects."""
    if project_id:
        verify_project_ownership(project_id, current_user, db)
        allowed_ids = [project_id]
    else:
        allowed_ids = get_user_authorized_project_ids(current_user, db)

    query = db.query(Investigation).filter(
        Investigation.project_id.in_(allowed_ids),
        Investigation.is_subinvestigation.is_(False),
    )
    if status:
        query = query.filter(Investigation.status == status)

    invs = query.order_by(Investigation.created_at.desc()).limit(limit).all()

    return [
        {
            "id": i.id,
            "project_id": i.project_id,
            "question": i.question,
            "status": i.status,
            "verdict_type": i.verdict_type,
            "confidence_score": i.confidence_score,
            "direct_answer": i.direct_answer,
            "main_finding": i.main_finding,
            "reproducible_manifest_hash": i.reproducible_manifest_hash,
            "created_at": i.created_at.isoformat() if i.created_at else None,
            "updated_at": i.updated_at.isoformat() if i.updated_at else None,
            "analysis_plan": (
                (db.query(InvestigationEvent)
                 .filter(InvestigationEvent.investigation_id == i.id, InvestigationEvent.event_type == "investigation.universal_analysis_plan")
                 .order_by(InvestigationEvent.sequence.desc()).first().event_payload_json)
                if db.query(InvestigationEvent).filter(InvestigationEvent.investigation_id == i.id, InvestigationEvent.event_type == "investigation.universal_analysis_plan").first() else None
            ),
        }
        for i in invs
    ]


@router.get("/history")
def list_analysis_history(
    limit: int = Query(default=100, ge=10, le=1000),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return persisted analysis and project records without rerunning calculations."""
    allowed_ids = get_user_authorized_project_ids(current_user, db)
    projects = db.query(Project).filter(Project.id.in_(allowed_ids)).all() if allowed_ids else []
    project_names = {project.id: project.name for project in projects}

    investigations = (
        db.query(Investigation)
        .filter(
            Investigation.project_id.in_(allowed_ids),
            Investigation.is_subinvestigation.is_(False),
        )
        .order_by(Investigation.created_at.desc())
        .limit(limit)
        .all()
        if allowed_ids else []
    )
    investigation_ids = [row.id for row in investigations]
    contracts_by_investigation = {}
    if investigation_ids:
        contracts = (
            db.query(InvestigationContract)
            .filter(InvestigationContract.investigation_id.in_(investigation_ids))
            .order_by(InvestigationContract.version.desc())
            .all()
        )
        for contract in contracts:
            contracts_by_investigation.setdefault(contract.investigation_id, contract)

    evidence_by_investigation: dict[str, list[Evidence]] = {}
    if investigation_ids:
        evidence_rows = (
            db.query(Evidence)
            .filter(Evidence.investigation_id.in_(investigation_ids))
            .all()
        )
        for row in evidence_rows:
            evidence_by_investigation.setdefault(row.investigation_id, []).append(row)
    all_evidence_ids = [row.id for rows in evidence_by_investigation.values() for row in rows]
    verification_statuses_by_evidence: dict[str, list[str]] = {}
    verification_details_by_evidence: dict[str, list[dict[str, Any]]] = {}
    if all_evidence_ids:
        for verification in db.query(EvidenceVerification).filter(
            EvidenceVerification.evidence_id.in_(all_evidence_ids)
        ).all():
            verification_statuses_by_evidence.setdefault(verification.evidence_id, []).append(
                str(verification.status or "UNVERIFIED").upper()
            )
            verification_details_by_evidence.setdefault(verification.evidence_id, []).append({
                "status": str(verification.status or "UNVERIFIED").upper(),
                "primary_tool": verification.primary_tool,
                "secondary_tool": verification.secondary_tool,
                "tolerance_threshold": verification.tolerance_threshold,
                "observed_delta_pct": verification.observed_delta_pct,
                "is_deterministic": verification.is_deterministic,
                "validator_fingerprint": verification.validator_fingerprint,
                "created_at": verification.created_at.isoformat() if verification.created_at else None,
            })

    experiment_ids = {
        evidence.experiment_id
        for rows in evidence_by_investigation.values()
        for evidence in rows
        if evidence.experiment_id
    }
    observations_by_experiment = {}
    if experiment_ids:
        observations = db.query(Observation).filter(
            Observation.experiment_id.in_(experiment_ids)
        ).all()
        observations_by_experiment = {row.experiment_id: row for row in observations}

    gate_rows_by_investigation: dict[str, ClaimGateDecision] = {}
    if investigation_ids:
        gate_rows = (
            db.query(ClaimGateDecision)
            .filter(ClaimGateDecision.investigation_id.in_(investigation_ids))
            .order_by(ClaimGateDecision.created_at.desc())
            .all()
        )
        for row in gate_rows:
            gate_rows_by_investigation.setdefault(row.investigation_id, row)

    source_dataset_ids = set()
    for contract in contracts_by_investigation.values():
        population = contract.population_json or {}
        identities = population.get("dataset_identities", {}) if isinstance(population, dict) else {}
        for metadata in identities.values() if isinstance(identities, dict) else []:
            if isinstance(metadata, dict) and metadata.get("dataset_id"):
                source_dataset_ids.add(str(metadata["dataset_id"]))
    active_datasets = (
        db.query(Dataset).filter(Dataset.id.in_(source_dataset_ids)).all()
        if source_dataset_ids else []
    )
    active_by_id = {dataset.id: dataset for dataset in active_datasets}
    source_descriptors_by_investigation: dict[str, list[dict[str, Any]]] = {}

    def _source_descriptors(investigation: Investigation) -> list[dict[str, Any]]:
        contract = contracts_by_investigation.get(investigation.id)
        population = contract.population_json or {} if contract else {}
        identities = population.get("dataset_identities", {}) if isinstance(population, dict) else {}
        descriptors = []
        for name, metadata in identities.items() if isinstance(identities, dict) else []:
            if not isinstance(metadata, dict):
                continue
            dataset_id = str(metadata.get("dataset_id") or "")
            snapshot_version = metadata.get("dataset_version")
            active = active_by_id.get(dataset_id)
            try:
                exact_version_available = bool(
                    active
                    and snapshot_version is not None
                    and int(active.current_version or 0) == int(snapshot_version)
                )
            except (TypeError, ValueError):
                exact_version_available = False
            descriptors.append({
                "dataset_id": dataset_id or None,
                "name": str(name),
                "dataset_version": snapshot_version,
                "dataset_version_id": metadata.get("dataset_version_id"),
                "fingerprint": metadata.get("fingerprint"),
                "available": exact_version_available,
            })
        if not descriptors:
            descriptors = [
                {
                    "dataset_id": str(dataset_id),
                    "name": str(dataset_id),
                    "dataset_version": None,
                    "dataset_version_id": None,
                    "fingerprint": None,
                    "available": False,
                }
                for dataset_id in (investigation.requested_dataset_ids_json or [])
            ]
        return descriptors

    records = []
    for investigation in investigations:
        evidence_rows = evidence_by_investigation.get(investigation.id, [])
        statuses = [
            status
            for evidence in evidence_rows
            for status in verification_statuses_by_evidence.get(evidence.id, [])
        ]
        if any(status in {"FAILED", "MISMATCH", "MISMATCHED"} for status in statuses) or any(
            str(evidence.validation_status or "").lower() == "failed" for evidence in evidence_rows
        ):
            verification_status = "FAILED"
        elif evidence_rows and all(
            str(evidence.validation_status or "").lower() == "verified"
            and any(status in {"PASSED", "VERIFIED"} for status in verification_statuses_by_evidence.get(evidence.id, []))
            for evidence in evidence_rows
        ):
            verification_status = "VERIFIED"
        else:
            verification_status = "UNVERIFIED"

        answer_event = (
            db.query(InvestigationEvent)
            .filter(
                InvestigationEvent.investigation_id == investigation.id,
                InvestigationEvent.event_type == "investigation.analyst_result",
            )
            .order_by(InvestigationEvent.sequence.desc())
            .first()
        )
        analyst_result = (answer_event.event_payload_json or {}) if answer_event else None
        gate = gate_rows_by_investigation.get(investigation.id)
        records.append({
            "id": investigation.id,
            "record_type": "investigation",
            "project_id": investigation.project_id,
            "project_name": project_names.get(investigation.project_id, investigation.project_id),
            "question": investigation.question,
            "created_at": investigation.created_at.isoformat() if investigation.created_at else None,
            "updated_at": investigation.updated_at.isoformat() if investigation.updated_at else None,
            "status": investigation.status,
            "verdict_type": investigation.verdict_type,
            "confidence_score": investigation.confidence_score,
            "headline": (analyst_result or {}).get("headline") or investigation.direct_answer,
            "direct_answer": investigation.direct_answer,
            "analyst_result": analyst_result,
            "verification_status": verification_status,
            "claim_status": gate.outcome if gate else "NOT_RECORDED",
            "source_datasets": _source_descriptors(investigation),
            "evidence_ids": [evidence.id for evidence in evidence_rows],
            "evidence": [
                {
                    "id": evidence.id,
                    "statement": evidence.statement,
                    "evidence_type": evidence.evidence_type,
                    "validation_status": evidence.validation_status,
                    "confidence_score": evidence.confidence_score,
                    "calculation_summary": evidence.calculation_summary,
                    "statistical_test_name": evidence.statistical_test_name,
                    "p_value": evidence.p_value,
                    "effect_size": evidence.effect_size,
                    "experiment_id": evidence.experiment_id,
                    "row_count_analyzed": (
                        observations_by_experiment[evidence.experiment_id].row_count_analyzed
                        if evidence.experiment_id in observations_by_experiment else None
                    ),
                    "sql_executed": (
                        observations_by_experiment[evidence.experiment_id].sql_executed
                        if evidence.experiment_id in observations_by_experiment else None
                    ),
                    "calculation_trace": (
                        (observations_by_experiment[evidence.experiment_id].result_json or {}).get("calculation_trace")
                        if evidence.experiment_id in observations_by_experiment else None
                    ),
                    "verification_records": verification_details_by_evidence.get(evidence.id, []),
                }
                for evidence in evidence_rows
            ],
        })

    investigation_id_set = set(investigation_ids)
    legacy_runs = (
        db.query(AnalysisRun)
        .filter(AnalysisRun.project_id.in_(allowed_ids))
        .order_by(AnalysisRun.created_at.desc())
        .limit(limit)
        .all()
        if allowed_ids else []
    )
    for legacy in legacy_runs:
        if legacy.id in investigation_id_set:
            continue
        records.append({
            "id": legacy.id,
            "record_type": "legacy_analysis",
            "project_id": legacy.project_id,
            "project_name": project_names.get(legacy.project_id, legacy.project_id),
            "question": legacy.question,
            "created_at": legacy.created_at.isoformat() if legacy.created_at else None,
            "updated_at": legacy.updated_at.isoformat() if legacy.updated_at else None,
            "status": legacy.status,
            "verdict_type": None,
            "confidence_score": None,
            "headline": legacy.direct_answer,
            "direct_answer": legacy.direct_answer,
            "analyst_result": None,
            "saved_result": {
                "main_finding": legacy.main_finding,
                "hypotheses": legacy.hypotheses_json or [],
                "steps": legacy.steps_json or [],
                "findings": legacy.findings_json or [],
                "evidence": legacy.evidence_json or [],
                "manifest": legacy.manifest_json,
            },
            "verification_status": "NOT_RECORDED",
            "claim_status": "NOT_RECORDED",
            "source_datasets": [],
            "evidence_ids": [],
            "evidence": legacy.evidence_json or [],
        })

    records.sort(key=lambda row: row.get("created_at") or "", reverse=True)
    return {"records": records[:limit], "projects": [
        {"id": project.id, "name": project.name, "description": project.description}
        for project in projects
    ]}


@router.delete("/history")
def clear_analysis_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete only the caller's saved analysis records, not projects or datasets."""
    allowed_ids = get_user_authorized_project_ids(current_user, db)
    if not allowed_ids:
        return {"status": "success", "removed_investigations": 0, "removed_legacy_analyses": 0}

    investigation_ids = [
        row[0]
        for row in db.query(Investigation.id).filter(Investigation.project_id.in_(allowed_ids)).all()
    ]
    legacy_query = db.query(AnalysisRun).filter(AnalysisRun.project_id.in_(allowed_ids))
    legacy_count = legacy_query.count()

    if investigation_ids:
        metadata = Investigation.__table__.metadata
        root_table = Investigation.__table__
        root_keys = {(str(value),) for value in investigation_ids}
        scoped_keys: dict[Any, set[tuple[Any, ...]]] = {root_table: root_keys}
        table_order = [root_table]
        pending_tables = [root_table]

        while pending_tables:
            parent_table = pending_tables.pop(0)
            parent_primary_key = list(parent_table.primary_key.columns)
            parent_rows = scoped_keys[parent_table]
            for foreign_table in metadata.tables.values():
                child_primary_key = list(foreign_table.primary_key.columns)
                for foreign_key in foreign_table.foreign_keys:
                    if foreign_key.column.table is not parent_table:
                        continue
                    parent_index = parent_primary_key.index(foreign_key.column)
                    parent_values = {row[parent_index] for row in parent_rows}
                    if not parent_values:
                        continue
                    matching_rows = db.execute(
                        select(*child_primary_key).where(foreign_key.parent.in_(parent_values))
                    ).all()
                    if not matching_rows:
                        continue
                    child_keys = {tuple(row) for row in matching_rows}
                    existing_keys = scoped_keys.setdefault(foreign_table, set())
                    new_keys = child_keys - existing_keys
                    if new_keys:
                        existing_keys.update(new_keys)
                        if foreign_table not in table_order:
                            table_order.append(foreign_table)
                        pending_tables.append(foreign_table)

        try:
            for table in reversed(table_order):
                keys = scoped_keys[table]
                primary_key = list(table.primary_key.columns)
                if len(primary_key) == 1:
                    predicate = primary_key[0].in_({row[0] for row in keys})
                else:
                    predicate = tuple_(*primary_key).in_(keys)
                db.execute(delete(table).where(predicate))
            legacy_query.delete(synchronize_session=False)
            db.commit()
        except Exception as exc:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"History could not be cleared: {exc}") from exc
    else:
        try:
            legacy_query.delete(synchronize_session=False)
            db.commit()
        except Exception as exc:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"History could not be cleared: {exc}") from exc

    return {
        "status": "success",
        "removed_investigations": len(investigation_ids),
        "removed_legacy_analyses": legacy_count,
        "projects_and_datasets_preserved": True,
    }


@router.get("/{investigation_id}")
def get_investigation(
    investigation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve full durable investigation state, DAG nodes, evidence, and active execution."""
    inv = db.query(Investigation).filter(Investigation.id == investigation_id).first()
    if not inv:
        raise HTTPException(status_code=404, detail="Investigation not found.")

    verify_project_ownership(inv.project_id, current_user, db)

    objectives = db.query(InvestigationObjective).filter(InvestigationObjective.investigation_id == inv.id).all()
    hyps = db.query(Hypothesis).filter(Hypothesis.investigation_id == inv.id).all()
    exps = db.query(Experiment).filter(Experiment.investigation_id == inv.id).all()
    evid = db.query(Evidence).filter(Evidence.investigation_id == inv.id).all()
    verdict = db.query(InvestigationVerdict).filter(InvestigationVerdict.investigation_id == inv.id).first()
    observations_by_experiment = {o.experiment_id: o for o in db.query(Observation).filter(Observation.experiment_id.in_([e.id for e in exps])).all()} if exps else {}
    experiments_by_id = {e.id: e for e in exps}
    decision_recs = db.query(DecisionRecommendationRecord).filter(DecisionRecommendationRecord.investigation_id == inv.id).all()
    hypothesis_by_code = {str(h.hypothesis_code): h for h in hyps}
    evidence_ids_by_hypothesis = {}
    for ev in evid:
        if ev.hypothesis_id:
            evidence_ids_by_hypothesis.setdefault(str(ev.hypothesis_id), []).append(str(ev.id))
    assumptions = db.query(Assumption).filter(Assumption.investigation_id == inv.id).all()
    predictions = db.query(Prediction).filter(Prediction.investigation_id == inv.id).order_by(Prediction.created_at.asc()).all()
    belief_updates = db.query(BeliefUpdate).filter(BeliefUpdate.investigation_id == inv.id).order_by(BeliefUpdate.update_step_index.asc(), BeliefUpdate.created_at.asc()).all()
    evidence_verifications = (
        db.query(EvidenceVerification)
        .filter(EvidenceVerification.evidence_id.in_([e.id for e in evid]))
        .order_by(EvidenceVerification.created_at.asc())
        .all()
    ) if evid else []
    event_rows = (
        db.query(InvestigationEvent)
        .filter(InvestigationEvent.investigation_id == inv.id)
        .order_by(InvestigationEvent.sequence.asc())
        .all()
    )
    latest_verification_status_by_id = {}
    for _vr in evidence_verifications:
        latest_verification_status_by_id[str(_vr.evidence_id)] = str(getattr(_vr, "status", ""))

    recommendation_grounding_events = {}
    for _event in event_rows:
        if _event.event_type == "decision.recommendation.grounding_evaluated":
            _payload = _event.event_payload_json or {}
            _recommendation_id = str(_payload.get("recommendation_id") or "")
            _hypothesis_id = str(_payload.get("grounded_hypothesis_id") or "")
            if _recommendation_id:
                recommendation_grounding_events[_recommendation_id] = _payload
            elif _hypothesis_id:
                # Backward-compatible lookup for pre-fix events. New writes
                # always carry a recommendation_id and are bound one-to-one.
                recommendation_grounding_events[_hypothesis_id] = _payload

    def _latest_payload(prefixes):
        for event in reversed(event_rows):
            if any(event.event_type == p or event.event_type.startswith(p) for p in prefixes):
                return event.event_payload_json or {}
        return None

    epistemic_payload = _latest_payload(["calculation.epistemic_calibration"])
    missingness_payload = _latest_payload(["investigation.missingness_sensitivity"])
    causal_gate_payload = _latest_payload(["investigation.causal_gate.evaluated", "investigation.causal_gate.fallback"])
    causal_effect_payload = _latest_payload(["investigation.causal_effect.estimated", "investigation.causal_effect.estimation_failed"])
    adversarial_payloads = [
        e.event_payload_json or {} for e in event_rows
        if e.event_type.startswith("investigation.adversarial_challenge")
        or e.event_type.startswith("calculation.adversarial")
    ]
    multiverse_payload = _latest_payload(["investigation.multiverse.skipped", "calculation.multiverse"])
    manifest_payload = _latest_payload(["calculation.provenance_manifest"])
    phase_semantic_payload = _latest_payload(["investigation.phase.semantic_model"])
    analyst_result_payload = _latest_payload(["investigation.analyst_result"])
    # `plan_payload` was dead code that referenced `analysis_plan` before it was
    # assigned later in this function (undefined-name crash, pyflakes-confirmed);
    # it was never actually consumed downstream, so it is removed rather than
    # reordered. `analysis_plan` itself remains defined and used below.
    latest_contract = (
        db.query(InvestigationContract)
        .filter(InvestigationContract.investigation_id == inv.id)
        .order_by(InvestigationContract.version.desc())
        .first()
    )
    # Session 8 (DEFECT-022): ``primary_dataset_name`` was referenced in the
    # response below but never defined in this function, so every
    # GET /investigations/{id} raised NameError -- the endpoint the analyst UI
    # polls for every result. Resolve it the same way /explanation does.
    scope_payload = _latest_payload(["investigation.dataset_scope_resolved"]) or {}
    primary_dataset_name = scope_payload.get("primary_dataset")
    readiness_rows = (
        db.query(InvestigationDataReadiness)
        .filter(InvestigationDataReadiness.investigation_id == inv.id)
        .order_by(InvestigationDataReadiness.dataset_name.asc(), InvestigationDataReadiness.assessed_at.desc())
        .all()
    )
    latest_readiness_by_dataset = {}
    for _row in readiness_rows:
        latest_readiness_by_dataset.setdefault(_row.dataset_name, _row)
    readiness = latest_readiness_by_dataset.get(primary_dataset_name)
    if readiness is None and latest_readiness_by_dataset:
        # Never let alphabetical ordering pick a misleading non-primary dataset
        # when the scope event is absent (older investigations).
        readiness = next(iter(latest_readiness_by_dataset.values()))
    high_risk_assumptions = [a for a in assumptions if a.sensitivity_risk == "high" and not a.is_validated]
    edges = db.query(InvestigationGraphEdge).filter(InvestigationGraphEdge.investigation_id == inv.id).all()
    pending_decision = (
        db.query(InvestigationDecisionRequest)
        .filter(
            InvestigationDecisionRequest.investigation_id == inv.id,
            InvestigationDecisionRequest.status == "PENDING",
        )
        .first()
    )
    pending_resource = (
        db.query(InvestigationResourceWait)
        .filter(
            InvestigationResourceWait.investigation_id == inv.id,
            InvestigationResourceWait.status == "PENDING",
        )
        .first()
    )

    claim_gate_row = (
        db.query(ClaimGateDecision)
        .filter(ClaimGateDecision.investigation_id == inv.id)
        .order_by(ClaimGateDecision.created_at.desc())
        .first()
    )

    plan_event = (
        db.query(InvestigationEvent)
        .filter(
            InvestigationEvent.investigation_id == inv.id,
            InvestigationEvent.event_type == "investigation.universal_analysis_plan",
        )
        .order_by(InvestigationEvent.sequence.desc())
        .first()
    )
    analysis_plan = plan_event.event_payload_json if plan_event else None
    latest_guidance_resolution_event = (
        db.query(InvestigationEvent)
        .filter(
            InvestigationEvent.investigation_id == inv.id,
            InvestigationEvent.event_type == "investigation.guidance_resolution",
        )
        .order_by(InvestigationEvent.timestamp.desc(), InvestigationEvent.id.desc())
        .first()
    )
    latest_locked_contract_event = (
        db.query(InvestigationEvent)
        .filter(
            InvestigationEvent.investigation_id == inv.id,
            InvestigationEvent.event_type == "investigation.canonical_contract_locked",
        )
        .order_by(InvestigationEvent.timestamp.desc(), InvestigationEvent.id.desc())
        .first()
    )
    decision_event = (
        db.query(InvestigationEvent)
        .filter(InvestigationEvent.investigation_id == inv.id, InvestigationEvent.event_type == "investigation.decision_impact")
        .order_by(InvestigationEvent.sequence.desc())
        .first()
    )
    decision_impact = decision_event.event_payload_json if decision_event else None
    compound_results = [
        (e.event_payload_json or {})
        for e in event_rows
        if e.event_type == "investigation.compound_objective.completed"
    ]

    # Human-reviewable cross-dataset execution ledger.
    requested_scope_ids = list(inv.requested_dataset_ids_json or [])
    requested_dataset_rows = (
        db.query(Dataset.id, Dataset.name)
        .filter(Dataset.project_id == inv.project_id, Dataset.id.in_(requested_scope_ids))
        .all()
        if requested_scope_ids else []
    )
    dataset_name_by_id = {str(row[0]): str(row[1]) for row in requested_dataset_rows}
    cross_dataset_review = build_cross_dataset_review(
        requested_dataset_ids=requested_scope_ids,
        dataset_name_by_id=dataset_name_by_id,
        experiments=[
            {
                "id": e.id,
                "test_code": e.test_code,
                "status": e.status,
                "arguments_json": e.arguments_json or {},
            }
            for e in exps
        ],
        evidences=[
            {
                "id": ev.id,
                "experiment_id": ev.experiment_id,
                "validation_status": ev.validation_status,
            }
            for ev in evid
        ],
        verifications=[
            {
                "evidence_id": verification.evidence_id,
                "status": verification.status,
            }
            for verification in evidence_verifications
        ],
        join_events=[
            dict(e.event_payload_json or {})
            for e in event_rows
            if e.event_type.startswith("relational.join.safety")
        ],
    )

    local_explanation = LocalExplanationEngine.build(
        question=inv.question,
        verdict_type=inv.verdict_type or "INCONCLUSIVE",
        direct_answer=inv.direct_answer,
        justification=verdict.justification if verdict else inv.main_finding or inv.direct_answer,
        hypotheses=hyps,
        experiments=exps,
        observations=[],
        evidence_rows=evid,
        method_selection=None,
        assumption_items=assumptions,
    )

    # Surface the recorded failure reason so the UI can show it instead of a generic message.
    _failure_message = None
    if str(inv.status).upper() == "FAILED":
        for _event in reversed(event_rows):
            _fp = _event.event_payload_json or {}
            if isinstance(_fp, dict) and _fp.get("error_message"):
                _failure_message = str(_fp["error_message"])
                break

    return {
        "id": inv.id,
        "project_id": inv.project_id,
        "question": inv.question,
        "status": inv.status,
        "error_message": _failure_message,
        "verdict_type": inv.verdict_type,
        "confidence_score": inv.confidence_score,
        "direct_answer": inv.direct_answer,
        "main_finding": inv.main_finding,
        "analysis_mode": "DETERMINISTIC",
        "analyst_result": analyst_result_payload,
        "claim_gate": ({
            "outcome": claim_gate_row.outcome,
            "requested_claim": claim_gate_row.requested_claim,
            "evidence_level": claim_gate_row.evidence_level,
            "design_status": claim_gate_row.design_status,
            "assumptions": claim_gate_row.assumptions_json or [],
            "allowed_claim": claim_gate_row.allowed_claim,
            "blocked_claim": claim_gate_row.blocked_claim,
            "reason": claim_gate_row.reason,
            "recovery_actions": claim_gate_row.recovery_actions_json or [],
            "computed_evidence": claim_gate_row.computed_evidence_json or {},
            "refusal_id": claim_gate_row.id if claim_gate_row.outcome == "REFUSE" else None,
            "requested_level": claim_gate_row.requested_level,
            "max_supported_level": claim_gate_row.max_supported_level,
            "identification_strategy": claim_gate_row.identification_strategy,
            "blocking_conditions": claim_gate_row.blocking_conditions_json or [],
            "missing_evidence": claim_gate_row.missing_evidence_json or [],
        } if claim_gate_row else None),
        "analysis_plan": analysis_plan,
        "guidance_resolution": (
            latest_guidance_resolution_event.event_payload_json
            if latest_guidance_resolution_event
            else None
        ),
        "locked_canonical_contract": (
            latest_locked_contract_event.event_payload_json
            if latest_locked_contract_event
            else None
        ),
        "decision_impact": decision_impact,
        "ai_status_message": "Analysis and explanation are generated locally from canonical investigation state. AI is optional.",
        "explanation": local_explanation.to_dict(),
        "reproducible_manifest_hash": inv.reproducible_manifest_hash,
        "objectives": [{"id": o.id, "statement": o.statement, "status": o.status} for o in objectives],
        "compound_objectives": compound_results,
        "cross_dataset_review": cross_dataset_review,
        "hypotheses": [
            {
                "id": h.hypothesis_code or h.id,
                "statement": h.statement,
                "prior_probability": h.prior_probability,
                "posterior_probability": h.posterior_probability,
                "belief_state": h.belief_state,
                "is_counter_hypothesis": h.is_counter_hypothesis,
            }
            for h in hyps
        ],
        "experiments": [
            {"id": e.test_code or e.id, "tool_name": e.tool_name, "status": e.status} for e in exps
        ],
        "predictions": [
            {
                "id": p.id,
                "prediction_code": p.prediction_code,
                "hypothesis_id": p.hypothesis_id,
                "statement": p.statement,
                "target_metric": p.target_metric,
                "target_dimension": p.target_dimension,
                "target_segment": p.target_segment,
                "expected_direction": p.expected_direction,
                "expected_value": p.expected_value,
                "threshold": p.threshold,
                "expected_magnitude": p.expected_magnitude,
                "expected_relationship": p.expected_relationship,
                "expected_effect": p.expected_effect,
                "confidence": p.confidence,
                "testability": p.testability,
                "status": p.status,
                "target_experiment_id": p.target_experiment_id,
                "actual_observed_result": p.actual_observed_result_json or {},
                "evaluation_reason": p.evaluation_reason,
                "created_at": p.created_at.isoformat() if p.created_at else None,
                "evaluated_at": p.evaluated_at.isoformat() if p.evaluated_at else None,
            } for p in predictions
        ],
        "belief_updates": [
            {
                "id": b.id,
                "hypothesis_id": b.hypothesis_id,
                "evidence_id": b.evidence_id,
                "prior_probability": b.prior_probability,
                "bayes_factor": b.bayes_factor,
                "likelihood": b.likelihood_p,  # legacy probability field; never populated by canonical updates
                "posterior_probability": b.posterior_probability,
                "entropy_delta": b.entropy_delta,
                "update_step_index": b.update_step_index,
                "created_at": b.created_at.isoformat() if b.created_at else None,
            } for b in belief_updates
        ],
        "evidence_verifications": [
            {
                "id": v.id,
                "evidence_id": v.evidence_id,
                "primary_tool": v.primary_tool,
                "secondary_tool": v.secondary_tool,
                "tolerance_threshold": v.tolerance_threshold,
                "observed_delta_pct": v.observed_delta_pct,
                "is_deterministic": v.is_deterministic,
                "validator_fingerprint": v.validator_fingerprint,
                "status": v.status,
                "created_at": v.created_at.isoformat() if v.created_at else None,
            } for v in evidence_verifications
        ],
        "semantic_world_model": {
            "status": "PARTIAL" if phase_semantic_payload or latest_contract else "UNAVAILABLE",
            "phase_payload": phase_semantic_payload,
            "contract": {
                "id": latest_contract.id,
                "version": latest_contract.version,
                "problem_class": latest_contract.problem_class,
                "claim_type": latest_contract.claim_type,
                "target": latest_contract.target_json,
                "explanatory_variables": latest_contract.explanatory_variables_json or [],
                "grain": latest_contract.grain_json,
                "scope": latest_contract.scope_json,
                "time_window": latest_contract.time_window_json,
                "semantic_interpretations": latest_contract.semantic_interpretations_json or [],
                "unresolved_questions": latest_contract.unresolved_questions_json or [],
            } if latest_contract else None,
            "primary_dataset": primary_dataset_name,
        "data_readiness": {
                "dataset_name": readiness.dataset_name,
                "dataset_version_ids": readiness.dataset_version_ids_json or [],
                "source_fingerprints": readiness.source_fingerprints_json or {},
                "row_count": readiness.row_count,
                "column_count": readiness.column_count,
                "fitness_verdict": readiness.fitness_verdict,
                "checks": readiness.checks_json or {},
            } if readiness else None,
            "limitation": "Full relational world-model projection is not persisted as a first-class investigation artifact in this schema; this response exposes only the durable semantic and readiness facts actually recorded.",
        },
        "adversarial_findings": adversarial_payloads,
        # Session 8: steps that failed were previously visible only in the raw
        # event log. A human verifier must be told when a planned check did not run.
        "failed_steps": [
            {
                "step_code": (e.event_payload_json or {}).get("step_code"),
                "step_type": (e.event_payload_json or {}).get("step_type"),
                "error": (e.event_payload_json or {}).get("error"),
            }
            for e in event_rows if e.event_type == "step.failed"
        ],
        "multiverse": multiverse_payload,
        "missingness": missingness_payload,
        "causal_status": {
            "gate": causal_gate_payload,
            "effect_estimation": causal_effect_payload,
        },
        "epistemic_assessment": epistemic_payload,
        "stopping_reason": inv.stopping_rationale,
        "stopping_criteria_met": bool(inv.stopping_criteria_met),
        "provenance": {
            "manifest_hash": inv.reproducible_manifest_hash,
            "manifest_event": manifest_payload,
            "dataset_version_ids": inv.dataset_version_ids_json or [],
            "requested_dataset_ids": inv.requested_dataset_ids_json,
            "claim_lineage": [
                {
                    "evidence_id": ev.id,
                    "hypothesis_id": ev.hypothesis_id,
                    "experiment_id": ev.experiment_id,
                    "evidence_identity_hash": ev.evidence_identity_hash,
                    "validation_status": ev.validation_status,
                    "sql_executed": observations_by_experiment.get(ev.experiment_id).sql_executed if observations_by_experiment.get(ev.experiment_id) else None,
                    "source_dataset": ((experiments_by_id.get(ev.experiment_id).arguments_json or {}).get("dataset_versions", {}) if experiments_by_id.get(ev.experiment_id) else {}),
                } for ev in evid
            ],
            "limitation": "Per-claim dataset UUID/version and schema fingerprint are exposed where the persisted experiment arguments contain them; absent values are returned as null rather than inferred.",
        },
        "evidence": [
            {
                "id": ev.id,
                "statement": ev.statement,
                "validation_status": ev.validation_status,
                "p_value": ev.p_value,
                "effect_size": ev.effect_size,
                "calculation_summary": ev.calculation_summary or "",
                "row_count_analyzed": int(observations_by_experiment.get(ev.experiment_id).row_count_analyzed) if observations_by_experiment.get(ev.experiment_id) else 0,
                "sql_executed": observations_by_experiment.get(ev.experiment_id).sql_executed if observations_by_experiment.get(ev.experiment_id) else None,
                "calculation_trace": (observations_by_experiment.get(ev.experiment_id).result_json or {}).get("calculation_trace") if observations_by_experiment.get(ev.experiment_id) else None,
                "experiment_id": ev.experiment_id,
                "source_dataset": ((experiments_by_id.get(ev.experiment_id).arguments_json or {}).get("dataset_versions", {}) if experiments_by_id.get(ev.experiment_id) else {}),
            }
            for ev in evid
        ],
        "verdict": {
            "verdict_type": verdict.verdict_type,
            "confidence": verdict.confidence_score,
            # OBSERVED answers are exact recomputations; the verdict row's justification is the
            # hypothesis loop's rationale ("posterior 0.99 ...") about a different claim.
            "justification": (
                inv.main_finding
                if str(verdict.verdict_type).upper() == "OBSERVED" and getattr(inv, "main_finding", None)
                else verdict.justification
            ),
            "epistemic_grade": verdict.epistemic_grade,
        } if verdict else None,
        "assumption_ledger": {
            "items": [
                {
                    "statement": a.statement,
                    "validated": bool(a.is_validated),
                    "sensitivity_risk": a.sensitivity_risk,
                } for a in assumptions
            ],
            "quality_card": {
                "overall_status": "RED" if len(high_risk_assumptions) > 2 else ("AMBER" if high_risk_assumptions else "GREEN"),
                "assumption_count": len(assumptions),
                "high_risk_unvalidated": len(high_risk_assumptions),
                "validated_count": sum(1 for a in assumptions if a.is_validated),
                "limitation_count": sum(1 for a in assumptions if not a.is_validated),
            },
        },
        "decision_recommendations": [
            {
                "recommendation_id": r.recommendation_id,
                "action_title": r.action_title,
                "action_description": r.action_description,
                "grounded_hypothesis_id": r.grounded_hypothesis_id,
                "target_metric": r.target_metric,
                "evidence_binding": (
                    lambda _hyp, _ground: {
                        "hypothesis_code": (_hyp.hypothesis_code if _hyp is not None else None),
                        "evidence_ids": list(_ground.supporting_evidence_ids),
                        "verified_evidence_ids": list(_ground.verified_supporting_evidence_ids),
                        "unverified_evidence_ids": list(_ground.unverified_supporting_evidence_ids),
                        "verified_contradicting_evidence_ids": list(_ground.verified_contradicting_evidence_ids),
                        "evidence_count": len(_ground.supporting_evidence_ids),
                        "binding_status": _ground.status,
                        "reason": _ground.reason,
                        "persisted_binding": recommendation_grounding_events.get(r.recommendation_id) or recommendation_grounding_events.get(r.grounded_hypothesis_id),
                    }
                )(
                    hypothesis_by_code.get(r.grounded_hypothesis_id.split("_", 1)[1]) if "_" in r.grounded_hypothesis_id else None,
                    evaluate_recommendation_grounding(
                        supporting_evidence_ids=(
                            getattr(hypothesis_by_code.get(r.grounded_hypothesis_id.split("_", 1)[1]), "supporting_evidence_ids", None) or []
                            if "_" in r.grounded_hypothesis_id else []
                        ),
                        contradicting_evidence_ids=(
                            getattr(hypothesis_by_code.get(r.grounded_hypothesis_id.split("_", 1)[1]), "contradicting_evidence_ids", None) or []
                            if "_" in r.grounded_hypothesis_id else []
                        ),
                        evidence_validation_status_by_id={
                            str(e.id): str(getattr(e, "validation_status", "")) for e in evid
                        },
                        latest_verification_status_by_id=latest_verification_status_by_id,
                    ),
                ),
                "target_metric": r.target_metric,
                "expected_utility": {
                    "expected_gain_metric": r.expected_gain_metric,
                    "downside_risk_metric": r.downside_risk_metric,
                    "probability_of_success": r.probability_of_success,
                    "net_expected_utility": r.net_expected_utility,
                    "utility_function_description": r.utility_function_description,
                },
                "policy_compliance_passed": r.policy_compliance_passed,
                "required_preconditions": r.required_preconditions_json or [],
            }
            for r in decision_recs
        ],
        "graph_edges": [
            {"source": e.source_node_id, "target": e.target_node_id, "type": e.relationship_type}
            for e in edges
        ],
        "pending_decision_request": {
            "id": pending_decision.id,
            "question": pending_decision.question,
            "options": pending_decision.options_json,
        } if pending_decision else None,
        "pending_resource_wait": {
            "id": pending_resource.id,
            "resource_id": pending_resource.resource_id,
            "reason": pending_resource.reason,
        } if pending_resource else None,
        "created_at": inv.created_at.isoformat() if inv.created_at else None,
        "updated_at": inv.updated_at.isoformat() if inv.updated_at else None,
    }


@router.get("/{investigation_id}/events")
async def stream_investigation_events(
    investigation_id: str,
    last_event_id: Optional[str] = Query(None),
    last_event_id_header: Optional[str] = Header(None, alias="Last-Event-ID"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Live Server-Sent Events (SSE) stream with Last-Event-ID cursor replay.
    Continuously streams new events as they are committed to database by worker processes.
    """
    inv = db.query(Investigation).filter(Investigation.id == investigation_id).first()
    if not inv:
        raise HTTPException(status_code=404, detail="Investigation not found.")
    verify_project_ownership(inv.project_id, current_user, db)

    effective_last_id = last_event_id or last_event_id_header

    async def event_generator():
        last_seen_seq = int(effective_last_id) if (effective_last_id and str(effective_last_id).isdigit()) else 0

        while True:
            session = SessionLocal()
            try:
                events = (
                    session.query(InvestigationEvent)
                    .filter(
                        InvestigationEvent.investigation_id == investigation_id,
                        InvestigationEvent.sequence > last_seen_seq,
                    )
                    .order_by(InvestigationEvent.sequence.asc())
                    .limit(50)
                    .all()
                )

                for ev in events:
                    last_seen_seq = ev.sequence or (last_seen_seq + 1)
                    payload = {
                        "id": ev.id,
                        "sequence": ev.sequence,
                        "type": ev.event_type,
                        "payload": ev.event_payload_json,
                        "timestamp": ev.timestamp.isoformat() if ev.timestamp else None,
                    }
                    yield f"id: {ev.sequence or ev.id}\nevent: {ev.event_type}\ndata: {json.dumps(payload)}\n\n"

                inv_curr = session.query(Investigation).filter(Investigation.id == investigation_id).first()
                if inv_curr and inv_curr.status in [
                    InvestigationState.COMPLETED,
                    InvestigationState.FAILED,
                    InvestigationState.CANCELLED,
                    InvestigationState.PARTIAL,
                ]:
                    # Terminal state reached, yield remaining and close
                    break
            finally:
                session.close()

            await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/{investigation_id}/cancel")
def cancel_investigation(
    investigation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Request clean cancellation of an active or queued investigation."""
    inv = db.query(Investigation).filter(Investigation.id == investigation_id).first()
    if not inv:
        raise HTTPException(status_code=404, detail="Investigation not found.")
    verify_project_ownership(inv.project_id, current_user, db)

    transition_investigation(
        db=db,
        investigation_id=investigation_id,
        to_state=InvestigationState.CANCEL_REQUESTED,
        actor=current_user.email,
    )

    job = db.query(InvestigationJob).filter(InvestigationJob.investigation_id == investigation_id).first()
    if job and job.status not in [JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED]:
        transition_job(db=db, job_id=job.id, to_state=JobState.CANCEL_REQUESTED)

    return {"status": "success", "message": f"Cancellation requested for {investigation_id}."}


@router.post("/{investigation_id}/decision")
def resolve_decision_request(
    investigation_id: str,
    payload: DecisionResponseRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Submit human decision for a suspended investigation and resume execution."""
    inv = db.query(Investigation).filter(Investigation.id == investigation_id).first()
    if not inv:
        raise HTTPException(status_code=404, detail="Investigation not found.")
    verify_project_ownership(inv.project_id, current_user, db)

    dec_req = (
        db.query(InvestigationDecisionRequest)
        .filter(
            InvestigationDecisionRequest.investigation_id == investigation_id,
            InvestigationDecisionRequest.status == "PENDING",
        )
        .first()
    )
    if not dec_req:
        raise HTTPException(status_code=400, detail="No pending decision request for this investigation.")

    dec_req.status = "RESOLVED"
    dec_req.user_response = payload.user_response
    dec_req.user_id = current_user.id
    dec_req.resolved_at = datetime.now(timezone.utc)

    # Transition state to QUEUED
    transition_investigation(
        db=db,
        investigation_id=investigation_id,
        to_state=InvestigationState.QUEUED,
        actor=current_user.email,
    )

    # Resume job in queue
    queue = get_default_queue_provider()
    queue.resume_job(investigation_id=investigation_id)

    return {
        "status": "success",
        "message": f"Decision submitted: '{payload.user_response}'. Investigation resumed.",
        "investigation_id": investigation_id,
    }


@router.get("/{investigation_id}/explanation")
def get_investigation_explanation(
    investigation_id: str,
    mode: str = Query("DETERMINISTIC", pattern="^(DETERMINISTIC|AI_AUGMENTED)$"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return a local explanation, with optional AI-only presentation augmentation.

    The deterministic investigation, evidence, verdict, and provenance remain the
    source of truth. AI can only rewrite the already-established explanation.
    """
    inv = db.query(Investigation).filter(Investigation.id == investigation_id).first()
    if not inv:
        raise HTTPException(status_code=404, detail="Investigation not found.")
    verify_project_ownership(inv.project_id, current_user, db)

    hyps = db.query(Hypothesis).filter(Hypothesis.investigation_id == inv.id).all()
    exps = db.query(Experiment).filter(Experiment.investigation_id == inv.id).all()
    evid = db.query(Evidence).filter(Evidence.investigation_id == inv.id).all()
    verdict = db.query(InvestigationVerdict).filter(InvestigationVerdict.investigation_id == inv.id).first()
    assumptions = db.query(Assumption).filter(Assumption.investigation_id == inv.id).all()
    predictions = db.query(Prediction).filter(Prediction.investigation_id == inv.id).order_by(Prediction.created_at.asc()).all()
    belief_updates = db.query(BeliefUpdate).filter(BeliefUpdate.investigation_id == inv.id).order_by(BeliefUpdate.update_step_index.asc(), BeliefUpdate.created_at.asc()).all()
    evidence_verifications = (
        db.query(EvidenceVerification)
        .filter(EvidenceVerification.evidence_id.in_([e.id for e in evid]))
        .order_by(EvidenceVerification.created_at.asc())
        .all()
    ) if evid else []
    event_rows = (
        db.query(InvestigationEvent)
        .filter(InvestigationEvent.investigation_id == inv.id)
        .order_by(InvestigationEvent.sequence.asc())
        .all()
    )

    def _latest_payload(prefixes):
        for event in reversed(event_rows):
            if any(event.event_type == p or event.event_type.startswith(p) for p in prefixes):
                return event.event_payload_json or {}
        return None

    epistemic_payload = _latest_payload(["calculation.epistemic_calibration"])
    missingness_payload = _latest_payload(["investigation.missingness_sensitivity"])
    causal_gate_payload = _latest_payload(["investigation.causal_gate.evaluated", "investigation.causal_gate.fallback"])
    causal_effect_payload = _latest_payload(["investigation.causal_effect.estimated", "investigation.causal_effect.estimation_failed"])
    adversarial_payloads = [
        e.event_payload_json or {} for e in event_rows
        if e.event_type.startswith("investigation.adversarial_challenge")
        or e.event_type.startswith("calculation.adversarial")
    ]
    multiverse_payload = _latest_payload(["investigation.multiverse.skipped", "calculation.multiverse"])
    manifest_payload = _latest_payload(["calculation.provenance_manifest"])
    phase_semantic_payload = _latest_payload(["investigation.phase.semantic_model"])
    analyst_result_payload = _latest_payload(["investigation.analyst_result"])
    # `plan_payload` was dead code that referenced `analysis_plan` before it was
    # assigned later in this function (undefined-name crash, pyflakes-confirmed);
    # it was never actually consumed downstream, so it is removed rather than
    # reordered. `analysis_plan` itself remains defined and used below.
    latest_contract = (
        db.query(InvestigationContract)
        .filter(InvestigationContract.investigation_id == inv.id)
        .order_by(InvestigationContract.version.desc())
        .first()
    )
    scope_payload = _latest_payload(["investigation.dataset_scope_resolved"]) or {}
    primary_dataset_name = scope_payload.get("primary_dataset")
    readiness_rows = (
        db.query(InvestigationDataReadiness)
        .filter(InvestigationDataReadiness.investigation_id == inv.id)
        .order_by(InvestigationDataReadiness.dataset_name.asc(), InvestigationDataReadiness.assessed_at.desc())
        .all()
    )
    latest_readiness_by_dataset = {}
    for row in readiness_rows:
        latest_readiness_by_dataset.setdefault(row.dataset_name, row)
    readiness = latest_readiness_by_dataset.get(primary_dataset_name)
    if readiness is None and latest_readiness_by_dataset:
        # Keep the legacy field deterministic even when an older investigation
        # predates dataset_scope_resolved; never let alphabetical ordering choose
        # a misleading non-primary dataset.
        readiness = next(iter(latest_readiness_by_dataset.values()))
    plan_event = (
        db.query(InvestigationEvent)
        .filter(InvestigationEvent.investigation_id == inv.id, InvestigationEvent.event_type == "investigation.universal_analysis_plan")
        .order_by(InvestigationEvent.sequence.desc())
        .first()
    )
    local = LocalExplanationEngine.build(
        question=inv.question,
        verdict_type=inv.verdict_type or "INCONCLUSIVE",
        direct_answer=inv.direct_answer,
        justification=verdict.justification if verdict else inv.main_finding or inv.direct_answer,
        hypotheses=hyps,
        experiments=exps,
        evidence_rows=evid,
        method_selection=None,
        assumption_items=assumptions,
    )

    result = local.to_dict()
    result["analysis_id"] = investigation_id
    result["analysis_mode"] = "DETERMINISTIC"
    result["ai_fallback_triggered"] = False
    result["ai_status_message"] = "Local deterministic explanation active; no AI required."
    # Attach the data fetched above (predictions, belief updates, evidence
    # verifications, and event-derived payloads) so this endpoint doesn't
    # silently drop them the way the sibling GET /investigations/{id}
    # endpoint's equivalent block does not. Serialization mirrors that
    # sibling endpoint's dict shape.
    result["predictions"] = [
        {
            "id": p.id,
            "prediction_code": p.prediction_code,
            "hypothesis_id": p.hypothesis_id,
            "statement": p.statement,
            "target_metric": p.target_metric,
            "target_dimension": p.target_dimension,
            "target_segment": p.target_segment,
            "expected_direction": p.expected_direction,
            "expected_value": p.expected_value,
            "threshold": p.threshold,
            "expected_magnitude": p.expected_magnitude,
            "expected_relationship": p.expected_relationship,
            "expected_effect": p.expected_effect,
            "confidence": p.confidence,
            "testability": p.testability,
            "status": p.status,
            "target_experiment_id": p.target_experiment_id,
            "actual_observed_result": p.actual_observed_result_json or {},
            "evaluation_reason": p.evaluation_reason,
            "created_at": p.created_at.isoformat() if p.created_at else None,
            "evaluated_at": p.evaluated_at.isoformat() if p.evaluated_at else None,
        } for p in predictions
    ]
    result["belief_updates"] = [
        {
            "id": b.id,
            "hypothesis_id": b.hypothesis_id,
            "evidence_id": b.evidence_id,
            "prior_probability": b.prior_probability,
            "bayes_factor": b.bayes_factor,
            "likelihood": b.likelihood_p,
            "posterior_probability": b.posterior_probability,
            "entropy_delta": b.entropy_delta,
            "update_step_index": b.update_step_index,
            "created_at": b.created_at.isoformat() if b.created_at else None,
        } for b in belief_updates
    ]
    result["evidence_verifications"] = [
        {
            "id": v.id,
            "evidence_id": v.evidence_id,
            "primary_tool": v.primary_tool,
            "secondary_tool": v.secondary_tool,
            "tolerance_threshold": v.tolerance_threshold,
            "observed_delta_pct": v.observed_delta_pct,
            "is_deterministic": v.is_deterministic,
            "validator_fingerprint": v.validator_fingerprint,
            "status": v.status,
            "created_at": v.created_at.isoformat() if v.created_at else None,
        } for v in evidence_verifications
    ]
    result["semantic_world_model"] = {
        "status": "PARTIAL" if phase_semantic_payload or latest_contract else "UNAVAILABLE",
        "phase_payload": phase_semantic_payload,
        "contract": {
            "id": latest_contract.id,
            "version": latest_contract.version,
            "problem_class": latest_contract.problem_class,
            "claim_type": latest_contract.claim_type,
            "target": latest_contract.target_json,
            "explanatory_variables": latest_contract.explanatory_variables_json or [],
            "grain": latest_contract.grain_json,
            "scope": latest_contract.scope_json,
            "time_window": latest_contract.time_window_json,
            "semantic_interpretations": latest_contract.semantic_interpretations_json or [],
            "unresolved_questions": latest_contract.unresolved_questions_json or [],
        } if latest_contract else None,
        "data_readiness": {
            "dataset_name": readiness.dataset_name,
            "dataset_version_ids": readiness.dataset_version_ids_json or [],
            "source_fingerprints": readiness.source_fingerprints_json or {},
            "row_count": readiness.row_count,
            "column_count": readiness.column_count,
            "fitness_verdict": readiness.fitness_verdict,
            "checks": readiness.checks_json or {},
        } if readiness else None,
        "dataset_scope_readiness": [
            {
                "dataset_name": row.dataset_name,
                "dataset_version_ids": row.dataset_version_ids_json or [],
                "source_fingerprints": row.source_fingerprints_json or {},
                "row_count": row.row_count,
                "column_count": row.column_count,
                "overall_quality_score": row.overall_quality_score,
                "fitness_verdict": row.fitness_verdict,
                "can_proceed": row.can_proceed,
                "critical_issues": row.critical_issues_json or [],
                "warnings": row.warnings_json or [],
            }
            for row in latest_readiness_by_dataset.values()
        ],
        "limitation": "Full relational world-model projection is not persisted as a first-class investigation artifact in this schema; this response exposes only the durable semantic and readiness facts actually recorded.",
    }
    result["adversarial_findings"] = adversarial_payloads
    result["multiverse"] = multiverse_payload
    result["missingness"] = missingness_payload
    result["causal_status"] = {
        "gate": causal_gate_payload,
        "effect_estimation": causal_effect_payload,
    }
    result["epistemic_assessment"] = epistemic_payload
    result["provenance_manifest_event"] = manifest_payload
    result["analysis_plan"] = plan_event.event_payload_json if plan_event else None
    result["analyst_result"] = analyst_result_payload

    if mode == "AI_AUGMENTED":
        try:
            from apps.api.src.ai.providers.factory import get_ai_provider as _get_ai_provider, is_ai_enabled
            _user_id = str(current_user.id) if hasattr(current_user, "id") and current_user.id else None
            provider = _get_ai_provider(user_id=_user_id) if is_ai_enabled(user_id=_user_id) else None
        except Exception:
            provider = None

        if provider is None:
            result["ai_fallback_triggered"] = True
            result["ai_status_message"] = "AI augmentation unavailable; local explanation retained."
            return result

        ok, message = provider.test_connection()
        if ok and provider.provider_type != "mock":
            prompt = (
                "Rewrite this verified AA-OS analytical explanation for clarity. "
                "Do not add facts, calculations, numbers, causal claims, or recommendations. "
                "Preserve uncertainty and limitations exactly. Plain text only.\n\n"
                + str(result)
            )
            ai_text = provider.generate_response(
                prompt,
                system_prompt=(
                    "You are a presentation layer only. The deterministic analysis and evidence are authoritative. "
                    "Never strengthen or alter the epistemic conclusion."
                ),
            )
            if ai_text and not ai_text.lower().startswith(("error:", "ollama execution error", "openai api error", "gemini api error")):
                result["ai_rewrite"] = ai_text.strip()
                result["mode"] = "AI_AUGMENTED"
                result["analysis_mode"] = "AI_AUGMENTED"
                result["ai_status_message"] = f"AI presentation augmentation active via {provider.provider_type}/{provider.model_name}; canonical analysis remains deterministic."
            else:
                result["ai_fallback_triggered"] = True
                result["ai_status_message"] = f"AI augmentation unavailable; local explanation retained. {message}"
        else:
            result["ai_fallback_triggered"] = True
            result["ai_status_message"] = f"AI augmentation unavailable; local explanation retained. {message}"

    return result


@router.get("/{investigation_id}/contract")
def get_investigation_contract(
    investigation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    inv = db.query(Investigation).filter(Investigation.id == investigation_id).first()
    if not inv:
        raise HTTPException(status_code=404, detail="Investigation not found.")
    verify_project_ownership(inv.project_id, current_user, db)
    contract = (db.query(InvestigationContract)
                .filter(InvestigationContract.investigation_id == investigation_id)
                .order_by(InvestigationContract.version.desc()).first())
    if not contract:
        raise HTTPException(status_code=404, detail="Analysis contract not available yet.")
    return {
        "id": contract.id, "investigation_id": contract.investigation_id, "version": contract.version,
        "status": contract.status, "original_question": contract.original_question,
        "normalized_question": contract.normalized_question, "problem_class": contract.problem_class,
        "claim_type": contract.claim_type, "target": contract.target_json,
        "explanatory_variables": contract.explanatory_variables_json, "population": contract.population_json,
        "grain": contract.grain_json, "scope": contract.scope_json, "time_window": contract.time_window_json,
        "estimand": contract.estimand_json, "assumptions": contract.assumptions_json,
        "data_requirements": contract.data_requirements_json, "candidate_methods": contract.candidate_methods_json,
        # v20-C4.2.3: proposal vs final.  selected_method is a projection of final_contract.
        "proposed_method": contract.proposed_method_json,
        "selected_method": contract.selected_method_json,
        "final_contract": contract.final_contract_json,
        "analytical_identity": contract.analytical_identity,
        "evidence_requirements": contract.evidence_requirements_json,
        "stopping_criteria": contract.stopping_criteria_json, "ambiguity_state": contract.ambiguity_state_json,
        "semantic_interpretations": contract.semantic_interpretations_json, "unresolved_questions": contract.unresolved_questions_json,
        "confidence": contract.confidence, "compiled_at": contract.compiled_at.isoformat() if contract.compiled_at else None,
    }


@router.get("/{investigation_id}/data-readiness")
def get_investigation_data_readiness(
    investigation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    inv = db.query(Investigation).filter(Investigation.id == investigation_id).first()
    if not inv:
        raise HTTPException(status_code=404, detail="Investigation not found.")
    verify_project_ownership(inv.project_id, current_user, db)
    readiness_rows = (db.query(InvestigationDataReadiness)
                 .filter(InvestigationDataReadiness.investigation_id == investigation_id)
                 .order_by(InvestigationDataReadiness.dataset_name.asc(), InvestigationDataReadiness.assessed_at.desc()).all())
    latest_readiness_by_dataset = {}
    for row in readiness_rows:
        latest_readiness_by_dataset.setdefault(row.dataset_name, row)
    scope_event = (db.query(InvestigationEvent)
                   .filter(InvestigationEvent.investigation_id == investigation_id,
                           InvestigationEvent.event_type == "investigation.dataset_scope_resolved")
                   .order_by(InvestigationEvent.sequence.desc()).first())
    scope_payload = scope_event.event_payload_json if scope_event else {}
    primary_dataset_name = (scope_payload or {}).get("primary_dataset")
    readiness = latest_readiness_by_dataset.get(primary_dataset_name) or next(iter(latest_readiness_by_dataset.values()), None)
    if not readiness:
        raise HTTPException(status_code=404, detail="Data readiness assessment not available yet.")
    return {
        "id": readiness.id, "investigation_id": readiness.investigation_id, "contract_id": readiness.contract_id,
        "dataset_name": readiness.dataset_name, "dataset_version_ids": readiness.dataset_version_ids_json,
        "source_fingerprints": readiness.source_fingerprints_json, "row_count": readiness.row_count,
        "column_count": readiness.column_count, "overall_quality_score": readiness.overall_quality_score,
        "fitness_verdict": readiness.fitness_verdict, "can_proceed": readiness.can_proceed,
        "checks": readiness.checks_json, "critical_issues": readiness.critical_issues_json,
        "warnings": readiness.warnings_json, "recommendations": readiness.recommendations_json,
        "assessed_at": readiness.assessed_at.isoformat() if readiness.assessed_at else None,
        "primary_dataset": primary_dataset_name,
        "datasets": [
            {
                "dataset_name": row.dataset_name,
                "dataset_version_ids": row.dataset_version_ids_json or [],
                "source_fingerprints": row.source_fingerprints_json or {},
                "row_count": row.row_count,
                "column_count": row.column_count,
                "overall_quality_score": row.overall_quality_score,
                "fitness_verdict": row.fitness_verdict,
                "can_proceed": row.can_proceed,
                "checks": row.checks_json or {},
                "critical_issues": row.critical_issues_json or [],
                "warnings": row.warnings_json or [],
                "recommendations": row.recommendations_json or [],
                "assessed_at": row.assessed_at.isoformat() if row.assessed_at else None,
            }
            for row in latest_readiness_by_dataset.values()
        ],
    }


# ---------------------------------------------------------------------------
# Human verification workflow: AA-OS does the analysis, the analyst verifies.
# The decision log is append-only (investigation_events), so no schema change
# and a full audit trail.  See packages/.../governance/verification_packet.py.
# ---------------------------------------------------------------------------
_VERIFICATION_ITEM_EVENT = "verification.item_reviewed"
_VERIFICATION_SIGNOFF_EVENT = "verification.signed_off"
_VERIFIABLE_STATES = {"COMPLETED", "PARTIAL"}


def _append_verification_event(db: Session, investigation_id: str, event_type: str, payload: Dict[str, Any]) -> None:
    """Append one event with a concurrency-safe sequence (same scheme as the checkpointer)."""
    from sqlalchemy import func
    from sqlalchemy.exc import IntegrityError

    safe = json.loads(json.dumps(payload, default=str))
    for attempt in range(5):
        max_seq = db.query(func.max(InvestigationEvent.sequence)).filter(
            InvestigationEvent.investigation_id == investigation_id
        ).scalar()
        db.add(InvestigationEvent(
            id=f"EVT-{gen_uuid()[:8]}",
            sequence=(max_seq or 0) + 1,
            investigation_id=investigation_id,
            event_type=event_type,
            event_payload_json=safe,
            timestamp=datetime.now(timezone.utc),
        ))
        try:
            db.commit()
            return
        except IntegrityError:
            db.rollback()
            if attempt == 4:
                raise HTTPException(status_code=503, detail="Could not record the review; please retry.")


def _verification_state(db: Session, investigation_id: str, current_user: User) -> Dict[str, Any]:
    # get_investigation performs the 404 and project-ownership checks.
    payload = get_investigation(investigation_id, current_user=current_user, db=db)
    if str(payload.get("status", "")).upper() not in _VERIFIABLE_STATES:
        raise HTTPException(
            status_code=409,
            detail=f"Investigation is {payload.get('status')}; verification opens once it has completed.",
        )
    packet = build_verification_packet(payload)
    rows = (
        db.query(InvestigationEvent)
        .filter(
            InvestigationEvent.investigation_id == investigation_id,
            InvestigationEvent.event_type.in_([_VERIFICATION_ITEM_EVENT, _VERIFICATION_SIGNOFF_EVENT]),
        )
        .order_by(InvestigationEvent.sequence.asc())
        .all()
    )
    decisions, signoffs = [], []
    for r in rows:
        rec = dict(r.event_payload_json or {})
        rec["at"] = r.timestamp.isoformat() if r.timestamp else None
        (decisions if r.event_type == _VERIFICATION_ITEM_EVENT else signoffs).append(rec)
    review = evaluate_review(packet, decisions)
    return {"packet": packet, "review": review, "signoff": current_signoff(packet, signoffs)}


@router.get("/{investigation_id}/verification")
def get_verification_packet(
    investigation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """The verifier's worklist: risk-ranked items, current decisions, and sign-off status."""
    return _verification_state(db, investigation_id, current_user)


@router.post("/{investigation_id}/verification/decisions")
def record_verification_decision(
    investigation_id: str,
    body: VerificationDecisionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Record the human verifier's decision on one item."""
    state = _verification_state(db, investigation_id, current_user)
    packet = state["packet"]
    item = next((i for i in packet["items"] if i["id"] == body.item_id), None)
    error = validate_decision(packet, body.item_id, body.decision, body.comment)
    if error:
        raise HTTPException(status_code=422, detail=error)
    if body.item_fingerprint and body.item_fingerprint != item["fingerprint"]:
        raise HTTPException(
            status_code=409,
            detail="This item changed since you opened it (the investigation was re-run). Reload and review it again.",
        )
    _append_verification_event(db, investigation_id, _VERIFICATION_ITEM_EVENT, {
        "item_id": body.item_id,
        "item_fingerprint": item["fingerprint"],
        "decision": body.decision,
        "comment": (body.comment or "").strip(),
        "reviewer": {"id": current_user.id, "email": current_user.email},
        "packet_fingerprint": packet["packet_fingerprint"],
    })
    return _verification_state(db, investigation_id, current_user)


@router.post("/{investigation_id}/verification/signoff")
def sign_off_verification(
    investigation_id: str,
    body: VerificationSignoffRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Record the overall verification outcome, bound to the exact result reviewed."""
    state = _verification_state(db, investigation_id, current_user)
    packet, review = state["packet"], state["review"]
    error = validate_signoff(review, body.outcome, body.comment)
    if error:
        raise HTTPException(status_code=422, detail=error)
    if body.packet_fingerprint and body.packet_fingerprint != packet["packet_fingerprint"]:
        raise HTTPException(
            status_code=409,
            detail="The result changed since you opened it (the investigation was re-run). Reload and review again.",
        )
    _append_verification_event(db, investigation_id, _VERIFICATION_SIGNOFF_EVENT, {
        "outcome": body.outcome,
        "comment": (body.comment or "").strip(),
        "reviewer": {"id": current_user.id, "email": current_user.email},
        "packet_fingerprint": packet["packet_fingerprint"],
        "manifest_hash": packet["manifest_hash"],
        "machine_verdict": packet["machine_claim"]["verdict"],
        "decisions_snapshot": {
            k: {"decision": v["decision"], "comment": v["comment"]} for k, v in review["per_item"].items() if not v["stale"]
        },
    })
    return _verification_state(db, investigation_id, current_user)


@router.get("/{investigation_id}/export")
def export_investigation(
    investigation_id: str,
    format: str = Query("pdf", pattern="^(pdf|docx)$"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Export investigation result as publication-grade PDF or structured DOCX in standard answer format."""
    inv_data = get_investigation(investigation_id=investigation_id, current_user=current_user, db=db)
    from fastapi.responses import Response
    from packages.analytics_core.src.reporting.exporter import (
        export_investigation_to_pdf,
        export_investigation_to_docx,
    )

    clean_id = investigation_id.replace(" ", "_")
    from packages.analytics_core.src.reporting.exporter import (
        compute_canonical_report_hash,
        compute_export_fingerprint,
        verify_report_integrity,
    )
    full_hash = compute_canonical_report_hash(inv_data)
    seal = compute_export_fingerprint(inv_data)

    if format.lower() == "docx":
        docx_bytes = export_investigation_to_docx(inv_data)
        return Response(
            content=docx_bytes,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                "Content-Disposition": f'attachment; filename="MindEd_Investigation_{clean_id}.docx"',
                "X-MindEd-Document-Seal": seal,
                "X-MindEd-Content-Digest": full_hash,
            },
        )

    pdf_bytes = export_investigation_to_pdf(inv_data)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="MindEd_Investigation_{clean_id}.pdf"',
            "X-MindEd-Document-Seal": seal,
            "X-MindEd-Content-Digest": full_hash,
        },
    )


@router.get("/{investigation_id}/export/verify")
def verify_export_fingerprint(
    investigation_id: str,
    seal: Optional[str] = Query(None, description="The visible document seal e.g. MDED-XXXX-XXXX-XXXX-XXXX"),
    digest: Optional[str] = Query(None, description="The full 64-char SHA-256 canonical content digest"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Verify that a supplied canonical report data seal or digest strictly matches the authoritative stored investigation record."""
    inv_data = get_investigation(investigation_id=investigation_id, current_user=current_user, db=db)
    from packages.analytics_core.src.reporting.exporter import verify_report_integrity

    is_valid, details = verify_report_integrity(inv_data, expected_full_hash=digest, expected_seal=seal)
    return {
        "investigation_id": investigation_id,
        "verification_scope": "CANONICAL_REPORT_DATA_FINGERPRINT",
        "authentic": is_valid,
        "details": details,
        "notice": (
            "Verified against authoritative stored investigation data. "
            "This confirms mathematical and textual integrity of the canonical investigation record; "
            "it is not a cryptographic digital signature over external PDF/DOCX binary byte encodings."
        ),
    }


@router.post("/interpret")
def interpret_investigation_question(
    payload: PreAnalysisInterpretRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Pre-analysis guided interpretation: reveals what MindEd understood, planned calculations, and consequential ambiguity."""
    verify_project_ownership(payload.project_id, current_user, db)

    from packages.analytics_core.src.engines.dataset_provider import DatabaseDatasetProvider
    from packages.analytics_core.src.governance.guided_investigation import (
        evaluate_guided_investigation,
        select_primary_dataset,
    )

    provider = DatabaseDatasetProvider(session_factory=SessionLocal)
    context = provider.acquire_context(payload.project_id, payload.dataset_ids)

    primary_name, primary_df = select_primary_dataset(payload.question, context.datasets_map or {})

    interpretation = evaluate_guided_investigation(
        question=payload.question,
        df=primary_df,
        available_columns=list(primary_df.columns) if primary_df is not None else None,
    )
    dataset_fp = context.dataset_fingerprints.get(primary_name) if primary_name else None
    from packages.analytics_core.src.governance.guided_investigation import (
        ResolverStatus,
        apply_human_guidance_to_contract,
        compute_resolved_plan_hash,
        normalize_guidance_payload,
        render_contract_preview_from_contract,
    )
    guidance = {
        key: value
        for key, value in payload.model_dump(exclude_unset=True).items()
        if key in _GUIDANCE_KEYS and value is not None
    }
    try:
        guidance = normalize_guidance_payload(guidance)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid intent guidance: {exc}") from exc
    if guidance.get("selected_dataset"):
        _sel = str(guidance["selected_dataset"]).strip()
        guidance["selected_dataset"] = next(
            (
                name
                for name, metadata in context.dataset_versions.items()
                if str(metadata.get("dataset_id")) == _sel
            ),
            _sel,
        )
    guidance_resolution = None
    if guidance:
        # Resolve through the same code path as creation so the hash shown here is the hash
        # the API will compute for this guidance.
        guidance_resolution = apply_human_guidance_to_contract(
            question=payload.question,
            data_context=context,
            guidance=guidance,
        )
        if guidance_resolution.contract is not None:
            interpretation.canonical_contract = guidance_resolution.contract
            contract = guidance_resolution.contract
            interpretation.target_metric = contract.target_column
            interpretation.group_dimension = (contract.grouping_columns or (None,))[0]
            known_slots = {
                "target_column", "numerator_column", "denominator_column", "time_column", "time_range",
                "grouping_columns", "requested_aggregation", "ranking_direction", "filter_conditions",
            }
            candidate_slot = str(guidance_resolution.unresolved_instruction or "").strip()
            if candidate_slot not in known_slots:
                candidate_slot = next(
                    (str(item.get("slot")) for item in (contract.intent_ledger or ())
                     if item.get("status") == "UNRESOLVED" and str(item.get("slot")) in known_slots),
                    "",
                )
            interpretation.pending_intent_slot = candidate_slot or None
            display_sql = render_contract_preview_from_contract(contract)
            if display_sql:
                interpretation.planned_calculation = display_sql
            if guidance_resolution.status != ResolverStatus.RESOLVED:
                interpretation.can_auto_proceed = False
                interpretation.ambiguity_status = (
                    "AMBIGUOUS_SCOPE"
                    if any(str(item.get("slot")) in {"time_column", "time_range"} for item in (contract.intent_ledger or ()))
                    else "MISSING_DATA"
                )
                interpretation.clarification_prompt = guidance_resolution.reason
                interpretation.clarification_options = list(guidance_resolution.suggested_resolutions)
                if guidance_resolution.suggested_patches:
                    interpretation.clarification_options = [item.get("label", "Apply clarification") for item in guidance_resolution.suggested_patches]
                    interpretation.clarification_option_patches = list(guidance_resolution.suggested_patches)
                interpretation.recommended_action = "Resolve the open intent slots before execution."
                interpretation.plan_hash = None
            elif contract.completeness == "INCOMPLETE" or contract.unresolved_roles:
                interpretation.can_auto_proceed = False
                interpretation.ambiguity_status = "MISSING_DATA"
                interpretation.clarification_prompt = "The canonical contract still has unresolved material intent slots."
                interpretation.clarification_options = list(guidance_resolution.suggested_resolutions)
                interpretation.plan_hash = None
            else:
                # The newly resolved contract supersedes stale clarification statuses from the
                # previous preview. Do not clear independent safety gates (causal/join risk,
                # failed verification, or unexpected assumptions) merely because a slot resolved.
                protected_statuses = {
                    "JOIN_FANOUT_RISK", "CAUSAL_ON_OBSERVATIONAL", "VERIFICATION_FAILED",
                    "UNEXPECTED_ASSUMPTION",
                }
                if interpretation.ambiguity_status not in protected_statuses:
                    interpretation.ambiguity_status = "CLEAR"
                    interpretation.can_auto_proceed = True
                    interpretation.clarification_prompt = None
                    interpretation.clarification_options = []
                    interpretation.clarification_option_patches = []
                    interpretation.recommended_action = "Execute the approved canonical contract."
                    interpretation.plan_hash = None
        resolved_name = guidance_resolution.schema_information.get("dataset") if guidance_resolution.schema_information else None
        if resolved_name:
            primary_name = resolved_name
            dataset_fp = context.dataset_fingerprints.get(primary_name)
    if interpretation.canonical_contract is not None and interpretation.can_auto_proceed and (
        guidance_resolution is None or guidance_resolution.status == ResolverStatus.RESOLVED
    ):
        interpretation.dataset_name = primary_name
        interpretation.dataset_fingerprint = dataset_fp
        interpretation.plan_hash = compute_resolved_plan_hash(
            contract=interpretation.canonical_contract,
            dataset_fingerprint=dataset_fp,
            schema_version=1,
        )

    result = interpretation.to_dict()
    if guidance_resolution is not None:
        result["guidance_resolution"] = guidance_resolution.to_dict()
        result["guidance_applied"] = guidance
    result["interpreted_dataset"] = primary_name
    result["dataset_fingerprint"] = dataset_fp
    return result


@router.post("/{investigation_id}/guidance")
def revise_investigation_guidance(
    investigation_id: str,
    payload: GuidanceRevisionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Resolve and append a correction while an investigation awaits clarification."""
    from packages.analytics_core.src.engines.dataset_provider import DatabaseDatasetProvider
    from packages.analytics_core.src.governance.guided_investigation import (
        ResolverStatus,
        apply_human_guidance_to_contract,
        normalize_guidance_payload,
    )
    from packages.analytics_core.src.intelligence.canonical_question_contract import (
        canonical_question_contract_from_dict,
    )

    inv = db.query(Investigation).filter(Investigation.id == investigation_id).first()
    if not inv:
        raise HTTPException(status_code=404, detail="Investigation not found.")
    verify_project_ownership(inv.project_id, current_user, db)
    if inv.status != InvestigationState.NEEDS_CLARIFICATION:
        raise HTTPException(
            status_code=409,
            detail="Guidance can only be revised while the investigation is awaiting clarification.",
        )
    patch = {
        key: value
        for key, value in payload.model_dump(exclude_unset=True).items()
        if value is not None and key != "plan_hash"
    }
    if not patch:
        raise HTTPException(status_code=422, detail="Provide at least one guidance field to revise.")

    event_rows = (
        db.query(InvestigationEvent)
        .filter(
            InvestigationEvent.investigation_id == investigation_id,
            InvestigationEvent.event_type.in_(
                ["investigation.human_guidance_intent", "investigation.guidance_resolution"]
            ),
        )
        .order_by(InvestigationEvent.timestamp.asc(), InvestigationEvent.id.asc())
        .all()
    )
    merged_guidance: Dict[str, Any] = {}
    for event in event_rows:
        event_payload = dict(event.event_payload_json or {})
        if event.event_type == "investigation.guidance_resolution":
            continue
        for key, value in event_payload.items():
            if key == "confirmed_assumptions":
                current = list(merged_guidance.get(key, []))
                if isinstance(value, list):
                    current = list(value)
                elif isinstance(value, dict):
                    if value.get("replace") is not None:
                        current = list(value["replace"])
                    else:
                        for item in value.get("remove") or []:
                            current = [entry for entry in current if entry != item]
                        for item in value.get("add") or []:
                            if item not in current:
                                current.append(item)
                merged_guidance[key] = current
            elif value == "":
                merged_guidance.pop(key, None)
            else:
                merged_guidance[key] = value

    for key, value in patch.items():
        if key == "confirmed_assumptions":
            current = list(merged_guidance.get(key, []))
            if isinstance(value, list):
                current = list(value)
            elif isinstance(value, dict):
                if value.get("replace") is not None:
                    current = list(value["replace"])
                else:
                    for item in value.get("remove") or []:
                        current = [entry for entry in current if entry != item]
                    for item in value.get("add") or []:
                        if item not in current:
                            current.append(item)
            merged_guidance[key] = current
        elif value == "":
            merged_guidance.pop(key, None)
        else:
            merged_guidance[key] = value

    ds_provider = DatabaseDatasetProvider(session_factory=SessionLocal)
    ds_context = ds_provider.acquire_context(
        inv.project_id,
        list(inv.requested_dataset_ids_json) if inv.requested_dataset_ids_json else None,
    )
    if ds_context.load_errors or ds_context.unavailable_requested_ids or ds_context.scope_errors:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "DATASET_SCOPE_UNAVAILABLE",
                "message": "The complete dataset scope could not be loaded for guidance resolution.",
                "load_errors": ds_context.load_errors,
                "unavailable_requested_ids": ds_context.unavailable_requested_ids,
                "scope_errors": ds_context.scope_errors,
            },
        )
    if merged_guidance.get("selected_dataset"):
        selected_dataset = str(merged_guidance["selected_dataset"]).strip()
        merged_guidance["selected_dataset"] = next(
            (
                name
                for name, metadata in ds_context.dataset_versions.items()
                if str(metadata.get("dataset_id")) == selected_dataset
            ),
            selected_dataset,
        )
    previous_resolution = next(
        (
            dict(event.event_payload_json or {})
            for event in reversed(event_rows)
            if event.event_type == "investigation.guidance_resolution"
        ),
        {},
    )
    previous_contract_payload = previous_resolution.get("contract")
    previous_contract = (
        canonical_question_contract_from_dict(previous_contract_payload)
        if isinstance(previous_contract_payload, dict)
        else None
    )
    created_event = (
        db.query(InvestigationEvent)
        .filter(
            InvestigationEvent.investigation_id == investigation_id,
            InvestigationEvent.event_type == "investigation.created",
        )
        .first()
    )
    resolution_question = (
        str((created_event.event_payload_json or {}).get("question") or inv.question)
        if created_event
        else inv.question
    )
    # A fresh slot-aware freeform answer supersedes the previously selected option patch.
    # Do not combine stale persisted patch values with a new answer to the pending slot.
    if patch.get("pending_intent_slot") and (patch.get("custom_instruction") or patch.get("clarification_response")) and "intent_patch" not in patch:
        merged_guidance.pop("intent_patch", None)
    try:
        merged_guidance = normalize_guidance_payload(merged_guidance)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid intent guidance: {exc}") from exc
    resolution = apply_human_guidance_to_contract(
        question=resolution_question,
        data_context=ds_context,
        guidance=merged_guidance,
        previous_contract=previous_contract,
    )

    db.add(InvestigationEvent(
        id=f"EVT-{gen_uuid()[:8]}",
        investigation_id=investigation_id,
        event_type="investigation.human_guidance_intent",
        event_payload_json=patch,
        timestamp=datetime.now(timezone.utc),
    ))
    db.add(InvestigationEvent(
        id=f"EVT-{gen_uuid()[:8]}",
        investigation_id=investigation_id,
        event_type="investigation.guidance_resolution",
        event_payload_json=resolution.to_dict(),
        timestamp=datetime.now(timezone.utc),
    ))
    if resolution.status != ResolverStatus.RESOLVED:
        db.commit()
        return {
            "investigation_id": investigation_id,
            "status": InvestigationState.NEEDS_CLARIFICATION,
            "job_id": None,
            "question": inv.question,
            "project_id": inv.project_id,
            "created_at": inv.created_at.isoformat() if inv.created_at else None,
            "resolution": resolution.to_dict(),
            "guidance_patch": patch,
        }
    if resolution.contract is None:
        raise HTTPException(status_code=500, detail="Resolved guidance did not produce a canonical contract.")

    dataset_name = resolution.schema_information.get("dataset")
    dataset_fp = ds_context.dataset_fingerprints.get(dataset_name)
    from packages.analytics_core.src.governance.guided_investigation import compute_resolved_plan_hash
    current_plan_hash = compute_resolved_plan_hash(
        contract=resolution.contract,
        dataset_fingerprint=dataset_fp,
        schema_version=1,
    )
    if not payload.plan_hash:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail={
                "code": "PLAN_APPROVAL_REQUIRED",
                "message": "Explicit plan approval is required to lock and execute guidance revision. Inspect and approve the plan preview.",
                "current_plan_hash": current_plan_hash,
                "resolution": resolution.to_dict(),
            },
        )
    if payload.plan_hash != current_plan_hash:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail={
                "code": "STALE_PLAN_APPROVAL",
                "message": "The approved plan hash does not match the current resolved plan. Refresh the plan preview before confirming execution.",
                "submitted_plan_hash": payload.plan_hash,
                "current_plan_hash": current_plan_hash,
                "resolution": resolution.to_dict(),
            },
        )
    db.add(InvestigationEvent(
        id=f"EVT-{gen_uuid()[:8]}",
        investigation_id=investigation_id,
        event_type="investigation.canonical_contract_locked",
        event_payload_json={
            "version": 1,
            "contract": resolution.contract.to_dict(),
            "contract_hash": resolution.contract.contract_hash(),
            "plan_hash": current_plan_hash,
            "guidance": merged_guidance,
            "dataset_name": dataset_name,
            "dataset_metadata": ds_context.dataset_versions.get(dataset_name, {}),
            "dataset_fingerprint": dataset_fp,
            "locked_at": datetime.now(timezone.utc).isoformat(),
        },
        timestamp=datetime.now(timezone.utc),
    ))
    transition_investigation(
        db=db,
        investigation_id=investigation_id,
        to_state=InvestigationState.READY,
        expected_from=InvestigationState.NEEDS_CLARIFICATION,
        actor=current_user.email,
    )
    return _dispatch_ready_investigation(
        db=db,
        investigation_id=investigation_id,
        priority=10,
        actor=current_user.email,
        question=inv.question,
        project_id=inv.project_id,
        created_at=inv.created_at,
    ) | {"resolution": resolution.to_dict(), "guidance_patch": patch}


@router.post("/{investigation_id}/review")
def submit_investigation_review(
    investigation_id: str,
    payload: HumanReviewSubmissionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Submit post-analysis human review: Accept for workflow, Request correction, Reject, or Redo.
    Enforces the invariant that human approval never overrides failed verification.
    """
    inv = db.query(Investigation).filter(Investigation.id == investigation_id).first()
    if not inv:
        raise HTTPException(status_code=404, detail="Investigation not found.")
    verify_project_ownership(inv.project_id, current_user, db)

    verifications = db.query(EvidenceVerification).filter(EvidenceVerification.investigation_id == investigation_id).all()
    evidence_items = db.query(Evidence).filter(Evidence.investigation_id == investigation_id).all()
    machine_verified = (
        len(evidence_items) > 0 and
        all(e.validation_status == "VERIFIED" for e in evidence_items) and
        all(v.status in ("PASSED", "VERIFIED") for v in verifications)
    )

    from packages.analytics_core.src.governance.guided_investigation import apply_human_review_decision
    try:
        review_record = apply_human_review_decision(
            investigation_id=investigation_id,
            action=payload.action,
            actor=current_user.email,
            comments=payload.comments,
            modified_parameters=payload.modified_parameters,
            verification_passed=machine_verified,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    _append_verification_event(db, investigation_id, "investigation.human_review", review_record)

    audit_entry = AuditLog(
        id=f"AUD-{gen_uuid()[:8]}",
        user_id=current_user.id,
        action=f"REVIEW_{payload.action.upper()}",
        resource_type="investigation",
        resource_id=investigation_id,
        details_json=review_record,
    )
    db.add(audit_entry)
    db.commit()

    return {
        "status": "success",
        "review": review_record,
        "investigation_id": investigation_id,
    }
