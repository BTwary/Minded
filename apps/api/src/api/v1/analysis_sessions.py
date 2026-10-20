"""Persistent multi-question analysis sessions and cumulative report exports."""
from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from apps.api.src.core.database import get_db
from apps.api.src.core.security import get_current_user, get_user_authorized_project_ids, verify_project_ownership
from apps.api.src.models.entities import AnalysisSession, Investigation, InvestigationEvent, InvestigationJob, Project, User, gen_uuid

router = APIRouter(prefix="/analysis-sessions", tags=["analysis-sessions"])


class CreateAnalysisSessionRequest(BaseModel):
    project_id: str
    title: Optional[str] = None
    dataset_ids: list[str] = Field(default_factory=list)
    scope_filter: Optional[str] = None


class UpdateAnalysisSessionRequest(BaseModel):
    title: Optional[str] = None
    scope_filter: Optional[str] = None


def _iso(value: Any) -> Optional[str]:
    return value.isoformat() if value else None


def _session_or_404(db: Session, current_user: User, session_id: str) -> AnalysisSession:
    row = db.query(AnalysisSession).filter(AnalysisSession.id == session_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Analysis session not found.")
    if row.user_id and row.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Analysis session not found.")
    verify_project_ownership(row.project_id, current_user, db)
    return row


def _session_payload(db: Session, session: AnalysisSession, include_questions: bool = True) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": session.id,
        "project_id": session.project_id,
        "user_id": session.user_id,
        "title": session.title,
        "status": session.status,
        "dataset_ids": list(session.dataset_ids_json or []),
        "scope_filter": session.scope_filter,
        "question_count": int(session.question_count or 0),
        "accepted_investigation_ids": list(session.accepted_investigation_ids_json or []),
        "accepted_question_count": len(session.accepted_investigation_ids_json or []),
        "report_revision": int(session.report_revision or 0),
        "created_at": _iso(session.created_at),
        "updated_at": _iso(session.updated_at),
        "closed_at": _iso(session.closed_at),
    }
    if include_questions:
        investigations = db.query(Investigation).filter(
            Investigation.analysis_session_id == session.id,
            Investigation.is_subinvestigation.is_(False),
        ).order_by(Investigation.created_at.asc(), Investigation.id.asc()).all()
        ids = [row.id for row in investigations]
        latest_reviews: dict[str, dict[str, Any]] = {}
        if ids:
            events = db.query(InvestigationEvent).filter(
                InvestigationEvent.investigation_id.in_(ids),
                InvestigationEvent.event_type == "investigation.human_review",
            ).order_by(InvestigationEvent.timestamp.asc(), InvestigationEvent.id.asc()).all()
            for event in events:
                latest_reviews[event.investigation_id] = dict(event.event_payload_json or {})
        payload["questions"] = [
            {
                "id": row.id,
                "question": row.question,
                "status": row.status,
                "verdict_type": row.verdict_type,
                "direct_answer": row.direct_answer,
                "main_finding": row.main_finding,
                "review_status": latest_reviews.get(row.id, {}).get("review_status"),
                "created_at": _iso(row.created_at),
                "updated_at": _iso(row.updated_at),
                "accepted_into_report": row.id in (session.accepted_investigation_ids_json or []),
            }
            for row in investigations
        ]
    return payload


@router.post("")
def create_analysis_session(
    payload: CreateAnalysisSessionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    verify_project_ownership(payload.project_id, current_user, db)
    dataset_ids = list(dict.fromkeys(str(value) for value in payload.dataset_ids if str(value).strip()))
    if len(dataset_ids) != len(payload.dataset_ids):
        raise HTTPException(status_code=400, detail="Dataset scope contains duplicate or blank IDs.")
    if dataset_ids:
        from apps.api.src.models.entities import Dataset
        found = {str(row[0]) for row in db.query(Dataset.id).filter(
            Dataset.project_id == payload.project_id, Dataset.id.in_(dataset_ids)
        ).all()}
        missing = [item for item in dataset_ids if item not in found]
        if missing:
            raise HTTPException(status_code=400, detail={"code": "DATASET_SCOPE_INVALID", "missing_dataset_ids": missing})
    title = (payload.title or "").strip() or f"Analysis · {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    session = AnalysisSession(
        id=gen_uuid(),
        project_id=payload.project_id,
        user_id=current_user.id,
        title=title[:255],
        status="OPEN",
        dataset_ids_json=dataset_ids,
        scope_filter=(payload.scope_filter or "").strip() or None,
        question_count=0,
        accepted_investigation_ids_json=[],
        report_revision=0,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return _session_payload(db, session)


@router.get("")
def list_analysis_sessions(
    project_id: Optional[str] = None,
    limit: int = Query(default=100, ge=1, le=1000),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    allowed = get_user_authorized_project_ids(current_user, db)
    if project_id:
        verify_project_ownership(project_id, current_user, db)
        allowed = [project_id]
    rows = db.query(AnalysisSession).filter(
        AnalysisSession.project_id.in_(allowed or ["__no_authorized_projects__"]),
        (AnalysisSession.user_id == current_user.id) | (AnalysisSession.user_id.is_(None)),
    ).order_by(AnalysisSession.updated_at.desc()).limit(limit).all()
    return {"sessions": [_session_payload(db, row, include_questions=False) for row in rows]}


@router.get("/{session_id}")
def get_analysis_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    session = _session_or_404(db, current_user, session_id)
    return _session_payload(db, session)


@router.patch("/{session_id}")
def update_analysis_session(
    session_id: str,
    payload: UpdateAnalysisSessionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    session = _session_or_404(db, current_user, session_id)
    if str(session.status).upper() != "OPEN":
        raise HTTPException(status_code=409, detail="Closed analysis sessions cannot be edited.")
    changed = False
    if payload.title is not None:
        title = payload.title.strip()
        if not title:
            raise HTTPException(status_code=422, detail="Analysis title cannot be empty.")
        session.title = title[:255]
        changed = True
    if payload.scope_filter is not None:
        session.scope_filter = payload.scope_filter.strip() or None
        changed = True
    if changed:
        session.updated_at = datetime.now(timezone.utc)
        session.report_revision = int(session.report_revision or 0) + 1
        db.commit()
        db.refresh(session)
    return _session_payload(db, session)


@router.post("/{session_id}/close")
def close_analysis_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    session = _session_or_404(db, current_user, session_id)
    if str(session.status).upper() != "CLOSED":
        active = db.query(Investigation).filter(
            Investigation.analysis_session_id == session.id,
            Investigation.status.in_(["PLANNED", "READY", "QUEUED", "CLAIMED", "RUNNING", "VERIFYING", "WAITING_FOR_RESOURCE", "CANCEL_REQUESTED"]),
        ).count()
        if active:
            raise HTTPException(status_code=409, detail=f"Cannot close this analysis while {active} question run(s) are still active.")
        session.status = "CLOSED"
        session.closed_at = datetime.now(timezone.utc)
        session.updated_at = session.closed_at
        session.report_revision = int(session.report_revision or 0) + 1
        db.commit()
        db.refresh(session)
    return {"status": "CLOSED", "session": _session_payload(db, session), "export_url": f"/api/v1/analysis-sessions/{session.id}/export?format=pdf"}


@router.get("/{session_id}/export")
def export_analysis_session(
    session_id: str,
    format: str = Query(default="pdf", pattern="^(pdf|docx)$"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    session = _session_or_404(db, current_user, session_id)
    accepted_ids = list(session.accepted_investigation_ids_json or [])
    if not accepted_ids:
        raise HTTPException(status_code=409, detail="The report has no accepted and verified question results yet.")
    investigations = db.query(Investigation).filter(
        Investigation.id.in_(accepted_ids),
        Investigation.analysis_session_id == session.id,
        Investigation.project_id == session.project_id,
        Investigation.status.in_(["COMPLETED", "PARTIAL"]),
    ).order_by(Investigation.created_at.asc(), Investigation.id.asc()).all()
    if not investigations:
        raise HTTPException(status_code=409, detail="No completed accepted question results are available for this report.")
    from apps.api.src.api.v1.investigations import get_investigation
    question_data = [get_investigation(row.id, current_user, db) for row in investigations]
    if any(str(row.get("review_status") or "").upper() != "ACCEPTED_AND_VERIFIED" for row in question_data):
        raise HTTPException(status_code=409, detail="One or more accepted report sections no longer have verified review status. Review the questions before exporting.")

    from packages.analytics_core.src.reporting.session_exporter import export_analysis_session_to_docx, export_analysis_session_to_pdf
    session_data = _session_payload(db, session, include_questions=False)
    try:
        if format.lower() == "docx":
            content = export_analysis_session_to_docx(session_data, question_data)
            media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ext = "docx"
        else:
            content = export_analysis_session_to_pdf(session_data, question_data)
            media_type = "application/pdf"
            ext = "pdf"
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Cumulative report generation failed: {type(exc).__name__}: {exc}") from exc
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="MindEd_Analysis_{session.id[:8]}.{ext}"',
                 "X-MindEd-Analysis-Session": session.id,
                 "X-MindEd-Report-Revision": str(session.report_revision or 0)},
    )
