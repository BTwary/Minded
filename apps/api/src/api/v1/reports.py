"""Reports API Endpoints with Strict Tenancy & Multi-Project Isolation."""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from apps.api.src.core.database import get_db
from apps.api.src.core.security import (
    get_current_user,
    get_user_authorized_project_ids,
    verify_project_ownership,
)
from apps.api.src.models.entities import AnalysisRun, Report, User
from packages.schemas.src.report import ReportCreate, ReportResponse

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("", response_model=List[ReportResponse])
def list_reports(
    project_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List reports strictly isolated to authorized user projects."""
    if project_id:
        verify_project_ownership(project_id, current_user, db)
        allowed_ids = [project_id]
    else:
        allowed_ids = get_user_authorized_project_ids(current_user, db)

    reports = db.query(Report).filter(Report.project_id.in_(allowed_ids)).all()
    return [
        ReportResponse(
            id=r.id,
            project_id=r.project_id,
            analysis_id=r.analysis_id,
            title=r.title,
            report_type=r.report_type,
            markdown_content=r.markdown_content,
            pdf_artifact_path=r.pdf_artifact_path,
            created_at=r.created_at,
        )
        for r in reports
    ]


@router.post("", response_model=ReportResponse)
def create_report(
    payload: ReportCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Generate executive and technical reports based on an analysis run with project authorization."""
    verify_project_ownership(payload.project_id, current_user, db)
    markdown_text = f"# {payload.title}\n\nGenerated on {payload.report_type.title()} Report.\n\n"
    
    if payload.analysis_id:
        run = db.query(AnalysisRun).filter(AnalysisRun.id == payload.analysis_id).first()
        if not run:
            raise HTTPException(status_code=404, detail="Analysis run not found.")
        if run.project_id != payload.project_id:
            raise HTTPException(
                status_code=403,
                detail="Cross-project isolation violation: analysis run belongs to a different project.",
            )
        markdown_text += (
            f"## Executive Summary\n\n"
            f"**Direct Answer**: {run.direct_answer}\n\n"
            f"**Main Finding**: {run.main_finding}\n\n"
            f"**Confidence**: {run.confidence}\n\n"
            f"### Evidence & Auditable Trail\n\n"
        )
        for ev in (run.evidence_json or []):
            markdown_text += f"- **Statement**: {ev.get('statement')}\n  - *Validation*: {ev.get('validation_status')}\n  - *Rows analyzed*: {ev.get('row_count_analyzed'):,}\n\n"

        # Preserve human verification sign-off state into exported report
        from apps.api.src.models.entities import InvestigationEvent
        signoff_event = (
            db.query(InvestigationEvent)
            .filter(
                InvestigationEvent.investigation_id == run.id,
                InvestigationEvent.event_type == "verification.signed_off",
            )
            .order_by(InvestigationEvent.sequence.desc())
            .first()
        )
        markdown_text += "### Human Verification & Sign-Off Status\n\n"
        if signoff_event and signoff_event.event_payload_json:
            so = signoff_event.event_payload_json
            reviewer = (so.get("reviewer") or {}).get("email", "unknown")
            outcome = so.get("outcome", "SIGNED_OFF")
            comment = so.get("comment", "")
            fingerprint = so.get("packet_fingerprint", "N/A")
            at = signoff_event.timestamp.isoformat() if signoff_event.timestamp else "N/A"
            markdown_text += (
                f"- **Status**: SIGNED_OFF ({outcome})\n"
                f"- **Reviewer**: {reviewer}\n"
                f"- **Signed At**: {at}\n"
                f"- **Packet Fingerprint**: `{str(fingerprint)[:16]}...`\n"
            )
            if comment:
                markdown_text += f"- **Verifier Comment**: {comment}\n"
            markdown_text += "\n"
        else:
            markdown_text += "- **Status**: PENDING_REVIEW (No formal human sign-off recorded)\n\n"

    report = Report(
        project_id=payload.project_id,
        analysis_id=payload.analysis_id,
        title=payload.title,
        report_type=payload.report_type,
        markdown_content=markdown_text,
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return ReportResponse(
        id=report.id,
        project_id=report.project_id,
        analysis_id=report.analysis_id,
        title=report.title,
        report_type=report.report_type,
        markdown_content=report.markdown_content,
        created_at=report.created_at,
    )
