"""Autonomous Analysis API Endpoints with Tenancy & JWT Enforcement."""
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from apps.api.src.core.database import get_db
from apps.api.src.core.security import get_current_user, verify_project_ownership
from apps.api.src.models.entities import AnalysisRun, ClaimGateDecision, User
from apps.api.src.services.analysis_service import AnalysisService
from packages.schemas.src.analysis import AnalysisCreate, AnalysisResponse

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.post("", response_model=AnalysisResponse)
def create_analysis(
    payload: AnalysisCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Run an autonomous AI analysis on project datasets with authenticated ownership check."""
    verify_project_ownership(payload.project_id, current_user, db)
    service = AnalysisService(db)
    try:
        response = service.execute_analysis(payload, current_user.id)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Autonomous analysis failed: {str(e)}")


@router.get("/{analysis_id}")
def get_analysis(
    analysis_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve full analysis run, steps, findings, evidence, and manifest."""
    run = db.query(AnalysisRun).filter(AnalysisRun.id == analysis_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Analysis run not found.")

    verify_project_ownership(run.project_id, current_user, db)

    gate = (
        db.query(ClaimGateDecision)
        .filter(ClaimGateDecision.investigation_id == analysis_id)
        .order_by(ClaimGateDecision.created_at.desc())
        .first()
    )

    return {
        "id": run.id,
        "project_id": run.project_id,
        "question": run.question,
        "status": run.status,
        "direct_answer": run.direct_answer,
        "main_finding": run.main_finding,
        "confidence": run.confidence,
        "hypotheses": run.hypotheses_json,
        "steps": run.steps_json,
        "findings": run.findings_json,
        "evidence": run.evidence_json,
        "evidence_integrity": (run.manifest_json or {}).get("evidence_integrity", {
            "evidence_count": len(run.evidence_json or []),
            "verified_evidence_count": sum(1 for item in (run.evidence_json or []) if item.get("validation_status") == "PASSED"),
            "has_evidence": bool(run.evidence_json),
        }),
        "manifest": run.manifest_json,
        "analysis_mode": (run.manifest_json or {}).get("analysis_mode", "DETERMINISTIC"),
        "claim_gate": (run.manifest_json or {}).get("claim_gate") or ({
            "outcome": gate.outcome,
            "requested_claim": gate.requested_claim,
            "evidence_level": gate.evidence_level,
            "design_status": gate.design_status,
            "assumptions": gate.assumptions_json or [],
            "allowed_claim": gate.allowed_claim,
            "blocked_claim": gate.blocked_claim,
            "reason": gate.reason,
            "recovery_actions": gate.recovery_actions_json or [],
            "computed_evidence": gate.computed_evidence_json or {},
            "refusal_id": gate.id if gate.outcome == "REFUSE" else None,
            "requested_level": gate.requested_level,
            "max_supported_level": gate.max_supported_level,
            "identification_strategy": gate.identification_strategy,
            "blocking_conditions": gate.blocking_conditions_json or [],
            "missing_evidence": gate.missing_evidence_json or [],
        } if gate else None),
        "explanation": (run.manifest_json or {}).get("explanation"),
        "created_at": run.created_at.isoformat() if run.created_at else None,
    }


@router.get("", response_model=List[Dict[str, Any]])
def list_analyses(
    project_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List historical analysis runs strictly isolated to authorized user projects."""
    if project_id:
        verify_project_ownership(project_id, current_user, db)
        allowed_ids = [project_id]
    else:
        from apps.api.src.core.security import get_user_authorized_project_ids
        allowed_ids = get_user_authorized_project_ids(current_user, db)

    runs = (
        db.query(AnalysisRun)
        .filter(AnalysisRun.project_id.in_(allowed_ids))
        .order_by(AnalysisRun.created_at.desc())
        .all()
    )

    return [
        {
            "id": r.id,
            "project_id": r.project_id,
            "question": r.question,
            "status": r.status,
            "direct_answer": r.direct_answer,
            "main_finding": r.main_finding,
            "confidence": r.confidence,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in runs
    ]
