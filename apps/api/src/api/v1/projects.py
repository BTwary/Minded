"""Project Management API Endpoints with Strict Tenancy & JWT Enforcement."""
import os
import shutil
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from apps.api.src.core.config import settings
from apps.api.src.core.database import get_db
from apps.api.src.core.security import get_current_user, verify_project_ownership
from apps.api.src.models.entities import AlertEvent, AlertRule, AnalysisRun, BusinessGlossary, BusinessMetric, Dashboard, Dataset, DatasetVersion, Project, Report, User
from apps.api.src.services.dataset_service import DatasetService

router = APIRouter(prefix="/projects", tags=["projects"])


class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None
    org_id: Optional[str] = "default-org"


class ProjectResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    org_id: str
    owner_id: str


@router.get("", response_model=List[ProjectResponse])
def list_projects(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List projects belonging to the authenticated user using centralized tenancy enforcement."""
    from apps.api.src.core.security import get_user_authorized_project_ids

    auth_ids = get_user_authorized_project_ids(current_user, db)
    projects = db.query(Project).filter(Project.id.in_(auth_ids)).all()

    return [
        ProjectResponse(
            id=p.id,
            name=p.name,
            description=p.description,
            org_id=p.org_id,
            owner_id=p.owner_id,
        )
        for p in projects
    ]


@router.post("", response_model=ProjectResponse)
def create_project(
    payload: ProjectCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new project owned by the authenticated user."""
    project = Project(
        name=payload.name,
        description=payload.description,
        org_id=current_user.organization_id or payload.org_id or "default-org",
        owner_id=current_user.id,
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    return ProjectResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        org_id=project.org_id,
        owner_id=project.owner_id,
    )


@router.get("/summary")
def get_project_summary(
    project_id: str = "proj-default",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Fetch real-time dynamic summary and metrics for the workspace with ownership validation."""
    verify_project_ownership(project_id, current_user, db)

    datasets = db.query(Dataset).filter(Dataset.project_id == project_id).all()
    total_datasets = len(datasets)
    
    valid_scores = [d.data_quality_score for d in datasets if d.data_quality_score is not None]
    avg_quality = round(sum(valid_scores) / len(valid_scores), 1) if valid_scores else 0.0
    total_rows = sum(d.row_count or 0 for d in datasets)
    total_cols = sum(d.column_count or 0 for d in datasets)

    recent_analyses = (
        db.query(AnalysisRun)
        .filter(AnalysisRun.project_id == project_id)
        .order_by(AnalysisRun.created_at.desc())
        .limit(5)
        .all()
    )

    recent_alerts = (
        db.query(AlertEvent)
        .order_by(AlertEvent.triggered_at.desc())
        .limit(5)
        .all()
    )

    return {
        "project_id": project_id,
        "total_datasets": total_datasets,
        "avg_data_quality": avg_quality,
        "total_rows": total_rows,
        "total_columns": total_cols,
        "engine": "DuckDB (In-Process OLAP)",
        "ai_provider": settings.AI_PROVIDER,
        "infra_cost": "$0.00",
        "datasets": [
            {
                "id": d.id,
                "name": d.name,
                "description": d.description,
                "format": d.format,
                "current_version": d.current_version,
                "row_count": d.row_count,
                "column_count": d.column_count,
                "data_quality_score": d.data_quality_score,
                "updated_at": d.updated_at.isoformat() if d.updated_at else None,
            }
            for d in datasets
        ],
        "recent_analyses": [
            {
                "id": a.id,
                "question": a.question,
                "direct_answer": a.direct_answer,
                "confidence": a.confidence,
                "status": a.status,
                "created_at": a.created_at.isoformat() if a.created_at else None,
            }
            for a in recent_analyses
        ],
        "recent_alerts": [
            {
                "id": al.id,
                "rule_id": al.rule_id,
                "severity": al.severity,
                "summary": al.title,
                "details": al.description,
                "created_at": al.triggered_at.isoformat() if al.triggered_at else None,
            }
            for al in recent_alerts
        ],
    }


@router.post("/reset")
def reset_project(
    project_id: str = "proj-default",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Reset all workspace data to a clean slate (0 datasets, 0 analyses)."""
    verify_project_ownership(project_id, current_user, db)

    # Delete only objects owned by this project. Never issue an unscoped bulk delete.
    project_dataset_ids = [row[0] for row in db.query(Dataset.id).filter(Dataset.project_id == project_id).all()]
    project_rule_ids = [row[0] for row in db.query(AlertRule.id).filter(AlertRule.project_id == project_id).all()]

    # Capture exact managed storage keys before deleting DB rows. This preserves
    # project isolation without deleting another project's objects.
    storage_records = []
    if project_dataset_ids:
        storage_records = db.query(DatasetVersion).filter(DatasetVersion.dataset_id.in_(project_dataset_ids)).all()
        db.query(DatasetVersion).filter(DatasetVersion.dataset_id.in_(project_dataset_ids)).delete(synchronize_session=False)
    db.query(Dataset).filter(Dataset.project_id == project_id).delete(synchronize_session=False)
    db.query(AnalysisRun).filter(AnalysisRun.project_id == project_id).delete(synchronize_session=False)
    if project_rule_ids:
        db.query(AlertEvent).filter(AlertEvent.rule_id.in_(project_rule_ids), AlertEvent.project_id == project_id).delete(synchronize_session=False)
    db.query(AlertRule).filter(AlertRule.project_id == project_id).delete(synchronize_session=False)
    db.query(Dashboard).filter(Dashboard.project_id == project_id).delete()
    db.query(Report).filter(Report.project_id == project_id).delete()
    db.query(BusinessMetric).filter(BusinessMetric.project_id == project_id).delete()
    db.query(BusinessGlossary).filter(BusinessGlossary.project_id == project_id).delete()
    db.commit()

    # Remove only storage objects referenced by this project's dataset versions.
    # Works for local and configured cloud providers through the same abstraction.
    try:
        from apps.api.src.services.storage_service import StorageService
        storage = StorageService()
        for record in storage_records:
            for key in (record.file_path,):
                if key:
                    try:
                        if os.path.isabs(key):
                            managed_root = os.path.abspath(settings.DATA_STORAGE_DIR)
                            candidate = os.path.abspath(key)
                            if candidate == managed_root or candidate.startswith(managed_root + os.sep):
                                if os.path.isfile(candidate):
                                    os.remove(candidate)
                        elif storage.provider.file_exists(key):
                            storage.provider.delete_file(key)
                    except Exception:
                        # DB reset remains authoritative; a storage cleanup failure
                        # must not expand deletion scope or delete unrelated data.
                        pass
            transformation = record.transformation_applied or {}
            raw_key = transformation.get("raw_file_path") if isinstance(transformation, dict) else None
            if raw_key and raw_key != record.file_path:
                try:
                    if os.path.isabs(raw_key):
                        managed_root = os.path.abspath(settings.DATA_STORAGE_DIR)
                        candidate = os.path.abspath(raw_key)
                        if candidate == managed_root or candidate.startswith(managed_root + os.sep):
                            if os.path.isfile(candidate):
                                os.remove(candidate)
                    elif storage.provider.file_exists(raw_key):
                        storage.provider.delete_file(raw_key)
                except Exception:
                    pass
    except Exception:
        pass

    return {"status": "success", "message": "Project reset complete; only this project's datasets, alerts, and artifacts were removed."}


@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get project details with ownership check."""
    project = verify_project_ownership(project_id, current_user, db)
    return ProjectResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        org_id=project.org_id,
        owner_id=project.owner_id,
    )
