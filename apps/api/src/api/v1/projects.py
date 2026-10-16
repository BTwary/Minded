"""Project Management API Endpoints with Strict Tenancy & JWT Enforcement."""
import os
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from apps.api.src.core.config import settings
from apps.api.src.core.database import get_db
from apps.api.src.core.security import get_current_user, verify_project_ownership
from apps.api.src.models.entities import (
    AlertEvent,
    AlertRule,
    AnalysisRun,
    AuditLog,
    BusinessGlossary,
    BusinessMetric,
    Dashboard,
    Dataset,
    DatasetTransformation,
    DatasetVersion,
    Project,
    Report,
    SemanticModel,
    User,
)

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
    """Remove active project data while preserving saved analysis history."""
    verify_project_ownership(project_id, current_user, db)

    # Delete only objects owned by this project. Never issue an unscoped bulk delete.
    project_dataset_ids = [row[0] for row in db.query(Dataset.id).filter(Dataset.project_id == project_id).all()]
    storage_records = (
        db.query(DatasetVersion).filter(DatasetVersion.dataset_id.in_(project_dataset_ids)).all()
        if project_dataset_ids else []
    )
    cleanup_keys = set()
    for record in storage_records:
        if record.file_path:
            cleanup_keys.add(str(record.file_path))
        transformation = record.transformation_applied or {}
        raw_key = transformation.get("raw_file_path") if isinstance(transformation, dict) else None
        if raw_key:
            cleanup_keys.add(str(raw_key))

    pending_cleanup = (
        db.query(AuditLog)
        .filter(
            AuditLog.project_id == project_id,
            AuditLog.action == "workspace_reset_storage_cleanup",
        )
        .all()
    )
    pending_cleanup = [
        item for item in pending_cleanup
        if (item.details_json or {}).get("status") == "PENDING"
    ]
    cleanup_record = None

    try:
        if cleanup_keys:
            cleanup_record = AuditLog(
                project_id=project_id,
                user_id=current_user.id,
                action="workspace_reset_storage_cleanup",
                details_json={"status": "PENDING", "keys": sorted(cleanup_keys)},
            )
            db.add(cleanup_record)
            pending_cleanup.append(cleanup_record)

        if project_dataset_ids:
            db.query(DatasetTransformation).filter(
                DatasetTransformation.dataset_id.in_(project_dataset_ids)
            ).delete(synchronize_session=False)
            db.query(DatasetVersion).filter(
                DatasetVersion.dataset_id.in_(project_dataset_ids)
            ).delete(synchronize_session=False)
        db.query(Dataset).filter(Dataset.project_id == project_id).delete(synchronize_session=False)

        # Semantic models are retained as historical references by investigations,
        # but are no longer active against a workspace with no source datasets.
        db.query(SemanticModel).filter(SemanticModel.project_id == project_id).update(
            {SemanticModel.status: "deprecated"},
            synchronize_session=False,
        )

        # Preserve monitoring history while disabling rules whose source
        # dataset has been removed. Historical event detail remains available
        # and can report that its source is no longer present.
        if project_dataset_ids:
            db.query(AlertRule).filter(
                AlertRule.project_id == project_id,
                AlertRule.dataset_id.in_(project_dataset_ids),
            ).update({AlertRule.is_active: False}, synchronize_session=False)
        db.query(Dashboard).filter(Dashboard.project_id == project_id).delete(synchronize_session=False)
        db.query(Report).filter(Report.project_id == project_id).delete(synchronize_session=False)
        db.query(BusinessMetric).filter(BusinessMetric.project_id == project_id).delete(synchronize_session=False)
        db.query(BusinessGlossary).filter(BusinessGlossary.project_id == project_id).delete(synchronize_session=False)
        # AnalysisRun and Investigation records are durable history and are
        # intentionally not removed by Reset Data.
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Reset could not commit its project-scoped database changes: {exc}") from exc

    if pending_cleanup:
        try:
            from packages.analytics_core.src.providers.manager import InfrastructureManager

            provider = InfrastructureManager.get_storage_provider()
            managed_root = os.path.abspath(settings.DATA_STORAGE_DIR)
            cleanup_errors = []
            for item in pending_cleanup:
                details = dict(item.details_json or {})
                remaining = []
                for key in details.get("keys", []):
                    key = str(key)
                    try:
                        if os.path.isabs(key):
                            candidate = os.path.abspath(key)
                            if candidate == managed_root or not candidate.startswith(managed_root + os.sep):
                                raise ValueError("Absolute storage path is outside the managed dataset store.")
                            if os.path.isfile(candidate):
                                os.remove(candidate)
                        elif provider.file_exists(key):
                            provider.delete_file(key)
                    except Exception as exc:
                        remaining.append(key)
                        cleanup_errors.append(f"{key}: {type(exc).__name__}")
                details["keys"] = remaining
                details["status"] = "PENDING" if remaining else "COMPLETE"
                details["last_error"] = "; ".join(cleanup_errors) if remaining else None
                item.details_json = details
            db.commit()
            if cleanup_errors:
                raise HTTPException(
                    status_code=503,
                    detail="Active dataset records were removed, but managed storage cleanup is incomplete. Retry Reset Data; the cleanup is recorded and no successful reset is reported.",
                )
        except HTTPException:
            raise
        except Exception as exc:
            db.rollback()
            raise HTTPException(
                status_code=503,
                detail=f"Active dataset records were removed, but storage cleanup could not be confirmed. Retry Reset Data. ({type(exc).__name__})",
            ) from exc

    return {
        "status": "success",
        "message": "Project data reset complete. Saved analysis history and application preferences were preserved.",
        "removed_datasets": len(project_dataset_ids),
        "removed_dataset_versions": len(storage_records),
        "history_preserved": True,
        "storage_cleanup_pending": False,
    }


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
