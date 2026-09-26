"""Dashboards API Endpoints with Strict Tenancy & Multi-Project Isolation."""
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from apps.api.src.core.database import get_db
from apps.api.src.core.security import (
    get_current_user,
    get_user_authorized_project_ids,
    verify_project_ownership,
)
from apps.api.src.models.entities import Dashboard, User
from packages.schemas.src.dashboard import DashboardCreate, DashboardResponse, WidgetSchema

router = APIRouter(prefix="/dashboards", tags=["dashboards"])


@router.get("", response_model=List[DashboardResponse])
def list_dashboards(
    project_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List dashboards strictly isolated to authorized user projects."""
    if project_id:
        verify_project_ownership(project_id, current_user, db)
        allowed_ids = [project_id]
    else:
        allowed_ids = get_user_authorized_project_ids(current_user, db)

    dashboards = db.query(Dashboard).filter(Dashboard.project_id.in_(allowed_ids)).all()
    return [
        DashboardResponse(
            id=d.id,
            project_id=d.project_id,
            title=d.title,
            description=d.description,
            widgets=[WidgetSchema.model_validate(w) for w in (d.widgets_json or [])],
            filters=d.filters_json or {},
            created_at=d.created_at,
            updated_at=d.updated_at,
        )
        for d in dashboards
    ]


@router.post("", response_model=DashboardResponse)
def create_dashboard(
    payload: DashboardCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new dashboard with project authorization."""
    verify_project_ownership(payload.project_id, current_user, db)
    dashboard = Dashboard(
        project_id=payload.project_id,
        title=payload.title,
        description=payload.description,
        widgets_json=[w.model_dump() for w in payload.widgets],
        filters_json=payload.filters or {},
    )
    db.add(dashboard)
    db.commit()
    db.refresh(dashboard)
    return DashboardResponse(
        id=dashboard.id,
        project_id=dashboard.project_id,
        title=dashboard.title,
        description=dashboard.description,
        widgets=payload.widgets,
        filters=dashboard.filters_json or {},
        created_at=dashboard.created_at,
        updated_at=dashboard.updated_at,
    )


@router.get("/{dashboard_id}", response_model=DashboardResponse)
def get_dashboard(
    dashboard_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get dashboard details with ownership verification."""
    dashboard = db.query(Dashboard).filter(Dashboard.id == dashboard_id).first()
    if not dashboard:
        raise HTTPException(status_code=404, detail="Dashboard not found.")
    verify_project_ownership(dashboard.project_id, current_user, db)
    return DashboardResponse(
        id=dashboard.id,
        project_id=dashboard.project_id,
        title=dashboard.title,
        description=dashboard.description,
        widgets=[WidgetSchema.model_validate(w) for w in (dashboard.widgets_json or [])],
        filters=dashboard.filters_json or {},
        created_at=dashboard.created_at,
        updated_at=dashboard.updated_at,
    )
