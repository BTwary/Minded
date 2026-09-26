"""Business Semantic Metrics API Endpoints with Strict Tenancy & Multi-Project Isolation."""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from apps.api.src.core.database import get_db
from apps.api.src.core.security import (
    get_current_user,
    get_user_authorized_project_ids,
    verify_project_ownership,
)
from apps.api.src.models.entities import BusinessMetric, User
from packages.schemas.src.semantic import BusinessMetricCreate, BusinessMetricResponse

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("", response_model=List[BusinessMetricResponse])
def list_metrics(
    project_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List business metrics strictly isolated to authorized user projects."""
    if project_id:
        verify_project_ownership(project_id, current_user, db)
        allowed_ids = [project_id]
    else:
        allowed_ids = get_user_authorized_project_ids(current_user, db)

    metrics = db.query(BusinessMetric).filter(BusinessMetric.project_id.in_(allowed_ids)).all()
    return [
        BusinessMetricResponse(
            id=m.id,
            project_id=m.project_id,
            name=m.name,
            display_name=m.display_name,
            description=m.description,
            sql_formula=m.sql_formula,
            unit=m.unit,
            category=m.category,
            dimensions=m.dimensions_json or [],
            filters=m.filters,
            synonyms=m.synonyms_json or [],
            created_at=m.created_at,
            updated_at=m.updated_at,
        )
        for m in metrics
    ]


@router.post("", response_model=BusinessMetricResponse)
def create_metric(
    payload: BusinessMetricCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new business metric definition with project verification."""
    verify_project_ownership(payload.project_id, current_user, db)
    metric = BusinessMetric(
        project_id=payload.project_id,
        name=payload.name,
        display_name=payload.display_name,
        description=payload.description,
        sql_formula=payload.sql_formula,
        unit=payload.unit,
        category=payload.category,
        dimensions_json=payload.dimensions,
        filters=payload.filters,
        synonyms_json=payload.synonyms,
    )
    db.add(metric)
    db.commit()
    db.refresh(metric)
    return BusinessMetricResponse(
        id=metric.id,
        project_id=metric.project_id,
        name=metric.name,
        display_name=metric.display_name,
        description=metric.description,
        sql_formula=metric.sql_formula,
        unit=metric.unit,
        category=metric.category,
        dimensions=metric.dimensions_json or [],
        filters=metric.filters,
        synonyms=metric.synonyms_json or [],
        created_at=metric.created_at,
        updated_at=metric.updated_at,
    )
