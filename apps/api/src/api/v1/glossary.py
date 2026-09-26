"""Business Glossary API Endpoints with Strict Tenancy & Multi-Project Isolation."""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from apps.api.src.core.database import get_db
from apps.api.src.core.security import (
    get_current_user,
    get_user_authorized_project_ids,
    verify_project_ownership,
)
from apps.api.src.models.entities import BusinessGlossary, User
from packages.schemas.src.semantic import GlossaryTermCreate, GlossaryTermResponse

router = APIRouter(prefix="/glossary", tags=["glossary"])


@router.get("", response_model=List[GlossaryTermResponse])
def list_glossary_terms(
    project_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List business glossary terms strictly isolated to authorized user projects."""
    if project_id:
        verify_project_ownership(project_id, current_user, db)
        allowed_ids = [project_id]
    else:
        allowed_ids = get_user_authorized_project_ids(current_user, db)

    terms = db.query(BusinessGlossary).filter(BusinessGlossary.project_id.in_(allowed_ids)).all()
    return [
        GlossaryTermResponse(
            id=t.id,
            project_id=t.project_id,
            term=t.term,
            definition=t.definition,
            category=t.category,
            synonyms=t.synonyms_json or [],
            related_metrics=t.related_metrics_json or [],
            related_columns=t.related_columns_json or [],
            created_at=t.created_at,
            updated_at=t.updated_at,
        )
        for t in terms
    ]


@router.post("", response_model=GlossaryTermResponse)
def create_glossary_term(
    payload: GlossaryTermCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new glossary definition with project verification."""
    verify_project_ownership(payload.project_id, current_user, db)
    term = BusinessGlossary(
        project_id=payload.project_id,
        term=payload.term,
        definition=payload.definition,
        category=payload.category,
        synonyms_json=payload.synonyms,
        related_metrics_json=payload.related_metrics,
        related_columns_json=payload.related_columns,
    )
    db.add(term)
    db.commit()
    db.refresh(term)
    return GlossaryTermResponse(
        id=term.id,
        project_id=term.project_id,
        term=term.term,
        definition=term.definition,
        category=term.category,
        synonyms=term.synonyms_json or [],
        related_metrics=term.related_metrics_json or [],
        related_columns=term.related_columns_json or [],
        created_at=term.created_at,
        updated_at=term.updated_at,
    )
