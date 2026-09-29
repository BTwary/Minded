"""Dataset Ingestion, Profiling, and Versioning Endpoints with Tenancy Enforcement."""
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from apps.api.src.core.database import get_db
from apps.api.src.core.security import (
    get_current_user,
    verify_dataset_ownership,
    verify_project_ownership,
)
from apps.api.src.models.entities import Dataset, DatasetVersion, User
from apps.api.src.services.dataset_service import DatasetService
from packages.schemas.src.dataset import DatasetProfileSchema, DatasetResponse, DatasetVersionSchema

router = APIRouter(prefix="/datasets", tags=["datasets"])


class CleanDatasetPayload(BaseModel):
    recipe: List[Dict[str, Any]]
    description: Optional[str] = None


class DatasetVersionDiffPayload(BaseModel):
    before_version: int
    after_version: int
    key_columns: List[str] = Field(default_factory=list)


MAX_UPLOAD_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB


@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    project_id: str = Form(...),
    description: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Upload a raw dataset file, auto-profile, and convert to Parquet with project authorization."""
    verify_project_ownership(project_id, current_user, db)
    service = DatasetService(db)
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(contents) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded file exceeds maximum size limit of 500 MB ({len(contents) / (1024 * 1024):.1f} MB received).",
        )

    try:
        dataset = service.ingest_dataset_file(
            project_id=project_id,
            filename=file.filename or "uploaded_data.csv",
            file_bytes=contents,
            description=description,
        )
        version = (db.query(DatasetVersion)
                   .filter(DatasetVersion.dataset_id == dataset.id, DatasetVersion.version_number == dataset.current_version)
                   .first())
        lineage = (version.transformation_applied or {}) if version else {}
        return {
            "message": "Dataset uploaded and profiled successfully.",
            "dataset_id": dataset.id,
            "name": dataset.name,
            "row_count": dataset.row_count,
            "column_count": dataset.column_count,
            "data_quality_score": dataset.data_quality_score,
            "current_version": dataset.current_version,
            "source_filename": lineage.get("filename"),
            "source_sha256": lineage.get("source_sha256"),
            "raw_file_path": lineage.get("raw_file_path"),
            "profile": dataset.profile_json,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process dataset: {str(e)}")


@router.get("", response_model=List[Dict[str, Any]])
def list_datasets(
    project_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all datasets strictly isolated to the authenticated user's authorized projects."""
    if project_id:
        verify_project_ownership(project_id, current_user, db)
        allowed_ids = [project_id]
    else:
        from apps.api.src.core.security import get_user_authorized_project_ids
        allowed_ids = get_user_authorized_project_ids(current_user, db)

    datasets = db.query(Dataset).filter(Dataset.project_id.in_(allowed_ids)).all()
    result = []
    for d in datasets:
        v = (db.query(DatasetVersion)
             .filter(DatasetVersion.dataset_id == d.id, DatasetVersion.version_number == d.current_version)
             .first())
        lineage = (v.transformation_applied or {}) if v else {}
        result.append({
            "id": d.id,
            "project_id": d.project_id,
            "name": d.name,
            "description": d.description,
            "format": d.format,
            "row_count": d.row_count,
            "column_count": d.column_count,
            "data_quality_score": d.data_quality_score,
            "current_version": d.current_version,
            "source_filename": lineage.get("filename"),
            "source_sha256": lineage.get("source_sha256"),
            "raw_file_path": lineage.get("raw_file_path"),
            "created_at": d.created_at.isoformat() if d.created_at else None,
            "updated_at": d.updated_at.isoformat() if d.updated_at else None,
        })
    return result


@router.get("/{dataset_id}")
def get_dataset(
    dataset_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve full dataset metadata and profile."""
    dataset = verify_dataset_ownership(dataset_id, current_user, db)
    version = (db.query(DatasetVersion)
               .filter(DatasetVersion.dataset_id == dataset.id, DatasetVersion.version_number == dataset.current_version)
               .first())
    lineage = (version.transformation_applied or {}) if version else {}
    return {
        "id": dataset.id,
        "project_id": dataset.project_id,
        "name": dataset.name,
        "description": dataset.description,
        "format": dataset.format,
        "row_count": dataset.row_count,
        "column_count": dataset.column_count,
        "data_quality_score": dataset.data_quality_score,
        "current_version": dataset.current_version,
        "source_filename": lineage.get("filename"),
        "source_sha256": lineage.get("source_sha256"),
        "raw_file_path": lineage.get("raw_file_path"),
        "profile": dataset.profile_json,
        "created_at": dataset.created_at.isoformat() if dataset.created_at else None,
        "updated_at": dataset.updated_at.isoformat() if dataset.updated_at else None,
    }


@router.get("/{dataset_id}/data")
def get_dataset_data(
    dataset_id: str,
    limit: int = 100,
    version: Optional[int] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve preview sample rows from dataset version with ownership check."""
    verify_dataset_ownership(dataset_id, current_user, db)
    service = DatasetService(db)
    try:
        rows = service.get_dataset_sample(dataset_id, limit=limit, version=version)
        return {"dataset_id": dataset_id, "row_count": len(rows), "rows": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch data preview: {str(e)}")


@router.post("/{dataset_id}/clean")
def clean_dataset(
    dataset_id: str,
    payload: CleanDatasetPayload,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Execute cleaning transformations and increment immutable dataset version."""
    verify_dataset_ownership(dataset_id, current_user, db)
    service = DatasetService(db)
    try:
        updated_dataset, new_version = service.clean_dataset(
            dataset_id=dataset_id,
            recipe=payload.recipe,
            description=payload.description or "Automated data cleaning transformation.",
        )
        return {
            "message": f"Successfully created dataset version v{new_version.version_number}.",
            "new_version": new_version.version_number,
            "row_count": new_version.row_count,
            "column_count": new_version.column_count,
            "data_quality_score": updated_dataset.data_quality_score,
            "applied_recipe": new_version.applied_recipe,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clean dataset: {str(e)}")


@router.post("/{dataset_id}/versions/diff")
def diff_dataset_versions(
    dataset_id: str,
    payload: DatasetVersionDiffPayload,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Compare two immutable versions of one authorized dataset without mutation."""
    verify_dataset_ownership(dataset_id, current_user, db)
    service = DatasetService(db)
    try:
        diff = service.compare_dataset_versions(
            dataset_id=dataset_id,
            before_version=payload.before_version,
            after_version=payload.after_version,
            key_columns=payload.key_columns or None,
        )
        return {
            "dataset_id": dataset_id,
            "before_version": payload.before_version,
            "after_version": payload.after_version,
            "diff": diff.to_dict(),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compare dataset versions: {e}")


@router.get("/{dataset_id}/versions")
def list_dataset_versions(
    dataset_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List immutable lineage versions for a dataset."""
    verify_dataset_ownership(dataset_id, current_user, db)
    versions = (
        db.query(DatasetVersion)
        .filter(DatasetVersion.dataset_id == dataset_id)
        .order_by(DatasetVersion.version_number.desc())
        .all()
    )
    return [
        {
            "id": v.id,
            "version_number": v.version_number,
            "row_count": v.row_count,
            "column_count": v.column_count,
            "file_size_bytes": v.file_size_bytes,
            "transformation_applied": v.transformation_applied,
            "created_at": v.created_at.isoformat() if v.created_at else None,
        }
        for v in versions
    ]
