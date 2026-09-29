"""Safe Direct SQL Execution API Endpoints with Tenancy & JWT Enforcement."""
from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from apps.api.src.core.database import get_db
from apps.api.src.core.security import get_current_user, verify_project_ownership
from apps.api.src.models.entities import Dataset, User
from apps.api.src.services.dataset_service import DatasetService
from packages.analytics_core.src.sql.engine import DuckDBSQLEngine, SQLSecurityError, sanitize_sql_identifier

router = APIRouter(prefix="/query", tags=["query"])


class SQLQueryRequest(BaseModel):
    project_id: str
    sql: str
    dataset_ids: Optional[List[str]] = None


@router.post("/sql")
@router.post("/execute")
def execute_safe_sql(
    payload: SQLQueryRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Execute a safe, read-only DuckDB SQL query across project datasets with strict authorization."""
    verify_project_ownership(payload.project_id, current_user, db)
    engine = DuckDBSQLEngine()
    service = DatasetService(db)

    # Register project datasets
    query = db.query(Dataset).filter(Dataset.project_id == payload.project_id)
    if payload.dataset_ids:
        query = query.filter(Dataset.id.in_(payload.dataset_ids))
    datasets = query.all()

    # The analyst asked for an explicit, named scope. Silently continuing
    # with fewer datasets than requested would let the query answer a
    # smaller question than the one actually asked, with no hard signal
    # that the scope shrank. Fail closed instead of returning a
    # warning-only partial result.
    if payload.dataset_ids:
        found_ids = {d.id for d in datasets}
        missing_ids = [did for did in payload.dataset_ids if did not in found_ids]
        if missing_ids:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Requested dataset scope could not be fully resolved. "
                    f"Missing or inaccessible dataset id(s): {', '.join(missing_ids)}. "
                    "No query was executed."
                ),
            )

    # Dataset identity for SQL registration is the analyst-visible display
    # name, but that name is not guaranteed unique within a project. Two
    # datasets sharing a sanitized name would silently collide in the
    # engine's table registry -- the second registration overwrites the
    # first, and the query would run against the wrong data with no error.
    # Detect that collision up front and fail closed rather than executing
    # against a corrupted table set.
    sanitized_names: Dict[str, List[str]] = {}
    for d in datasets:
        sanitized = sanitize_sql_identifier(d.name)
        sanitized_names.setdefault(sanitized, []).append(d.name)
    colliding = {k: v for k, v in sanitized_names.items() if len(v) > 1}
    if colliding:
        collision_desc = "; ".join(
            f"{sanitized!r} <- {names}" for sanitized, names in colliding.items()
        )
        raise HTTPException(
            status_code=409,
            detail=(
                "Dataset name collision in requested scope: multiple datasets "
                f"resolve to the same SQL table identity ({collision_desc}). "
                "Rename one of the datasets, or narrow dataset_ids to exclude "
                "the collision, before querying. No query was executed."
            ),
        )

    failed_datasets = []
    loaded_count = 0
    for d in datasets:
        try:
            df = service.get_dataset_dataframe(d.id, d.current_version)
            engine.register_dataframe(d.name, df)
            loaded_count += 1
        except Exception as e:
            failed_datasets.append(f"{d.name} ({str(e)})")

    if failed_datasets:
        # A dataset that was found but failed to load is still a shrunken
        # scope relative to what was requested/expected -- same reasoning
        # as the missing-id case above, so this fails closed too rather
        # than degrading to a "warnings" field on a 200 response.
        raise HTTPException(
            status_code=500,
            detail=f"Dataset registration failed: Could not load requested project datasets: {', '.join(failed_datasets)}",
        )

    try:
        res = engine.execute_query(payload.sql)
        res["rows"] = res.get("data", [])
        return res
    except SQLSecurityError as se:
        raise HTTPException(status_code=403, detail=f"Security policy violation: {str(se)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Query execution error: {str(e)}")
