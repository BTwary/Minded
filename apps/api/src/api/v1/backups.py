"""Optional user-controlled cloud backup and migration APIs."""
from __future__ import annotations

import os
import socket
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from apps.api.src.core.database import get_db, engine
from apps.api.src.core.security import get_current_user
from apps.api.src.models.entities import Base as OrmBase, User
from apps.api.src.services.backup_service import BackupService, CloudStorageConfig
from apps.api.src.core.config import settings

router = APIRouter(prefix="/backups", tags=["backups"])


class CloudStorageRequest(BaseModel):
    provider: str = Field(pattern="^(s3|gcs)$")
    bucket: str = Field(min_length=1, max_length=512)
    region: str = "us-east-1"
    endpoint_url: Optional[str] = None
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    project_id: Optional[str] = None


class BackupCreateRequest(CloudStorageRequest):
    passphrase: str = Field(min_length=8)
    include_source_files: bool = True


class BackupRestoreRequest(CloudStorageRequest):
    object_key: str = Field(min_length=1)
    passphrase: str = Field(min_length=8)
    replace_existing: bool = False


class BackupInspectRequest(BaseModel):
    passphrase: Optional[str] = None


def _service() -> BackupService:
    return BackupService(
        engine=engine,
        metadata=OrmBase.metadata,
        storage_root=settings.DATA_STORAGE_DIR,
        app_version=os.getenv("AAOS_VERSION", "dev"),
        schema_revision=os.getenv("AAOS_SCHEMA_REVISION", "current"),
    )


from packages.analytics_core.src.security.ssrf import assert_safe_endpoint, SSRFSecurityError


def _config(req: CloudStorageRequest) -> CloudStorageConfig:
    if req.endpoint_url:
        try:
            assert_safe_endpoint(req.endpoint_url, allow_loopback=False, label="Cloud storage endpoint")
        except SSRFSecurityError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err
    return CloudStorageConfig(
        provider=req.provider,
        bucket=req.bucket,
        region=req.region,
        endpoint_url=req.endpoint_url,
        access_key_id=req.access_key_id,
        secret_access_key=req.secret_access_key,
        project_id=req.project_id,
    )


@router.post("/cloud/test")
def test_cloud_storage(req: CloudStorageRequest, current_user: User = Depends(get_current_user)):
    provider = _service().provider_from_config(_config(req))
    ok, message = provider.test_connection()
    if not ok:
        raise HTTPException(status_code=400, detail=message)
    return {"status": "CONNECTED", "provider": provider.provider_type, "message": message, "cost_owner": "user"}


@router.post("/cloud/create")
def create_cloud_backup(req: BackupCreateRequest, current_user: User = Depends(get_current_user)):
    provider = _service().provider_from_config(_config(req))
    ok, message = provider.test_connection()
    if not ok:
        raise HTTPException(status_code=400, detail=message)
    try:
        return _service().create_backup(provider=provider, user_id=current_user.id, passphrase=req.passphrase, include_source_files=req.include_source_files)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"AA-OS backup creation failed: {exc}") from exc


@router.post("/cloud/migrate-local")
def migrate_existing_local_storage(req: CloudStorageRequest, current_user: User = Depends(get_current_user)):
    """Copy current local dataset/artifact files to the user's cloud provider without deleting local copies."""
    provider_obj = _service().provider_from_config(_config(req))
    ok, message = provider_obj.test_connection()
    if not ok:
        raise HTTPException(status_code=400, detail=message)
    try:
        from apps.api.src.services.storage_service import StorageService
        result = StorageService().migrate_local_to_provider(provider_obj, user_id=current_user.id)
        result["cost_owner"] = "user"
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Local-to-cloud migration failed: {exc}") from exc


@router.get("/cloud")
def list_cloud_backups(provider: str, bucket: str, region: str = "us-east-1", endpoint_url: Optional[str] = None, current_user: User = Depends(get_current_user)):
    if endpoint_url:
        try:
            assert_safe_endpoint(endpoint_url, allow_loopback=False, label="Cloud storage endpoint")
        except SSRFSecurityError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err
    provider_obj = _service().provider_from_config(CloudStorageConfig(provider=provider, bucket=bucket, region=region, endpoint_url=endpoint_url))
    try:
        return {"provider": provider_obj.provider_type, "backups": _service().list_backups(provider=provider_obj, user_id=current_user.id)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to list backups: {exc}") from exc


@router.post("/cloud/restore")
def restore_cloud_backup(req: BackupRestoreRequest, current_user: User = Depends(get_current_user)):
    provider_obj = _service().provider_from_config(_config(req))
    try:
        data = _service().download_backup(provider=provider_obj, object_key=req.object_key, user_id=current_user.id)
        is_super = getattr(current_user, "role", "") == "superadmin"
        return _service().restore_backup(
            backup_bytes=data,
            passphrase=req.passphrase,
            replace_existing=req.replace_existing,
            restoring_user_id=current_user.id,
            is_system_restore=is_super,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"AA-OS cloud restore failed: {exc}") from exc



def _is_loopback(request: Request) -> bool:
    host = request.client.host if request.client else ""
    return host in {"127.0.0.1", "::1", "localhost"}


def _assert_local_recovery_allowed(request: Request) -> None:
    if os.getenv("AAOS_LOCAL_RECOVERY_ONLY", "true").lower() not in {"1", "true", "yes"}:
        raise HTTPException(status_code=403, detail="Local recovery endpoint is disabled in hosted mode.")
    if not _is_loopback(request):
        raise HTTPException(status_code=403, detail="Recovery is available only from the local AA-OS device.")


@router.post("/recover-upload")
async def recover_uploaded_backup(request: Request, passphrase: str, replace_existing: bool = False, file: UploadFile = File(...)):
    """First-run local recovery after reinstall. No account token is required because the entire request is loopback-only and the encrypted backup requires its own passphrase."""
    _assert_local_recovery_allowed(request)
    data = await file.read()
    if len(data) > 10 * 1024 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Backup exceeds the local recovery size limit (10 GiB).")
    try:
        return _service().restore_backup(backup_bytes=data, passphrase=passphrase, replace_existing=replace_existing)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"AA-OS local recovery failed: {exc}") from exc
