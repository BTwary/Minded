"""User-controlled AA-OS backup, cloud continuity, and migration service."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from packages.analytics_core.src.backup.portable_backup import PortableBackupBuilder, PortableBackupRestorer, BackupError
from packages.analytics_core.src.providers.storage import BaseStorageProvider, LocalStorageProvider, S3StorageProvider, GCSStorageProvider


@dataclass
class CloudStorageConfig:
    provider: str
    bucket: str
    region: str = "us-east-1"
    endpoint_url: Optional[str] = None
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    project_id: Optional[str] = None


class BackupService:
    """Build encrypted portable backups and store them through a BYOI provider."""

    PREFIX = "aaos-backups"

    def __init__(self, *, engine: Any, metadata: Any, storage_root: str, app_version: str = "dev", schema_revision: str = "current"):
        self.engine = engine
        self.metadata = metadata
        self.storage_root = storage_root
        self.builder = PortableBackupBuilder(app_version=app_version, schema_revision=schema_revision)
        self.restorer = PortableBackupRestorer()

    @staticmethod
    def provider_from_config(config: CloudStorageConfig) -> BaseStorageProvider:
        provider = config.provider.lower().strip()
        if provider == "s3":
            return S3StorageProvider(
                bucket_name=config.bucket,
                region_name=config.region,
                access_key_id=config.access_key_id,
                secret_access_key=config.secret_access_key,
                endpoint_url=config.endpoint_url,
            )
        if provider == "gcs":
            return GCSStorageProvider(bucket_name=config.bucket, project_id=config.project_id)
        raise BackupError("Unsupported cloud storage provider. Use 's3' (AWS/R2/MinIO) or 'gcs'.")

    def create_backup(self, *, provider: BaseStorageProvider, user_id: str, passphrase: str, include_source_files: bool = True) -> Dict[str, Any]:
        from packages.analytics_core.src.backup.portable_backup import BackupScope
        data = self.builder.build_bytes(
            engine=self.engine,
            metadata=self.metadata,
            storage_root=self.storage_root,
            passphrase=passphrase,
            include_source_files=include_source_files,
            scope=BackupScope.USER,
            scope_user_id=user_id,
        )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_id = f"{stamp}-{uuid4().hex[:12]}"
        key = f"{self.PREFIX}/{user_id}/{backup_id}.aaosbackup"
        uri = provider.save_file(key, data)
        return {
            "backup_id": backup_id,
            "object_key": key,
            "uri": uri,
            "provider": provider.provider_type,
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "encrypted": True,
            "includes_source_files": include_source_files,
            "cost_owner": "user",
            "note": "AA-OS does not receive or pay cloud storage/query charges; the connected provider account owns the data and charges.",
        }

    def list_backups(self, *, provider: BaseStorageProvider, user_id: str) -> List[Dict[str, Any]]:
        prefix = f"{self.PREFIX}/{user_id}/"
        keys = provider.list_files(prefix)
        return [
            {
                "backup_id": Path(key).name.removesuffix(".aaosbackup"),
                "object_key": key,
                "provider": provider.provider_type,
            }
            for key in keys
            if key.endswith(".aaosbackup")
        ]

    def download_backup(self, *, provider: BaseStorageProvider, object_key: str, user_id: Optional[str] = None) -> bytes:
        clean = object_key.replace("\\", "/").lstrip("/")
        if ".." in clean.split("/"):
            raise BackupError("Invalid backup object key.")
        if user_id:
            expected_user_prefix = f"{self.PREFIX}/{user_id}/"
            if not clean.startswith(expected_user_prefix):
                raise BackupError("Access denied: cannot access another user's backup.")
        elif not clean.startswith(f"{self.PREFIX}/"):
            raise BackupError("Invalid backup object key.")
        return provider.read_file(clean)

    def restore_backup(
        self,
        *,
        backup_bytes: bytes,
        passphrase: str,
        replace_existing: bool = False,
        restoring_user_id: Optional[str] = None,
        is_system_restore: bool = False,
    ) -> Dict[str, Any]:
        return self.restorer.restore_bytes(
            backup_bytes=backup_bytes,
            passphrase=passphrase,
            engine=self.engine,
            metadata=self.metadata,
            storage_root=self.storage_root,
            replace_existing=replace_existing,
            restoring_user_id=restoring_user_id,
            is_system_restore=is_system_restore,
        )

    def local_test_provider(self, path: str) -> Dict[str, Any]:
        provider = LocalStorageProvider(path, retention_days=3650)
        ok, message = provider.test_connection()
        return {"ok": ok, "message": message, "provider": provider.provider_type}
