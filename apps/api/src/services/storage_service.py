"""Provider-backed dataset storage with local-first compatibility."""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from apps.api.src.core.config import settings
from packages.analytics_core.src.ingestion.robust_loader import IngestionReport, RobustFileLoader
from packages.analytics_core.src.providers.manager import InfrastructureManager
from packages.analytics_core.src.providers.retention import LocalFileRetentionManager, RetentionResult
from packages.analytics_core.src.providers.storage import BaseStorageProvider, LocalStorageProvider


class StorageService:
    """Manages dataset versioning through the active AA-OS storage provider.

    Existing absolute local paths remain readable. New versions use logical
    provider keys so local, S3-compatible, and GCS storage can be selected
    without changing the analytical layer.
    """

    def __init__(self, base_dir: Optional[str] = None, provider: Optional[BaseStorageProvider] = None):
        self.base_dir = base_dir or settings.DATA_STORAGE_DIR
        self.provider = provider or InfrastructureManager.get_storage_provider()
        os.makedirs(self.base_dir, exist_ok=True)
        self.retention = LocalFileRetentionManager(self.base_dir, settings.DATA_RETENTION_DAYS)
        self.last_retention_result: Optional[RetentionResult] = self.retention.prune() if self.provider.is_local else None

    def enforce_retention(self) -> RetentionResult:
        """Prune local source files. Cloud objects are controlled by the user's provider policy."""
        self.last_retention_result = self.retention.prune()
        return self.last_retention_result

    def _logical_raw_key(self, dataset_id: str, version: int, filename: str) -> str:
        safe_name = Path(filename or "uploaded_data.csv").name
        return f"datasets/{dataset_id}/v{version}/{safe_name}"

    def _logical_parquet_key(self, dataset_id: str, version: int) -> str:
        return f"datasets/{dataset_id}/v{version}/data.parquet"

    def save_raw_file(self, dataset_id: str, version: int, filename: str, content: bytes) -> str:
        logical_key = self._logical_raw_key(dataset_id, version, filename)
        saved = self.provider.save_file(logical_key, content)
        if self.provider.is_local:
            self.last_retention_result = self.retention.prune()
        return logical_key if not self.provider.is_local else saved

    def save_dataframe_as_parquet(self, dataset_id: str, version: int, df: pd.DataFrame) -> str:
        logical_key = self._logical_parquet_key(dataset_id, version)
        buf = io.BytesIO()
        df.to_parquet(buf, engine="pyarrow", compression="snappy", index=False)
        payload = buf.getvalue()
        saved = self.provider.save_file(logical_key, payload)
        if self.provider.is_local:
            self.last_retention_result = self.retention.prune()
        return logical_key if not self.provider.is_local else saved

    def _resolve_read_path(self, file_path: str) -> str:
        # Backward compatibility with older database rows containing absolute
        # local filesystem paths.
        if os.path.isabs(file_path) and os.path.exists(file_path):
            return file_path
        # Provider keys are logical paths. For cloud storage this downloads the
        # object into its isolated cache; for local storage it resolves safely.
        return self.provider.get_local_path(file_path)

    def migrate_local_to_provider(
        self,
        target_provider: BaseStorageProvider,
        user_id: Optional[str] = None,
        db_session: Optional[Any] = None,
    ) -> dict:
        """Copy existing local dataset/artifact files to a user-selected provider.

        No local data is deleted. DatasetVersion paths under the managed local
        directory are rewritten to provider logical keys after successful upload.
        When user_id is provided, only datasets belonging to projects owned by
        user_id are migrated, strictly enforcing tenant boundaries.
        """
        if target_provider.is_local:
            raise ValueError("Target provider must be a cloud/non-local provider.")

        from apps.api.src.core.database import SessionLocal
        from apps.api.src.models.entities import Dataset, DatasetVersion, Project

        root = Path(self.base_dir).expanduser().resolve()
        uploaded = 0
        skipped = 0
        errors = []
        db_updated = 0

        db = db_session if db_session is not None else SessionLocal()
        should_close_db = db_session is None

        try:
            # Determine allowed dataset versions
            query = db.query(DatasetVersion)
            if user_id:
                query = (
                    query.join(Dataset, DatasetVersion.dataset_id == Dataset.id)
                    .join(Project, Dataset.project_id == Project.id)
                    .filter(Project.owner_id == user_id)
                )

            target_versions = query.all()
            target_file_paths = set()

            for dv in target_versions:
                raw_fp = str(dv.file_path or "")
                if not raw_fp:
                    continue
                p = Path(raw_fp).expanduser()
                if p.is_absolute() and p.exists():
                    target_file_paths.add(p.resolve())
                else:
                    cand = (root / raw_fp.replace("datasets/", "")).resolve()
                    if cand.exists():
                        target_file_paths.add(cand)

            # In un-scoped/CLI mode, fallback to all files under root if no user_id specified
            files_to_upload = target_file_paths if user_id else {
                path.resolve() for path in root.rglob("*")
                if path.is_file() and not path.is_symlink()
            }

            for path in files_to_upload:
                if not path.is_file() or path.is_symlink():
                    continue
                try:
                    rel = path.relative_to(root).as_posix()
                except ValueError:
                    continue
                if any(part in {"s3_cache", "gcs_cache", ".aaos_cache", "__pycache__"} for part in path.relative_to(root).parts):
                    skipped += 1
                    continue
                try:
                    payload = path.read_bytes()
                    key = rel if rel.startswith("datasets/") else f"datasets/{rel}"
                    target_provider.save_file(key, payload)
                    uploaded += 1
                except Exception as exc:
                    errors.append(f"{rel}: {exc}")

            # Update DB references for migrated versions
            versions_to_update = target_versions if user_id else db.query(DatasetVersion).all()
            for record in versions_to_update:
                old = str(record.file_path or "")
                if not old:
                    continue
                old_path = Path(old).expanduser()
                if not old_path.is_absolute() or not old_path.exists():
                    continue
                try:
                    relative = old_path.resolve().relative_to(root).as_posix()
                except ValueError:
                    continue
                new_key = relative if relative.startswith("datasets/") else f"datasets/{relative}"
                if target_provider.file_exists(new_key):
                    record.file_path = new_key
                    db_updated += 1
            db.commit()
        except Exception as exc:
            errors.append(f"database-path-rewrite: {exc}")
        finally:
            if should_close_db:
                db.close()

        return {
            "provider": target_provider.provider_type,
            "uploaded_files": uploaded,
            "skipped_files": skipped,
            "database_paths_updated": db_updated,
            "errors": errors,
            "local_copy_retained": True,
        }


    def load_dataframe(self, file_path: str) -> pd.DataFrame:
        """Load CSV/XLSX/JSON/Parquet from the active provider."""
        resolved = self._resolve_read_path(file_path)
        ext = os.path.splitext(resolved)[1].lower()
        if ext == ".parquet":
            return pd.read_parquet(resolved)
        df, _report = RobustFileLoader().load(file_path=resolved)
        return df

    def load_dataframe_with_report(self, file_path: str) -> Tuple[pd.DataFrame, Optional[IngestionReport]]:
        resolved = self._resolve_read_path(file_path)
        ext = os.path.splitext(resolved)[1].lower()
        if ext == ".parquet":
            return pd.read_parquet(resolved), None
        return RobustFileLoader().load(file_path=resolved)
