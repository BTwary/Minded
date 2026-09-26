"""BYOI pluggable storage providers for AA-OS.

Local storage is the default. Cloud providers are explicit, user-owned options
used for datasets and optional continuity backups.
"""
from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import Optional, Tuple, List

from packages.analytics_core.src.providers.retention import LocalFileRetentionManager, RetentionResult


def _safe_relative(relative_path: str) -> str:
    clean = relative_path.replace("\\", "/").lstrip("/")
    parts = [p for p in clean.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        raise ValueError(f"Unsafe storage path: {relative_path}")
    return "/".join(parts)


def _safe_local_path(root: Path, relative_path: str) -> Path:
    target = (root / _safe_relative(relative_path)).resolve()
    if root.resolve() not in target.parents and target != root.resolve():
        raise ValueError(f"Storage path escapes provider root: {relative_path}")
    return target


class BaseStorageProvider(ABC):
    """Abstract interface for dataset, artifact, and backup storage."""

    @property
    @abstractmethod
    def provider_type(self) -> str:
        pass

    @property
    @abstractmethod
    def is_local(self) -> bool:
        pass

    @abstractmethod
    def save_file(self, relative_path: str, content: bytes) -> str:
        pass

    @abstractmethod
    def read_file(self, relative_path: str) -> bytes:
        pass

    @abstractmethod
    def file_exists(self, relative_path: str) -> bool:
        pass

    @abstractmethod
    def delete_file(self, relative_path: str) -> bool:
        pass

    @abstractmethod
    def get_local_path(self, relative_path: str) -> str:
        pass

    @abstractmethod
    def test_connection(self) -> Tuple[bool, str]:
        pass

    def list_files(self, prefix: str = "") -> List[str]:
        """List logical object paths when the provider supports listing."""
        raise NotImplementedError(f"{self.provider_type} does not support listing")


class LocalStorageProvider(BaseStorageProvider):
    """Zero-setup local filesystem storage provider for Free / Community Edition."""

    def __init__(self, base_dir: str = "./data_store", retention_days: Optional[int] = None):
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        configured_days = retention_days if retention_days is not None else int(os.getenv("DATA_RETENTION_DAYS", "180"))
        self.retention = LocalFileRetentionManager(str(self.base_dir), configured_days)
        self.last_retention_result: Optional[RetentionResult] = self.retention.prune()

    @property
    def provider_type(self) -> str:
        return "local"

    @property
    def is_local(self) -> bool:
        return True

    def _resolve(self, relative_path: str) -> Path:
        return _safe_local_path(self.base_dir, relative_path)

    def save_file(self, relative_path: str, content: bytes) -> str:
        full_path = self._resolve(relative_path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(content)
        self.last_retention_result = self.retention.prune()
        return str(full_path)

    def read_file(self, relative_path: str) -> bytes:
        full_path = self._resolve(relative_path)
        if not full_path.exists():
            raise FileNotFoundError(f"Storage file not found: {relative_path}")
        return full_path.read_bytes()

    def file_exists(self, relative_path: str) -> bool:
        return self._resolve(relative_path).exists()

    def delete_file(self, relative_path: str) -> bool:
        full_path = self._resolve(relative_path)
        if full_path.exists():
            full_path.unlink()
            return True
        return False

    def get_local_path(self, relative_path: str) -> str:
        return str(self._resolve(relative_path))

    def list_files(self, prefix: str = "") -> List[str]:
        clean_prefix = _safe_relative(prefix)
        return sorted(
            p.relative_to(self.base_dir).as_posix()
            for p in self.base_dir.rglob("*")
            if p.is_file() and (not clean_prefix or p.relative_to(self.base_dir).as_posix().startswith(clean_prefix))
        )

    def test_connection(self) -> Tuple[bool, str]:
        probe = self._resolve(".health_check_probe.tmp")
        try:
            probe.write_text("probe", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return True, f"Local filesystem storage operational ({self.base_dir})"
        except Exception as e:
            return False, f"Local filesystem error: {e}"


class S3StorageProvider(BaseStorageProvider):
    """Actual user-owned S3/S3-compatible provider (AWS S3, R2, MinIO, etc.)."""

    def __init__(self, bucket_name: str, region_name: str = "us-east-1", access_key_id: Optional[str] = None, secret_access_key: Optional[str] = None, endpoint_url: Optional[str] = None, cache_dir: str = "./data_store/s3_cache"):
        self.bucket_name = bucket_name
        self.region_name = region_name
        self.access_key_id = access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        self.secret_access_key = secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        self.endpoint_url = endpoint_url or os.getenv("AWS_S3_ENDPOINT_URL")
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = None

    @property
    def provider_type(self) -> str:
        return "s3"

    @property
    def is_local(self) -> bool:
        return False

    def _get_client(self):
        if self._client is None:
            if self.endpoint_url:
                # Defense-in-depth: callers (apps/api/src/api/v1/backups.py) already
                # validate endpoint_url before building this provider, but re-check
                # here immediately before the client is created so a provider built
                # once and reused later (or constructed by a future caller that
                # skips that pre-check) can't skip validation. Checked before the
                # optional-dependency import so the rejection reason is always the
                # security failure, never masked by an unrelated ImportError. Note
                # this cannot pin the resolved IP the way the AI-provider HTTP
                # calls do: boto3/botocore resolves and connects to endpoint_url
                # itself on every request, outside our control, so a DNS-rebinding
                # window remains between this check and boto3's own connection.
                # Fully closing it would require a custom botocore endpoint
                # resolver/connection pool; flagged as accepted residual risk
                # rather than solved here.
                from packages.analytics_core.src.security.ssrf import assert_safe_endpoint
                assert_safe_endpoint(self.endpoint_url, allow_loopback=False, label="S3-compatible endpoint URL")
            try:
                import boto3
            except ImportError as exc:
                raise RuntimeError("S3 cloud support is optional; install requirements-cloud.txt (boto3).") from exc
            self._client = boto3.client("s3", region_name=self.region_name, aws_access_key_id=self.access_key_id, aws_secret_access_key=self.secret_access_key, endpoint_url=self.endpoint_url)
        return self._client

    def _cache_path(self, relative_path: str) -> Path:
        return _safe_local_path(self.cache_dir, relative_path)

    def save_file(self, relative_path: str, content: bytes) -> str:
        key = _safe_relative(relative_path)
        self._get_client().put_object(Bucket=self.bucket_name, Key=key, Body=content)
        cache = self._cache_path(key)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(content)
        return f"s3://{self.bucket_name}/{key}"

    def read_file(self, relative_path: str) -> bytes:
        key = _safe_relative(relative_path)
        try:
            response = self._get_client().get_object(Bucket=self.bucket_name, Key=key)
            data = response["Body"].read()
            cache = self._cache_path(key)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(data)
            return data
        except Exception:
            cache = self._cache_path(key)
            if cache.exists():
                return cache.read_bytes()
            raise

    def file_exists(self, relative_path: str) -> bool:
        try:
            self._get_client().head_object(Bucket=self.bucket_name, Key=_safe_relative(relative_path))
            return True
        except Exception:
            return False

    def delete_file(self, relative_path: str) -> bool:
        key = _safe_relative(relative_path)
        self._get_client().delete_object(Bucket=self.bucket_name, Key=key)
        self._cache_path(key).unlink(missing_ok=True)
        return True

    def get_local_path(self, relative_path: str) -> str:
        key = _safe_relative(relative_path)
        cache = self._cache_path(key)
        if not cache.exists():
            self.read_file(key)
        return str(cache)

    def list_files(self, prefix: str = "") -> List[str]:
        token = None
        keys: List[str] = []
        kwargs = {"Bucket": self.bucket_name, "Prefix": _safe_relative(prefix) if prefix else ""}
        while True:
            if token:
                kwargs["ContinuationToken"] = token
            response = self._get_client().list_objects_v2(**kwargs)
            keys.extend(o["Key"] for o in response.get("Contents", []))
            if not response.get("IsTruncated"):
                break
            token = response.get("NextContinuationToken")
        return sorted(keys)

    def test_connection(self) -> Tuple[bool, str]:
        if not self.bucket_name:
            return False, "S3 bucket name is required."
        try:
            self._get_client().head_bucket(Bucket=self.bucket_name)
            endpoint_note = f" via {self.endpoint_url}" if self.endpoint_url else ""
            return True, f"S3 bucket '{self.bucket_name}' reachable{endpoint_note}."
        except Exception as exc:
            return False, f"S3 connection failed: {exc}"


class GCSStorageProvider(BaseStorageProvider):
    """Actual user-owned Google Cloud Storage provider."""

    def __init__(self, bucket_name: str, project_id: Optional[str] = None, cache_dir: str = "./data_store/gcs_cache"):
        self.bucket_name = bucket_name
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID")
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._bucket = None

    @property
    def provider_type(self) -> str:
        return "gcs"

    @property
    def is_local(self) -> bool:
        return False

    def _get_bucket(self):
        if self._bucket is None:
            try:
                from google.cloud import storage as gcs
            except ImportError as exc:
                raise RuntimeError("GCS cloud support is optional; install requirements-cloud.txt (google-cloud-storage).") from exc
            self._client = gcs.Client(project=self.project_id)
            self._bucket = self._client.bucket(self.bucket_name)
        return self._bucket

    def _cache_path(self, relative_path: str) -> Path:
        return _safe_local_path(self.cache_dir, relative_path)

    def save_file(self, relative_path: str, content: bytes) -> str:
        key = _safe_relative(relative_path)
        self._get_bucket().blob(key).upload_from_string(content)
        cache = self._cache_path(key)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(content)
        return f"gs://{self.bucket_name}/{key}"

    def read_file(self, relative_path: str) -> bytes:
        key = _safe_relative(relative_path)
        data = self._get_bucket().blob(key).download_as_bytes()
        cache = self._cache_path(key)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(data)
        return data

    def file_exists(self, relative_path: str) -> bool:
        return self._get_bucket().blob(_safe_relative(relative_path)).exists()

    def delete_file(self, relative_path: str) -> bool:
        key = _safe_relative(relative_path)
        self._get_bucket().blob(key).delete()
        self._cache_path(key).unlink(missing_ok=True)
        return True

    def get_local_path(self, relative_path: str) -> str:
        key = _safe_relative(relative_path)
        cache = self._cache_path(key)
        if not cache.exists():
            self.read_file(key)
        return str(cache)

    def list_files(self, prefix: str = "") -> List[str]:
        return sorted(blob.name for blob in self._get_bucket().list_blobs(prefix=_safe_relative(prefix) if prefix else None))

    def test_connection(self) -> Tuple[bool, str]:
        if not self.bucket_name:
            return False, "GCS bucket name is required."
        try:
            exists = self._get_bucket().exists()
            return (True, f"Google Cloud Storage bucket '{self.bucket_name}' reachable.") if exists else (False, f"GCS bucket '{self.bucket_name}' does not exist or is inaccessible.")
        except Exception as exc:
            return False, f"GCS connection failed: {exc}"
