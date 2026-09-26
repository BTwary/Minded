"""Portable, encrypted AA-OS backup format.

The backup is deliberately storage-provider agnostic: a single `.aaosbackup`
artifact can be kept locally or uploaded to any provider supported by AA-OS.
Cloud credentials, AI API keys, and environment files are never included.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import zipfile
from dataclasses import dataclass, asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


BACKUP_FORMAT = "AAOS-BACKUP"
BACKUP_VERSION = "1.0"
DEFAULT_KDF_ITERATIONS = 390_000

MAX_BACKUP_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB
MAX_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024  # 1 GB
MAX_ZIP_ENTRIES = 10_000


class BackupError(RuntimeError):
    """Raised for invalid, incompatible, or corrupted backup artifacts."""


class BackupScope:
    """Defines what data is included in a backup artifact.

    USER scope (default): only rows whose user_id column matches scope_user_id,
    and only storage files under storage_root/<scope_user_id>/.

    SYSTEM scope: all rows, all files. Requires explicit opt-in by the caller
    and must only be used for system-administrator operations.
    """
    USER = "USER"
    SYSTEM = "SYSTEM"


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bytes):
        return {"__bytes_b64__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, default=_json_default, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _derive_key(passphrase: str, salt: bytes, iterations: int = DEFAULT_KDF_ITERATIONS) -> bytes:
    if not passphrase:
        raise BackupError("A backup passphrase is required for encrypted backups.")
    try:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    except ImportError as exc:
        raise BackupError("cryptography is required for encrypted AA-OS backups.") from exc

    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=iterations)
    return kdf.derive(passphrase.encode("utf-8"))


def _encrypt(data: bytes, passphrase: str, salt: bytes, iterations: int) -> bytes:
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:
        raise BackupError("cryptography is required for encrypted AA-OS backups.") from exc
    key = _derive_key(passphrase, salt, iterations)
    nonce = os.urandom(12)
    ciphertext = AESGCM(key).encrypt(nonce, data, BACKUP_FORMAT.encode("ascii"))
    return nonce + ciphertext


def _decrypt(data: bytes, passphrase: str, salt: bytes, iterations: int) -> bytes:
    if len(data) < 12:
        raise BackupError("Encrypted payload is truncated.")
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:
        raise BackupError("cryptography is required for encrypted AA-OS backups.") from exc
    key = _derive_key(passphrase, salt, iterations)
    nonce, ciphertext = data[:12], data[12:]
    try:
        return AESGCM(key).decrypt(nonce, ciphertext, BACKUP_FORMAT.encode("ascii"))
    except Exception as exc:
        raise BackupError("Backup passphrase is incorrect or the backup payload is corrupted.") from exc


@dataclass(frozen=True)
class AAOSBackupManifest:
    format: str
    format_version: str
    created_at: str
    app_version: str
    schema_revision: str
    encrypted: bool
    kdf: str
    kdf_iterations: int
    salt_b64: Optional[str]
    database_tables: Sequence[str]
    storage_files: Sequence[str]
    excluded_sensitive_config: Sequence[str]
    checksums: Mapping[str, str]
    backup_scope: str
    scope_user_id: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)



class PortableBackupBuilder:
    """Build a logical AA-OS backup from SQLAlchemy metadata plus local files."""

    EXCLUDED_FILENAMES = {".env", ".env.local", ".env.production", "secrets.json", "credentials.json"}
    EXCLUDED_DIRS = {"s3_cache", "gcs_cache", ".aaos_cache", "__pycache__"}
    EXCLUDED_CONFIG_FIELDS = {
        "AI_API_KEY",
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GROQ_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "CLOUD_BACKUP_SECRET",
    }

    def __init__(self, *, app_version: str = "dev", schema_revision: str = "unknown"):
        self.app_version = app_version
        self.schema_revision = schema_revision

    @staticmethod
    def _serialize_rows(engine: Any, tables: Iterable[Any]) -> tuple[Dict[str, Any], list[str]]:
        rows_payload: Dict[str, Any] = {}
        names: list[str] = []
        with engine.connect() as conn:
            for table in tables:
                names.append(table.name)
                result = conn.execute(table.select())
                rows_payload[table.name] = [dict(row._mapping) for row in result.fetchall()]
        return rows_payload, names

    @staticmethod
    def _filter_rows_by_scope(
        rows_payload: Dict[str, Any],
        scope: str,
        scope_user_id: Optional[str],
    ) -> Dict[str, Any]:
        """Return a filtered copy of rows_payload containing only rows authorized for the scope.

        For BackupScope.USER:
        Follows the strict ownership graph:
        user -> authorized projects (owner_id == scope_user_id)
             -> datasets (project_id) -> dataset_versions & dataset_transformations
             -> semantic_models (project_id) -> dimensions, entities, metrics, relationships, etc.
             -> conversations (project_id) -> messages
             -> reports (project_id), dashboards, alerts, analysis_runs, glossary, metrics
             -> investigations (owner_id/user_id == scope_user_id OR project_id in user_projects)
                -> contracts, events, executions, edges, jobs, objectives, unknowns, verdicts,
                   data_readiness, decision_requests, resource_waits, step_executions
                -> hypotheses -> counter_hypotheses
                -> predictions
                -> experiments -> observations
                -> evidence -> evidence_verifications
                -> belief_updates
             -> user_ai_configs, organization_members, audit_logs (user_id == scope_user_id)
             -> users (id == scope_user_id)

        Tables without a user_id or project link are excluded unless explicitly mapped to the user.
        System tables ('alembic_', 'pg_', 'information_schema') are excluded.

        For BackupScope.SYSTEM: rows_payload is returned unmodified.
        """
        if scope == BackupScope.SYSTEM or not scope_user_id:
            return rows_payload

        SYSTEM_TABLE_PREFIXES = ('alembic_', 'pg_', 'information_schema', 'sqlite_')
        uid = str(scope_user_id)

        filtered: Dict[str, Any] = {}
        for table_name in rows_payload:
            if any(table_name.startswith(pfx) for pfx in SYSTEM_TABLE_PREFIXES):
                continue
            filtered[table_name] = []

        # 1. User row & organization
        user_org_id = None
        user_rows = rows_payload.get("users", [])
        filtered["users"] = [r for r in user_rows if str(r.get("id", "")) == uid]
        if filtered["users"]:
            user_org_id = filtered["users"][0].get("organization_id")
        if user_org_id and "organizations" in rows_payload:
            filtered["organizations"] = [r for r in rows_payload["organizations"] if str(r.get("id", "")) == str(user_org_id)]
        elif "organizations" in rows_payload:
            filtered["organizations"] = []

        # 2. User-specific config & membership
        if "organization_members" in rows_payload:
            filtered["organization_members"] = [r for r in rows_payload["organization_members"] if str(r.get("user_id", "")) == uid]
        if "user_ai_configs" in rows_payload:
            filtered["user_ai_configs"] = [r for r in rows_payload["user_ai_configs"] if str(r.get("user_id", "")) == uid]
        if "api_keys" in rows_payload:
            filtered["api_keys"] = [r for r in rows_payload["api_keys"] if str(r.get("user_id", "")) == uid]
        if "audit_logs" in rows_payload:
            filtered["audit_logs"] = [r for r in rows_payload["audit_logs"] if str(r.get("user_id", "")) == uid]

        # 3. Projects owned by user
        project_rows = rows_payload.get("projects", [])
        filtered["projects"] = [r for r in project_rows if str(r.get("owner_id", "")) == uid]
        auth_project_ids = {str(r["id"]) for r in filtered["projects"] if r.get("id")}

        # 4. Project-scoped domain tables
        def _filter_by_project(table_name: str, id_col: str = "id") -> set[str]:
            rows = rows_payload.get(table_name, [])
            res = [r for r in rows if str(r.get("project_id", "")) in auth_project_ids or (r.get("user_id") and str(r.get("user_id")) == uid)]
            filtered[table_name] = res
            return {str(r[id_col]) for r in res if r.get(id_col)}

        auth_dataset_ids = _filter_by_project("datasets")
        auth_semantic_model_ids = _filter_by_project("semantic_models")
        auth_conversation_ids = _filter_by_project("conversations")
        _filter_by_project("reports")
        _filter_by_project("dashboards")
        _filter_by_project("analysis_runs")
        _filter_by_project("business_metrics")
        _filter_by_project("business_glossary")
        auth_alert_rule_ids = _filter_by_project("alert_rules")

        # Alert events
        alert_event_rows = rows_payload.get("alert_events", [])
        filtered["alert_events"] = [
            r for r in alert_event_rows
            if str(r.get("project_id", "")) in auth_project_ids or str(r.get("rule_id", "")) in auth_alert_rule_ids
        ]

        # Messages
        message_rows = rows_payload.get("messages", [])
        filtered["messages"] = [
            r for r in message_rows
            if str(r.get("conversation_id", "")) in auth_conversation_ids
        ]

        # Datasets children
        dataset_version_rows = rows_payload.get("dataset_versions", [])
        filtered["dataset_versions"] = [
            r for r in dataset_version_rows
            if str(r.get("dataset_id", "")) in auth_dataset_ids
        ]

        dataset_transformation_rows = rows_payload.get("dataset_transformations", [])
        filtered["dataset_transformations"] = [
            r for r in dataset_transformation_rows
            if str(r.get("dataset_id", "")) in auth_dataset_ids
        ]

        # Semantic models children
        for child_tbl in (
            "semantic_dimensions", "semantic_entities", "semantic_metrics",
            "semantic_relationships", "semantic_time_definitions", "semantic_business_rules"
        ):
            rows = rows_payload.get(child_tbl, [])
            filtered[child_tbl] = [
                r for r in rows
                if str(r.get("semantic_model_id", "")) in auth_semantic_model_ids
            ]

        # 5. Investigations and children
        inv_rows = rows_payload.get("investigations", [])
        filtered["investigations"] = [
            r for r in inv_rows
            if str(r.get("user_id", "")) == uid or str(r.get("project_id", "")) in auth_project_ids
        ]
        auth_inv_ids = {str(r["id"]) for r in filtered["investigations"] if r.get("id")}

        for inv_child in (
            "investigation_contracts", "investigation_events", "investigation_executions",
            "investigation_graph_edges", "investigation_jobs", "investigation_objectives",
            "investigation_unknowns", "investigation_verdicts", "investigation_data_readiness",
            "investigation_decision_requests", "investigation_resource_waits",
            "investigation_step_executions", "predictions", "belief_updates"
        ):
            rows = rows_payload.get(inv_child, [])
            filtered[inv_child] = [
                r for r in rows
                if str(r.get("investigation_id", "")) in auth_inv_ids
            ]

        hyp_rows = rows_payload.get("hypotheses", [])
        filtered["hypotheses"] = [
            r for r in hyp_rows
            if str(r.get("investigation_id", "")) in auth_inv_ids
        ]
        auth_hyp_ids = {str(r["id"]) for r in filtered["hypotheses"] if r.get("id")}

        counter_hyp_rows = rows_payload.get("counter_hypotheses", [])
        filtered["counter_hypotheses"] = [
            r for r in counter_hyp_rows
            if str(r.get("primary_hypothesis_id", "")) in auth_hyp_ids or str(r.get("counter_hypothesis_id", "")) in auth_hyp_ids
        ]

        exp_rows = rows_payload.get("experiments", [])
        filtered["experiments"] = [
            r for r in exp_rows
            if str(r.get("investigation_id", "")) in auth_inv_ids
        ]
        auth_exp_ids = {str(r["id"]) for r in filtered["experiments"] if r.get("id")}

        obs_rows = rows_payload.get("observations", [])
        filtered["observations"] = [
            r for r in obs_rows
            if str(r.get("experiment_id", "")) in auth_exp_ids
        ]

        evidence_rows = rows_payload.get("evidence", [])
        filtered["evidence"] = [
            r for r in evidence_rows
            if str(r.get("investigation_id", "")) in auth_inv_ids
        ]
        auth_evidence_ids = {str(r["id"]) for r in filtered["evidence"] if r.get("id")}

        verif_rows = rows_payload.get("evidence_verifications", [])
        filtered["evidence_verifications"] = [
            r for r in verif_rows
            if str(r.get("evidence_id", "")) in auth_evidence_ids
        ]

        # 6. Any other table with user_id or owner_id fallback
        for tbl, rows in rows_payload.items():
            if tbl in filtered:
                continue
            if any(tbl.startswith(pfx) for pfx in SYSTEM_TABLE_PREFIXES):
                continue
            if not rows:
                filtered[tbl] = []
                continue
            sample = rows[0]
            if "user_id" in sample:
                filtered[tbl] = [r for r in rows if str(r.get("user_id", "")) == uid]
            elif "owner_id" in sample:
                filtered[tbl] = [r for r in rows if str(r.get("owner_id", "")) == uid]
            else:
                filtered[tbl] = []

        return filtered

    @staticmethod
    def _collect_storage_files(
        storage_root: Optional[str],
        scope_prefix: Optional[str] = None,
        scope: str = BackupScope.SYSTEM,
        scope_user_id: Optional[str] = None,
        authorized_dataset_ids: Optional[Iterable[str]] = None,
        dataset_file_paths: Optional[Iterable[str]] = None,
    ) -> Dict[str, bytes]:
        if not storage_root:
            return {}
        root = Path(storage_root).expanduser().resolve()
        if not root.exists():
            return {}

        uid = scope_user_id or scope_prefix
        auth_datasets = set(authorized_dataset_ids or [])
        auth_paths = set(dataset_file_paths or [])

        def _is_safe(p: Path) -> bool:
            if not p.is_file() or p.is_symlink():
                return False
            if p.name in PortableBackupBuilder.EXCLUDED_FILENAMES:
                return False
            rel_parts = p.relative_to(root).parts
            if any(part in PortableBackupBuilder.EXCLUDED_DIRS for part in rel_parts):
                return False
            if p.relative_to(root).as_posix().startswith('.git/'):
                return False
            return True

        output: Dict[str, bytes] = {}

        if scope == BackupScope.SYSTEM or (not uid and not auth_datasets):
            # SYSTEM scope: collect all files in storage_root
            for path in root.rglob('*'):
                if _is_safe(path):
                    rel = path.relative_to(root).as_posix()
                    output[rel] = path.read_bytes()
            return output

        # USER scope:
        # 1. Collect files in storage_root/datasets/{dataset_id}/... for all user's datasets
        datasets_dir = root / "datasets"
        if datasets_dir.exists():
            for ds_id in auth_datasets:
                ds_dir = datasets_dir / str(ds_id)
                if ds_dir.exists():
                    for path in ds_dir.rglob('*'):
                        if _is_safe(path):
                            rel = path.relative_to(root).as_posix()
                            output[rel] = path.read_bytes()

        # 2. Collect any explicitly referenced dataset_versions file_paths
        for fp in auth_paths:
            p = Path(fp)
            if not p.is_absolute():
                p = (root / p).resolve()
            if p.exists() and _is_safe(p):
                try:
                    rel = p.relative_to(root).as_posix()
                    output[rel] = p.read_bytes()
                except ValueError:
                    safe_name = f"datasets/external/{p.name}"
                    output[safe_name] = p.read_bytes()

        # 3. Collect files in storage_root/<user_id>/... if it exists
        if uid:
            user_dir = root / str(uid)
            if user_dir.exists():
                for path in user_dir.rglob('*'):
                    if _is_safe(path):
                        rel = path.relative_to(root).as_posix()
                        output[rel] = path.read_bytes()

        return output


    def build_bytes(
        self,
        *,
        engine: Any,
        metadata: Any,
        storage_root: Optional[str],
        passphrase: str,
        include_source_files: bool = True,
        scope: str = BackupScope.USER,
        scope_user_id: Optional[str] = None,
    ) -> bytes:
        """Build an encrypted portable backup.

        scope=USER (default): only rows and files belonging to scope_user_id
        are included. scope=SYSTEM: all rows and all files (for admins only).
        """
        if scope == BackupScope.USER and not scope_user_id:
            raise BackupError(
                "scope_user_id is required for USER-scoped backups. "
                "Pass scope=BackupScope.SYSTEM only for administrator system backups."
            )
        tables = list(metadata.sorted_tables)
        rows_payload, table_names = self._serialize_rows(engine, tables)
        # Filter rows to authorized scope before encrypting
        rows_payload = self._filter_rows_by_scope(rows_payload, scope, scope_user_id)
        auth_dataset_ids = {str(r["id"]) for r in rows_payload.get("datasets", []) if r.get("id")}
        auth_file_paths = {str(r["file_path"]) for r in rows_payload.get("dataset_versions", []) if r.get("file_path")}

        storage_files = (
            self._collect_storage_files(
                storage_root,
                scope_prefix=scope_user_id,
                scope=scope,
                scope_user_id=scope_user_id,
                authorized_dataset_ids=auth_dataset_ids,
                dataset_file_paths=auth_file_paths,
            )
            if include_source_files else {}
        )

        logical_db = _canonical_json({"tables": rows_payload})
        salt = os.urandom(16)
        encrypted_db = _encrypt(logical_db, passphrase, salt, DEFAULT_KDF_ITERATIONS)
        encrypted_storage = {
            name: _encrypt(content, passphrase, salt, DEFAULT_KDF_ITERATIONS)
            for name, content in storage_files.items()
        }

        manifest_without_hash = {
            "format": BACKUP_FORMAT,
            "format_version": BACKUP_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat() + "Z",
            "app_version": self.app_version,
            "schema_revision": self.schema_revision,
            "encrypted": True,
            "kdf": "PBKDF2-HMAC-SHA256",
            "kdf_iterations": DEFAULT_KDF_ITERATIONS,
            "salt_b64": base64.b64encode(salt).decode("ascii"),
            "database_tables": table_names,
            "storage_files": sorted(storage_files),
            "excluded_sensitive_config": sorted(self.EXCLUDED_CONFIG_FIELDS),
            "backup_scope": scope,
            "scope_user_id": scope_user_id,
        }


        checksums: Dict[str, str] = {
            "database.enc": hashlib.sha256(encrypted_db).hexdigest(),
        }
        for name, payload in encrypted_storage.items():
            checksums[f"storage/{name}.enc"] = hashlib.sha256(payload).hexdigest()
        manifest_without_hash["checksums"] = checksums

        manifest = AAOSBackupManifest(**manifest_without_hash)
        manifest_bytes = _canonical_json(manifest.to_dict())
        manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()

        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", manifest_bytes)
            zf.writestr("manifest.sha256", manifest_hash)
            zf.writestr("database.enc", encrypted_db)
            for name, payload in encrypted_storage.items():
                zf.writestr(f"storage/{name}.enc", payload)
            zf.writestr(
                "README.txt",
                "AA-OS encrypted portable backup. Cloud/provider credentials and AI API keys are intentionally excluded.\n",
            )
        return output.getvalue()

    def build_file(self, output_path: str, **kwargs: Any) -> str:
        data = self.build_bytes(**kwargs)
        path = Path(output_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(path)


class PortableBackupRestorer:
    """Restore a portable backup into an initialized SQLAlchemy database."""

    @staticmethod
    def _decode(value: Any) -> Any:
        if isinstance(value, dict) and "__bytes_b64__" in value:
            return base64.b64decode(value["__bytes_b64__"])
        return value

    @staticmethod
    def _coerce_for_column(column: Any, value: Any) -> Any:
        if value is None:
            return None
        value = PortableBackupRestorer._decode(value)
        try:
            python_type = column.type.python_type
        except Exception:
            return value
        if python_type is datetime and isinstance(value, str):
            return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
        if python_type is date and isinstance(value, str):
            return date.fromisoformat(value[:10])
        if python_type is bool:
            return bool(value)
        if python_type is str:
            return str(value)
        if python_type in (int, float):
            return python_type(value)
        return value

    def inspect_manifest(self, backup_bytes: bytes, passphrase: Optional[str] = None) -> Dict[str, Any]:
        if len(backup_bytes) > MAX_BACKUP_SIZE_BYTES:
            raise BackupError(
                f"Backup size ({len(backup_bytes)/(1024*1024):.1f} MB) exceeds maximum allowed size of 500 MB."
            )
        try:
            with zipfile.ZipFile(io.BytesIO(backup_bytes), "r") as zf:
                infolist = zf.infolist()
                if len(infolist) > MAX_ZIP_ENTRIES:
                    raise BackupError(f"Backup archive entry count ({len(infolist)}) exceeds maximum limit of {MAX_ZIP_ENTRIES}.")
                uncompressed_total = sum(info.file_size for info in infolist)
                if uncompressed_total > MAX_UNCOMPRESSED_BYTES:
                    raise BackupError(
                        f"Uncompressed backup size ({uncompressed_total/(1024*1024):.1f} MB) exceeds maximum allowed limit of 1 GB."
                    )
                manifest = json.loads(zf.read("manifest.json"))
                expected_hash = zf.read("manifest.sha256").decode("ascii")
                db_payload = zf.read("database.enc") if passphrase else None
        except BackupError:
            raise
        except Exception as exc:
            raise BackupError("Invalid AA-OS backup container.") from exc
        actual_hash = hashlib.sha256(_canonical_json(manifest)).hexdigest()
        if actual_hash != expected_hash:
            raise BackupError("Backup manifest integrity check failed.")
        if manifest.get("format") != BACKUP_FORMAT:
            raise BackupError("This file is not an AA-OS backup.")
        if manifest.get("format_version") != BACKUP_VERSION:
            raise BackupError(f"Unsupported AA-OS backup format: {manifest.get('format_version')}")
        if passphrase:
            salt = base64.b64decode(manifest["salt_b64"])
            payload = db_payload if db_payload is not None else self._read_checked(None, backup_bytes, "database.enc", manifest)
            expected = manifest.get("checksums", {}).get("database.enc")
            if expected and hashlib.sha256(payload).hexdigest() != expected:
                raise BackupError("Backup integrity check failed for database.enc.")
            _decrypt(payload, passphrase, salt, int(manifest["kdf_iterations"]))
        return manifest

    @staticmethod
    def _read_checked(zf: Any, backup_bytes: bytes, name: str, manifest: Mapping[str, Any]) -> bytes:
        if zf is None:
            zf = zipfile.ZipFile(io.BytesIO(backup_bytes), "r")
            close_after = True
        else:
            close_after = False
        try:
            payload = zf.read(name)
        except KeyError as exc:
            raise BackupError(f"Backup is missing required entry: {name}") from exc
        finally:
            if close_after:
                zf.close()
        expected = manifest.get("checksums", {}).get(name)
        if expected and hashlib.sha256(payload).hexdigest() != expected:
            raise BackupError(f"Backup integrity check failed for {name}.")
        return payload

    @staticmethod
    def _scoped_delete_user_data(conn: Any, table_map: Dict[str, Any], user_id: str) -> None:
        """Safely delete ONLY data belonging to user_id across the ownership graph."""
        uid = str(user_id)

        # 1. Discover user's project IDs
        user_project_ids = []
        if "projects" in table_map and hasattr(table_map["projects"].c, "owner_id"):
            res = conn.execute(table_map["projects"].select().where(table_map["projects"].c.owner_id == uid)).fetchall()
            user_project_ids = [str(r[0]) for r in res]

        # 2. Discover user's investigation IDs
        user_inv_ids = []
        if "investigations" in table_map:
            inv_table = table_map["investigations"]
            conds = []
            if hasattr(inv_table.c, "user_id"):
                conds.append(inv_table.c.user_id == uid)
            if user_project_ids and hasattr(inv_table.c, "project_id"):
                conds.append(inv_table.c.project_id.in_(user_project_ids))
            if conds:
                from sqlalchemy import or_
                res = conn.execute(inv_table.select().where(or_(*conds))).fetchall()
                user_inv_ids = [str(r[0]) for r in res]

        # 3. Discover user's dataset IDs
        user_dataset_ids = []
        if "datasets" in table_map and user_project_ids and hasattr(table_map["datasets"].c, "project_id"):
            res = conn.execute(table_map["datasets"].select().where(table_map["datasets"].c.project_id.in_(user_project_ids))).fetchall()
            user_dataset_ids = [str(r[0]) for r in res]

        # 4. Discover user's semantic model IDs
        user_sm_ids = []
        if "semantic_models" in table_map and user_project_ids and hasattr(table_map["semantic_models"].c, "project_id"):
            res = conn.execute(table_map["semantic_models"].select().where(table_map["semantic_models"].c.project_id.in_(user_project_ids))).fetchall()
            user_sm_ids = [str(r[0]) for r in res]

        # 5. Discover user's conversation IDs
        user_conv_ids = []
        if "conversations" in table_map and user_project_ids and hasattr(table_map["conversations"].c, "project_id"):
            res = conn.execute(table_map["conversations"].select().where(table_map["conversations"].c.project_id.in_(user_project_ids))).fetchall()
            user_conv_ids = [str(r[0]) for r in res]

        # 6. Discover user's alert rule IDs
        user_alert_ids = []
        if "alert_rules" in table_map and user_project_ids and hasattr(table_map["alert_rules"].c, "project_id"):
            res = conn.execute(table_map["alert_rules"].select().where(table_map["alert_rules"].c.project_id.in_(user_project_ids))).fetchall()
            user_alert_ids = [str(r[0]) for r in res]

        # 7. Discover user's hypothesis IDs
        user_hyp_ids = []
        if "hypotheses" in table_map and user_inv_ids and hasattr(table_map["hypotheses"].c, "investigation_id"):
            res = conn.execute(table_map["hypotheses"].select().where(table_map["hypotheses"].c.investigation_id.in_(user_inv_ids))).fetchall()
            user_hyp_ids = [str(r[0]) for r in res]

        # 8. Discover user's experiment IDs
        user_exp_ids = []
        if "experiments" in table_map and user_inv_ids and hasattr(table_map["experiments"].c, "investigation_id"):
            res = conn.execute(table_map["experiments"].select().where(table_map["experiments"].c.investigation_id.in_(user_inv_ids))).fetchall()
            user_exp_ids = [str(r[0]) for r in res]

        # 9. Discover user's evidence IDs
        user_ev_ids = []
        if "evidence" in table_map and user_inv_ids and hasattr(table_map["evidence"].c, "investigation_id"):
            res = conn.execute(table_map["evidence"].select().where(table_map["evidence"].c.investigation_id.in_(user_inv_ids))).fetchall()
            user_ev_ids = [str(r[0]) for r in res]

        def _del(tname: str, col_name: str, vals: Any):
            if tname not in table_map or not vals:
                return
            tbl = table_map[tname]
            if hasattr(tbl.c, col_name):
                col = getattr(tbl.c, col_name)
                if isinstance(vals, (list, set, tuple)):
                    if vals:
                        conn.execute(tbl.delete().where(col.in_(list(vals))))
                else:
                    conn.execute(tbl.delete().where(col == vals))

        # Leaf-to-root deletion to maintain foreign key integrity
        _del("evidence_verifications", "evidence_id", user_ev_ids)
        _del("observations", "experiment_id", user_exp_ids)
        _del("counter_hypotheses", "primary_hypothesis_id", user_hyp_ids)
        _del("counter_hypotheses", "counter_hypothesis_id", user_hyp_ids)
        _del("belief_updates", "investigation_id", user_inv_ids)
        _del("evidence", "investigation_id", user_inv_ids)
        _del("experiments", "investigation_id", user_inv_ids)
        _del("predictions", "investigation_id", user_inv_ids)
        _del("hypotheses", "investigation_id", user_inv_ids)

        for inv_child in (
            "investigation_step_executions", "investigation_resource_waits",
            "investigation_decision_requests", "investigation_data_readiness",
            "investigation_verdicts", "investigation_unknowns", "investigation_objectives",
            "investigation_jobs", "investigation_graph_edges", "investigation_executions",
            "investigation_events", "investigation_contracts"
        ):
            _del(inv_child, "investigation_id", user_inv_ids)

        _del("investigations", "id", user_inv_ids)

        for sm_child in (
            "semantic_business_rules", "semantic_time_definitions", "semantic_relationships",
            "semantic_metrics", "semantic_entities", "semantic_dimensions"
        ):
            _del(sm_child, "semantic_model_id", user_sm_ids)

        _del("semantic_models", "id", user_sm_ids)
        _del("messages", "conversation_id", user_conv_ids)
        _del("conversations", "id", user_conv_ids)
        _del("dataset_transformations", "dataset_id", user_dataset_ids)
        _del("dataset_versions", "dataset_id", user_dataset_ids)
        _del("datasets", "id", user_dataset_ids)

        _del("alert_events", "project_id", user_project_ids)
        _del("alert_rules", "project_id", user_project_ids)
        _del("reports", "project_id", user_project_ids)
        _del("dashboards", "project_id", user_project_ids)
        _del("analysis_runs", "project_id", user_project_ids)
        _del("business_metrics", "project_id", user_project_ids)
        _del("business_glossary", "project_id", user_project_ids)

        _del("projects", "id", user_project_ids)
        _del("user_ai_configs", "user_id", uid)
        _del("organization_members", "user_id", uid)
        _del("api_keys", "user_id", uid)
        _del("audit_logs", "user_id", uid)

    def restore_bytes(
        self,
        *,
        backup_bytes: bytes,
        passphrase: str,
        engine: Any,
        metadata: Any,
        storage_root: Optional[str] = None,
        replace_existing: bool = False,
        restoring_user_id: Optional[str] = None,
        is_system_restore: bool = False,
    ) -> Dict[str, Any]:
        manifest = self.inspect_manifest(backup_bytes)
        backup_scope = manifest.get('backup_scope', BackupScope.SYSTEM)
        manifest_user_id = manifest.get('scope_user_id')

        # Tenant isolation check:
        # A non-admin restoring a USER backup cannot restore a backup created by another user.
        if restoring_user_id and backup_scope == BackupScope.USER and manifest_user_id:
            if str(manifest_user_id) != str(restoring_user_id):
                raise BackupError(
                    f"Access denied: cannot restore backup belonging to user {manifest_user_id} "
                    f"into account {restoring_user_id}."
                )

        if replace_existing and backup_scope == BackupScope.SYSTEM and not is_system_restore:
            raise BackupError(
                "replace_existing=True for SYSTEM backups requires explicit is_system_restore=True authorization."
            )

        salt = base64.b64decode(manifest["salt_b64"])
        iterations = int(manifest["kdf_iterations"])

        with zipfile.ZipFile(io.BytesIO(backup_bytes), "r") as zf:
            db_payload = self._read_checked(zf, backup_bytes, "database.enc", manifest)
            raw_db = _decrypt(db_payload, passphrase, salt, iterations)
            database = json.loads(raw_db.decode("utf-8"))["tables"]
            storage_payloads = {}
            for name in manifest.get("storage_files", []):
                entry = f"storage/{name}.enc"
                payload = self._read_checked(zf, backup_bytes, entry, manifest)
                storage_payloads[name] = _decrypt(payload, passphrase, salt, iterations)

        effective_user_id = manifest_user_id or restoring_user_id
        # In USER scope: ALWAYS filter the restored rows through the ownership graph
        # to ensure foreign/tampered rows inside the backup cannot be inserted.
        if backup_scope == BackupScope.USER and effective_user_id:
            database = PortableBackupBuilder._filter_rows_by_scope(
                database, scope=BackupScope.USER, scope_user_id=effective_user_id
            )

        table_map = {t.name: t for t in metadata.sorted_tables}
        restored_counts: Dict[str, int] = {}
        warnings: list[str] = []

        with engine.begin() as conn:
            if replace_existing:
                if is_system_restore and backup_scope == BackupScope.SYSTEM:
                    # Global deletion allowed ONLY for authorized system restore
                    for table_name in reversed(list(table_map.keys())):
                        table = table_map[table_name]
                        conn.execute(table.delete())
                elif effective_user_id:
                    # Scoped deletion: delete ONLY existing rows belonging to effective_user_id
                    self._scoped_delete_user_data(conn, table_map, effective_user_id)

            for table_name, rows in database.items():
                table = table_map.get(table_name)
                if table is None:
                    warnings.append(f"Skipped unknown table from backup: {table_name}")
                    continue
                if not rows:
                    restored_counts[table_name] = 0
                    continue
                valid_columns = {c.name: c for c in table.columns}
                inserted = 0
                for row in rows:
                    mapped = {
                        name: self._coerce_for_column(valid_columns[name], value)
                        for name, value in row.items()
                        if name in valid_columns
                    }
                    if table_name in ("users", "organizations") and "id" in mapped and hasattr(table.c, "id"):
                        existing = conn.execute(table.select().where(table.c.id == mapped["id"])).fetchone()
                        if existing:
                            if replace_existing:
                                conn.execute(table.update().where(table.c.id == mapped["id"]).values(**mapped))
                                inserted += 1
                            continue
                    conn.execute(table.insert().values(**mapped))
                    inserted += 1
                restored_counts[table_name] = inserted

        if storage_root and storage_payloads:
            root = Path(storage_root).expanduser().resolve()
            root.mkdir(parents=True, exist_ok=True)
            for name, content in storage_payloads.items():
                target = (root / name).resolve()
                if root not in target.parents and target != root:
                    raise BackupError(f"Unsafe storage path in backup: {name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)

        return {
            "status": "RESTORED",
            "backup_format": manifest["format_version"],
            "created_at": manifest["created_at"],
            "restored_tables": restored_counts,
            "restored_storage_files": len(storage_payloads),
            "warnings": warnings,
            "backup_scope": backup_scope,
            "scope_user_id": manifest_user_id,
            "migration_note": "Rows are restored by logical column intersection so backups can migrate across additive schema changes; run current migrations before restore.",
        }

