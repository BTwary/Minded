import tempfile
from pathlib import Path

import pytest
from sqlalchemy import create_engine

from apps.api.src.models.entities import Base, Project
from packages.analytics_core.src.backup.portable_backup import BackupError, PortableBackupBuilder, PortableBackupRestorer
from packages.analytics_core.src.providers.storage import LocalStorageProvider


def test_encrypted_backup_round_trip_and_migration_shape():
    with tempfile.TemporaryDirectory() as td:
        db1 = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(db1)
        with db1.begin() as conn:
            conn.execute(Project.__table__.insert().values(id="proj-1", org_id="org-1", owner_id="user-1", name="AAOS"))
        storage = Path(td) / "storage"
        storage.mkdir()
        (storage / "sales.csv").write_text("a,b\n1,2\n", encoding="utf-8")

        backup = PortableBackupBuilder(app_version="test", schema_revision="test").build_bytes(
            engine=db1,
            metadata=Base.metadata,
            storage_root=str(storage),
            passphrase="correct horse battery staple",
            include_source_files=True,
        )
        restorer = PortableBackupRestorer()
        manifest = restorer.inspect_manifest(backup, passphrase="correct horse battery staple")
        assert manifest["encrypted"] is True
        assert "sales.csv" in manifest["storage_files"]
        assert "AI_API_KEY" in manifest["excluded_sensitive_config"]

        db2 = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(db2)
        result = restorer.restore_bytes(
            backup_bytes=backup,
            passphrase="correct horse battery staple",
            engine=db2,
            metadata=Base.metadata,
            storage_root=str(Path(td) / "restored"),
        )
        assert result["status"] == "RESTORED"
        with db2.connect() as conn:
            row = conn.execute(Project.__table__.select()).mappings().first()
            assert row["id"] == "proj-1"
        assert (Path(td) / "restored" / "sales.csv").read_text(encoding="utf-8") == "a,b\n1,2\n"

        with pytest.raises(BackupError):
            restorer.restore_bytes(
                backup_bytes=backup,
                passphrase="wrong-passphrase",
                engine=db2,
                metadata=Base.metadata,
                storage_root=str(Path(td) / "wrong"),
            )


def test_local_storage_rejects_path_escape():
    with tempfile.TemporaryDirectory() as td:
        provider = LocalStorageProvider(td, retention_days=3650)
        with pytest.raises(ValueError):
            provider.save_file("../../escape.txt", b"blocked")
        provider.save_file("safe/file.txt", b"ok")
        assert provider.read_file("safe/file.txt") == b"ok"

class _FakeCloudProvider:
    provider_type = "fake-cloud"
    is_local = False

    def __init__(self):
        self.objects = {}

    def save_file(self, relative_path, content):
        self.objects[relative_path] = content
        return f"fake://bucket/{relative_path}"

    def read_file(self, relative_path):
        return self.objects[relative_path]

    def file_exists(self, relative_path):
        return relative_path in self.objects

    def delete_file(self, relative_path):
        return self.objects.pop(relative_path, None) is not None

    def get_local_path(self, relative_path):
        raise AssertionError("Fake provider should not need a local read for this write-only test")

    def test_connection(self):
        return True, "ok"

def test_storage_service_uses_provider_for_new_objects():
    from apps.api.src.services.storage_service import StorageService
    provider = _FakeCloudProvider()
    with tempfile.TemporaryDirectory() as td:
        service = StorageService(base_dir=td, provider=provider)
        key = service.save_raw_file("d1", 1, "sales.csv", b"a,b\n1,2\n")
        assert key == "datasets/d1/v1/sales.csv"
        assert provider.objects[key] == b"a,b\n1,2\n"
