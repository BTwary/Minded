"""P0 Regression Suite: Backup Scoping, Tenant Isolation, and Safe Restoration.

Verifies:
1. USER-scoped backup only includes data in the requesting user's ownership graph:
   user -> projects -> datasets, dataset_versions, semantic_models, investigations,
   hypotheses, experiments, observations, evidence, evidence_verifications, reports.
   Other users' entities are strictly excluded.
2. Source file collection in USER backup captures only datasets belonging to the user
   (under datasets/{dataset_id}/...) and does NOT capture other users' dataset files.
3. USER-scoped restore with replace_existing=True deletes ONLY the restoring user's
   existing rows and NEVER deletes other users' data in the database.
4. Non-admin / USER restore cannot perform global database deletion (replace_existing=True
   for SYSTEM backups requires is_system_restore=True).
5. Cross-user restore denial: A user cannot restore another user's backup archive.
6. Container limits: Oversized backups and zip bomb archives are rejected.
"""
import base64
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import sqlalchemy as sa

sys.path.insert(0, '.')

from packages.analytics_core.src.backup.portable_backup import (
    PortableBackupBuilder,
    PortableBackupRestorer,
    BackupScope,
    BackupError,
    _decrypt,
    DEFAULT_KDF_ITERATIONS,
)


class TestBackupScopeAndRestoration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.storage_root = os.path.join(self.temp_dir, "storage")
        os.makedirs(self.storage_root, exist_ok=True)

        # Real in-memory SQLite database with SQLAlchemy Core schema
        self.engine = sa.create_engine("sqlite:///:memory:")
        self.metadata = sa.MetaData()

        self.users = sa.Table(
            "users", self.metadata,
            sa.Column("id", sa.String, primary_key=True),
            sa.Column("email", sa.String),
        )
        self.projects = sa.Table(
            "projects", self.metadata,
            sa.Column("id", sa.String, primary_key=True),
            sa.Column("owner_id", sa.String),
            sa.Column("name", sa.String),
        )
        self.datasets = sa.Table(
            "datasets", self.metadata,
            sa.Column("id", sa.String, primary_key=True),
            sa.Column("project_id", sa.String),
            sa.Column("name", sa.String),
        )
        self.dataset_versions = sa.Table(
            "dataset_versions", self.metadata,
            sa.Column("id", sa.String, primary_key=True),
            sa.Column("dataset_id", sa.String),
            sa.Column("version", sa.String),
            sa.Column("file_path", sa.String),
        )
        self.semantic_models = sa.Table(
            "semantic_models", self.metadata,
            sa.Column("id", sa.String, primary_key=True),
            sa.Column("project_id", sa.String),
            sa.Column("name", sa.String),
        )
        self.semantic_dimensions = sa.Table(
            "semantic_dimensions", self.metadata,
            sa.Column("id", sa.String, primary_key=True),
            sa.Column("semantic_model_id", sa.String),
            sa.Column("name", sa.String),
        )
        self.investigations = sa.Table(
            "investigations", self.metadata,
            sa.Column("id", sa.String, primary_key=True),
            sa.Column("project_id", sa.String),
            sa.Column("user_id", sa.String),
            sa.Column("question", sa.String),
        )
        self.hypotheses = sa.Table(
            "hypotheses", self.metadata,
            sa.Column("id", sa.String, primary_key=True),
            sa.Column("investigation_id", sa.String),
            sa.Column("hypothesis_text", sa.String),
        )
        self.experiments = sa.Table(
            "experiments", self.metadata,
            sa.Column("id", sa.String, primary_key=True),
            sa.Column("investigation_id", sa.String),
            sa.Column("name", sa.String),
        )
        self.observations = sa.Table(
            "observations", self.metadata,
            sa.Column("id", sa.String, primary_key=True),
            sa.Column("experiment_id", sa.String),
            sa.Column("data", sa.String),
        )
        self.evidence = sa.Table(
            "evidence", self.metadata,
            sa.Column("id", sa.String, primary_key=True),
            sa.Column("investigation_id", sa.String),
            sa.Column("evidence_text", sa.String),
        )
        self.evidence_verifications = sa.Table(
            "evidence_verifications", self.metadata,
            sa.Column("id", sa.String, primary_key=True),
            sa.Column("evidence_id", sa.String),
            sa.Column("status", sa.String),
        )
        self.reports = sa.Table(
            "reports", self.metadata,
            sa.Column("id", sa.String, primary_key=True),
            sa.Column("project_id", sa.String),
            sa.Column("title", sa.String),
        )

        self.metadata.create_all(self.engine)
        self._populate_multi_tenant_data()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _populate_multi_tenant_data(self):
        with self.engine.begin() as conn:
            # Users
            conn.execute(self.users.insert(), [
                {"id": "alice", "email": "alice@example.com"},
                {"id": "bob", "email": "bob@example.com"},
            ])
            # Projects
            conn.execute(self.projects.insert(), [
                {"id": "proj-alice-1", "owner_id": "alice", "name": "Alice Project"},
                {"id": "proj-bob-1", "owner_id": "bob", "name": "Bob Secret Project"},
            ])
            # Datasets
            conn.execute(self.datasets.insert(), [
                {"id": "ds-alice-1", "project_id": "proj-alice-1", "name": "Alice Dataset"},
                {"id": "ds-bob-1", "project_id": "proj-bob-1", "name": "Bob Confidential Dataset"},
            ])
            # Dataset Versions
            conn.execute(self.dataset_versions.insert(), [
                {"id": "dsv-alice-1", "dataset_id": "ds-alice-1", "version": "v1", "file_path": "datasets/ds-alice-1/v1/alice.csv"},
                {"id": "dsv-bob-1", "dataset_id": "ds-bob-1", "version": "v1", "file_path": "datasets/ds-bob-1/v1/bob.csv"},
            ])
            # Semantic Models & Dimensions
            conn.execute(self.semantic_models.insert(), [
                {"id": "sm-alice-1", "project_id": "proj-alice-1", "name": "Alice Model"},
                {"id": "sm-bob-1", "project_id": "proj-bob-1", "name": "Bob Model"},
            ])
            conn.execute(self.semantic_dimensions.insert(), [
                {"id": "smd-alice-1", "semantic_model_id": "sm-alice-1", "name": "Alice Dim"},
                {"id": "smd-bob-1", "semantic_model_id": "sm-bob-1", "name": "Bob Dim"},
            ])
            # Investigations
            conn.execute(self.investigations.insert(), [
                {"id": "inv-alice-1", "project_id": "proj-alice-1", "user_id": "alice", "question": "Why did Alice sales grow?"},
                {"id": "inv-bob-1", "project_id": "proj-bob-1", "user_id": "bob", "question": "Bob confidential research"},
            ])
            # Hypotheses
            conn.execute(self.hypotheses.insert(), [
                {"id": "hyp-alice-1", "investigation_id": "inv-alice-1", "hypothesis_text": "Marketing boost"},
                {"id": "hyp-bob-1", "investigation_id": "inv-bob-1", "hypothesis_text": "Secret factor"},
            ])
            # Experiments
            conn.execute(self.experiments.insert(), [
                {"id": "exp-alice-1", "investigation_id": "inv-alice-1", "name": "Alice Test"},
                {"id": "exp-bob-1", "investigation_id": "inv-bob-1", "name": "Bob Test"},
            ])
            # Observations
            conn.execute(self.observations.insert(), [
                {"id": "obs-alice-1", "experiment_id": "exp-alice-1", "data": "alice-obs"},
                {"id": "obs-bob-1", "experiment_id": "exp-bob-1", "data": "bob-obs"},
            ])
            # Evidence & Verifications
            conn.execute(self.evidence.insert(), [
                {"id": "ev-alice-1", "investigation_id": "inv-alice-1", "evidence_text": "Alice strong evidence"},
                {"id": "ev-bob-1", "investigation_id": "inv-bob-1", "evidence_text": "Bob strong evidence"},
            ])
            conn.execute(self.evidence_verifications.insert(), [
                {"id": "ver-alice-1", "evidence_id": "ev-alice-1", "status": "VERIFIED"},
                {"id": "ver-bob-1", "evidence_id": "ev-bob-1", "status": "VERIFIED"},
            ])
            # Reports
            conn.execute(self.reports.insert(), [
                {"id": "rep-alice-1", "project_id": "proj-alice-1", "title": "Alice Q3 Report"},
                {"id": "rep-bob-1", "project_id": "proj-bob-1", "title": "Bob Proprietary Report"},
            ])

        # Create files on disk under storage_root
        alice_file = Path(self.storage_root) / "datasets" / "ds-alice-1" / "v1" / "alice.csv"
        alice_file.parent.mkdir(parents=True, exist_ok=True)
        alice_file.write_text("id,val\n1,alice_data\n", encoding="utf-8")

        bob_file = Path(self.storage_root) / "datasets" / "ds-bob-1" / "v1" / "bob.csv"
        bob_file.parent.mkdir(parents=True, exist_ok=True)
        bob_file.write_text("id,val\n1,bob_confidential\n", encoding="utf-8")

    def _extract_database(self, backup_bytes: bytes, passphrase: str) -> dict:
        with zipfile.ZipFile(io.BytesIO(backup_bytes)) as zf:
            manifest = json.loads(zf.read("manifest.json"))
            db_enc = zf.read("database.enc")
        salt = base64.b64decode(manifest["salt_b64"])
        raw = _decrypt(db_enc, passphrase, salt, int(manifest["kdf_iterations"]))
        return json.loads(raw)["tables"]

    def test_user_backup_ownership_graph_isolation(self):
        """Alice's USER backup includes Alice's complete ownership graph and zero Bob rows."""
        builder = PortableBackupBuilder(app_version="v29", schema_revision="v29")
        backup_bytes = builder.build_bytes(
            engine=self.engine,
            metadata=self.metadata,
            storage_root=self.storage_root,
            passphrase="passphrase-alice-123",
            include_source_files=True,
            scope=BackupScope.USER,
            scope_user_id="alice",
        )

        db = self._extract_database(backup_bytes, "passphrase-alice-123")

        # 1. Projects
        proj_ids = {r["id"] for r in db.get("projects", [])}
        self.assertIn("proj-alice-1", proj_ids)
        self.assertNotIn("proj-bob-1", proj_ids)

        # 2. Datasets & Versions
        ds_ids = {r["id"] for r in db.get("datasets", [])}
        self.assertIn("ds-alice-1", ds_ids)
        self.assertNotIn("ds-bob-1", ds_ids)

        dsv_ids = {r["id"] for r in db.get("dataset_versions", [])}
        self.assertIn("dsv-alice-1", dsv_ids)
        self.assertNotIn("dsv-bob-1", dsv_ids)

        # 3. Semantic Models & Dimensions
        sm_ids = {r["id"] for r in db.get("semantic_models", [])}
        self.assertIn("sm-alice-1", sm_ids)
        self.assertNotIn("sm-bob-1", sm_ids)

        smd_ids = {r["id"] for r in db.get("semantic_dimensions", [])}
        self.assertIn("smd-alice-1", smd_ids)
        self.assertNotIn("smd-bob-1", smd_ids)

        # 4. Investigations
        inv_ids = {r["id"] for r in db.get("investigations", [])}
        self.assertIn("inv-alice-1", inv_ids)
        self.assertNotIn("inv-bob-1", inv_ids)

        # 5. Hypotheses, Experiments, Observations, Evidence, Verifications
        hyp_ids = {r["id"] for r in db.get("hypotheses", [])}
        self.assertIn("hyp-alice-1", hyp_ids)
        self.assertNotIn("hyp-bob-1", hyp_ids)

        exp_ids = {r["id"] for r in db.get("experiments", [])}
        self.assertIn("exp-alice-1", exp_ids)
        self.assertNotIn("exp-bob-1", exp_ids)

        obs_ids = {r["id"] for r in db.get("observations", [])}
        self.assertIn("obs-alice-1", obs_ids)
        self.assertNotIn("obs-bob-1", obs_ids)

        ev_ids = {r["id"] for r in db.get("evidence", [])}
        self.assertIn("ev-alice-1", ev_ids)
        self.assertNotIn("ev-bob-1", ev_ids)

        ver_ids = {r["id"] for r in db.get("evidence_verifications", [])}
        self.assertIn("ver-alice-1", ver_ids)
        self.assertNotIn("ver-bob-1", ver_ids)

        # 6. Reports
        rep_ids = {r["id"] for r in db.get("reports", [])}
        self.assertIn("rep-alice-1", rep_ids)
        self.assertNotIn("rep-bob-1", rep_ids)

    def test_user_backup_dataset_source_files_collected_and_isolated(self):
        """Alice's USER backup includes datasets/ds-alice-1/v1/alice.csv and NOT Bob's dataset files."""
        builder = PortableBackupBuilder(app_version="v29", schema_revision="v29")
        backup_bytes = builder.build_bytes(
            engine=self.engine,
            metadata=self.metadata,
            storage_root=self.storage_root,
            passphrase="passphrase-alice-123",
            include_source_files=True,
            scope=BackupScope.USER,
            scope_user_id="alice",
        )

        with zipfile.ZipFile(io.BytesIO(backup_bytes)) as zf:
            manifest = json.loads(zf.read("manifest.json"))
            storage_files = manifest.get("storage_files", [])

        self.assertIn("datasets/ds-alice-1/v1/alice.csv", storage_files)
        self.assertNotIn("datasets/ds-bob-1/v1/bob.csv", storage_files)

    def test_user_restore_replace_existing_preserves_other_user_data(self):
        """Restoring Alice's backup with replace_existing=True deletes only Alice's data, preserving Bob's data."""
        builder = PortableBackupBuilder(app_version="v29", schema_revision="v29")
        alice_backup = builder.build_bytes(
            engine=self.engine,
            metadata=self.metadata,
            storage_root=self.storage_root,
            passphrase="passphrase-alice-123",
            include_source_files=False,
            scope=BackupScope.USER,
            scope_user_id="alice",
        )

        # Mutate Alice's project name and add a new row
        with self.engine.begin() as conn:
            conn.execute(
                self.projects.update()
                .where(self.projects.c.id == "proj-alice-1")
                .values(name="Alice Mutated Name")
            )

        restorer = PortableBackupRestorer()
        restore_result = restorer.restore_bytes(
            backup_bytes=alice_backup,
            passphrase="passphrase-alice-123",
            engine=self.engine,
            metadata=self.metadata,
            storage_root=self.storage_root,
            replace_existing=True,
            restoring_user_id="alice",
            is_system_restore=False,
        )

        # Verify Alice's project was restored to original
        with self.engine.connect() as conn:
            alice_proj = conn.execute(
                self.projects.select().where(self.projects.c.id == "proj-alice-1")
            ).fetchone()
            self.assertIsNotNone(alice_proj)
            self.assertEqual(alice_proj.name, "Alice Project")

            # CRITICAL: Bob's data must be completely preserved!
            bob_proj = conn.execute(
                self.projects.select().where(self.projects.c.id == "proj-bob-1")
            ).fetchone()
            self.assertIsNotNone(bob_proj, "Bob project MUST NOT be deleted by Alice's replace_existing restore!")
            self.assertEqual(bob_proj.name, "Bob Secret Project")

            bob_ds = conn.execute(
                self.datasets.select().where(self.datasets.c.id == "ds-bob-1")
            ).fetchone()
            self.assertIsNotNone(bob_ds, "Bob dataset MUST NOT be deleted!")

            bob_inv = conn.execute(
                self.investigations.select().where(self.investigations.c.id == "inv-bob-1")
            ).fetchone()
            self.assertIsNotNone(bob_inv, "Bob investigation MUST NOT be deleted!")

            bob_rep = conn.execute(
                self.reports.select().where(self.reports.c.id == "rep-bob-1")
            ).fetchone()
            self.assertIsNotNone(bob_rep, "Bob report MUST NOT be deleted!")

    def test_non_admin_cannot_system_restore_global_delete(self):
        """Non-admin cannot perform global delete via replace_existing on a SYSTEM backup."""
        builder = PortableBackupBuilder(app_version="v29", schema_revision="v29")
        sys_backup = builder.build_bytes(
            engine=self.engine,
            metadata=self.metadata,
            storage_root=self.storage_root,
            passphrase="passphrase-sys-123",
            include_source_files=False,
            scope=BackupScope.SYSTEM,
            scope_user_id=None,
        )

        restorer = PortableBackupRestorer()
        with self.assertRaises(BackupError) as ctx:
            restorer.restore_bytes(
                backup_bytes=sys_backup,
                passphrase="passphrase-sys-123",
                engine=self.engine,
                metadata=self.metadata,
                storage_root=self.storage_root,
                replace_existing=True,
                restoring_user_id="alice",
                is_system_restore=False,  # NOT authorized for system restore
            )
        self.assertIn("requires explicit is_system_restore=True authorization", str(ctx.exception))

    def test_user_cannot_restore_other_user_backup(self):
        """Alice cannot restore a backup generated by Bob."""
        builder = PortableBackupBuilder(app_version="v29", schema_revision="v29")
        bob_backup = builder.build_bytes(
            engine=self.engine,
            metadata=self.metadata,
            storage_root=self.storage_root,
            passphrase="passphrase-bob-123",
            include_source_files=False,
            scope=BackupScope.USER,
            scope_user_id="bob",
        )

        restorer = PortableBackupRestorer()
        with self.assertRaises(BackupError) as ctx:
            restorer.restore_bytes(
                backup_bytes=bob_backup,
                passphrase="passphrase-bob-123",
                engine=self.engine,
                metadata=self.metadata,
                storage_root=self.storage_root,
                replace_existing=False,
                restoring_user_id="alice",  # Alice trying to restore Bob's backup
                is_system_restore=False,
            )
        self.assertIn("cannot restore backup belonging to user bob", str(ctx.exception))

    def test_oversized_backup_bytes_rejected(self):
        restorer = PortableBackupRestorer()
        fake_oversized = b"X" * (500 * 1024 * 1024 + 1)
        with self.assertRaises(BackupError) as ctx:
            restorer.inspect_manifest(fake_oversized)
        self.assertIn("exceeds maximum allowed size", str(ctx.exception))

    def test_zip_bomb_excess_entries_rejected(self):
        restorer = PortableBackupRestorer()
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for i in range(10_001):
                zf.writestr(f"file_{i}.txt", b"x")
        with self.assertRaises(BackupError) as ctx:
            restorer.inspect_manifest(buf.getvalue())
        self.assertIn("entry count", str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
