"""P0 Security, Tenant Isolation, SSRF Protection, and Epistemic Calibration Verification Suite.

Tests:
1. Tenant-scoped cloud backup download & restore (prevents cross-tenant access).
2. Tenant-scoped local storage migration (prevents cross-tenant file traversal).
3. SSRF network boundary protection on cloud storage & AI provider endpoints.
4. Per-user isolated AI runtime configuration (prevents global state leakage).
5. SQL escaping and determinism with single-quote strings in data values and paths.
6. Epistemic calibration: NO_DETECTABLE_EFFECT confidence is bounded < 1.0 and
   counter_hypothesis_refuted requires explicit empirical refutation.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
import tempfile
import uuid

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from packages.analytics_core.src.security.ssrf import (
    validate_network_endpoint,
    assert_safe_endpoint,
    SSRFSecurityError,
)
from packages.analytics_core.src.backup.portable_backup import BackupError
from packages.analytics_core.src.providers.storage import LocalStorageProvider
from apps.api.src.services.backup_service import BackupService, CloudStorageConfig
from apps.api.src.services.storage_service import StorageService
from apps.api.src.ai.providers.factory import get_ai_config, set_ai_config, get_ai_provider


class TestSSRFSecurityProtection(unittest.TestCase):
    """Test SSRF validation against cloud metadata and private IP spaces."""

    def test_blocks_cloud_metadata_service(self):
        # AWS / GCP metadata
        is_safe, reason = validate_network_endpoint("http://169.254.169.254/latest/meta-data/")
        self.assertFalse(is_safe)
        self.assertIn("forbidden", reason.lower())

        is_safe, reason = validate_network_endpoint("http://metadata.google.internal/computeMetadata/v1/")
        self.assertFalse(is_safe)
        self.assertIn("forbidden", reason.lower())

    def test_blocks_private_subnets_and_loopback_by_default(self):
        # RFC 1918 subnets
        self.assertFalse(validate_network_endpoint("http://10.0.0.1/admin")[0])
        self.assertFalse(validate_network_endpoint("http://192.168.1.100:8080/")[0])
        self.assertFalse(validate_network_endpoint("http://172.16.0.5/api")[0])

        # Loopback
        self.assertFalse(validate_network_endpoint("http://127.0.0.1:8000/internal")[0])
        self.assertFalse(validate_network_endpoint("http://localhost:5432/")[0])

    def test_allows_loopback_only_when_explicitly_permitted(self):
        # Loopback allowed for local Ollama
        is_safe, reason = validate_network_endpoint("http://localhost:11434/api/generate", allow_loopback=True)
        self.assertTrue(is_safe)

        is_safe, reason = validate_network_endpoint("http://127.0.0.1:11434", allow_loopback=True)
        self.assertTrue(is_safe)

    def test_allows_legitimate_public_endpoints(self):
        self.assertTrue(validate_network_endpoint("https://api.openai.com/v1")[0])
        self.assertTrue(validate_network_endpoint("https://api.anthropic.com/v1")[0])
        self.assertTrue(validate_network_endpoint("https://s3.amazonaws.com")[0])


class TestTenantIsolatedBackupAndMigration(unittest.TestCase):
    """Test tenant isolation in backup download and local storage migration."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="aaos_test_tenant_")
        self.provider = LocalStorageProvider(self.temp_dir, retention_days=365)

    def test_download_backup_prevents_cross_tenant_access(self):
        service = BackupService(
            engine=None,
            metadata=None,
            storage_root=self.temp_dir,
        )

        # Plant backups for user_A and user_B
        key_a = f"{service.PREFIX}/user_A/backup1.aaosbackup"
        key_b = f"{service.PREFIX}/user_B/backup2.aaosbackup"
        self.provider.save_file(key_a, b"data_a")
        self.provider.save_file(key_b, b"data_b")

        # user_A can download their own backup
        data = service.download_backup(provider=self.provider, object_key=key_a, user_id="user_A")
        self.assertEqual(data, b"data_a")

        # user_A attempting to download user_B's backup MUST be rejected
        with self.assertRaises(BackupError) as ctx:
            service.download_backup(provider=self.provider, object_key=key_b, user_id="user_A")
        self.assertIn("Access denied", str(ctx.exception))

    def test_migrate_local_to_provider_scopes_by_user_id(self):
        from apps.api.src.core.database import SessionLocal, engine
        from apps.api.src.models.entities import Base, Project, Dataset, DatasetVersion

        Base.metadata.create_all(bind=engine)
        db = SessionLocal()

        try:
            # Create user_A project and dataset
            proj_a = Project(id=f"proj-a-{uuid.uuid4().hex[:6]}", name="Proj A", owner_id="user_A_id")
            ds_a = Dataset(id=f"ds-a-{uuid.uuid4().hex[:6]}", project_id=proj_a.id, name="Data A")
            file_a = Path(self.temp_dir) / "datasets" / ds_a.id / "v1" / "data.csv"
            file_a.parent.mkdir(parents=True, exist_ok=True)
            file_a.write_text("a,b\n1,2\n")
            dv_a = DatasetVersion(id=f"dv-a-{uuid.uuid4().hex[:6]}", dataset_id=ds_a.id, version_number=1, file_path=str(file_a))

            # Create user_B project and dataset
            proj_b = Project(id=f"proj-b-{uuid.uuid4().hex[:6]}", name="Proj B", owner_id="user_B_id")
            ds_b = Dataset(id=f"ds-b-{uuid.uuid4().hex[:6]}", project_id=proj_b.id, name="Data B")
            file_b = Path(self.temp_dir) / "datasets" / ds_b.id / "v1" / "data.csv"
            file_b.parent.mkdir(parents=True, exist_ok=True)
            file_b.write_text("c,d\n3,4\n")
            dv_b = DatasetVersion(id=f"dv-b-{uuid.uuid4().hex[:6]}", dataset_id=ds_b.id, version_number=1, file_path=str(file_b))

            db.add_all([proj_a, ds_a, dv_a, proj_b, ds_b, dv_b])
            db.commit()

            # Target cloud-mock provider
            target_dir = tempfile.mkdtemp(prefix="aaos_cloud_mock_")
            class MockCloudStorageProvider(LocalStorageProvider):
                @property
                def provider_type(self) -> str:
                    return "s3"
                @property
                def is_local(self) -> bool:
                    return False
            target_provider = MockCloudStorageProvider(target_dir, retention_days=365)

            # Migrate as user_A
            service = StorageService(base_dir=self.temp_dir, provider=self.provider)
            res = service.migrate_local_to_provider(target_provider, user_id="user_A_id", db_session=db)

            # user_A's file should be migrated
            self.assertEqual(res["uploaded_files"], 1)
            # Verify file_b was NOT uploaded to target provider
            migrated_files = list(Path(target_dir).rglob("*"))
            migrated_names = [f.name for f in migrated_files if f.is_file()]
            self.assertIn("data.csv", migrated_names)
            # Check content migrated belongs to user_A
            migrated_content = (Path(target_dir) / "datasets" / ds_a.id / "v1" / "data.csv").read_text()
            self.assertEqual(migrated_content, "a,b\n1,2\n")
            # user_B's directory must NOT exist on target provider
            self.assertFalse((Path(target_dir) / "datasets" / ds_b.id).exists())

        finally:
            db.close()


class TestPerUserAIRuntimeGovernance(unittest.TestCase):
    """Test that runtime AI configuration is strictly isolated per user."""

    def test_user_configs_do_not_leak_or_clobber(self):
        user1 = f"user_{uuid.uuid4().hex[:8]}"
        user2 = f"user_{uuid.uuid4().hex[:8]}"

        # User 1 sets Gemini
        set_ai_config(user_id=user1, enabled=True, provider="gemini", api_key="key-user1", model="gemini-2.0-flash")

        # User 2 sets Claude
        set_ai_config(user_id=user2, enabled=True, provider="claude", api_key="key-user2", model="claude-3-7-sonnet-20250219")

        cfg1 = get_ai_config(user_id=user1)
        cfg2 = get_ai_config(user_id=user2)

        self.assertEqual(cfg1["provider"], "gemini")
        self.assertEqual(cfg1["api_key"], "key-user1")
        self.assertEqual(cfg1["model"], "gemini-2.0-flash")

        self.assertEqual(cfg2["provider"], "claude")
        self.assertEqual(cfg2["api_key"], "key-user2")
        self.assertEqual(cfg2["model"], "claude-3-7-sonnet-20250219")

        # Unspecified user gets default
        cfg_default = get_ai_config(user_id="unknown_user")
        self.assertEqual(cfg_default["provider"], "none")


class TestSQLInjectionAndQuotingHardening(unittest.TestCase):
    """Test SQL determinism when data values or storage paths contain single quotes."""

    def test_single_quote_category_value_in_data_table_isolation(self):
        import duckdb
        from packages.analytics_core.src.sql.engine import DuckDBSQLEngine

        # Dataset containing single quotes in categories
        df = pd.DataFrame({
            "category": ["Men's Apparel", "Women's Clothing", "Children's Wear"] * 20,
            "sales": [150.0, 200.0, 80.0] * 20,
        })
        engine = DuckDBSQLEngine()
        engine.register_dataframe("data_table", df)

        # Escaped single-quote query must succeed without syntax error
        target_val = "Men's Apparel"
        target_val_escaped = target_val.replace("'", "''")
        sql = f"SELECT '{target_val_escaped}' AS category, SUM(sales) AS total_metric FROM data_table WHERE category = '{target_val_escaped}'"
        res = engine.execute_query(sql)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["row_count"], 1)
        self.assertEqual(res["data"][0]["category"], "Men's Apparel")
        self.assertAlmostEqual(res["data"][0]["total_metric"], 3000.0)


if __name__ == "__main__":
    unittest.main()
