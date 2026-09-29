import os
import tempfile
import time
from pathlib import Path
import unittest

from packages.analytics_core.src.providers.retention import LocalFileRetentionManager


class TestLocalFileRetention(unittest.TestCase):
    def test_deletes_files_older_than_180_days_and_keeps_new_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_file = root / "old.csv"
            new_file = root / "new.parquet"
            old_file.write_bytes(b"old")
            new_file.write_bytes(b"new")

            now = time.time()
            old_mtime = now - (181 * 24 * 60 * 60)
            os.utime(old_file, (old_mtime, old_mtime))

            manager = LocalFileRetentionManager(str(root), retention_days=180)
            result = manager.prune(now=now)

            self.assertFalse(old_file.exists())
            self.assertTrue(new_file.exists())
            self.assertEqual(result.deleted_files, 1)
            self.assertEqual(result.deleted_bytes, 3)

    def test_retention_does_not_follow_symlinked_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outside = Path(tmp).parent / f"aaos_retention_outside_{os.getpid()}"
            outside.mkdir(exist_ok=True)
            try:
                protected = outside / "protected.csv"
                protected.write_bytes(b"protected")
                old_mtime = time.time() - (365 * 24 * 60 * 60)
                os.utime(protected, (old_mtime, old_mtime))
                link = root / "linked"
                try:
                    link.symlink_to(outside, target_is_directory=True)
                except (OSError, NotImplementedError):
                    self.skipTest("Symlink creation not available")

                manager = LocalFileRetentionManager(str(root), retention_days=180)
                manager.prune()
                self.assertTrue(protected.exists())
            finally:
                try:
                    protected.unlink()
                except FileNotFoundError:
                    pass
                try:
                    outside.rmdir()
                except OSError:
                    pass


if __name__ == "__main__":
    unittest.main()
