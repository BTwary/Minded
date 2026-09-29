"""Local dataset retention policy for AA-OS.

The retention layer governs *stored dataset files*, not analytical history.
SQLite/metadata records are intentionally outside this file-based retention policy.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class RetentionResult:
    scanned_files: int
    deleted_files: int
    skipped_files: int
    deleted_bytes: int
    errors: List[str]


class LocalFileRetentionManager:
    """Delete local stored dataset/artifact files older than the configured TTL.

    Retention is based on file modification time, which is stable across subsequent
    reads/analyses and therefore represents the stored-file lifetime rather than
    last access time. Symlinks are never followed.
    """

    DEFAULT_RETENTION_DAYS = 180

    def __init__(self, base_dir: str, retention_days: int = DEFAULT_RETENTION_DAYS):
        if retention_days <= 0:
            raise ValueError("retention_days must be > 0")
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.retention_days = retention_days
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @property
    def cutoff_timestamp(self) -> float:
        return time.time() - (self.retention_days * 24 * 60 * 60)

    def iter_files(self) -> Iterable[Path]:
        """Yield regular files under base_dir without following symlinks."""
        if not self.base_dir.exists():
            return
        for root, dirs, files in os.walk(self.base_dir, topdown=True, followlinks=False):
            # Prevent traversal through symlinked directories.
            dirs[:] = [d for d in dirs if not (Path(root) / d).is_symlink()]
            for filename in files:
                path = Path(root) / filename
                if path.is_symlink() or not path.is_file():
                    continue
                yield path

    def prune(self, now: Optional[float] = None) -> RetentionResult:
        """Delete files older than the retention window and tidy empty directories."""
        now = time.time() if now is None else now
        cutoff = now - (self.retention_days * 24 * 60 * 60)
        scanned = deleted = skipped = deleted_bytes = 0
        errors: List[str] = []

        files = list(self.iter_files())
        scanned = len(files)
        for path in files:
            try:
                stat = path.stat()
                if stat.st_mtime >= cutoff:
                    continue
                size = stat.st_size
                path.unlink()
                deleted += 1
                deleted_bytes += size
            except (FileNotFoundError, PermissionError, OSError) as exc:
                skipped += 1
                errors.append(f"{path}: {exc}")

        # Remove empty directories, deepest first, but never remove base_dir itself.
        if self.base_dir.exists():
            for root, dirs, _files in os.walk(self.base_dir, topdown=False, followlinks=False):
                root_path = Path(root)
                for dirname in dirs:
                    candidate = root_path / dirname
                    if candidate.is_symlink():
                        continue
                    try:
                        candidate.rmdir()
                    except OSError:
                        pass

        return RetentionResult(
            scanned_files=scanned,
            deleted_files=deleted,
            skipped_files=skipped,
            deleted_bytes=deleted_bytes,
            errors=errors,
        )
