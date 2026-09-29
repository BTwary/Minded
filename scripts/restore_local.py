"""Restore an encrypted AA-OS local backup."""
from __future__ import annotations
import argparse
import getpass
from pathlib import Path
from packages.analytics_core.src.backup.portable_backup import PortableBackupRestorer
from apps.api.src.core.config import settings
from apps.api.src.core.database import engine, init_db
from apps.api.src.models.entities import Base as OrmBase


def main() -> None:
    init_db()
    ap = argparse.ArgumentParser()
    ap.add_argument("backup", type=Path)
    ap.add_argument("--replace-existing", action="store_true")
    args = ap.parse_args()
    if not args.backup.exists():
        raise SystemExit(f"Backup not found: {args.backup}")
    passphrase = getpass.getpass("Backup passphrase: ")
    result = PortableBackupRestorer().restore_bytes(
        backup_bytes=args.backup.read_bytes(), passphrase=passphrase,
        engine=engine, metadata=OrmBase.metadata,
        storage_root=settings.DATA_STORAGE_DIR,
        replace_existing=args.replace_existing,
    )
    print(result)


if __name__ == "__main__":
    main()
