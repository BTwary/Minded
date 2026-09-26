"""Create an encrypted portable local AA-OS backup."""
from __future__ import annotations
import argparse
import getpass
import os
from pathlib import Path
from packages.analytics_core.src.backup.portable_backup import PortableBackupBuilder
from apps.api.src.core.config import settings
from apps.api.src.core.database import engine, init_db
from apps.api.src.models.entities import Base as OrmBase


def main() -> None:
    init_db()
    ap = argparse.ArgumentParser()
    ap.add_argument("output", nargs="?", help="Backup file path")
    ap.add_argument("--no-source-files", action="store_true")
    args = ap.parse_args()
    output = Path(args.output or (Path(settings.DATA_STORAGE_DIR).parent / "backups" / "minded-backup.aaosbackup"))
    output.parent.mkdir(parents=True, exist_ok=True)
    passphrase = getpass.getpass("Backup passphrase: ")
    if not passphrase:
        raise SystemExit("A non-empty backup passphrase is required.")
    data = PortableBackupBuilder(app_version=os.getenv("AAOS_VERSION", "dev"), schema_revision=os.getenv("AAOS_SCHEMA_REVISION", "current")).build_bytes(
        engine=engine, metadata=OrmBase.metadata, storage_root=settings.DATA_STORAGE_DIR,
        passphrase=passphrase, include_source_files=not args.no_source_files,
    )
    output.write_bytes(data)
    print(f"Encrypted AA-OS backup written to {output}")


if __name__ == "__main__":
    main()
