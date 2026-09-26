"""Backfill content-addressed hashes for existing DatasetVersion records.

Run from the repository root in the application's production environment. Missing
or unreadable historical files are reported and left NULL rather than guessed.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps.api.src.core.database import SessionLocal
from apps.api.src.models.entities import DatasetVersion
from apps.api.src.services.dataset_service import DatasetService
from packages.analytics_core.src.data.content_identity import compute_content_hash


def main() -> int:
    db = SessionLocal()
    changed = 0
    failed = 0
    try:
        for version in db.query(DatasetVersion).order_by(DatasetVersion.id).all():
            if version.content_hash:
                continue
            try:
                df = DatasetService(db).get_dataset_dataframe(version.dataset_id, version.version_number)
                version.content_hash = compute_content_hash(df)
                changed += 1
            except Exception as exc:
                failed += 1
                print(f"UNRESOLVED {version.id}: {exc}")
        db.commit()
    finally:
        db.close()
    print(f"BACKFILL_COMPLETE changed={changed} unresolved={failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
