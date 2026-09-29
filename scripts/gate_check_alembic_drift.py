"""Run Alembic against a fresh temporary SQLite database and detect schema drift."""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.util.exc import CommandError


def _config(root: Path, db_path: Path) -> Config:
    cfg = Config(str(root / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path.as_posix()}")
    cfg.set_main_option("script_location", str(root / "migrations"))
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--upgrade-only", action="store_true")
    mode.add_argument("--check-drift", action="store_true")
    args = parser.parse_args(argv)
    check_drift = args.check_drift or not args.upgrade_only

    root = Path(__file__).resolve().parents[1]
    tmpdir = Path(tempfile.mkdtemp(prefix="aaos_gate_"))
    db_path = tmpdir / "gate.db"
    try:
        cfg = _config(root, db_path)
        try:
            command.upgrade(cfg, "head")
        except Exception as exc:  # Alembic wraps many DB-specific errors.
            print("ALEMBIC_SCHEMA_AUDIT: FAIL - upgrade head failed")
            print(f"  {type(exc).__name__}: {exc}")
            return 1

        if args.upgrade_only:
            print("ALEMBIC_SCHEMA_AUDIT: PASS - fresh database reached head")
            return 0

        if check_drift:
            try:
                command.check(cfg)
            except CommandError as exc:
                print("ALEMBIC_SCHEMA_AUDIT: FAIL - ORM/migration drift detected")
                print(f"  {exc}")
                return 1
            except Exception as exc:
                print("ALEMBIC_SCHEMA_AUDIT: FAIL - drift check errored")
                print(f"  {type(exc).__name__}: {exc}")
                return 1

        print("ALEMBIC_SCHEMA_AUDIT: PASS - head reached and no drift detected")
        return 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
