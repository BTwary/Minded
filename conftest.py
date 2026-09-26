"""Root pytest bootstrap: an isolated, freshly-initialized SQLite database
for the whole test session.

Without this, `apps.api.src.core.database` binds its engine (at import
time, driven by `apps.api.src.core.config.settings.DATABASE_URL`) to
whatever `DATABASE_URL` happens to be set to -- by default a relative
on-disk file (`./autonomous_analyst.db`) that this repo does not ship
pre-created or pre-seeded. Any test module that imports `SessionLocal`
from that module and opens a session then hits
`sqlite3.OperationalError: no such table: investigations` (or similar)
the moment it queries, because nothing in the test run ever created the
schema on that file -- exactly the failure mode this conftest exists to
close.

This is placed at the repository root (rather than under tests/) so
pytest loads and executes it, per pytest's conftest.py collection order,
before importing *any* test module anywhere in the tree -- test modules
import `apps.api.src.core.database` at their own module level, so the
environment must be patched before that first import happens anywhere
in the process, not inside a fixture that only runs once a test starts.

Behavior:
  - If the environment already has an explicit `DATABASE_URL` set (e.g. a
    developer intentionally pointing tests at a real dev DB), that choice
    is respected and left untouched (`os.environ.setdefault`).
  - Otherwise, tests get their own fresh temporary SQLite file for the
    session so runs are isolated from each other and from any stale
    `./autonomous_analyst.db` left over in the working directory.
  - The schema is created via the same `init_db()` used in production/dev
    (not a hand-rolled schema), and the standard seed data
    (`scripts/seed_database.py`) is loaded once per session so tests that
    exercise real end-to-end analysis (e.g. `AnalysisService.execute_analysis`
    against `project_id="proj-default"`) have real datasets to run against,
    matching what `test_verification_status_contract.py` already documents
    as a precondition ("Requires a seeded database").
"""
import os
import sys
import tempfile

import pytest

_REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_tmp_dir = tempfile.mkdtemp(prefix="aaos_test_db_")
_tmp_db_path = os.path.join(_tmp_dir, "test_autonomous_analyst.db")

os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp_db_path}")
# The local-first migration (packages/analytics_core/src/platform_local_first.py)
# defaults dataset storage to a per-OS user-data directory rather than the
# repo tree, which is correct for real usage -- but for tests it means a
# full run writes into the *actual* machine's ~/.local/share/Minded/AAOS (or
# platform equivalent) unless redirected, same class of isolation problem
# DATABASE_URL above already solves for the metadata DB. AAOS_DATA_ROOT is
# platform_local_first's own explicit override, so route it to the same
# disposable temp directory.
os.environ.setdefault("AAOS_DATA_ROOT", os.path.join(_tmp_dir, "aaos_data_root"))
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("AAOS_AUTO_CREATE_DB_SCHEMA", "true")
# Deterministic, non-network-dependent test runs unless a developer opts in.
os.environ.setdefault("AI_ENABLED", "false")
os.environ.setdefault("AI_PROVIDER", "none")
os.environ.setdefault("AAOS_BUSINESS_TIMEZONE", "UTC")


@pytest.fixture(scope="session", autouse=True)
def _bootstrap_test_database():
    """Create the schema and load seed data once per test session.

    autouse so every test collected under this root gets a working,
    seeded database without needing to opt in -- tests that construct
    their own isolated in-memory engine directly (as several already do)
    are unaffected; this only governs the shared engine bound inside
    `apps.api.src.core.database`.
    """
    from apps.api.src.core.database import init_db

    init_db()

    try:
        from scripts.seed_database import seed_db

        seed_db(generate=True)
    except Exception as exc:  # noqa: BLE001
        # Seeding is best-effort here: schema creation (above) is the part
        # every test actually depends on for table existence. A seed
        # script failure (e.g. a missing optional seed CSV) should not
        # block the whole session -- tests that need seeded rows and don't
        # get them will fail on their own with a clear assertion instead
        # of a mysterious collection-time crash.
        print(f"[conftest] seed_database.seed_db() failed, continuing without seed data: {exc}")

    yield

    # Runtime-generated synthetic fixtures are disposable test artifacts; never
    # leave them in the source tree after the session ends.
    import shutil
    seed_dir = os.path.join(_REPO_ROOT, "data", "seed")
    shutil.rmtree(seed_dir, ignore_errors=True)
