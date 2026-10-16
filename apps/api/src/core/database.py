"""Database Connection, Session Management, and Production Connection Pooling."""
import os
from pathlib import Path
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import SingletonThreadPool, StaticPool
from alembic.config import Config as AlembicConfig
from alembic.script import ScriptDirectory
from apps.api.src.core.config import settings
from apps.api.src.models.entities import Base

# Normalize PostgreSQL URL for Supabase / Cloud Postgres
db_url = settings.DATABASE_URL
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

is_sqlite = db_url.startswith("sqlite")
connect_args = {"check_same_thread": False} if is_sqlite else {}

if is_sqlite:
    # SQLite concurrency hardening. Two independent problems, both real
    # under FastAPI + pytest's request/session-per-call pattern (each call
    # opens its own connection on a shared engine):
    #
    # 1. Without a shared single connection, `:memory:` databases lose
    #    their schema the moment the first connection closes (a fresh
    #    in-memory DB is created per pooled connection), and even
    #    file-based SQLite can serialize writers unpredictably across
    #    pooled connections. StaticPool pins the engine to exactly one
    #    underlying DBAPI connection, reused for every session, so the
    #    schema and any in-flight write are always visible to the next
    #    caller instead of racing a second physical connection.
    # 2. SQLite's default locking raises "database is locked" the moment
    #    two sessions overlap a write, even briefly -- there is no
    #    queuing/waiting by default. WAL journal mode lets readers and a
    #    writer proceed concurrently instead of blocking each other
    #    outright, and `busy_timeout` makes a genuine writer/writer
    #    collision retry for a bounded window instead of failing
    #    immediately, which is what real concurrent controller/test
    #    access needs.
    # CONCURRENCY FIX (found with the real 4-CSV workspace): StaticPool shares ONE DBAPI connection
    # between every session AND every thread.  Any session's close() issues a rollback on that shared
    # connection, so the UI polling investigation status from the API thread silently rolled back the
    # worker thread's uncommitted rows ("Contract <id> not found" at finalize_contract).
    #
    # For a FILE database we therefore use one connection PER THREAD: sessions on the same thread still
    # share a connection (exactly the semantics the controller/tests were written against), but a
    # different thread's close()/rollback() can no longer touch this thread's open transaction.
    # WAL + busy_timeout (below) let the per-thread connections read while another writes.
    # In-memory databases must keep StaticPool (a second connection would be a different, empty DB).
    _in_memory = db_url in {"sqlite://", "sqlite:///:memory:"} or ":memory:" in db_url or "mode=memory" in db_url
    if _in_memory:
        engine = create_engine(
            db_url,
            connect_args=connect_args,
            poolclass=StaticPool,
            echo=False,
        )
    else:
        engine = create_engine(
            db_url,
            connect_args=connect_args,
            poolclass=SingletonThreadPool,
            pool_size=256,  # max distinct threads holding a live connection; must exceed worker+API threads
            echo=False,
        )

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, connection_record):  # noqa: ARG001
        cursor = dbapi_connection.cursor()
        try:
            # WAL is a no-op (falls back to "memory") on ":memory:" databases;
            # harmless either way, and correct for the file-based case.
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=30000")
        finally:
            cursor.close()
else:
    # Production-ready PostgreSQL connection pooling for Supabase / Cloud Postgres
    engine = create_engine(
        db_url,
        connect_args=connect_args,
        pool_pre_ping=True,      # Automatically recycles dead/closed connections
        pool_size=10,            # Max persistent connections in pool
        max_overflow=20,         # Max burst connections beyond pool_size
        pool_recycle=300,        # Recycle connections every 5 minutes
        echo=False,
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize the local schema without pretending ORM creation is a migration system.

    Auto-create remains available only for local development. Production must run
    Alembic explicitly so migration state cannot diverge from ORM metadata, and is
    verified fail-closed via `_assert_production_schema_current` below.
    """
    auto_create = os.getenv("AAOS_AUTO_CREATE_DB_SCHEMA", "true").lower() in {"1", "true", "yes"}
    if settings.ENVIRONMENT.lower() == "production":
        _assert_production_schema_current()
        return
    if auto_create:
        Base.metadata.create_all(bind=engine)
        if engine.dialect.name == "sqlite":
            _auto_sync_sqlite_columns(engine)


def _auto_sync_sqlite_columns(bind) -> None:
    """Ensure existing SQLite tables receive newly added ORM columns in local dev/desktop."""
    try:
        inspector = inspect(bind)
        existing_tables = set(inspector.get_table_names())
        with bind.connect() as conn:
            for table_name, table in Base.metadata.tables.items():
                if table_name in existing_tables:
                    existing_cols = {col["name"] for col in inspector.get_columns(table_name)}
                    for col in table.columns:
                        if col.name not in existing_cols:
                            col_type = col.type.compile(bind.dialect)
                            conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {col.name} {col_type}"))
            conn.commit()
    except Exception:
        pass


def _assert_production_schema_current() -> None:
    """Fail closed when production DB has no recorded/current Alembic revision."""
    inspector = inspect(engine)
    if "alembic_version" not in inspector.get_table_names():
        raise RuntimeError("Production database is not Alembic-managed: alembic_version table is missing.")
    with engine.connect() as conn:
        revision = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
    # Resolve the authoritative migration head from Alembic itself instead of
    # hard-coding a revision that becomes stale as soon as a new migration lands.
    repo_root = Path(__file__).resolve().parents[4]
    alembic_cfg = AlembicConfig(str(repo_root / "alembic.ini"))
    alembic_cfg.set_main_option("script_location", str(repo_root / "migrations"))
    expected_heads = tuple(ScriptDirectory.from_config(alembic_cfg).get_heads())
    if len(expected_heads) != 1:
        raise RuntimeError(f"Production migration chain must have exactly one head; found {expected_heads!r}.")
    expected = expected_heads[0]
    if revision != expected:
        raise RuntimeError(f"Production database schema revision {revision!r} does not match Alembic head {expected!r}. Run 'alembic upgrade head'.")


def get_db():
    """Dependency that provides a database session per request."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
