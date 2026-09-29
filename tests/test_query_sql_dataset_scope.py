"""Regression tests for the /query/sql dataset-scope authorization fixes.

Covers two P0 findings from the C4.2.4 audit that were still open in this
build:

1. A requested dataset scope (``dataset_ids``) that partially resolves used
   to fall back to a warning-only partial result instead of failing the
   request -- the executed query could silently answer a smaller question
   than the one actually requested.
2. Two datasets in the same project sharing a sanitized SQL table name
   (e.g. "Sales Data" and "Sales-Data" both -> "Sales_Data") used to
   silently collide in the engine's table registry, so the query would run
   against whichever dataset registered last with no error raised.

Both are exercised through ``execute_safe_sql`` directly (rather than over
HTTP) against the real ingestion pipeline and the real test database set up
by the root conftest, so this proves the actual production code path, not
a mock of it.
"""
import io

import pytest
from fastapi import HTTPException

from apps.api.src.api.v1.query import SQLQueryRequest, execute_safe_sql
from apps.api.src.core.database import SessionLocal
from apps.api.src.models.entities import Organization, Project, User
from apps.api.src.services.dataset_service import DatasetService


def _make_user_and_project(db):
    org = Organization(id="org-query-scope-test", name="Query Scope Test Org", slug="query-scope-test-org")
    db.merge(org)
    user = User(
        id="user-query-scope-test",
        email="query-scope-test@example.com",
        hashed_password="not-a-real-hash",
        full_name="Query Scope Test User",
        role="analyst",
        organization_id=org.id,
    )
    db.merge(user)
    project = Project(
        id="proj-query-scope-test",
        org_id=org.id,
        owner_id=user.id,
        name="Query Scope Test Project",
    )
    db.merge(project)
    db.commit()
    return user, project


def _ingest_csv(service: DatasetService, project_id: str, filename: str, csv_text: str):
    return service.ingest_dataset_file(
        project_id=project_id,
        filename=filename,
        file_bytes=csv_text.encode("utf-8"),
    )


@pytest.fixture
def scope_test_context():
    db = SessionLocal()
    try:
        user, project = _make_user_and_project(db)
        yield db, user, project
    finally:
        db.close()


def test_missing_requested_dataset_id_fails_closed(scope_test_context):
    """Requesting a dataset_id that doesn't resolve must abort, not warn-and-continue."""
    db, user, project = scope_test_context
    service = DatasetService(db)
    ds = _ingest_csv(service, project.id, "orders.csv", "amount\n10\n20\n")
    db.commit()

    payload = SQLQueryRequest(
        project_id=project.id,
        sql="SELECT COUNT(*) AS n FROM orders",
        dataset_ids=[ds.id, "dataset-id-that-does-not-exist"],
    )

    with pytest.raises(HTTPException) as exc_info:
        execute_safe_sql(payload, current_user=user, db=db)

    assert exc_info.value.status_code == 404
    assert "dataset-id-that-does-not-exist" in exc_info.value.detail


def test_valid_full_scope_still_executes(scope_test_context):
    """Sanity check: a fully-resolvable scope is unaffected by the fix."""
    db, user, project = scope_test_context
    service = DatasetService(db)
    ds = _ingest_csv(service, project.id, "orders2.csv", "amount\n10\n20\n30\n")
    db.commit()

    payload = SQLQueryRequest(
        project_id=project.id,
        sql="SELECT COUNT(*) AS n FROM orders2",
        dataset_ids=[ds.id],
    )

    res = execute_safe_sql(payload, current_user=user, db=db)
    assert res["status"] == "success"
    assert res["data"][0]["n"] == 3


def test_colliding_dataset_names_fail_closed_instead_of_silently_overwriting(scope_test_context):
    """Two datasets that sanitize to the same table identity must abort the query.

    Ingestion now disambiguates colliding names at creation time (see
    ``tests/test_dataset_ingestion_name_collision.py``), so this can no
    longer happen for datasets created going forward. This test simulates
    data that predates that fix -- e.g. rows inserted directly, a bulk
    import, or a migration -- to prove the query-time guard still catches
    it as defense-in-depth: before the fix, ``engine.register_dataframe``
    registered datasets by display name with no collision check, so the
    second dataset registered would silently replace the first in the
    DuckDB table registry and the query would run against the wrong data
    with no error or warning.
    """
    db, user, project = scope_test_context
    service = DatasetService(db)
    ds_a = _ingest_csv(service, project.id, "Sales Data.csv", "amount\n1\n")
    db.commit()

    # Simulate a pre-existing collision that ingestion-time disambiguation
    # did not create (e.g. legacy data, a direct DB insert, a bulk import
    # path that bypasses ingest_dataset_file) by inserting the second
    # colliding dataset directly rather than through the service.
    from apps.api.src.models.entities import Dataset, DatasetVersion
    import uuid

    ds_b_id = str(uuid.uuid4())
    db.add(Dataset(
        id=ds_b_id,
        project_id=project.id,
        name="Sales-Data",
        format="csv",
        current_version=1,
        row_count=3,
        column_count=1,
    ))
    db.commit()

    payload = SQLQueryRequest(
        project_id=project.id,
        sql="SELECT COUNT(*) AS n FROM Sales_Data",
        dataset_ids=[ds_a.id, ds_b_id],
    )

    with pytest.raises(HTTPException) as exc_info:
        execute_safe_sql(payload, current_user=user, db=db)

    assert exc_info.value.status_code == 409
    assert "collision" in exc_info.value.detail.lower()
