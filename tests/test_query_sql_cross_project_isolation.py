"""Regression proof for the actual cross-project authorization boundary on
``/query/sql``.

Earlier audits of this codebase flagged that ``GovernanceEngine.
inject_rbac_filters`` (a tenant_id/region-keyed row-level-security rewrite)
is never called from the real query execution path, and read that as "RLS
is not enforced." Investigating it turned up a design mismatch, not just a
missing wire: nothing in ``User``/``Project``/``Dataset`` has a
``tenant_id`` or ``region`` column, so that module has no data to filter
on for this application's actual multi-tenancy model.

The authorization boundary this codebase actually relies on is different:
a project belongs to one owner, and a SQL query can only ever see datasets
that (a) belong to the requesting project and (b) were explicitly loaded
and registered into a fresh, request-scoped ``DuckDBSQLEngine`` instance,
whose ``validate_sql`` rejects any relation name that was not registered.
There is no shared engine, no shared table namespace, and no code path
that registers a dataset from a project other than the one named in the
request.

Rather than wire in a filter that has no column to filter on, this test
proves that boundary actually holds end-to-end: a user cannot read another
project's data through this endpoint, whether by naming a foreign
project directly, by naming a foreign dataset ID, or by guessing a table
name that happens to match another project's dataset.
"""
import pytest
from fastapi import HTTPException

from apps.api.src.api.v1.query import SQLQueryRequest, execute_safe_sql
from apps.api.src.core.database import SessionLocal
from apps.api.src.models.entities import Organization, Project, User
from apps.api.src.services.dataset_service import DatasetService


def _make_org_and_users(db):
    org = Organization(id="org-cross-project-test", name="Cross Project Test Org", slug="cross-project-test-org")
    db.merge(org)
    owner_a = User(
        id="user-cross-project-test-a",
        email="cross-project-test-a@example.com",
        hashed_password="not-a-real-hash",
        full_name="Owner A",
        role="analyst",
        organization_id=org.id,
    )
    owner_b = User(
        id="user-cross-project-test-b",
        email="cross-project-test-b@example.com",
        hashed_password="not-a-real-hash",
        full_name="Owner B",
        role="analyst",
        organization_id=org.id,
    )
    db.merge(owner_a)
    db.merge(owner_b)
    db.commit()
    return owner_a, owner_b


@pytest.fixture
def cross_project_context():
    db = SessionLocal()
    try:
        owner_a, owner_b = _make_org_and_users(db)
        project_a = Project(id="proj-cross-test-a", org_id=owner_a.organization_id, owner_id=owner_a.id, name="Project A")
        project_b = Project(id="proj-cross-test-b", org_id=owner_b.organization_id, owner_id=owner_b.id, name="Project B")
        db.merge(project_a)
        db.merge(project_b)
        db.commit()

        service = DatasetService(db)
        ds_b = service.ingest_dataset_file(project_b.id, "secret_customers.csv", b"revenue\n999\n")
        db.commit()

        yield db, owner_a, owner_b, project_a, project_b, ds_b
    finally:
        db.close()


def test_user_cannot_query_a_project_they_do_not_own(cross_project_context):
    """Naming another user's project directly must be rejected before any data loads."""
    db, owner_a, owner_b, project_a, project_b, ds_b = cross_project_context

    payload = SQLQueryRequest(
        project_id=project_b.id,
        sql="SELECT * FROM secret_customers",
    )

    with pytest.raises(HTTPException) as exc_info:
        execute_safe_sql(payload, current_user=owner_a, db=db)

    assert exc_info.value.status_code == 403


def test_foreign_dataset_id_inside_own_project_scope_is_not_loaded(cross_project_context):
    """Naming a foreign project's dataset_id while scoped to your own project
    must not load that dataset -- it should be treated as missing, not silently
    ignored or (worse) loaded anyway."""
    db, owner_a, owner_b, project_a, project_b, ds_b = cross_project_context

    payload = SQLQueryRequest(
        project_id=project_a.id,
        sql="SELECT * FROM secret_customers",
        dataset_ids=[ds_b.id],
    )

    with pytest.raises(HTTPException) as exc_info:
        execute_safe_sql(payload, current_user=owner_a, db=db)

    # Caught by the requested-vs-resolved dataset scope check: the foreign
    # dataset_id was filtered out by `Dataset.project_id == payload.project_id`
    # before it ever reached table registration.
    assert exc_info.value.status_code == 404
    assert ds_b.id in exc_info.value.detail


def test_guessing_a_foreign_table_name_is_rejected_by_the_engine(cross_project_context):
    """Even with no dataset_ids named, a query cannot reference a table that
    was never registered for this request -- there is no shared table
    namespace across projects for a name to be guessed into."""
    db, owner_a, owner_b, project_a, project_b, ds_b = cross_project_context

    service = DatasetService(db)
    service.ingest_dataset_file(project_a.id, "orders.csv", b"amount\n1\n")
    db.commit()

    payload = SQLQueryRequest(
        project_id=project_a.id,
        sql="SELECT * FROM secret_customers",  # project B's table name, guessed
    )

    res = execute_safe_sql(payload, current_user=owner_a, db=db)
    assert res["status"] == "security_violation"
    assert "not registered" in res["error"].lower()
