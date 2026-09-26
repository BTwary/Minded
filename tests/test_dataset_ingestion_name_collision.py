"""Regression tests for ingestion-time dataset-name collision prevention.

This closes the same P0 gap that ``test_query_sql_dataset_scope.py``'s
collision test detects at query time, but one layer earlier: instead of
only failing a query closed after a collision has already been created,
``DatasetService.ingest_dataset_file`` now refuses to let two datasets in
the same project ever end up with names that sanitize to the same SQL
table identity in the first place.
"""
from apps.api.src.core.database import SessionLocal
from apps.api.src.models.entities import Organization, Project, User
from apps.api.src.services.dataset_service import DatasetService
from packages.analytics_core.src.sql.engine import sanitize_sql_identifier


def _make_project(db):
    org = Organization(id="org-name-collision-test", name="Name Collision Test Org", slug="name-collision-test-org")
    db.merge(org)
    user = User(
        id="user-name-collision-test",
        email="name-collision-test@example.com",
        hashed_password="not-a-real-hash",
        full_name="Name Collision Test User",
        role="analyst",
        organization_id=org.id,
    )
    db.merge(user)
    project = Project(
        id="proj-name-collision-test",
        org_id=org.id,
        owner_id=user.id,
        name="Name Collision Test Project",
    )
    db.merge(project)
    db.commit()
    return project


def test_ingestion_disambiguates_colliding_names_within_a_project():
    db = SessionLocal()
    try:
        project = _make_project(db)
        service = DatasetService(db)

        ds_a = service.ingest_dataset_file(project.id, "Sales Data.csv", b"amount\n1\n")
        db.commit()
        ds_b = service.ingest_dataset_file(project.id, "Sales-Data.csv", b"amount\n2\n")
        db.commit()

        assert ds_a.name != ds_b.name
        assert sanitize_sql_identifier(ds_a.name) != sanitize_sql_identifier(ds_b.name)
        # The second dataset should be recognizably derived from the same
        # requested name, not an unrelated name.
        assert "Sales" in ds_b.name
    finally:
        db.close()


def test_ingestion_third_colliding_name_still_disambiguates():
    db = SessionLocal()
    try:
        project = _make_project(db)
        service = DatasetService(db)

        names = []
        for fname in ("Orders.csv", "Orders!!.csv", "Orders###.csv"):
            ds = service.ingest_dataset_file(project.id, fname, b"amount\n1\n")
            db.commit()
            names.append(ds.name)

        sanitized = [sanitize_sql_identifier(n) for n in names]
        assert len(set(sanitized)) == len(sanitized), f"collision among {names} -> {sanitized}"
    finally:
        db.close()


def test_different_projects_may_reuse_the_same_dataset_name():
    """Disambiguation is project-scoped -- it should not rename datasets
    unnecessarily just because another project happens to use the same name."""
    db = SessionLocal()
    try:
        org = Organization(id="org-name-collision-test-2", name="Name Collision Test Org 2", slug="name-collision-test-org-2")
        db.merge(org)
        user = User(
            id="user-name-collision-test-2",
            email="name-collision-test-2@example.com",
            hashed_password="not-a-real-hash",
            full_name="Name Collision Test User 2",
            role="analyst",
            organization_id=org.id,
        )
        db.merge(user)
        project_a = Project(id="proj-name-collision-test-a", org_id=org.id, owner_id=user.id, name="Project A")
        project_b = Project(id="proj-name-collision-test-b", org_id=org.id, owner_id=user.id, name="Project B")
        db.merge(project_a)
        db.merge(project_b)
        db.commit()

        service = DatasetService(db)
        ds_a = service.ingest_dataset_file(project_a.id, "Orders.csv", b"amount\n1\n")
        db.commit()
        ds_b = service.ingest_dataset_file(project_b.id, "Orders.csv", b"amount\n2\n")
        db.commit()

        assert ds_a.name == "Orders" == ds_b.name
    finally:
        db.close()
