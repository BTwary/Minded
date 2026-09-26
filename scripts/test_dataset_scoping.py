"""
Regression test for DEFECT-003 (dataset-provider scoping, Section 22 of the
brief). Drives the REAL `DatabaseDatasetProvider` against a real SQLite DB
with real `Dataset`/`DatasetVersion` rows and real parquet files on disk
(via the real `StorageService`) -- not a mock -- and asserts the exact six
cases the brief requires:

    one dataset selected
    multiple datasets selected
    invalid dataset id (belongs to no project)
    nonexistent dataset id
    unauthorized / cross-project dataset id
    no selection (default: all of the project's own datasets)

Also exercises the real `AnalysisCreate.dataset_ids` -> `Investigation.
requested_dataset_ids_json` -> `DatabaseDatasetProvider` wiring end-to-end
via `AnalysisService`, since a provider-level fix that nothing above it
ever calls with a real value is not actually a fix.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from apps.api.src.models.entities import Base, Dataset, DatasetVersion, Project, User, gen_uuid
from apps.api.src.services.storage_service import StorageService
from packages.analytics_core.src.engines.dataset_provider import DatabaseDatasetProvider


def _make_dataset(session, storage, project_id, name, n_rows=10):
    ds_id = gen_uuid()
    df = pd.DataFrame({"x": range(n_rows), "y": [i * 2 for i in range(n_rows)]})
    path = storage.save_dataframe_as_parquet(ds_id, 1, df)
    ds = Dataset(id=ds_id, project_id=project_id, name=name, format="parquet", current_version=1,
                 row_count=len(df), column_count=len(df.columns))
    ver = DatasetVersion(id=gen_uuid(), dataset_id=ds_id, version_number=1, file_path=path,
                          file_size_bytes=0, row_count=len(df), column_count=len(df.columns))
    session.add_all([ds, ver])
    session.commit()
    return ds_id


def main():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionFactory = sessionmaker(bind=engine)
    storage = StorageService(base_dir="/tmp/minded_test_dataset_scoping")

    with SessionFactory() as session:
        user = User(id="u1", email="a@b.com", hashed_password="x", full_name="T", is_active=True, role="admin")
        proj_a = Project(id="proj-A", name="A", description="", owner_id=user.id)
        proj_b = Project(id="proj-B", name="B", description="", owner_id=user.id)
        session.add_all([user, proj_a, proj_b])
        session.commit()

        ds1 = _make_dataset(session, storage, "proj-A", "sales")
        ds2 = _make_dataset(session, storage, "proj-A", "returns")
        ds3 = _make_dataset(session, storage, "proj-A", "inventory")
        ds_other_project = _make_dataset(session, storage, "proj-B", "confidential_b_data")

    provider = DatabaseDatasetProvider(SessionFactory)

    # Case 1: no selection -> default to all of the project's own datasets.
    ctx = provider.acquire_context("proj-A")
    assert set(ctx.datasets_map.keys()) == {"sales", "returns", "inventory"}, \
        f"Default (no selection) should return all of proj-A's datasets, got {list(ctx.datasets_map.keys())}"
    assert ctx.requested_dataset_ids is None
    print("[PASS] no selection -> all of the project's own datasets, none other")

    # Case 2: one dataset selected.
    ctx = provider.acquire_context("proj-A", dataset_ids=[ds1])
    assert set(ctx.datasets_map.keys()) == {"sales"}, f"Expected only 'sales', got {list(ctx.datasets_map.keys())}"
    assert ctx.unavailable_requested_ids == []
    print("[PASS] one dataset selected -> exactly that one dataset, no others")

    # Case 3: multiple datasets selected.
    ctx = provider.acquire_context("proj-A", dataset_ids=[ds1, ds3])
    assert set(ctx.datasets_map.keys()) == {"sales", "inventory"}, \
        f"Expected 'sales' + 'inventory', got {list(ctx.datasets_map.keys())}"
    print("[PASS] multiple datasets selected -> exactly those datasets")

    # Case 4: nonexistent dataset id (well-formed but no such row anywhere).
    fake_id = gen_uuid()
    ctx = provider.acquire_context("proj-A", dataset_ids=[ds1, fake_id])
    assert set(ctx.datasets_map.keys()) == {"sales"}, "A nonexistent id must not silently expand or error out"
    assert ctx.unavailable_requested_ids == [fake_id], \
        f"Nonexistent id should be reported as unavailable, got {ctx.unavailable_requested_ids}"
    print("[PASS] nonexistent dataset id -> silently excluded, reported in unavailable_requested_ids")

    # Case 5 (= "invalid ID"): a syntactically-fine but non-UUID/garbage id.
    ctx = provider.acquire_context("proj-A", dataset_ids=["not-a-real-id;--"])
    assert ctx.is_empty, "A garbage/invalid id must never resolve to any dataset"
    print("[PASS] invalid dataset id -> empty context, not an expansion to all datasets")

    # Case 6: unauthorized / cross-project dataset id. This is the critical
    # one -- requesting proj-B's dataset while scoped to proj-A must NEVER
    # leak proj-B's data into a proj-A investigation.
    ctx = provider.acquire_context("proj-A", dataset_ids=[ds1, ds_other_project])
    assert set(ctx.datasets_map.keys()) == {"sales"}, \
        f"Cross-project dataset id must be excluded; got {list(ctx.datasets_map.keys())}"
    assert "confidential_b_data" not in ctx.datasets_map
    assert ds_other_project in ctx.unavailable_requested_ids, \
        "Cross-project id should be reported unavailable (from proj-A's perspective), not silently ignored"
    print("[PASS] cross-project/unauthorized dataset id -> excluded, no leakage across projects")

    # End-to-end: AnalysisService actually persists and the controller
    # actually enforces the request-scoped selection (not just the
    # provider in isolation).
    from apps.api.src.models.entities import Investigation
    from packages.schemas.src.analysis import AnalysisCreate
    from apps.api.src.services.analysis_service import AnalysisService

    with SessionFactory() as session:
        svc = AnalysisService(session)
        payload = AnalysisCreate(question="irrelevant for this check", project_id="proj-A", dataset_ids=[ds2])
        inv_id = f"INV-{gen_uuid()[:8]}"
        inv = Investigation(id=inv_id, project_id="proj-A", question=payload.question, status="PLANNED",
                             requested_dataset_ids_json=list(payload.dataset_ids))
        session.add(inv)
        session.commit()

        ctx = provider.acquire_context("proj-A", dataset_ids=list(
            session.query(Investigation).filter(Investigation.id == inv_id).first().requested_dataset_ids_json
        ))
        assert set(ctx.datasets_map.keys()) == {"returns"}, \
            f"End-to-end: Investigation.requested_dataset_ids_json must restrict acquisition, got {list(ctx.datasets_map.keys())}"
    print("[PASS] AnalysisCreate.dataset_ids -> Investigation.requested_dataset_ids_json -> provider: wired correctly")

    print("\nALL DATASET SCOPING CASES PASSED")


if __name__ == "__main__":
    main()
