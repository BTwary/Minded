"""M1 end-to-end: the hash the analyst approved on the preview is the hash the API accepts.

These tests deliberately do NOT take a hash from a rejected API response. They take the hash
from ``POST /investigations/interpret`` (what the UI shows the analyst) and submit exactly that
value to ``POST /investigations``, which is the real UI -> API path.
"""
from __future__ import annotations

import tempfile
import uuid

import pandas as pd
import pytest

from packages.analytics_core.src.governance.guided_investigation import (
    apply_human_guidance_to_contract,
    bind_default_aggregation,
    compute_resolved_plan_hash,
    evaluate_guided_investigation,
    ResolverStatus,
)
from packages.analytics_core.src.intelligence.canonical_question_contract import (
    compile_canonical_question_contract,
)

QUESTION = "Rank revenue by region"


@pytest.fixture()
def api_env():
    from fastapi.testclient import TestClient
    from apps.api.src.main import app
    from apps.api.src.core.database import SessionLocal
    from apps.api.src.api.v1.auth import get_current_user
    from apps.api.src.models.entities import User, Project, Dataset, DatasetVersion

    db = SessionLocal()
    user = db.query(User).first()
    if not user:
        user = User(id=str(uuid.uuid4()), email="m1_e2e@minded.ai", full_name="M1 E2E", hashed_password="pw", is_active=True)
        db.add(user)
        db.commit()
        db.refresh(user)
    project = Project(id=str(uuid.uuid4()), name="M1 E2E Project", owner_id=user.id)
    db.add(project)
    db.commit()
    ds = Dataset(id=str(uuid.uuid4()), project_id=project.id, name="e2e_data", row_count=4, column_count=2)
    db.add(ds)
    db.commit()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        f.write("revenue,region\n100.0,North\n200.0,South\n300.0,North\n150.0,South\n")
        path = f.name
    db.add(DatasetVersion(
        id=str(uuid.uuid4()), dataset_id=ds.id, version_number=1, file_path=path,
        row_count=4, column_count=2, content_hash="fp_e2e_123",
    ))
    db.commit()
    app.dependency_overrides[get_current_user] = lambda: user
    try:
        yield TestClient(app), project.id, ds.id
    finally:
        app.dependency_overrides.clear()
        db.close()


def _interpret(client, project_id, ds_id, **guidance):
    res = client.post(
        "/api/v1/investigations/interpret",
        json={"question": QUESTION, "project_id": project_id, "dataset_ids": [ds_id], **guidance},
    )
    assert res.status_code == 200, res.text
    return res.json()


def _create(client, project_id, ds_id, plan_hash, **guidance):
    return client.post(
        "/api/v1/investigations",
        json={
            "question": QUESTION, "project_id": project_id, "dataset_ids": [ds_id],
            "plan_hash": plan_hash, **guidance,
        },
    )


def test_preview_hash_is_accepted_by_creation_without_guidance(api_env):
    client, project_id, ds_id = api_env
    preview = _interpret(client, project_id, ds_id)
    assert preview["plan_hash"]
    res = _create(client, project_id, ds_id, preview["plan_hash"])
    assert res.status_code == 200, res.text
    assert res.json()["status"] == "QUEUED"


def test_preview_hash_with_guidance_is_accepted_when_guidance_is_previewed(api_env):
    client, project_id, ds_id = api_env
    guidance = {
        "selected_metric": "revenue",
        "selected_group_dimension": "region",
        "selected_aggregation": "SUM",
    }
    preview = _interpret(client, project_id, ds_id, **guidance)
    assert preview["guidance_resolution"]["status"] == ResolverStatus.RESOLVED.value
    res = _create(client, project_id, ds_id, preview["plan_hash"], **guidance)
    assert res.status_code == 200, res.text


def test_unguided_preview_hash_is_rejected_when_guidance_changes_the_plan(api_env):
    client, project_id, ds_id = api_env
    preview = _interpret(client, project_id, ds_id)
    res = _create(
        client, project_id, ds_id, preview["plan_hash"],
        selected_metric="revenue", selected_group_dimension="region", selected_aggregation="MEAN",
    )
    assert res.status_code == 409
    assert res.json()["detail"]["code"] == "STALE_PLAN_APPROVAL"


def test_preview_binds_default_sum_into_contract_and_displayed_plan():
    df = pd.DataFrame({"revenue": [1.0, 2.0], "region": ["N", "S"]})
    preview = evaluate_guided_investigation(QUESTION, df=df)
    contract = preview.canonical_contract
    assert ('SUM("revenue")' in preview.planned_calculation) or ("SUM(revenue)" in preview.planned_calculation)
    assert contract.requested_aggregation == "SUM"
    assert contract.resolution_evidence.get("defaulted_aggregation") == "SUM"
    assert any("default" in a for a in preview.assumptions)


def test_resolver_without_guidance_returns_same_hashed_plan_as_preview():
    df = pd.DataFrame({"revenue": [1.0, 2.0], "region": ["N", "S"]})
    preview = evaluate_guided_investigation(QUESTION, df=df)
    outcome = apply_human_guidance_to_contract(QUESTION, {"data": df}, {})
    assert outcome.status == ResolverStatus.RESOLVED
    assert compute_resolved_plan_hash(outcome.contract, "fp") == compute_resolved_plan_hash(
        preview.canonical_contract, "fp"
    )


def test_resolved_plan_hash_ignores_provenance_but_not_material_fields():
    df = pd.DataFrame({"revenue": [1.0, 2.0], "region": ["N", "S"]})
    from dataclasses import replace

    base = bind_default_aggregation(compile_canonical_question_contract(QUESTION, df))
    noisy = replace(base, resolution_evidence={**base.resolution_evidence, "submitted_guidance": {"x": 1}})
    assert compute_resolved_plan_hash(base, "fp") == compute_resolved_plan_hash(noisy, "fp")
    assert compute_resolved_plan_hash(base, "fp") != compute_resolved_plan_hash(base, "other_fp")
    assert compute_resolved_plan_hash(base, "fp") != compute_resolved_plan_hash(
        replace(base, requested_aggregation="MEAN"), "fp"
    )
