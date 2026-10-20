"""End-to-end reliability: "Compare regions A, B and C by total revenue in 2025."

Drives the real FastAPI app (interpret -> create -> controller/worker execution -> detail ->
review -> export) against a fixture whose expected values are computed with plain Python,
independently of pandas/DuckDB/the verifier.
"""
import csv
import collections
import tempfile
import uuid

import pandas as pd
import pytest

QUESTION = "Compare regions A, B and C by total revenue in 2025."


def _rows():
    rows, n = [], 0
    for region, base in (("A", 100), ("B", 150), ("C", 200), ("D", 999)):
        for year in (2024, 2025):
            for m in range(1, 13):
                n += 1
                rows.append({"order_id": f"O{n:04d}", "region": region, "order_date": f"{year}-{m:02d}-15",
                             "revenue": base + m + (1000 if year == 2024 else 0), "units": m})
    return rows


def _expected(regions, agg="sum"):
    vals = collections.defaultdict(list)
    for r in _rows():
        if r["order_date"].startswith("2025") and r["region"] in regions:
            vals[r["region"]].append(r["revenue"])
    return {k: (sum(v) if agg == "sum" else sum(v) / len(v)) for k, v in vals.items()}


@pytest.fixture()
def frame():
    return pd.DataFrame(_rows()).assign(order_date=lambda d: pd.to_datetime(d["order_date"]))


@pytest.fixture()
def env():
    from fastapi.testclient import TestClient
    from apps.api.src.main import app
    from apps.api.src.core.database import SessionLocal
    from apps.api.src.api.v1.auth import get_current_user
    from apps.api.src.models.entities import User, Project, Dataset, DatasetVersion
    db = SessionLocal()
    user = User(id=str(uuid.uuid4()), email=f"{uuid.uuid4().hex[:6]}@rc.ai", full_name="RC", hashed_password="pw", is_active=True)
    db.add(user); db.commit(); db.refresh(user)
    project = Project(id=str(uuid.uuid4()), name="RC", owner_id=user.id)
    db.add(project); db.commit()
    rows = _rows()
    ds = Dataset(id=str(uuid.uuid4()), project_id=project.id, name="sales", row_count=len(rows), column_count=5)
    db.add(ds); db.commit()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows); path = f.name
    db.add(DatasetVersion(id=str(uuid.uuid4()), dataset_id=ds.id, version_number=1, file_path=path,
                          row_count=len(rows), column_count=5, content_hash=f"fp_{uuid.uuid4().hex[:8]}"))
    db.commit()
    app.dependency_overrides[get_current_user] = lambda: user
    try:
        yield TestClient(app), project.id, ds.id
    finally:
        app.dependency_overrides.clear(); db.close()


def _interpret(env, question=QUESTION, **guidance):
    client, pid, did = env
    res = client.post("/api/v1/investigations/interpret",
                      json={"question": question, "project_id": pid, "dataset_ids": [did], **guidance})
    assert res.status_code == 200, res.text
    return res.json()


def _run(env, body, question=QUESTION, **guidance):
    from apps.api.src.core.database import SessionLocal
    from packages.analytics_core.src.engines.dataset_provider import DatabaseDatasetProvider
    from packages.analytics_core.src.runtime.controller import InvestigationController
    client, pid, did = env
    res = client.post("/api/v1/investigations", json={"question": question, "project_id": pid, "dataset_ids": [did],
                                                      "plan_hash": body["plan_hash"], **guidance})
    assert res.status_code == 200, res.text
    iid = res.json()["investigation_id"]
    InvestigationController(session_factory=SessionLocal,
                            dataset_provider=DatabaseDatasetProvider(session_factory=SessionLocal)
                            ).execute_investigation(iid, worker_id="rc-test")
    detail = client.get(f"/api/v1/investigations/{iid}")
    assert detail.status_code == 200, detail.text
    return iid, detail.json()


def _numbers(answer):
    """Parse 'C: 2,478; B: 1,878; A: 1,278 ...' into {region: value}."""
    head = answer.split(" Executed")[0]
    return {k.strip(): float(v.replace(",", "")) for k, v in (p.split(":") for p in head.split(";"))}


# ---- Checkpoint B: resolution binds the right roles and keeps every selection ---------------
def test_contract_binds_measure_dimension_period_and_entities(frame):
    from packages.analytics_core.src.intelligence.canonical_question_contract import compile_canonical_question_contract
    c = compile_canonical_question_contract(QUESTION, frame)
    assert (c.target_column, c.requested_aggregation, tuple(c.grouping_columns)) == ("revenue", "SUM", ("region",))
    assert tuple(c.comparison_values) == ("A", "B", "C")
    filters = [(f["column"], f["operator"], f["value"]) for f in c.filter_conditions]
    assert ("region", "IN", ["A", "B", "C"]) in filters
    assert ("order_date", ">=", "2025-01-01") in filters and ("order_date", "<", "2026-01-01") in filters


@pytest.mark.parametrize("q", [
    "Compare regions A, B and Z by total revenue in 2025.",   # Z does not exist -> never compare a subset
    "Compare regions A and A by total revenue in 2025.",      # duplicate level
    "Compare revenue A, B and C by total region in 2025.",    # roles inverted
])
def test_unsupported_variants_fail_closed(frame, q):
    from packages.analytics_core.src.intelligence.canonical_question_contract import compile_canonical_question_contract
    c = compile_canonical_question_contract(q, frame)
    assert not (c.target_column == "region" and tuple(c.grouping_columns) == ("revenue",))   # no inverted binding
    assert tuple(c.comparison_values) != ("A", "B")                                         # no silent subset


def test_inverted_roles_are_never_bound_by_generic_x_by_y(frame):
    from packages.analytics_core.src.intelligence.canonical_question_contract import compile_canonical_question_contract
    c = compile_canonical_question_contract("Show region by revenue", frame)
    assert not (c.target_column == "region" and tuple(c.grouping_columns) == ("revenue",))


def test_existing_x_by_y_still_binds_measure_by_dimension(frame):
    from packages.analytics_core.src.intelligence.canonical_question_contract import compile_canonical_question_contract
    c = compile_canonical_question_contract("Show total revenue by region", frame)
    assert (c.target_column, tuple(c.grouping_columns)) == ("revenue", ("region",))


# ---- Checkpoints A-E + review: the whole path through the real application -------------------
def test_full_path_matches_independent_values_and_review_export(env):
    client, _, _ = env
    body = _interpret(env)
    assert body["can_auto_proceed"] is True and body["plan_hash"]
    assert "\"region\" IN ('A', 'B', 'C')" in body["planned_calculation"] and "2025-01-01" in body["planned_calculation"]

    iid, detail = _run(env, body)
    assert detail["status"] == "COMPLETED"
    assert _numbers(detail["direct_answer"]) == _expected("ABC")            # D and 2024 excluded
    assert detail["evidence"] and all(e["validation_status"] == "VERIFIED" for e in detail["evidence"])
    assert detail["evidence_verifications"] and all(v["status"] == "VERIFIED" for v in detail["evidence_verifications"])
    assert detail["execution_record"]["comparison_values"] == ["A", "B", "C"]

    # accept -> verified acceptance recorded -> export available
    rv = client.post(f"/api/v1/investigations/{iid}/review", json={"action": "ACCEPT", "comments": "looks right"})
    assert rv.status_code == 200, rv.text
    assert rv.json()["review"]["review_status"] == "ACCEPTED_AND_VERIFIED"
    for fmt, ctype in (("pdf", "application/pdf"), ("docx", "officedocument")):
        ex = client.get(f"/api/v1/investigations/{iid}/export?format={fmt}")
        assert ex.status_code == 200 and ctype in ex.headers["content-type"] and len(ex.content) > 1000


def test_entity_subset_is_honoured_not_widened(env):
    body = _interpret(env, "Compare regions A and C by total revenue in 2025.")
    assert body["can_auto_proceed"] is True
    _, detail = _run(env, body, question="Compare regions A and C by total revenue in 2025.")
    assert _numbers(detail["direct_answer"]) == _expected("AC") and "B" not in _numbers(detail["direct_answer"])


def test_targeted_correction_changes_only_the_aggregation_and_reruns(env):
    first = _interpret(env)
    corrected = _interpret(env, selected_aggregation="AVG")
    c1, c2 = first["canonical_contract"], corrected["canonical_contract"]
    assert (c2["requested_aggregation"], c1["requested_aggregation"]) == ("MEAN", "SUM")
    for keep in ("target_column", "grouping_columns", "comparison_values", "filter_conditions"):
        assert c1[keep] == c2[keep], keep                                      # nothing else was lost
    assert corrected["plan_hash"] and corrected["plan_hash"] != first["plan_hash"]
    _, detail = _run(env, corrected, selected_aggregation="AVG")
    assert _numbers(detail["direct_answer"]) == pytest.approx(_expected("ABC", "avg"))


@pytest.mark.parametrize("action,extra", [
    ("ACCEPT", {}), ("REQUEST_CORRECTION", {"comments": "use average"}),
    ("REJECT", {"comments": "wrong period"}), ("REDO", {"modified_parameters": {"selected_aggregation": "AVG"}}),
])
def test_every_review_action_is_recorded_instead_of_crashing(env, action, extra):
    client, _, _ = env
    iid, _ = _run(env, _interpret(env))
    r = client.post(f"/api/v1/investigations/{iid}/review", json={"action": action, **extra})
    assert r.status_code == 200, r.text                                         # was a 500 for every action
    assert r.json()["review"]["action"] == action
