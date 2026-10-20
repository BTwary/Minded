"""Trace "Compare regions A, B and C by total revenue in 2025." through the real API.
Run: python scripts/trace_region_compare.py
"""
import csv, json, os, sys, tempfile, uuid, collections
_tmp = tempfile.mkdtemp(prefix="trace_")
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/t.db"
os.environ["AAOS_DATA_ROOT"] = f"{_tmp}/data"
os.environ.update(ENVIRONMENT="development", AAOS_AUTO_CREATE_DB_SCHEMA="true", AI_ENABLED="false", AI_PROVIDER="none", AAOS_BUSINESS_TIMEZONE="UTC")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

QUESTION = "Compare regions A, B and C by total revenue in 2025."

def build_fixture(path):
    rows = []
    n = 0
    for region, base in (("A", 100), ("B", 150), ("C", 200), ("D", 999)):
        for year in (2024, 2025):
            for m in range(1, 13):
                n += 1
                rows.append({"order_id": f"O{n:04d}", "region": region, "order_date": f"{year}-{m:02d}-15",
                             "revenue": base + m + (1000 if year == 2024 else 0), "units": m})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    exp = collections.defaultdict(float)   # independent expectation: plain python, no pandas/duckdb
    for r in rows:
        if r["order_date"].startswith("2025") and r["region"] in "ABC":
            exp[r["region"]] += r["revenue"]
    return dict(exp)

def main():
    from fastapi.testclient import TestClient
    from apps.api.src.main import app
    from apps.api.src.core.database import SessionLocal, init_db
    from apps.api.src.api.v1.auth import get_current_user
    from apps.api.src.models.entities import User, Project, Dataset, DatasetVersion
    init_db()
    csv_path = f"{_tmp}/sales.csv"
    expected = build_fixture(csv_path)
    print("EXPECTED (independent):", expected)
    db = SessionLocal()
    user = User(id=str(uuid.uuid4()), email="t@x.ai", full_name="T", hashed_password="pw", is_active=True)
    db.add(user); db.commit(); db.refresh(user)
    project = Project(id=str(uuid.uuid4()), name="P", owner_id=user.id); db.add(project); db.commit()
    ds = Dataset(id=str(uuid.uuid4()), project_id=project.id, name="sales", row_count=96, column_count=5); db.add(ds); db.commit()
    db.add(DatasetVersion(id=str(uuid.uuid4()), dataset_id=ds.id, version_number=1, file_path=csv_path, row_count=96, column_count=5, content_hash="fp1")); db.commit()
    app.dependency_overrides[get_current_user] = lambda: user
    c = TestClient(app)
    r = c.post("/api/v1/investigations/interpret", json={"question": QUESTION, "project_id": project.id, "dataset_ids": [ds.id]})
    print("STATUS", r.status_code)
    body = r.json()
    keep = ["can_auto_proceed","ambiguity_status","planned_calculation","clarification_prompt","clarification_options","pending_intent_slot","plan_hash","target_metric","group_dimension"]
    print(json.dumps({k: body.get(k) for k in keep}, indent=2, default=str))
    gr = body.get("guidance_resolution", {})
    print("RESOLVER status:", gr.get("status"), "| reason:", gr.get("reason"))
    ctr = (body.get("canonical_contract") or {})
    print("CONTRACT:", json.dumps({k: ctr.get(k) for k in ("target_column","requested_aggregation","grouping_columns","filter_conditions","time_column","time_range","completeness","unresolved_roles")}, indent=2, default=str))
    print("LEDGER:", [(i["slot"], i["status"], i["provenance"]) for i in (ctr.get("intent_ledger") or [])])
    return c, project, ds, body, expected

if __name__ == "__main__":
    main()


def stage_execute(c, project, ds, body, expected):
    """Create via the real endpoint (with the approved plan hash), then run the real controller/worker path."""
    from apps.api.src.core.database import SessionLocal
    from packages.analytics_core.src.engines.dataset_provider import DatabaseDatasetProvider
    from packages.analytics_core.src.runtime.controller import InvestigationController
    r = c.post("/api/v1/investigations", json={"question": QUESTION, "project_id": project.id, "dataset_ids": [ds.id], "plan_hash": body["plan_hash"]})
    print("\n== CREATE", r.status_code, json.dumps(r.json(), default=str)[:400])
    iid = r.json()["investigation_id"]
    ctl = InvestigationController(session_factory=SessionLocal, dataset_provider=DatabaseDatasetProvider(session_factory=SessionLocal))
    ctl.execute_investigation(iid, worker_id="trace")
    det = c.get(f"/api/v1/investigations/{iid}")
    print("== DETAIL", det.status_code)
    d = det.json()
    print("keys:", sorted(d.keys())[:60])
    for k in ("status", "verdict_type", "direct_answer", "answer", "headline"):
        if k in d: print(k, "=", json.dumps(d[k], default=str)[:800])
    return iid, d

if __name__ == "__main__":
    pass
