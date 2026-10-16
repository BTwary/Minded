"""End-to-end regression for the real 4-CSV workspace failure.

Symptom seen in the UI: asking "Which product category generates the highest total revenue?" returned
``Investigation Execution Error: Contract <uuid> not found`` / INCONCLUSIVE with an empty finding.

Cause: on a file-backed SQLite DB one shared connection served every thread, so the UI's status polling
rolled back the worker's uncommitted contract row. This test runs the REAL controller on a worker thread
against a fresh file DB (in a subprocess, so the module-level engine really is file-backed) while another
thread polls the investigation exactly like the UI does, and checks the answer against pandas.
"""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

_SCRIPT = textwrap.dedent(
    """
    import json, os, sys, threading, time
    tmp, csv = sys.argv[1], sys.argv[2]
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp}/t.db"
    os.environ["DATA_STORAGE_DIR"] = f"{tmp}/storage"
    sys.path.insert(0, os.getcwd())
    from apps.api.src.core import database as dbmod
    dbmod.init_db()
    from apps.api.src.models.entities import Project, Investigation
    from apps.api.src.services.dataset_service import DatasetService
    from packages.analytics_core.src.runtime.controller import InvestigationController
    db = dbmod.SessionLocal(); p = Project(name="ws"); db.add(p); db.commit()
    DatasetService(db).ingest_dataset_file(p.id, "orders.csv", open(csv, "rb").read())
    iid = "INV-E2E"
    s = dbmod.SessionLocal()
    s.add(Investigation(id=iid, project_id=p.id, question="Which product category generates the highest total revenue?", status="PLANNED"))
    s.commit(); s.close()
    err = []
    def work():
        try:
            InvestigationController(session_factory=dbmod.SessionLocal).execute_investigation(investigation_id=iid, worker_id="w")
        except Exception as e:
            err.append(repr(e))
    th = threading.Thread(target=work); th.start()
    while th.is_alive():
        ps = dbmod.SessionLocal(); ps.query(Investigation).filter(Investigation.id == iid).first(); ps.close(); time.sleep(0.02)
    s = dbmod.SessionLocal(); inv = s.query(Investigation).filter(Investigation.id == iid).first()
    print("RESULT=" + json.dumps({"err": err, "status": inv.status, "answer": inv.direct_answer}))
    """
)


def test_polled_investigation_completes_and_names_the_right_category(tmp_path) -> None:
    rng = np.random.default_rng(7)
    n = 120_000
    cats = rng.choice(["Electronics", "Furniture", "Books", "Sports"], size=n, p=[0.3, 0.3, 0.25, 0.15])
    price = {"Electronics": 900.0, "Furniture": 600.0, "Books": 20.0, "Sports": 70.0}
    qty = rng.integers(1, 4, size=n)
    revenue = np.round(np.array([price[c] for c in cats]) * qty * rng.uniform(0.9, 1.1, size=n), 2)
    df = pd.DataFrame(
        {
            "order_id": [f"O{i:06d}" for i in range(n)],
            "order_date": pd.date_range("2023-01-01", periods=n, freq="min").strftime("%Y-%m-%d"),
            "category": cats,
            "quantity": qty,
            "revenue": revenue,
        }
    )
    csv = tmp_path / "orders.csv"
    df.to_csv(csv, index=False)
    expected_top = df.groupby("category").revenue.sum().idxmax()

    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(tmp_path), str(csv)],
        cwd=str(ROOT), capture_output=True, text=True, timeout=280,
    )
    line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT=")), None)
    assert line, proc.stdout[-2000:] + proc.stderr[-2000:]
    out = json.loads(line[len("RESULT="):])
    assert out["err"] == [], out["err"]
    assert out["status"] == "COMPLETED"
    assert expected_top in out["answer"]
    assert "not found" not in out["answer"].lower()
