"""Release regressions for the human-guided analysis path (answers -> analysis -> output).

Each case builds a deterministic dataset, runs the REAL guided resolver + canonical controller in a
subprocess (fresh SQLite DB), and compares the printed answer with an independent pandas computation.
Covered defects: RATE questions inconclusive/"Simpson" override, "the churn rate" grouped by itself,
"how many X churned" demanding an identifier, missing measure (profit) silently substituted, multi-dataset
scope not auto-selecting the one dataset that covers the question, integer/percent formatting.
"""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]

_SCRIPT = textwrap.dedent(
    """
    import sys, os, uuid, json, hashlib, warnings, time
    warnings.filterwarnings("ignore")
    tmp, tables_dir, question, scope = sys.argv[1:5]
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp}/t.db"
    sys.path.insert(0, os.getcwd())
    import pandas as pd
    from datetime import datetime, timezone
    from apps.api.src.core.database import SessionLocal, init_db
    init_db()
    from apps.api.src.models.entities import Investigation, InvestigationEvent
    from packages.analytics_core.src.engines.dataset_provider import BaseDatasetProvider, InvestigationDataContext
    from packages.analytics_core.src.runtime.controller import InvestigationController
    from packages.analytics_core.src.governance.guided_investigation import apply_human_guidance_to_contract, ResolverStatus
    names = ["customers", "sales"]
    allt = {n: pd.read_csv(f"{tables_dir}/{n}.csv") for n in names}
    tables = allt if scope == "all" else {scope: allt[scope]}
    ctx = InvestigationDataContext(project_id="p", datasets_map=tables,
        dataset_fingerprints={n: hashlib.sha256(pd.util.hash_pandas_object(x).values.tobytes()).hexdigest() for n, x in tables.items()},
        requested_dataset_ids=None, dataset_versions={n: {"dataset_id": n} for n in tables})
    class P(BaseDatasetProvider):
        def acquire_context(self, project_id, dataset_ids=None): return ctx
    res = apply_human_guidance_to_contract(question=question, data_context=ctx, guidance={})
    if res.status != ResolverStatus.RESOLVED:
        print("RESULT" + json.dumps({"stage": "resolver", "reason": res.reason})); raise SystemExit
    iid = "INV-" + uuid.uuid4().hex[:6]; db = SessionLocal()
    db.add(Investigation(id=iid, project_id="p", question=question, status="QUEUED"))
    ds = res.schema_information.get("dataset")
    db.add(InvestigationEvent(id="EVT-" + uuid.uuid4().hex[:8], sequence=1, investigation_id=iid,
        event_type="investigation.canonical_contract_locked",
        event_payload_json={"version": 1, "contract": res.contract.to_dict(), "dataset_name": ds, "dataset_metadata": {},
            "dataset_fingerprint": ctx.dataset_fingerprints.get(ds), "locked_at": datetime.now(timezone.utc).isoformat()},
        timestamp=datetime.now(timezone.utc)))
    db.commit(); db.close()
    InvestigationController(session_factory=SessionLocal, dataset_provider=P()).execute_investigation(investigation_id=iid, worker_id="w")
    db = SessionLocal(); i = db.query(Investigation).filter_by(id=iid).first()
    print("RESULT" + json.dumps({"stage": "executed", "status": i.status, "verdict": i.verdict_type,
                                 "answer": i.direct_answer or "", "dataset": ds}))
    """
)


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    rng = np.random.default_rng(7)
    n = 900
    customers = pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(n)],
        "region": rng.choice(["North", "South", "East", "West"], n),
        "segment": rng.choice(["Consumer", "Corporate", "Small Business"], n, p=[.5, .2, .3]),
    })
    base = customers["segment"].map({"Consumer": .25, "Corporate": .10, "Small Business": .20})
    customers["churned"] = (rng.random(n) < base).astype(int)
    sales = pd.DataFrame({
        "order_id": range(1500),
        "region": rng.choice(["North", "South", "East", "West"], 1500),
        "product_category": rng.choice(["Electronics", "Furniture", "Apparel"], 1500),
        "revenue": np.round(rng.gamma(2.0, 300.0, 1500), 2),
    })
    d = tmp_path_factory.mktemp("tbl")
    customers.to_csv(d / "customers.csv", index=False)
    sales.to_csv(d / "sales.csv", index=False)
    return d, customers, sales


def run(tables, tmp_path, question, scope):
    d = tables[0]
    out = subprocess.run([sys.executable, "-c", _SCRIPT, str(tmp_path), str(d), question, scope],
                         cwd=ROOT, capture_output=True, text=True, timeout=600)
    line = [l for l in out.stdout.splitlines() if l.startswith("RESULT")]
    assert line, out.stdout[-2000:] + out.stderr[-2000:]
    return json.loads(line[-1][6:])


def test_churn_rate_ranking_is_answered_not_inconclusive(tables, tmp_path):
    _, customers, _ = tables
    r = run(tables, tmp_path, "Which segment has the highest churn rate?", "customers")
    assert r["status"] == "COMPLETED" and r["verdict"] == "OBSERVED", r
    assert "simpson" not in r["answer"].lower()
    expected = customers.groupby("segment")["churned"].mean().sort_values(ascending=False)
    parts = [p.strip() for p in r["answer"].split("Executed")[0].split(";")]
    assert [p.split(":")[0] for p in parts] == list(expected.index)
    for p, (_, v) in zip(parts, expected.items()):
        assert p.split(":")[1].strip() == f"{v * 100:.2f}%"


def test_overall_churn_rate_is_not_grouped_by_itself(tables, tmp_path):
    _, customers, _ = tables
    r = run(tables, tmp_path, "What is the churn rate?", "customers")
    assert r["answer"].startswith(f"{customers['churned'].mean() * 100:.2f}%"), r


def test_how_many_churned_counts_positive_flag(tables, tmp_path):
    _, customers, _ = tables
    r = run(tables, tmp_path, "How many customers churned?", "customers")
    assert r["status"] == "COMPLETED"
    assert r["answer"].startswith(f"{int(customers['churned'].sum()):,}"), r


def test_missing_measure_is_refused_not_substituted(tables, tmp_path):
    r = run(tables, tmp_path, "What is the total profit?", "sales")
    assert r["stage"] == "resolver" and "profit" in r["reason"], r


def test_multi_dataset_scope_auto_selects_covering_dataset(tables, tmp_path):
    _, _, sales = tables
    r = run(tables, tmp_path, "What is the total revenue by product_category?", "all")
    assert r["stage"] == "executed" and r["dataset"] == "sales", r
    top = sales.groupby("product_category")["revenue"].sum().sort_values(ascending=False)
    assert r["answer"].startswith(f"{top.index[0]}:"), r


def test_multi_dataset_churn_goes_to_customers(tables, tmp_path):
    r = run(tables, tmp_path, "Which segment has the highest churn rate?", "all")
    assert r["stage"] == "executed" and r["dataset"] == "customers", r
