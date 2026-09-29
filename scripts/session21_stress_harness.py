"""Session 21 stress-test harness.

Drives several messy, real-world-style synthetic datasets through the REAL
InvestigationController.execute_investigation path (not unit-level shortcuts), the same way
the 2026-09-13 broad-dataset e2e stress test did. Goal: find crashes, blank terminal states,
and questionable verdicts across a spread of question types and data shapes, to inform the
release-readiness assessment.
"""
import os
import sys
import tempfile
import traceback

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from apps.api.src.models.entities import Base, User, Project, Investigation, gen_uuid
from packages.analytics_core.src.runtime.controller import InvestigationController
from packages.analytics_core.src.engines.dataset_provider import InMemoryDatasetProvider

np.random.seed(42)


def _new_session_factory():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "stress.db")
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False}, poolclass=StaticPool)

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_connection, _):
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.close()

    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def ds_saas_churn():
    n = 2000
    tenure = np.random.exponential(12, n).clip(0, 60)
    support_tickets = np.random.poisson(1.5, n)
    plan = np.random.choice(["Free", "Starter", "Pro", "Enterprise"], n, p=[0.4, 0.3, 0.2, 0.1])
    churn_prob = 1 / (1 + np.exp(-(0.3 * support_tickets - 0.05 * tenure - 1.0)))
    churned = (np.random.random(n) < churn_prob).astype(int)
    df = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "tenure_months": np.round(tenure, 1),
        "support_tickets": support_tickets,
        "plan_type": plan,
        "monthly_revenue": np.where(plan == "Free", 0, np.random.gamma(4, 15, n)),
        "churned": churned,
        "signup_date": pd.to_datetime("2024-01-01") + pd.to_timedelta(np.random.randint(0, 600, n), unit="D"),
    })
    # messiness: some nulls, a duplicate-key row, a stray whitespace category
    df.loc[df.sample(30, random_state=1).index, "support_tickets"] = np.nan
    df = pd.concat([df, df.iloc[[5]]], ignore_index=True)  # duplicate customer_id
    df.loc[df.sample(10, random_state=2).index, "plan_type"] = "Pro "  # trailing space
    return df


def ds_regional_sales():
    n = 1500
    dates = pd.date_range("2026-01-01", periods=180, freq="D")
    region = np.random.choice(["North", "South", "East", "West"], n)
    base = np.random.gamma(6, 40, n)
    d = np.random.choice(dates, n)
    # planted effect: West region drops hard after March 20th
    revenue = np.where((region == "West") & (d >= pd.Timestamp("2026-03-20")), base * 0.2, base)
    df = pd.DataFrame({"order_date": d, "region": region, "revenue": revenue, "channel": np.random.choice(["Online", "Retail", "Partner"], n)})
    return df


def ds_manufacturing_defects():
    n = 3000
    line = np.random.choice(["Line-A", "Line-B", "Line-C"], n)
    shift = np.random.choice(["Day", "Night"], n)
    temp = np.random.normal(70, 5, n)
    # planted: Night shift on Line-B has elevated defect rate
    defect_prob = 0.03 + 0.15 * ((line == "Line-B") & (shift == "Night"))
    defects = (np.random.random(n) < defect_prob).astype(int)
    df = pd.DataFrame({"production_line": line, "shift": shift, "temperature_f": temp, "is_defective": defects, "units_produced": np.random.randint(80, 150, n)})
    return df


def ds_tiny_hr():
    # deliberately tiny -- tests small-n handling
    return pd.DataFrame({
        "department": ["Eng", "Eng", "Sales", "Sales", "Sales", "HR"],
        "attrition": [1, 0, 1, 1, 0, 0],
        "tenure_years": [2.1, 5.3, 0.8, 1.2, 3.0, 4.4],
    })


def ds_all_null_column():
    n = 500
    df = pd.DataFrame({
        "product": np.random.choice(["Widget", "Gadget", "Gizmo"], n),
        "units_sold": np.random.poisson(20, n),
        "warranty_claim_reason": [None] * n,  # 100% null, irrelevant to most questions
        "revenue": np.random.gamma(5, 30, n),
    })
    return df


DATASETS = {
    "saas_churn": (ds_saas_churn(), {
        "Why are customers churning?",
        "Is churn correlated with support tickets?",
        "What is the churn rate by plan type?",
        "Why did revenue drop for free-tier customers?",
        "Is there a relationship between tenure and churn?",
    }),
    "regional_sales": (ds_regional_sales(), {
        "Why did revenue fall in the West region since the start of the quarter?",
        "Which region has the highest revenue?",
        "Why did revenue drop this year?",
        "How does revenue differ between channels?",
    }),
    "manufacturing": (ds_manufacturing_defects(), {
        "Why did the defect rate spike on Line-B?",
        "Is defect rate correlated with temperature?",
        "Which production line has the most defects?",
        "Why are night shift defects higher?",
    }),
    "tiny_hr": (ds_tiny_hr(), {
        "Why is attrition higher in Sales?",
        "What is the average tenure by department?",
    }),
    "all_null_col": (ds_all_null_column(), {
        "Which product has the highest revenue?",
        "Why did units_sold vary by product?",
        "What is the average warranty_claim_reason?",  # deliberately asks about the null column
    }),
}

results = []

for ds_name, (df, questions) in DATASETS.items():
    for q in questions:
        SessionFactory = _new_session_factory()
        db = SessionFactory()
        user = User(id=f"usr-{gen_uuid()[:8]}", email="stress@aaos.ai", hashed_password="pw", full_name="Stress Tester", is_active=True)
        proj = Project(id=f"prj-{gen_uuid()[:8]}", name="Stress Project", owner_id=user.id)
        db.add_all([user, proj])
        db.commit()

        inv = Investigation(id=f"INV-{gen_uuid()[:8]}", project_id=proj.id, user_id=user.id, question=q, status="QUEUED")
        db.add(inv)
        db.commit()

        provider = InMemoryDatasetProvider({ds_name: df})
        controller = InvestigationController(session_factory=SessionFactory, dataset_provider=provider)
        row = {"dataset": ds_name, "question": q}
        try:
            success = controller.execute_investigation(investigation_id=inv.id, worker_id="stress-worker")
            db.expire_all()
            completed = db.query(Investigation).filter(Investigation.id == inv.id).first()
            row["success"] = success
            row["status"] = completed.status if completed else None
            row["verdict_type"] = getattr(completed, "verdict_type", None)
            row["direct_answer_len"] = len(completed.direct_answer or "") if hasattr(completed, "direct_answer") and completed.direct_answer else 0
            row["main_finding_blank"] = not bool(getattr(completed, "main_finding", None))
            row["error"] = None
        except Exception as e:  # noqa: BLE001
            row["success"] = False
            row["status"] = "EXCEPTION"
            row["verdict_type"] = None
            row["direct_answer_len"] = 0
            row["main_finding_blank"] = True
            row["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=6)}"
        results.append(row)
        db.close()

print(f"{'dataset':15} {'status':10} {'verdict':22} {'ans_len':8} {'blank?':7} question")
for r in results:
    print(f"{r['dataset']:15} {str(r['status']):10} {str(r['verdict_type']):22} {r['direct_answer_len']:<8} {str(r['main_finding_blank']):7} {r['question']}")

print("\n=== ERRORS/CRASHES ===")
for r in results:
    if r["error"]:
        print(f"--- {r['dataset']} / {r['question']!r} ---")
        print(r["error"])

print("\n=== SUSPICIOUS (blank finding or non-success) ===")
for r in results:
    if r["main_finding_blank"] or not r["success"]:
        print(r)
