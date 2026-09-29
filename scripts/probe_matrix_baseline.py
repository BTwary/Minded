"""
probe_matrix_baseline.py

One-off, throwaway diagnostic script. NOT a test. Its only job is to run a
handful of representative questions through the REAL InvestigationController
end-to-end and print exactly what terminal state each one currently reaches,
so we can write test_controller_question_matrix.py against observed fact
instead of assumption -- per instruction: "do not invent expected verdicts
for missing-variable cases; first determine what the existing terminal-state
contract says."

Run: .venv/bin/python scripts/probe_matrix_baseline.py
"""
import hashlib
import os
import sys
import uuid
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from apps.api.src.core.database import SessionLocal, Base, engine as _engine
from apps.api.src.models.entities import Investigation, InvestigationVerdict
from packages.analytics_core.src.engines.dataset_provider import BaseDatasetProvider, InvestigationDataContext
from packages.analytics_core.src.runtime.controller import InvestigationController


class InMemoryProvider(BaseDatasetProvider):
    def __init__(self, d):
        self.d = d

    def acquire_context(self, project_id, dataset_ids=None):
        return InvestigationDataContext(
            project_id=project_id,
            datasets_map=self.d,
            dataset_fingerprints={n: hashlib.sha256(x.to_json().encode()).hexdigest() for n, x in self.d.items()},
            requested_dataset_ids=dataset_ids,
        )


def run(df, question, label):
    db = SessionLocal()
    iid = f"INV-PROBE-{uuid.uuid4().hex[:8]}"
    db.add(Investigation(id=iid, project_id=f"p-{uuid.uuid4().hex[:6]}", question=question, status="PLANNED"))
    db.commit()
    db.close()
    err = None
    try:
        InvestigationController(
            session_factory=SessionLocal,
            dataset_provider=InMemoryProvider({"t": df}),
        ).execute_investigation(investigation_id=iid, worker_id="w-probe")
    except Exception as e:  # noqa: BLE001 -- diagnostic only, we want to SEE crashes, not hide them
        err = f"{type(e).__name__}: {e}"

    db = SessionLocal()
    inv = db.query(Investigation).filter(Investigation.id == iid).first()
    verdict_record = db.query(InvestigationVerdict).filter(InvestigationVerdict.investigation_id == iid).first()
    print(f"\n=== {label} ===")
    print(f"question: {question!r}")
    print(f"raised exception out of execute_investigation: {err!r}")
    if inv is None:
        print("Investigation row: MISSING (never persisted)")
    else:
        print(f"status: {inv.status}")
        print(f"verdict_type: {inv.verdict_type}")
        print(f"confidence_score: {inv.confidence_score}")
        print(f"direct_answer: {(inv.direct_answer or '')[:300]!r}")
        print(f"main_finding: {(inv.main_finding or '')[:300]!r}")
    print(f"verdict_record: {verdict_record.verdict_type if verdict_record else None}")
    db.close()


def main():
    Base.metadata.create_all(_engine)
    rng = np.random.RandomState(1)

    # ---------------------------------------------------------------- MISSING VARIABLE
    df_basic = pd.DataFrame({
        "revenue": rng.normal(1000, 100, 60),
        "region": rng.choice(["East", "West"], 60),
        "date": pd.date_range("2024-01-01", periods=60, freq="D"),
    })
    run(df_basic, "Is revenue correlated with marketing_spend?", "MISSING-VAR / correlation (marketing_spend absent)")
    run(df_basic, "Does churn_flag differ by region?", "MISSING-VAR / rate (churn-worded, churn_flag absent)")
    run(df_basic, "What proportion of orders were returned?", "MISSING-VAR / rate (non-churn-worded, 'returned' absent)")
    run(df_basic, "What is the trend in blorptastic_index over time?", "MISSING-VAR / trend (nonsense column)")
    run(df_basic, "Which category has the highest widget_count?", "MISSING-VAR / ranking (widget_count+category absent)")

    # ---------------------------------------------------------------- CORRELATION: genuine null / underpowered
    n = 200
    df_corr_null = pd.DataFrame({
        "x_metric": rng.normal(50, 10, n),
        "y_metric": rng.normal(50, 10, n),  # independently drawn -- true r ~ 0
    })
    run(df_corr_null, "Is x_metric correlated with y_metric?", "CORRELATION / genuine null (n=200, independent)")

    rng2 = np.random.RandomState(7)
    n_small = 6
    x_small = rng2.normal(50, 10, n_small)
    y_small = 0.6 * x_small + rng2.normal(0, 10, n_small)  # real but weak/noisy signal, tiny n
    df_corr_under = pd.DataFrame({"x_metric": x_small, "y_metric": y_small})
    run(df_corr_under, "Is x_metric correlated with y_metric?", "CORRELATION / underpowered (n=6, real weak signal)")

    # ---------------------------------------------------------------- RATE: genuine null / underpowered
    n_grp = 300
    df_rate_null = pd.DataFrame({
        "group": ["A"] * n_grp + ["B"] * n_grp,
        "converted": list(rng.binomial(1, 0.20, n_grp)) + list(rng.binomial(1, 0.20, n_grp)),
    })
    run(df_rate_null, "Does the conversion rate differ between group A and group B?", "RATE / genuine null (n=300/300, p=0.20 both)")

    n_small_grp = 5
    df_rate_under = pd.DataFrame({
        "group": ["A"] * n_small_grp + ["B"] * n_small_grp,
        "converted": [1, 1, 0, 1, 0] + [0, 0, 1, 0, 0],
    })
    run(df_rate_under, "Does the conversion rate differ between group A and group B?", "RATE / underpowered (n=5/5)")

    # ---------------------------------------------------------------- TREND: genuine null / underpowered
    df_trend_null = pd.DataFrame({
        "month": pd.date_range("2024-01-01", periods=32, freq="MS"),
        "flat_metric": 1000 + rng.normal(0, 30, 32),  # noise only, no slope
    })
    run(df_trend_null, "Is there a trend in flat_metric over time?", "TREND / genuine null (n=32, flat + noise)")

    df_trend_under = pd.DataFrame({
        "month": pd.date_range("2024-01-01", periods=4, freq="MS"),
        "flat_metric": [1000, 1050, 1080, 1200],  # only 4 points
    })
    run(df_trend_under, "Is there a trend in flat_metric over time?", "TREND / underpowered (n=4)")

    # ---------------------------------------------------------------- RANKING: near-tie / tiny-n
    df_rank_tie = pd.DataFrame({
        "category": ["A"] * 50 + ["B"] * 50 + ["C"] * 50,
        "sales": list(rng.normal(1000, 50, 50)) + list(rng.normal(1002, 50, 50)) + list(rng.normal(998, 50, 50)),
    })
    run(df_rank_tie, "Which category has the highest sales?", "RANKING / near-tie (A~B~C)")

    df_rank_tiny = pd.DataFrame({
        "category": ["A", "B", "C"],
        "sales": [10, 11, 9],
    })
    run(df_rank_tiny, "Which category has the highest sales?", "RANKING / tiny-n (1 row per category)")


if __name__ == "__main__":
    main()
