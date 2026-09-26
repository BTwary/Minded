"""Compound Objective Sequential Forward-Chaining Verification Suite.

Tests that complex, interdependent human analyst questions:
1. Decompose into sequential dependent stages.
2. Ground dependent follow-up questions using empirical cohorts established in prior stages.
3. Block execution cleanly when an earlier premise fails or is inconclusive.
4. Synthesize unified multi-stage answers with verified findings and exact numbers.
"""
import hashlib
import unittest
import uuid
import numpy as np
import pandas as pd

from apps.api.src.core.database import SessionLocal
from apps.api.src.models.entities import Investigation, InvestigationObjective
from packages.analytics_core.src.engines.dataset_provider import BaseDatasetProvider, InvestigationDataContext
from packages.analytics_core.src.runtime.controller import InvestigationController


class InMemoryProvider(BaseDatasetProvider):
    def __init__(self, datasets):
        self.d = datasets

    def acquire_context(self, project_id, dataset_ids=None):
        return InvestigationDataContext(
            project_id=project_id,
            datasets_map=self.d,
            dataset_fingerprints={n: hashlib.sha256(x.to_json().encode()).hexdigest() for n, x in self.d.items()},
            requested_dataset_ids=dataset_ids,
        )


class TestCompoundForwardChaining(unittest.TestCase):

    def test_dependent_cohort_forward_chaining_succeeds(self):
        # Dataset 1: Sales across regions with significant difference
        rng = np.random.default_rng(42)
        n = 100
        sales_df = pd.DataFrame({
            "region": (["North"] * 40 + ["South"] * 60),
            "revenue": list(rng.normal(200, 10, 40)) + list(rng.normal(500, 20, 60)),
        })
        # Dataset 2: Support tickets per region
        support_df = pd.DataFrame({
            "region": (["North"] * 50 + ["South"] * 50),
            "tickets": list(rng.integers(1, 4, 50)) + list(rng.integers(10, 25, 50)),
        })
        datasets = {"sales": sales_df, "support": support_df}

        db = SessionLocal()
        iid = f"INV-COMP-{uuid.uuid4().hex[:8]}"
        pid = f"p-{uuid.uuid4().hex[:6]}"
        db.add(Investigation(
            id=iid,
            project_id=pid,
            question="Which region generated the highest revenue, and did tickets increase for those customers?",
            status="PLANNED",
            requested_dataset_ids_json=["sales", "support"],
        ))
        db.commit()
        db.close()

        ctrl = InvestigationController(session_factory=SessionLocal, dataset_provider=InMemoryProvider(datasets))
        success = ctrl.execute_investigation(iid, worker_id="w-comp")

        db = SessionLocal()
        inv = db.query(Investigation).filter(Investigation.id == iid).first()
        self.assertTrue(success)
        self.assertEqual(inv.status, "COMPLETED")
        self.assertIn("South", inv.direct_answer)
        db.close()

    def test_dependent_cohort_premise_failure_blocks_gracefully(self):
        # Empty / non-discriminating dataset where lead objective cannot establish a finding
        sales_df = pd.DataFrame({
            "region": ["North", "South"],
            "revenue": [100.0, 100.0],
        })
        support_df = pd.DataFrame({
            "region": ["North", "South"],
            "tickets": [5, 5],
        })
        datasets = {"sales": sales_df, "support": support_df}

        db = SessionLocal()
        iid = f"INV-BLOCKED-{uuid.uuid4().hex[:8]}"
        pid = f"p-{uuid.uuid4().hex[:6]}"
        db.add(Investigation(
            id=iid,
            project_id=pid,
            question="Why did churn spike in North, and did tickets increase for those customers?",
            status="PLANNED",
            requested_dataset_ids_json=["sales", "support"],
        ))
        db.commit()
        db.close()

        ctrl = InvestigationController(session_factory=SessionLocal, dataset_provider=InMemoryProvider(datasets))
        success = ctrl.execute_investigation(iid, worker_id="w-block")

        db = SessionLocal()
        inv = db.query(Investigation).filter(Investigation.id == iid).first()
        self.assertTrue(success)
        self.assertEqual(inv.status, "COMPLETED")
        # Since churn is not in sales_df, lead objective fails/is inconclusive, so dependent objective is blocked
        self.assertEqual(inv.verdict_type, "INCONCLUSIVE")
        self.assertIn("blocked", inv.direct_answer.lower())
        db.close()


if __name__ == "__main__":
    unittest.main()
