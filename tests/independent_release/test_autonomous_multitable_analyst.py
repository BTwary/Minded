"""Autonomous Multi-Table Relational Question Solving Verification Suite.

Tests that complex enterprise questions traversing rows, columns, and multiple tables:
1. Identify multi-table safe relational plans across 3 tables (e.g. orders -> customers, orders -> products).
2. Apply row-level filters on terminal tables (e.g. products.category = 'Premium').
3. Prevent fan-out multiplication via grain safety verification.
4. Synthesize the joined row-level dataset for AnalystResult.
5. Elevate the final investigation verdict to OBSERVED with numbers-first breakdown.
"""
import hashlib
import unittest
import uuid
import pandas as pd
import numpy as np

from apps.api.src.core.database import SessionLocal
from apps.api.src.models.entities import Investigation
from packages.analytics_core.src.engines.dataset_provider import BaseDatasetProvider, InvestigationDataContext
from packages.analytics_core.src.runtime.controller import InvestigationController
from tests.independent_release.test_milestone_relational_execution import _make_fixture


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


class TestAutonomousMultiTableAnalyst(unittest.TestCase):

    def test_three_table_relational_question_resolves_to_observed(self):
        customers, products, orders = _make_fixture()
        datasets = {"customers": customers, "products": products, "orders": orders}

        db = SessionLocal()
        iid = f"INV-REL-AUTO-{uuid.uuid4().hex[:8]}"
        pid = f"p-{uuid.uuid4().hex[:6]}"
        db.add(Investigation(
            id=iid,
            project_id=pid,
            question="Which customer segment generated the highest revenue from the premium product category?",
            status="PLANNED",
            requested_dataset_ids_json=["customers", "products", "orders"],
        ))
        db.commit()
        db.close()

        ctrl = InvestigationController(session_factory=SessionLocal, dataset_provider=InMemoryProvider(datasets))
        success = ctrl.execute_investigation(iid, worker_id="w-rel-auto")

        db = SessionLocal()
        inv = db.query(Investigation).filter(Investigation.id == iid).first()
        self.assertTrue(success)
        self.assertEqual(inv.status, "COMPLETED")
        self.assertEqual(inv.verdict_type, "OBSERVED")
        self.assertEqual(inv.confidence_score, 1.0)
        self.assertIn("Enterprise", inv.direct_answer)
        self.assertIn("highest total revenue", inv.direct_answer)
        self.assertIn("Ranking by total revenue", inv.direct_answer)
        db.close()


if __name__ == "__main__":
    unittest.main()
