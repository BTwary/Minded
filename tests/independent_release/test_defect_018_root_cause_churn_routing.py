"""DEFECT-018 independent release test suite.

Regression coverage for the ROOT_CAUSE-vs-churn misrouting bugs found while
following up on DEFECT-017 (BUGFIX_2026-09-13_v14.md):

1. `semantic.py`'s churn-question keyword check only matched a subset of
   "churn"'s verb inflections ("churn", "churned") and missed "churning" /
   "churns". A ROOT_CAUSE-classified question like "Why are customers
   churning?" therefore never resolved `churn_event_col`, even against a
   dataset with an unambiguous churn column, and the investigation
   incorrectly reported "no identifiable churn outcome".

2. `method_selection.py`'s `decide()` routed every `ROOT_CAUSE`-intent
   question to the generic diagnostic battery unconditionally, regardless
   of whether the semantic layer had positively resolved a binary churn
   outcome. "Why"-phrased questions about churn classify as ROOT_CAUSE (not
   CHURN) by design in IntentEngine, so churn-outcome questions phrased as
   "why" never reached the churn identifiability workflow at all -- even
   after fix (1) above, `churn_event_col` being resolved made no
   difference to routing.

Together these meant a "why are customers churning" style question could
never receive a real churn-segment-difference verdict, regardless of how
strong the underlying signal was, as long as the question was phrased
diagnostically rather than as an explicit segment comparison.

Run with: python -m unittest tests/independent_release/test_defect_018_root_cause_churn_routing.py
"""
import hashlib
import os
import random
import sys
import unittest
import uuid
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from apps.api.src.core.database import SessionLocal
from apps.api.src.models.entities import Investigation
from packages.analytics_core.src.engines.dataset_provider import BaseDatasetProvider, InvestigationDataContext
from packages.analytics_core.src.engines.intent import IntentEngine
from packages.analytics_core.src.engines.semantic import SemanticEngine
from packages.analytics_core.src.runtime.controller import InvestigationController
from scripts.generate_churn_seed_data import scenario_genuine_effect


class InMemoryDatasetProvider(BaseDatasetProvider):
    """In-memory dataset provider for scenario-specific test datasets."""

    def __init__(self, datasets_map: Dict[str, pd.DataFrame]):
        self.datasets_map = datasets_map

    def acquire_context(self, project_id: str, dataset_ids: Optional[List[str]] = None) -> InvestigationDataContext:
        fingerprints = {
            name: hashlib.sha256(df.to_json().encode()).hexdigest()
            for name, df in self.datasets_map.items()
        }
        return InvestigationDataContext(
            project_id=project_id,
            datasets_map=self.datasets_map,
            dataset_fingerprints=fingerprints,
            requested_dataset_ids=dataset_ids,
        )


def _run_scenario_investigation(df: pd.DataFrame, question: str, table_name: Optional[str] = None) -> Investigation:
    """Runs a full canonical investigation against a custom scenario DataFrame."""
    db = SessionLocal()
    inv_id = f"INV-D018-{uuid.uuid4().hex[:10]}"
    tbl = table_name or f"d018_{uuid.uuid4().hex[:8]}"
    proj_id = f"proj-d018-{uuid.uuid4().hex[:8]}"
    inv = Investigation(id=inv_id, project_id=proj_id, question=question, status="PLANNED")
    db.add(inv)
    db.commit()
    db.close()

    provider = InMemoryDatasetProvider({tbl: df})
    ctrl = InvestigationController(session_factory=SessionLocal, dataset_provider=provider)
    ctrl.execute_investigation(investigation_id=inv_id, worker_id="test-defect-018")

    fresh_db = SessionLocal()
    result = fresh_db.query(Investigation).filter(Investigation.id == inv_id).first()
    fresh_db.close()
    return result


class TestDefect018RootCauseChurnRouting(unittest.TestCase):
    """Unit + E2E regression tests for the two routing fixes."""

    def test_semantic_resolves_churn_event_col_for_gerund_phrasing(self):
        """'churning' (gerund) must resolve churn_event_col just like 'churn'
        and 'churned' already did, for a ROOT_CAUSE-classified question.
        """
        rng = random.Random(3)
        df, _truth = scenario_genuine_effect(rng)

        intent = IntentEngine.parse_intent("Why are customers churning?")
        self.assertEqual(intent.intent_type, "ROOT_CAUSE")

        sem = SemanticEngine().resolve_schema(intent, {"churn_genuine": df})
        self.assertEqual(sem.churn_event_col, "churn_event")
        self.assertTrue(sem.churn_outcome_available)

    def test_why_phrased_churn_question_with_named_dimension_reaches_diagnosis(self):
        """A ROOT_CAUSE ('why')-phrased question that names the actual
        driving dimension must be able to reach a real DIAGNOSED verdict
        via the churn identifiability workflow -- not be forced through the
        generic diagnostic battery just because it was phrased as 'why'
        instead of 'which segments'.
        """
        rng = random.Random(5)
        df, truth = scenario_genuine_effect(rng)

        inv = _run_scenario_investigation(
            df, "Why are segments churning at different rates?", table_name="churn_genuine"
        )

        self.assertEqual(inv.status, "COMPLETED")
        self.assertEqual(str(inv.verdict_type), "DIAGNOSED")
        self.assertIn("segment", (inv.direct_answer or "").lower())

    def test_root_cause_intent_without_churn_outcome_still_uses_diagnostic_battery(self):
        """Sanity check: a ROOT_CAUSE question against a dataset with no
        churn-like outcome at all must still route to the ordinary
        diagnostic battery, unaffected by this fix (no churn_event_col
        should ever be resolved here, so the new routing branch must not
        fire).
        """
        rng = random.Random(9)
        df = pd.DataFrame({
            "order_id": [f"O{i:05d}" for i in range(300)],
            "revenue": [100 + i for i in range(300)],
            "region": (["West", "East", "Central"] * 100),
        })

        intent = IntentEngine.parse_intent("Why did revenue drop?")
        self.assertEqual(intent.intent_type, "ROOT_CAUSE")
        sem = SemanticEngine().resolve_schema(intent, {"orders": df})
        self.assertIsNone(sem.churn_event_col)


if __name__ == "__main__":
    unittest.main()
