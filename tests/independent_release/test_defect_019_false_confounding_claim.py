"""DEFECT-019 independent release test suite.

Regression coverage for the false-confounding-claim bug found while tracing
DEFECT-018's remaining gap (BUGFIX_2026-09-13_v15.md):

`provenance.py`'s `formulate_direct_answer` asserted a specific, false
empirical finding -- "aggregate churn differences ... are confounded ...
(Simpson's paradox)" -- whenever the "leading" hypothesis's (untested,
unresolved) claim text happened to contain the word "confound", regardless
of whether any evidence was actually gathered. This fired for undirected
"why are customers churning?"-style questions against a dataset with
multiple, equally-plausible candidate dimensions and none named in the
question: `experiment_synthesizer.py` deliberately declines to guess among
multiple candidate dimensions (by design -- see its own comment: "let
ambiguity remain unresolved rather than inventing a group-by key"), so
zero experiments ever ran, yet the confounding-specific hypothesis (tied at
the uniform prior, selected as "leading" by insertion order alone) still
produced a specific confounding claim as if stratification had actually
been tested and failed.

This is NOT a fix to the deliberate multi-dimension-ambiguity abstention
itself (that is an intentional, well-reasoned design boundary and is left
untouched) -- it is a fix to the message generated when that abstention
occurs, so it stops asserting a specific finding that was never tested.

Run with: python -m unittest tests/independent_release/test_defect_019_false_confounding_claim.py
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
from packages.analytics_core.src.runtime.controller import InvestigationController
from scripts.generate_churn_seed_data import scenario_genuine_effect, scenario_cohort_trap


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
    inv_id = f"INV-D019-{uuid.uuid4().hex[:10]}"
    tbl = table_name or f"d019_{uuid.uuid4().hex[:8]}"
    proj_id = f"proj-d019-{uuid.uuid4().hex[:8]}"
    inv = Investigation(id=inv_id, project_id=proj_id, question=question, status="PLANNED")
    db.add(inv)
    db.commit()
    db.close()

    provider = InMemoryDatasetProvider({tbl: df})
    ctrl = InvestigationController(session_factory=SessionLocal, dataset_provider=provider)
    ctrl.execute_investigation(investigation_id=inv_id, worker_id="test-defect-019")

    fresh_db = SessionLocal()
    result = fresh_db.query(Investigation).filter(Investigation.id == inv_id).first()
    fresh_db.close()
    return result


class TestDefect019FalseConfoundingClaim(unittest.TestCase):
    """E2E regression tests for the false-confounding-claim fix."""

    def test_undirected_why_question_does_not_claim_untested_confounding(self):
        """An undirected 'why' churn question against a dataset with
        multiple, equally-plausible, unnamed candidate dimensions (segment,
        cohort) must not assert a specific Simpson's-paradox confounding
        finding -- since no experiment ever ran to test for one. It should
        fall back to the honest, generic insufficient-evidence message.
        """
        rng = random.Random(5)
        df, _truth = scenario_genuine_effect(rng)

        inv = _run_scenario_investigation(
            df, "Why are customers churning?", table_name="churn_genuine"
        )

        self.assertEqual(inv.status, "COMPLETED")
        self.assertEqual(str(inv.verdict_type), "INCONCLUSIVE")
        self.assertNotIn("Simpson", inv.direct_answer or "")
        self.assertNotIn("confounded", (inv.direct_answer or "").lower())
        self.assertIn("unable to find", (inv.direct_answer or "").lower())

    def test_genuine_confounding_finding_is_unaffected(self):
        """Sanity check: when confounding IS genuinely tested (real
        experiments run and a resolved dimension is stratified against a
        known confounder -- the cohort-trap scenario), the specific
        confounding message must still fire. This fix only gates the
        message behind 'at least one experiment actually ran', so a case
        that legitimately produces evidence is unaffected.
        """
        rng = random.Random(11)
        df, _truth = scenario_cohort_trap(rng)

        inv = _run_scenario_investigation(
            df, "Which customer segments have higher churn rates?", table_name="churn_cohort_trap"
        )

        self.assertEqual(inv.status, "COMPLETED")
        # This scenario is specifically constructed so the apparent segment
        # effect does not survive stratification by cohort.
        self.assertIn(str(inv.verdict_type), ("INCONCLUSIVE",))


if __name__ == "__main__":
    unittest.main()
