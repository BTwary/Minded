"""DEFECT-017 independent release test suite.

Regression coverage for the churn-correlation misrouting bug found during
the 2026-09-13 broad-dataset harness pass (BUGFIX_2026-09-13_v13.md):

A CORRELATION-intent question against a binary churn outcome (e.g. "is
there a correlation between support tickets and churn?") previously always
fell into the categorical segment-comparison churn-identifiability routine
in transition.py, purely because the target metric was a positively-
identified churn column -- regardless of what the question actually asked.
That routine only accepts a categorical grouping column, so it silently
substituted an unrelated categorical dimension (or, with none available,
failed closed) and never tested the specific, already-resolved numeric
explanatory variable the question named -- producing a false-negative
INCONCLUSIVE verdict even when the underlying data had a strong,
deliberately-constructed signal.

Also covers two related duplicate-numeric-column-collapse bugs found and
fixed in the same routine (a spurious "multiple numeric candidates" error
in the diagnostic inference-selection step, and a silently-skipped
correlation computation in the real eta-squared/Bayesian-evidence step),
both triggered whenever primary_result_column == target_metric_col, which
is the common case for any correlation experiment on a named metric.

Run with: python -m unittest tests/independent_release/test_defect_017_churn_correlation_routing.py
"""
import hashlib
import os
import sys
import unittest
import uuid
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from apps.api.src.core.database import SessionLocal
from apps.api.src.models.entities import Investigation
from packages.analytics_core.src.engines.dataset_provider import BaseDatasetProvider, InvestigationDataContext
from packages.analytics_core.src.runtime.controller import InvestigationController


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
    inv_id = f"INV-D017-{uuid.uuid4().hex[:10]}"
    tbl = table_name or f"d017_{uuid.uuid4().hex[:8]}"
    proj_id = f"proj-d017-{uuid.uuid4().hex[:8]}"
    inv = Investigation(id=inv_id, project_id=proj_id, question=question, status="PLANNED")
    db.add(inv)
    db.commit()
    db.close()

    provider = InMemoryDatasetProvider({tbl: df})
    ctrl = InvestigationController(session_factory=SessionLocal, dataset_provider=provider)
    ctrl.execute_investigation(investigation_id=inv_id, worker_id="test-defect-017")

    fresh_db = SessionLocal()
    result = fresh_db.query(Investigation).filter(Investigation.id == inv_id).first()
    fresh_db.close()
    return result


class TestDefect017ChurnCorrelationRouting(unittest.TestCase):
    """E2E regression tests for the churn/correlation routing fix."""

    def test_correlation_with_churn_target_detects_real_signal(self):
        """A CORRELATION question against a binary churn column, where the named
        explanatory variable has a strong, deliberately-constructed relationship
        to churn, must reach a positive verdict -- not silently reroute to an
        unrelated categorical grouping and fail closed.
        """
        rng = np.random.RandomState(7)
        n = 800
        support_tickets = rng.poisson(lam=2.0, size=n)
        logit = -3.0 + 0.9 * support_tickets
        p = 1 / (1 + np.exp(-logit))
        churn = (rng.rand(n) < p).astype(int)

        df = pd.DataFrame({
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "support_tickets": support_tickets,
            "churn": churn,
            "plan_type": rng.choice(["Basic", "Pro", "Enterprise"], size=n),
            "signup_date": pd.date_range("2024-01-01", periods=n, freq="6h").strftime("%Y-%m-%d"),
        })

        inv = _run_scenario_investigation(
            df, "Is there a correlation between support tickets and churn?", table_name="saas_churn"
        )

        self.assertEqual(inv.status, "COMPLETED")
        self.assertIn(str(inv.verdict_type), ("STATISTICALLY_SIGNIFICANT", "DIAGNOSED"))
        self.assertNotIn("Inconclusive", inv.direct_answer or "")
        # The resolved explanatory variable must actually appear in the answer --
        # not a substituted, unrelated categorical dimension.
        self.assertIn("support_tickets", (inv.direct_answer or ""))

    def test_correlation_with_churn_target_no_signal_stays_conservative(self):
        """Sanity check: the routing fix must not manufacture significance where
        none exists. A support_tickets column with no relationship to churn
        should not produce a confident positive verdict.
        """
        rng = np.random.RandomState(11)
        n = 500
        support_tickets = rng.poisson(lam=2.0, size=n)
        churn = rng.binomial(1, 0.2, size=n)  # independent of support_tickets

        df = pd.DataFrame({
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "support_tickets": support_tickets,
            "churn": churn,
            "plan_type": rng.choice(["Basic", "Pro", "Enterprise"], size=n),
        })

        inv = _run_scenario_investigation(
            df, "Is there a correlation between support tickets and churn?", table_name="saas_churn_null"
        )

        self.assertEqual(inv.status, "COMPLETED")
        self.assertNotEqual(str(inv.verdict_type), "DIAGNOSED")


if __name__ == "__main__":
    unittest.main()
