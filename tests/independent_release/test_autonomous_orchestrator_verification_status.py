"""P0 epistemic-integrity fix: AutonomousOrchestrator -> InvestigationRuntime
-> AnalysisResponse black-box regression.

This is the THIRD independent construction site for the same
case-sensitive validation_status bug (after AnalysisService and
state_reconstruction.py): apps/api/src/ai/runtime.py drives the real
InvestigationController against its own ephemeral in-memory SQLite
database and previously mapped Evidence.validation_status the same
broken way, so this path -- used by AutonomousOrchestrator and the
benchmark/acceptance scripts that call it directly (test_ai_fallback.py,
test_zero_ai_mode.py) -- also silently reported every evidence row as
SKIPPED regardless of real verification outcome.

This test drives the actual public entrypoint,
AutonomousOrchestrator.run_investigation(...), rather than calling
InvestigationRuntime or InvestigationController directly, so it proves
the full path a real caller uses.

Run with: python -m unittest discover -s tests/independent_release -p "test_*.py"
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from apps.api.src.ai.orchestrator import AutonomousOrchestrator
from apps.api.src.ai.providers.mock import NoneAIProvider
from packages.schemas.src.analysis import ValidationStatus


class TestAutonomousOrchestratorVerificationStatus(unittest.TestCase):
    """Black-box proof: a genuine VERIFIED result reaching AnalysisResponse
    through AutonomousOrchestrator -> InvestigationRuntime cannot be
    silently reported as SKIPPED."""

    def setUp(self):
        np.random.seed(7)
        n_rows = 400
        dates = pd.date_range("2026-01-01", periods=90, freq="D")
        revenue = np.round(np.random.uniform(50.0, 500.0, size=n_rows), 2)
        # Deterministic, genuine positive correlation -- not a coincidence
        # of random data -- so the dual-engine (duckdb/polars) pearsonr
        # check has real predictive structure to independently verify.
        cost = np.round(revenue * 0.6 + np.random.normal(0, 5.0, size=n_rows), 2)

        df_sales = pd.DataFrame({
            "order_id": [f"ORD-{i:05d}" for i in range(n_rows)],
            "order_date": np.random.choice(dates, size=n_rows),
            "customer_id": [f"CUST-{np.random.randint(1, 100):03d}" for _ in range(n_rows)],
            "region": np.random.choice(["North", "South", "East", "West"], size=n_rows),
            "revenue": revenue,
            "cost": cost,
            "quantity": np.random.randint(1, 10, size=n_rows),
        })
        df_sales.attrs["dataset_id"] = "ds-sales-orchestrator-test"
        df_sales.attrs["version"] = 1
        self.datasets = {"sales": df_sales}

    def test_orchestrator_verified_correlation_is_not_reported_as_skipped(self):
        orchestrator = AutonomousOrchestrator(ai_provider=NoneAIProvider())
        response = orchestrator.run_investigation(
            question="Is there a correlation between revenue and cost?",
            project_id="proj-orchestrator-verification-test",
            datasets=self.datasets,
        )

        self.assertGreater(len(response.evidence), 0, "expected at least one evidence row")
        statuses = {e.validation_status for e in response.evidence}

        # This is the exact invariant the P0 fix establishes: a real,
        # dual-engine-verified result must never surface as SKIPPED through
        # this path. Before the fix, every status in `statuses` here would
        # have been ValidationStatus.SKIPPED regardless of this outcome.
        self.assertNotIn(ValidationStatus.SKIPPED, statuses)
        self.assertIn(ValidationStatus.PASSED, statuses)

        # Bonus coverage: this path also had its own num_steps==0-style gap
        # (steps were built from InvestigationEvent rows, but is exercised
        # here as a secondary check on the same real run rather than a
        # separate synthetic case).
        self.assertGreater(len(response.steps), 0)


if __name__ == "__main__":
    unittest.main()
