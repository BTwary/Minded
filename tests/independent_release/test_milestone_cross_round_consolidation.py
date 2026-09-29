"""Milestone forensic test: does cross-round hypothesis consolidation
actually work end-to-end through the real InvestigationController TODAY?

This reproduces the exact symptom the defect register describes for
DEFECT-007-B ("Region B collapsed" rediscovered across multiple replanning
rounds, spawning near-duplicate hypotheses that fragment posterior mass),
using the real controller with no monkeypatching, plus the "shoe size"
no-signal regression the register says a naive dedup fix broke previously.

This does NOT assume the register's "OPEN" status is current -- it tests
the actual code as shipped in this tree.
"""
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from apps.api.src.models.entities import Base, User, Project, Investigation, gen_uuid
from packages.analytics_core.src.runtime.controller import InvestigationController
from packages.analytics_core.src.engines.dataset_provider import InMemoryDatasetProvider


class BaseIsolatedControllerTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "isolated_test.db")
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        self.SessionFactory = sessionmaker(bind=self.engine)
        Base.metadata.create_all(bind=self.engine)
        self.db = self.SessionFactory()
        self.user = User(id=f"usr-{gen_uuid()[:8]}", email="tester@aaos.ai",
                          hashed_password="pw", full_name="Forensic Tester", is_active=True)
        self.proj = Project(id=f"prj-{gen_uuid()[:8]}", name="Isolated Test Project",
                             owner_id=self.user.id)
        self.db.add_all([self.user, self.proj])
        self.db.commit()

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def _run(self, question: str, df: pd.DataFrame, table_name: str, tag: str):
        inv = Investigation(
            id=f"INV-{tag}-{gen_uuid()[:8]}",
            project_id=self.proj.id,
            user_id=self.user.id,
            question=question,
            status="QUEUED",
        )
        self.db.add(inv)
        self.db.commit()
        provider = InMemoryDatasetProvider({table_name: df})
        controller = InvestigationController(session_factory=self.SessionFactory, dataset_provider=provider)
        success = controller.execute_investigation(investigation_id=inv.id, worker_id=f"worker-{tag}")
        self.db.expire_all()
        refreshed = self.db.query(Investigation).filter(Investigation.id == inv.id).first()
        return success, refreshed


class TestCrossRoundFragmentation(BaseIsolatedControllerTest):
    """Reproduces the exact 'Region B rediscovered 3x' scenario from the
    register's original DEFECT-007 diagnosis (HYP-04/06/07 all independently
    describing the same localized anomaly, fragmenting posterior mass across
    three near-duplicate hypotheses instead of consolidating onto one)."""

    def test_localized_anomaly_rediscovered_does_not_fragment(self):
        rng = np.random.default_rng(42)
        n_per_region = 250
        regions = ["Region A", "Region B", "Region C", "Region D"]
        rows = []
        for region in regions:
            for i in range(n_per_region):
                if region == "Region B":
                    # Region B has genuinely collapsed -- a strong, real,
                    # localized anomaly that multiple detector passes are
                    # likely to independently rediscover across rounds.
                    value = rng.normal(12.0, 2.0)
                else:
                    value = rng.normal(100.0, 5.0)
                rows.append({"region": region, "order_id": f"{region}-{i}", "revenue": max(value, 0.0)})
        df = pd.DataFrame(rows)

        success, inv = self._run(
            "Why did revenue collapse in some regions?", df, "sales", "fragtest"
        )
        self.assertTrue(success, "Controller execution must succeed.")

        hyps = list(inv.hypotheses) if hasattr(inv, "hypotheses") else []
        # Fall back to querying if relationship isn't eagerly loaded
        if not hyps:
            from apps.api.src.models.entities import Hypothesis
            hyps = self.db.query(Hypothesis).filter(Hypothesis.investigation_id == inv.id).all()

        region_b_hyps = [
            h for h in hyps
            if "region b" in (getattr(h, "claim", "") or "").lower()
            or str(getattr(h, "target_value", "")).lower() == "region b"
        ]

        print(f"\n[cross-round test] verdict={inv.verdict_type} confidence={inv.confidence_score}")
        print(f"[cross-round test] total hypotheses={len(hyps)}, Region-B-targeting hypotheses={len(region_b_hyps)}")
        for h in hyps:
            print(f"    ALL: code={getattr(h,'hypothesis_code',None)} claim={(getattr(h,'claim','') or '')[:60]!r} "
                  f"posterior={getattr(h,'posterior_probability',None)} "
                  f"supporting_ev={getattr(h,'supporting_evidence_ids',None)} "
                  f"contradicting_ev={getattr(h,'contradicting_evidence_ids',None)}")

        # THE CORE ASSERTION: if consolidation genuinely works cross-round,
        # there should be at most ONE surviving hypothesis object targeting
        # Region B's collapse (others merged into it), not 2-3 fragments
        # each holding a sliver of posterior mass.
        self.assertLessEqual(
            len(region_b_hyps), 1,
            f"Cross-round consolidation did NOT collapse duplicate Region-B "
            f"hypotheses -- found {len(region_b_hyps)} distinct hypothesis "
            f"objects referencing Region B. This is the exact DEFECT-007-B "
            f"fragmentation symptom."
        )
        if region_b_hyps:
            self.assertGreaterEqual(
                region_b_hyps[0].posterior_probability, 0.70,
                "The single surviving, consolidated, directly-tested Region B "
                "hypothesis should reach a diagnostic posterior, not have its "
                "mass still trapped below threshold."
            )


class TestNoRegressionOnUnanswerableQuestion(BaseIsolatedControllerTest):
    """The exact regression the register documents: consolidation must NOT
    make a genuinely unanswerable question return a false confident answer
    by shrinking the hypothesis count and inflating generic hypotheses'
    posterior share through normalization alone."""

    def test_unanswerable_question_stays_inconclusive(self):
        rng = np.random.default_rng(7)
        df = pd.DataFrame({
            "customer_id": [f"C{i}" for i in range(500)],
            "revenue": rng.normal(100, 10, 500),
            "region": rng.choice(["Region A", "Region B", "Region C"], 500),
        })
        success, inv = self._run(
            "What is the average shoe size of our customers?", df, "sales", "shoesize"
        )
        self.assertTrue(success, "Controller execution must succeed (fail-closed, not crash).")
        print(f"\n[no-signal test] verdict={inv.verdict_type} confidence={inv.confidence_score}")
        self.assertIn(
            str(inv.verdict_type),
            {"INCONCLUSIVE", "INSUFFICIENT_EVIDENCE", "INSUFFICIENT_DATA"},
            f"A question about data that does not exist in the schema must not "
            f"reach a confident verdict. Got: {inv.verdict_type} "
            f"(confidence={inv.confidence_score})"
        )


if __name__ == "__main__":
    unittest.main()
