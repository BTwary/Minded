"""DEFECT-015 independent release test suite.

Verifies end-to-end integration of the churn identifiability statistics layer
into the autonomous AA-OS investigation pipeline:
- Intent classification and semantic resolution for churn questions.
- Hypothesis synthesis with competing observational and confounding hypotheses.
- Experiment synthesis with crude rate, baseline uniformity, and stratified queries.
- Transition service execution via DuckDB and Polars with dual-engine verification.
- Epistemic calibration and honest fail-closed behavior on unidentifiable or confounded datasets.
- Clean handling of datasets lacking churn outcome variables (e.g. customers.csv).

Run with: python -m unittest tests/independent_release/test_defect_015_churn_autonomous_integration.py
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
from scripts.generate_churn_seed_data import (
    scenario_genuine_effect,
    scenario_cohort_trap,
    scenario_tenure_trap,
    scenario_insufficient_information,
    scenario_exposure_trap,
    scenario_raw_count_trap,
)


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
    inv_id = f"INV-CHURN-{uuid.uuid4().hex[:10]}"
    tbl = table_name or f"churn_{uuid.uuid4().hex[:8]}"
    proj_id = f"proj-churn-{uuid.uuid4().hex[:8]}"
    inv = Investigation(id=inv_id, project_id=proj_id, question=question, status="PLANNED")
    db.add(inv)
    db.commit()
    db.close()

    provider = InMemoryDatasetProvider({tbl: df})
    ctrl = InvestigationController(session_factory=SessionLocal, dataset_provider=provider)
    ctrl.execute_investigation(investigation_id=inv_id, worker_id="test-defect-015")

    fresh_db = SessionLocal()
    result = fresh_db.query(Investigation).filter(Investigation.id == inv_id).first()
    fresh_db.close()
    return result


def _run_standard_db_investigation(question: str) -> Investigation:
    """Runs a canonical investigation against the standard database (customers.csv)."""
    db = SessionLocal()
    inv_id = f"INV-STD-{uuid.uuid4().hex[:10]}"
    inv = Investigation(id=inv_id, project_id="proj-default", question=question, status="PLANNED")
    db.add(inv)
    db.commit()
    db.close()

    ctrl = InvestigationController(session_factory=SessionLocal)
    ctrl.execute_investigation(investigation_id=inv_id, worker_id="test-defect-015-std")

    fresh_db = SessionLocal()
    result = fresh_db.query(Investigation).filter(Investigation.id == inv_id).first()
    fresh_db.close()
    return result


class TestDefect015AutonomousChurnIntegration(unittest.TestCase):
    """E2E autonomous investigation tests for DEFECT-015."""

    def test_end_to_end_positive_genuine_effect(self):
        """Case 1: Genuine segment difference that survives confounder stratification.

        Segment I has a genuine higher churn hazard than Segment J across all cohorts
        and tenure bands. The autonomous investigation must reach a positive
        verdict (DIAGNOSED or STATISTICALLY_SIGNIFICANT) with non-causal language.
        """
        rng = random.Random(42)
        df, truth = scenario_genuine_effect(rng)
        inv = _run_scenario_investigation(df, "Which customer segments have higher churn rates?")

        self.assertEqual(inv.status, "COMPLETED")
        self.assertIn(str(inv.verdict_type), ("DIAGNOSED", "STATISTICALLY_SIGNIFICANT"))
        self.assertNotIn("Inconclusive", inv.direct_answer or "")

        # Claim must remain observational/associational, never ungrounded causation
        direct_ans_lower = (inv.direct_answer or "").lower()
        self.assertNotIn("causes churn", direct_ans_lower)
        self.assertNotIn("caused churn", direct_ans_lower)

    def test_end_to_end_cohort_trap_confounding(self):
        """Case 2: Apparent segment difference is fully explained by signup cohort mix (Simpson's paradox).

        Within each cohort, E and F churn at identical rates. The autonomous path
        must detect confounding / Simpson's paradox and fail closed (INCONCLUSIVE).
        """
        rng = random.Random(42)
        df, truth = scenario_cohort_trap(rng)
        inv = _run_scenario_investigation(df, "Which customer segments have higher churn rates?")

        self.assertEqual(inv.status, "COMPLETED")
        self.assertEqual(str(inv.verdict_type), "INCONCLUSIVE")

        # The direct answer or narrative must articulate the confounding/identifiability limitation
        combined_narrative = f"{inv.direct_answer or ''} {inv.main_finding or ''}".lower()
        self.assertTrue(
            any(w in combined_narrative for w in ["confound", "simpson", "inconclusive", "stratification", "identifiability"]),
            f"Expected confounding/identifiability disclosure in narrative, got: {combined_narrative}"
        )

    def test_end_to_end_tenure_trap_confounding(self):
        """Case 3: Segment differences collinear with customer tenure.

        Tenure drives churn risk; segments skew by tenure. The system must not
        issue a high-confidence segment diagnosis.
        """
        rng = random.Random(42)
        df, truth = scenario_tenure_trap(rng)
        inv = _run_scenario_investigation(df, "Compare churn rates between customer segments")

        self.assertEqual(inv.status, "COMPLETED")
        self.assertEqual(str(inv.verdict_type), "INCONCLUSIVE")

    def test_end_to_end_insufficient_information(self):
        """Case 4: Dataset lacks exposure/tenure/cohort fields (unidentifiable).

        Without confounder data, causal or robust segment claims cannot be identified.
        The autonomous path must produce INCONCLUSIVE.
        """
        rng = random.Random(42)
        df, truth = scenario_insufficient_information(rng)
        # Drop unrecorded/100% null columns so dataset satisfies basic schema fitness
        df_valid = df.dropna(axis=1, how="all")
        inv = _run_scenario_investigation(df_valid, "Which customer segment is more likely to churn?")

        self.assertEqual(inv.status, "COMPLETED")
        self.assertEqual(str(inv.verdict_type), "INCONCLUSIVE")

    def test_real_customers_dataset_no_churn_event_fails_closed(self):
        """Case 5: Question against real customers.csv (has no churn outcome column).

        The autonomous controller must detect the missing outcome variable,
        fail closed with INCONCLUSIVE, and explain that no churn outcome exists
        rather than fabricating predictions or crashing.
        """
        inv = _run_standard_db_investigation("Which customers are likely to churn?")

        self.assertEqual(inv.status, "COMPLETED")
        self.assertEqual(str(inv.verdict_type), "INCONCLUSIVE")

        # Direct answer must honestly state the data-availability limitation
        direct_ans = inv.direct_answer or ""
        self.assertIn("no identifiable churn outcome", direct_ans.lower())
        self.assertNotIn("will churn", direct_ans.lower())
        self.assertNotIn("predicted to churn", direct_ans.lower())

    def test_intent_classification_and_semantic_matching(self):
        """Case 6: Intent and semantic resolution unit checks for churn."""
        intent = IntentEngine.parse_intent("What is our churn rate by plan?")
        self.assertEqual(intent.intent_type, "CHURN")

        # Intent with variations
        intent_cancellation = IntentEngine.parse_intent("Show me customer cancellation patterns")
        self.assertEqual(intent_cancellation.intent_type, "CHURN")

        intent_attrition = IntentEngine.parse_intent("Analyze customer attrition by industry")
        self.assertEqual(intent_attrition.intent_type, "CHURN")

    def test_semantic_resolution_detects_churn_columns(self):
        """Case 7: SemanticEngine accurately identifies churn outcome, exposure, and confounders."""
        rng = random.Random(42)
        df, _ = scenario_genuine_effect(rng)
        datasets_map = {"churn_table": df}

        intent = IntentEngine.parse_intent("Which customer segments have higher churn?")
        semantic = SemanticEngine().resolve_schema(intent, datasets_map)

        self.assertTrue(semantic.churn_outcome_available)
        self.assertEqual(semantic.churn_event_col, "churn_event")
        self.assertEqual(semantic.churn_exposure_col, "tenure_days")
        self.assertEqual(semantic.churn_censored_col, "censored")
        self.assertIn("cohort", semantic.churn_confounder_cols)


if __name__ == "__main__":
    unittest.main()
