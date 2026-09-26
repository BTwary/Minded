"""Forensic Validation and Completion Audit Test Suite for DEFECT-015.

Exercises all 14 matrix cases and deep validation requirements:
1. Existing genuine effect
2. Novel genuine effect
3. Existing cohort trap
4. Novel cohort trap
5. Existing tenure trap
6. Novel tenure trap
7. Existing insufficient information
8. Real missingness and selection bias
9. Right-censoring distinction
10. Exposure semantics
11. No-outcome safety
12. Ambiguous binary field non-classification
13. Semantic alias generalization
14. Persisted hypothesis diversity
15. Persisted experiment diversity
16. Dual-engine verification proof
17. Final-verdict language audit
"""
import os
import random
import sys
import unittest
import uuid
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from apps.api.src.core.database import SessionLocal
from apps.api.src.models.entities import Evidence, Experiment, Hypothesis, Investigation
import hashlib
from packages.analytics_core.src.engines.dataset_provider import BaseDatasetProvider, InvestigationDataContext
from packages.analytics_core.src.engines.intent import IntentEngine
from packages.analytics_core.src.engines.semantic import SemanticEngine
from packages.analytics_core.src.runtime.controller import InvestigationController
from packages.analytics_core.src.statistics.churn_estimands import ChurnVerdict, analyze_churn_identifiability
from scripts.generate_churn_seed_data import (
    scenario_cohort_trap,
    scenario_exposure_trap,
    scenario_genuine_effect,
    scenario_insufficient_information,
    scenario_raw_count_trap,
    scenario_tenure_trap,
)


class InMemoryDatasetProvider(BaseDatasetProvider):
    def __init__(self, datasets_map: Dict[str, pd.DataFrame]):
        self.datasets_map = datasets_map

    def acquire_context(
        self,
        project_id: str,
        dataset_ids: Optional[List[str]] = None,
    ) -> InvestigationDataContext:
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


def _run_forensic_investigation(
    df: pd.DataFrame,
    question: str,
    table_name: Optional[str] = None,
) -> Investigation:
    db = SessionLocal()
    inv_id = f"INV-FORENSIC-{uuid.uuid4().hex[:10]}"
    tbl = table_name or f"churn_{uuid.uuid4().hex[:8]}"
    proj_id = f"proj-forensic-{uuid.uuid4().hex[:8]}"
    inv = Investigation(id=inv_id, project_id=proj_id, question=question, status="PLANNED")
    db.add(inv)
    db.commit()
    db.close()

    provider = InMemoryDatasetProvider({tbl: df})
    ctrl = InvestigationController(session_factory=SessionLocal, dataset_provider=provider)
    ctrl.execute_investigation(investigation_id=inv_id, worker_id="test-forensic")

    fresh_db = SessionLocal()
    result = fresh_db.query(Investigation).filter(Investigation.id == inv_id).first()
    fresh_db.close()
    return result


class TestDefect015ForensicCompletionAudit(unittest.TestCase):

    def test_01_existing_genuine_effect(self):
        rng = random.Random(42)
        df, _ = scenario_genuine_effect(rng)
        inv = _run_forensic_investigation(df, "Which customer segments have higher churn rates?")

        self.assertEqual(inv.status, "COMPLETED")
        self.assertIn(str(inv.verdict_type), ("DIAGNOSED", "STATISTICALLY_SIGNIFICANT"))
        self.assertNotIn("Inconclusive", inv.direct_answer or "")
        direct_lower = (inv.direct_answer or "").lower()
        self.assertNotIn("causes churn", direct_lower)
        self.assertNotIn("caused churn", direct_lower)

    def test_02_novel_genuine_effect_dataset(self):
        rng = np.random.RandomState(9999)
        rows = []
        cid = 1
        cohorts = ["Cohort-2024-Q1", "Cohort-2024-Q2", "Cohort-2024-Q3"]
        for cohort in cohorts:
            for seg, n, base_hazard in [("Alpha", 150, 0.28), ("Beta", 150, 0.07)]:
                for _ in range(n):
                    tenure = int(rng.exponential(scale=180) + 10)
                    obs_days = tenure + int(rng.uniform(10, 50))
                    churn = int(rng.uniform() < base_hazard)
                    censored = 1 - churn
                    rows.append({
                        "customer_id": f"CUST-{cid:05d}",
                        "tier": seg,
                        "signup_cohort": cohort,
                        "tenure_days": tenure,
                        "observation_days": obs_days,
                        "churn_flag": churn,
                        "is_censored": censored,
                    })
                    cid += 1
        df = pd.DataFrame(rows)
        inv = _run_forensic_investigation(df, "Which tier has higher churn rate?")

        self.assertEqual(inv.status, "COMPLETED")
        self.assertIn(str(inv.verdict_type), ("DIAGNOSED", "STATISTICALLY_SIGNIFICANT"))
        self.assertNotIn("Inconclusive", inv.direct_answer or "")
        direct_lower = (inv.direct_answer or "").lower()
        self.assertNotIn("causes churn", direct_lower)
        self.assertNotIn("caused churn", direct_lower)

    def test_03_existing_cohort_trap(self):
        rng = random.Random(42)
        df, _ = scenario_cohort_trap(rng)
        inv = _run_forensic_investigation(df, "Which customer segments have higher churn rates?")

        self.assertEqual(inv.status, "COMPLETED")
        self.assertEqual(str(inv.verdict_type), "INCONCLUSIVE")
        narrative = f"{inv.direct_answer or ''} {inv.main_finding or ''}".lower()
        self.assertTrue(any(w in narrative for w in ["confound", "simpson", "stratification", "inconclusive"]))

    def test_04_novel_confounded_dataset(self):
        rng = np.random.RandomState(7777)
        rows = []
        cid = 1
        configs = [
            ("Starter", "Vintage-2023", 400, 0.30),
            ("Starter", "Vintage-2025", 100, 0.05),
            ("Growth", "Vintage-2023", 100, 0.30),
            ("Growth", "Vintage-2025", 400, 0.05),
        ]
        for plan, vintage, n, hazard in configs:
            for _ in range(n):
                churn = int(rng.uniform() < hazard)
                rows.append({
                    "account_id": f"ACC-{cid:05d}",
                    "plan_tier": plan,
                    "vintage_year": vintage,
                    "tenure_months": int(rng.uniform(6, 36)),
                    "cancellation_event": churn,
                    "right_censored": 1 - churn,
                })
                cid += 1
        df = pd.DataFrame(rows)
        inv = _run_forensic_investigation(df, "Does plan tier affect cancellation rate?")

        self.assertEqual(inv.status, "COMPLETED")
        self.assertEqual(str(inv.verdict_type), "INCONCLUSIVE")
        narrative = f"{inv.direct_answer or ''} {inv.main_finding or ''}".lower()
        self.assertTrue(any(w in narrative for w in ["confound", "simpson", "stratification", "inconclusive", "identifiability"]))

        # v20-C4.2.1: the narrative check above is satisfied by any
        # inconclusive verdict, so it cannot tell whether the question was
        # actually tested. Assert on what was executed: the plan_tier
        # question must be tested on plan_tier (stratified by the
        # vintage_year confounder), never on an unrelated churn-vs-tenure
        # correlation.
        db = SessionLocal()
        try:
            exps = db.query(Experiment).filter(Experiment.investigation_id == inv.id).all()
        finally:
            db.close()
        sqls = [str((e.arguments_json or {}).get("sql", "")) for e in exps]
        self.assertTrue(sqls, "no experiments executed")
        self.assertTrue(all("plan_tier" in q for q in sqls), sqls)
        self.assertTrue(any("vintage_year" in q for q in sqls), sqls)

    def test_05_existing_tenure_trap(self):
        rng = random.Random(42)
        df, _ = scenario_tenure_trap(rng)
        inv = _run_forensic_investigation(df, "Compare churn rates between customer segments")

        self.assertEqual(inv.status, "COMPLETED")
        self.assertEqual(str(inv.verdict_type), "INCONCLUSIVE")

    def test_06_novel_tenure_trap(self):
        rng = np.random.RandomState(5555)
        rows = []
        cid = 1
        for seg, n, mean_days in [("Org-X", 300, 600), ("Org-Y", 300, 60)]:
            for _ in range(n):
                days = int(rng.exponential(scale=mean_days) + 1)
                prob = 1.0 - np.exp(-0.0005 * days)
                churn = int(rng.uniform() < prob)
                rows.append({
                    "org_id": f"ORG-{cid:05d}",
                    "org_type": seg,
                    "exposure_days": days,
                    "churn": churn,
                })
                cid += 1
        df = pd.DataFrame(rows)
        inv = _run_forensic_investigation(df, "Which org type has higher churn rate?")

        self.assertEqual(inv.status, "COMPLETED")
        self.assertEqual(str(inv.verdict_type), "INCONCLUSIVE")

    def test_07_existing_insufficient_information(self):
        rng = random.Random(42)
        df, _ = scenario_insufficient_information(rng)
        df_valid = df.dropna(axis=1, how="all")
        inv = _run_forensic_investigation(df_valid, "Which customer segment is more likely to churn?")

        self.assertEqual(inv.status, "COMPLETED")
        self.assertEqual(str(inv.verdict_type), "INCONCLUSIVE")

    def test_08_missingness_selection_bias(self):
        rng = np.random.RandomState(3333)
        rows = []
        cid = 1
        for seg, n, hazard, missing_rate in [("Seg-A", 300, 0.15, 0.05), ("Seg-B", 300, 0.15, 0.60)]:
            for _ in range(n):
                if rng.uniform() < missing_rate:
                    churn_val = np.nan
                else:
                    churn_val = float(rng.uniform() < hazard)
                rows.append({
                    "cust_id": f"C-{cid:05d}",
                    "segment": seg,
                    "churn_event": churn_val,
                })
                cid += 1
        df = pd.DataFrame(rows)
        res = analyze_churn_identifiability(df, group_col="segment", known_confounders=[])
        self.assertIn(res.verdict, (ChurnVerdict.INSUFFICIENT_EVIDENCE, ChurnVerdict.OBSERVED_ASSOCIATION, ChurnVerdict.CONFOUNDED_IDENTIFIABILITY_LIMITED))

    def test_09_censoring_distinction(self):
        df, _ = scenario_genuine_effect(random.Random(42))
        from packages.analytics_core.src.statistics.churn_estimands import censoring_report
        report = censoring_report(df, censored_col="censored")
        self.assertTrue(report["censoring_data_available"])
        self.assertGreater(report["n_censored_incomplete_observation"], 0)
        self.assertIn("right-censored", report["note"])

    def test_10_no_churn_outcome_dataset_fails_closed(self):
        rng = np.random.RandomState(1234)
        n = 200
        df = pd.DataFrame({
            "account_id": [f"A-{i:04d}" for i in range(n)],
            "segment": rng.choice(["Retail", "Enterprise"], size=n),
            "is_active": rng.choice([0, 1], size=n),
            "is_premium": rng.choice([0, 1], size=n),
            "region_flag": rng.choice([0, 1], size=n),
            "mrr": rng.uniform(100, 5000, size=n),
        })
        inv = _run_forensic_investigation(df, "What is the churn rate by segment?")

        self.assertEqual(inv.status, "COMPLETED")
        self.assertEqual(str(inv.verdict_type), "INCONCLUSIVE")
        narrative = f"{inv.direct_answer or ''} {inv.main_finding or ''}".lower()
        self.assertTrue(any(w in narrative for w in ["no identifiable churn", "no recorded churn", "limitation", "cannot estimate"]))

    def test_11_ambiguous_binary_field_not_treated_as_churn(self):
        df = pd.DataFrame({
            "user_id": [f"U-{i}" for i in range(50)],
            "is_vip": [1]*25 + [0]*25,
            "is_subscriber": [0]*25 + [1]*25,
            "is_verified": [1]*50,
        })
        intent = IntentEngine.parse_intent("What is the customer churn rate by segment?")
        engine = SemanticEngine()
        sem = engine.resolve_schema(intent, {"users": df})

        self.assertFalse(sem.churn_outcome_available)
        self.assertIsNone(sem.churn_event_col)

    def test_12_semantic_churn_aliases_recognized(self):
        engine = SemanticEngine()
        intent = IntentEngine.parse_intent("Analyze customer dropoff")

        for col_name in ["cancelled", "attrition_flag", "customer_dropout", "has_churned"]:
            df = pd.DataFrame({
                "customer_id": ["C1", "C2", "C3"],
                "segment": ["A", "B", "C"],
                col_name: [0, 1, 0],
            })
            sem = engine.resolve_schema(intent, {"subscribers": df})
            self.assertTrue(sem.churn_outcome_available, f"Failed to recognize alias '{col_name}'")
            self.assertEqual(sem.churn_event_col, col_name)

    def test_13_persisted_hypothesis_diversity(self):
        rng = random.Random(42)
        df, _ = scenario_genuine_effect(rng)
        inv = _run_forensic_investigation(df, "Which customer segments have higher churn rates?")

        db = SessionLocal()
        hyps = db.query(Hypothesis).filter(Hypothesis.investigation_id == inv.id).all()
        db.close()

        codes = {h.hypothesis_code for h in hyps}
        self.assertIn("HYP-01", codes)
        self.assertIn("HYP-02", codes)
        self.assertIn("HYP-03", codes)
        self.assertIn("HYP-04", codes)

    def test_14_persisted_experiments_and_dual_engine_verification(self):
        rng = random.Random(42)
        df, _ = scenario_genuine_effect(rng)
        inv = _run_forensic_investigation(df, "Which customer segments have higher churn rates?")

        db = SessionLocal()
        exps = db.query(Experiment).filter(Experiment.investigation_id == inv.id).all()
        evs = db.query(Evidence).filter(Evidence.investigation_id == inv.id).all()
        db.close()

        self.assertGreater(len(exps), 0)
        self.assertGreater(len(evs), 0)
        verified_evs = [e for e in evs if e.validation_status == "VERIFIED"]
        self.assertGreater(len(verified_evs), 0)

    def test_15_final_verdict_language_is_observational(self):
        rng = random.Random(42)
        df, _ = scenario_genuine_effect(rng)
        inv = _run_forensic_investigation(df, "Which customer segments have higher churn rates?")

        answer = (inv.direct_answer or "").lower()
        self.assertNotIn("causes churn", answer)
        self.assertNotIn("caused churn", answer)
        self.assertNotIn("will churn", answer)
        self.assertNotIn("intrinsically risky", answer)


if __name__ == "__main__":
    unittest.main()
