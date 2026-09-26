"""Dedicated Scientific Calibration & Evidence Integrity Test Suite for DEFECT-015.

Audits the statistical, relational, and epistemic soundness of the churn integration:
1. Bayes rule responsiveness under varying priors.
2. Sensitivity to statistical strength (discrete likelihood mapping analysis).
3. End-to-end missingness / selection-bias enforcement through InvestigationController.
4. End-to-end censoring semantics through InvestigationController (hardened non-vacuous assertions).
5. Exposure adjustment causal diagnostics (Case A: same hazard/different exposure with numerical assertions).
6. Exposure adjustment causal diagnostics (Case B: different hazard/different exposure with rate ratio assertions).
7. Controller-level row-order, label, and table-name invariance (asserting posteriors, exps, and confidences).
8. Group dimension generalization.
9. Identifier exclusion from group dimension.
10. Confounder discovery generalization.
11. End-to-end persisted relational provenance chain strictly scoped to current investigation:
    (Investigation -> Hypothesis -> Experiment -> Observation -> Evidence -> EvidenceVerification -> BeliefUpdate -> InvestigationVerdict).
12. Large raw effect collapsing after stratification (negative calibration inspecting likelihood & posterior).
13. Fail-closed safety under missing churn outcomes.
14. Ambiguous group dimensions fail-closed safety (no arbitrary selection).
15. Confounder vs group dimension strict role separation.
16. Semantic alias negative cases (rejecting non-binary churn fields).
17. Bayesian likelihood monotonicity under progressive evidence strength.
18. Empirical Bayesian calibration across weak, moderate, and strong datasets.
19. Same-verdict different-strength limitation audit (N=200 vs N=5000).
20. Semantic grouping does not leak into causal confirmation.
"""
import hashlib
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
from apps.api.src.models.entities import (
    BeliefUpdate,
    Evidence,
    EvidenceVerification,
    Experiment,
    Hypothesis,
    Investigation,
    InvestigationVerdict,
    Observation,
)
from packages.analytics_core.src.engines.belief import BeliefEngine
from packages.analytics_core.src.engines.dataset_provider import (
    BaseDatasetProvider,
    InvestigationDataContext,
)
from packages.analytics_core.src.engines.intent import IntentEngine
from packages.analytics_core.src.engines.semantic import SemanticEngine
from packages.analytics_core.src.runtime.controller import InvestigationController
from packages.analytics_core.src.statistics.churn_estimands import (
    ChurnVerdict,
    analyze_churn_identifiability,
    censoring_report,
    exposure_adjusted_rate,
)


def _compute_churn_transition_likelihoods(verdict: ChurnVerdict, warnings: Optional[List[str]] = None) -> List[float]:
    """Helper calculating base likelihoods [H1, H2, H3, H4] according to transition service rules."""
    warns = warnings or []
    if verdict == ChurnVerdict.SUPPORTED_SEGMENT_DIFFERENCE:
        h4_like = 0.80 if any("exposure" in w.lower() for w in warns) else 0.20
        return [0.85, 0.15, 0.15, h4_like]
    elif verdict == ChurnVerdict.CONFOUNDED_IDENTIFIABILITY_LIMITED:
        return [0.15, 0.35, 0.85, 0.20]
    elif verdict == ChurnVerdict.OBSERVED_ASSOCIATION:
        return [0.65, 0.35, 0.40, 0.20]
    else:
        return [0.30, 0.70, 0.35, 0.20]


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


def _run_calibrated_investigation(
    df: pd.DataFrame,
    question: str,
    table_name: Optional[str] = None,
) -> Investigation:
    db = SessionLocal()
    inv_id = f"INV-CALIB-{uuid.uuid4().hex[:10]}"
    tbl = table_name or f"churn_{uuid.uuid4().hex[:8]}"
    proj_id = f"proj-calib-{uuid.uuid4().hex[:8]}"
    inv = Investigation(id=inv_id, project_id=proj_id, question=question, status="PLANNED")
    db.add(inv)
    db.commit()
    db.close()

    provider = InMemoryDatasetProvider({tbl: df})
    ctrl = InvestigationController(session_factory=SessionLocal, dataset_provider=provider)
    ctrl.execute_investigation(investigation_id=inv_id, worker_id="test-calib")

    fresh_db = SessionLocal()
    saved_inv = fresh_db.query(Investigation).filter(Investigation.id == inv_id).first()
    return saved_inv


class TestDefect015ScientificCalibrationAudit(unittest.TestCase):
    """Deep scientific calibration and evidence integrity tests for autonomous churn identifiability."""

    def test_01_bayes_prior_responsiveness(self):
        """Proves that with identical likelihoods, different priors produce mathematically different posteriors."""
        priors_uniform = [0.25, 0.25, 0.25, 0.25]
        priors_favor_h1 = [0.70, 0.10, 0.10, 0.10]
        priors_favor_h2 = [0.10, 0.70, 0.10, 0.10]
        likelihoods = [0.85, 0.15, 0.15, 0.15]

        post_uniform, _ = BeliefEngine.compute_bayesian_posteriors(priors_uniform, likelihoods)
        post_favor_h1, _ = BeliefEngine.compute_bayesian_posteriors(priors_favor_h1, likelihoods)
        post_favor_h2, _ = BeliefEngine.compute_bayesian_posteriors(priors_favor_h2, likelihoods)

        self.assertAlmostEqual(post_uniform[0], 0.653846, places=4)
        self.assertAlmostEqual(post_favor_h1[0], 0.929688, places=4)
        self.assertAlmostEqual(post_favor_h2[0], 0.386364, places=4)
        self.assertGreater(post_favor_h1[0], post_uniform[0])
        self.assertGreater(post_uniform[0], post_favor_h2[0])

    def test_02_circularity_audit_and_discrete_likelihood_behavior(self):
        """Audits the mapping from statistical estimands to likelihood vectors."""
        rng = np.random.RandomState(42)
        n = 1000
        df_strong = pd.DataFrame({
            "segment": ["A"]*n + ["B"]*n,
            "cohort": rng.choice(["2024-Q1", "2024-Q2"], size=2*n),
            "churn_event": list(rng.uniform(size=n) < 0.35) + list(rng.uniform(size=n) < 0.05),
        })
        df_strong["churn_event"] = df_strong["churn_event"].astype(int)
        res_strong = analyze_churn_identifiability(df_strong, group_col="segment", known_confounders=["cohort"])
        self.assertEqual(res_strong.verdict, ChurnVerdict.SUPPORTED_SEGMENT_DIFFERENCE)
        self.assertLess(res_strong.aggregate_association["p_value"], 1e-10)

    def test_03_end_to_end_missingness_selection_bias_controller(self):
        """End-to-end controller execution on dataset with 95% vs 40% observed outcomes."""
        rng = np.random.RandomState(999)
        n = 400
        rows = []
        for i in range(n):
            rows.append({
                "account_id": f"ACC-{i:05d}",
                "segment": "Tier-A",
                "cohort": rng.choice(["2024-Q1", "2024-Q2"]),
                "churn_event": int(rng.uniform() < 0.15) if rng.uniform() < 0.95 else np.nan,
            })
        for i in range(n, 2*n):
            rows.append({
                "account_id": f"ACC-{i:05d}",
                "segment": "Tier-B",
                "cohort": rng.choice(["2024-Q1", "2024-Q2"]),
                "churn_event": int(rng.uniform() < 0.15) if rng.uniform() < 0.40 else np.nan,
            })
        df = pd.DataFrame(rows)

        inv = _run_calibrated_investigation(df, "Which segment has higher churn rate?")
        self.assertEqual(inv.status, "COMPLETED")
        self.assertEqual(str(inv.verdict_type), "INCONCLUSIVE")
        narrative = f"{inv.direct_answer or ''} {inv.main_finding or ''}".lower()
        self.assertTrue(any(w in narrative for w in ["limitation", "missing", "inconclusive", "unidentifiable", "cannot confirm", "selection"]))

    def test_04_end_to_end_censoring_semantics_controller(self):
        """End-to-end controller execution with non-vacuous assertions on censoring execution and provenance."""
        rng = np.random.RandomState(888)
        n = 300
        rows = []
        for i in range(n):
            is_censored = 1 if rng.uniform() < 0.40 else 0
            churn = 0 if is_censored else int(rng.uniform() < 0.20)
            rows.append({
                "account_id": f"ACC-{i:05d}",
                "segment": "Alpha",
                "censored": is_censored,
                "churn_event": churn,
            })
        for i in range(n, 2*n):
            is_censored = 1 if rng.uniform() < 0.40 else 0
            churn = 0 if is_censored else int(rng.uniform() < 0.20)
            rows.append({
                "account_id": f"ACC-{i:05d}",
                "segment": "Beta",
                "censored": is_censored,
                "churn_event": churn,
            })
        df = pd.DataFrame(rows)

        inv = _run_calibrated_investigation(df, "What is the churn rate by segment?")
        self.assertEqual(inv.status, "COMPLETED")
        
        db = SessionLocal()
        censor_exps = db.query(Experiment).filter(
            Experiment.investigation_id == inv.id,
            Experiment.test_code == "EXP-CHURN-CENSOR"
        ).all()
        self.assertGreaterEqual(len(censor_exps), 1, "EXP-CHURN-CENSOR must be synthesized and executed")
        self.assertEqual(censor_exps[0].status, "EXECUTED")

        # Assert observation record exists and contains structured breakdown scoped to this experiment
        obs = db.query(Observation).filter(Observation.experiment_id == censor_exps[0].id).first()
        self.assertIsNotNone(obs, "Observation record must exist for censoring experiment")
        self.assertIn("structured_result", obs.result_json)

        # Assert evidence verification was performed and verified scoped to this evidence
        verifs = db.query(EvidenceVerification).filter(
            EvidenceVerification.evidence_id == f"EV_{censor_exps[0].id}"
        ).all()
        self.assertGreaterEqual(len(verifs), 1, "Verification proof must exist for censoring experiment")
        self.assertEqual(verifs[0].status, "VERIFIED")
        db.close()

        # Assert narrative does not conflate censored with confirmed non-churn
        narrative = f"{inv.direct_answer or ''} {inv.main_finding or ''}".lower()
        self.assertNotIn("confirmed non-churn", narrative)

    def test_05_exposure_adjustment_causally_diagnostic_case_a(self):
        """Case A: Same underlying hazard (0.0005/day), 10x exposure duration -> numerical proof of convergence."""
        rng = np.random.RandomState(777)
        rows = []
        cid = 1
        for seg, n, mean_days in [("Enterprise", 400, 600), ("SMB", 400, 60)]:
            for _ in range(n):
                days = int(rng.exponential(scale=mean_days) + 1)
                prob = 1.0 - np.exp(-0.0005 * days)
                churn = int(rng.uniform() < prob)
                rows.append({
                    "account_id": f"ACC-{cid:05d}",
                    "segment": seg,
                    "exposure_days": days,
                    "churn_event": churn,
                })
                cid += 1
        df = pd.DataFrame(rows)

        # Numerical assertion 1: Crude rates differ substantially due to exposure duration
        crude_ent = df[df["segment"] == "Enterprise"]["churn_event"].mean()
        crude_smb = df[df["segment"] == "SMB"]["churn_event"].mean()
        self.assertGreater(crude_ent / crude_smb, 3.0, "Crude churn rate in Enterprise must be >3x higher than SMB")

        # Numerical assertion 2: Person-time rates converge to identical underlying hazard
        exp_df = exposure_adjusted_rate(df, "segment", "exposure_days")
        rate_ent = float(exp_df.loc[exp_df["segment"] == "Enterprise", "persontime_rate"].iloc[0])
        rate_smb = float(exp_df.loc[exp_df["segment"] == "SMB", "persontime_rate"].iloc[0])
        self.assertAlmostEqual(rate_ent, rate_smb, delta=0.0003, msg="Person-time hazard rates must converge")

        inv = _run_calibrated_investigation(df, "Which segment has higher churn?")
        self.assertEqual(inv.status, "COMPLETED")
        self.assertEqual(str(inv.verdict_type), "INCONCLUSIVE")

        db = SessionLocal()
        exp_exps = db.query(Experiment).filter(
            Experiment.investigation_id == inv.id,
            Experiment.test_code == "EXP-CHURN-EXPOSURE"
        ).all()
        self.assertGreaterEqual(len(exp_exps), 1, "EXP-CHURN-EXPOSURE must be executed")
        verifs = db.query(EvidenceVerification).filter(
            EvidenceVerification.evidence_id == f"EV_{exp_exps[0].id}"
        ).all()
        self.assertGreaterEqual(len(verifs), 1)
        self.assertEqual(verifs[0].status, "VERIFIED")
        db.close()

        narrative = f"{inv.direct_answer or ''} {inv.main_finding or ''}".lower()
        self.assertTrue(any(w in narrative for w in ["exposure", "duration", "tenure", "person-time", "inconclusive"]))

    def test_06_exposure_adjustment_causally_diagnostic_case_b(self):
        """Case B: Different underlying hazards (0.0020 vs 0.0004/day) with unequal exposure -> genuine difference survives."""
        rng = np.random.RandomState(666)
        rows = []
        cid = 1
        for seg, n, mean_days, hazard in [("HighRiskTier", 400, 300, 0.0020), ("LowRiskTier", 400, 150, 0.0004)]:
            for _ in range(n):
                days = int(rng.exponential(scale=mean_days) + 1)
                prob = 1.0 - np.exp(-hazard * days)
                churn = int(rng.uniform() < prob)
                rows.append({
                    "account_id": f"ACC-{cid:05d}",
                    "segment": seg,
                    "exposure_days": days,
                    "churn_event": churn,
                })
                cid += 1
        df = pd.DataFrame(rows)

        exp_df = exposure_adjusted_rate(df, "segment", "exposure_days")
        rate_high = float(exp_df.loc[exp_df["segment"] == "HighRiskTier", "persontime_rate"].iloc[0])
        rate_low = float(exp_df.loc[exp_df["segment"] == "LowRiskTier", "persontime_rate"].iloc[0])
        self.assertGreater(rate_high / rate_low, 2.0, "High risk person-time hazard must remain >2x higher")

        inv = _run_calibrated_investigation(df, "Which segment has higher churn?")
        self.assertEqual(inv.status, "COMPLETED")
        
        db = SessionLocal()
        exp_exps = db.query(Experiment).filter(
            Experiment.investigation_id == inv.id,
            Experiment.test_code == "EXP-CHURN-EXPOSURE"
        ).all()
        self.assertGreaterEqual(len(exp_exps), 1, "EXP-CHURN-EXPOSURE must be executed")
        verifs = db.query(EvidenceVerification).filter(
            EvidenceVerification.evidence_id == f"EV_{exp_exps[0].id}"
        ).all()
        self.assertGreaterEqual(len(verifs), 1)
        self.assertEqual(verifs[0].status, "VERIFIED")
        self.assertEqual(verifs[0].observed_delta_pct, 0.0)
        db.close()

    def test_07_controller_row_order_and_label_invariance(self):
        """Shuffled rows, relabeled segments and cohorts reach identical posterior & logical conclusion."""
        rng = np.random.RandomState(42)
        n = 300
        rows = []
        for i in range(n):
            rows.append({"cust_id": f"C{i}", "tier_code": "Gold", "wave": "W1", "attrition_flag": int(rng.uniform() < 0.35)})
        for i in range(n, 2*n):
            rows.append({"cust_id": f"C{i}", "tier_code": "Silver", "wave": "W1", "attrition_flag": int(rng.uniform() < 0.05)})
        df1 = pd.DataFrame(rows)

        df2 = df1.sample(frac=1.0, random_state=1234).reset_index(drop=True)
        df2 = df2.rename(columns={"tier_code": "plan_level", "wave": "cohort_vintage"})

        inv1 = _run_calibrated_investigation(df1, "Which tier has higher churn?", table_name="cust_data_1")
        inv2 = _run_calibrated_investigation(df2, "Which plan level has higher churn?", table_name="cust_data_2")

        self.assertEqual(inv1.status, inv2.status)
        self.assertEqual(inv1.verdict_type, inv2.verdict_type)
        self.assertAlmostEqual(inv1.confidence_score, inv2.confidence_score, places=4)

        db = SessionLocal()
        h1_1 = db.query(Hypothesis).filter(Hypothesis.investigation_id == inv1.id, Hypothesis.hypothesis_code == "HYP-01").first()
        h1_2 = db.query(Hypothesis).filter(Hypothesis.investigation_id == inv2.id, Hypothesis.hypothesis_code == "HYP-01").first()
        exps1 = [e.test_code for e in db.query(Experiment).filter(Experiment.investigation_id == inv1.id).all()]
        exps2 = [e.test_code for e in db.query(Experiment).filter(Experiment.investigation_id == inv2.id).all()]
        db.close()

        self.assertIsNotNone(h1_1)
        self.assertIsNotNone(h1_2)
        self.assertAlmostEqual(h1_1.posterior_probability, h1_2.posterior_probability, places=4)
        self.assertEqual(set(exps1), set(exps2))

    def test_08_group_dimension_generalization(self):
        """Validates semantic resolution across diverse group column names."""
        engine = SemanticEngine()
        for col_name in ["customer_class", "market_band", "service_level", "account_group"]:
            df = pd.DataFrame({
                "account_id": [f"A{i}" for i in range(12)],
                col_name: ["Enterprise", "SMB", "MidMarket"] * 4,
                "churn_event": [0, 1, 0] * 4,
            })
            intent = IntentEngine.parse_intent(f"What is the churn rate by {col_name}?")
            sem = engine.resolve_schema(intent, {"dataset": df})
            self.assertEqual(sem.group_dimension_col, col_name)

    def test_09_identifier_exclusion_from_group_dimension(self):
        """Validates that identifier columns are excluded from being selected as the group dimension."""
        df = pd.DataFrame({
            "account_id": [f"ID-{i:04d}" for i in range(100)],
            "user_uuid": [f"UUID-{i:04d}" for i in range(100)],
            "service_tier": ["Free"]*50 + ["Paid"]*50,
            "churn_event": [0]*50 + [1]*50,
        })
        intent = IntentEngine.parse_intent("What is the churn rate by segment?")
        engine = SemanticEngine()
        sem = engine.resolve_schema(intent, {"dataset": df})
        self.assertNotEqual(sem.group_dimension_col, "account_id")
        self.assertNotEqual(sem.group_dimension_col, "user_uuid")
        self.assertEqual(sem.group_dimension_col, "service_tier")

    def test_10_confounder_discovery_generalization(self):
        """Tests confounder identification across semantically equivalent concepts."""
        df = pd.DataFrame({
            "account_id": [f"ID-{i}" for i in range(100)],
            "tier": ["A"]*50 + ["B"]*50,
            "signup_vintage": ["2023"]*50 + ["2024"]*50,
            "onboarding_wave": ["W1"]*50 + ["W2"]*50,
            "account_tenure": [10]*50 + [20]*50,
            "churn_event": [0]*50 + [1]*50,
        })
        intent = IntentEngine.parse_intent("What is the churn rate by tier?")
        sem = SemanticEngine().resolve_schema(intent, {"dataset": df})
        self.assertTrue(len(sem.churn_confounder_cols) >= 1)
        self.assertIn("signup_vintage", sem.churn_confounder_cols)

    def test_11_persisted_evidence_traceability_reproducibility(self):
        """Asserts complete relational provenance chain strictly scoped to current investigation."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "cust_id": [f"C{i}" for i in range(200)],
            "tier": ["Gold"]*100 + ["Silver"]*100,
            "cohort": ["2024-Q1"]*100 + ["2024-Q2"]*100,
            "churn_event": list(rng.uniform(size=100) < 0.35) + list(rng.uniform(size=100) < 0.05),
        })
        df["churn_event"] = df["churn_event"].astype(int)

        inv = _run_calibrated_investigation(df, "Which tier has higher churn rate?")
        self.assertEqual(inv.status, "COMPLETED")

        db = SessionLocal()
        # 1. Hypotheses strictly scoped to this investigation
        hyps = db.query(Hypothesis).filter(Hypothesis.investigation_id == inv.id).all()
        self.assertGreaterEqual(len(hyps), 3, "Expected at least 3 competing hypotheses")

        # 2. Experiments strictly scoped to this investigation
        exps = db.query(Experiment).filter(Experiment.investigation_id == inv.id).all()
        self.assertGreaterEqual(len(exps), 1, "Expected at least 1 experiment executed")
        exp_ids = [e.id for e in exps]

        # 3. Observations strictly scoped to this investigation's experiments via foreign key
        obs = db.query(Observation).filter(Observation.experiment_id.in_(exp_ids)).all()
        self.assertGreaterEqual(len(obs), 1, "Expected at least 1 observation record")

        # 4. Evidence strictly scoped to this investigation
        evs = db.query(Evidence).filter(Evidence.investigation_id == inv.id).all()
        self.assertGreaterEqual(len(evs), 1, "Expected at least 1 evidence record")
        ev_ids = [ev.id for ev in evs]

        # 5. EvidenceVerifications strictly scoped to this investigation's evidence via foreign key
        verifs = db.query(EvidenceVerification).filter(EvidenceVerification.evidence_id.in_(ev_ids)).all()
        self.assertGreaterEqual(len(verifs), 1, "Expected at least 1 verification proof")

        # 6. BeliefUpdates strictly scoped to this investigation
        belief_updates = db.query(BeliefUpdate).filter(BeliefUpdate.investigation_id == inv.id).all()
        self.assertGreaterEqual(len(belief_updates), 1, "Expected at least 1 belief update record")

        # 7. InvestigationVerdict strictly scoped to this investigation
        verdict = db.query(InvestigationVerdict).filter(InvestigationVerdict.investigation_id == inv.id).first()
        self.assertIsNotNone(verdict, "InvestigationVerdict entity must be persisted")
        self.assertEqual(verdict.investigation_id, inv.id)
        self.assertEqual(verdict.verdict_type, inv.verdict_type)

        # 8. Prove the verification belongs to the evidence and downstream chain
        primary_ev = evs[0]
        matching_verif = db.query(EvidenceVerification).filter(EvidenceVerification.evidence_id == primary_ev.id).first()
        self.assertIsNotNone(matching_verif, "Verification record must exist for primary evidence")
        self.assertEqual(matching_verif.status, "VERIFIED")
        self.assertEqual(matching_verif.primary_tool, "duckdb_sql")
        self.assertEqual(matching_verif.secondary_tool, "polars_vectorized")
        self.assertLessEqual(matching_verif.observed_delta_pct, 5.0)

        # 9. Prove belief update connects verified evidence to hypothesis
        matching_bu = db.query(BeliefUpdate).filter(
            BeliefUpdate.investigation_id == inv.id,
            BeliefUpdate.evidence_id == primary_ev.id
        ).first()
        self.assertIsNotNone(matching_bu, "BeliefUpdate must reference the verified evidence ID")
        self.assertIn(matching_bu.hypothesis_id, [h.id for h in hyps])
        db.close()

    def test_12_strong_raw_effect_collapses_after_stratification(self):
        """Negative calibration: 30% crude gap collapsing in strata forces INCONCLUSIVE and prevents false positive."""
        rows = []
        cid = 1
        for seg, cohort, n, churn_prob in [
            ("Tier-A", "EarlyCohort", 300, 0.40),
            ("Tier-A", "LateCohort", 50, 0.10),
            ("Tier-B", "EarlyCohort", 50, 0.40),
            ("Tier-B", "LateCohort", 300, 0.10),
        ]:
            rng = np.random.RandomState(cid)
            for _ in range(n):
                churn = int(rng.uniform() < churn_prob)
                rows.append({
                    "account_id": f"ACC-{cid:05d}",
                    "segment": seg,
                    "signup_cohort": cohort,
                    "churn_event": churn,
                })
                cid += 1
        df = pd.DataFrame(rows)

        # 1. Crude difference is large (35.7% vs 14.3%)
        crude_a = df[df["segment"] == "Tier-A"]["churn_event"].mean()
        crude_b = df[df["segment"] == "Tier-B"]["churn_event"].mean()
        self.assertGreater(crude_a - crude_b, 0.10, "Crude difference must be substantial before stratification")

        # 2. Statistical identifiability captures Simpson's paradox
        res = analyze_churn_identifiability(df, group_col="segment", known_confounders=["signup_cohort"])
        self.assertEqual(res.verdict, ChurnVerdict.CONFOUNDED_IDENTIFIABILITY_LIMITED)

        # 3. Transition likelihoods penalize H1 and boost H3
        base_likes = _compute_churn_transition_likelihoods(verdict=res.verdict, warnings=res.warnings)
        self.assertEqual(base_likes[0], 0.15, "H1 (Segment Diff) must receive low likelihood under confounding")
        self.assertEqual(base_likes[2], 0.85, "H3 (Confounded) must receive high likelihood under confounding")

        # 4. Controller execution terminates at INCONCLUSIVE with zero confidence
        inv = _run_calibrated_investigation(df, "Which segment has higher churn rate?")
        self.assertEqual(inv.status, "COMPLETED")
        self.assertEqual(str(inv.verdict_type), "INCONCLUSIVE")
        self.assertEqual(inv.confidence_score, 0.0)

    def test_13_fail_closed_edge_cases(self):
        """Verifies fail-closed behavior under missing churn outcome."""
        df = pd.DataFrame({
            "user_id": [f"U-{i}" for i in range(100)],
            "is_vip": [1]*50 + [0]*50,
            "is_subscriber": [0]*50 + [1]*50,
            "score": np.random.uniform(10, 100, size=100),
        })
        inv = _run_calibrated_investigation(df, "What is the churn rate by segment?")
        self.assertEqual(inv.status, "COMPLETED")
        self.assertEqual(str(inv.verdict_type), "INCONCLUSIVE")
        self.assertEqual(inv.confidence_score, 0.0)

    def test_14_ambiguous_group_dimensions_fail_closed(self):
        """Dataset with multiple legitimate categoricals fails closed on ambiguous query; resolves on explicit query."""
        df = pd.DataFrame({
            "account_id": [f"ACC-{i:04d}" for i in range(120)],
            "segment": ["Enterprise", "SMB"] * 60,
            "region": ["US", "EU", "APAC"] * 40,
            "plan": ["Pro", "Basic"] * 60,
            "customer_type": ["B2B", "B2C"] * 60,
            "churn_event": [0, 1] * 60,
        })
        # 1. Ambiguous question without named group dimension
        intent_ambig = IntentEngine.parse_intent("Which customers have higher churn?")
        sem_ambig = SemanticEngine().resolve_schema(intent_ambig, {"cust_data": df})
        self.assertIsNone(sem_ambig.group_dimension_col, "Ambiguous question must not guess a group dimension")
        
        inv_ambig = _run_calibrated_investigation(df, "Which customers have higher churn?")
        self.assertEqual(inv_ambig.status, "COMPLETED")
        self.assertEqual(str(inv_ambig.verdict_type), "INCONCLUSIVE")

        # 2. Explicit question naming region
        intent_expl = IntentEngine.parse_intent("What is churn by region?")
        sem_expl = SemanticEngine().resolve_schema(intent_expl, {"cust_data": df})
        self.assertEqual(sem_expl.group_dimension_col, "region", "Explicit question must select named dimension")

    def test_15_confounder_vs_group_separation(self):
        """Requested group dimension is never added to churn_confounder_cols."""
        df = pd.DataFrame({
            "account_id": [f"ID-{i}" for i in range(100)],
            "region": ["East"]*50 + ["West"]*50,
            "signup_vintage": ["2023"]*50 + ["2024"]*50,
            "customer_type": ["SMB"]*50 + ["Enterprise"]*50,
            "churn_event": [0]*50 + [1]*50,
        })
        intent = IntentEngine.parse_intent("What is the churn rate by region?")
        sem = SemanticEngine().resolve_schema(intent, {"dataset": df})
        self.assertEqual(sem.group_dimension_col, "region")
        self.assertNotIn("region", sem.churn_confounder_cols, "Group dimension must not be treated as its own confounder")
        self.assertIn("signup_vintage", sem.churn_confounder_cols)

    def test_16_semantic_alias_negative_cases(self):
        """Rejects non-binary churn fields such as reasons, notes, scores, offers."""
        df = pd.DataFrame({
            "customer_id": [f"C-{i}" for i in range(50)],
            "cancel_reason": ["Price", "Competitor"] * 25,
            "churn_score": np.random.uniform(0.1, 0.9, size=50),
            "retention_offer": ["Discount", "Upgrade"] * 25,
            "attrition_note": ["Called support", "No response"] * 25,
            "customer_status": ["Active", "Paused", "Trial"] * 16 + ["Active", "Paused"],
        })
        intent = IntentEngine.parse_intent("What is the churn rate by status?")
        sem = SemanticEngine().resolve_schema(intent, {"dataset": df})
        self.assertIsNone(sem.churn_event_col, "Non-binary churn fields must not be classified as churn events")
        self.assertFalse(sem.churn_outcome_available)

    def test_17_bayesian_likelihood_monotonicity(self):
        """Verifies strictly monotonic posterior growth under progressively stronger hypothesis likelihoods."""
        prior = [0.25, 0.25, 0.25, 0.25]
        
        # Weak evidence
        post_weak, _ = BeliefEngine.compute_bayesian_posteriors(prior, [0.40, 0.20, 0.20, 0.20])
        # Moderate evidence
        post_mod, _ = BeliefEngine.compute_bayesian_posteriors(prior, [0.65, 0.15, 0.10, 0.10])
        # Strong evidence
        post_strong, _ = BeliefEngine.compute_bayesian_posteriors(prior, [0.85, 0.05, 0.05, 0.05])

        self.assertAlmostEqual(post_weak[0], 0.40, places=4)
        self.assertAlmostEqual(post_mod[0], 0.65, places=4)
        self.assertAlmostEqual(post_strong[0], 0.85, places=4)
        self.assertLess(post_weak[0], post_mod[0])
        self.assertLess(post_mod[0], post_strong[0])

    def test_18_bayesian_calibration_weak_moderate_strong_empirical(self):
        """Audits empirical weak, moderate, and strong datasets through the statistical identifiability pipeline."""
        rng = np.random.RandomState(42)

        # 1. Weak: N=60, modest effect
        df_weak = pd.DataFrame({
            "segment": ["A"]*30 + ["B"]*30,
            "churn_event": [1]*6 + [0]*24 + [1]*3 + [0]*27
        })
        res_weak = analyze_churn_identifiability(df_weak, group_col="segment")
        self.assertEqual(res_weak.verdict, ChurnVerdict.INSUFFICIENT_EVIDENCE)
        self.assertGreater(res_weak.aggregate_association["p_value"], 0.05)

        # 2. Moderate: N=300, moderate effect
        df_mod = pd.DataFrame({
            "segment": ["A"]*150 + ["B"]*150,
            "churn_event": list(rng.uniform(size=150) < 0.35) + list(rng.uniform(size=150) < 0.15)
        })
        df_mod["churn_event"] = df_mod["churn_event"].astype(int)
        res_mod = analyze_churn_identifiability(df_mod, group_col="segment")
        self.assertEqual(res_mod.verdict, ChurnVerdict.OBSERVED_ASSOCIATION)
        self.assertLess(res_mod.aggregate_association["p_value"], 1e-4)

        # 3. Strong: N=2000, large effect
        df_strong = pd.DataFrame({
            "segment": ["A"]*1000 + ["B"]*1000,
            "churn_event": list(rng.uniform(size=1000) < 0.45) + list(rng.uniform(size=1000) < 0.05)
        })
        df_strong["churn_event"] = df_strong["churn_event"].astype(int)
        res_strong = analyze_churn_identifiability(df_strong, group_col="segment")
        self.assertEqual(res_strong.verdict, ChurnVerdict.OBSERVED_ASSOCIATION)
        self.assertLess(res_strong.aggregate_association["p_value"], 1e-50)

        # Compute posteriors across all 3 tiers under uniform prior
        priors = [0.25, 0.25, 0.25, 0.25]
        likes_weak = _compute_churn_transition_likelihoods(res_weak.verdict, res_weak.warnings)
        likes_mod = _compute_churn_transition_likelihoods(res_mod.verdict, res_mod.warnings)
        likes_strong = _compute_churn_transition_likelihoods(res_strong.verdict, res_strong.warnings)

        post_weak, _ = BeliefEngine.compute_bayesian_posteriors(priors, likes_weak)
        post_mod, _ = BeliefEngine.compute_bayesian_posteriors(priors, likes_mod)
        post_strong, _ = BeliefEngine.compute_bayesian_posteriors(priors, likes_strong)

        # Monotonicity check
        self.assertLess(post_weak[0], post_mod[0])
        self.assertAlmostEqual(post_weak[0], 0.193548, places=4)
        self.assertAlmostEqual(post_mod[0], 0.406250, places=4)

    def test_19_same_verdict_different_strength_limitation_audit(self):
        """Audits behavior when two datasets with different sample sizes and effect sizes produce the same ChurnVerdict."""
        rng = np.random.RandomState(42)
        
        # Dataset A: N=200, effect delta = 0.20
        df_a = pd.DataFrame({
            "segment": ["A"]*100 + ["B"]*100,
            "cohort": rng.choice(["2024-Q1", "2024-Q2"], size=200),
            "churn_event": list(rng.uniform(size=100) < 0.35) + list(rng.uniform(size=100) < 0.15)
        })
        df_a["churn_event"] = df_a["churn_event"].astype(int)
        res_a = analyze_churn_identifiability(df_a, group_col="segment", known_confounders=["cohort"])

        # Dataset B: N=5000, effect delta = 0.55
        df_b = pd.DataFrame({
            "segment": ["A"]*2500 + ["B"]*2500,
            "cohort": rng.choice(["2024-Q1", "2024-Q2"], size=5000),
            "churn_event": list(rng.uniform(size=2500) < 0.60) + list(rng.uniform(size=2500) < 0.05)
        })
        df_b["churn_event"] = df_b["churn_event"].astype(int)
        res_b = analyze_churn_identifiability(df_b, group_col="segment", known_confounders=["cohort"])

        self.assertEqual(res_a.verdict, ChurnVerdict.SUPPORTED_SEGMENT_DIFFERENCE)
        self.assertEqual(res_b.verdict, ChurnVerdict.SUPPORTED_SEGMENT_DIFFERENCE)

        # Audit finding: Base likelihoods assigned are identical (0.85) because of categorical mapping
        likes_a = _compute_churn_transition_likelihoods(res_a.verdict, res_a.warnings)
        likes_b = _compute_churn_transition_likelihoods(res_b.verdict, res_b.warnings)
        self.assertEqual(likes_a, likes_b, "Documented limitation: categorical verdict mapping produces identical base likelihoods")

    def test_20_semantic_grouping_does_not_leak_into_causal_confirmation(self):
        """Predictive association on requested dimension does not claim causation."""
        rng = np.random.RandomState(42)
        n = 200
        df = pd.DataFrame({
            "region": ["East"]*n + ["West"]*n,
            "plan": rng.choice(["Basic", "Pro"], size=2*n),
            "customer_type": rng.choice(["B2B", "B2C"], size=2*n),
            "channel": rng.choice(["Direct", "Partner"], size=2*n),
            "churn_event": list(rng.uniform(size=n) < 0.40) + list(rng.uniform(size=n) < 0.10)
        })
        df["churn_event"] = df["churn_event"].astype(int)

        intent = IntentEngine.parse_intent("What is churn by region?")
        sem = SemanticEngine().resolve_schema(intent, {"dataset": df})
        self.assertEqual(sem.group_dimension_col, "region")

        inv = _run_calibrated_investigation(df, "What is churn by region?")
        self.assertEqual(inv.status, "COMPLETED")
        
        # Narrative must use honest associative terminology and never claim proven causation
        narrative = f"{inv.direct_answer or ''} {inv.main_finding or ''}".lower()
        self.assertNotIn("proves that region causes", narrative)
        self.assertNotIn("causally determined by", narrative)


if __name__ == "__main__":
    unittest.main()
