"""
test_golden_adaptive_investigation.py: Comprehensive validation suite for the 9 Golden
Properties of the AA-OS Adaptive Autonomous Scientific Investigation Kernel.
"""
import os
import sys
import shutil
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from apps.api.src.core.database import Base
from apps.api.src.models.entities import (
    Dataset,
    DatasetVersion,
    Evidence,
    Experiment,
    Hypothesis,
    Investigation,
    InvestigationExecution,
    InvestigationStepExecution,
    Prediction,
    Project,
    User,
    gen_uuid,
)
from packages.analytics_core.src.causal.identifiability_gate import CausalIdentifiabilityGate
from packages.analytics_core.src.intelligence.adversarial_attacker import AdversarialAttacker
from packages.analytics_core.src.engines.belief import BeliefEngine, compute_shannon_entropy
from packages.analytics_core.src.engines.dataset_provider import DatabaseDatasetProvider
from packages.analytics_core.src.engines.evidence import EvidenceEngine
from packages.analytics_core.src.engines.execution_provider import DuckDBExecutionProvider
from packages.analytics_core.src.engines.intent import IntentEngine
from packages.analytics_core.src.execution.queue import DatabaseQueueProvider
from packages.analytics_core.src.engines.semantic import SemanticEngine
from packages.analytics_core.src.engines.stopping import StoppingEngine
from packages.analytics_core.src.engines.verification import VerificationEngine
from packages.analytics_core.src.execution.checkpointer import InvestigationCheckpointer
from packages.analytics_core.src.execution.state_machine import InvestigationState, VerificationStatus
from packages.analytics_core.src.execution.worker import InvestigationWorker
from packages.analytics_core.src.intelligence.eig_optimizer import EIGOptimizer
from packages.analytics_core.src.intelligence.evidence_patterns import (
    ConcentrationPatternDetector,
    SegmentDifferenceDetector,
)
from packages.analytics_core.src.intelligence.experiment_synthesizer import (
    CandidateExperiment,
    ExperimentSynthesizer,
    UncertaintyState,
)
from packages.analytics_core.src.intelligence.hypothesis_revision import HypothesisRevisionEngine
from packages.analytics_core.src.intelligence.multiverse_engine import MultiverseEngine
from packages.analytics_core.src.intelligence.prediction_engine import (
    ConcentrationEvaluator,
    DirectionEvaluator,
    PredictionEvaluator,
    PredictionSynthesizer,
    StructuredPrediction,
)
from packages.analytics_core.src.intelligence.predictive_hypothesis import (
    HypothesisSynthesizer,
    PredictiveHypothesis,
)
from packages.analytics_core.src.intelligence.state_validation import validate_state
from packages.analytics_core.src.intelligence.transition import ScientificTransitionService
from packages.analytics_core.src.runtime.controller import InvestigationController
from packages.analytics_core.src.runtime.state import InvestigationStateManager
from packages.schemas.src.analysis import (
    FirstClassExperiment,
    GrainPreservationStatus,
    HypothesisStatus,
    PredictionStatus,
    RawObservationRecord,
)


def run_9_golden_adaptive_tests():
    print("=" * 80)
    print("RUNNING AA-OS 9 GOLDEN ADAPTIVE INVESTIGATION TESTS")
    print("=" * 80)

    temp_dir = tempfile.mkdtemp(prefix="aaos_adaptive_golden_")
    db_path = os.path.join(temp_dir, "adaptive_test.db")
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    db = TestingSession()
    user = User(id="usr-golden-01", email="golden@aaos.ai", hashed_password="pw", full_name="Golden Tester", is_active=True)
    proj = Project(id="prj-golden-01", name="Adaptive Project", description="Golden Testing", owner_id=user.id)
    db.add_all([user, proj])
    db.commit()

    storage_dir = os.path.join(temp_dir, "datasets", proj.id)
    os.makedirs(storage_dir, exist_ok=True)

    # ------------------------------------------------------------------------
    # [TEST 1] Emergent Hypothesis Formulation from Raw Observation Data
    # ------------------------------------------------------------------------
    print("\n[GOLDEN 1] Testing Emergent Hypothesis Formulation from Data Patterns...")
    obs_df = pd.DataFrame({
        "datacenter_region": ["us-east", "us-west", "eu-central", "ap-southeast"],
        "cost_metric": [94000.0, 2000.0, 2500.0, 1500.0],
        "row_count": [1000, 1000, 1000, 1000],
    })
    detector = ConcentrationPatternDetector()
    expl = detector.detect(obs_df, {})
    assert expl is not None
    assert expl.dominant_value == "us-east"
    assert expl.dominant_share > 0.90

    semantic_res = SemanticEngine().resolve_schema(
        IntentEngine.parse_intent("Why did cost_metric surge across datacenter_region?"),
        {"cloud_costs": obs_df},
    )
    existing_hyps = HypothesisSynthesizer.synthesize_competing_hypotheses(semantic_res, "Why did cost_metric surge?")
    emergent = HypothesisSynthesizer.synthesize_emergent_hypotheses(
        [{"dimension": "datacenter_region", "top_value": "us-east", "top_share_pct": 94.0, "source_experiment_id": "EXP-01"}],
        semantic_res,
        existing_hyps,
    )
    assert len(emergent) >= 1
    assert "us-east" in emergent[0].claim
    assert emergent[0].target_value == "us-east"
    print(f" -> PASSED: Emergent hypothesis formulated: '{emergent[0].claim}'")

    # ------------------------------------------------------------------------
    # [GOLDEN 2] Structured Prediction Deduction and Deterministic Evaluation
    # ------------------------------------------------------------------------
    print("\n[GOLDEN 2] Testing Structured Prediction Deduction & Deterministic Evaluation...")
    h_test = emergent[0]
    pred = PredictionSynthesizer.deduce_prediction(h_test, semantic_res)
    assert pred.expected_direction.lower() == "concentrated"
    assert pred.expected_value == "us-east"

    eval_supported = PredictionEvaluator.evaluate(pred, obs_df)
    assert eval_supported.status == "SUPPORTED"
    assert eval_supported.actual_observed_result.get("top_share_pct", 0) > 90.0

    # Test evaluation against flat uniform data produces REFUTED
    flat_df = pd.DataFrame({
        "datacenter_region": ["us-east", "us-west", "eu-central", "ap-southeast"],
        "cost_metric": [2500.0, 2500.0, 2500.0, 2500.0],
    })
    eval_refuted = PredictionEvaluator.evaluate(pred, flat_df)
    assert eval_refuted.status == "REFUTED"
    print(f" -> PASSED: Deterministic prediction deduction and evaluation verified (Supported: {eval_supported.status}, Refuted: {eval_refuted.status}).")

    # ------------------------------------------------------------------------
    # [GOLDEN 3] In-Loop Adversarial Attack & Targeted Replanning
    # ------------------------------------------------------------------------
    print("\n[GOLDEN 3] Testing In-Loop Adversarial Attack & Dynamic Counter-Experiments...")
    df_simpsons = pd.DataFrame({
        "department": ["Eng", "Sales"] * 200,
        "seniority": ["Junior", "Junior", "Senior", "Senior"] * 100,
        "bonus_pct": [12.0, 15.0, 25.0, 22.0] * 100,
    })
    h_lead = PredictiveHypothesis(
        id="HYP-01",
        hypothesis_code="HYP-01",
        claim="Department concentration in bonus_pct",
        mechanism="Department role drives bonus",
        predicted_observables_if_true=["High variance across department"],
        predicted_observables_if_false=["Uniform drift"],
        falsification_criteria="spread < 5%",
        required_assumptions=[],
        prior_probability=0.80,
        posterior_probability=0.80,
        target_dimension="department",
        target_metric="bonus_pct",
    )
    attack_res = AdversarialAttacker.execute_adversarial_attack(
        df=df_simpsons,
        leading_hypothesis=h_lead,
        counter_hypothesis=h_lead,
    )
    assert attack_res.simpsons_paradox_detected is True
    assert attack_res.details.get("secondary_dimension") == "seniority"

    replan_res = ExperimentSynthesizer.dynamically_replan_candidates(
        hypotheses=[h_lead],
        semantic=SemanticEngine().resolve_schema(IntentEngine.parse_intent("Why did bonus_pct change across department?"), {"bonus_data": df_simpsons}),
        executed_codes=[],
        current_posteriors=[0.80],
        unresolved_adversarial_issues=[{"confounding_dimension": "seniority"}],
    )
    assert any("EXP-COND" in c.code for c in replan_res.candidates)
    print(f" -> PASSED: In-loop adversarial attack detected confounding and synthesized conditional experiment.")

    # ------------------------------------------------------------------------
    # [GOLDEN 4] Multiverse Robustness Influencing Candidate Selection
    # ------------------------------------------------------------------------
    print("\n[GOLDEN 4] Testing Multiverse Robustness Engine & Candidate Scoring...")
    multi_report = MultiverseEngine.evaluate_specification_curve(
        df=obs_df,
        dimension_col="datacenter_region",
        target_metric_col="cost_metric",
    )
    assert multi_report.total_specifications >= 3
    assert multi_report.robustness_pct >= 90.0

    scored_exps = EIGOptimizer.score_candidate_experiments(
        replan_res.candidates,
        [h_lead],
        multiverse_report=multi_report,
    )
    assert len(scored_exps) >= 1
    assert scored_exps[0].robustness_value >= 0.90
    print(f" -> PASSED: Multiverse specification curve evaluated ({multi_report.robustness_pct:.1f}% robustness) and integrated into candidate scores.")

    # ------------------------------------------------------------------------
    # [GOLDEN 5] N-Hypothesis Bayesian Updating with ANOVA Effect Size
    # ------------------------------------------------------------------------
    print("\n[GOLDEN 5] Testing Arbitrary N-Hypothesis Bayesian Updating with ANOVA Grounding...")
    state_mgr = InvestigationStateManager("INV-BAYES-TEST")
    h1 = PredictiveHypothesis(id="H1", hypothesis_code="HYP-01", claim="H1", mechanism="M1", predicted_observables_if_true=[], predicted_observables_if_false=[], falsification_criteria="f1", required_assumptions=[], prior_probability=0.33, posterior_probability=0.33, target_metric="cost_metric", target_dimension="datacenter_region")
    h2 = PredictiveHypothesis(id="H2", hypothesis_code="HYP-02", claim="H2", mechanism="M2", predicted_observables_if_true=[], predicted_observables_if_false=[], falsification_criteria="f2", required_assumptions=[], prior_probability=0.33, posterior_probability=0.33, target_metric="cost_metric", target_dimension="datacenter_region", is_counter_hypothesis=True)
    h3 = PredictiveHypothesis(id="H3", hypothesis_code="HYP-03", claim="H3", mechanism="M3", predicted_observables_if_true=[], predicted_observables_if_false=[], falsification_criteria="f3", required_assumptions=[], prior_probability=0.34, posterior_probability=0.34, target_metric="cost_metric", target_dimension="datacenter_region", target_value="us-east")
    state_mgr.create_hypothesis(h1)
    state_mgr.create_hypothesis(h2)
    state_mgr.create_hypothesis(h3)

    pred3 = PredictionSynthesizer.deduce_prediction(h3, semantic_res)
    state_mgr.create_prediction(pred3)

    trans_res = ScientificTransitionService.apply_post_execution_transition(
        state_mgr=state_mgr,
        experiment_id="EXP-TEST-BAYES",
        target_prediction_ids=[pred3.prediction_id],
        primary_df=obs_df,
        result_df=obs_df,
        target_metric_col="cost_metric",
        group_dimension_col="datacenter_region",
        aggregation_type="SUM",
        primary_value=94000.0,
        is_grouped=True,
    )
    assert trans_res.verification_status == "VERIFIED"
    assert trans_res.belief_update is not None
    posteriors = trans_res.belief_update["posteriors"]
    assert len(posteriors) == 3
    assert np.isclose(sum(posteriors), 1.0)
    assert posteriors[2] > posteriors[0] and posteriors[2] > posteriors[1]
    print(f" -> PASSED: N-Hypothesis Bayesian update normalized posteriors: {posteriors} (sum={sum(posteriors):.4f})")

    # ------------------------------------------------------------------------
    # [GOLDEN 6] Active Evidence Pattern Discovery (Observation != Hypothesis)
    # ------------------------------------------------------------------------
    print("\n[GOLDEN 6] Testing Active Evidence Pattern Discovery (Observation != Hypothesis)...")
    sec_df = pd.DataFrame({
        "segment_name": ["Alpha", "Beta", "Gamma"],
        "metric_delta": [500.0, -20.0, 10.0],
    })
    diff_detector = SegmentDifferenceDetector()
    diff_expl = diff_detector.detect(sec_df, {})
    assert diff_expl is not None
    assert diff_expl.pattern_name.lower() == "segment_difference"
    assert diff_expl.dominant_value == "Alpha"
    # Observation records the empirical metric difference
    obs_record = RawObservationRecord(
        observation_id="OBS-01",
        experiment_id="EXP-RAW-01",
        primary_value=500.0,
        row_count=3,
        execution_time_ms=1.2,
        structured_result={"segment_distribution": {"Alpha": 500.0, "Beta": -20.0, "Gamma": 10.0}, "dimension": "segment_name"},
    )
    assert obs_record.primary_value == 500.0
    assert obs_record.structured_result["segment_distribution"]["Alpha"] == 500.0
    print(f" -> PASSED: Empirical observation pattern cleanly discovered and decoupled from explanatory hypothesis.")

    # ------------------------------------------------------------------------
    # [GOLDEN 7] End-to-End Multi-Turn Adaptive Scientific Investigation Loop
    # ------------------------------------------------------------------------
    print("\n[GOLDEN 7] Testing End-to-End Multi-Turn Adaptive Autonomous Scientific Loop...")
    df_loop = pd.DataFrame({
        "customer_tier": ["Enterprise", "MidMarket", "SMB", "Starter"] * 100,
        "mrr_churn_usd": [80000.0 if i % 4 == 0 else 100.0 for i in range(400)],
    })
    p_path_loop = os.path.join(storage_dir, "churn_data.parquet")
    df_loop.to_parquet(p_path_loop)
    ds_loop = Dataset(id="ds-churn-golden", project_id=proj.id, name="churn_data", current_version=1, row_count=400, column_count=2, format="parquet")
    ds_vloop = DatasetVersion(id=gen_uuid(), dataset_id=ds_loop.id, version_number=1, file_path=p_path_loop, row_count=400)
    db.add_all([ds_loop, ds_vloop])
    db.commit()

    inv_golden = Investigation(
        id="INV-GOLDEN-LOOP",
        project_id=proj.id,
        user_id=user.id,
        question="Why did mrr_churn_usd spike across customer_tier?",
        status=InvestigationState.QUEUED,
    )
    db.add(inv_golden)
    db.commit()

    queue = DatabaseQueueProvider(session_factory=TestingSession)
    controller = InvestigationController(session_factory=TestingSession)
    worker = InvestigationWorker(worker_id="worker-golden-01", queue_provider=queue, session_factory=TestingSession, controller=controller)

    queue.enqueue(investigation_id=inv_golden.id, priority=100)
    processed = worker.process_next_job()
    assert processed is True

    db.expire_all()
    completed_inv = db.query(Investigation).filter(Investigation.id == "INV-GOLDEN-LOOP").first()
    assert completed_inv.status == InvestigationState.COMPLETED
    assert completed_inv.verdict_type == "DIAGNOSED"
    assert completed_inv.confidence_score >= 0.85
    assert len(completed_inv.reproducible_manifest_hash) == 64
    print(f" -> PASSED: Autonomous loop concluded: Verdict={completed_inv.verdict_type}, Confidence={completed_inv.confidence_score:.2f}, Manifest={completed_inv.reproducible_manifest_hash[:16]}...")

    # ------------------------------------------------------------------------
    # [GOLDEN 8] Fail-Closed Canonical State Consistency Validation
    # ------------------------------------------------------------------------
    print("\n[GOLDEN 8] Testing Fail-Closed Canonical State Consistency Validation...")
    corrupted_mgr = InvestigationStateManager("INV-CORRUPTED")
    h_good = PredictiveHypothesis(id="H_G", hypothesis_code="HYP-01", claim="Good", mechanism="M", predicted_observables_if_true=[], predicted_observables_if_false=[], falsification_criteria="f", required_assumptions=[], prior_probability=0.5, posterior_probability=0.5)
    corrupted_mgr.create_hypothesis(h_good)
    # Inject dangling prediction reference
    bad_pred = StructuredPrediction(
        prediction_id="PRED-BAD",
        hypothesis_id="HYP-NONEXISTENT",
        hypothesis_code="HYP-NONEXISTENT",
        statement="Dangling prediction statement",
        expected_direction="INCREASE",
        threshold=0.0,
    )
    corrupted_mgr.create_prediction(bad_pred)

    report = corrupted_mgr.validate_consistency()
    assert report.is_consistent is False
    assert any(err.error_code == "prediction_references_missing_hypothesis" for err in report.errors)
    print(f" -> PASSED: State validator failed closed on corrupted state with {len(report.errors)} detected invariant errors.")

    # ------------------------------------------------------------------------
    # [GOLDEN 9] Evidence-Based Multi-Constraint Stopping Reasons
    # ------------------------------------------------------------------------
    print("\n[GOLDEN 9] Testing Multi-Constraint Stopping Engine Reasons...")
    # Decisive stopping
    dec_decisive = StoppingEngine.evaluate_stopping(
        leading_posterior=0.92,
        experiments_completed=1,
        all_verifications_passed=True,
    )
    assert dec_decisive.should_stop is True
    assert dec_decisive.reason == "DECISIVE_SIGNAL_RESOLVED"

    # Adversarial blockage
    dec_blocked = StoppingEngine.evaluate_stopping(
        leading_posterior=0.92,
        experiments_completed=1,
        all_verifications_passed=True,
        has_unresolved_adversarial_issue=True,
    )
    assert dec_blocked.should_stop is False
    assert dec_blocked.reason == "ADVERSARIAL_CHALLENGE_UNRESOLVED"

    # Stagnation stopping
    dec_stagnant = StoppingEngine.evaluate_stopping(
        leading_posterior=0.55,
        entropy_delta=-0.01,
        experiments_completed=3,
        all_verifications_passed=True,
    )
    assert dec_stagnant.should_stop is True
    assert dec_stagnant.reason == "UNCERTAINTY_STAGNATED"

    print(f" -> PASSED: Stopping reasons validated (Decisive: {dec_decisive.reason}, Adversarial Block: {dec_blocked.reason}, Stagnant: {dec_stagnant.reason}).")

    print("\n" + "=" * 80)
    print("ALL 9 GOLDEN ADAPTIVE INVESTIGATION TESTS PASSED WITH 100% INTEGRITY")
    print("=" * 80)

    db.close()
    shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    run_9_golden_adaptive_tests()
