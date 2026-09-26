"""Build 13 Core Acceptance Suite: Autonomous Experimentation Loop, EIG Optimization & Multiverse Analysis."""
import os
import sys
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from apps.api.src.models.entities import (
    Base,
    User,
    Organization,
    Project,
    Dataset,
    DatasetVersion,
    Investigation,
    InvestigationObjective,
    Hypothesis,
    Experiment,
    Evidence,
    EvidenceVerification,
    BeliefUpdate,
    InvestigationVerdict,
    gen_uuid,
)
from packages.analytics_core.src.execution.queue import DatabaseQueueProvider
from packages.analytics_core.src.execution.state_machine import InvestigationState
from packages.analytics_core.src.engines.intent import IntentEngine
from packages.analytics_core.src.engines.semantic import SemanticEngine
from packages.analytics_core.src.engines.dataset_provider import DatabaseDatasetProvider
from packages.analytics_core.src.intelligence.predictive_hypothesis import HypothesisSynthesizer, PredictiveHypothesis
from packages.analytics_core.src.intelligence.experiment_synthesizer import ExperimentSynthesizer
from packages.analytics_core.src.intelligence.eig_optimizer import EIGOptimizer
from packages.analytics_core.src.intelligence.adversarial_attacker import AdversarialAttacker
from packages.analytics_core.src.intelligence.multiverse_engine import MultiverseEngine
from packages.analytics_core.src.intelligence.epistemic_calibration import EpistemicCalibrationEngine
from packages.analytics_core.src.runtime.controller import InvestigationController
from packages.analytics_core.src.execution.worker import InvestigationWorker


def run_build13_tests():
    print("=" * 80)
    print("RUNNING AA-OS CORE / BUILD 13: AUTONOMOUS EXPERIMENTATION & EIG ACCEPTANCE SUITE")
    print("=" * 80)

    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_build13_core.db"))
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except Exception:
            pass

    test_engine = create_engine(
        f"sqlite:///{db_path}?timeout=30",
        connect_args={"check_same_thread": False},
        echo=False,
    )
    Base.metadata.create_all(bind=test_engine)
    TestingSession = sessionmaker(bind=test_engine)
    db = TestingSession()

    # Seed base tenant
    org = Organization(id="org-b13", name="Build 13 Autonomous Org", slug="b13-org")
    user = User(id="user-b13", email="core@aaos.ai", hashed_password="pw", full_name="AA-OS Core Engineer", organization_id=org.id)
    proj = Project(id="proj-b13", org_id=org.id, owner_id=user.id, name="AA-OS Build 13 Core Project")
    db.add_all([org, user, proj])
    db.commit()

    # Create unfamiliar multi-dimensional business dataset: Cloud Compute Billing
    storage_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "storage", "datasets", "proj-b13"))
    os.makedirs(storage_dir, exist_ok=True)
    parquet_path = os.path.join(storage_dir, "cloud_infrastructure_cost.parquet")
    df_cloud = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=500, freq="h"),
        "instance_family": ["gpu-accelerated", "memory-optimized", "general-compute", "storage-dense", "network-enhanced"] * 100,
        "region": ["us-east-1", "eu-central-1", "ap-southeast-1", "us-west-2"] * 125,
        "cost_usd": [50.0 + (i % 5) * 120.0 + (i * 0.15) for i in range(500)],
        "utilization_pct": [0.30 + (i % 4) * 0.15 for i in range(500)],
    })
    df_cloud.to_parquet(parquet_path)

    ds = Dataset(id="ds-cloud-01", project_id=proj.id, name="cloud_infrastructure_cost", current_version=1, row_count=500, column_count=5, format="parquet")
    ds_ver = DatasetVersion(id=gen_uuid(), dataset_id=ds.id, version_number=1, file_path=parquet_path, row_count=500)
    db.add_all([ds, ds_ver])
    db.commit()

    # ------------------------------------------------------------------------
    # [TEST 1] Predictive Hypotheses Synthesis from Semantic World Model
    # ------------------------------------------------------------------------
    print("\n[TEST 1] Testing Predictive Hypotheses Synthesis...")
    intent = IntentEngine.parse_intent("Why did cost_usd surge across instance_family?")
    semantic = SemanticEngine().resolve_schema(intent, {"cloud_infrastructure_cost": df_cloud})
    hyps = HypothesisSynthesizer.synthesize_competing_hypotheses(semantic, intent.raw_question)
    
    assert len(hyps) >= 2
    assert hyps[0].falsification_criteria is not None
    assert len(hyps[0].predicted_observables_if_true) >= 1
    assert abs(sum(h.prior_probability for h in hyps) - 1.0) < 1e-4
    print(f" -> PASSED: Synthesized {len(hyps)} predictive hypotheses with falsification boundaries & sum(priors)=1.0.")

    # ------------------------------------------------------------------------
    # [TEST 2] Candidate Experiment Pool Synthesis
    # ------------------------------------------------------------------------
    print("\n[TEST 2] Testing Candidate Experiment Pool Synthesis...")
    candidate_exps = ExperimentSynthesizer.synthesize_candidate_experiments(hyps, semantic)
    assert len(candidate_exps) >= 3
    assert any(e.code == "EXP-CONC" for e in candidate_exps)
    assert any(e.code == "EXP-UNIF" for e in candidate_exps)
    assert any(e.code == "EXP-ANOVA" for e in candidate_exps)
    print(f" -> PASSED: Synthesized {len(candidate_exps)} candidate tests with discriminating weights.")

    # ------------------------------------------------------------------------
    # [TEST 3] Expected Information Gain (EIG) & Utility Ranking
    # ------------------------------------------------------------------------
    print("\n[TEST 3] Testing EIG & Multi-Objective Utility Optimization...")
    scored = EIGOptimizer.score_candidate_experiments(candidate_exps, hyps, executed_codes=[])
    assert len(scored) == len(candidate_exps)
    assert scored[0].utility_score >= scored[-1].utility_score
    top_exp, rationale = EIGOptimizer.select_next_experiment(candidate_exps, hyps, executed_codes=[])
    assert top_exp is not None
    print(f" -> PASSED: EIG Optimizer ranked {len(scored)} tests. Top choice: {top_exp.code} ({top_exp.description}).")

    # ------------------------------------------------------------------------
    # [TEST 4] Active Adversarial Attacker Falsification
    # ------------------------------------------------------------------------
    print("\n[TEST 4] Testing Active Adversarial Attacker...")
    attack_res = AdversarialAttacker.design_attack(hyps[0], hyps[1:], df_cloud)
    assert attack_res.attack_status in ["SURVIVED", "WEAKENED", "REFUTED", "CONTRADICTED"]
    assert attack_res.discriminating_test_sql is not None
    print(f" -> PASSED: Adversarial Attack executed against {hyps[0].hypothesis_code} -> Status: {attack_res.attack_status}.")

    # ------------------------------------------------------------------------
    # [TEST 5] Multiverse Specification Curve Robustness
    # ------------------------------------------------------------------------
    print("\n[TEST 5] Testing Multiverse Specification Curve Engine...")
    mv_report = MultiverseEngine.evaluate_specification_curve(df_cloud, "instance_family", "cost_usd")
    assert mv_report.total_specifications >= 3
    assert 0.0 <= mv_report.robustness_pct <= 100.0
    print(f" -> PASSED: {mv_report.epistemic_summary}")

    # ------------------------------------------------------------------------
    # [TEST 6] 8 Disentangled Epistemic Vectors
    # ------------------------------------------------------------------------
    print("\n[TEST 6] Testing 8 Disentangled Epistemic Vectors...")
    vectors = EpistemicCalibrationEngine.calibrate_vectors(
        df=df_cloud,
        posterior_probability=0.88,
        verification_delta_pct=0.00001,
        variance_explained_pct=85.4,
        multiverse_robustness_pct=mv_report.robustness_pct,
    )
    assert 0.0 <= vectors.evidence_strength <= 1.0
    assert 0.0 <= vectors.data_quality <= 1.0
    assert 0.0 <= vectors.statistical_uncertainty <= 1.0
    assert 0.0 <= vectors.model_uncertainty <= 1.0
    assert 0.0 <= vectors.hypothesis_belief <= 1.0
    print(f" -> PASSED: Calibrated 8 epistemic vectors (Evidence: {vectors.evidence_strength:.2f}, Belief: {vectors.hypothesis_belief:.2f}, Robustness: {vectors.multiverse_robustness_pct:.1f}%).")

    # ------------------------------------------------------------------------
    # [TEST 7] End-to-End Autonomous Investigation Loop on Unfamiliar Dataset
    # ------------------------------------------------------------------------
    print("\n[TEST 7] Testing End-to-End Autonomous Investigation Loop (Build 13 North Star)...")
    queue = DatabaseQueueProvider(session_factory=TestingSession)
    controller = InvestigationController(session_factory=TestingSession)
    worker = InvestigationWorker(worker_id="worker-b13-northstar", queue_provider=queue, session_factory=TestingSession, controller=controller)

    inv = Investigation(
        id="INV-B13-NORTHSTAR",
        project_id=proj.id,
        user_id=user.id,
        question="Why did cost_usd surge across instance_family?",
        status=InvestigationState.QUEUED,
    )
    db.add(inv)
    db.commit()

    queue.enqueue(investigation_id="INV-B13-NORTHSTAR", priority=100)
    
    # Run pure worker loop
    processed = worker.process_next_job()
    assert processed is True

    db.expire_all()
    inv_res = db.query(Investigation).filter(Investigation.id == "INV-B13-NORTHSTAR").first()
    assert inv_res.status == InvestigationState.COMPLETED
    assert inv_res.verdict_type in ["DIAGNOSED", "OBSERVED", "STATISTICALLY_SIGNIFICANT", "INCONCLUSIVE"]
    assert len(inv_res.reproducible_manifest_hash) == 64
    assert "Multiverse Robustness" in inv_res.direct_answer

    # Verify that autonomous loop executed experiments and produced verified evidence
    executed_experiments = db.query(Experiment).filter(Experiment.investigation_id == "INV-B13-NORTHSTAR").all()
    assert len(executed_experiments) >= 1

    verified_evidences = db.query(Evidence).filter(Evidence.investigation_id == "INV-B13-NORTHSTAR").all()
    assert len(verified_evidences) >= 1
    assert all(e.validation_status == "VERIFIED" for e in verified_evidences)

    print(f" -> PASSED: Autonomous Investigation DAG concluded independently in {len(executed_experiments)} steps.")
    print(f"    Direct Answer: {inv_res.direct_answer[:120]}...")
    print(f"    Manifest Hash: {inv_res.reproducible_manifest_hash[:16]}...")

    print("\n" + "=" * 80)
    print("ALL 7 BUILD 13 AUTONOMOUS CORE ACCEPTANCE TESTS PASSED WITH 100% INTEGRITY")
    print("=" * 80)


if __name__ == "__main__":
    run_build13_tests()
