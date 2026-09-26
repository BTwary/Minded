"""Phase 3 Full Certification Acceptance Suite: Modular InvestigationRuntime & Epistemic Micro-Engines."""
import concurrent.futures
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta
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
    InvestigationJob,
    InvestigationExecution,
    InvestigationStepExecution,
    InvestigationObjective,
    Hypothesis,
    Experiment,
    Observation,
    Evidence,
    EvidenceVerification,
    BeliefUpdate,
    InvestigationVerdict,
    InvestigationGraphEdge,
    InvestigationEvent,
    gen_uuid,
)
from packages.analytics_core.src.execution.queue import DatabaseQueueProvider
from packages.analytics_core.src.execution.state_machine import (
    InvestigationState,
    JobState,
    ExecutionState,
    VerificationStatus,
    transition_investigation,
)
from packages.analytics_core.src.execution.checkpointer import InvestigationCheckpointer
from packages.analytics_core.src.engines.intent import IntentEngine, InvestigationIntent
from packages.analytics_core.src.engines.semantic import SemanticEngine, SemanticResolution
from packages.analytics_core.src.engines.hypothesis import HypothesisEngine, CandidateHypothesis
from packages.analytics_core.src.engines.experiment import ExperimentEngine, ExperimentSpecification
from packages.analytics_core.src.engines.execution_provider import BaseExecutionProvider, DuckDBExecutionProvider
from packages.analytics_core.src.engines.evidence import EvidenceEngine, StructuredEvidence
from packages.analytics_core.src.engines.verification import VerificationEngine, VerificationResult
from packages.analytics_core.src.engines.belief import BeliefEngine, compute_shannon_entropy
from packages.analytics_core.src.engines.adversarial import AdversarialEngine, AdversarialEvaluation
from packages.analytics_core.src.engines.stopping import StoppingEngine, StoppingDecision
from packages.analytics_core.src.engines.human_decision import HumanDecisionController, DecisionPolicy
from packages.analytics_core.src.engines.verdict import VerdictEngine, VerdictEvaluation
from packages.analytics_core.src.engines.provenance import ProvenanceEngine
from packages.analytics_core.src.runtime.controller import InvestigationController
from packages.analytics_core.src.execution.worker import InvestigationWorker


def run_phase3_tests():
    print("=" * 80)
    print("RUNNING PHASE 3: MODULAR RUNTIME DECOMPOSITION & EPISTEMIC ENGINES TEST SUITE")
    print("=" * 80)

    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_phase3_runtime.db"))
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
    org = Organization(id="org-p3", name="Phase 3 Org", slug="phase3-org")
    user = User(id="user-p3", email="architect@aaos.ai", hashed_password="pw", full_name="Principal Architect", organization_id=org.id)
    proj = Project(id="proj-p3", org_id=org.id, owner_id=user.id, name="Phase 3 Architecture Project")
    db.add_all([org, user, proj])
    db.commit()

    # Create real multi-dimensional test dataset
    storage_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "storage", "datasets", "proj-p3"))
    os.makedirs(storage_dir, exist_ok=True)
    parquet_path = os.path.join(storage_dir, "web_metrics.parquet")
    df_sample = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=300, freq="h"),
        "channel": ["organic", "paid_search", "direct", "referral", "social"] * 60,
        "conversions": [10.0 + (i % 5) * 20.0 + (i * 0.05) for i in range(300)],
    })
    df_sample.to_parquet(parquet_path)

    ds = Dataset(id="ds-web-01", project_id=proj.id, name="web_traffic", current_version=1, row_count=300, column_count=3, format="parquet")
    ds_ver = DatasetVersion(id=gen_uuid(), dataset_id=ds.id, version_number=1, file_path=parquet_path, row_count=300)
    db.add_all([ds, ds_ver])
    db.commit()

    # ------------------------------------------------------------------------
    # [TEST 1] IntentEngine Classification & Extraction
    # ------------------------------------------------------------------------
    print("\n[TEST 1] Testing IntentEngine Question Parsing...")
    intent_rc = IntentEngine.parse_intent("Why did conversions drop in social channel?")
    assert intent_rc.intent_type == "ROOT_CAUSE"
    assert "conversions" in intent_rc.keywords
    intent_fc = IntentEngine.parse_intent("Forecast next month's sales trajectory")
    assert intent_fc.intent_type == "FORECAST"
    intent_corr = IntentEngine.parse_intent("What is the correlation between latency and churn?")
    assert intent_corr.intent_type == "CORRELATION"
    print(f" -> PASSED: Intent classified as {intent_rc.intent_type} with keywords {intent_rc.keywords[:3]}.")

    # ------------------------------------------------------------------------
    # [TEST 2] DatasetProvider & SemanticEngine Schema Resolution
    # ------------------------------------------------------------------------
    print("\n[TEST 2] Testing DatasetProvider & SemanticEngine Schema Resolution...")
    from packages.analytics_core.src.engines.dataset_provider import DatabaseDatasetProvider
    ds_provider = DatabaseDatasetProvider(session_factory=TestingSession)
    context = ds_provider.acquire_context(proj.id)
    assert not context.is_empty
    assert "web_traffic" in context.datasets_map
    assert "web_traffic" in context.dataset_fingerprints

    semantic = SemanticEngine().resolve_schema(intent_rc, context.datasets_map)
    assert semantic.target_metric_col == "conversions"
    assert semantic.group_dimension_col == "channel"
    assert semantic.time_col == "timestamp"
    print(f" -> PASSED: DatasetProvider ({len(context.datasets_map)} tables) and Semantic bindings resolved (target='{semantic.target_metric_col}', dimension='{semantic.group_dimension_col}').")

    # ------------------------------------------------------------------------
    # [TEST 3] HypothesisEngine Dynamic Synthesis
    # ------------------------------------------------------------------------
    print("\n[TEST 3] Testing HypothesisEngine Synthesis...")
    hyps = HypothesisEngine.generate_hypotheses(semantic)
    assert len(hyps) == 2
    assert "channel" in hyps[0].statement
    assert "conversions" in hyps[0].statement
    assert hyps[0].prior_probability + hyps[1].prior_probability == 1.0
    print(f" -> PASSED: Synthesized {len(hyps)} competing hypotheses with sum(priors)=1.0.")

    # ------------------------------------------------------------------------
    # [TEST 4] ExperimentEngine Executable SQL Design & ExecutionProvider
    # ------------------------------------------------------------------------
    print("\n[TEST 4] Testing ExperimentEngine & DuckDBExecutionProvider...")
    exp_specs = ExperimentEngine.design_experiments(hyps, semantic)
    assert len(exp_specs) == 2
    exec_provider = DuckDBExecutionProvider()
    val, cnt, dur, _ = exec_provider.execute_query(df_sample, "data_table", exp_specs[0].query_sql)
    assert val > 0.0
    assert cnt == 300
    assert dur >= 0.0
    print(f" -> PASSED: Experiment executed via ExecutionProvider (val={val:.2f}, rows={cnt}, dur={dur:.1f}ms).")

    # ------------------------------------------------------------------------
    # [TEST 5] EvidenceEngine Structured Output
    # ------------------------------------------------------------------------
    print("\n[TEST 5] Testing EvidenceEngine Structured Output...")
    ev_struct = EvidenceEngine.synthesize_evidence(
        exp_code="EXP-01",
        ev_index=1,
        primary_metric=val,
        row_count=cnt,
    )
    assert ev_struct.code == "EVID-01"
    assert ev_struct.validation_status == "UNVERIFIED"
    print(" -> PASSED: Evidence initialized strictly as UNVERIFIED with empirical statement.")

    # ------------------------------------------------------------------------
    # [TEST 6] VerificationEngine Dual-Engine Cross-Validation
    # ------------------------------------------------------------------------
    print("\n[TEST 6] Testing VerificationEngine Dual-Engine Cross-Validation...")
    verif_res = VerificationEngine.verify_secondary(
        primary_df=df_sample,
        target_metric_col="conversions",
        aggregation_type="SUM",
        primary_metric=val,
    )
    assert verif_res.status == VerificationStatus.VERIFIED
    assert verif_res.observed_delta_pct <= 1e-4
    print(f" -> PASSED: Verified DuckDB vs Polars equality (delta = {verif_res.observed_delta_pct:.6f}).")

    # ------------------------------------------------------------------------
    # [TEST 7] BeliefEngine Bayesian Updating & ANOVA Variance
    # ------------------------------------------------------------------------
    print("\n[TEST 7] Testing BeliefEngine Bayesian Updating & ANOVA Variance...")
    eta_sq = BeliefEngine.compute_variance_explained(df_sample, "channel", "conversions")
    assert eta_sq is not None
    assert eta_sq > 0.0

    posteriors, delta_ent = BeliefEngine.compute_bayesian_posteriors([0.60, 0.40], [0.85, 0.15])
    assert posteriors[0] > 0.85
    assert delta_ent < 0.0
    print(f" -> PASSED: ANOVA eta-squared = {eta_sq:.2f}%, Posterior = {posteriors[0]:.4f}, delta_H = {delta_ent:.4f}.")

    # ------------------------------------------------------------------------
    # [TEST 8] AdversarialEngine, StoppingEngine & HumanDecisionController
    # ------------------------------------------------------------------------
    print("\n[TEST 8] Testing AdversarialEngine, StoppingEngine, and HumanDecisionController...")
    adv_eval = AdversarialEngine.evaluate_counter_hypothesis("HYP-02", counter_posterior=0.1053, discriminating_evidence_count=2)
    assert adv_eval.status in ["WEAKENED", "REFUTED"]

    stopping = StoppingEngine.evaluate_stopping(
        leading_posterior=0.8947,
        entropy_delta=-0.4855,
        experiments_completed=2,
        all_verifications_passed=True,
        counter_hypothesis_evaluated=True,
    )
    assert stopping.should_stop is True

    dec_eval = HumanDecisionController.evaluate_ambiguity(["conversions", "orders"], "conversions", True)
    assert dec_eval.policy == DecisionPolicy.CONTINUE_AUTONOMOUSLY
    print(f" -> PASSED: Adversarial ({adv_eval.status}), Stopping ({stopping.should_stop}), and Decision ({dec_eval.policy}) verified.")

    # ------------------------------------------------------------------------
    # [TEST 9] VerdictEngine Multi-Constraint Diagnostic Gate
    # ------------------------------------------------------------------------
    print("\n[TEST 9] Testing VerdictEngine Multi-Constraint Gate...")
    verdict_ok = VerdictEngine.evaluate_verdict(
        question="Why did conversions drop?",
        leading_hypothesis_code="HYP-01",
        leading_posterior=0.8947,
        initial_entropy=0.9709,
        final_entropy=0.4855,
        all_verifications_passed=True,
        adversarial_eval=adv_eval,
        variance_explained_pct=eta_sq,
    )
    assert verdict_ok.verdict_type == "DIAGNOSED"
    assert verdict_ok.confidence_score >= 0.75
    assert verdict_ok.stopping_criteria_met is True

    # Test Gate Rejection: If verification fails, verdict must NOT be DIAGNOSED
    verdict_rejected = VerdictEngine.evaluate_verdict(
        question="Why did conversions drop?",
        leading_hypothesis_code="HYP-01",
        leading_posterior=0.8947,
        initial_entropy=0.9709,
        final_entropy=0.4855,
        all_verifications_passed=False,
        adversarial_eval=adv_eval,
        variance_explained_pct=eta_sq,
    )
    assert verdict_rejected.verdict_type == "INCONCLUSIVE"
    print(" -> PASSED: Multi-gate criteria correctly required verification pass to issue DIAGNOSED.")

    # ------------------------------------------------------------------------
    # [TEST 10] Complete Autonomous Investigation DAG via Controller & Decoupled Worker
    # ------------------------------------------------------------------------
    print("\n[TEST 10] Testing Complete Autonomous Investigation DAG (Controller + Pure Worker)...")
    queue = DatabaseQueueProvider(session_factory=TestingSession)
    controller = InvestigationController(session_factory=TestingSession)
    worker = InvestigationWorker(worker_id="worker-p3-e2e", queue_provider=queue, session_factory=TestingSession, controller=controller)

    inv = Investigation(id="INV-P3-E2E", project_id=proj.id, user_id=user.id, question="Why did conversions change across channel?", status=InvestigationState.QUEUED)
    db.add(inv)
    db.commit()

    queue.enqueue(investigation_id="INV-P3-E2E", priority=100)
    processed = worker.process_next_job()
    assert processed is True

    db.expire_all()
    inv_res = db.query(Investigation).filter(Investigation.id == "INV-P3-E2E").first()
    assert inv_res.status == InvestigationState.COMPLETED
    assert inv_res.verdict_type in ["DIAGNOSED", "OBSERVED", "STATISTICALLY_SIGNIFICANT", "INCONCLUSIVE"]
    assert len(inv_res.reproducible_manifest_hash) == 64

    # Check relational graph completeness
    assert len(db.query(InvestigationObjective).filter(InvestigationObjective.investigation_id == "INV-P3-E2E").all()) >= 1
    assert len(db.query(Hypothesis).filter(Hypothesis.investigation_id == "INV-P3-E2E").all()) >= 2
    assert len(db.query(Experiment).filter(Experiment.investigation_id == "INV-P3-E2E").all()) >= 1
    assert len(db.query(Evidence).filter(Evidence.investigation_id == "INV-P3-E2E").all()) >= 1
    assert len(db.query(EvidenceVerification).all()) >= 1
    assert len(db.query(InvestigationVerdict).filter(InvestigationVerdict.investigation_id == "INV-P3-E2E").all()) >= 1
    print(f" -> PASSED: Complete Investigation DAG executed cleanly through decomposed micro-engines (Manifest: {inv_res.reproducible_manifest_hash[:16]}...).")

    print("\n" + "=" * 80)
    print("ALL 10 PHASE 3 RUNTIME DECOMPOSITION TESTS PASSED WITH 100% INTEGRITY")
    print("=" * 80)


if __name__ == "__main__":
    run_phase3_tests()
