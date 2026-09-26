from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from apps.api.src.models.entities import (
    Base,
    Investigation,
    InvestigationEvent,
    Hypothesis,
    Prediction,
    Experiment,
    Evidence,
    BeliefUpdate,
)
from packages.analytics_core.src.runtime.state_reconstruction import (
    reconstruct_investigation_state,
    replay_investigation_events,
)


def make_session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)()


def test_replay_event_stream_restores_lifecycle_facts():
    events = [
        InvestigationEvent(sequence=1, investigation_id="I1", event_type="investigation.universal_analysis_plan", event_payload_json={"task": "ASSOCIATION"}),
        InvestigationEvent(sequence=2, investigation_id="I1", event_type="investigation.contract.replanned", event_payload_json={"reason": "initial evidence ambiguous"}),
        InvestigationEvent(sequence=3, investigation_id="I1", event_type="investigation.experiment.executed", event_payload_json={"experiment_code": "EXP-1"}),
        InvestigationEvent(sequence=4, investigation_id="I1", event_type="belief.updated", event_payload_json={"delta_entropy": -0.25}),
        InvestigationEvent(sequence=5, investigation_id="I1", event_type="investigation.stopping_decision", event_payload_json={"reason": "ANSWER_ESTABLISHED"}),
        InvestigationEvent(sequence=6, investigation_id="I1", event_type="investigation.verdict.formulated", event_payload_json={"verdict_type": "SUPPORTED"}),
    ]
    result = replay_investigation_events(events)
    assert result["analysis_plan"]["task"] == "ASSOCIATION"
    assert result["replan_count"] == 1
    assert result["last_replan_reason"] == "initial evidence ambiguous"
    assert result["executed_experiments"] == ["EXP-1"]
    assert result["last_entropy_delta"] == -0.25
    assert result["stopping_reason"] == "ANSWER_ESTABLISHED"
    assert result["final_verdict"] == "SUPPORTED"


def test_reconstructs_hypotheses_predictions_experiments_evidence_and_beliefs():
    db = make_session()
    inv = Investigation(id="I1", project_id="P1", question="Does plan_tier affect churn?", status="RUNNING")
    db.add(inv)
    db.add(Hypothesis(
        id="I1_HYP-01", investigation_id="I1", hypothesis_code="HYP-01",
        canonical_identity="event-sourced-recovery-test-hyp-01",
        statement="plan_tier is associated with churn", rationale="different rates",
        prior_probability=0.5, posterior_probability=0.8, belief_state="supported", status="SUPPORTED",
        target_metric="churn_event", target_dimension="plan_tier", mechanism_detail="association",
    ))
    db.add(Prediction(
        id="PRED-001", investigation_id="I1", hypothesis_id="I1_HYP-01",
        prediction_code="PRED-01", statement="churn rate differs by plan",
        target_metric="churn_event", target_dimension="plan_tier", expected_direction="difference",
        confidence=0.8, testability=True, status="SUPPORTED",
        actual_observed_result_json={"difference": 0.12},
        evaluation_reason="verified",
    ))
    db.add(Experiment(
        id="I1_EXP-01", investigation_id="I1", hypothesis_id="I1_HYP-01",
        target_prediction_id="PRED-001", test_code="EXP-01", tool_name="duckdb",
        fingerprint="abc123", status="EXECUTED",
    ))
    db.add(Evidence(
        id="EV-001", investigation_id="I1", experiment_id="I1_EXP-01", hypothesis_id="I1_HYP-01",
        statement="churn rates differ by plan", evidence_type="DIRECT_MEASUREMENT",
        effect_size=0.12, validation_status="VERIFIED",
    ))
    db.add(BeliefUpdate(
        id="BU-001", investigation_id="I1", hypothesis_id="I1_HYP-01", evidence_id="EV-001",
        prior_probability=0.5, likelihood_p=0.9, posterior_probability=0.8,
        entropy_delta=-0.2, update_step_index=1,
    ))
    db.add_all([
        InvestigationEvent(sequence=1, investigation_id="I1", event_type="investigation.experiment.executed", event_payload_json={"experiment_code": "EXP-01"}),
        InvestigationEvent(sequence=2, investigation_id="I1", event_type="investigation.stopping_decision", event_payload_json={"reason": "ANSWER_ESTABLISHED"}),
    ])
    db.commit()

    state = reconstruct_investigation_state("I1", db, question=inv.question)
    assert len(state.get_active_hypotheses()) == 1
    assert state.get_hypothesis("HYP-01").posterior_probability == 0.8
    assert state.get_predictions_for_hypothesis("HYP-01")[0].status == "SUPPORTED"
    assert state.get_executed_experiments() == ["I1_EXP-01"]
    assert len(state.evidence_ledger.entries) == 1
    assert state.state.posterior_beliefs["HYP-01"] == 0.8
    db.close()
