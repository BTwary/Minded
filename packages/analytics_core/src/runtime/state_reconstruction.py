"""Durable reconstruction of autonomous investigation state.

The controller is deliberately allowed to keep hot in-memory state while running,
but a restart must not require the original Python process.  Reconstruction uses
both durable domain projections (hypotheses/evidence/experiments/predictions) and
the immutable investigation event stream.  The event stream supplies lifecycle
position and re-planning metadata; domain rows supply the scientific objects.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from packages.analytics_core.src.intelligence.hypothesis_identity import compute_semantic_identity
from packages.analytics_core.src.intelligence.predictive_hypothesis import PredictiveHypothesis
from packages.analytics_core.src.intelligence.prediction_engine import StructuredPrediction
from packages.analytics_core.src.runtime.state import InvestigationStateManager
from packages.schemas.src.analysis import CanonicalInvestigationState


def classify_evidence_for_hypothesis(
    validation_status: Optional[str],
    effect_size: Optional[float],
) -> Optional[str]:
    """Return supporting/contradicting/neutral using canonical status vocabulary."""
    status_upper = (validation_status or "").strip().upper()
    if status_upper == "VERIFIED" or (effect_size or 0) != 0:
        return "supporting"
    if status_upper == "FAILED":
        return "contradicting"
    return None


def reconstruct_investigation_state(
    investigation_id: str,
    session: Session,
    question: str = "",
) -> InvestigationStateManager:
    """Rebuild the hot runtime state from durable rows and replayed events.

    This function is intentionally deterministic: given the same persisted rows
    and event sequence it reconstructs the same hypothesis identities, posterior
    beliefs, executed-experiment set and lifecycle metadata.
    """
    from apps.api.src.models.entities import (
        Hypothesis as HypothesisRow,
        Evidence as EvidenceRow,
        Experiment as ExperimentRow,
        Prediction as PredictionRow,
        BeliefUpdate as BeliefUpdateRow,
        InvestigationEvent as InvestigationEventRow,
        Investigation as InvestigationRow,
    )

    inv = session.query(InvestigationRow).filter(InvestigationRow.id == investigation_id).first()
    state_mgr = InvestigationStateManager(investigation_id=investigation_id, question=question)

    # 1. Scientific objects: hypotheses are the durable authority for identity + beliefs.
    hyp_rows: List[HypothesisRow] = (
        session.query(HypothesisRow)
        .filter(HypothesisRow.investigation_id == investigation_id)
        .order_by(HypothesisRow.created_at.asc(), HypothesisRow.id.asc())
        .all()
    )
    for row in hyp_rows:
        h = PredictiveHypothesis(
            id=row.id,
            hypothesis_code=row.hypothesis_code,
            claim=row.statement or "",
            mechanism=row.rationale or "",
            predicted_observables_if_true=[],
            predicted_observables_if_false=[],
            falsification_criteria="",
            required_assumptions=[],
            prior_probability=float(row.prior_probability or 0.0),
            posterior_probability=float(row.posterior_probability or 0.0),
            belief_state=row.belief_state or "ACTIVE",
            is_counter_hypothesis=bool(row.is_counter_hypothesis),
            target_metric=row.target_metric or "",
            target_dimension=row.target_dimension or "",
            target_value=row.target_value,
            mechanism_detail=row.mechanism_detail or "",
            generated_reason=row.generated_reason or "",
            source_evidence=list(row.source_evidence_json or []),
            parent_hypotheses=list(row.parent_hypotheses_json or []),
        )
        h.canonical_identity = compute_semantic_identity(h)
        state_mgr.create_hypothesis(h)

    # 2. Predictions are reconstructed from their durable records, including their
    # evaluation result, so pending predictions are not regenerated after a restart.
    pred_rows: List[PredictionRow] = (
        session.query(PredictionRow)
        .filter(PredictionRow.investigation_id == investigation_id)
        .order_by(PredictionRow.created_at.asc(), PredictionRow.id.asc())
        .all()
    )
    for row in pred_rows:
        hyp_code = _hypothesis_code_from_id(row.hypothesis_id, investigation_id)
        pred = StructuredPrediction(
            prediction_id=row.id,
            hypothesis_id=row.hypothesis_id,
            hypothesis_code=hyp_code,
            statement=row.statement,
            target_metric=row.target_metric,
            target_dimension=row.target_dimension,
            expected_direction=row.expected_direction,
            expected_value=row.expected_value,
            threshold=row.threshold,
            expected_relationship=row.expected_relationship,
            expected_effect=row.expected_effect,
            # Never synthesize a confidence value on restart. `row.confidence`
            # is a nullable column and `StructuredPrediction.confidence` is
            # Optional[float] = None by design (no epistemic confidence is
            # implied merely by persisting a prediction) -- coercing a NULL
            # into 0.5 here would silently manufacture a belief that was
            # never actually held, purely as a side effect of the process
            # restarting. `row.confidence or 0.5` was also wrong for a
            # genuine persisted value of 0.0 (falsy), not only for NULL.
            confidence=(float(row.confidence) if row.confidence is not None else None),
            testability=bool(row.testability),
            status=row.status or "PENDING",
            target_experiment_id=row.target_experiment_id,
            actual_observed_result=dict(row.actual_observed_result_json or {}),
            evaluation_reason=row.evaluation_reason,
        )
        state_mgr.create_prediction(pred)

    # 3. Evidence links are restored onto the exact persisted hypothesis IDs.
    ev_rows: List[EvidenceRow] = (
        session.query(EvidenceRow)
        .filter(EvidenceRow.investigation_id == investigation_id)
        .order_by(EvidenceRow.created_at.asc(), EvidenceRow.id.asc())
        .all()
    )
    for ev in ev_rows:
        if not ev.hypothesis_id:
            continue
        target = _find_hypothesis_by_db_id(state_mgr, ev.hypothesis_id)
        if target is None:
            continue
        classification = classify_evidence_for_hypothesis(ev.validation_status, ev.effect_size)
        if classification == "supporting" and ev.id not in target.supporting_evidence_ids:
            target.supporting_evidence_ids.append(ev.id)
        elif classification == "contradicting" and ev.id not in target.contradicting_evidence_ids:
            target.contradicting_evidence_ids.append(ev.id)
        state_mgr.append_evidence(
            claim=ev.statement,
            metric=ev.statistical_test_name,
            value=ev.effect_size,
            proof=ev.calculation_summary,
        )

    # 4. Completed/failed experiments and fingerprints are reconstructed so the
    # EIG loop sees the same search frontier after a restart.
    exp_rows: List[ExperimentRow] = (
        session.query(ExperimentRow)
        .filter(ExperimentRow.investigation_id == investigation_id)
        .order_by(ExperimentRow.created_at.asc(), ExperimentRow.id.asc())
        .all()
    )
    for exp in exp_rows:
        status = (exp.status or "").upper()
        if status == "EXECUTED":
            state_mgr.record_experiment_executed(exp.id, fingerprint=exp.fingerprint)
        elif status in {"FAILED", "SKIPPED"}:
            state_mgr.record_experiment_failed(exp.id, error_msg=exp.rationale or status)

    # 5. Rebuild posterior trajectory from the last durable Bayesian updates.
    belief_rows: List[BeliefUpdateRow] = (
        session.query(BeliefUpdateRow)
        .filter(BeliefUpdateRow.investigation_id == investigation_id)
        .order_by(BeliefUpdateRow.update_step_index.asc(), BeliefUpdateRow.created_at.asc())
        .all()
    )
    if belief_rows:
        latest: Dict[str, float] = {}
        entropy_delta = 0.0
        for bu in belief_rows:
            code = _hypothesis_code_from_id(bu.hypothesis_id, investigation_id)
            latest[code] = float(bu.posterior_probability)
            entropy_delta += float(bu.entropy_delta or 0.0)
        if latest:
            state_mgr.update_beliefs(latest, delta_entropy=entropy_delta)

    # 6. Replay immutable lifecycle events.  The projection rows above supply the
    # scientific payload; events supply the exact lifecycle location and replan history.
    events: List[InvestigationEventRow] = (
        session.query(InvestigationEventRow)
        .filter(InvestigationEventRow.investigation_id == investigation_id)
        .order_by(InvestigationEventRow.sequence.asc())
        .all()
    )
    lifecycle = replay_investigation_events(events)
    state_mgr.current_turn = lifecycle["experiment_round"]

    # Prefer the latest immutable canonical runtime snapshot when one exists.
    # Domain rows above remain the structural authority for runtime lookup maps;
    # the snapshot restores the complete Pydantic scientific state that cannot be
    # losslessly inferred from normalized rows alone (uncertainty, raw observations,
    # candidate/completed experiment payloads, stopping state, verdict, lineage, etc.).
    snapshot_state = None
    for event in reversed(events):
        if str(getattr(event, "event_type", "")) == "investigation.scientific_state_snapshot":
            payload = getattr(event, "event_payload_json", {}) or {}
            snapshot_state = payload.get("snapshot", {}).get("canonical_runtime_state")
            if snapshot_state:
                break
    if snapshot_state:
        try:
            restored = CanonicalInvestigationState.model_validate(snapshot_state)
            state_mgr.state = restored
            state_mgr.current_turn = lifecycle["experiment_round"]
        except Exception:
            # Never discard recoverable normalized state because an old or partial
            # snapshot cannot be parsed after a schema evolution.
            pass
    if lifecycle["last_entropy_delta"] is not None and not belief_rows:
        state_mgr.state.information_gain_accumulated = abs(float(lifecycle["last_entropy_delta"]))

    # Persisted canonical state fields are restored where available.
    if inv is not None:
        state_mgr.state.original_question = inv.question or state_mgr.state.original_question
    # AnalysisPlan is persisted on InvestigationContract, not CanonicalInvestigationState.
    # Keep the reducer free of non-canonical dynamic fields; callers load the active
    # contract separately when they need the compiled plan.
    return state_mgr


def replay_investigation_events(events: List[Any]) -> Dict[str, Any]:
    """Pure event reducer for restart/audit tests.

    It intentionally does not recreate domain rows; it derives lifecycle facts
    that must survive process death: phase, replan count/reason, experiments seen,
    stopping reason, final verdict, and the last durable analysis plan.
    """
    result: Dict[str, Any] = {
        "current_phase": None,
        "last_replan_reason": None,
        "replan_count": 0,
        "experiment_round": 0,
        "executed_experiments": [],
        "last_entropy_delta": None,
        "stopping_reason": None,
        "final_verdict": None,
        "analysis_plan": None,
    }
    for event in events:
        et = str(getattr(event, "event_type", ""))
        payload = getattr(event, "event_payload_json", {}) or {}
        phase = payload.get("phase")
        if phase:
            result["current_phase"] = phase
        if et == "investigation.universal_analysis_plan":
            result["analysis_plan"] = payload
        elif et == "investigation.contract.replanned":
            result["replan_count"] += 1
            result["last_replan_reason"] = payload.get("reason") or payload.get("replan_reason")
        elif et in {"investigation.experiment.executed", "experiment.executed", "experiment.completed"}:
            code = payload.get("experiment_code") or payload.get("experiment_id")
            if code and code not in result["executed_experiments"]:
                result["executed_experiments"].append(code)
            result["experiment_round"] += 1
        elif et == "belief.updated":
            result["last_entropy_delta"] = payload.get("delta_entropy")
        elif et in {"investigation.stopping_decision", "investigation.stopping"}:
            result["stopping_reason"] = payload.get("reason") or payload.get("stopping_reason")
        elif et in {"investigation.verdict", "investigation.verdict.formulated"}:
            result["final_verdict"] = payload.get("verdict") or payload.get("verdict_type")
    return result


def _hypothesis_code_from_id(hypothesis_id: Optional[str], investigation_id: str) -> str:
    if not hypothesis_id:
        return ""
    prefix = f"{investigation_id}_"
    return hypothesis_id[len(prefix):] if hypothesis_id.startswith(prefix) else str(hypothesis_id)


def _find_hypothesis_by_db_id(state_mgr: InvestigationStateManager, db_id: str) -> Optional[Any]:
    for h in state_mgr.get_active_hypotheses():
        if getattr(h, "id", None) == db_id:
            return h
    return None
