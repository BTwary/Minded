"""Durable Asynchronous Investigations API with Distributed Job Queue, Live SSE Replay, and Human Decision Support."""
import asyncio
import json
import time
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Query, Header, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from apps.api.src.core.database import get_db, SessionLocal
from apps.api.src.core.security import get_current_user, verify_project_ownership, get_user_authorized_project_ids
from apps.api.src.models.entities import (
    User,
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
    InvestigationDecisionRequest,
    InvestigationResourceWait,
    InvestigationEvent,
    DecisionRecommendationRecord,
    Dataset,
    ClaimGateDecision,
    InvestigationContract,
    InvestigationDataReadiness,
    Assumption,
    Prediction,
    gen_uuid,
)
from packages.analytics_core.src.execution.queue import get_default_queue_provider
from packages.analytics_core.src.engines.local_explanation import LocalExplanationEngine
from packages.analytics_core.src.runtime.cross_dataset_review import build_cross_dataset_review
from packages.analytics_core.src.governance.recommendation_grounding import evaluate_recommendation_grounding
from packages.analytics_core.src.governance.verification_packet import (
    build_verification_packet,
    current_signoff,
    evaluate_review,
    validate_decision,
    validate_signoff,
)
from packages.analytics_core.src.execution.state_machine import (
    InvestigationState,
    JobState,
    transition_investigation,
    transition_job,
)

router = APIRouter(prefix="/investigations", tags=["investigations"])


class InvestigationCreateRequest(BaseModel):
    project_id: str
    question: str
    target_metric: Optional[str] = "revenue"
    priority: Optional[int] = 10
    dataset_ids: Optional[List[str]] = None


class DecisionResponseRequest(BaseModel):
    user_response: str


class VerificationDecisionRequest(BaseModel):
    item_id: str
    decision: str  # CONFIRMED | REJECTED | NEEDS_REWORK
    comment: Optional[str] = None
    # Fingerprint of the item as the reviewer saw it; if the result changed since
    # they loaded the packet the decision is refused rather than misattributed.
    item_fingerprint: Optional[str] = None


class VerificationSignoffRequest(BaseModel):
    outcome: str  # VERIFIED | REJECTED | NEEDS_REWORK
    comment: Optional[str] = None
    packet_fingerprint: Optional[str] = None


@router.post("")
def create_investigation(
    payload: InvestigationCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Create a new durable investigation and enqueue it to the QueueProvider.
    Returns immediately with investigation_id, job_id, and status: 'QUEUED'.
    Worker processes claim and execute the job completely asynchronously.
    """
    verify_project_ownership(payload.project_id, current_user, db)

    # Explicit multi-dataset scope is fail-closed: every requested dataset must
    # exist in the selected project, and duplicate IDs are rejected. The
    # runtime must never silently drop a requested dataset and continue with a
    # narrower population than the analyst selected.
    requested_dataset_ids = list(payload.dataset_ids or [])
    if len(requested_dataset_ids) != len(set(requested_dataset_ids)):
        raise HTTPException(status_code=400, detail="Duplicate dataset IDs are not allowed in an investigation scope.")
    if requested_dataset_ids:
        found = db.query(Dataset.id).filter(
            Dataset.project_id == payload.project_id,
            Dataset.id.in_(requested_dataset_ids),
        ).all()
        found_rows = db.query(Dataset.id, Dataset.name).filter(
            Dataset.project_id == payload.project_id,
            Dataset.id.in_(requested_dataset_ids),
        ).all()
        found_ids = {row[0] for row in found_rows}
        missing = [dataset_id for dataset_id in requested_dataset_ids if dataset_id not in found_ids]
        if missing:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "DATASET_SCOPE_INVALID",
                    "message": "One or more requested datasets are unavailable in this project.",
                    "missing_dataset_ids": missing,
                },
            )
        names_to_ids: dict[str, list[str]] = {}
        for dataset_id, dataset_name in found_rows:
            names_to_ids.setdefault(str(dataset_name), []).append(str(dataset_id))
        duplicate_names = {name: ids for name, ids in names_to_ids.items() if len(ids) > 1}
        if duplicate_names:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "DATASET_SCOPE_AMBIGUOUS",
                    "message": "Selected datasets contain duplicate display names. Rename the datasets or select only one of each duplicate name; AA-OS refuses to collapse two physical datasets into one table.",
                    "duplicate_names": duplicate_names,
                },
            )

    investigation_id = f"INV-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{gen_uuid()[:8]}"
    
    # 1. Create Investigation record in PLANNED status
    inv = Investigation(
        id=investigation_id,
        project_id=payload.project_id,
        user_id=current_user.id,
        question=payload.question,
        status=InvestigationState.PLANNED,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        requested_dataset_ids_json=requested_dataset_ids or None,
        parent_investigation_id=None,
        is_subinvestigation=False,
    )
    db.add(inv)
    db.commit()

    # 2. Transition state to QUEUED
    transition_investigation(
        db=db,
        investigation_id=investigation_id,
        to_state=InvestigationState.QUEUED,
        actor=current_user.email,
    )

    # 3. Enqueue in durable queue
    queue = get_default_queue_provider()
    job_id = queue.enqueue(investigation_id=investigation_id, priority=payload.priority or 10)

    # 4. Record event
    event = InvestigationEvent(
        id=f"EVT-{gen_uuid()[:8]}",
        investigation_id=investigation_id,
        event_type="investigation.created",
        event_payload_json={"job_id": job_id, "question": payload.question, "project_id": payload.project_id},
        timestamp=datetime.now(timezone.utc),
    )
    db.add(event)
    db.commit()

    return {
        "investigation_id": investigation_id,
        "job_id": job_id,
        "status": InvestigationState.QUEUED,
        "question": payload.question,
        "project_id": payload.project_id,
        "created_at": inv.created_at.isoformat(),
    }


@router.get("")
def list_investigations(
    project_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List investigations isolated strictly to authorized user projects."""
    if project_id:
        verify_project_ownership(project_id, current_user, db)
        allowed_ids = [project_id]
    else:
        allowed_ids = get_user_authorized_project_ids(current_user, db)

    query = db.query(Investigation).filter(
        Investigation.project_id.in_(allowed_ids),
        Investigation.is_subinvestigation.is_(False),
    )
    if status:
        query = query.filter(Investigation.status == status)

    invs = query.order_by(Investigation.created_at.desc()).limit(limit).all()

    return [
        {
            "id": i.id,
            "project_id": i.project_id,
            "question": i.question,
            "status": i.status,
            "verdict_type": i.verdict_type,
            "confidence_score": i.confidence_score,
            "direct_answer": i.direct_answer,
            "main_finding": i.main_finding,
            "reproducible_manifest_hash": i.reproducible_manifest_hash,
            "created_at": i.created_at.isoformat() if i.created_at else None,
            "updated_at": i.updated_at.isoformat() if i.updated_at else None,
            "analysis_plan": (
                (db.query(InvestigationEvent)
                 .filter(InvestigationEvent.investigation_id == i.id, InvestigationEvent.event_type == "investigation.universal_analysis_plan")
                 .order_by(InvestigationEvent.sequence.desc()).first().event_payload_json)
                if db.query(InvestigationEvent).filter(InvestigationEvent.investigation_id == i.id, InvestigationEvent.event_type == "investigation.universal_analysis_plan").first() else None
            ),
        }
        for i in invs
    ]


@router.get("/{investigation_id}")
def get_investigation(
    investigation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve full durable investigation state, DAG nodes, evidence, and active execution."""
    inv = db.query(Investigation).filter(Investigation.id == investigation_id).first()
    if not inv:
        raise HTTPException(status_code=404, detail="Investigation not found.")

    verify_project_ownership(inv.project_id, current_user, db)

    objectives = db.query(InvestigationObjective).filter(InvestigationObjective.investigation_id == inv.id).all()
    hyps = db.query(Hypothesis).filter(Hypothesis.investigation_id == inv.id).all()
    exps = db.query(Experiment).filter(Experiment.investigation_id == inv.id).all()
    evid = db.query(Evidence).filter(Evidence.investigation_id == inv.id).all()
    verdict = db.query(InvestigationVerdict).filter(InvestigationVerdict.investigation_id == inv.id).first()
    observations_by_experiment = {o.experiment_id: o for o in db.query(Observation).filter(Observation.experiment_id.in_([e.id for e in exps])).all()} if exps else {}
    experiments_by_id = {e.id: e for e in exps}
    decision_recs = db.query(DecisionRecommendationRecord).filter(DecisionRecommendationRecord.investigation_id == inv.id).all()
    hypothesis_by_code = {str(h.hypothesis_code): h for h in hyps}
    evidence_ids_by_hypothesis = {}
    for ev in evid:
        if ev.hypothesis_id:
            evidence_ids_by_hypothesis.setdefault(str(ev.hypothesis_id), []).append(str(ev.id))
    assumptions = db.query(Assumption).filter(Assumption.investigation_id == inv.id).all()
    predictions = db.query(Prediction).filter(Prediction.investigation_id == inv.id).order_by(Prediction.created_at.asc()).all()
    belief_updates = db.query(BeliefUpdate).filter(BeliefUpdate.investigation_id == inv.id).order_by(BeliefUpdate.update_step_index.asc(), BeliefUpdate.created_at.asc()).all()
    evidence_verifications = (
        db.query(EvidenceVerification)
        .filter(EvidenceVerification.evidence_id.in_([e.id for e in evid]))
        .order_by(EvidenceVerification.created_at.asc())
        .all()
    ) if evid else []
    event_rows = (
        db.query(InvestigationEvent)
        .filter(InvestigationEvent.investigation_id == inv.id)
        .order_by(InvestigationEvent.sequence.asc())
        .all()
    )
    latest_verification_status_by_id = {}
    for _vr in evidence_verifications:
        latest_verification_status_by_id[str(_vr.evidence_id)] = str(getattr(_vr, "status", ""))

    recommendation_grounding_events = {}
    for _event in event_rows:
        if _event.event_type == "decision.recommendation.grounding_evaluated":
            _payload = _event.event_payload_json or {}
            _recommendation_id = str(_payload.get("recommendation_id") or "")
            _hypothesis_id = str(_payload.get("grounded_hypothesis_id") or "")
            if _recommendation_id:
                recommendation_grounding_events[_recommendation_id] = _payload
            elif _hypothesis_id:
                # Backward-compatible lookup for pre-fix events. New writes
                # always carry a recommendation_id and are bound one-to-one.
                recommendation_grounding_events[_hypothesis_id] = _payload

    def _latest_payload(prefixes):
        for event in reversed(event_rows):
            if any(event.event_type == p or event.event_type.startswith(p) for p in prefixes):
                return event.event_payload_json or {}
        return None

    epistemic_payload = _latest_payload(["calculation.epistemic_calibration"])
    missingness_payload = _latest_payload(["investigation.missingness_sensitivity"])
    causal_gate_payload = _latest_payload(["investigation.causal_gate.evaluated", "investigation.causal_gate.fallback"])
    causal_effect_payload = _latest_payload(["investigation.causal_effect.estimated", "investigation.causal_effect.estimation_failed"])
    adversarial_payloads = [
        e.event_payload_json or {} for e in event_rows
        if e.event_type.startswith("investigation.adversarial_challenge")
        or e.event_type.startswith("calculation.adversarial")
    ]
    multiverse_payload = _latest_payload(["investigation.multiverse.skipped", "calculation.multiverse"])
    manifest_payload = _latest_payload(["calculation.provenance_manifest"])
    phase_semantic_payload = _latest_payload(["investigation.phase.semantic_model"])
    analyst_result_payload = _latest_payload(["investigation.analyst_result"])
    # `plan_payload` was dead code that referenced `analysis_plan` before it was
    # assigned later in this function (undefined-name crash, pyflakes-confirmed);
    # it was never actually consumed downstream, so it is removed rather than
    # reordered. `analysis_plan` itself remains defined and used below.
    latest_contract = (
        db.query(InvestigationContract)
        .filter(InvestigationContract.investigation_id == inv.id)
        .order_by(InvestigationContract.version.desc())
        .first()
    )
    # Session 8 (DEFECT-022): ``primary_dataset_name`` was referenced in the
    # response below but never defined in this function, so every
    # GET /investigations/{id} raised NameError -- the endpoint the analyst UI
    # polls for every result. Resolve it the same way /explanation does.
    scope_payload = _latest_payload(["investigation.dataset_scope_resolved"]) or {}
    primary_dataset_name = scope_payload.get("primary_dataset")
    readiness_rows = (
        db.query(InvestigationDataReadiness)
        .filter(InvestigationDataReadiness.investigation_id == inv.id)
        .order_by(InvestigationDataReadiness.dataset_name.asc(), InvestigationDataReadiness.assessed_at.desc())
        .all()
    )
    latest_readiness_by_dataset = {}
    for _row in readiness_rows:
        latest_readiness_by_dataset.setdefault(_row.dataset_name, _row)
    readiness = latest_readiness_by_dataset.get(primary_dataset_name)
    if readiness is None and latest_readiness_by_dataset:
        # Never let alphabetical ordering pick a misleading non-primary dataset
        # when the scope event is absent (older investigations).
        readiness = next(iter(latest_readiness_by_dataset.values()))
    high_risk_assumptions = [a for a in assumptions if a.sensitivity_risk == "high" and not a.is_validated]
    edges = db.query(InvestigationGraphEdge).filter(InvestigationGraphEdge.investigation_id == inv.id).all()
    pending_decision = (
        db.query(InvestigationDecisionRequest)
        .filter(
            InvestigationDecisionRequest.investigation_id == inv.id,
            InvestigationDecisionRequest.status == "PENDING",
        )
        .first()
    )
    pending_resource = (
        db.query(InvestigationResourceWait)
        .filter(
            InvestigationResourceWait.investigation_id == inv.id,
            InvestigationResourceWait.status == "PENDING",
        )
        .first()
    )

    claim_gate_row = (
        db.query(ClaimGateDecision)
        .filter(ClaimGateDecision.investigation_id == inv.id)
        .order_by(ClaimGateDecision.created_at.desc())
        .first()
    )

    plan_event = (
        db.query(InvestigationEvent)
        .filter(
            InvestigationEvent.investigation_id == inv.id,
            InvestigationEvent.event_type == "investigation.universal_analysis_plan",
        )
        .order_by(InvestigationEvent.sequence.desc())
        .first()
    )
    analysis_plan = plan_event.event_payload_json if plan_event else None
    decision_event = (
        db.query(InvestigationEvent)
        .filter(InvestigationEvent.investigation_id == inv.id, InvestigationEvent.event_type == "investigation.decision_impact")
        .order_by(InvestigationEvent.sequence.desc())
        .first()
    )
    decision_impact = decision_event.event_payload_json if decision_event else None
    compound_results = [
        (e.event_payload_json or {})
        for e in event_rows
        if e.event_type == "investigation.compound_objective.completed"
    ]

    # Human-reviewable cross-dataset execution ledger.
    requested_scope_ids = list(inv.requested_dataset_ids_json or [])
    requested_dataset_rows = (
        db.query(Dataset.id, Dataset.name)
        .filter(Dataset.project_id == inv.project_id, Dataset.id.in_(requested_scope_ids))
        .all()
        if requested_scope_ids else []
    )
    dataset_name_by_id = {str(row[0]): str(row[1]) for row in requested_dataset_rows}
    cross_dataset_review = build_cross_dataset_review(
        requested_dataset_ids=requested_scope_ids,
        dataset_name_by_id=dataset_name_by_id,
        experiments=[
            {
                "id": e.id,
                "test_code": e.test_code,
                "status": e.status,
                "arguments_json": e.arguments_json or {},
            }
            for e in exps
        ],
        evidences=[
            {
                "id": ev.id,
                "experiment_id": ev.experiment_id,
                "validation_status": ev.validation_status,
            }
            for ev in evid
        ],
        verifications=[
            {
                "evidence_id": verification.evidence_id,
                "status": verification.status,
            }
            for verification in evidence_verifications
        ],
        join_events=[
            dict(e.event_payload_json or {})
            for e in event_rows
            if e.event_type.startswith("relational.join.safety")
        ],
    )

    local_explanation = LocalExplanationEngine.build(
        question=inv.question,
        verdict_type=inv.verdict_type or "INCONCLUSIVE",
        direct_answer=inv.direct_answer,
        justification=verdict.justification if verdict else inv.main_finding or inv.direct_answer,
        hypotheses=hyps,
        experiments=exps,
        observations=[],
        evidence_rows=evid,
        method_selection=None,
        assumption_items=assumptions,
    )

    return {
        "id": inv.id,
        "project_id": inv.project_id,
        "question": inv.question,
        "status": inv.status,
        "verdict_type": inv.verdict_type,
        "confidence_score": inv.confidence_score,
        "direct_answer": inv.direct_answer,
        "main_finding": inv.main_finding,
        "analysis_mode": "DETERMINISTIC",
        "analyst_result": analyst_result_payload,
        "claim_gate": ({
            "outcome": claim_gate_row.outcome,
            "requested_claim": claim_gate_row.requested_claim,
            "evidence_level": claim_gate_row.evidence_level,
            "design_status": claim_gate_row.design_status,
            "assumptions": claim_gate_row.assumptions_json or [],
            "allowed_claim": claim_gate_row.allowed_claim,
            "blocked_claim": claim_gate_row.blocked_claim,
            "reason": claim_gate_row.reason,
            "recovery_actions": claim_gate_row.recovery_actions_json or [],
            "computed_evidence": claim_gate_row.computed_evidence_json or {},
            "refusal_id": claim_gate_row.id if claim_gate_row.outcome == "REFUSE" else None,
            "requested_level": claim_gate_row.requested_level,
            "max_supported_level": claim_gate_row.max_supported_level,
            "identification_strategy": claim_gate_row.identification_strategy,
            "blocking_conditions": claim_gate_row.blocking_conditions_json or [],
            "missing_evidence": claim_gate_row.missing_evidence_json or [],
        } if claim_gate_row else None),
        "analysis_plan": analysis_plan,
        "decision_impact": decision_impact,
        "ai_status_message": "Analysis and explanation are generated locally from canonical investigation state. AI is optional.",
        "explanation": local_explanation.to_dict(),
        "reproducible_manifest_hash": inv.reproducible_manifest_hash,
        "objectives": [{"id": o.id, "statement": o.statement, "status": o.status} for o in objectives],
        "compound_objectives": compound_results,
        "cross_dataset_review": cross_dataset_review,
        "hypotheses": [
            {
                "id": h.hypothesis_code or h.id,
                "statement": h.statement,
                "prior_probability": h.prior_probability,
                "posterior_probability": h.posterior_probability,
                "belief_state": h.belief_state,
                "is_counter_hypothesis": h.is_counter_hypothesis,
            }
            for h in hyps
        ],
        "experiments": [
            {"id": e.test_code or e.id, "tool_name": e.tool_name, "status": e.status} for e in exps
        ],
        "predictions": [
            {
                "id": p.id,
                "prediction_code": p.prediction_code,
                "hypothesis_id": p.hypothesis_id,
                "statement": p.statement,
                "target_metric": p.target_metric,
                "target_dimension": p.target_dimension,
                "target_segment": p.target_segment,
                "expected_direction": p.expected_direction,
                "expected_value": p.expected_value,
                "threshold": p.threshold,
                "expected_magnitude": p.expected_magnitude,
                "expected_relationship": p.expected_relationship,
                "expected_effect": p.expected_effect,
                "confidence": p.confidence,
                "testability": p.testability,
                "status": p.status,
                "target_experiment_id": p.target_experiment_id,
                "actual_observed_result": p.actual_observed_result_json or {},
                "evaluation_reason": p.evaluation_reason,
                "created_at": p.created_at.isoformat() if p.created_at else None,
                "evaluated_at": p.evaluated_at.isoformat() if p.evaluated_at else None,
            } for p in predictions
        ],
        "belief_updates": [
            {
                "id": b.id,
                "hypothesis_id": b.hypothesis_id,
                "evidence_id": b.evidence_id,
                "prior_probability": b.prior_probability,
                "bayes_factor": b.bayes_factor,
                "likelihood": b.likelihood_p,  # legacy probability field; never populated by canonical updates
                "posterior_probability": b.posterior_probability,
                "entropy_delta": b.entropy_delta,
                "update_step_index": b.update_step_index,
                "created_at": b.created_at.isoformat() if b.created_at else None,
            } for b in belief_updates
        ],
        "evidence_verifications": [
            {
                "id": v.id,
                "evidence_id": v.evidence_id,
                "primary_tool": v.primary_tool,
                "secondary_tool": v.secondary_tool,
                "tolerance_threshold": v.tolerance_threshold,
                "observed_delta_pct": v.observed_delta_pct,
                "is_deterministic": v.is_deterministic,
                "validator_fingerprint": v.validator_fingerprint,
                "status": v.status,
                "created_at": v.created_at.isoformat() if v.created_at else None,
            } for v in evidence_verifications
        ],
        "semantic_world_model": {
            "status": "PARTIAL" if phase_semantic_payload or latest_contract else "UNAVAILABLE",
            "phase_payload": phase_semantic_payload,
            "contract": {
                "id": latest_contract.id,
                "version": latest_contract.version,
                "problem_class": latest_contract.problem_class,
                "claim_type": latest_contract.claim_type,
                "target": latest_contract.target_json,
                "explanatory_variables": latest_contract.explanatory_variables_json or [],
                "grain": latest_contract.grain_json,
                "scope": latest_contract.scope_json,
                "time_window": latest_contract.time_window_json,
                "semantic_interpretations": latest_contract.semantic_interpretations_json or [],
                "unresolved_questions": latest_contract.unresolved_questions_json or [],
            } if latest_contract else None,
            "primary_dataset": primary_dataset_name,
        "data_readiness": {
                "dataset_name": readiness.dataset_name,
                "dataset_version_ids": readiness.dataset_version_ids_json or [],
                "source_fingerprints": readiness.source_fingerprints_json or {},
                "row_count": readiness.row_count,
                "column_count": readiness.column_count,
                "fitness_verdict": readiness.fitness_verdict,
                "checks": readiness.checks_json or {},
            } if readiness else None,
            "limitation": "Full relational world-model projection is not persisted as a first-class investigation artifact in this schema; this response exposes only the durable semantic and readiness facts actually recorded.",
        },
        "adversarial_findings": adversarial_payloads,
        # Session 8: steps that failed were previously visible only in the raw
        # event log. A human verifier must be told when a planned check did not run.
        "failed_steps": [
            {
                "step_code": (e.event_payload_json or {}).get("step_code"),
                "step_type": (e.event_payload_json or {}).get("step_type"),
                "error": (e.event_payload_json or {}).get("error"),
            }
            for e in event_rows if e.event_type == "step.failed"
        ],
        "multiverse": multiverse_payload,
        "missingness": missingness_payload,
        "causal_status": {
            "gate": causal_gate_payload,
            "effect_estimation": causal_effect_payload,
        },
        "epistemic_assessment": epistemic_payload,
        "stopping_reason": inv.stopping_rationale,
        "stopping_criteria_met": bool(inv.stopping_criteria_met),
        "provenance": {
            "manifest_hash": inv.reproducible_manifest_hash,
            "manifest_event": manifest_payload,
            "dataset_version_ids": inv.dataset_version_ids_json or [],
            "requested_dataset_ids": inv.requested_dataset_ids_json,
            "claim_lineage": [
                {
                    "evidence_id": ev.id,
                    "hypothesis_id": ev.hypothesis_id,
                    "experiment_id": ev.experiment_id,
                    "evidence_identity_hash": ev.evidence_identity_hash,
                    "validation_status": ev.validation_status,
                    "sql_executed": observations_by_experiment.get(ev.experiment_id).sql_executed if observations_by_experiment.get(ev.experiment_id) else None,
                    "source_dataset": ((experiments_by_id.get(ev.experiment_id).arguments_json or {}).get("dataset_versions", {}) if experiments_by_id.get(ev.experiment_id) else {}),
                } for ev in evid
            ],
            "limitation": "Per-claim dataset UUID/version and schema fingerprint are exposed where the persisted experiment arguments contain them; absent values are returned as null rather than inferred.",
        },
        "evidence": [
            {
                "id": ev.id,
                "statement": ev.statement,
                "validation_status": ev.validation_status,
                "p_value": ev.p_value,
                "effect_size": ev.effect_size,
                "calculation_summary": ev.calculation_summary or "",
                "row_count_analyzed": int(observations_by_experiment.get(ev.experiment_id).row_count_analyzed) if observations_by_experiment.get(ev.experiment_id) else 0,
                "sql_executed": observations_by_experiment.get(ev.experiment_id).sql_executed if observations_by_experiment.get(ev.experiment_id) else None,
                "calculation_trace": (observations_by_experiment.get(ev.experiment_id).result_json or {}).get("calculation_trace") if observations_by_experiment.get(ev.experiment_id) else None,
                "experiment_id": ev.experiment_id,
                "source_dataset": ((experiments_by_id.get(ev.experiment_id).arguments_json or {}).get("dataset_versions", {}) if experiments_by_id.get(ev.experiment_id) else {}),
            }
            for ev in evid
        ],
        "verdict": {
            "verdict_type": verdict.verdict_type,
            "confidence": verdict.confidence_score,
            "justification": verdict.justification,
            "epistemic_grade": verdict.epistemic_grade,
        } if verdict else None,
        "assumption_ledger": {
            "items": [
                {
                    "statement": a.statement,
                    "validated": bool(a.is_validated),
                    "sensitivity_risk": a.sensitivity_risk,
                } for a in assumptions
            ],
            "quality_card": {
                "overall_status": "RED" if len(high_risk_assumptions) > 2 else ("AMBER" if high_risk_assumptions else "GREEN"),
                "assumption_count": len(assumptions),
                "high_risk_unvalidated": len(high_risk_assumptions),
                "validated_count": sum(1 for a in assumptions if a.is_validated),
                "limitation_count": sum(1 for a in assumptions if not a.is_validated),
            },
        },
        "decision_recommendations": [
            {
                "recommendation_id": r.recommendation_id,
                "action_title": r.action_title,
                "action_description": r.action_description,
                "grounded_hypothesis_id": r.grounded_hypothesis_id,
                "target_metric": r.target_metric,
                "evidence_binding": (
                    lambda _hyp, _ground: {
                        "hypothesis_code": (_hyp.hypothesis_code if _hyp is not None else None),
                        "evidence_ids": list(_ground.supporting_evidence_ids),
                        "verified_evidence_ids": list(_ground.verified_supporting_evidence_ids),
                        "unverified_evidence_ids": list(_ground.unverified_supporting_evidence_ids),
                        "verified_contradicting_evidence_ids": list(_ground.verified_contradicting_evidence_ids),
                        "evidence_count": len(_ground.supporting_evidence_ids),
                        "binding_status": _ground.status,
                        "reason": _ground.reason,
                        "persisted_binding": recommendation_grounding_events.get(r.recommendation_id) or recommendation_grounding_events.get(r.grounded_hypothesis_id),
                    }
                )(
                    hypothesis_by_code.get(r.grounded_hypothesis_id.split("_", 1)[1]) if "_" in r.grounded_hypothesis_id else None,
                    evaluate_recommendation_grounding(
                        supporting_evidence_ids=(
                            getattr(hypothesis_by_code.get(r.grounded_hypothesis_id.split("_", 1)[1]), "supporting_evidence_ids", None) or []
                            if "_" in r.grounded_hypothesis_id else []
                        ),
                        contradicting_evidence_ids=(
                            getattr(hypothesis_by_code.get(r.grounded_hypothesis_id.split("_", 1)[1]), "contradicting_evidence_ids", None) or []
                            if "_" in r.grounded_hypothesis_id else []
                        ),
                        evidence_validation_status_by_id={
                            str(e.id): str(getattr(e, "validation_status", "")) for e in evid
                        },
                        latest_verification_status_by_id=latest_verification_status_by_id,
                    ),
                ),
                "target_metric": r.target_metric,
                "expected_utility": {
                    "expected_gain_metric": r.expected_gain_metric,
                    "downside_risk_metric": r.downside_risk_metric,
                    "probability_of_success": r.probability_of_success,
                    "net_expected_utility": r.net_expected_utility,
                    "utility_function_description": r.utility_function_description,
                },
                "policy_compliance_passed": r.policy_compliance_passed,
                "required_preconditions": r.required_preconditions_json or [],
            }
            for r in decision_recs
        ],
        "graph_edges": [
            {"source": e.source_node_id, "target": e.target_node_id, "type": e.relationship_type}
            for e in edges
        ],
        "pending_decision_request": {
            "id": pending_decision.id,
            "question": pending_decision.question,
            "options": pending_decision.options_json,
        } if pending_decision else None,
        "pending_resource_wait": {
            "id": pending_resource.id,
            "resource_id": pending_resource.resource_id,
            "reason": pending_resource.reason,
        } if pending_resource else None,
        "created_at": inv.created_at.isoformat() if inv.created_at else None,
        "updated_at": inv.updated_at.isoformat() if inv.updated_at else None,
    }


@router.get("/{investigation_id}/events")
async def stream_investigation_events(
    investigation_id: str,
    last_event_id: Optional[str] = Query(None),
    last_event_id_header: Optional[str] = Header(None, alias="Last-Event-ID"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Live Server-Sent Events (SSE) stream with Last-Event-ID cursor replay.
    Continuously streams new events as they are committed to database by worker processes.
    """
    inv = db.query(Investigation).filter(Investigation.id == investigation_id).first()
    if not inv:
        raise HTTPException(status_code=404, detail="Investigation not found.")
    verify_project_ownership(inv.project_id, current_user, db)

    effective_last_id = last_event_id or last_event_id_header

    async def event_generator():
        last_seen_seq = int(effective_last_id) if (effective_last_id and str(effective_last_id).isdigit()) else 0

        while True:
            session = SessionLocal()
            try:
                events = (
                    session.query(InvestigationEvent)
                    .filter(
                        InvestigationEvent.investigation_id == investigation_id,
                        InvestigationEvent.sequence > last_seen_seq,
                    )
                    .order_by(InvestigationEvent.sequence.asc())
                    .limit(50)
                    .all()
                )

                for ev in events:
                    last_seen_seq = ev.sequence or (last_seen_seq + 1)
                    payload = {
                        "id": ev.id,
                        "sequence": ev.sequence,
                        "type": ev.event_type,
                        "payload": ev.event_payload_json,
                        "timestamp": ev.timestamp.isoformat() if ev.timestamp else None,
                    }
                    yield f"id: {ev.sequence or ev.id}\nevent: {ev.event_type}\ndata: {json.dumps(payload)}\n\n"

                inv_curr = session.query(Investigation).filter(Investigation.id == investigation_id).first()
                if inv_curr and inv_curr.status in [
                    InvestigationState.COMPLETED,
                    InvestigationState.FAILED,
                    InvestigationState.CANCELLED,
                    InvestigationState.PARTIAL,
                ]:
                    # Terminal state reached, yield remaining and close
                    break
            finally:
                session.close()

            await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/{investigation_id}/cancel")
def cancel_investigation(
    investigation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Request clean cancellation of an active or queued investigation."""
    inv = db.query(Investigation).filter(Investigation.id == investigation_id).first()
    if not inv:
        raise HTTPException(status_code=404, detail="Investigation not found.")
    verify_project_ownership(inv.project_id, current_user, db)

    transition_investigation(
        db=db,
        investigation_id=investigation_id,
        to_state=InvestigationState.CANCEL_REQUESTED,
        actor=current_user.email,
    )

    job = db.query(InvestigationJob).filter(InvestigationJob.investigation_id == investigation_id).first()
    if job and job.status not in [JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED]:
        transition_job(db=db, job_id=job.id, to_state=JobState.CANCEL_REQUESTED)

    return {"status": "success", "message": f"Cancellation requested for {investigation_id}."}


@router.post("/{investigation_id}/decision")
def resolve_decision_request(
    investigation_id: str,
    payload: DecisionResponseRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Submit human decision for a suspended investigation and resume execution."""
    inv = db.query(Investigation).filter(Investigation.id == investigation_id).first()
    if not inv:
        raise HTTPException(status_code=404, detail="Investigation not found.")
    verify_project_ownership(inv.project_id, current_user, db)

    dec_req = (
        db.query(InvestigationDecisionRequest)
        .filter(
            InvestigationDecisionRequest.investigation_id == investigation_id,
            InvestigationDecisionRequest.status == "PENDING",
        )
        .first()
    )
    if not dec_req:
        raise HTTPException(status_code=400, detail="No pending decision request for this investigation.")

    dec_req.status = "RESOLVED"
    dec_req.user_response = payload.user_response
    dec_req.user_id = current_user.id
    dec_req.resolved_at = datetime.now(timezone.utc)

    # Transition state to QUEUED
    transition_investigation(
        db=db,
        investigation_id=investigation_id,
        to_state=InvestigationState.QUEUED,
        actor=current_user.email,
    )

    # Resume job in queue
    queue = get_default_queue_provider()
    queue.resume_job(investigation_id=investigation_id)

    return {
        "status": "success",
        "message": f"Decision submitted: '{payload.user_response}'. Investigation resumed.",
        "investigation_id": investigation_id,
    }


@router.get("/{investigation_id}/explanation")
def get_investigation_explanation(
    investigation_id: str,
    mode: str = Query("DETERMINISTIC", pattern="^(DETERMINISTIC|AI_AUGMENTED)$"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return a local explanation, with optional AI-only presentation augmentation.

    The deterministic investigation, evidence, verdict, and provenance remain the
    source of truth. AI can only rewrite the already-established explanation.
    """
    inv = db.query(Investigation).filter(Investigation.id == investigation_id).first()
    if not inv:
        raise HTTPException(status_code=404, detail="Investigation not found.")
    verify_project_ownership(inv.project_id, current_user, db)

    hyps = db.query(Hypothesis).filter(Hypothesis.investigation_id == inv.id).all()
    exps = db.query(Experiment).filter(Experiment.investigation_id == inv.id).all()
    evid = db.query(Evidence).filter(Evidence.investigation_id == inv.id).all()
    verdict = db.query(InvestigationVerdict).filter(InvestigationVerdict.investigation_id == inv.id).first()
    assumptions = db.query(Assumption).filter(Assumption.investigation_id == inv.id).all()
    predictions = db.query(Prediction).filter(Prediction.investigation_id == inv.id).order_by(Prediction.created_at.asc()).all()
    belief_updates = db.query(BeliefUpdate).filter(BeliefUpdate.investigation_id == inv.id).order_by(BeliefUpdate.update_step_index.asc(), BeliefUpdate.created_at.asc()).all()
    evidence_verifications = (
        db.query(EvidenceVerification)
        .filter(EvidenceVerification.evidence_id.in_([e.id for e in evid]))
        .order_by(EvidenceVerification.created_at.asc())
        .all()
    ) if evid else []
    event_rows = (
        db.query(InvestigationEvent)
        .filter(InvestigationEvent.investigation_id == inv.id)
        .order_by(InvestigationEvent.sequence.asc())
        .all()
    )

    def _latest_payload(prefixes):
        for event in reversed(event_rows):
            if any(event.event_type == p or event.event_type.startswith(p) for p in prefixes):
                return event.event_payload_json or {}
        return None

    epistemic_payload = _latest_payload(["calculation.epistemic_calibration"])
    missingness_payload = _latest_payload(["investigation.missingness_sensitivity"])
    causal_gate_payload = _latest_payload(["investigation.causal_gate.evaluated", "investigation.causal_gate.fallback"])
    causal_effect_payload = _latest_payload(["investigation.causal_effect.estimated", "investigation.causal_effect.estimation_failed"])
    adversarial_payloads = [
        e.event_payload_json or {} for e in event_rows
        if e.event_type.startswith("investigation.adversarial_challenge")
        or e.event_type.startswith("calculation.adversarial")
    ]
    multiverse_payload = _latest_payload(["investigation.multiverse.skipped", "calculation.multiverse"])
    manifest_payload = _latest_payload(["calculation.provenance_manifest"])
    phase_semantic_payload = _latest_payload(["investigation.phase.semantic_model"])
    analyst_result_payload = _latest_payload(["investigation.analyst_result"])
    # `plan_payload` was dead code that referenced `analysis_plan` before it was
    # assigned later in this function (undefined-name crash, pyflakes-confirmed);
    # it was never actually consumed downstream, so it is removed rather than
    # reordered. `analysis_plan` itself remains defined and used below.
    latest_contract = (
        db.query(InvestigationContract)
        .filter(InvestigationContract.investigation_id == inv.id)
        .order_by(InvestigationContract.version.desc())
        .first()
    )
    scope_payload = _latest_payload(["investigation.dataset_scope_resolved"]) or {}
    primary_dataset_name = scope_payload.get("primary_dataset")
    readiness_rows = (
        db.query(InvestigationDataReadiness)
        .filter(InvestigationDataReadiness.investigation_id == inv.id)
        .order_by(InvestigationDataReadiness.dataset_name.asc(), InvestigationDataReadiness.assessed_at.desc())
        .all()
    )
    latest_readiness_by_dataset = {}
    for row in readiness_rows:
        latest_readiness_by_dataset.setdefault(row.dataset_name, row)
    readiness = latest_readiness_by_dataset.get(primary_dataset_name)
    if readiness is None and latest_readiness_by_dataset:
        # Keep the legacy field deterministic even when an older investigation
        # predates dataset_scope_resolved; never let alphabetical ordering choose
        # a misleading non-primary dataset.
        readiness = next(iter(latest_readiness_by_dataset.values()))
    plan_event = (
        db.query(InvestigationEvent)
        .filter(InvestigationEvent.investigation_id == inv.id, InvestigationEvent.event_type == "investigation.universal_analysis_plan")
        .order_by(InvestigationEvent.sequence.desc())
        .first()
    )
    local = LocalExplanationEngine.build(
        question=inv.question,
        verdict_type=inv.verdict_type or "INCONCLUSIVE",
        direct_answer=inv.direct_answer,
        justification=verdict.justification if verdict else inv.main_finding or inv.direct_answer,
        hypotheses=hyps,
        experiments=exps,
        evidence_rows=evid,
        method_selection=None,
        assumption_items=assumptions,
    )

    result = local.to_dict()
    result["analysis_id"] = investigation_id
    result["analysis_mode"] = "DETERMINISTIC"
    result["ai_fallback_triggered"] = False
    result["ai_status_message"] = "Local deterministic explanation active; no AI required."
    # Attach the data fetched above (predictions, belief updates, evidence
    # verifications, and event-derived payloads) so this endpoint doesn't
    # silently drop them the way the sibling GET /investigations/{id}
    # endpoint's equivalent block does not. Serialization mirrors that
    # sibling endpoint's dict shape.
    result["predictions"] = [
        {
            "id": p.id,
            "prediction_code": p.prediction_code,
            "hypothesis_id": p.hypothesis_id,
            "statement": p.statement,
            "target_metric": p.target_metric,
            "target_dimension": p.target_dimension,
            "target_segment": p.target_segment,
            "expected_direction": p.expected_direction,
            "expected_value": p.expected_value,
            "threshold": p.threshold,
            "expected_magnitude": p.expected_magnitude,
            "expected_relationship": p.expected_relationship,
            "expected_effect": p.expected_effect,
            "confidence": p.confidence,
            "testability": p.testability,
            "status": p.status,
            "target_experiment_id": p.target_experiment_id,
            "actual_observed_result": p.actual_observed_result_json or {},
            "evaluation_reason": p.evaluation_reason,
            "created_at": p.created_at.isoformat() if p.created_at else None,
            "evaluated_at": p.evaluated_at.isoformat() if p.evaluated_at else None,
        } for p in predictions
    ]
    result["belief_updates"] = [
        {
            "id": b.id,
            "hypothesis_id": b.hypothesis_id,
            "evidence_id": b.evidence_id,
            "prior_probability": b.prior_probability,
            "bayes_factor": b.bayes_factor,
            "likelihood": b.likelihood_p,
            "posterior_probability": b.posterior_probability,
            "entropy_delta": b.entropy_delta,
            "update_step_index": b.update_step_index,
            "created_at": b.created_at.isoformat() if b.created_at else None,
        } for b in belief_updates
    ]
    result["evidence_verifications"] = [
        {
            "id": v.id,
            "evidence_id": v.evidence_id,
            "primary_tool": v.primary_tool,
            "secondary_tool": v.secondary_tool,
            "tolerance_threshold": v.tolerance_threshold,
            "observed_delta_pct": v.observed_delta_pct,
            "is_deterministic": v.is_deterministic,
            "validator_fingerprint": v.validator_fingerprint,
            "status": v.status,
            "created_at": v.created_at.isoformat() if v.created_at else None,
        } for v in evidence_verifications
    ]
    result["semantic_world_model"] = {
        "status": "PARTIAL" if phase_semantic_payload or latest_contract else "UNAVAILABLE",
        "phase_payload": phase_semantic_payload,
        "contract": {
            "id": latest_contract.id,
            "version": latest_contract.version,
            "problem_class": latest_contract.problem_class,
            "claim_type": latest_contract.claim_type,
            "target": latest_contract.target_json,
            "explanatory_variables": latest_contract.explanatory_variables_json or [],
            "grain": latest_contract.grain_json,
            "scope": latest_contract.scope_json,
            "time_window": latest_contract.time_window_json,
            "semantic_interpretations": latest_contract.semantic_interpretations_json or [],
            "unresolved_questions": latest_contract.unresolved_questions_json or [],
        } if latest_contract else None,
        "data_readiness": {
            "dataset_name": readiness.dataset_name,
            "dataset_version_ids": readiness.dataset_version_ids_json or [],
            "source_fingerprints": readiness.source_fingerprints_json or {},
            "row_count": readiness.row_count,
            "column_count": readiness.column_count,
            "fitness_verdict": readiness.fitness_verdict,
            "checks": readiness.checks_json or {},
        } if readiness else None,
        "dataset_scope_readiness": [
            {
                "dataset_name": row.dataset_name,
                "dataset_version_ids": row.dataset_version_ids_json or [],
                "source_fingerprints": row.source_fingerprints_json or {},
                "row_count": row.row_count,
                "column_count": row.column_count,
                "overall_quality_score": row.overall_quality_score,
                "fitness_verdict": row.fitness_verdict,
                "can_proceed": row.can_proceed,
                "critical_issues": row.critical_issues_json or [],
                "warnings": row.warnings_json or [],
            }
            for row in latest_readiness_by_dataset.values()
        ],
        "limitation": "Full relational world-model projection is not persisted as a first-class investigation artifact in this schema; this response exposes only the durable semantic and readiness facts actually recorded.",
    }
    result["adversarial_findings"] = adversarial_payloads
    result["multiverse"] = multiverse_payload
    result["missingness"] = missingness_payload
    result["causal_status"] = {
        "gate": causal_gate_payload,
        "effect_estimation": causal_effect_payload,
    }
    result["epistemic_assessment"] = epistemic_payload
    result["provenance_manifest_event"] = manifest_payload
    result["analysis_plan"] = plan_event.event_payload_json if plan_event else None
    result["analyst_result"] = analyst_result_payload

    if mode == "AI_AUGMENTED":
        try:
            from apps.api.src.ai.providers.factory import get_ai_provider as _get_ai_provider, is_ai_enabled
            _user_id = str(current_user.id) if hasattr(current_user, "id") and current_user.id else None
            provider = _get_ai_provider(user_id=_user_id) if is_ai_enabled(user_id=_user_id) else None
        except Exception:
            provider = None

        if provider is None:
            result["ai_fallback_triggered"] = True
            result["ai_status_message"] = "AI augmentation unavailable; local explanation retained."
            return result

        ok, message = provider.test_connection()
        if ok and provider.provider_type != "mock":
            prompt = (
                "Rewrite this verified AA-OS analytical explanation for clarity. "
                "Do not add facts, calculations, numbers, causal claims, or recommendations. "
                "Preserve uncertainty and limitations exactly. Plain text only.\n\n"
                + str(result)
            )
            ai_text = provider.generate_response(
                prompt,
                system_prompt=(
                    "You are a presentation layer only. The deterministic analysis and evidence are authoritative. "
                    "Never strengthen or alter the epistemic conclusion."
                ),
            )
            if ai_text and not ai_text.lower().startswith(("error:", "ollama execution error", "openai api error", "gemini api error")):
                result["ai_rewrite"] = ai_text.strip()
                result["mode"] = "AI_AUGMENTED"
                result["analysis_mode"] = "AI_AUGMENTED"
                result["ai_status_message"] = f"AI presentation augmentation active via {provider.provider_type}/{provider.model_name}; canonical analysis remains deterministic."
            else:
                result["ai_fallback_triggered"] = True
                result["ai_status_message"] = f"AI augmentation unavailable; local explanation retained. {message}"
        else:
            result["ai_fallback_triggered"] = True
            result["ai_status_message"] = f"AI augmentation unavailable; local explanation retained. {message}"

    return result


@router.get("/{investigation_id}/contract")
def get_investigation_contract(
    investigation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    inv = db.query(Investigation).filter(Investigation.id == investigation_id).first()
    if not inv:
        raise HTTPException(status_code=404, detail="Investigation not found.")
    verify_project_ownership(inv.project_id, current_user, db)
    contract = (db.query(InvestigationContract)
                .filter(InvestigationContract.investigation_id == investigation_id)
                .order_by(InvestigationContract.version.desc()).first())
    if not contract:
        raise HTTPException(status_code=404, detail="Analysis contract not available yet.")
    return {
        "id": contract.id, "investigation_id": contract.investigation_id, "version": contract.version,
        "status": contract.status, "original_question": contract.original_question,
        "normalized_question": contract.normalized_question, "problem_class": contract.problem_class,
        "claim_type": contract.claim_type, "target": contract.target_json,
        "explanatory_variables": contract.explanatory_variables_json, "population": contract.population_json,
        "grain": contract.grain_json, "scope": contract.scope_json, "time_window": contract.time_window_json,
        "estimand": contract.estimand_json, "assumptions": contract.assumptions_json,
        "data_requirements": contract.data_requirements_json, "candidate_methods": contract.candidate_methods_json,
        # v20-C4.2.3: proposal vs final.  selected_method is a projection of final_contract.
        "proposed_method": contract.proposed_method_json,
        "selected_method": contract.selected_method_json,
        "final_contract": contract.final_contract_json,
        "analytical_identity": contract.analytical_identity,
        "evidence_requirements": contract.evidence_requirements_json,
        "stopping_criteria": contract.stopping_criteria_json, "ambiguity_state": contract.ambiguity_state_json,
        "semantic_interpretations": contract.semantic_interpretations_json, "unresolved_questions": contract.unresolved_questions_json,
        "confidence": contract.confidence, "compiled_at": contract.compiled_at.isoformat() if contract.compiled_at else None,
    }


@router.get("/{investigation_id}/data-readiness")
def get_investigation_data_readiness(
    investigation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    inv = db.query(Investigation).filter(Investigation.id == investigation_id).first()
    if not inv:
        raise HTTPException(status_code=404, detail="Investigation not found.")
    verify_project_ownership(inv.project_id, current_user, db)
    readiness_rows = (db.query(InvestigationDataReadiness)
                 .filter(InvestigationDataReadiness.investigation_id == investigation_id)
                 .order_by(InvestigationDataReadiness.dataset_name.asc(), InvestigationDataReadiness.assessed_at.desc()).all())
    latest_readiness_by_dataset = {}
    for row in readiness_rows:
        latest_readiness_by_dataset.setdefault(row.dataset_name, row)
    scope_event = (db.query(InvestigationEvent)
                   .filter(InvestigationEvent.investigation_id == investigation_id,
                           InvestigationEvent.event_type == "investigation.dataset_scope_resolved")
                   .order_by(InvestigationEvent.sequence.desc()).first())
    scope_payload = scope_event.event_payload_json if scope_event else {}
    primary_dataset_name = (scope_payload or {}).get("primary_dataset")
    readiness = latest_readiness_by_dataset.get(primary_dataset_name) or next(iter(latest_readiness_by_dataset.values()), None)
    if not readiness:
        raise HTTPException(status_code=404, detail="Data readiness assessment not available yet.")
    return {
        "id": readiness.id, "investigation_id": readiness.investigation_id, "contract_id": readiness.contract_id,
        "dataset_name": readiness.dataset_name, "dataset_version_ids": readiness.dataset_version_ids_json,
        "source_fingerprints": readiness.source_fingerprints_json, "row_count": readiness.row_count,
        "column_count": readiness.column_count, "overall_quality_score": readiness.overall_quality_score,
        "fitness_verdict": readiness.fitness_verdict, "can_proceed": readiness.can_proceed,
        "checks": readiness.checks_json, "critical_issues": readiness.critical_issues_json,
        "warnings": readiness.warnings_json, "recommendations": readiness.recommendations_json,
        "assessed_at": readiness.assessed_at.isoformat() if readiness.assessed_at else None,
        "primary_dataset": primary_dataset_name,
        "datasets": [
            {
                "dataset_name": row.dataset_name,
                "dataset_version_ids": row.dataset_version_ids_json or [],
                "source_fingerprints": row.source_fingerprints_json or {},
                "row_count": row.row_count,
                "column_count": row.column_count,
                "overall_quality_score": row.overall_quality_score,
                "fitness_verdict": row.fitness_verdict,
                "can_proceed": row.can_proceed,
                "checks": row.checks_json or {},
                "critical_issues": row.critical_issues_json or [],
                "warnings": row.warnings_json or [],
                "recommendations": row.recommendations_json or [],
                "assessed_at": row.assessed_at.isoformat() if row.assessed_at else None,
            }
            for row in latest_readiness_by_dataset.values()
        ],
    }


# ---------------------------------------------------------------------------
# Human verification workflow: AA-OS does the analysis, the analyst verifies.
# The decision log is append-only (investigation_events), so no schema change
# and a full audit trail.  See packages/.../governance/verification_packet.py.
# ---------------------------------------------------------------------------
_VERIFICATION_ITEM_EVENT = "verification.item_reviewed"
_VERIFICATION_SIGNOFF_EVENT = "verification.signed_off"
_VERIFIABLE_STATES = {"COMPLETED", "PARTIAL"}


def _append_verification_event(db: Session, investigation_id: str, event_type: str, payload: Dict[str, Any]) -> None:
    """Append one event with a concurrency-safe sequence (same scheme as the checkpointer)."""
    from sqlalchemy import func
    from sqlalchemy.exc import IntegrityError

    safe = json.loads(json.dumps(payload, default=str))
    for attempt in range(5):
        max_seq = db.query(func.max(InvestigationEvent.sequence)).filter(
            InvestigationEvent.investigation_id == investigation_id
        ).scalar()
        db.add(InvestigationEvent(
            id=f"EVT-{gen_uuid()[:8]}",
            sequence=(max_seq or 0) + 1,
            investigation_id=investigation_id,
            event_type=event_type,
            event_payload_json=safe,
            timestamp=datetime.now(timezone.utc),
        ))
        try:
            db.commit()
            return
        except IntegrityError:
            db.rollback()
            if attempt == 4:
                raise HTTPException(status_code=503, detail="Could not record the review; please retry.")


def _verification_state(db: Session, investigation_id: str, current_user: User) -> Dict[str, Any]:
    # get_investigation performs the 404 and project-ownership checks.
    payload = get_investigation(investigation_id, current_user=current_user, db=db)
    if str(payload.get("status", "")).upper() not in _VERIFIABLE_STATES:
        raise HTTPException(
            status_code=409,
            detail=f"Investigation is {payload.get('status')}; verification opens once it has completed.",
        )
    packet = build_verification_packet(payload)
    rows = (
        db.query(InvestigationEvent)
        .filter(
            InvestigationEvent.investigation_id == investigation_id,
            InvestigationEvent.event_type.in_([_VERIFICATION_ITEM_EVENT, _VERIFICATION_SIGNOFF_EVENT]),
        )
        .order_by(InvestigationEvent.sequence.asc())
        .all()
    )
    decisions, signoffs = [], []
    for r in rows:
        rec = dict(r.event_payload_json or {})
        rec["at"] = r.timestamp.isoformat() if r.timestamp else None
        (decisions if r.event_type == _VERIFICATION_ITEM_EVENT else signoffs).append(rec)
    review = evaluate_review(packet, decisions)
    return {"packet": packet, "review": review, "signoff": current_signoff(packet, signoffs)}


@router.get("/{investigation_id}/verification")
def get_verification_packet(
    investigation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """The verifier's worklist: risk-ranked items, current decisions, and sign-off status."""
    return _verification_state(db, investigation_id, current_user)


@router.post("/{investigation_id}/verification/decisions")
def record_verification_decision(
    investigation_id: str,
    body: VerificationDecisionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Record the human verifier's decision on one item."""
    state = _verification_state(db, investigation_id, current_user)
    packet = state["packet"]
    item = next((i for i in packet["items"] if i["id"] == body.item_id), None)
    error = validate_decision(packet, body.item_id, body.decision, body.comment)
    if error:
        raise HTTPException(status_code=422, detail=error)
    if body.item_fingerprint and body.item_fingerprint != item["fingerprint"]:
        raise HTTPException(
            status_code=409,
            detail="This item changed since you opened it (the investigation was re-run). Reload and review it again.",
        )
    _append_verification_event(db, investigation_id, _VERIFICATION_ITEM_EVENT, {
        "item_id": body.item_id,
        "item_fingerprint": item["fingerprint"],
        "decision": body.decision,
        "comment": (body.comment or "").strip(),
        "reviewer": {"id": current_user.id, "email": current_user.email},
        "packet_fingerprint": packet["packet_fingerprint"],
    })
    return _verification_state(db, investigation_id, current_user)


@router.post("/{investigation_id}/verification/signoff")
def sign_off_verification(
    investigation_id: str,
    body: VerificationSignoffRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Record the overall verification outcome, bound to the exact result reviewed."""
    state = _verification_state(db, investigation_id, current_user)
    packet, review = state["packet"], state["review"]
    error = validate_signoff(review, body.outcome, body.comment)
    if error:
        raise HTTPException(status_code=422, detail=error)
    if body.packet_fingerprint and body.packet_fingerprint != packet["packet_fingerprint"]:
        raise HTTPException(
            status_code=409,
            detail="The result changed since you opened it (the investigation was re-run). Reload and review again.",
        )
    _append_verification_event(db, investigation_id, _VERIFICATION_SIGNOFF_EVENT, {
        "outcome": body.outcome,
        "comment": (body.comment or "").strip(),
        "reviewer": {"id": current_user.id, "email": current_user.email},
        "packet_fingerprint": packet["packet_fingerprint"],
        "manifest_hash": packet["manifest_hash"],
        "machine_verdict": packet["machine_claim"]["verdict"],
        "decisions_snapshot": {
            k: {"decision": v["decision"], "comment": v["comment"]} for k, v in review["per_item"].items() if not v["stale"]
        },
    })
    return _verification_state(db, investigation_id, current_user)
