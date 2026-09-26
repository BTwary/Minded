"""Canonical-state consistency validation.

Detects inconsistent scientific state WITHOUT silently repairing it.
Returns a structured StateConsistencyReport so the caller can fail safely at
investigation boundaries rather than continue with corrupted state.
"""
from typing import Any, Dict, List

from packages.schemas.src.analysis import (
    CanonicalInvestigationState,
    ConsistencyError,
    PredictionStatus,
    StateConsistencyReport,
)

_RESOLVED_STATUSES = {PredictionStatus.SUPPORTED, PredictionStatus.REFUTED}


def validate_state(
    state: CanonicalInvestigationState,
    runtime_hypotheses: Dict[str, Any],
    runtime_predictions: Dict[str, Any],
) -> StateConsistencyReport:
    """Validate canonical investigation state for internal consistency."""
    errors: List[ConsistencyError] = []

    hypothesis_codes = set(runtime_hypotheses.keys())
    prediction_ids = set(runtime_predictions.keys())

    # 1. Duplicate prediction IDs
    pred_id_list = [p.prediction_id for p in state.predictions]
    seen_pred: Dict[str, int] = {}
    dup_preds: List[str] = []
    for pid in pred_id_list:
        seen_pred[pid] = seen_pred.get(pid, 0) + 1
        if seen_pred[pid] == 2:
            dup_preds.append(pid)
    if dup_preds:
        errors.append(ConsistencyError(
            error_code="duplicate_prediction_id",
            message=f"Duplicate prediction IDs present in canonical state: {sorted(dup_preds)}",
            offending_ids=sorted(dup_preds),
        ))

    # 2. Duplicate experiment IDs
    exp_id_list = [e.experiment_id for e in state.completed_experiments]
    seen_exp: Dict[str, int] = {}
    dup_exps: List[str] = []
    for eid in exp_id_list:
        seen_exp[eid] = seen_exp.get(eid, 0) + 1
        if seen_exp[eid] == 2:
            dup_exps.append(eid)
    if dup_exps:
        errors.append(ConsistencyError(
            error_code="duplicate_experiment_id",
            message=f"Duplicate experiment IDs present in canonical state: {sorted(dup_exps)}",
            offending_ids=sorted(dup_exps),
        ))

    # 3. Prediction references missing hypothesis
    for p in state.predictions:
        hyp_id = p.hypothesis_id or ""
        matched = hyp_id in hypothesis_codes or any(hyp_id.endswith(f"_{code}") for code in hypothesis_codes)
        if not matched and hypothesis_codes:
            errors.append(ConsistencyError(
                error_code="prediction_references_missing_hypothesis",
                message=f"Prediction {p.prediction_id} references hypothesis '{hyp_id}' which is not in canonical state.",
                offending_ids=[p.prediction_id, hyp_id],
            ))

    # 4. Hypothesis references nonexistent prediction
    for h in state.hypotheses:
        for pid in (h.prediction_ids or []):
            if pid not in prediction_ids and pid not in pred_id_list:
                errors.append(ConsistencyError(
                    error_code="hypothesis_references_nonexistent_prediction",
                    message=f"Hypothesis {h.id} references prediction '{pid}' which is not in canonical state.",
                    offending_ids=[h.id, pid],
                ))

    # 5. Prediction marked resolved but no observation
    # A resolved prediction must be backed by either an actual_observed_result
    # already attached, or a target_experiment_id that points at a *real*
    # raw_observations entry. A target_experiment_id alone is not sufficient
    # -- it can reference an experiment that was never actually observed
    # (e.g. a stale/misassigned ID), which would otherwise pass this check
    # with no evidence behind the resolution.
    observation_exp_ids = {o.experiment_id for o in state.raw_observations}
    for p in state.predictions:
        if p.status in _RESOLVED_STATUSES:
            has_real_observation = bool(p.actual_observed_result) or (
                bool(p.target_experiment_id) and p.target_experiment_id in observation_exp_ids
            )
            if not has_real_observation:
                errors.append(ConsistencyError(
                    error_code="prediction_marked_resolved_but_no_observation",
                    message=f"Prediction {p.prediction_id} is {p.status} but has no target_experiment_id matching a real observation, and no observation result.",
                    offending_ids=[p.prediction_id],
                ))

    # 6. Executed experiment absent from history
    executed_ids = set(state.executed_experiment_ids)
    completed_ids = {e.experiment_id for e in state.completed_experiments}
    missing_completed = executed_ids - completed_ids
    if missing_completed and completed_ids:
        errors.append(ConsistencyError(
            error_code="executed_experiment_absent_from_history",
            message=f"Experiments in executed_experiment_ids are absent from completed_experiments: {sorted(missing_completed)}",
            offending_ids=sorted(missing_completed),
        ))

    # 7. Refuted hypothesis unexplained posterior increase
    for h_code, h in runtime_hypotheses.items():
        belief = getattr(h, "belief_state", "")
        post = getattr(h, "posterior_probability", 0.0)
        prior = getattr(h, "prior_probability", 0.0)
        if belief == "REFUTED" and post > prior + 0.15:
            errors.append(ConsistencyError(
                error_code="refuted_hypothesis_unexplained_posterior_increase",
                message=f"Hypothesis {h_code} is REFUTED but posterior ({post:.3f}) increased significantly above prior ({prior:.3f}).",
                offending_ids=[h_code],
            ))

    return StateConsistencyReport(
        is_consistent=(len(errors) == 0),
        errors=errors,
    )
