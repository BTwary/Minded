"""Deterministic scientific-state snapshot for replay/audit/recovery checks."""
from __future__ import annotations
import hashlib, json
from typing import Any

from apps.api.src.models.entities import Hypothesis, Prediction, Experiment, Evidence, BeliefUpdate, InvestigationVerdict, InvestigationContract, DatasetVersion, DatasetTransformation


def _rows(session, model, investigation_id: str, fields: list[str]) -> list[dict[str, Any]]:
    rows = session.query(model).filter(model.investigation_id == investigation_id).all()
    out = []
    for row in rows:
        item = {}
        for f in fields:
            v = getattr(row, f, None)
            if isinstance(v, (dict, list)):
                item[f] = v
            else:
                item[f] = str(v) if v is not None else None
        out.append(item)
    return sorted(out, key=lambda x: json.dumps(x, sort_keys=True, default=str))


def build_scientific_state_snapshot(session, investigation_id: str, runtime_state: Any = None) -> dict[str, Any]:
    state = {
        "investigation_id": investigation_id,
        "hypotheses": _rows(session, Hypothesis, investigation_id, ["id", "hypothesis_code", "canonical_identity", "analytical_identity", "posterior_probability", "status", "belief_state"]),
        "predictions": _rows(session, Prediction, investigation_id, ["id", "prediction_code", "hypothesis_id", "status", "actual_observed_result_json", "evaluation_reason"]),
        "experiments": _rows(session, Experiment, investigation_id, ["id", "test_code", "experiment_role", "analytical_identity", "hypothesis_canonical_identity", "contract_id", "hypothesis_id", "status", "fingerprint", "rationale", "execution_result_json", "verification_result_json"]),
        "evidence": _rows(session, Evidence, investigation_id, ["id", "experiment_id", "hypothesis_id", "validation_status", "effect_size", "calculation_summary", "evidence_identity_hash", "analytical_identity"]),
        "belief_updates": _rows(session, BeliefUpdate, investigation_id, ["id", "hypothesis_id", "evidence_id", "prior_probability", "bayes_factor", "likelihood_p", "posterior_probability", "entropy_delta", "update_step_index"]),
        "verdicts": _rows(session, InvestigationVerdict, investigation_id, ["id", "verdict_type", "confidence_score"]),
    }
    contracts = (session.query(InvestigationContract).filter(InvestigationContract.investigation_id == investigation_id).order_by(InvestigationContract.version.asc()).all())
    state["contracts"] = [{
        "id": c.id, "version": c.version, "status": c.status, "original_question": c.original_question,
        "problem_class": c.problem_class, "claim_type": c.claim_type, "target_json": c.target_json,
        "population_json": c.population_json, "grain_json": c.grain_json, "scope_json": c.scope_json,
        "time_window_json": c.time_window_json, "estimand_json": c.estimand_json,
        "proposed_method_json": c.proposed_method_json, "selected_method_json": c.selected_method_json,
        "final_contract_json": c.final_contract_json, "analytical_identity": c.analytical_identity,
        "evidence_requirements_json": c.evidence_requirements_json,
        "stopping_criteria_json": c.stopping_criteria_json, "execution_state_json": c.execution_state_json,
    } for c in contracts]
    if runtime_state is not None:
        try:
            state["canonical_runtime_state"] = runtime_state.model_dump(mode="json") if hasattr(runtime_state, "model_dump") else dict(runtime_state)
        except Exception:
            state["canonical_runtime_state"] = None
    canonical = json.dumps(state, sort_keys=True, separators=(",", ":"), default=str)
    state["snapshot_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return state
