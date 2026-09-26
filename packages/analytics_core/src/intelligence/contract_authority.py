"""Authoritative durable analytical-contract state for AA-OS.

The controller may emit transient events, but these helpers make the database
contract the durable source of truth for the current analytical phase and
replanning lineage. They deliberately fail closed on a missing contract.
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from sqlalchemy.orm import Session

from apps.api.src.models.entities import Evidence, Experiment, Hypothesis, Investigation, InvestigationContract
from packages.analytics_core.src.intelligence.analytical_identity import (
    AnalyticalIdentityError,
    FinalAnalyticalContract,
    method_identifier,
)


class ContractImmutabilityError(AnalyticalIdentityError):
    """A finalized contract's analytical decision may not be changed in place.
    A material change must create a new contract version (supersede_and_replan)."""


class ContractIntegrityError(AnalyticalIdentityError):
    """The persisted contract is internally inconsistent (fail closed)."""


def _selected_method_projection(final: FinalAnalyticalContract) -> Dict[str, Any]:
    """The ONLY thing ever written to selected_method_json: a projection of the
    final contract, so the column can never carry a competing decision."""
    return {
        "method_code": final.selected_method_code,
        "method_identifier": (method_identifier(final.selected_method_code) if final.selected_method_code else None),
        "fallback_used": final.fallback_used,
        "method_family": final.method_family,
        "claim_ceiling": final.claim_ceiling,
        "verification_regime": final.verification_regime,
        "analytical_identity": final.analytical_identity,
    }


def finalize_contract(db: Session, contract_id: str, final: FinalAnalyticalContract) -> str:
    """Persist the FINAL reconciled analytical decision on a contract row.

    Writes ``final_contract_json`` / ``analytical_identity`` / ``selected_method_json``
    together, in one commit.  Idempotent for the identical final contract; raises
    ``ContractImmutabilityError`` if the row is already finalized with a different
    analytical identity.  Deliberately NOT best-effort: the caller must treat any
    exception as "no final contract" and refuse to execute.
    """
    contract = db.query(InvestigationContract).filter(InvestigationContract.id == contract_id).first()
    if contract is None:
        raise ValueError(f"Contract {contract_id} not found")
    if contract.final_contract_json:
        # v20-C4.2.4 section 3: identical identity is NOT sufficient by
        # itself to accept an already-finalized contract. The persisted row
        # must be fully re-derived and re-validated every time, so a
        # corrupted final_contract_json / analytical_identity / selected_
        # method_json can never be silently accepted just because the
        # caller's freshly-computed `final` happens to match the (possibly
        # corrupted) identity column. load_final_contract() re-derives the
        # identity from final_contract_json's own stored components and
        # cross-checks analytical_identity + selected_method_json against
        # that re-derivation, raising ContractIntegrityError on any mismatch
        # -- exactly the "fail closed" path this invariant requires.
        persisted = load_final_contract(db, contract_id)
        if persisted.analytical_identity != final.analytical_identity:
            raise ContractImmutabilityError(
                f"contract {contract_id} is already finalized as {persisted.analytical_identity}; "
                f"refusing to overwrite with {final.analytical_identity}. "
                "A materially different decision requires a new contract version."
            )
        return persisted.analytical_identity
    contract.final_contract_json = final.to_dict()
    contract.analytical_identity = final.analytical_identity
    contract.selected_method_json = _selected_method_projection(final)
    contract.finalized_at = datetime.now(timezone.utc)
    db.commit()
    return contract.analytical_identity


def load_final_contract(db: Session, contract_id: str) -> FinalAnalyticalContract:
    """Reconstruct the final contract from the database, verifying integrity.

    Fails closed when the contract is missing, not yet finalized, or when any of
    the stored projections (identity column, selected_method_json) disagree with
    the authoritative ``final_contract_json``.
    """
    contract = db.query(InvestigationContract).filter(InvestigationContract.id == contract_id).first()
    if contract is None:
        raise ValueError(f"Contract {contract_id} not found")
    if not contract.final_contract_json:
        raise ContractIntegrityError(f"contract {contract_id} has no final analytical contract (not finalized)")
    final = FinalAnalyticalContract.from_dict(contract.final_contract_json)  # re-derives + verifies
    if contract.analytical_identity != final.analytical_identity:
        raise ContractIntegrityError("analytical_identity column disagrees with final_contract_json")
    if contract.selected_method_json != _selected_method_projection(final):
        raise ContractIntegrityError("selected_method_json disagrees with final_contract_json")
    return final


def load_active_final_contract(db: Session, investigation_id: str) -> FinalAnalyticalContract:
    inv = db.query(Investigation).filter(Investigation.id == investigation_id).first()
    if inv is None or not inv.active_contract_id:
        raise ValueError(f"Investigation {investigation_id} has no active contract")
    return load_final_contract(db, inv.active_contract_id)


def set_phase(
    db: Session,
    investigation_id: str,
    phase: str,
    *,
    state_patch: Optional[Dict[str, Any]] = None,
) -> str:
    inv = db.query(Investigation).filter(Investigation.id == investigation_id).first()
    if inv is None:
        raise ValueError(f"Investigation {investigation_id} not found")
    if not inv.active_contract_id:
        raise ValueError(f"Investigation {investigation_id} has no active analytical contract")
    contract = db.query(InvestigationContract).filter(InvestigationContract.id == inv.active_contract_id).first()
    if contract is None:
        raise ValueError(f"Active contract {inv.active_contract_id} not found")
    inv.current_phase = phase
    contract.current_phase = phase
    current = dict(contract.execution_state_json or {})
    current.update(state_patch or {})
    current["current_phase"] = phase
    current["updated_at"] = datetime.now(timezone.utc).isoformat()
    contract.execution_state_json = current
    db.commit()
    return contract.id


_ESTIMAND_MATERIAL_FIELDS = (
    "target_column", "predictor_columns", "comparison_dimension",
    "time_column", "numerator_column", "denominator_column", "weight_column",
)


def _stale_projection(field_name: str, version: int) -> Dict[str, Any]:
    """Marker written in place of a legacy projection this replan cannot
    faithfully regenerate from the new final contract (rather than either
    fabricating a value or silently carrying the superseded one forward).
    A downstream consumer that reads this must not treat it as an
    analytical answer -- ``final_contract_json`` is the authority."""
    return {
        "_stale_projection": True,
        "_field": field_name,
        "_reason": (
            "this replan materially changed the analytical contract but the "
            "caller did not supply enough information to regenerate this "
            "legacy projection; it is intentionally invalidated rather than "
            "left carrying the superseded analytical interpretation"
        ),
        "_superseded_by_contract_version": version,
    }


def supersede_and_replan(
    db: Session,
    investigation_id: str,
    *,
    replan_reason: str,
    candidate_experiments: list[Dict[str, Any]],
    active_hypotheses: list[str],
    phase: str = "REPLANNING",
    new_final_contract: Optional[FinalAnalyticalContract] = None,
    new_semantic_binding_set: Optional[Any] = None,
    new_dataset_diagnostics: Optional[Dict[str, Any]] = None,
) -> str:
    """``new_semantic_binding_set`` (a ``SemanticBindingSet``, with a
    ``to_dict_list()`` method) and ``new_dataset_diagnostics`` (a dict with
    optional ``grain``/``scope`` keys) let the caller supply fresh
    full-fidelity data for legacy projections that cannot be losslessly
    rebuilt from ``FinalAnalyticalContract`` alone (its ``bindings`` are a
    reduced (role, table, column) projection with no resolution_status/
    confidence/provenance, and it carries no dataset-schema/grain data at
    all). When the analytical claim materially changes and the caller did
    not supply that fresh data, legacy projections are regenerated from
    whatever the new final contract genuinely knows (marked as a reduced
    projection where fidelity was lost) or explicitly marked stale (where
    nothing in the final contract can regenerate them, e.g. grain/scope
    after a dataset change) -- never silently carried forward as if they
    still described the superseded analytical claim.
    """
    inv = db.query(Investigation).filter(Investigation.id == investigation_id).first()
    if inv is None or not inv.active_contract_id:
        raise ValueError(f"Investigation {investigation_id} has no active contract")
    parent = db.query(InvestigationContract).filter(InvestigationContract.id == inv.active_contract_id).first()
    if parent is None:
        raise ValueError(f"Active contract {inv.active_contract_id} not found")
    # v20-C4.2.3: a replan may change the PROPOSAL (candidate experiments) without
    # changing the analytical claim.  The child inherits the parent's final
    # contract verbatim (same analytical identity) unless the caller supplies a
    # materially different one, in which case the child carries the new identity
    # and the parent keeps its own, unchanged.
    parent_final: Optional[FinalAnalyticalContract] = None
    if parent.final_contract_json:
        parent_final = load_final_contract(db, parent.id)  # fail closed on a corrupt parent
    child_final = new_final_contract if new_final_contract is not None else parent_final
    material_diff: list = []
    if parent_final is not None and child_final is not None:
        material_diff = parent_final.material_diff(child_final)

    # v20-C4.2.4 section 2: every legacy proposal-era projection whose value
    # is a function of the analytical claim must be regenerated from the NEW
    # contract when a material change touches it -- never left as a copy of
    # the superseded contract's projection. A field not itself touched by
    # the diff is left as the parent's value (byte-identical-identity path).
    estimand_changed = any(f in material_diff for f in _ESTIMAND_MATERIAL_FIELDS)
    bindings_changed = "bindings" in material_diff
    dataset_changed = "dataset_identity" in material_diff
    new_version = int(parent.version) + 1

    if estimand_changed and child_final is not None:
        estimand_json = {
            **child_final.estimand_components(),
            "_derived_from": "final_contract",
            "_note": "regenerated from the new final analytical contract at replan",
        }
        time_col = (
            new_semantic_binding_set.projected_time_column()
            if (new_semantic_binding_set and hasattr(new_semantic_binding_set, "projected_time_column") and new_semantic_binding_set.projected_time_column() is not None)
            else child_final.time_column
        )
        time_window_json = {"time_column": time_col}
    else:
        time_window_json = parent.time_window_json
        estimand_json = parent.estimand_json

    if bindings_changed:
        if new_semantic_binding_set is not None:
            # Full-fidelity data supplied by the caller: use it as-is.
            semantic_bindings_json = new_semantic_binding_set.to_dict_list()
        else:
            # FinalAnalyticalContract.bindings is a reduced (role, table,
            # column) projection -- it carries no resolution_status/
            # confidence/provenance, so rebuilding a full SemanticBinding
            # record from it would mean fabricating those fields. Persist
            # the reduced projection instead, explicitly marked as such, so
            # the superseded (and now-wrong) full bindings never survive
            # under the new identity, without inventing data this system
            # does not have.
            semantic_bindings_json = [
                {"role": role, "table": table, "column": column, "_reduced_projection": True}
                for (role, table, column) in child_final.bindings
            ]
    else:
        semantic_bindings_json = parent.semantic_bindings_json

    if dataset_changed:
        diagnostics = new_dataset_diagnostics or {}
        grain_json = diagnostics.get("grain", _stale_projection("grain_json", new_version))
        scope_json = diagnostics.get("scope", _stale_projection("scope_json", new_version))
    else:
        grain_json = parent.grain_json
        scope_json = parent.scope_json

    parent.status = "SUPERSEDED"
    parent.superseded_at = datetime.now(timezone.utc)
    version = new_version
    child_target_col = (
        new_semantic_binding_set.projected_target_column()
        if (new_semantic_binding_set and hasattr(new_semantic_binding_set, "projected_target_column") and new_semantic_binding_set.projected_target_column() is not None)
        else (child_final.target_column if child_final is not None else None)
    )
    child_expl_cols = (
        list(new_semantic_binding_set.projected_explanatory_columns())
        if (new_semantic_binding_set and hasattr(new_semantic_binding_set, "projected_explanatory_columns") and new_semantic_binding_set.projected_explanatory_columns())
        else (list(child_final.predictor_columns) if child_final is not None else [])
    )
    child = InvestigationContract(
        investigation_id=investigation_id,
        version=version,
        parent_contract_id=parent.id,
        status="ACTIVE",
        original_question=parent.original_question,
        normalized_question=parent.normalized_question,
        problem_class=(child_final.canonical_task if ("canonical_task" in material_diff and child_final is not None) else parent.problem_class),
        claim_type=parent.claim_type,
        # Legacy problem-statement columns are proposal-era diagnostics; when the
        # replan materially changes the claim they are re-derived from the new
        # final contract and semantic binding set so the row never carries two competing truths.
        target_json=(
            {"target": child_target_col}
            if ("target_column" in material_diff and child_target_col is not None) else parent.target_json
        ),
        explanatory_variables_json=(
            child_expl_cols
            if ("predictor_columns" in material_diff and child_expl_cols)
            else parent.explanatory_variables_json
        ),
        population_json=parent.population_json,
        grain_json=grain_json,
        scope_json=scope_json,
        time_window_json=time_window_json,
        estimand_json=estimand_json,
        assumptions_json=parent.assumptions_json,
        data_requirements_json=parent.data_requirements_json,
        candidate_methods_json=candidate_experiments,
        # The replanned candidate is a PROPOSAL.  It never becomes the selected
        # method; selected_method_json is the projection of the child's final contract.
        proposed_method_json=(candidate_experiments[0] if candidate_experiments else parent.proposed_method_json),
        selected_method_json=(_selected_method_projection(child_final) if child_final is not None else None),
        final_contract_json=(child_final.to_dict() if child_final is not None else None),
        analytical_identity=(child_final.analytical_identity if child_final is not None else None),
        finalized_at=(datetime.now(timezone.utc) if child_final is not None else None),
        evidence_requirements_json=parent.evidence_requirements_json,
        stopping_criteria_json=parent.stopping_criteria_json,
        ambiguity_state_json=parent.ambiguity_state_json,
        semantic_interpretations_json=parent.semantic_interpretations_json,
        unresolved_questions_json=parent.unresolved_questions_json,
        confidence=parent.confidence,
        semantic_bindings_json=semantic_bindings_json,
        current_phase=phase,
        execution_state_json={
            # v19 fix: a replan creates a new (child) contract row per the
            # immutable-contract-per-version design, but that must not
            # silently drop durable, mutable-by-design execution-state keys
            # already recorded on the parent (e.g. the reconciled_method_decision
            # this phase's fix writes right after MethodSelectionEngine
            # reconciliation). Previously this dict was built from scratch,
            # so any such key present on the parent was lost the moment a
            # replan happened -- verified directly: a live investigation
            # that replanned mid-run left its reconciled_method_decision
            # stranded on the superseded v1 row, invisible from the (now
            # active) v2 row any consumer actually reads. Start from the
            # parent's state and only overlay the fields this replan
            # actually changes.
            # v20-C4.2.3: the legacy reconciled_method_decision scratch key is a
            # competing copy of the final decision and is never inherited.
            **{k: v for k, v in dict(parent.execution_state_json or {}).items() if k != "reconciled_method_decision"},
            "replan_material_diff": list(material_diff),
            "replan_reason": replan_reason,
            "active_hypotheses": active_hypotheses,
            "candidate_count": len(candidate_experiments),
            "current_phase": phase,
        },
        last_replan_reason=replan_reason,
        last_replanned_at=datetime.now(timezone.utc),
    )
    db.add(child)
    db.flush()
    inv.active_contract_id = child.id
    inv.contract_revision = version
    inv.current_phase = phase
    db.commit()
    return child.id


def trace_evidence_to_contract(db: Session, evidence_id: str) -> Dict[str, Any]:
    """Trace Evidence -> Experiment -> Hypothesis -> Analytical Identity -> Final Contract.

    Every link is checked against DURABLE identity columns.  Nothing is inferred
    from narrative text (``Evidence.statement`` / ``Hypothesis.statement`` are never
    read), and any break in the chain raises ``ContractIntegrityError`` -- a
    missing identity is an error, not something to reconstruct.
    """
    ev = db.query(Evidence).filter(Evidence.id == evidence_id).first()
    if ev is None:
        raise ValueError(f"Evidence {evidence_id} not found")
    if not ev.experiment_id:
        raise ContractIntegrityError(f"evidence {evidence_id} references no experiment")
    exp = db.query(Experiment).filter(Experiment.id == ev.experiment_id).first()
    if exp is None:
        raise ContractIntegrityError(f"evidence {evidence_id}: experiment {ev.experiment_id} not found")
    if not exp.contract_id:
        raise ContractIntegrityError(f"experiment {exp.id} is not attributed to a contract")
    identity = exp.analytical_identity
    if not identity:
        raise ContractIntegrityError(f"experiment {exp.id} carries no analytical identity")
    final = load_final_contract(db, exp.contract_id)  # re-derives + verifies the contract
    if identity not in (set(final.pair_identities().values()) | {final.analytical_identity}):
        raise ContractIntegrityError(
            f"experiment {exp.id} identity {identity[:12]} matches no analytical pair of contract {exp.contract_id}"
        )
    if ev.analytical_identity != identity:
        raise ContractIntegrityError(
            f"evidence {evidence_id} identity does not match its experiment's identity"
        )
    hyp = db.query(Hypothesis).filter(Hypothesis.id == exp.hypothesis_id).first()
    if hyp is None:
        raise ContractIntegrityError(f"experiment {exp.id}: hypothesis {exp.hypothesis_id} not found")
    if exp.hypothesis_canonical_identity and hyp.canonical_identity != exp.hypothesis_canonical_identity:
        raise ContractIntegrityError(f"experiment {exp.id} references a different hypothesis than it targets")
    if exp.experiment_role == "PRIMARY" and hyp.analytical_identity != identity:
        raise ContractIntegrityError(
            f"PRIMARY experiment {exp.id} and its hypothesis {hyp.id} belong to different analytical identities"
        )
    return {
        "evidence_id": ev.id, "experiment_id": exp.id, "experiment_role": exp.experiment_role,
        "hypothesis_id": hyp.id, "analytical_identity": identity,
        "contract_id": exp.contract_id, "contract_analytical_identity": final.analytical_identity,
    }
