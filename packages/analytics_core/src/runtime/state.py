"""InvestigationStateManager: Single authoritative scientific state management for AA-OS kernel.

Phase 5 (Canonical State Unification): the manager OWNS the runtime hypothesis
and prediction registries (keyed by hypothesis_code and prediction_id respectively,
so a single hypothesis may carry MULTIPLE predictions). The controller reads the next
loop state FROM the manager; it does not maintain a parallel authoritative copy of
hypotheses / pending predictions / executed experiments / uncertainty.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from packages.schemas.src.analysis import (
    CanonicalInvestigationState,
    CalculationTraceSchema,
    ConsistencyError,
    EpistemicClaimType,
    EvidenceLedgerSchema,
    FirstClassExperiment,
    HypothesisSchema,
    HypothesisStatus,
    ObjectiveType,
    PredictionEvidenceRecord,
    PredictionSchema,
    PredictionStatus,
    RawObservationRecord,
    StateConsistencyReport,
    UncertaintyStateSchema,
)
from packages.shared.src.enums import AnalyticalVerdict
from packages.analytics_core.src.intelligence.state_validation import validate_state
from packages.analytics_core.src.intelligence.hypothesis_consolidation import (
    HypothesisConsolidator,
)


_BELIEF_STATE_TO_LIFECYCLE = {
    "untested": HypothesisStatus.ACTIVE,
    "active": HypothesisStatus.ACTIVE,
    "proposed": HypothesisStatus.ACTIVE,
    "highly_likely": HypothesisStatus.SUPPORTED,
    "supported": HypothesisStatus.SUPPORTED,
    "weakened": HypothesisStatus.WEAKENED,
    "refuted": HypothesisStatus.REFUTED,
    "falsified": HypothesisStatus.REFUTED,
    "inconclusive": HypothesisStatus.INCONCLUSIVE,
    "retired": HypothesisStatus.RETIRED,
}


def _to_hypothesis_schema(h: Any) -> HypothesisSchema:
    lifecycle = _BELIEF_STATE_TO_LIFECYCLE.get(str(getattr(h, "belief_state", "active")).lower(), HypothesisStatus.ACTIVE)
    return HypothesisSchema(
        id=h.hypothesis_code,
        statement=h.claim,
        rationale=h.mechanism,
        priority=max(0.0, min(1.0, h.posterior_probability)),
        status=h.belief_state,
        lifecycle_status=lifecycle,
        prior_probability=h.prior_probability,
        posterior_probability=h.posterior_probability,
        target_metric=h.target_metric or None,
        target_dimension=h.target_dimension or None,
        target_value=getattr(h, "target_value", None),
        mechanism=getattr(h, "mechanism_detail", "") or h.mechanism,
        source_evidence=list(getattr(h, "source_evidence", []) or []),
        parent_hypotheses=list(getattr(h, "parent_hypotheses", []) or []),
        generated_reason=getattr(h, "generated_reason", "") or None,
        prediction_ids=list(getattr(h, "prediction_ids", []) or []),
        supporting_prediction_count=getattr(h, "supporting_prediction_count", 0),
        refuted_prediction_count=getattr(h, "refuted_prediction_count", 0),
        unresolved_prediction_count=getattr(h, "unresolved_prediction_count", 0),
    )


def _to_prediction_schema(prediction: Any) -> PredictionSchema:
    status_val = getattr(prediction, "status", "PENDING")
    try:
        status_enum = PredictionStatus(status_val) if isinstance(status_val, str) else PredictionStatus.PENDING
    except ValueError:
        status_enum = PredictionStatus.PENDING
    return PredictionSchema(
        prediction_id=prediction.prediction_id,
        hypothesis_id=prediction.hypothesis_id,
        statement=prediction.statement,
        target_metric=prediction.target_metric,
        target_dimension=prediction.target_dimension,
        expected_direction=prediction.expected_direction,
        expected_value=prediction.expected_value,
        expected_range=prediction.expected_range,
        threshold=prediction.threshold,
        expected_relationship=prediction.expected_relationship,
        expected_effect=prediction.expected_effect,
        confidence=prediction.confidence,
        testability=prediction.testability,
        status=status_enum,
        target_experiment_id=prediction.target_experiment_id,
        actual_observed_result=prediction.actual_observed_result,
        evaluation_reason=prediction.evaluation_reason,
    )


@dataclass
class EvidenceEntry:
    claim: str
    metric: Optional[str] = None
    value: Any = None
    proof: Optional[str] = None


class EvidenceLedgerContainer:
    def __init__(self):
        self.entries: List[EvidenceEntry] = []

    def append(self, entry: EvidenceEntry):
        self.entries.append(entry)


class InvestigationStateManager:
    """Authoritative state manager governing the lifecycle of CanonicalInvestigationState."""

    def __init__(self, investigation_id: str, question: str = "", objective: ObjectiveType = ObjectiveType.DESCRIBE):
        self.state = CanonicalInvestigationState(
            investigation_id=investigation_id,
            original_question=question,
            interpreted_objective=objective,
            evidence_ledger=EvidenceLedgerSchema(investigation_id=investigation_id),
        )
        self._hypotheses: Dict[str, Any] = {}
        self._predictions: Dict[str, Any] = {}
        self._executed_experiment_ids: List[str] = []
        self._executed_fingerprints: List[str] = []
        self._failed_experiment_ids: List[str] = []
        self._evidence_ledger_container = EvidenceLedgerContainer()
        self.parent_investigation_id: Optional[str] = None
        self.fork_turn_index: Optional[int] = None
        self.fork_constraints: Optional[Dict[str, Any]] = None
        self.current_turn: int = 0

    @property
    def evidence_ledger(self) -> EvidenceLedgerContainer:
        return self._evidence_ledger_container

    def advance_turn(self) -> None:
        self.current_turn += 1
        self.state.updated_at = datetime.now(timezone.utc)

    def append_evidence(self, claim: str, metric: Optional[str] = None, value: Any = None, proof: Optional[str] = None) -> None:
        entry = EvidenceEntry(claim=claim, metric=metric, value=value, proof=proof)
        self._evidence_ledger_container.append(entry)
        self.state.updated_at = datetime.now(timezone.utc)

    def register_hypothesis(self, hyp_id: str, prior: float = 0.5, description: str = "") -> None:
        class SimpleHypothesis:
            def __init__(self, code, pr, desc):
                self.hypothesis_code = code
                self.claim = desc or code
                self.mechanism = desc
                self.prior_probability = float(pr)
                self.posterior_probability = float(pr)
                self.belief_state = "ACTIVE"
                self.target_metric = None
                self.target_dimension = None

            @property
            def posterior(self) -> float:
                return self.posterior_probability

            @posterior.setter
            def posterior(self, val: float):
                self.posterior_probability = float(val)

        h = SimpleHypothesis(hyp_id, prior, description)
        self.create_hypothesis(h)

    def update_hypothesis_posterior(self, hyp_id: str, new_posterior: float) -> None:
        if hyp_id in self._hypotheses:
            self._hypotheses[hyp_id].posterior_probability = float(new_posterior)
            self.state.posterior_beliefs[hyp_id] = float(new_posterior)
            self._refresh_hypotheses_schema()
            self._refresh_best_explanation()

    def fork(
        self,
        child_investigation_id: str,
        at_turn_index: int = 0,
        new_constraints: Optional[Dict[str, Any]] = None,
    ) -> "InvestigationStateManager":
        import copy
        child = InvestigationStateManager(
            investigation_id=child_investigation_id,
            question=self.state.original_question,
            objective=self.state.interpreted_objective,
        )
        child.parent_investigation_id = self.state.investigation_id
        child.fork_turn_index = at_turn_index
        child.fork_constraints = dict(new_constraints or {})
        child.current_turn = at_turn_index

        child._hypotheses = copy.deepcopy(self._hypotheses)
        child._predictions = copy.deepcopy(self._predictions)
        child._evidence_ledger_container.entries = copy.deepcopy(self._evidence_ledger_container.entries)
        child.state.priors = dict(self.state.priors)
        child.state.posterior_beliefs = dict(self.state.posterior_beliefs)
        child._refresh_hypotheses_schema()
        child._refresh_best_explanation()
        return child

    def set_semantic_world_model(self, swm_dict: Dict[str, Any], known_facts: List[str], unknowns: List[str]) -> None:
        self.state.semantic_world_model = swm_dict
        self.state.known_facts = known_facts
        self.state.unknowns = unknowns
        self.state.updated_at = datetime.now(timezone.utc)

    # Hypothesis operations
    def create_hypothesis(self, hypothesis: Any) -> Tuple[str, str, Any]:
        """Register a hypothesis, consolidating it with any existing entry
        that shares the same canonical semantic identity rather than
        creating a duplicate. This is the single authoritative merge path
        used by every caller (initial synthesis, emergent hypotheses
        discovered mid-loop, and state reconstruction after a restart).

        Returns (action, canonical_code, resulting_object):
          action is "created" when this is a genuinely new claim, or
          "merged" when it was folded into an existing hypothesis under a
          different code. Callers that were tracking the hypothesis by its
          own object/code must switch to the returned canonical_code/object
          when action == "merged" -- the original code is retired.
        """
        action, code, resolved = HypothesisConsolidator.register_or_merge(self._hypotheses, hypothesis)
        self._refresh_hypotheses_schema()
        self.state.priors[code] = resolved.prior_probability
        self.state.posterior_beliefs[code] = resolved.posterior_probability
        self._refresh_best_explanation()
        self.state.updated_at = datetime.now(timezone.utc)
        return action, code, resolved

    def revise_hypothesis(self, hypothesis: Any) -> None:
        self._hypotheses[hypothesis.hypothesis_code] = hypothesis
        self._refresh_hypotheses_schema()
        self.state.posterior_beliefs[hypothesis.hypothesis_code] = hypothesis.posterior_probability
        self._refresh_best_explanation()
        self.state.updated_at = datetime.now(timezone.utc)

    def expand_hypothesis_space(self, hypothesis: Any, max_discovery_prior: float = 0.05) -> Tuple[str, str, Any]:
        """Add a newly discovered hypothesis without rewriting accumulated evidence.

        Hypotheses discovered after evidence already exists represent a model-space
        expansion, not fresh evidence for the new claim.  We therefore allocate a
        bounded discovery prior to the new hypothesis and proportionally rescale the
        existing prior/posterior mass.  Existing posterior odds are preserved exactly
        and their priors are never replaced by their posteriors.
        """
        action, code, resolved = self.create_hypothesis(hypothesis)
        if action != "created":
            return action, code, resolved

        active = [h for h in self.get_active_hypotheses() if h.hypothesis_code != code]
        declared = float(getattr(resolved, "prior_probability", 0.0) or 0.0)
        discovery_mass = min(max(float(max_discovery_prior), 0.0), max(declared, 0.0))
        if not active:
            discovery_mass = 1.0

        # Ensure an admissible, normalized expansion even when a caller supplies
        # an invalid/zero declared prior for a genuinely new hypothesis.
        if active and discovery_mass <= 0.0:
            discovery_mass = min(float(max_discovery_prior), 0.05)

        existing_prior_total = sum(max(float(getattr(h, "prior_probability", 0.0) or 0.0), 0.0) for h in active)
        existing_post_total = sum(max(float(getattr(h, "posterior_probability", 0.0) or 0.0), 0.0) for h in active)

        if active:
            if existing_prior_total <= 1e-12:
                existing_prior_weights = {h.hypothesis_code: 1.0 / len(active) for h in active}
            else:
                existing_prior_weights = {
                    h.hypothesis_code: max(float(h.prior_probability), 0.0) / existing_prior_total
                    for h in active
                }
            if existing_post_total <= 1e-12:
                existing_post_weights = {h.hypothesis_code: 1.0 / len(active) for h in active}
            else:
                existing_post_weights = {
                    h.hypothesis_code: max(float(h.posterior_probability), 0.0) / existing_post_total
                    for h in active
                }

            keep_mass = 1.0 - discovery_mass
            for h in active:
                # Relative prior odds and posterior odds among already-observed
                # hypotheses are unchanged; only the expanded model space receives
                # the new bounded prior/posterior mass.
                h.prior_probability = existing_prior_weights[h.hypothesis_code] * keep_mass
                h.posterior_probability = existing_post_weights[h.hypothesis_code] * keep_mass
                self.state.priors[h.hypothesis_code] = h.prior_probability
                self.state.posterior_beliefs[h.hypothesis_code] = h.posterior_probability

        resolved.prior_probability = discovery_mass
        resolved.posterior_probability = discovery_mass
        self.state.priors[code] = discovery_mass
        self.state.posterior_beliefs[code] = discovery_mass
        self._refresh_hypotheses_schema()
        self._refresh_best_explanation()
        self.state.updated_at = datetime.now(timezone.utc)
        return action, code, resolved

    def _refresh_hypotheses_schema(self) -> None:
        self.state.hypotheses = [_to_hypothesis_schema(h) for h in self._hypotheses.values()]

    def _refresh_best_explanation(self) -> None:
        if not self.state.posterior_beliefs:
            return
        top_code = max(self.state.posterior_beliefs, key=self.state.posterior_beliefs.get)
        top_h = self._hypotheses.get(top_code)
        if top_h is not None:
            self.state.current_best_explanation = top_h.claim
        self.state.rejected_explanations = [
            h.claim for h in self._hypotheses.values() if h.posterior_probability < 0.20
        ]

    def sync_hypotheses(self, hypotheses: List[Any]) -> None:
        # Defense-in-depth: even though callers are expected to already have
        # gone through create_hypothesis's merge path, run the same batch
        # consolidation here too so a caller can never reintroduce a
        # same-identity duplicate by wholesale-replacing the registry.
        consolidated = HypothesisConsolidator.consolidate_batch(list(hypotheses))
        self._hypotheses = {h.hypothesis_code: h for h in consolidated}
        self._refresh_hypotheses_schema()
        # Bug fix: priors/posterior_beliefs must be derived from the same
        # `consolidated` set as `_hypotheses`, not the pre-consolidation
        # `hypotheses` argument. Building them from the raw list let obsolete
        # hypothesis_codes that HypothesisConsolidator had just merged away
        # linger in these dicts -- so `_refresh_best_explanation` below could
        # pick a `top_code` no longer present in `_hypotheses` (via
        # `max(posterior_beliefs, ...)`), silently failing to update
        # `current_best_explanation` for a real, consolidated investigation.
        self.state.priors = {h.hypothesis_code: h.prior_probability for h in consolidated}
        self.state.posterior_beliefs = {h.hypothesis_code: h.posterior_probability for h in consolidated}
        self._refresh_best_explanation()
        self.state.updated_at = datetime.now(timezone.utc)

    def set_hypotheses(self, hypotheses: List[Any], priors: Optional[Dict[str, float]] = None) -> None:
        self.sync_hypotheses(hypotheses)

    def get_hypothesis(self, hypothesis_code: str) -> Optional[Any]:
        return self._hypotheses.get(hypothesis_code)

    def get_active_hypotheses(self) -> List[Any]:
        return [
            h for h in self._hypotheses.values()
            if str(getattr(h, "belief_state", "active")).lower() not in ("retired",)
        ]

    def get_all_hypotheses(self) -> List[Any]:
        return list(self._hypotheses.values())

    def get_runtime_hypotheses_dict(self) -> Dict[str, Any]:
        return self._hypotheses

    # Prediction operations
    def create_prediction(self, prediction: Any) -> None:
        self._predictions[prediction.prediction_id] = prediction
        self._refresh_predictions_schema()
        self.state.updated_at = datetime.now(timezone.utc)

    def add_prediction(self, prediction: Any) -> None:
        self.create_prediction(prediction)

    def evaluate_prediction(
        self,
        prediction_id: str,
        status: str,
        reason: str,
        actual_observed_result: Optional[Dict[str, Any]] = None,
        target_experiment_id: Optional[str] = None,
    ) -> None:
        pred = self._predictions.get(prediction_id)
        if pred is not None:
            pred.status = status
            pred.evaluation_reason = reason
            if actual_observed_result is not None:
                pred.actual_observed_result = actual_observed_result
            if target_experiment_id is not None:
                pred.target_experiment_id = target_experiment_id
        self._refresh_predictions_schema()
        self.state.updated_at = datetime.now(timezone.utc)

    def _refresh_predictions_schema(self) -> None:
        self.state.predictions = [_to_prediction_schema(p) for p in self._predictions.values()]

    def get_prediction(self, prediction_id: str) -> Optional[Any]:
        return self._predictions.get(prediction_id)

    def get_predictions_for_hypothesis(self, hypothesis_code: str) -> List[Any]:
        return [p for p in self._predictions.values() if p.hypothesis_code == hypothesis_code]

    def get_pending_predictions(self) -> List[Any]:
        return [p for p in self._predictions.values() if p.status == "PENDING"]

    def get_unresolved_predictions(self) -> List[Any]:
        return [p for p in self._predictions.values() if p.status in ("PENDING", "INCONCLUSIVE", "NOT_TESTABLE")]

    def get_runtime_predictions_dict(self) -> Dict[str, Any]:
        return self._predictions

    def record_calculation_trace(self, trace: Dict[str, Any] | CalculationTraceSchema) -> None:
        """Persist a canonical calculation lineage object in runtime state."""
        obj = trace if isinstance(trace, CalculationTraceSchema) else CalculationTraceSchema.model_validate(trace)
        self.state.calculation_traces.append(obj)
        self.state.updated_at = datetime.now(timezone.utc)

    # Observation & Evidence operations
    def record_raw_observation(self, observation: RawObservationRecord) -> None:
        self.state.raw_observations.append(observation)
        self.state.updated_at = datetime.now(timezone.utc)

    def record_prediction_evidence(self, ev_record: PredictionEvidenceRecord) -> None:
        self.state.prediction_evidence.append(ev_record)
        self.state.updated_at = datetime.now(timezone.utc)

    # Experiment operations
    def record_experiment_executed(self, experiment_id: str, fingerprint: Optional[str] = None) -> None:
        if experiment_id not in self._executed_experiment_ids:
            self._executed_experiment_ids.append(experiment_id)
        if experiment_id not in self.state.executed_experiment_ids:
            self.state.executed_experiment_ids.append(experiment_id)
        if fingerprint:
            if fingerprint not in self._executed_fingerprints:
                self._executed_fingerprints.append(fingerprint)
        self.state.updated_at = datetime.now(timezone.utc)

    def record_experiment_failed(self, experiment_id: str, error_msg: str = "") -> None:
        if experiment_id not in self._failed_experiment_ids:
            self._failed_experiment_ids.append(experiment_id)
        if experiment_id not in self.state.failed_experiment_ids:
            self.state.failed_experiment_ids.append(experiment_id)
        self.state.updated_at = datetime.now(timezone.utc)

    def get_executed_experiments(self) -> List[str]:
        return list(self._executed_experiment_ids)

    def get_executed_fingerprints(self) -> List[str]:
        return list(self._executed_fingerprints)

    def has_experiment_been_executed(self, experiment_id_or_code: str, fingerprint: Optional[str] = None) -> bool:
        if fingerprint and fingerprint in self._executed_fingerprints:
            return True
        if experiment_id_or_code in self._executed_experiment_ids:
            return True
        for e_id in self._executed_experiment_ids:
            if e_id == experiment_id_or_code or e_id.endswith(f"_{experiment_id_or_code}"):
                return True
        return False

    def mark_candidate(self, candidate_exp: FirstClassExperiment) -> None:
        cid = candidate_exp.experiment_id
        if cid not in self.state.candidate_experiment_ids:
            self.state.candidate_experiment_ids.append(cid)
        if not any(c.experiment_id == cid for c in self.state.candidate_experiments):
            self.state.candidate_experiments.append(candidate_exp)
        self.state.updated_at = datetime.now(timezone.utc)

    def record_completed_experiment(
        self,
        exp: FirstClassExperiment,
        observation: Dict[str, Any],
        verification: Dict[str, Any],
    ) -> None:
        exp.execution_result = observation
        exp.verification_result = verification
        self.state.completed_experiments.append(exp)
        self.record_experiment_executed(exp.experiment_id)
        self.state.candidate_experiments = [c for c in self.state.candidate_experiments if c.experiment_id != exp.experiment_id]
        self.state.updated_at = datetime.now(timezone.utc)

    def record_failed_experiment(self, exp: FirstClassExperiment, error_msg: str) -> None:
        exp.execution_result = {"error": error_msg, "status": "FAILED"}
        self.state.failed_experiments.append(exp)
        self.record_experiment_failed(exp.experiment_id, error_msg)
        self.state.candidate_experiments = [c for c in self.state.candidate_experiments if c.experiment_id != exp.experiment_id]
        self.state.updated_at = datetime.now(timezone.utc)

    # Uncertainty operations
    def update_uncertainty(self, uncertainty: Any) -> None:
        if isinstance(uncertainty, UncertaintyStateSchema):
            self.state.uncertainty_state = uncertainty
        elif hasattr(uncertainty, "unresolved_hypotheses"):
            self.state.uncertainty_state = UncertaintyStateSchema(
                unresolved_hypotheses=list(getattr(uncertainty, "unresolved_hypotheses", [])),
                discriminating_variables=list(getattr(uncertainty, "discriminating_variables", [])),
                missing_evidence=list(getattr(uncertainty, "missing_evidence", [])),
                conflicting_evidence=list(getattr(uncertainty, "conflicting_evidence", [])),
                failed_predictions=list(getattr(uncertainty, "failed_predictions", [])),
                assumption_failures=list(getattr(uncertainty, "assumption_failures", [])),
                high_value_unknowns=list(getattr(uncertainty, "high_value_unknowns", [])),
                decision_critical_unknowns=list(getattr(uncertainty, "decision_critical_unknowns", [])),
            )
        self.state.unresolved_predictions = [p.prediction_id for p in self.get_unresolved_predictions()]
        self.state.refuted_predictions = [p.prediction_id for p in self._predictions.values() if p.status == "REFUTED"]
        self.state.updated_at = datetime.now(timezone.utc)

    def get_current_uncertainty(self) -> Optional[UncertaintyStateSchema]:
        return self.state.uncertainty_state

    def update_beliefs(self, posteriors: Dict[str, float], delta_entropy: float) -> None:
        self.state.posterior_beliefs = dict(posteriors)
        self.state.information_gain_accumulated += abs(delta_entropy)
        for h_code, post_val in posteriors.items():
            if h_code in self._hypotheses:
                self._hypotheses[h_code].posterior_probability = post_val
        self._refresh_hypotheses_schema()
        self._refresh_best_explanation()
        self.state.updated_at = datetime.now(timezone.utc)

    def record_adversarial_attack(self, attack_summary: Dict[str, Any]) -> None:
        self.state.adversarial_attacks.append(attack_summary)
        self.state.updated_at = datetime.now(timezone.utc)

    def set_stopping_state(self, stopping_dict: Dict[str, Any]) -> None:
        self.state.stopping_state = stopping_dict
        self.state.updated_at = datetime.now(timezone.utc)

    def finalize_verdict(self, verdict: AnalyticalVerdict, provenance_hash: str) -> None:
        self.state.final_verdict = (
            verdict if isinstance(verdict, AnalyticalVerdict)
            else AnalyticalVerdict(str(verdict))
        )
        self.state.provenance_hash = provenance_hash
        self.state.updated_at = datetime.now(timezone.utc)

    def validate_consistency(self) -> StateConsistencyReport:
        return validate_state(self.state, self._hypotheses, self._predictions)

    def get_state(self) -> CanonicalInvestigationState:
        return self.state
