"""
AnalysisStateMachine: LEGACY / RESERVED state vocabulary.

AUTHORITATIVE STATE MACHINE: packages.analytics_core.src.execution.state_machine.
This module is retained only for backwards-compatible imports and historical
tests. Production orchestration must not import or mutate this state machine.
New lifecycle states and transitions belong in execution/state_machine.py.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------
class AnalysisState(str, Enum):
    QUESTION_RECEIVED        = "QUESTION_RECEIVED"
    QUESTION_UNDERSTOOD      = "QUESTION_UNDERSTOOD"
    AWAITING_HUMAN_INPUT     = "AWAITING_HUMAN_INPUT"
    SEMANTICS_RESOLVED       = "SEMANTICS_RESOLVED"
    DATA_READINESS_EVALUATED = "DATA_READINESS_EVALUATED"
    CONTRACT_CREATED         = "CONTRACT_CREATED"
    CONTRACT_REPLANNED       = "CONTRACT_REPLANNED"
    HYPOTHESES_FORMED        = "HYPOTHESES_FORMED"
    EXPERIMENT_SELECTED      = "EXPERIMENT_SELECTED"
    EXPERIMENT_EXECUTING     = "EXPERIMENT_EXECUTING"
    RESULT_AVAILABLE         = "RESULT_AVAILABLE"
    RESULT_VERIFIED          = "RESULT_VERIFIED"
    BELIEFS_UPDATED          = "BELIEFS_UPDATED"
    REPLAN_DECISION          = "REPLAN_DECISION"
    ADVERSARIAL_REVIEW       = "ADVERSARIAL_REVIEW"
    STOPPING_DECISION        = "STOPPING_DECISION"
    VERDICT                  = "VERDICT"
    DECISION_GUIDANCE        = "DECISION_GUIDANCE"
    FINALIZED                = "FINALIZED"
    FAILED                   = "FAILED"


_TERMINAL: Set[AnalysisState] = {AnalysisState.FINALIZED, AnalysisState.FAILED}

_LEGAL: Dict[AnalysisState, Set[AnalysisState]] = {
    AnalysisState.QUESTION_RECEIVED: {
        AnalysisState.QUESTION_UNDERSTOOD,
        AnalysisState.AWAITING_HUMAN_INPUT,
        AnalysisState.FAILED,
    },
    AnalysisState.QUESTION_UNDERSTOOD: {
        AnalysisState.SEMANTICS_RESOLVED,
        AnalysisState.AWAITING_HUMAN_INPUT,
        AnalysisState.FAILED,
    },
    AnalysisState.AWAITING_HUMAN_INPUT: {
        AnalysisState.QUESTION_UNDERSTOOD,
        AnalysisState.SEMANTICS_RESOLVED,
        AnalysisState.STOPPING_DECISION,
        AnalysisState.FAILED,
    },
    AnalysisState.SEMANTICS_RESOLVED: {
        AnalysisState.DATA_READINESS_EVALUATED,
        AnalysisState.FAILED,
    },
    AnalysisState.DATA_READINESS_EVALUATED: {
        AnalysisState.CONTRACT_CREATED,
        AnalysisState.FAILED,
    },
    AnalysisState.CONTRACT_CREATED: {
        AnalysisState.HYPOTHESES_FORMED,
        AnalysisState.AWAITING_HUMAN_INPUT,
        AnalysisState.FAILED,
    },
    AnalysisState.CONTRACT_REPLANNED: {
        AnalysisState.EXPERIMENT_SELECTED,
        AnalysisState.HYPOTHESES_FORMED,
        AnalysisState.STOPPING_DECISION,
        AnalysisState.FAILED,
    },
    AnalysisState.HYPOTHESES_FORMED: {
        AnalysisState.EXPERIMENT_SELECTED,
        AnalysisState.STOPPING_DECISION,
        AnalysisState.FAILED,
    },
    AnalysisState.EXPERIMENT_SELECTED: {
        AnalysisState.EXPERIMENT_EXECUTING,
        AnalysisState.STOPPING_DECISION,
        AnalysisState.FAILED,
    },
    AnalysisState.EXPERIMENT_EXECUTING: {
        AnalysisState.RESULT_AVAILABLE,
        AnalysisState.FAILED,
    },
    AnalysisState.RESULT_AVAILABLE: {
        AnalysisState.RESULT_VERIFIED,
        AnalysisState.BELIEFS_UPDATED,
        AnalysisState.FAILED,
    },
    AnalysisState.RESULT_VERIFIED: {
        AnalysisState.BELIEFS_UPDATED,
        AnalysisState.FAILED,
    },
    AnalysisState.BELIEFS_UPDATED: {
        AnalysisState.REPLAN_DECISION,
        AnalysisState.FAILED,
    },
    AnalysisState.REPLAN_DECISION: {
        AnalysisState.CONTRACT_REPLANNED,
        AnalysisState.EXPERIMENT_SELECTED,
        AnalysisState.ADVERSARIAL_REVIEW,
        AnalysisState.STOPPING_DECISION,
        AnalysisState.FAILED,
    },
    AnalysisState.ADVERSARIAL_REVIEW: {
        AnalysisState.STOPPING_DECISION,
        AnalysisState.CONTRACT_REPLANNED,
        AnalysisState.FAILED,
    },
    AnalysisState.STOPPING_DECISION: {
        AnalysisState.VERDICT,
        AnalysisState.EXPERIMENT_SELECTED,
        AnalysisState.AWAITING_HUMAN_INPUT,
        AnalysisState.FAILED,
    },
    AnalysisState.VERDICT: {
        AnalysisState.DECISION_GUIDANCE,
        AnalysisState.FINALIZED,
        AnalysisState.FAILED,
    },
    AnalysisState.DECISION_GUIDANCE: {
        AnalysisState.FINALIZED,
        AnalysisState.FAILED,
    },
    AnalysisState.FINALIZED: set(),
    AnalysisState.FAILED: set(),
}

PHASE_LABELS: Dict[AnalysisState, str] = {
    AnalysisState.QUESTION_RECEIVED:        "Receiving question",
    AnalysisState.QUESTION_UNDERSTOOD:      "Understanding question",
    AnalysisState.AWAITING_HUMAN_INPUT:     "Awaiting clarification",
    AnalysisState.SEMANTICS_RESOLVED:       "Building data model",
    AnalysisState.DATA_READINESS_EVALUATED: "Checking data readiness",
    AnalysisState.CONTRACT_CREATED:         "Committing analytical contract",
    AnalysisState.CONTRACT_REPLANNED:       "Replanning (new evidence)",
    AnalysisState.HYPOTHESES_FORMED:        "Forming hypotheses",
    AnalysisState.EXPERIMENT_SELECTED:      "Selecting next experiment",
    AnalysisState.EXPERIMENT_EXECUTING:     "Running experiment",
    AnalysisState.RESULT_AVAILABLE:         "Processing result",
    AnalysisState.RESULT_VERIFIED:          "Verifying result (dual-engine)",
    AnalysisState.BELIEFS_UPDATED:          "Updating beliefs",
    AnalysisState.REPLAN_DECISION:          "Deciding: replan or continue?",
    AnalysisState.ADVERSARIAL_REVIEW:       "Adversarial challenge",
    AnalysisState.STOPPING_DECISION:        "Evaluating stopping criteria",
    AnalysisState.VERDICT:                  "Issuing verdict",
    AnalysisState.DECISION_GUIDANCE:        "Generating decision guidance",
    AnalysisState.FINALIZED:               "Analysis complete",
    AnalysisState.FAILED:                  "Analysis failed",
}


# ---------------------------------------------------------------------------
# Reason codes
# ---------------------------------------------------------------------------
class TransitionReason(str, Enum):
    INTENT_CLASSIFIED            = "INTENT_CLASSIFIED"
    INTENT_AMBIGUOUS             = "INTENT_AMBIGUOUS"
    HUMAN_CLARIFICATION_PROVIDED = "HUMAN_CLARIFICATION_PROVIDED"
    SCHEMA_BOUND                 = "SCHEMA_BOUND"
    DATA_FIT_CONFIRMED           = "DATA_FIT_CONFIRMED"
    DATA_CRITICAL_CONSTRAINT     = "DATA_CRITICAL_CONSTRAINT"
    DATA_INADEQUATE              = "DATA_INADEQUATE"
    CONTRACT_COMMITTED           = "CONTRACT_COMMITTED"
    REPLAN_EMERGENT_HYPOTHESIS   = "REPLAN_EMERGENT_HYPOTHESIS"
    REPLAN_ADVERSARIAL           = "REPLAN_ADVERSARIAL"
    REPLAN_EVIDENCE_SHIFT        = "REPLAN_EVIDENCE_SHIFT"
    HYPOTHESES_GENERATED         = "HYPOTHESES_GENERATED"
    NO_TESTABLE_HYPOTHESIS       = "NO_TESTABLE_HYPOTHESIS"
    EXPERIMENT_CHOSEN_BY_EIG     = "EXPERIMENT_CHOSEN_BY_EIG"
    NO_VIABLE_EXPERIMENT         = "NO_VIABLE_EXPERIMENT"
    EXPERIMENT_STARTED           = "EXPERIMENT_STARTED"
    EXPERIMENT_COMPLETED         = "EXPERIMENT_COMPLETED"
    EXPERIMENT_FAILED            = "EXPERIMENT_FAILED"
    VERIFICATION_PASSED          = "VERIFICATION_PASSED"
    VERIFICATION_FAILED          = "VERIFICATION_FAILED"
    VERIFICATION_SKIPPED         = "VERIFICATION_SKIPPED"
    BAYESIAN_UPDATE_APPLIED      = "BAYESIAN_UPDATE_APPLIED"
    REPLAN_NEEDED                = "REPLAN_NEEDED"
    NO_REPLAN_CONTINUE           = "NO_REPLAN_CONTINUE"
    NO_REPLAN_ADVERSARIAL        = "NO_REPLAN_ADVERSARIAL"
    NO_REPLAN_STOP               = "NO_REPLAN_STOP"
    ADVERSARIAL_SURVIVED         = "ADVERSARIAL_SURVIVED"
    ADVERSARIAL_CHALLENGE_FOUND  = "ADVERSARIAL_CHALLENGE_FOUND"
    ANSWER_ESTABLISHED           = "ANSWER_ESTABLISHED"
    ANSWER_WITH_LIMITATIONS      = "ANSWER_WITH_LIMITATIONS"
    INSUFFICIENT_EVIDENCE        = "INSUFFICIENT_EVIDENCE"
    IDENTIFICATION_FAILED        = "IDENTIFICATION_FAILED"
    VERIFICATION_FAILED_STOP     = "VERIFICATION_FAILED_STOP"
    AMBIGUITY_HUMAN_REQUIRED     = "AMBIGUITY_HUMAN_REQUIRED"
    MAX_SAFETY_BUDGET            = "MAX_SAFETY_BUDGET"
    UNCERTAINTY_STAGNATED        = "UNCERTAINTY_STAGNATED"
    VERDICT_ISSUED               = "VERDICT_ISSUED"
    GUIDANCE_GENERATED           = "GUIDANCE_GENERATED"
    FINALIZED_COMPLETE           = "FINALIZED_COMPLETE"
    INTERNAL_ERROR               = "INTERNAL_ERROR"
    STATE_INCONSISTENCY          = "STATE_INCONSISTENCY"
    JOIN_SAFETY_BLOCKED          = "JOIN_SAFETY_BLOCKED"
    UNSUPPORTED_METHOD           = "UNSUPPORTED_METHOD"


# ---------------------------------------------------------------------------
# Transition record
# ---------------------------------------------------------------------------
@dataclass
class StateTransition:
    """Immutable record of one state transition. Callers persist this to DB."""
    transition_id: str
    investigation_id: str
    from_state: AnalysisState
    to_state: AnalysisState
    reason_code: TransitionReason
    reason_text: str
    contract_version: int
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "investigation_id": self.investigation_id,
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "reason_code": self.reason_code.value,
            "reason_text": self.reason_text,
            "contract_version": self.contract_version,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    def to_sse_event(self) -> Dict[str, Any]:
        """Shape consumed by SSE stream / StepTracker.tsx."""
        return {
            "event_type": "investigation.state_transition",
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "reason_code": self.reason_code.value,
            "reason_text": self.reason_text,
            "contract_version": self.contract_version,
            "phase_label": PHASE_LABELS.get(self.to_state, self.to_state.value),
            "timestamp": self.timestamp.isoformat(),
        }


class IllegalTransitionError(ValueError):
    """Raised when a transition is not in the legal transition map."""


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------
class AnalysisStateMachine:
    """
    Explicit state machine for one investigation.

    Usage in controller.py:
        sm = AnalysisStateMachine(investigation_id)
        t = sm.transition(
            AnalysisState.QUESTION_UNDERSTOOD,
            TransitionReason.INTENT_CLASSIFIED,
            f"Classified as {problem_class}",
            contract_version=0,
            metadata={"problem_class": problem_class},
        )
        # Persist t.to_dict() and emit t.to_sse_event()
    """

    def __init__(self, investigation_id: str) -> None:
        self.investigation_id = investigation_id
        self.current_state: AnalysisState = AnalysisState.QUESTION_RECEIVED
        self.transitions: List[StateTransition] = []

    def transition(
        self,
        to_state: AnalysisState,
        reason_code: TransitionReason,
        reason_text: str,
        contract_version: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StateTransition:
        """Validate and execute a transition. Raises IllegalTransitionError if invalid."""
        if self.current_state in _TERMINAL:
            raise IllegalTransitionError(
                f"{self.investigation_id} is terminal ({self.current_state.value}) — "
                "no further transitions are valid."
            )
        allowed = _LEGAL.get(self.current_state, set())
        if to_state not in allowed:
            raise IllegalTransitionError(
                f"Illegal transition {self.current_state.value} -> {to_state.value} "
                f"for {self.investigation_id}. "
                f"Allowed: {sorted(s.value for s in allowed)}"
            )
        t = StateTransition(
            transition_id=str(uuid.uuid4()),
            investigation_id=self.investigation_id,
            from_state=self.current_state,
            to_state=to_state,
            reason_code=reason_code,
            reason_text=reason_text,
            contract_version=contract_version,
            timestamp=datetime.now(timezone.utc),
            metadata=metadata or {},
        )
        self.current_state = to_state
        self.transitions.append(t)
        return t

    def is_terminal(self) -> bool:
        return self.current_state in _TERMINAL

    def phase_label(self) -> str:
        return PHASE_LABELS.get(self.current_state, self.current_state.value)

    def transition_history(self) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self.transitions]

    def last_transition(self) -> Optional[StateTransition]:
        return self.transitions[-1] if self.transitions else None

    def safe_fail(
        self,
        reason_code: TransitionReason,
        reason_text: str,
        contract_version: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StateTransition:
        """
        Transition to FAILED from any non-terminal state.
        Bypasses legal-transition check — FAILED is always reachable as an emergency exit.
        """
        if self.current_state in _TERMINAL:
            # Already terminal — return sentinel
            return StateTransition(
                transition_id=str(uuid.uuid4()),
                investigation_id=self.investigation_id,
                from_state=self.current_state,
                to_state=self.current_state,
                reason_code=reason_code,
                reason_text=f"[already terminal] {reason_text}",
                contract_version=contract_version,
                timestamp=datetime.now(timezone.utc),
                metadata=metadata or {},
            )
        t = StateTransition(
            transition_id=str(uuid.uuid4()),
            investigation_id=self.investigation_id,
            from_state=self.current_state,
            to_state=AnalysisState.FAILED,
            reason_code=reason_code,
            reason_text=reason_text,
            contract_version=contract_version,
            timestamp=datetime.now(timezone.utc),
            metadata=metadata or {},
        )
        self.current_state = AnalysisState.FAILED
        self.transitions.append(t)
        return t
