# ADR 001: Canonical Investigation State Machine

## Decision

`packages/analytics_core/src/execution/state_machine.py` is the sole authoritative
lifecycle/state vocabulary for investigation execution. It owns `AnalyticalPhase`,
`InvestigationState`, `ExecutionState`, and the durable transition validators.

`packages/analytics_core/src/runtime/analytical_state_machine.py` is legacy/reserved
and must not receive new production dependencies.

## Rationale

Two overlapping state vocabularies make it possible for orchestration, persistence,
and tests to disagree about whether an investigation legally progressed. A scientific
runtime must have one transition authority so invalid lifecycle progress fails closed.

## Consequence

New lifecycle states, transitions, and phase semantics belong in `execution/state_machine.py`.
Legacy consumers may be migrated incrementally, but no new production import of the
legacy module is permitted.
