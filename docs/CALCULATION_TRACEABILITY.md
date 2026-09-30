# AA-OS Calculation Traceability

AA-OS treats calculation traceability as a canonical analytical invariant.
Every material result must be inspectable as:

`dataset snapshot -> analytical scope -> metric/estimand formula -> executable expression -> deterministic execution -> independent verification -> statistical/prediction derivations -> Bayesian update -> final evidence`

## Trace contents

Each `CalculationTrace` contains:

- immutable dataset fingerprints and row counts;
- exact executable SQL/scope expression;
- metric semantics and aggregation formula;
- formula inputs, parameters, and outputs;
- primary and independent secondary verification results;
- grain/fanout verification;
- prediction evaluation;
- statistical outputs when present;
- Bayesian prior/likelihood/posterior updates;
- a canonical SHA-256 trace hash.

## Persistence

The same trace is stored in the canonical raw observation, included in evidence computation proof, and emitted as an immutable `calculation.trace.created` investigation event. Final verdict determination, epistemic calibration, decision utility, and the reproducible manifest hash are also emitted as calculation/audit events.

No AI provider is required to create or read a trace. AI augmentation may explain the trace but may not replace, alter, or fabricate the canonical calculation.

## Human review contract

A human reviewer must be able to start from any material evidence item and drill down to the exact source dataset fingerprint, scope, formula, executable expression, intermediate result, independent verification, and final claim supported by that calculation.

AA-OS must never silently replace an unavailable calculation with an invented value or an unverifiable narrative.
