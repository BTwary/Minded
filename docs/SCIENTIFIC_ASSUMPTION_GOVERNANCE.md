# Scientific Assumption & Limitation Governance

AA-OS now creates a durable assumption ledger for each canonical investigation. The ledger is generated from already-resolved analytical facts and does not manufacture evidence.

## What is recorded

- data-scope stability and source snapshot assumptions
- metric semantics and aggregation
- unit-of-analysis/grain binding
- data-quality warnings and critical issues
- temporal scope interpretation and ambiguity
- forecasting generalization and backtesting assumptions
- causal identification limits
- missingness/selection sensitivity
- relational join safety
- independent-verification dependency
- the no-silent-sampling execution contract

Each item records a stable code, category, statement, sensitivity risk, validation status, and optional validation evidence.

## Human review

The investigation API exposes `assumption_ledger` and a derived quality card. Local deterministic explanations include unvalidated assumptions in their limitations. High-risk unvalidated assumptions are never converted into positive evidence and are intended to trigger human review before consequential use.

## Design invariant

A conclusion is not considered fully decision-grade merely because a calculation is numerically correct. The user must be able to see the assumptions on which the calculation and interpretation depend.
