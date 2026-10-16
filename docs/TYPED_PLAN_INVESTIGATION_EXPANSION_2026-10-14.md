# MindEd AA-OS — Typed Plan and Investigation Expansion

Date: 2026-10-14

## What changed

- Added `typed_investigation_planner.py`, a deterministic, type-aware planning layer with 18 task types: profile, aggregation, distribution, missingness, comparison, association, temporal trend, forecast, prediction, causal effect, segmentation, ranking, contribution, reconciliation, anomaly, survival, text analysis, governance, and decision analysis.
- Added conservative schema type inference for numeric, datetime, boolean, categorical, text/mixed, and unknown columns; serializes the schema and proposed semantic role types into the plan.
- Emits explicit required and unresolved roles and type warnings rather than treating an incomplete role proposal as execution-ready.
- Adds task-family-specific conditional branches (e.g. causal identification, prediction leakage, forecast time order, group cardinality, cluster stability, decision context, and reconciliation formula readiness).
- Adds a safe structured condition evaluator (`eq`, `ne`, comparisons, membership, truthy, exists). It never uses `eval`; `UNKNOWN` blocks rather than passes. Step resolution returns auditable `RUN`, `SKIP`, or `BLOCKED` outcomes and checks prerequisite statuses.
- Integrates the typed plan into `UniversalQuestionCompiler` output and semantic evidence, preserving the existing plan and method-registry authority boundaries.

## Validation performed

- Focused merge + typed planner regression suite: **89 passed**.
- Typed planner + autonomy hardening + method admissibility integration suite: **15 passed**.
- Python `compileall` for modified source and tests: **passed**.

## Important boundary

This adds the typed plan representation and a working conditional-resolution API. It does **not** mean every existing runtime experiment has been converted to consume every branch automatically, nor does a branch or a file's presence authorize execution. Runtime method-registry admissibility, data-readiness checks, contract completeness, and execution authorization remain mandatory. In particular, production end-to-end conditional wiring and the complete release manifest/Windows clean-machine EXE test have not been certified here. DuckDB is absent from this audit environment, so database-backed release checks were not run.
