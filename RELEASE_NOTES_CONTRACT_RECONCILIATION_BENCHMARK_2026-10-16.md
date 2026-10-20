# MindEd: execution reconciliation, answer report, and acceptance scoring

Date: 2026-10-16

## Implemented

### 1. Contract-to-execution reconciliation

- Added a lightweight, independently testable reconciliation module at `packages/analytics_core/src/runtime/execution_reconciliation.py`.
- The approved aggregate path now checks the metric, aggregation, grouping, effective filters (including time/comparison scope), denominator semantics, requested outputs, ranking direction, exact SQL, dataset identity, dataset version metadata, dataset fingerprints, and input content fingerprints.
- A mismatch is recorded as `contract_status=MISMATCH`, includes named failed checks, and blocks publishing a verified answer.
- Share-of-total executions record the denominator derived from the returned grouped relation. This describes the actual denominator source rather than hiding it in prose.

### 2. Investigation report

- The answer section now places verification status beside the answer, so a number is not presented without its trust status.
- The execution summary can show denominator details when applicable.
- Added a collapsible contract-to-execution reconciliation report with named checks; actual SQL and execution specification remain expandable instead of dominating the initial report.
- Existing output order retains the answer first, then scope/filters/row counts, result table, calculation explanation, and technical evidence.

### 3. Reproducible acceptance scoring

- Added `scripts/summarize_adjudicated_benchmark.py` and `docs/ADJUDICATED_ACCEPTANCE_BENCHMARK.md`.
- The scorer requires exact case-ID parity and separates correct answers, wrong answers, safe refusals, and answerable refusals. It refuses to score a partial or duplicate run as a pass.
- Added regression tests for the four-way scoring and missing/duplicate cases.

## Validation

- Focused suite: **41 passed, 1 skipped** across execution reconciliation, benchmark scoring, filter ingestion, SQL predicate guard, and SQL literal safety.
- `compileall` passed for the changed Python runtime modules and benchmark scorer.
- ZIP integrity check passed after packaging.

## Acceptance benchmark status: BLOCKED, not passed

The supplied archive contains a 120-case independent audit matrix and a separate 32-case stress-results fixture. It does not establish that these are the same original 152 adjudicated cases, nor does it include a fresh production-run result file for that exact benchmark. I did not combine them and claim a 152-case pass. Provide the original adjudicated fixture and capture results from the updated production run, then run the documented scorer command to produce the required four-way totals.

## Environment limits

- Frontend build was not run because `apps/web/node_modules` is absent in this working copy.
- The full production execution path was not run here because this environment lacks the `polars` dependency required to import the relational compiler (and prior broader collection also lacked `duckdb`). The focused reconciliation tests use a lightweight fake receipt to test pass/fail conditions; they are not represented as a database integration test.
