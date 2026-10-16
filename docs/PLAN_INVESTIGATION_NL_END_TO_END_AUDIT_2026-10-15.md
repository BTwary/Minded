# MindEd AA-OS — Plan, Investigation, and Natural-Language Audit

Date: 2026-10-15

## Verdict

**Improved and focused-tested; not end-to-end release-certified.** The planning layer now has broader schema-aware question parsing and safer specialist gates. Full controller execution and the complete release manifest could not be verified in this runner because critical locked dependencies are missing, and the release-gate invocation did not complete within the available execution window.

## Fixes in this revision

1. Bumped the typed plan schema to version 2.1.
2. Added conservative natural-language contract extraction for operation cues, grouping candidates, time grain, top/bottom-N ranking, comparison mode, filter candidates, relative date-window candidates, schema column mentions, and explicit ambiguity flags.
3. Kept filter/date candidates non-executable until their columns and semantics are bound; these fields are marked `execution_safe: false`.
4. Added schema-grounded binding for unambiguous mentioned numeric measures, grouping columns, and association X/Y columns.
5. Added invalid-role binding detection when an existing semantic proposal names a column absent from the observed schema.
6. Fixed semantic aliases so `target_metric_col`, `group_dimension_col`, and `time_col` can populate typed roles; the controller now reads these aliases when building runtime facts.
7. Added compound `all`/`any` predicates to the safe condition evaluator. The evaluator remains declarative and does not execute Python expressions.
8. Association eligibility now requires observed, supported types for both X and Y, rather than checking only X.
9. Forecast authorization now requires both a monotonic observed time index and an explicitly stated horizon. Missing horizon remains unknown and cannot authorize forecasting.
10. Added regression coverage for real-world question families, schema binding, invalid semantic columns, compound conditions, and forecast horizon gating.

## Validation evidence

| Check | Result |
|---|---|
| Focused plan / intent / conditional / universal compiler tests | **133 passed** |
| Broad task-routing matrix | **30/30 expected task-family routes** |
| Python compileall | **PASS** |
| Release-status audit | **PASS** |
| Canonical serialization audit | **PASS; 193 files scanned** |
| ZIP integrity | PASS on the generated archive |
| Full independent-release collection | **BLOCKED**: 629 tests collected, 54 collection errors |
| Controller-level question matrix | **BLOCKED**: import requires missing `duckdb` |
| API integration tests | **BLOCKED**: missing `python-jose` (`jose`) |
| Full release gate | **NOT COMPLETED** within runner execution window |
| Clean-machine Windows EXE | **NOT TESTED** |

The 30/30 routing matrix is a bounded regression set, not proof of general natural-language understanding. The 133 passing tests exercise selected modules and integration seams; they do not substitute for running the complete locked environment.

## Current capability assessment

- **Question-family routing:** broader, deterministic lexical routing; good on the tested 30-question matrix. Paraphrases, domain-specific jargon, negation, nested logic, and competing intents still require wider adversarial evaluation.
- **Semantic role binding:** uses observed schema and existing semantic aliases; binds only when the evidence is unambiguous. Unresolved or absent columns are surfaced instead of being silently accepted.
- **Natural-language conditions:** candidate filters and date windows are extracted, but deliberately not executed automatically. Column binding, operator typing, timezone/calendar interpretation, and explicit contract authorization remain necessary.
- **Conditional runtime:** controller source wiring and the selected conditional policy are tested statically/unit-level. Prediction, segmentation, decision analysis, and some specialist paths correctly remain blocked until an independent evidence producer exists. This is safe refusal, not full specialist capability.
- **Analyst understanding:** cannot be rated as fully reliable from this evidence. There is no completed full controller question matrix, no end-to-end execution with DuckDB/Polars, and no fresh-install EXE run in this environment.

## Required next verification

1. Install the exact hashed lock in a clean Python 3.12 environment with package-index access.
2. Run the complete `scripts/release_gate.py` manifest to completion and preserve its JSON evidence.
3. Run the controller question matrix against real CSV/XLSX/parquet samples with independent expected answers and verify operation, filters, grouping, joins, time scope, and verdict—not only task routing.
4. Run the Windows EXE from a fresh extraction and fresh user-data directory; verify import, ingestion, clarification behavior, computation, evidence, cancellation/recovery, and no stale database/cache influence.
5. Do not promote unresolved filter candidates, predicted roles, or conditional plan metadata to executable authority without validation.
