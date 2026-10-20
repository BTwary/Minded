# Filter parsing and investigation output polish

Date: 2026-10-16

## What changed

### 1. Threshold intent ledger is column-aware

The canonical question contract now records the explicit schema column, comparator, and numeric value for each recognized threshold clause. A ledger entry is `CONSUMED` only when a typed predicate matches its comparator, numeric value, and (when named) column. This closes the prior count-only loophole where an unrelated numeric predicate could satisfy a dropped threshold clause.

Example: `revenue above 500 and quantity below 3` should produce `revenue > 500` and `quantity < 3`, with a separate consumed ledger entry for each clause.

### 2. Investigation UI uses operational language

The investigation graph now describes itself as an investigation workbench, uses “Priority” rather than implying the priority field is a posterior probability, clarifies that scores prioritize checks and are not probabilities, and uses less jargon-heavy labels for rationale and steps. The selected inspector item is synchronized when a new result set arrives, preventing stale selection after a new investigation run.

### 3. SQL filter documentation expanded

`docs/FILTERS_AND_QUERY_EXECUTION.md` now explains the end-to-end lifecycle: intent extraction, schema binding, value validation, typed predicates, allowlist/AST validation, SQL compilation, database semantics, execution, contract reconciliation, and user-facing output. It includes a bound-parameter example and explains why a syntactically valid SQL query can still answer the wrong question if the parser omitted intent.

## Validation

- `tests/independent_release/test_plan_filter_ingestion_regressions.py`: 20 passed.
- `tests/independent_release/test_filter_sql_predicate_guard.py` and `tests/independent_release/test_p0_sql_literal_safety.py`: combined with the above, 33 passed, 1 skipped.
- Python `compileall` for the modified canonical contract: passed.
- The broader `tests/test_generalized_filtering_and_export.py` could not be collected because `duckdb` is not installed in this environment.
- A frontend production build was not run; dependencies are not installed in this working copy.

## Remaining limitations

This is not a claim of full natural-language token coverage or full post-execution contract reconciliation. The current ledger is specific to detected filter clauses. The 152-case adjudicated benchmark was not included as a runnable fixture, so no overall accuracy improvement is claimed. Before release, run the full dependency-locked suite, frontend build, and original 152-case benchmark on the target platform.
