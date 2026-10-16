# MindEd AA-OS — Milestones 2/3 implementation report

**Candidate:** derived from `MindEd_AAOS_stateful_intent_patch(1).zip`  
**Scope:** bounded analytical primitive expansion and a constrained compound-operation path.  
**Overall status:** real implementation delivered; focused tests pass; application-level certification remains incomplete.

## Code changes made

1. **New `packages/analytics_core/src/intelligence/analytical_operations.py`**
   - Central registry for `ROW_COUNT`, `COUNT`, `COUNT_DISTINCT`, `SUM`, `MEAN`, `MEDIAN`, `MIN`, `MAX`, `VARIANCE`, `STDDEV`, and `PERCENTILE`.
   - Explicit sample semantics for variance and standard deviation (`ddof=1`).
   - Explicit percentile argument requirement (0–100); no inferred percentile value.
   - Multi-dimensional grouping with explicit null groups, typed filter application, independent recomputation, and fail-closed validation.
   - Versioned `OperationGraph` / `OperationNode` data structures, canonical JSON, SHA-256 identity, dependency validation, cycle detection, deterministic topological order, aggregate→rank execution.
   - A deliberately narrow compiler for the safe same-measure aggregate→rank human-question form.

2. **`canonical_question_contract.py`**
   - Added `operation_graph` as identity-bearing contract data, included in serialization and contract hash.
   - Recognizes variance and standard-deviation vocabulary.
   - Resolves explicit multiple grouping dimensions only when each dimension maps to a unique physical column.
   - Corrects ranking role resolution for both “Rank revenue by region” and “Rank regions by total revenue”.

3. **`guided_investigation.py`**
   - Preserves multiple grouping dimensions through clarification resolution and SQL preview generation.
   - Emits preview steps and graph hash for the supported aggregate→rank graph.
   - Generates a graph only for a narrowly validated same-measure/same-dimension pattern; other compound questions remain blocked.
   - Validates numeric compatibility for variance and standard deviation.
   - Keeps acknowledgement-only metadata from changing the calculation contract hash.

4. **`analyst_answer.py`**
   - Routes multi-dimensional grouped calculations and variance/stddev/percentile through the typed registry.
   - Executes an operation graph only when it is present on the supplied approved contract.
   - Returns graph outputs and verification metadata.

5. **Tests and release manifest**
   - Added `tests/independent_release/test_analytical_operation_registry.py` with 15 cases covering real natural-language questions, aggregation semantics, grouping/filter correctness, null handling, ranking role resolution, graph persistence/hash, approval-bound graph execution and safe refusal.
   - Registered the new test file as a required production-path check in `tests/independent_release/manifest.toml`.

## Real human-question examples verified

### Multi-dimensional grouping

Question: `Calculate total revenue by region and channel`

Dataset rows: East/web = 10, East/store = 20, West/web = 30, West/store = 40.

The canonical contract retains both `region` and `channel`; the preview SQL groups by both; execution returns all four groups and independently recomputes each value.

### Statistical aggregation

- `Calculate variance of revenue by region` resolves `VARIANCE` and previews sample variance (`VAR_SAMP`) by region.
- `Calculate standard deviation of revenue` resolves `STDDEV` and previews sample standard deviation (`STDDEV_SAMP`).
- A numeric sample `[1, 2, 3, 4]` returns sample variance `1.666666...`, sample standard deviation `1.290994...`, and median percentile `2.5`.

### Ranking language

Both `Rank regions by total revenue` and `Rank revenue by region` resolve `target_column=revenue`, `grouping_columns=(region,)`, and `SUM`, avoiding the prior role reversal that could generate `SUM(region)`.

### Supported compound question

Question: `Calculate total revenue by region and rank regions by revenue`

The preview now creates a complete canonical contract containing an aggregate→rank graph and its hash. The approved-contract execution returns both grouped sums and ranked rows. On the sample data East totals 40 and ranks 1; West totals 20 and ranks 2.

A different-metric compound question such as `Rank revenue by region and compare profit by region` remains non-executable; it is not reduced to only the first clause.

## Verification results

All commands below were run from the extracted candidate root in the development container (Linux, Python 3.13.5), not the pinned Windows environment.

| Command | Result |
|---|---|
| `python -m pytest tests/independent_release/test_analytical_operation_registry.py -vv --tb=short` | **15 passed**, exit 0 |
| `python -m pytest tests/independent_release/test_intent_state_contract.py -q --tb=short` | **18 passed**, exit 0 |
| `python -m pytest tests/independent_release/test_complex_objective_planner.py -q --tb=short` | **4 passed**, exit 0 |
| `python -m pytest tests/independent_release/test_m1_reconciliation.py -q --tb=short` | **7 passed**, exit 0 |
| `python -m compileall -q packages apps scripts tests` | **Passed**, exit 0 |
| TOML parse of `tests/independent_release/manifest.toml` | **33 checks registered**, new required check present |
| `tests/independent_release/test_m1_preview_to_approval_e2e.py` | **3 passed, 3 setup errors**, exit 1; API setup blocked by missing `python-jose` and `duckdb` |
| `tests/independent_release/test_controller_question_matrix.py` | **Collection blocked** by missing `duckdb` |
| A broader contract-authority suite | **Collection blocked** by missing `polars` |

Do not count the blocked API/controller suites as passes. The full release manifest, pinned Windows dependencies, Windows installer, clean installation, and offline operation have not been verified.

## Boundaries and remaining work

- Compound graph generation currently supports only the same-measure, one-dimension aggregate→rank shape. It does not support arbitrary multi-operation graphs from natural language.
- The API/controller lock-event lifecycle was not executed end to end here. Contract serialization and hash behavior are unit-tested; the real persisted event flow must still be tested in the project's pinned environment.
- The graph executor currently supports `AGGREGATE` and `RANK`; comparison, period-over-period, derived ratios, and other graph node types are not claimed as implemented.
- The operation registry's percentile primitive is implemented, but natural-language percentile extraction and clarification of the percentile value are not. Missing percentile is blocked.
- New operations have not been independently cross-checked against DuckDB and Polars in the target pinned environment. The result's registry recomputation is not a substitute for backend parity certification.
- No 1.0.0 release certification is claimed.

## Recommended next acceptance gate

Run the new registered test and the existing API preview→approval→controller execution tests inside the project's pinned Windows Python environment. Confirm that the operation graph survives the lock event and that the exact graph hash used at preview is the one executed. Then run the full release manifest and packaged clean-install tests. Do not broaden graph operations until these checks pass.
