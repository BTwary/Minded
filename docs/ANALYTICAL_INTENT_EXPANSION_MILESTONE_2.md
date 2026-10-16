# Analytical intent expansion — implementation notes

## Delivered in this candidate

This change adds a typed aggregation registry in `packages/analytics_core/src/intelligence/analytical_operations.py` and routes production canonical descriptive-contract execution through it for multi-dimensional grouping and the new `VARIANCE`, `STDDEV`, and explicit `PERCENTILE` primitives. The canonical contract now serializes an optional `operation_graph` field into its identity hash, and the descriptive executor will execute a graph only when it arrives via the approved-contract argument. The graph payload is therefore contract-persistable; end-to-end API persistence remains unverified because the pinned API dependencies are absent here.

The natural-language contract compiler now recognizes `variance`/`var` and `standard deviation`/`stddev`/`stdev`, and it can resolve explicit conjunctions of grouping dimensions (for example, `Calculate total revenue by region and channel`) only when every grouping phrase resolves to a dataset column. The guided resolver and preview SQL were also corrected to preserve all resolved grouping dimensions rather than collapsing the second dimension. Ranking parsing now distinguishes `Rank regions by total revenue` from `Rank revenue by region`, keeping the numeric measure and entity dimension in their correct roles. Acknowledgement-only human guidance is excluded from calculation-contract identity so a mere acknowledgement does not change the approved plan hash. The result path returns contract and population evidence plus independent recomputation status.

The registry's semantics are explicit:

- `COUNT` counts non-null values; `COUNT_DISTINCT` excludes nulls; `ROW_COUNT` counts rows.
- `VARIANCE` and `STDDEV` use sample semantics (`ddof=1`).
- `PERCENTILE` requires an explicit numeric percentile from 0 to 100; it is not inferred from an ambiguous phrase.
- Null grouping values are retained as a separate `<missing>` group, using an internal sentinel so a literal category named `<missing>` cannot merge with actual nulls.
- Unknown columns, unsupported operations, invalid filter conditions and invalid graph dependencies fail closed.
- Derived ratio/share behavior is not enabled by this registry unless a separate explicit denominator plan exists. Existing `RATE` and `share_of_total` semantics remain on their existing path.

A validated, serializable `OperationGraph` data model is also included for typed aggregate and rank nodes. It has canonical JSON, a SHA-256 graph hash, unique node identity checks, missing-dependency and cycle detection, requested-output validation, deterministic topological ordering, and execution tests.

## Verified test evidence in the development container

Commands run from the candidate root:

- `python -m pytest tests/independent_release/test_analytical_operation_registry.py -vv --tb=short` — **15 passed, exit 0**.
- `python -m pytest tests/independent_release/test_intent_state_contract.py -q --tb=short` — **18 passed, exit 0**.
- `python -m pytest tests/independent_release/test_complex_objective_planner.py -q --tb=short` — **4 passed, exit 0**.
- `python -m py_compile` on the changed Python modules — **passed**.

The API preview-to-approval integration test was attempted but could not initialize because the environment lacks the pinned runtime dependencies `python-jose` and `duckdb`; it reported 3 setup errors and 3 passed tests. A combined authority-suite attempt also failed collection because `polars` is not installed. These are environment limitations, not passes.

## Human-question acceptance examples exercised

1. `Calculate total revenue by region and channel` — schema-resolves both dimensions, computes all four groups and verifies values.
2. `Calculate variance of revenue by region` — uses sample variance per region and verifies each group.
3. `Calculate standard deviation of revenue` — resolves the measure, computes sample standard deviation and verifies the result.
4. Filtered multi-dimensional revenue grouping — excludes cancelled records before aggregation and preserves null group values.
5. Compound typed graph: aggregate revenue by region, then rank the aggregate output — validates dependencies, produces deterministic order and verifies graph structure.
6. Negative cases: unknown group, unsupported aggregation, missing percentile, nonnumeric measure, missing dependency, cycle, invalid rank input, and literal `<missing>` category versus actual null — rejected or distinguished as appropriate.

## Remaining limitations — do not overclaim

- The `OperationGraph` is a validated typed execution primitive and is representable in the canonical contract/hash. One narrow natural-language compound form is now compiled into a graph: aggregate one resolved measure by one resolved dimension, then rank that same dimension by the same measure (for example, `Calculate total revenue by region and rank regions by revenue`). Other compound questions, including different-metric comparisons, remain blocked rather than partially executed. The API's persisted lock-event path has not been verified end to end in this environment.
- The end-to-end API/controller/approval path and pinned Windows environment have not been certified in this container.
- The expanded aggregation executor currently supports explicit multi-column grouping and the new variance/stddev/percentile primitives. It does not claim universal parsing, arbitrary derived-metric semantics, SQL/Polars parity for all new operations, or 1.0.0 release readiness.
- `PERCENTILE` can be executed when a contract supplies `resolution_evidence.percentile`, but natural-language percentile extraction and user-facing clarification are not yet added. Missing percentile remains blocked.

## Acceptance decision

**Milestone 2: BOUNDED CAPABILITIES IMPLEMENTED AND FOCUSED-TEST VERIFIED; Milestone 3: PARTIAL.** Multi-dimensional grouping, variance/stddev, contract-bound graph serialization, and one safe aggregate-then-rank compound question pass focused tests. The API/controller lock-event lifecycle and target Windows environment remain unverified; arbitrary compound operations and general derived-ratio definitions remain incomplete.
