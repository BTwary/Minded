# Acceptance matrix: analytical intent expansion

| Capability | Status | Evidence / boundary |
|---|---|---|
| Typed aggregation registry | PASS (focused tests) | Registry and fail-closed validation in `intelligence/analytical_operations.py` |
| Two-dimensional grouping in production descriptive executor and preview | PASS (focused tests) | Guided resolver, preview SQL and `_execute_canonical_descriptive_contract` preserve both dimensions |
| Variance and standard deviation | PASS (focused tests) | Sample variance/stddev; natural-language contract recognition and executor tests |
| Explicit percentile operation | PASS (primitive only) | Requires explicit `percentile` parameter; natural-language percentile resolution remains unimplemented |
| Null group vs literal `<missing>` category | PASS (regression test) | Internal sentinel preserves distinct groups |
| Typed operation graph, canonical hash and topological validation | PASS (focused tests) | Aggregate→rank graph and cycle/missing-dependency rejection |
| Graph field in canonical contract and approval-gated executor | PASS (unit-level) | Graph participates in contract serialization/hash; executor refuses graph-bearing fresh parse; API lock-event E2E not verified |
| Graph generation from compound natural language | PARTIAL | One strict same-measure aggregate→rank shape is compiled; different-metric or otherwise unsupported compounds remain refused |
| Arbitrary compound natural-language execution | NOT IMPLEMENTED | Existing resolver continues to refuse unsupported compound requests |
| Clarification/approval-to-execution API E2E | NOT TESTED | Environment lacks `python-jose` and `duckdb`; API test setup failed |
| Polars/relational authority regression suite | NOT TESTED | Environment lacks `polars`; collection failure observed |
| Pinned Windows build, installer, offline operation | NOT TESTED | Requires the user's pinned Windows environment |
| 1.0.0 release certification | NOT CLAIMED | Full required manifest and packaged app have not passed in the target environment |
