# MindEd AA-OS Analytical Engine Capability Contract

This document states measured and enforced behavior. It is not a claim of unlimited dataset support.

## Deterministic execution

- Source-of-truth calculations run from the observed dataset snapshot.
- A scalar observation must identify its `primary_result_column`; arbitrary numeric-column guessing is blocked.
- Correlation observations compute the observed Pearson correlation from the paired result relation.
- Variance observations compute population variance (`ddof=0`) from the observed result values.
- Multi-row scalar observations are accepted only when the primary result is explicitly identified and deterministically ordered.
- Secondary verification recomputes the result relation independently in Polars and independently reduces that relation to the scalar before comparing it with DuckDB.

## Relational capability

- Explicit typed `RelationalPlan` supports an arbitrary list of join hops subject to join-safety and execution-time/resource gates.
- Autonomous relational discovery is bounded by `AAOS_MAX_AUTONOMOUS_RELATIONAL_HOPS` (default **5**, meaning up to **6 tables**) to keep search deterministic and bounded.
- Unsafe/unknown many-to-many joins are blocked unless explicitly authorized; raw source-table aggregation across fanout-risk joins is blocked.
- The autonomous planner now searches beyond the former two-/three-table special cases.

## Dataset size and efficiency limits

Current hard runtime controls:

- Maximum uploaded source artifact: **500 MB**.
- Per analytical query wall-clock budget: **30 s** by default (`AAOS_MAX_QUERY_EXECUTION_TIME_SECONDS`).
- Per investigation runtime budget: **300 s** by default (`AAOS_MAX_INVESTIGATION_RUNTIME_SECONDS`).
- Maximum experiment count per investigation: **8**.
- Maximum returned result rows from an analytical query: **50,000**.

The data provider currently materializes database-backed datasets into Pandas. Therefore practical row/column limits are ultimately bounded by local RAM and dataset width, not by a universal fixed row count. A 500 MB file can require substantially more RAM after parsing and type/object overhead.

## Local benchmark in the audit container

These measurements use the deterministic Pandas profiling layer only; DuckDB/Polars were unavailable in the audit container, so no unsupported SQL-engine throughput number is claimed.

- 100,000 rows × 10 numeric columns: ~0.24 s
- 500,000 rows × 10 numeric columns: ~1.50 s
- Five-table safe join-path discovery: ~0.0018 s on the small synthetic schema used for the planner test

These are engineering benchmarks, not guarantees for arbitrary hardware or arbitrary data types.
