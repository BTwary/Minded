# MindEd AA-OS v1.0.0 combined remediation report

Date: 2026-10-18

## Summary

This release candidate completes the remediation of all defects identified in the 2026-10-18 independent verification audit of MindEd AA-OS. It reconciles preview, authoritative resolver, and compiler layers for end-to-end consistency, enforces strict independent evidence verification, secures database test isolation, refines clarification candidate selection, and ensures cross-platform archive portability.

## Defect Remediation Summary

### High Severity (H1–H4)
- **H1 — Compound-plan preview override & graph execution**: Restored the verified aggregate-then-rank operation graph in both preview and resolver contracts. Unsupported multi-metric compound questions fail closed with `operation_graph` unresolved, refusing partial execution.
- **H2 — Filter rules & compiler integrity**: Conjunctions and disjunctions are correctly anchored. Existing compiler-derived filters (`NOT IN`) survive natural-language clauses without overwriting or string pollution.
- **H3 — Top-N limits and ranking role binding**: Multi-ranking compound shapes and out-of-bounds limits (>100) fail closed. Explicit ranking role binding (`Top region by gross_sales`) is aligned end-to-end across `canonical_question_contract.py`, `guided_investigation.py`, and `intent_state.py`: preview and authoritative resolver both bind `target_column="gross_sales"` (measure) and `grouping_columns=("region",)` (dimension) with `ResolverStatus.RESOLVED`.
- **H4 — Average order value (AOV) & multi-amount consistency**: Unified candidate priority sequence `("aov", "order_value", "total_amount", "order_total", "order_amount", "revenue", "sales_amount", "gross_sales", "sales", "amount")` across preview and compiler. On frames containing both `revenue` and `order_total`, both layers deterministically select `order_total`. For `average order value per customer`, the contract grouping by `customer_id` satisfies the per-customer requirement, resolving cleanly without an unneeded denominator slot. Identifier columns (`order_id`) are strictly excluded as additive measures.

### Medium Severity (M1–M4)
- **M1 — Group-comparison verification gate**: `analyst_group_comparison_dual_statistical_confirmation` requires executed experiments and is explicitly tied to an independently verified evidence record in the database (`EvidenceVerification.status == VERIFIED` and `Evidence.validation_status == 'VERIFIED'`), preventing self-attested zero-evidence passes.
- **M2 — Temporal comparison & partial-year warning**: `Compare revenue this year and last year` resolves consistently in preview and resolver with `time_column="order_date"` and paired calendar-year bounds, emitting the partial-period warning in assumptions when data ends mid-year.
- **M3 — Clarification candidate quality**: Dimension candidate filtering for comparisons strictly excludes numeric columns, measure synonyms (revenue, profit, sales, amount, cost), and the target metric, ensuring only valid categorical dimensions are presented without generic dead-end options.
- **M4 — Test database isolation**: In `test_approved_plan_execution.py`, `test_history_reopens_source_identity_from_actual_execution_receipt` snapshots pre-existing `Dataset(id="sales")` state before testing and restores the original row in `finally`, avoiding destructive deletion of seeded data.

### Low Severity & Governance (L1–L3)
- **L1 — Clarification slot**: Multi-part compound requests consistently designate `operation_graph` as the pending intent slot.
- **L2 — Quoted SQL identifiers**: Relational compiler and plan execution assert strict quoting of SQL column/table identifiers.
- **L3 — Test count & environment reconciliation**: Documented test breakdown:
  - 175 tests in full environment with DuckDB and full locked dependencies.
  - 120 tests passed in portable test runner environment; 8 tests blocked solely by external runtime binaries (`duckdb`, `python-jose`).
  - 7 focused audit remediation tests (`test_audit_remediation_2026_10_18.py`) and 40 semantic boundary tests (`test_semantic_boundaries.py`) passing at 100%.

## Packaging & Portability
- The candidate archive is packaged using POSIX forward slashes (`/`) for all 1,057+ entries, ensuring clean extraction on Linux, macOS, and Windows without path flattening or backslash escaping issues.
