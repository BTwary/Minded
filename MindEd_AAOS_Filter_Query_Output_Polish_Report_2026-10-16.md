# MindEd AA-OS — filter semantics, investigation and output polish

**Candidate:** source tree derived from the 16 October 2026 plan/investigation/filter/ingestion fixes  
**Decision:** source-level regression batch passes; full release and Windows EXE remain unverified in this environment.

## What was wrong

For a request such as:

> Calculate the total `realized_revenue` for each region, excluding records where `order_status` is Returned or Cancelled. Return the total for every region.

the question contract could swallow the trailing sentence into a filter value or emit conflicting filters. That made the plan ambiguous or wrong despite presenting a complete-looking plan. The output panel could show a proposed SQL statement that looked authoritative even though a proposed plan is not evidence that the same query ran successfully.

## Changes made

### 1. Natural-language filter parsing

- Same-column alternatives are converted to typed list predicates when intent is explicit: `status is Delivered or Shipped` → `status IN ('Delivered', 'Shipped')`; explicit exclusion of two statuses → `status NOT IN ('Returned', 'Cancelled')`.
- Exclusion parsing does not use the preceding grouping dimension as an implicit filter column when a unique lifecycle/status field is available. If a valid field cannot be determined, the contract must stay unresolved instead of guessing.
- The explicit question predicate is hidden from the generic fallback regex after conversion so it cannot be duplicated as a conflicting equality/exclusion.
- The regression test checks the screenshot’s business question, selected metric (`realized_revenue`), `SUM`, grouping (`region`), the `NOT IN` filter, and parity between question preview and creation contract. With the test fixture, the post-filter grouped totals are North = 30 and South = 40.

### 2. User-supplied filter language and SQL/database behavior

The filter path now follows these stages:

1. **Interpret intent against schema.** The question parser creates typed predicates (column, operator, value) only when the intent and column/value are grounded in the selected dataset. It does not accept a user sentence as an arbitrary SQL query.
2. **Review in the pre-analysis panel.** The panel exposes dataset-backed fields and supported operators. It previews the question-derived filter and any extra user-supplied filter separately. Additional filters are combined with the original conditions using `AND`, rather than silently replacing them.
3. **Resolve and validate.** The scope-filter validator parses the filter as a SQLGlot predicate AST, checks each column against the selected dataframe schema, allows only the documented predicate nodes, rejects unsupported expressions and normalizes the accepted AST into DuckDB dialect.
4. **Execute the validated expression.** The controller sends the normalized predicate—not the original raw filter text—to DuckDB. If the filter returns zero rows, execution fails closed; AA-OS does not silently retry against the unfiltered population.
5. **Report the actual run.** The proposed SQL is only a preview. The saved execution record and verification result are the authority for which scope was actually run and whether it was independently verified.

Supported advanced filter forms are comparisons (`=`, `!=`, `<`, `<=`, `>`, `>=`), `IN` / `NOT IN`, inclusive `BETWEEN`, `IS NULL` / `IS NOT NULL`, joined with `AND`. Natural-language alternatives on one column can normalize to `IN`/`NOT IN`; the advanced filter field does **not** allow arbitrary SQL or `OR`. Use the typed UI builder for ordinary filtering. `NULL` comparisons must use `IS NULL` / `IS NOT NULL`; `BETWEEN` includes both bounds. `WHERE` is applied before `GROUP BY` and aggregation.

**Security note:** this path uses schema checks, an AST whitelist and normalized SQL. It does not yet bind each literal using prepared-statement placeholders; the advanced filter grammar remains deliberately restricted and arbitrary SQL is rejected. This limitation is documented, not represented as parameterized SQL.

### 3. Investigation panel and result presentation

- The planned calculation is shown as a preview, with explicit separation between planned logic and actual execution/verification.
- The condition builder uses selected-schema columns, operator-appropriate values, escaped string literals, numeric/boolean validation, and range validation for `BETWEEN`.
- The question-derived scope is preserved when the analyst adds an extra scope predicate; the extra predicate adds another `AND` term instead of dropping the original conditions.
- Dataset-ingestion/selection failures stop planning with an actionable error rather than pretending that a partial dataset scope is complete.
- The analysis output adds an “Analysis at a glance” section from the saved execution record: dataset, population row count, result/group count, target metric/aggregation, executed filters, date range, and verification state.
- Text values are displayed as quoted SQL-like literals for clarity; list filters are shown as `IN (...)` / `NOT IN (...)`, and exclusive date-end semantics are labeled explicitly.
- Errors from the API are surfaced to the user instead of being collapsed into generic failure text.
- Windows packaging is configured to rebuild the frontend from the lockfile and fresh sources rather than silently reuse stale static output.
- Removed six obsolete `.bak` source copies from the release tree; none is imported by the application. This avoids shipping stale alternate implementations that are easy to mistake for active source.

## Verification performed

The following focused batch was run in the current environment:

- **131 passed, 1 skipped** in the selected test batch.
- Python syntax compilation: pass.
- TypeScript transpile/syntax checks for `ProposedAnalysisPanel.tsx`, `StructuredAnalysisAnswer.tsx`, and `apps/web/src/app/analyst/page.tsx`: zero syntax/transpile diagnostics.
- Canonical serialization audit: pass (199 source files scanned).
- Release-status audit: pass.
- Frontend lock audit: pass (`next=15.5.26`).
- Attestation verification: pass; attestation/source/report hashes match.

The one skipped test module is the SQLGlot filter predicate guard; `sqlglot` is not installed in this environment. The wider environment also does not match the project’s pinned runtime: DuckDB, Polars, PyArrow, and `python-jose` are missing or version-mismatched. Two API/controller integration tests could not collect because `python-jose` was unavailable in an earlier wider run. The frontend `node_modules` are absent, so a full Next.js production build/typecheck was not run.

The refreshed `release_evidence/GATE_REPORT.json` deliberately says **BLOCKED**, with all manifest checks marked `NOT_RUN`. The manifest has 36 checks. The final attestation binds the shipped source tree (711 hashed source files; SHA-256 `891ae081c34480a8332c322f7abe8e2be60a759560098b67dbe04d949593d15f`) to that blocked gate report. Its previous gate report is retained as `release_evidence/GATE_REPORT_PRE_FILTER_OUTPUT_POLISH_2026-10-15.json` for historical reference only. Attestation verification confirms tree/report identity; it does **not** mean the full release gate passed.

## What remains before the real EXE can be certified

1. In the pinned CPython 3.12 Windows environment, install dependencies from the repository’s lock and run the full 36-check release manifest.
2. Run the SQLGlot predicate tests and DuckDB/Polars controller/investigation integration tests against the exact candidate.
3. Rebuild the Windows EXE using the fresh frontend export, then test the screenshot question in the actual executable. The plan should show `order_status NOT IN ('Returned', 'Cancelled')`; the result should show the correct grouped totals for the actual test data; saved execution filters and verification receipts must match the approved plan.
4. Re-generate and verify release evidence only after no more code changes are made.

This report is a source audit record, not an EXE certification or a declaration that release readiness has passed.
