# MindEd AA-OS Phase 1 Remediation Report

Date: 2026-10-15  
Baseline: `MindEd_AAOS_Plan_Investigation_NL_Enriched_Verified_2026-10-15.zip`  
Baseline SHA-256: `5ce0b1f0569557cae0c87300c69620003cbe483499fa557afd55487cc4e5fcf3`  
Audit bundle supplied by user: `minded_audit.zip`  
Audit-bundle SHA-256: `b44a9aca273d9f7f243d200009e5e3d5e05308f16c6d44789ba4dcc985a4e789`

## Verdict

**Phase 1 source remediation candidate created; not release certified.** Confirmed code paths for the reported epistemic inversion, omitted stopping-verification argument, verification masking, narrow filter extraction, demo-column dimension lookup, row-order-sensitive forecast gate, sparse-period false positive, and malformed quoted CSV were addressed. Focused regression tests pass. The complete production-path audit and release gate could not be re-run in this environment because locked runtime dependencies including `duckdb` and `polars` are unavailable here; the root lock file also omits document-export packages declared by `requirements.txt`. Do not interpret this package as a release-ready certification.

## Baseline evidence preserved

The supplied `audit_matrix_results.json` contains 120 cases (not 126). Its recorded baseline labels are 56 PASS, 16 P1, and 48 P2. The auditor's progress notes separately report 126 real-app-path cases. The raw 120-case matrix, `audit_harness.py`, and `summarize.py` are preserved under `release_evidence/independent_audit_2026-10-15/` and were not rewritten to fit the fixes.

## Changes made

1. **Evidence polarity — `packages/analytics_core/src/runtime/state_reconstruction.py`**
   - FAILED/FAIL/INVALID/REJECTED status is handled before any other inference and cannot be labeled supporting due to a non-zero effect size.
   - Unknown/pending status no longer becomes supporting solely because the effect is non-zero. VERIFIED/PASSED/PASS is required for supporting classification.

2. **Stopping verification gate — `packages/analytics_core/src/engines/stopping.py` and `packages/analytics_core/src/runtime/controller.py`**
   - The stopping API's verification argument now defaults to unknown and resolves unknown to false, rather than defaulting to true.
   - The controller passes its accumulated verification state explicitly into the stopping engine.
   - Final verdict and epistemic calibration now require both accumulated verification success and verified evidence on the leading hypothesis, preventing one successful leading-hypothesis proof from masking a failed/blocked verification elsewhere in the run.

3. **Schema-grounded filter extraction — `packages/analytics_core/src/intelligence/canonical_question_contract.py`**
   - Added arbitrary-schema explicit comparison parsing for `=`, `!=`, `<>`, `>`, `>=`, `<`, `<=`, `is`, and null checks.
   - Added year filters and ISO date ranges as half-open temporal predicates when exactly one date-like column can be identified.
   - Added conservative natural-language dimension filters for region/state/country/segment/category/status/channel/tier/product/SKU and common lifecycle states.
   - Unresolved filter intent with no parsed predicate marks the filter role unresolved rather than silently claiming a complete filter contract.
   - The existing query executor applies parsed predicates through the established typed filter mask; regression test verifies a filtered population and expected total.

4. **Dimension discovery and unknown metric — `canonical_question_contract.py` and `engines/semantic.py`**
   - Removed the demo-dataset-specific dimension candidate loop in favor of phrases derived from the actual dataset schema, with aliases only when they resolve to a real column.
   - Removed the generic fallback to the first table-scoped world-model metric when the question has no semantically matched numeric measure. Unresolved measures remain unresolved instead of being guessed from dataframe/metric ordering.

5. **Forecast temporal gate — `packages/analytics_core/src/runtime/controller.py`**
   - Forecast eligibility no longer requires raw CSV rows to arrive pre-sorted. It requires a parseable, non-degenerate time axis; the forecast engine already sorts cleaned observations chronologically before fitting.

6. **Data-quality false positive — `packages/analytics_core/src/profiling/data_quality_gate.py`**
   - Raised the minimum observations per temporal bucket for the hard-blocking median-discontinuity heuristic from 3 to 5. Three- or four-row buckets no longer trigger this severe blocker solely on a volatile median ratio. This is a conservative guard, not a complete solution to every possible unit-change detection problem.

7. **Malformed CSV quote handling — `packages/analytics_core/src/ingestion/robust_loader.py`**
   - Added strict quote-structure validation before parser/repair fallbacks. Unterminated quoted fields now fail with a clear `ValueError` rather than absorbing subsequent physical rows into a corrupted cell.

8. **Regression suite added**
   - `tests/independent_release/test_independent_audit_remediation_2026_10_15.py`
   - Covers failed evidence polarity, omitted/failed verification stopping, arbitrary-column filters, year/date ranges, natural-language region/status filters, actual filtered aggregation, unfamiliar enterprise dimensions, shuffled forecasting dates, sparse median buckets, malformed quotes, controller verification wiring, and no generic metric fallback.

## Validation performed on the modified tree

- Focused regression and adjacent tests: **29 passed**.
- Python compileall: **PASS** (`python -m compileall -q packages apps scripts tests`).
- Release-status audit script: **PASS** (`python scripts/audit_release_status.py --root .`).
- Canonical serialization audit: **PASS**, 193 files scanned.
- Malformed quote reproduction: **PASS**; loader raises the expected fail-closed error.
- Filter execution reproduction: **PASS**; `discount_rate > 0.1 AND region = North` returns only the matching population and its expected revenue total.

## Tests not completed / remaining blockers

- The full suite and original real-application 120/126-case matrix were **not rerun end-to-end** in this execution environment because importing the production controller requires `duckdb`, and relational-plan tests require `polars`; those dependencies are not installed here.
- `requirements-lock.txt` does not include `python-docx`, `reportlab`, or `pypdf`, although `requirements.txt` declares them. The hash-locked release environment must be repaired with verified package hashes before the documented full suite can collect cleanly. Hashes were not fabricated or added in this patch.
- The existing `tests/test_semantic_boundaries.py` selected run has six assertion failures that also reproduce on the unmodified baseline archive, plus dependency-related errors when `duckdb` is absent. These baseline failures are not counted as regressions introduced by this patch, but they remain real test debt.
- `scripts/release_gate.py`, clean installation, desktop EXE launch, and full analyst answer matrix remain **NOT TESTED** here.
- The natural-language filter parser intentionally supports a conservative deterministic subset. Unsupported or ambiguous filters must still be caught by the contract's unresolved-intent checks; a comprehensive grammar and end-to-end proof across all question families remain necessary.

## Required next acceptance run

1. Repair the lock file through the project's lock-generation workflow and verify hashes against published distributions.
2. Install only from the corrected hash-locked requirements into a clean Python 3.12 environment.
3. Run the complete test suite and release gate; capture exact commands, exit codes, and test counts.
4. Re-run the untouched original 120-case matrix and the auditor's 126-case real-app-path matrix against the patched production application; preserve raw output and compare each case with its original expected answer.
5. Specifically verify failed verification on leading and companion hypotheses, multi-predicate filter preservation, unsupported filter refusal, unknown business measures, shuffled date input, and malformed quoted CSVs.
6. Run fresh-install and desktop EXE smoke tests before making any release-readiness claim.
