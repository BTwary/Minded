# MindEd AA-OS — Plan, Investigation, Filter and Pre-analysis Fix Report

**Candidate:** source-only patch based on `aaos_expand_source_no_dependencies.zip`  
**Date:** 2026-10-16  
**Status:** Focused regression batch passes; real Windows EXE and dependency-backed controller verification are still required.

## Fixed source paths

1. `packages/analytics_core/src/intelligence/canonical_question_contract.py`
   - A comma before an exclusion no longer breaks the dimension association: `show revenue by region, excluding East` resolves `region != East` and groups by `region`.
   - The explicit `by <dimension> <value> split by <dimension>` form is resolved schema-first as a filter followed by a grouping (for example, `Calculate revenue by region West split by channel` filters `region = West` and groups by `channel`).
   - Ambiguous filter dimensions should not be guessed by adding broad fallback rules.

2. `packages/analytics_core/src/governance/guided_investigation.py`
   - `apply_human_guidance_to_contract()` binds the same default descriptive aggregation as the pre-analysis preview before it calculates unresolved roles. This removes a preview-vs-creation mismatch where preview showed `SUM` while creation still asked for an aggregation.

3. `apps/api/src/api/v1/investigations.py`
   - `/interpret` refuses to approve a plan if any selected dataset failed to load, was unavailable, or has a scope error. It returns a structured error rather than silently planning from a partial subset.
   - Pre-analysis always runs the canonical guided contract resolver, even without optional human guidance, so the preview, plan hash and creation path use the same resolution rules.

4. `apps/web/src/components/ProposedAnalysisPanel.tsx`
   - The “Planned Calculation Logic” panel displays the backend's canonical calculation unchanged until the analyst edits a parameter.
   - After edits, the calculation preview keeps all question-derived filters and combines a new UI scope filter as an additional `AND` predicate.
   - Assumptions and scope controls distinguish predicates supplied in the original question from additional UI filters.

5. `apps/web/src/lib/api.ts`
   - API error handling can read FastAPI structured `detail` objects and display their human-readable message instead of an opaque object string.

6. `packaging/windows/build_exe_scope.ps1`
   - The desktop bundle build now deletes any existing `apps/web/out` and reruns `npm ci` and `npm run build` before embedding the frontend. This prevents the real EXE build from silently including a stale frontend export from a previous source revision.

7. `tests/independent_release/test_plan_filter_ingestion_regressions.py`
   - Added regression coverage for comma exclusions, filter-then-split semantics, pre-analysis/creation contract parity, preserving user-question filters when adding UI filters, failing closed on incomplete dataset context, UI preview scope, and stale EXE frontend exports.

## Validation run

Command:

```bash
python -m pytest \
  tests/independent_release/test_plan_filter_ingestion_regressions.py \
  tests/independent_release/test_independent_audit_remediation_2026_10_15.py \
  tests/independent_release/test_plan_intent_merge_regressions.py \
  tests/independent_release/test_typed_investigation_planner.py \
  tests/independent_release/test_intent_state_contract.py \
  -q --tb=short
```

Result: **67 passed in 3.82 seconds**.

Python syntax compilation also passed for:

- `apps/api/src/api/v1/investigations.py`
- `packages/analytics_core/src/governance/guided_investigation.py`
- `packages/analytics_core/src/intelligence/canonical_question_contract.py`

## Important limits

- This environment does not have the pinned Windows release environment and lacks DuckDB, Polars, PyArrow and `python-jose`; controller/API integration tests requiring those packages could not be run here.
- The partial `npm ci` did not complete and the local TypeScript compiler was unavailable. Therefore no frontend production build was certified here.
- No Windows executable was produced in this Linux environment.
- The generic runtime warning “one or more experiments are unverified” is not claimed fixed solely by these source changes. The scope/planning causes are addressed, but the actual calculation’s DuckDB/Polars verification outcome needs to be checked in the real locked Windows build using the screenshot question and dataset.

## Required real EXE validation

From the extracted repository root in Windows PowerShell, with the lock-matching `.venv` and Node/npm installed:

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
.\packaging\windows\build_exe_scope.ps1 -PythonRuntimePath ".\.venv\Scripts\python.exe"
```

The build script now rebuilds the static frontend from the lock rather than reusing an existing `apps/web/out`. After installation, test the same real-dataset scenario: preview the original question, inspect the planned calculation, add a UI filter if needed, run the investigation, and verify both the result population and the verification receipt. A failed verification should remain visibly failed; it must not be bypassed.
