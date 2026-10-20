# AA-OS v1.0.0 Release Status

**Date:** 2026-10-17
**Archive:** `MindEd_AAOS_v1.0.0_complete_source_candidate.zip`
**Release Status:** NOT RELEASE READY — source candidate; release gate BLOCKED

This record supersedes `docs/history/AAOS_V40_RELEASE_STATUS_SUPERSEDED.md` and the earlier
"Release 1.0 candidate / 32-of-32" claims, which did not reproduce when re-run.

## Measured results (independent audit, 2026-10-17, Linux, CPython 3.12, hash-locked dependencies)

- Release gate: 36 manifest checks executed; 26 passed, 10 failed (before the fixes listed below).
- Test suite: 36 failed / 1830 passed before fixes; 35 failed / 1831 passed after the COUNT-metric fix.
  Roughly 6 failures are stale expectations from identifier quoting (`SUM(revenue)` vs `SUM("revenue")`),
  1 needs live DNS, and the remainder are real behaviour failures that are not yet root-caused.
- Real-data stress run (`scripts/session23_real_data_stress.py`), raw variant: 23 of 32 correct after the fixes
  (21 of 32 before); clean variant: 22 of 32 after the fixes (19 before). The previously shipped 32/32 evidence was stale.
- Passing: compileall, Alembic upgrade-to-head and no-drift, canonical serialization audit, dependency lock, frontend lock.

## Known open defects

- Approved-plan executor fails closed (INCONCLUSIVE/FAILED) for derived rates and rankings without an operation graph
  (e.g. churn-rate questions).
- Stress run: 5 CONTRACT_WRONG (trend questions), 2 NUMERICALLY_WRONG, 2 INCOMPLETE_OUTPUT.
- Remaining failing tests listed by `pytest tests packages`; stale quoting tests need adjudication, not silent edits.
- Windows installer/exe and desktop lock were not rebuilt or verified.

## Fixed in this archive

- Row-count metric with no column no longer crashes SQL generation; replan fallback failure no longer escapes the controller.
- `DatasetVersion` import restored (dataset version-change assessment).
- `POST /settings/feedback` NameError.
- `ProposedAnalysisPanel.tsx` TypeScript error (`next build` blocker).
- `requirements-lock.txt` now includes pypdf, python-docx, reportlab and their dependencies.
- Release identity aligned to v1.0.0; hardcoded absolute path removed from the audit harness; caches and `.bak` files removed.

## Merge audit addendum (2026-10-17)

Fixed on top of this archive (details and tests: `AUDIT_2026-10-17_merge_and_fixes.md`):

- `execution_reconciliation`: `ranking_direction_matches` wrongly flagged MISMATCH for grouped descriptive tasks carrying a rank
  instruction (broke `test_clarification_lifecycle_end_to_end`); comparison values without a grouping column now fail closed instead of `IndexError`.
- Filter extraction swallowed a trailing task clause into an `IN` list; scope-filter validator rejected negative numbers and `;`/`--` inside
  string literals, and let `BETWEEN 1 AND <column>` / NULL bounds through; `1e3` parsed as text; preview rendered `SUM(<category>)` for
  distinct-values questions; UI rendered `BETWEEN` as `(a, b)` and abbreviated audit row counts.
- Test assertion in `test_method_selection_spine.py` updated to the quoted-identifier form (one of the stale-quoting tests noted above).

Second pass (see audit note items 8-12): causal-wording guard, row-unique grouping guard, forecast horizon, quoted-identifier recompiler, one test adjudicated. All previously failing tests now pass in the files run. Gate remains BLOCKED (full gate not run).

## Third pass (2026-10-17): first-instance understanding

Observed-value population filters with negation, scope-clause merging in the compound splitter, verified order-grain "per order",
count rankings ("which region has the most orders"), contract-faithful preview SQL/assumptions, and a minimal-question clarification
model (`focus_objective`, <=3 ranked options, no column dumps). See `AUDIT_2026-10-17_merge_and_fixes.md` items 13-20.
Gate remains BLOCKED (full gate not run).

## Fourth pass (2026-10-18): branch merge
Merged the intent_first branch into the merged_audited branch (audit note items 21-26). Targeted 81-file set: 3 failures, all present in both source zips.
Full suite and release gate not run; ATTESTATION regenerated for this tree but its gate report is the older one. Gate remains BLOCKED.
