# Audit + merge, 2026-10-17

Inputs: (A) MindEd_AAOS_Filter_Query_Output_Polished_2026-10-16.zip (earlier audit, fixes below), (B) MindEd_AAOS_v1_0_0_complete_source_candidate_patched.zip.
Result: B is the base (it is strictly ahead: 13 new files, 19 modified, release identity/status/docs corrected); audit fixes from A were re-applied on top.

## Merge map
- Taken from B unchanged: all 13 new files (execution_reconciliation.py, benchmark scoring/summary, new tests, release notes, V1.0.0 status docs),
  controller.py (human-correction replay for locked contracts, reconciliation hook, denominator record), approved_execution.py (LF normalisation + import),
  metric_semantics.py (COUNT no longer needs a column), universal_question_planner.py (confidence<0.70 refusal), settings.py (missing `settings` import),
  InvestigationGraphView.tsx, release docs/evidence/scripts, requirements-lock.txt, deleted .bak files.
- Merged (both changed): canonical_question_contract.py, ProposedAnalysisPanel.tsx, StructuredAnalysisAnswer.tsx (my patches applied on B's version; no conflicts).
- From my audit (B did not touch): guided_investigation.py, controller_scope.py, regression tests.
- Dropped: my earlier root AUDIT note and ATTESTATION (regenerated); docs/_c4_2_2b_raw_capture.json kept as in B (my copy was a test side effect).

## Bugs fixed (earlier audit, still present in B)
1. WHERE-body clause splitting: ", calculate ..." swallowed into IN list. 2. Scope validator: negative numbers, ;/-- inside literals, BETWEEN column/NULL bypass, NULL bounds.
3. `1e3` literal parsed as text. 4. `SUM(<category>)` preview for distinct-values questions. 5. UI: BETWEEN display, apostrophes, exact audit counts.

## New bugs found in B's new code
6. execution_reconciliation.ranking_direction_matches: false MISMATCH for non-RANKING tasks that carry a rank direction. This was a regression vs. the previous zip:
   test_acceptance_criteria_matrix::test_clarification_lifecycle_end_to_end passed on the 2026-10-16 zip and failed on B (answer refused). Fixed + 3 tests.
7. execution_reconciliation: IndexError when comparison_values exist without a grouping column; now fails closed.

## Tests
New: tests/independent_release/test_audit_2026_10_17_filter_regressions.py (23 cases). One stale assertion updated (test_method_selection_spine quoted identifiers) — please review, B's status doc asks for adjudication of such tests.
Run on the merged tree: first 48 + 49-62 files of tests/independent_release, 40 files touching planner/semantics/guidance/controller/reconciliation, all filter and release-audit files.

## Second pass: the previously open failures, now fixed
8. "Why did cost_metric surge across datacenter_region?" was compiled as a COMPLETE grouped SUM (the compound grouping rule ignored causal wording), which dropped the "why" and made the controller downgrade DIAGNOSED -> OBSERVED. One guard in canonical_question_contract fixed test_defect038 and 4 test_defect_007 controller tests.
9. "Average revenue per order" bound `order_id` (unique on every row) as the grouping column -> one-row groups. Row-unique identifiers are no longer bound as grouping; the question is refused. "per customer" with repeated customers still works.
10. Forecast: horizon regex needed a digit, so "next month" was UNKNOWN and the plan was blocked; the refusal message also wrongly blamed the time index. Regex now accepts an implicit single period; message names the horizon.
11. Dual-engine verification: polars_query_recompiler only understood unquoted identifiers, but the executor now emits quoted ones (SUM("revenue")), so the independent engine computed nothing and valid forecast evidence was marked FAILED. Simple quoted identifiers are normalised (string literals untouched; identifiers needing quotes still fail closed).
12. Judgment call (please review): test_defect_005 "Which products are performing badly?" asserted DIAGNOSED, which required silently choosing unit_cost as the outcome. Test now asserts the fail-closed INCONCLUSIVE with a missing-outcome message. Also updated earlier: test_method_selection_spine quoted-identifier assertion.

Regression tests: 32 cases in test_audit_2026_10_17_filter_regressions.py. First 48 files of tests/independent_release: 482 passed, 0 failed. Previously failing: all now pass.

## Third pass: understand the question at first instance, ask the fewest questions
Driven by the product screenshots (a clear question produced a 17-column radio list and a dead end).

Understanding (no question asked):
13. Filter extraction ignored population qualifiers named by value. "Among Corporate orders that were delivered and not returned" bound only order_status=Delivered and silently dropped "Corporate" and "not returned". New `extract_observed_value_filters`: a value becomes a filter only if it occurs in exactly one categorical column (never guessed), supports negation (not/non/excluding/except/without/other than -> !=) and "A or B" -> IN; skips group/target columns and values that are also column-name words.
14. The compound splitter cut ", which ..." and treated the scope clause as a second objective ("multiple analytical operations"). Scope-only clauses (no task/aggregation word) now merge into the following clause; real multi-part questions still split.
15. "average X per order" on an order-level table (order_id unique on every row) is the mean over rows. Resolved with a recorded `row_grain` assumption, no denominator question. Not guessed when the grain is not verified, and "total X per order" is still refused.
16. "Which region has the most / fewest orders" was shown as CLEAR with `SUM(units)` / `SUM(gross_sales)` (a different measure silently substituted). Now compiled as COUNT(DISTINCT order_id) per group with the right direction; preview, assumptions and execution agree.
17. Preview SQL and the "Population" assumption ignored bound filters ("All available records" next to a WHERE clause). Both now come from the contract; the displayed metric can no longer differ from the one the contract computes.

Asking (only when truly ambiguous):
18. New `focus_objective` patch: the options for a multi-part question ARE its parts, in the user's words; for "per customer" ratios there are two executable readings. A click, an ordinal ("1") or one typed question resolves it; a typed question that still contains two asks is refused, not half-answered. Replaces the single "Split the request" dead end and the "Freeform patching is not supported" error.
19. Clarification suggestions were `every column in the table`. Now `compact_resolution_candidates`: at most 3, ranked by overlap with the question, role-aware (measures exclude identifiers; denominators are identifiers; groupings exclude numeric/ID/time).
20. UI: filter builder no longer pre-selects the first column (usually an ID) and no longer shows "Enter a value for filter 1" before the user has edited anything.

Tests: tests/independent_release/test_first_instance_understanding.py (8 tests, incl. the exact screenshot question). Two tests that encoded the old "average per order is refused" premise were updated to "resolved when order grain is verified, refused otherwise" (please review).
Run on the merged tree: first 48 + files 49-62 of independent_release, 45 files touching the changed modules (512 passed), plus acceptance/defect_001/005/007/038 (56 passed): 0 failures.

## Not done
Tests 63-148 of independent_release and most of tests/ (slow: ~15 s per file here), the 36-check release gate, next build/typecheck (TSX syntax-checked only), Windows build.
ATTESTATION.json regenerated and verifies, but binds B's BLOCKED gate report: it is evidence binding, not certification.

## Fourth pass (2026-10-18): merge of two diverged branches
`MindEd_AAOS_v1_0_0_merged_audited_2026-10-17` (second+third pass) and `MindEd_AAOS_v1_0_0_intent_first_2026-10-18` were sibling
branches of the same first-pass tree, not a chain. Base = merged_audited; the intent_first fixes it lacked were ported by hand
(overlapping fixes kept in the merged_audited form: row-grain, scope-clause merge, forecast horizon, quoted-identifier recompiler).

Ported from intent_first:
21. Negated lifecycle words ("orders that were **not delivered**") were bound as `= Delivered` - the opposite population, answered silently. Now `!=`.
22. 0/1 flag columns bound from plain words ("not returned" -> `returned = 0`), alongside the observed-value filters.
23. Ranking contracts always raised in the approved-execution verifier; the executed ordering is now independently checked instead (count rankings could compile but never run). analyst_answer also records `top`/`top_value`.
24. "Compare X between Y" / "X versus Y" compiles as a group comparison (was read as a date range). Broader rank verbs ("generated", "earned"...), measure scope tails ("lowest discount among ..."), "what was/were", return/cancellation-rate synonyms.
25. A capitalised population word that matches no observed value ("Among Enterprise customers") is no longer dropped: the person is asked, with <=6 observed values as options.
26. SQL scope filters reject injected constants (`x = 1 OR 1=1`, `OR TRUE`, standalone constants); categorical SUM defaults to ROW_COUNT; MISSING_DATA gets its own message; "sales" channel synonyms no longer swallow "sales rep".
Dropped: `data_store/s3_cache` (runtime cache).

Tests: tests/independent_release/test_merge_2026_10_18_unique_branch_fixes.py (11), tests/test_first_pass_question_understanding.py (2).
tests/test_semantic_boundaries.py takes the quoting-tolerant assertion from intent_first.

Targeted set (81 files touching the changed modules): merged_audited 12 failures -> merged 3. The 3 remaining fail identically in BOTH source zips and are not root-caused:
- test_session10 `test_ranking_is_answered_and_observed`: answer is the plain table, not "<top> has the highest ...".
- test_session10 `test_multi_group_numeric_comparison_end_to_end_controller`.
- test_session6 `test_plateau_after_early_learning_is_recognised_as_stagnation` (stopping rule returns INSUFFICIENT_DISCRIMINATION).
The full suite and the release gate were NOT run. Gate remains BLOCKED.
