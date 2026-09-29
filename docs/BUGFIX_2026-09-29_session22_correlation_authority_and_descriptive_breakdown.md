# Session 22 (2026-09-29): the two session-21 stress-test findings

Baseline: v37 session-21 zip. Deps installed from PyPI; `test_hf_space_app.py` excluded (needs `gradio`, unrelated).

## DEFECT-039 -- clean two-column correlation question failed (two independent bugs)
"Is churn correlated with support tickets?" was refused with "Analytical authority conflict ... predictor_columns, target_column".

1. `intelligence/experiment_contract_validation.py`: compiler plan said target=support_tickets/predictor=churned, canonical
   resolution said the reverse -- same two columns, roles swapped. Pearson is symmetric, yet the guard marked it BLOCKING.
   Now OVERRIDDEN (recorded, not blocking) ONLY when: task ASSOCIATION, method in `SYMMETRIC_ASSOCIATION_METHODS`
   (`association_numeric`), exactly one predictor each side, identical swapped column pair. Everything else still BLOCKS.
2. `engines/belief.py` (found only after fixing #1): the correlation Bayes factor preferred the wider `primary_df` over the
   experiment's `result_df` whenever primary_df was longer. `support_tickets` has nulls, so the pairwise-complete result was
   shorter, more than one extra numeric column was found, and the evidence silently became BF=1.0 (NEUTRAL) for r=0.125,
   p=2.6e-8. `tenure` (no nulls) always worked, hiding this. The CORRELATION branch now uses `result_df` when available.
Result: STATISTICALLY_SIGNIFICANT for both phrasings.

Note: the session-21 write-up guessed this shared a root cause with `test_v19_canonical_authority::test_general_intent_vs_comparison_task`.
That test STILL FAILS (identical before/after), so that link is unproven.

## DEFECT-040 -- descriptive "<measure> by <group>" questions refused / mis-owned
1. `engines/method_selection.py`: the "every group needs >= 2 observations" preflight is a precondition for a group TEST, not for
   reporting a mean. Waived only for generic_diagnostic_battery + task DESCRIPTIVE + that single reason; waived reasons are
   recorded as `descriptive_small_n_waived` in the gate payload. Row-count/quality/insufficient_groups blocks still apply.
2. `engines/analyst_answer.py`: new `is_pure_descriptive_breakdown()` (needs a measure word AND a group cue; ANY comparative,
   causal, ranking, temporal or predictive word disqualifies).
3. `runtime/controller.py`: churn-loop ownership of the final answer no longer applies to a pure breakdown, so the analyst
   layer's exact recomputation answers it. Ranking / "why" / "differ" / time-scoped churn questions are unchanged.
Result: "What is the churn rate by plan type?" -> OBSERVED with per-tier rates; tiny-HR average question now returns all three
department averages with the small-group caveat.

## Verification
- Full suite (analytics_core + benchmark + independent_release, hf_space excluded), before vs after: 9 failed/1062 passed both;
  failure sets IDENTICAL (zero regressions).
- New: `tests/independent_release/test_session22_correlation_and_descriptive_breakdown.py` (9 tests). Against the unfixed code
  7 fail (the 2 that pass are "still blocks" guards). All 9 pass after.
- Session-21 stress harness re-run: 18/18 COMPLETED, no crashes, no blank findings.

## Still open / honest gaps
- tiny-HR "average tenure by department" now shows the right numbers but the VERDICT is still INCONCLUSIVE (claim admission at
  n=6, not investigated), the headline still says "HR has the highest average (n=1)" (over-assertive for n=1), and
  `main_finding` is still hypothesis-loop boilerplate ("aggregated as SUM ...").
- "Which plan has the highest churn rate?" and "Why are customers churning?" still return the generic INCONCLUSIVE
  (ranking/why churn path, untouched, deliberately out of scope).
- Manufacturing `VARIABLE_NOT_FOUND` (no literal defect_rate column) left as fail-closed; binary-flag-to-rate inference not built.
- The 9 pre-existing failures are unchanged. Ledger has no entries for DEFECT-031..038 (only 039/040 added here); numbering follows
  the session-21 doc's "DEFECT-030 through 038".
- Release status: NOT RELEASE READY.
