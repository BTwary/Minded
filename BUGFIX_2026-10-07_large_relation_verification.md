# 2026-10-07 — real 120k-row dataset fixes

Reproduced on orders.csv / sales_orders_flat_120k.csv with "Which region has the highest total revenue?".

1. Raw-row relation summarised as groups (transition.py): the "leading category" was the region of the single
   largest ORDER (Central, 0.0%). Raw-row additive relations are now summed per group from the full analysis frame.
2. Truncated relations could never be verified (execution_provider.py, verification.py, transition.py, controller.py):
   results over MAX_QUERY_RESULT_ROWS (50k) failed dual-engine verification by design. The executor now fingerprints
   the FULL relation (rows, non-null counts, exact fsum column sums, category counts); Polars must reproduce it.
   No digest or no independent result still fails closed.
3. Segmentation/correlation/forecast hypothesis priors were never normalised (both 1.0 -> "P(H|E)=100%"). Shared normaliser.
4. Experiment.analysis_row_count / full_scope / sampling were never updated after execution (stayed 0). Now persisted.
   The persistent instance from session.merge is kept (a second merge queued a duplicate INSERT). Side effect:
   calculation_trace_id/hash written after the merge now persist in arguments_json.
5. Multiverse grid pinned to the question's estimand (a total => SUM only; mean metric => MEAN only), and concordance
   tracks the baseline group's own side (top stays top) instead of the largest absolute deviation, which flipped to a
   different group under trimming and produced a spurious "2/9 fragile" blocker.
6. API main_finding for OBSERVED answers is the deterministic headline, not the hypothesis loop's posterior text.
7. Repro SQL for raw-row evidence is a per-group SUM ... GROUP BY query a human can compare to the headline.
8. "dataset scope recorded" no longer FAILs for a single-dataset analysis whose experiments record their dataset.

Tests: tests/independent_release/test_large_relation_and_prior_fixes.py (new, 4);
test_session8_verifier_workflow.py updated 9 -> 3 specifications (pinned MEAN grid).

9. decision_status is synced after data-quality/version reviews append open questions (RESOLVED -> PARTIALLY_RESOLVED).
10. Evidence "Dataset:" shows name@fingerprint (was hard-coded "1"); row_count_analyzed falls back to the experiment's
    recorded analysis_row_count if the observation lookup misses (the likely source of UI "Rows Analyzed: 0").
11. Analysis plan response carries executed_experiments; the analyst page shows "Executed: EXP-CONC-HYP-01 (120,000 rows)"
    beside the planner's proposed EXP-DESCRIPTIVE instead of leaving them silently inconsistent.
12. Investigation DAG: banner for OBSERVED answers ("hypothesis probabilities do not describe the answer"); target
    variable no longer hard-coded to "Revenue".

Still true: the hypothesis loop still RUNS for descriptive questions (it produces the verified EXP-CONC evidence); only
its presentation changed. Web changes (page.tsx, InvestigationGraphView.tsx, types) were not compiled or rendered -
no node_modules in the sandbox. The SSRF network test fails offline (also fails on the original zip).

13. Follow-up from auditing the evidence layer on 4 question types: multiverse pin now maps rate/ratio/proportion/average
    to MEAN (return-rate question went 6/9 -> 3/3 variants), and the Verify endpoint (/investigations/{id}) returns the
    deterministic headline as verdict.justification for OBSERVED answers instead of the loop's posterior text.
