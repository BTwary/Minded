# BUGFIX 2026-09-27 (session 20): cross-sectional diagnostic questions silently downgraded to RANKING (DEFECT-038)

## What was asked

Priority #1 from the project's own priority list: root-cause the scientific-calibration
backlog rather than continuing NL-parsing work.

## Baseline established first (this session's own verification standard)

Prior sessions verified only a DB-independent slice of the suite. This session installed
the full dependency set (`requirements.txt`: duckdb, polars, pydantic-settings, alembic,
statsmodels, gradio, etc.), ran `alembic upgrade head` against a real local SQLite DB, and
ran the actual full suite for the first time in several sessions:

- `tests/independent_release/`: **17 failed / 961 passed** (before this session's fix)
- `tests/` (excluding independent_release): 311 passed / 1 skipped
- `packages/analytics_core/tests`: 77 passed

## Root cause found (DEFECT-038)

Several of the 17 failures shared an identical symptom -- a test expecting `DIAGNOSED`
getting `OBSERVED` instead (`test_session7_o3_simpsons_refutation`,
`test_session8_verifier_workflow`), and others expecting `DIAGNOSED` getting `INCONCLUSIVE`.
Traced the `DIAGNOSED` -> `OBSERVED` pair to a single shared root cause:

`classify_question` (`analyst_answer.py`) has a final catch-all --
`if has_group: return "RANKING"` -- that fires for *any* group-shaped question not caught by
a more specific branch. This includes genuinely causal/diagnostic questions like "Why did
cost_metric surge across datacenter_region?": no time column is involved (it's cross-
sectional, not a before/after comparison), so none of `PERIOD_CHANGE`, the DEFECT-037
fail-closed branch, `TREND`, `RANKING`'s own explicit patterns, or `GROUP_COMPARISON` match
it -- it falls all the way to the bottom and is classified as a plain `RANKING` question.

`RANKING` answers are marked `descriptive=True, supersedes_loop=True`. `controller.py`
(~line 4349) uses exactly that flag to overwrite the hypothesis loop's own verdict --
*including a correct `DIAGNOSED` verdict backed by real Bayesian evidence* -- with a shallow
"group X has the highest total" recomputation that never actually answers the causal "why"
the question asked. This is not a scientific-calibration bug in the Bayesian engine itself;
the engine was reaching `DIAGNOSED` correctly, and the NL classification layer was throwing
that verdict away afterward.

## The fix

`packages/analytics_core/src/engines/analyst_answer.py`: added a guard in `classify_question`,
placed after `GROUP_COMPARISON` and before the `RANKING` catch-all: if the question matches
both `_PERIOD_CHANGE_RE` (genuine diagnostic language -- "why", "what caused", etc.) and
`_CHANGE_WORDS_RE` (genuine change language -- "surge", "spike", "drop", etc.), return `"NONE"`
instead of falling into `RANKING`. `build_analyst_result` already treats `"NONE"` as "say
nothing rather than guess" (the same signal already used elsewhere in this function, e.g. for
missing required columns) -- returning it here means `build_analyst_result` returns `None`,
the controller's superseding block is skipped entirely (it's gated on
`analyst_result is not None`), and the hypothesis loop's own verdict is left untouched.

This is scoped narrowly: the guard checks the *question's own* content, not whether the
dataset happens to carry an unreferenced time column, so it fires identically whether or not
`has_time` is `True`. It sits after `RANK_RE`/`BREAKDOWN_RE` and `COMPARE_RE`/`AVG_RE`, so a
question that is *both* diagnostic-worded *and* an explicit ranking/comparison request (e.g.
"why is the West region the top performer") is unaffected -- those still correctly resolve to
`RANKING`/`GROUP_COMPARISON`.

## Verification

- New suite `tests/independent_release/test_defect038_crosssectional_diagnostic_not_ranking.py`
  (7 tests): confirms the fix fires regardless of `has_time`, confirms `build_analyst_result`
  returns `None` for this shape end-to-end, and confirms `RANKING`/`GROUP_COMPARISON`/
  `PERIOD_CHANGE`/`PERIOD_CHANGE_UNRESOLVED_TIMEFRAME` are all unaffected (the DEFECT-037 and
  DEFECT-038 guards don't shadow each other).
- Updated one DEFECT-037 test (`test_question_with_no_time_reference_at_all_is_unaffected`)
  whose prior assertion (`"Why did revenue fall in the West region?"` -> `RANKING`) was itself
  an instance of this exact bug and is now correctly `NONE`; the test now asserts what it
  actually needs to (that this isn't wrongly caught as `PERIOD_CHANGE_UNRESOLVED_TIMEFRAME`).
- Ran all eight recency/diagnostic-classification suites together (DEFECT-031/033/034/035/036/
  037/038): 61/61 pass.
- **`test_session7_o3_simpsons_refutation` and `test_session8_verifier_workflow` now pass** --
  confirmed root cause.
- Full `tests/independent_release/` re-run: **15 failed / 970 passed** (was 17 failed / 961
  passed) -- 2 genuine fixes, +9 passing (7 new tests + the 2 flipped), 0 regressions (every
  previously-passing test still passes; the failing-test list is a strict subset of before).
- Re-ran `tests/` (311 passed / 1 skipped) and `packages/analytics_core/tests` (77 passed)
  unchanged.
- This is the first session in this line to actually install the full dependency set and run
  `alembic upgrade head` against a real DB rather than verifying only a DB-independent slice --
  the "still owed" caveat from the DEFECT-036/037 sessions is now closed for this run.

## Remaining 15 failures (not touched this session, for the record)

Grouped by apparent shared cause where discernible:

- **Genuinely open scientific-calibration/verdict gaps** (the actual DEFECT-005/007/015
  backlog): `test_defect_005_workflows` (correlation & segmentation positive cases),
  `test_defect_015_churn_autonomous_integration` (churn fail-closed), `test_p0_closure` (BYOM
  schema validation, isotonic calibration usability), `test_controller_question_matrix` (rate
  family null-result verdict propagation), `test_v19_canonical_authority` (intent classifier
  vs. compiler task disagreement on "what drives higher revenue between regions?").
- **A second cluster, same `INCONCLUSIVE`-instead-of-`DIAGNOSED` symptom as this session's
  fix but NOT resolved by it** -- worth investigating next, likely a distinct root cause in
  the decisive-single-segment-driver path itself, not the NL classification layer:
  `test_defect_007_controller_closure::test_stopping_case_a_decisive_resolution_stops_early`
  and `::test_continuous_metric_controller_execution` (both: a single dominant segment should
  produce `DIAGNOSED` in one round, but the verdict computation itself reaches `INCONCLUSIVE`).
- **A naming/sequencing assertion, not a verdict-correctness bug**:
  `test_defect_007_controller_closure::test_real_adaptive_replanning_multi_round_trace` now
  fails on a *different* assertion than before this session (previously: no isolation
  experiment ran at all; now: an isolation experiment does run, named
  `EXP-VERIFY-ISOLATE-HYP-01` instead of the `EXP-ISOLATE*` prefix the test expects) --
  possibly a test expectation that's stale relative to a legitimate naming change elsewhere,
  not re-investigated this session.
- **Two-table/relational and idempotency**: `test_autonomous_multitable_analyst`,
  `test_autonomous_orchestrator_verification_status`, `test_session5_regression_closure`,
  `test_zero_ai_local_autonomy` (this last one's `INCONCLUSIVE` result was already present
  before this session's fix -- confirmed unrelated, not a regression).

## Still open

- The 15 remaining failures above are real and untouched -- this session fixed one specific,
  now-confirmed root cause (2 tests) out of the larger backlog, not the whole thing.
- The architectural note from DEFECT-036/037 stands: NL classification here is still a
  hand-enumerated regex table. DEFECT-038 is the same fail-closed philosophy as DEFECT-037
  (disclose/abstain rather than silently substitute a shallower answer), applied to a
  cross-sectional case instead of a temporal one.
