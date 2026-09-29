# Session 21: stress test + release-readiness assessment (2026-09-29)

## Method

Built `scripts/session21_stress_harness.py`: 5 messy, real-world-style synthetic datasets
(SaaS churn, regional sales with a planted regional drop, manufacturing defects with a
planted shift/line interaction, a deliberately tiny HR dataset, a dataset with a 100%-null
column) x 18 natural-language questions total, driven through the REAL
`InvestigationController.execute_investigation` path (not unit-level shortcuts) -- the same
methodology as the 2026-09-13 broad-dataset stress test.

## Result: zero crashes, zero blank terminal states

Every one of the 18 investigations reached `COMPLETED` with a non-empty `direct_answer` and
`main_finding`. No unhandled exceptions, no silent hangs. That's a genuinely solid floor --
the crash-hardening from prior sessions (blank-terminal-state fix, DataQualityGate scoping,
etc.) is holding up under a fresh, independently-constructed set of messy datasets.

## What's actually blocking release readiness

The existing `docs/RELEASE_CERTIFICATION_CURRENT.md` already says **NOT CERTIFIED — RELEASE
BLOCKED** for the v32 lineage this codebase descends from, for reasons that still hold
(independent real-data acceptance not done, no unified semantic-understanding layer, known
rate/correlation verdict inconsistency, no desktop/clean-room/Postgres/browser re-run). This
session's stress test surfaces two additional, concrete findings that sharpen *why* that
verdict is still correct, plus confirms the scientific-calibration backlog tracked since
session 20 is unrelated but equally blocking:

### 1. A clean, answerable correlation question can be refused outright by an internal authority conflict

`"Is churn correlated with support tickets?"` on the SaaS churn dataset (2,000 rows, both
columns present, a genuine planted logistic relationship) returned:

> Investigation inconclusive: Analytical authority conflict: the compiler proposal disagrees
> with the final reconciled contract on predictor_columns, target_column. Refusing to execute
> rather than let a stale plan redefine the analysis.

This is a *deliberate* fail-closed guard (better than silently picking the wrong columns),
but it means the correlation feature doesn't reliably work even for the simplest possible
case: two real, unambiguous, correctly-named columns. This is very likely the same root
cause `test_v19_canonical_authority::test_general_intent_vs_comparison_task` already pins as
a known failure (`IntentEngine` and `UniversalQuestionCompiler` disagreeing on task/intent for
the same question) -- this stress test shows that disagreement isn't just a narrow unit-test
artifact, it's actively blocking real end-user questions in the live controller path. **Not
fixed this session** -- reconciling two independent classifiers is a deeper change than a
targeted regex fix, and doing it carelessly risks exactly the kind of silent-wrong-answer
regression this project has been hardening against.

### 2. Purely descriptive "rate by group" / "average by group" questions inconsistently bypass direct computation

Three examples from this stress test:

- `"What is the churn rate by plan type?"` -> `INCONCLUSIVE`: *"unable to find sufficient
  evidence to distinguish competing explanations"* -- generic hypothesis-testing boilerplate,
  even though this is a purely descriptive groupby-mean request with an obvious, computable
  answer (a rate breakdown per plan tier), not a competing-hypotheses question at all.
- `"What is the average tenure by department?"` on the tiny 6-row HR dataset -> `INCONCLUSIVE`:
  *"Method admissibility gate blocked generic_diagnostic_battery:
  statistical_preflight:group_contains_fewer_than_two_observations"* -- again, a descriptive
  average-by-group request refused outright because one department (HR, n=1) can't support a
  hypothesis test, when a real analyst would just report the three group averages and flag the
  n=1 one.
- `"Why did the defect rate spike on Line-B?"` / `"Is defect rate correlated with
  temperature?"` -> `VARIABLE_NOT_FOUND`: *"the requested rate/proportion outcome could not be
  resolved to any column in the available dataset"*. This one is arguably **correct**
  fail-closed behavior (there's no literal `defect_rate` column, only a binary `is_defective`
  flag a human would obviously aggregate into a rate) rather than a bug -- but it also shows
  rate-phrase resolution only works for the specific pre-known "churn" special case noted in
  the session-20-era backlog notes, not binary-flag-to-rate inference generally.

This is the same *shape* of bug DEFECT-038 (session 20) fixed for cross-sectional "why"
questions -- a purely descriptive request getting routed through the full Bayesian
hypothesis-testing machinery instead of being answered directly -- but recurring here for
"rate"/"average"-by-group phrasing instead of plain aggregates. **Not fixed this session**:
unlike DEFECT-038 (a single, localized `classify_question` guard), this manifests across at
least two different subsystems (the hypothesis hypothesis-testing hypothesis loop's general
"insufficient evidence" path, and the statistical preflight gate), so a safe fix needs more
investigation than this session's budget allowed to avoid papering over the preflight gate's
legitimate small-n protection.

### 3. The session-20 calibration backlog (confirmed still open, unrelated to the above)

Re-confirmed present in this stress test too: `"Is churn correlated with support tickets?"`
would likely still resolve `INCONCLUSIVE` even without the authority-conflict refusal above,
consistent with the known `DEFECT-005`/`test_defect_005_workflows` correlation positive-case
gap already tracked as open.

## Updated release-readiness verdict

**Still NOT RELEASE READY.** Nothing in this session changes that; the certification
document's blockers stand, and two new concrete manifestations of the "no single semantic
understanding layer" and "descriptive vs. hypothesis-testing routing" gaps are now on record
with real repro cases. The good news from this session: no crashes, no blank states, across a
completely fresh set of messy data -- the engineering hardening work (DEFECT-030 through 038,
DataQualityGate scoping, blank-terminal-state fix) is holding.

## Priority list, updated with this session's findings

1. **Scientific-calibration backlog** (session 20's 15 remaining failures, esp. correlation/
   segmentation positive cases and the second `INCONCLUSIVE`-not-`DIAGNOSED` cluster in the
   decisive-single-segment-driver path) -- unchanged top priority.
2. **NEW: compiler-vs-intent authority conflict blocking simple correlation questions** --
   directly blocks a core, advertised capability (correlation analysis) even on trivially
   clean data. Worth investigating before #4/#5 below, since it may share root cause with
   `test_v19_canonical_authority`'s known failure.
3. **NEW: descriptive rate/average-by-group questions misrouted into hypothesis-testing
   machinery** -- same bug shape as DEFECT-038, different trigger words; produces unhelpful
   non-answers to simple, legitimate descriptive questions.
4. Null/zero-denominator/empty-population disclosure (flagged, not yet done).
5. Missing DEFECT-032 ledger entry (housekeeping).
6. Ambiguity alternatives surfaced instead of silent refusal.
7. Shared vocabulary module for recency/causal/diagnostic markers (architectural, lower
   urgency now that DEFECT-037/038 provide fail-closed backstops).
8. Independent real-data acceptance with externally-sourced datasets (per the existing
   certification doc) -- this session's synthetic stress test is a useful complement but does
   not substitute for it.
9. Desktop/clean-room/PostgreSQL/browser re-certification against the current artifact.
10. Local-first packaging / HF demo + Windows .exe -- sequenced last, as before.
