# BUGFIX 2026-09-27 (session 18): recency-vocabulary and diagnostic-vocabulary gaps (DEFECT-036)

## What was asked

Independent audit of `Minded_AAOS_v33_2026-09-27_defect032-035_intent_period_fixes.zip`
("identify why it is failing in analyzing real datasets and why its NL question layer
fails to parse and generate the required output"), followed by "fix these" on the
findings.

## What I did

Verified DEFECT-033/034/035 first: extracted the zip, ran all 25 of their regression
tests in isolation -- all pass, the fixes are real. (One documentation note: the zip
name and test docstrings reference "DEFECT-032", but no DEFECT-032 ledger entry or
artifact exists anywhere in the delivered tree -- only a stray comment in `intent.py`
referencing it. Not touched this session; flagged for the record.)

Then independently stress-probed the same two functions those defects just patched
(`classify_question`/`parse_period` in `analyst_answer.py`, `_DIAGNOSTIC_RE` in
`intent.py`) with ordinary paraphrases *not* covered by the new fixes, since every prior
session in this line found the same failure shape recurring under new wording. It did
again.

## The bugs (DEFECT-036)

**A. `parse_period` still returned `None` for several common recency phrasings**, causing
the same silent RANKING-fallback failure as DEFECT-031/035:
- `"year to date"` / `"YTD"`
- `"since the start/beginning of the quarter/month/year"`
- a bare `"fortnight"` with no explicit count (`"in the past fortnight"`)
- `"compared to a year ago"` / `"a year ago"` used as a trailing-window anchor
- `"last/past/trailing <N> quarters"` and `"... months"` where N > 10, or where the
  unit was `quarter`/`fortnight`/`year` (the word-number table stopped at "ten" and the
  unit set only accepted day/week/month)

**B. `_DIAGNOSTIC_RE` still missed contractions and gerunds**, reopening the exact
DEFECT-033 bug (a diagnostic "why"-style question getting hijacked into FORECAST by a
bare "this quarter"/"this month"/"this year") under different wording:
- `"What's behind the drop this quarter?"` (contraction of "what is behind")
- `"What's causing the decline this quarter?"` (gerund/inflection of "what caused")
- `"What is driving the decline this year?"` (the gerund gap DEFECT-033's own test suite
  had already documented as a known, deliberately-unfixed boundary)

## The fix

1. `packages/analytics_core/src/engines/analyst_answer.py`:
   - Added `_YTD_RE` (calendar-year-to-date: Jan 1 of the latest date's year through the
     latest date, explicitly flagged in the resolved-period note as a partial year, not
     the full calendar year -- distinct from the existing "this year" phrase).
   - Added `_SINCE_START_OF_RE` (maps "since the start/beginning of the quarter/month/year"
     onto the same calendar-start logic the existing "this <grain>" phrases already use).
   - Added `_BARE_RECENCY_RE` for phrases that imply a count of 1 with no explicit number
     ("fortnight", "a year ago") and extended `_QUANTIFIED_RECENCY_RE`'s unit set to
     `quarter`/`fortnight`/`year` and its word-number table to "eleven"/"twelve", all
     reusing the existing rolling-window code path (still the "week" grain bucket, per
     DEFECT-035's `KeyError`-avoidance rationale -- no new grain value introduced).
   - Each new note explicitly states its rolling-window approximation (a "quarter" here
     is 91 days, not a calendar quarter; a "year" here is 365 rolling days, not a
     calendar year) so this is legible as an approximation downstream, not silently
     conflated with the exact calendar arithmetic used elsewhere in the function.
2. `packages/analytics_core/src/engines/intent.py`:
   - Widened `_DIAGNOSTIC_RE`'s driver-word alternative from `driver\w*` to
     `driv(?:er\w*|ing)` (covers "driving").
   - Added a bare `caus\w*` alternative (covers "cause"/"causes"/"caused"/"causing"),
     which composes safely with the existing branch order: `causal` (a different,
     narrower regex for genuinely causal-inference questions) is still checked *before*
     `diagnostic` in `parse_intent`, so this doesn't change causal-question routing.
   - Added a contraction-tolerant `what(?:\s+is|'s)\s+behind` in place of the literal
     `what is behind`.

## Verification

- New regression suite `tests/independent_release/test_defect036_recency_vocabulary_gaps.py`
  (16 tests): all 9 previously-`None`/misrouted phrasings now resolve/classify correctly;
  genuine FORECAST ("What is our revenue outlook this quarter?") and PREDICTION ("What is
  the probability revenue will grow this quarter?") questions are confirmed unaffected.
- Updated `test_defect033_forecast_diagnostic_precedence.py`'s
  `test_documented_remaining_gap_driving_is_not_recognised_as_diagnostic` -- that test
  existed specifically to pin the "driving" gap as a known bug so a future change would
  be deliberate; this session is that deliberate change, so it's now split into three
  passing assertions (`driving`, `what's behind`, `what's causing` all route to
  ROOT_CAUSE) instead of asserting the old buggy FORECAST outcome.
- Re-ran all prior recency/diagnostic-precedence suites in the same run
  (DEFECT-031, 033, 034, 035, 036): 49/49 pass, 0 regressions.
- Ran the broader consumers of these two modules also present in this environment
  (`test_q6_1_intent_correlation_inflections.py`, `test_session12_human_analyst_depth.py`
  excepting its two DB-backed tests, which fail on missing `alembic` in this sandbox --
  an environment gap, not a logic failure): all passing tests remain passing.
- Did **not** attempt to run the full suite end-to-end in this environment (no DB, no
  `alembic upgrade head`) -- the DB-independent portions covering `intent.py` and
  `analyst_answer.py` show zero regressions, but a full clean-room `tests/` +
  `tests/independent_release/` + `packages/analytics_core/tests` run against a real
  Postgres instance is still owed before this is claimed release-verified the way prior
  sessions' full-suite counts were.

## Still open (unchanged by this session, documented so it isn't lost)

- No DEFECT-032 ledger entry exists despite being referenced by name in this codebase's
  own comments and the input zip's filename.
- The architectural gap named in DEFECT-025 and reiterated in the audit that led to this
  session: recency/causal-marker vocabulary is a hand-enumerated regex table with no
  shared, tested lexicon across `intent.py` and `analyst_answer.py`. This session (like
  032/033/034/035 before it) patches the specific phrasings a stress-probe happened to
  try; a genuinely unbounded space of paraphrases (typos, other contractions, regional
  phrasing, "the last little while", "since Q1 kicked off", etc.) remains unenumerated.
  A shared vocabulary/synonym module is still the durable fix, not another round of
  point patches.
- The ~31-test scientific-calibration backlog (DEFECT-005/007/015/019 family: churn/
  segmentation/correlation resolving INCONCLUSIVE despite genuine injected signal)
  remains open and untouched -- unrelated to NL parsing, and the reason real datasets
  with genuine effects can still fail to analyze correctly even once the question itself
  parses without issue.
