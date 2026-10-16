# BUGFIX 2026-09-26 (session 17): recency-phrase period-parsing gap (DEFECT-031)

## What was asked

"Check why the natural-language question parsing and analysis doesn't work
in real, fix the bugs." (Same prompt as session 16, run again against the
session-16 output zip, which had already fixed DEFECT-030.)

## What I did

Rather than trusting the session-16 changelog at face value, I independently
reproduced its claims first (fresh venv from `requirements.txt`, fresh
`alembic upgrade head` from zero, `alembic check` clean, DEFECT-030's own
regression test passing, full 5-question real-run repro from that changelog
re-run with warnings promoted to hard errors -- clean). DEFECT-030 holds.

Then, per the actual ask, I drove a *new* real ingestion + real
natural-language question set through the exact production path
(`DatasetService.ingest_dataset_file()` -> `AnalysisService.execute_analysis()`,
what `POST /analysis` calls) against a CSV with a genuine injected signal:
West region revenue collapsing in the most recent ~11 days of a 90-day
range, mixed into otherwise-normal noise.

## The bug (DEFECT-031)

Question: "Why did revenue fall in the West region recently?"

Real-run result: `direct_answer` was **byte-identical** to the answer for an
unrelated question asked in the same run, "What is the total revenue by
region?" -- no mention of West, "why," "fall," or "recently" anywhere.

Root cause, `packages/analytics_core/src/engines/analyst_answer.py`:

- `parse_period(question, dates)` only recognises a fixed vocabulary of
  explicit relative-period phrases (`"last month"`, `"this quarter"`,
  `"last week"`, ...) plus explicit quarter/month/year tokens. Anything else
  -- including extremely common vague recency phrasing like "recently",
  "lately", "of late" -- returns `None`.
- `classify_question(...)` requires `period is not None` to route to
  `"PERIOD_CHANGE"`, even when the question text obviously asks for a
  temporal root-cause answer (it matches both `_PERIOD_CHANGE_RE`, e.g.
  "why", and `_CHANGE_WORDS_RE`, e.g. "fall"). With `period is None`, the
  question falls through every other branch (no TREND word; no
  ASSOCIATION columns were passed in for this call shape; no RANK/
  BREAKDOWN/COMPARE keyword matches "why did revenue fall in the West
  region recently") and hits the final, unconditional fallback:
  `if has_group: return "RANKING"`.
- Result: a root-cause "why did X change" question silently downgrades
  into a plain top-line "who's highest" ranking answer that names no
  segment, no "why," and no change -- and is indistinguishable from the
  answer to a completely different, non-causal breakdown question.

This is the same failure shape the user's prompt described ("doesn't work
in real") but a distinct bug from DEFECT-030: not a crash, not a warning --
a silently wrong/irrelevant answer to one of the most natural ways to ask
a root-cause question.

## The fix

Extended the existing relative-period phrase table in `parse_period` with:
`"recently"`, `"lately"`, `"of late"`, `"in recent days"`,
`"in recent weeks"`, `"this past week"`, `"the last few days"` -- all
mapped onto the *already-correct* "last week" / "this week" resolution
logic (anchored to the data's own observed date range, never wall-clock
time). No new `grain` value was introduced (a downstream `dict` keyed by
exactly `{"month","quarter","year","week"}` would `KeyError` on anything
else), so this reuses an existing, already-tested code path rather than
adding a new one.

Deliberately not fixed this session (documented, not silently dropped): the
adjective form ("the recent decline", "recent revenue decline") is not
recognised, nor is unadorned "recent" -- both are more ambiguous to match
safely (risk of false-positiving on unrelated uses of "recent") and are
left as a documented remaining gap, covered by an explicit
"not-yet-fixed" regression test so a future change to this boundary is
deliberate, not a silent drift.

## Verification

- New regression test
  `tests/independent_release/test_defect031_recency_phrase_period_parsing.py`
  (6 tests): confirms `parse_period` resolves the new phrases, confirms
  `classify_question` now routes to `PERIOD_CHANGE` not `RANKING`,
  confirms the real answer is no longer the unrelated ranking fallback,
  confirms "recently" now behaves identically to the already-supported
  explicit "last week" phrasing, and pins down the documented remaining
  gap (adjective form) as a `None` result rather than silently changing
  behavior later without a test noticing.
- Re-ran the real production-path repro (real ingestion, real
  `execute_analysis`, 5 question shapes): the "why...recently" question now
  returns a genuine before/after temporal comparison with mechanistic
  decomposition (correctly isolates the injected Grocery/West-adjacent
  driver), no crash, no Pydantic warning (DEFECT-030 fix still holds under
  the same warnings-as-errors repro).
- Full suite (`tests/` + `packages/analytics_core/tests`, excluding
  `test_hf_space_app.py` which needs `gradio`, not in core
  `requirements.txt`): **before fix, 31 failed / 1281 passed**; **after fix
  (incl. the 6 new DEFECT-031 tests), 31 failed / 1287 passed** -- identical
  31 pre-existing failures both times, 0 regressions. `python -m compileall`
  clean.
- Diffed the delivered zip against a completely fresh independent unzip of
  the uploaded input archive: only `analyst_answer.py` and the new test
  file differ.

Release status remains **NOT RELEASE READY** -- the same pre-existing
31-test backlog (churn-confounding calibration, DEFECT-015/007/019-adjacent
gaps) from prior sessions is untouched by this session, as scoped.
