# BUGFIX 2026-09-27 (session 19): fail-closed guard for unrecognized time-window phrasing (DEFECT-037)

## What was asked

"This script.txt has a research output related to my project, look into it and check if
something worth taking implement it to my project." The file is a ~5,300-line external
research document ("Final specification" / "Question Understanding Contract Specification
for an Autonomous Analytical System") proposing a full enterprise semantic-layer contract:
six representation layers (raw utterance -> parser hypothesis -> semantic-model binding ->
proof -> immutable contract -> execution plan), an authority hierarchy, an executability
state machine, provenance/proof obligations, and a "Minimum Rejection Policy" classifying
every ambiguity into HARD BLOCK / SOFT WARNING / EXPLICIT INTERPRETATION ACCEPTABLE / SAFE
DETERMINISTIC INFERENCE.

## Assessment of the document

Most of it does not fit this project at its current stage: it assumes a governed metric
catalog, multi-tenant security scopes, currency/jurisdiction governance, and a full
provenance/immutable-contract-hash infrastructure that AA-OS has no equivalent of and that
would be a multi-month rebuild for a solo project, not an incremental fix. Adopting the full
six-layer architecture wholesale is not recommended right now.

One principle in it is directly, concretely applicable and cheap to adopt: the **Minimum
Rejection Policy**'s core idea that a system must not silently compile a request under an
unresolved material condition (explicitly listed: "Fiscal/calendar period unresolved where
material" is a named HARD BLOCK case) -- and that when full blocking isn't appropriate, an
"explicit interpretation acceptable" tier requires disclosing the assumption in the answer
rather than hiding it.

This maps exactly onto the failure class this codebase has been re-discovering, one phrasing
at a time, across DEFECT-031/033/034/035/036: `parse_period` doesn't recognise a given time
phrase, `classify_question` silently falls through to an unrelated RANKING answer, and the
resulting answer never mentions the time window at all. Every prior fix in that line closed
one specific phrasing; the phrase space is unbounded, so the class kept recurring under new
wording. This session implements the general principle instead of another specific-phrase
patch.

## The fix (DEFECT-037)

`packages/analytics_core/src/engines/analyst_answer.py`:

- Added `_GENERIC_TEMPORAL_MARKER_RE`: a broad, phrasing-agnostic detector for "this question
  names *some* time window" (last/past/trailing/this/recent/lately + up to two filler words +
  a time-unit word; "year to date"/"YTD"; a bare "... ago"; "since the start/beginning/end
  of"). This is deliberately looser than `parse_period`'s specific-phrase table -- it doesn't
  need to know the exact phrase, only that a time reference is present.
- `classify_question` now returns a new kind, `PERIOD_CHANGE_UNRESOLVED_TIMEFRAME`, when the
  question reads as a period-change question (a change/why word matched) and the generic
  marker fires, but `parse_period` returned `None` for the specific phrase. A question with a
  change word but *no* time reference at all is unaffected (still `RANKING`, as before) --
  this only catches the case where a time window was clearly named but not understood.
- `build_analyst_result` handles the new kind by falling back to the honest full-range
  answer (reusing `_ranking`, picking a fallback dimension from `candidate_dimensions` or the
  dataframe when no `group` was resolved) and prepending a caveat that states plainly the
  named time window could not be parsed and the figures cover the full available range
  instead -- an explicitly disclosed, reversible assumption (the reviewed spec's tier 11.3)
  rather than a silent misclassification (this module's own design rule 1: "if a required
  role is missing it returns `None` (say nothing rather than guess)" is extended here to "or
  say something honest, with the gap disclosed").
- This is additive to, not a replacement for, `parse_period`'s specific-phrase table: every
  phrasing DEFECT-031/033/034/035/036 already resolve continues to return a real `Period` and
  classify as plain `PERIOD_CHANGE`, unaffected by the new fallback path.

## Verification

- New suite `tests/independent_release/test_defect037_unresolved_timeframe_failclosed.py`
  (5 tests): a genuinely novel phrasing not in any existing table ("the last handful of
  quarters") and a bare "... ago" phrase now flag correctly instead of silently misrouting;
  the real `build_analyst_result` output discloses the gap in both headline and caveats; a
  question with no time reference at all is confirmed unaffected (still `RANKING`); every
  phrasing from DEFECT-031/033/034/035/036 is re-confirmed to still resolve as a real,
  correctly-scoped `Period`/`PERIOD_CHANGE` (the new generic marker doesn't shadow them).
- Ran all six recency/diagnostic regression suites together (031, 033, 034, 035, 036, 037):
  54/54 pass, 0 regressions.
- Re-ran the broader consumers of this module present in this sandbox
  (`test_q6_1_intent_correlation_inflections.py`, `test_session12_human_analyst_depth.py`):
  all DB-independent tests still pass (one pre-existing `alembic`-missing import error,
  unrelated, same as the prior session).
- Did not attempt a full clean-room suite run (no Postgres/alembic available here) --
  same caveat as the DEFECT-036 session.

## Still open

- This is a phrasing-agnostic *backstop*, not a semantic understanding layer. It can tell
  "a time window was named" from surface shape, but can't resolve genuinely novel windows
  ("last three sprints", "the previous product cycle") into an actual date range -- those
  still correctly fall back to the disclosed full-range answer rather than guessing, which
  is the intended fail-closed behavior, not a further gap to chase.
- The rest of the reviewed spec (six-layer contract, provenance, authority hierarchy, full
  Minimum Rejection Policy across metrics/joins/grain/causal claims) is not implemented. If
  this project reaches a stage with a real governed-metric catalog and multi-source joins,
  those sections are worth revisiting -- they're well-reasoned for that scale, just premature
  for the current single-layer regex-based NL parser.
- The ~31-test scientific-calibration backlog (DEFECT-005/007/015/019 family) remains open
  and untouched, as in every prior NL-focused session.
