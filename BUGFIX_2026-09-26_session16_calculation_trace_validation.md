# BUGFIX 2026-09-26 (session 16): `calculation_trace` Pydantic validation bypass (DEFECT-030)

## What was asked

"Check why the natural-language question parsing and analysis doesn't work
in real, fix the bugs."

## What I did

Rather than trusting the existing test suite (which was already
1,279/1,310 green), I drove the actual production code path end-to-end
against **real, ingested data**: real CSV upload through
`DatasetService.ingest_dataset_file()`, then real natural-language questions
through `AnalysisService.execute_analysis()` — the exact path
`POST /analysis` uses — for five different question shapes (root-cause,
bivariate correlation, ranking/breakdown, rate-by-segment, undirected
"why").

No exceptions, no crashes. But every question that produced evidence threw
a `PydanticSerializationUnexpectedValue` warning during the investigation
run — invisible to the existing test suite because it only surfaces inside
a full `model_dump()` of the whole runtime-state snapshot, which most tests
never call.

## The bug (DEFECT-030)

`packages/analytics_core/src/intelligence/transition.py`,
`apply_post_execution_transition`:

```python
raw_obs.calculation_trace = trace_dict   # BEFORE
```

`raw_obs` is a `RawObservationRecord` (Pydantic `BaseModel`) whose
`calculation_trace` field is typed `Optional[CalculationTraceSchema]`.
Direct attribute assignment of a plain `dict` bypasses Pydantic validation
entirely (this model doesn't set `validate_assignment=True`), so the field
silently held an unvalidated raw `dict`. `raw_obs` had already been stored
by reference into the investigation's runtime state
(`state_mgr.record_raw_observation`) earlier in the same function, so the
mistyped field stayed live for the rest of the investigation and got
serialized on every `build_scientific_state_snapshot()` call — corrupting
the very audit/reproducibility trace this schema exists to guarantee.

## The fix

```python
raw_obs.calculation_trace = CalculationTraceSchema.model_validate(trace_dict)  # AFTER
```

Matches the pattern already correctly used two lines below
(`state_mgr.record_calculation_trace(trace_dict)`) and in
`evidence_ledger.py`'s `EvidenceRecord` construction.

Audited the rest of the codebase for the same defect class (raw dict
assigned post-construction to a Pydantic field typed as a nested schema):
`TransitionResult.calculation_trace` is a plain dataclass field (correctly
typed `Optional[Dict[str, Any]]`, no bypass possible) and
`FirstClassExperiment.execution_result`/`verification_result` are correctly
typed `Dict[str, Any]` to begin with. No other instance found.

## Verification

- New regression test `tests/independent_release/test_defect030_calculation_trace_validation.py`
  (2 tests), run end-to-end through real ingestion + `execute_analysis`
  (not mocked). Confirmed fail-before / pass-after the fix.
- Re-ran the full 5-question real-run repro with the warning promoted to a
  hard error (`warnings.filterwarnings("error", ...)`): clean on all 5
  questions post-fix.
- Full suite (`tests/` + `packages/analytics_core/tests`), before and after
  the fix: identical 31 pre-existing failures both times (pre-dating this
  session, in the already-documented churn-confounding / DEFECT-015 /
  DEFECT-007 / DEFECT-019 backlog), 1281 passed after (1279 before, +2 new),
  0 regressions. `python -m compileall` clean.

## Not bugs (checked and ruled out)

Two other results from the real-run repro looked surprising at first glance
but turned out to match already-documented, intentional design decisions,
not new bugs:

- **"Why are customers churning?"** (no dimension named, multiple plausible
  candidate columns) → `"Unable to find a testable analysis for this
  question against the supplied data."` This is the DEFECT-019 abstention
  design: `experiment_synthesizer.py` deliberately declines to guess a
  group-by dimension among several equally-plausible candidates rather than
  invent one.
- **"What is the churn rate by plan_type?"** → an Inconclusive/Simpson's-
  paradox-style answer even though real experiments ran and a genuine
  confounding check was performed (per the DEFECT-019 fix, this message
  only fires when `experiments_summary` is non-empty). Churn-rate-by-segment
  questions are exactly the kind of comparison this system is built to
  stress-test for confounding before asserting a naive segment ranking, so
  this is intended behavior, not a false claim.

Release status remains **NOT RELEASE READY** — the pre-existing 31-test
backlog (churn-confounding calibration, DEFECT-015/007/019-adjacent gaps)
is untouched by this session, as scoped.
