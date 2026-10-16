# MindEd AA-OS merged fix audit — 2026-10-14

## Scope

This is a merged source candidate based on `MindEd_AAOS_Common_Intent_Expansion_Candidate (1).zip`, with the derived-rate changes retained from the prior fixed candidate and selected plan/investigation safety components ported from `MindEd_AAOS_Plan_Investigation_Enhanced_v3.zip`.

## Fixes applied

1. **Joint relationship preservation:** the compound planner now recognizes explicit relationship/correlation/association phrasing before metric-list expansion. Questions such as `Check the relationship between price and profit by region` and `Analyze correlation between revenue and profit by month` remain one joint objective rather than being split into independent metric summaries.
2. **Missing percentage operation routing:** missingness-percentage questions are classified before generic rate/proportion wording, and a direct pattern supports forms such as `What percentage of age values are missing?`.
3. **Natural-language operation mappings:** added deterministic patterns for skewness, kurtosis, median absolute deviation, coefficient of variation, absolute sum, positive count and negative count. Resolution remains schema-grounded; unresolved targets are not guessed.
4. **Derived binary rate execution compatibility:** retained unique binary-indicator resolution but tightened it to native boolean or numeric 0/1 columns, matching the current mean-based executor. Categorical labels such as `defective/non-defective` are rejected until an explicit typed transform exists.
5. **Plan dependency metadata:** compound objectives record exact prerequisite ordinals and serialize a small audit representation. Invalid prerequisite ordinals are rejected before child execution.
6. **Safer follow-up grounding:** ported v3's conservative follow-up helper. A dependent entity reference is blocked if a verified prior entity label cannot be extracted or safely substituted; ungrounded text is not executed as if it were a resolved cohort filter.
7. **Investigation plan audit:** integrated the structural plan audit into the universal question compiler. Its output always keeps `execution_authorized=false`; runtime authorization remains separate.
8. **Regression coverage and packaging hygiene:** added tests for relation preservation, operation routing, numeric/bool rate resolution and follow-up grounding. Generated `.pyc` and `__pycache__` entries are excluded from the ZIP.

## Verification performed

- `python -m compileall -q packages apps tests scripts`: PASS.
- Focused regression set: **84 passed** across merged-intent, compound-planner, descriptive-intent, derived-rate, intent-state, operation-registry and descriptive-execution tests.
- Four selected v3 integration tests for compound-plan ambiguity and plan-validation serialization/grounding: PASS.
- ZIP integrity check: PASS.

## Known limitations

- The full release manifest was not run end-to-end. `duckdb` is unavailable in the current audit environment, so database-backed integration and full clean-machine Windows packaging are not certified.
- The relationship guard preserves explicit pairwise relationship wording. It does not itself guarantee that every predictor/outcome/grouping column is correctly resolved; the canonical semantic contract and downstream validation must still confirm those roles.
- Categorical binary labels are deliberately rejected for derived rates rather than silently converted. A future implementation can support them only by representing and testing an explicit label-to-indicator transform in the approved contract.
- Structural plan validity does not mean a method is authorized to execute, nor does it prove that the requested business estimand is correct.

## Release status

**Improved merged candidate — not release certified.** Do not remove the remaining release gate or mark 1.0 ready based on focused tests alone.
