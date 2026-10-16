# Analyst understanding and minimal clarification UX — 2026-10-14

## Principle

Do not turn every question into a form. Resolve what is unambiguous from the user's wording, schema, data types, and validated semantic model. Ask one compact clarification only when a material choice remains unresolved. Never silently pick a metric, denominator, time grain, join, population, or causal interpretation.

## Minimal interaction contract

1. **No clarification needed:** show a compact interpretation summary (metric, population/filter, grouping, period, method) with `Run analysis` and a secondary `Edit interpretation` action. Do not require checkboxes.
2. **One material ambiguity:** ask one direct question and present at most 2–4 mutually exclusive choices as radio options, plus `Other / specify`. Use checkboxes only when multiple selections are genuinely valid (e.g. select several output metrics).
3. **Compound question:** show a concise, editable checklist of the user's requested deliverables, preselected from their wording. Keep dependencies and unresolved items visible; do not let unchecked clauses silently disappear. Require explicit confirmation before removing requested deliverables.
4. **Optional preferences:** hide advanced options behind `More options`; defaults must be displayed and reversible. Never make users choose routine calculation details they did not ask to control.
5. **Unsafe or unsupported plan:** explain the exact blocker and offer the nearest supported alternative; do not display a misleading Run button.

## Analyst reasoning record

Every plan should retain: original question; parsed objectives; metric/estimand and formula; population and filters; dimensions/grain; period boundaries; join path and cardinality checks; assumptions and their provenance; candidate methods and why the selected method fits; dependency graph; expected output schema; validation checks; known limitations; and a clear proceed/ask/refuse state.

## Broader aggregation vocabulary implemented in this increment

Added deterministic primitives: absolute sum, positive/negative/zero/non-zero counts, median absolute deviation, skewness, kurtosis, and coefficient of variation. These are typed operations, not automatic natural-language recognition guarantees. Skewness/kurtosis return null when sample sizes are insufficient; coefficient of variation returns null when the mean is zero or fewer than two non-null observations exist. CV is a ratio (not a percentage); callers must label units explicitly.

## UI test obligations

- Routine unambiguous question: no forced checkbox.
- Ambiguous denominator or date field: one targeted clarification, no silent default.
- Multi-output request: preserve every clause and allow explicit selection/editing.
- User edits interpretation: contract hash and plan must change; stale results must not be reused.
- Cancel/back: no partial execution or lost original question.
- Unresolved required slot: execution blocked with a readable reason.
