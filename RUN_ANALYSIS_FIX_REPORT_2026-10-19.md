# MindEd AA-OS: Run Analysis clarification-loop fix

## Defect reproduced from the screenshots

Question: `Calculate total realized_revenue by region, ranked from highest to lowest.`

The planner treated `highest` and `lowest` as two competing ranking instructions, blocked the canonical contract, and showed misleading clarification choices. Once a user selected a choice, the first click on **Run Analysis** only regenerated the preview. The refreshed plan caused the review component to remount, but the selected clarification and related state were not retained, so the next click could lose the choice and repeat the preview step. Stale `sessionStorage` values could also override the current contract's target/group defaults.

## Changes

1. Treat common paired rank phrases such as `ranked from highest to lowest` as one sort direction. Truly conflicting requests such as `top 5 and bottom 5` remain blocked.
2. Exclude the `from` inside a rank-direction phrase from the filter-intent scan. Genuine scope expressions such as `from 2024` remain untouched.
3. Preserve clarification choice/patch, scope, dataset and custom guidance in `guidance_context` when re-previewing; restore it if the review panel remounts.
4. Make the button accurately describe the state transition: **Apply Decisions & Refresh Preview** before the updated plan has been reviewed, then **Confirm & Run Analysis** once the same guidance has been previewed. A still-unresolved plan is blocked from execution until changed.
5. Prefer current canonical/parsed fields over stale saved field choices so `realized_revenue` and `region` are not silently replaced by an old metric/group.
6. Add regression tests for the exact wording, direction, aggregate, grouping and preview-to-approval UI contract.

## Verification performed in this environment

- 21 tests passed across the ranking-direction regression and intent-state suites.
- 27 tests passed across the common-descriptive-intent and plan/intent-merge regression suites.
- The exact question was evaluated against a synthetic fixture and resolved to `SUM(realized_revenue)` grouped by `region`, descending; the preview was `CLEAR` and the human-guidance resolver returned `RESOLVED`.
- Python compilation passed for the touched backend modules.
- TypeScript/TSX parse-transpilation passed for `ProposedAnalysisPanel.tsx` and `analyst/page.tsx`.

## Not claimed

A full Next.js production build and live desktop/browser end-to-end test were **not** run in this container. Dependency installation stalled, and several broader tests could not be collected here because DuckDB is not installed in this runtime. Rebuild on the Windows machine with the project's existing offline desktop build script, then verify the exact question using the user's dataset.
