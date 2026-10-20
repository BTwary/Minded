# MindEd review, filter, ranking, and report corrections

Date: 2026-10-20

## What the uploaded live screenshots and reports showed

1. The question review panel could restore an old metric/group from browser session storage while the current question was unresolved. This made unrelated fields such as `customer_tenure_months` and `order_id` look like parser-selected intent.
2. Quoted custom instructions such as `'Exclude returns and cancellations'` were rejected by the freeform guidance parser even when the canonical question contract already contained the same lifecycle exclusions.
3. “High support load” was not always bound to `support_load_band = 'High'` because `high` also occurred in a separate column name (`high_support_flag`).
4. Guided review could change a ranking request into a descriptive task even while retaining its ranking direction.
5. Reports for a question asking for the *lowest* group could state that the highest group was the answer. Ranking tables also showed shares for non-additive averages, and positive-only bar charts had negative y-axis labels.

## Changes

- Only restore saved field selections when the current canonical preview is `CLEAR`; resolved canonical values and explicit current guidance continue to take precedence.
- Strip a pair of outer straight, curly, or backtick quotes from custom instruction text before parsing.
- Treat lifecycle exclusion text as a no-op when the canonical contract already contains equivalent typed exclusion predicates; do not resolve the same exclusion a second time against unrelated columns.
- Preserve `RANKING` task family / `ranking` estimand when a user adjusts review parameters, and leave required grouping unresolved if it is missing.
- Bind explicit ordinal phrases such as “high support load” to a categorical `*_band` column, even when a token overlaps with a separate column name.
- Render lowest/least/smallest/worst ranking results in ascending order, write direction-aware narrative claims, use “Average Value” for mean rankings, and suppress misleading shares for non-additive average values.
- Keep positive-only bar charts anchored at zero and remove numeric axes from donut charts.
- Add regression tests for the parser, review state, ranking outputs, narrative direction, and chart scale.
- Retain the prior Windows startup fix: the exposed pywebview API object no longer keeps a native window reference in its inspected object graph, and boot logs separate API import latency.

## Verification performed

- Python compilation of the modified analytics, governance, export, and narrative modules passed.
- Focused plus related regression suites: **83 passed**.
- Manual validation on `northstar_commerce_analytics_test(2).csv` (9,000 rows): the question “Among orders with high support load that were delivered and not returned, which region had the lowest average satisfaction score?” resolves to `MEAN(satisfaction_score)` grouped by `region`, ascending, with these three filters:
  - `order_status = 'Delivered'`
  - `support_load_band = 'High'`
  - `returned = 0`
- Those filters select **623 rows** and reproduce the saved group means to four decimals: South 3.7101, North 3.7681, West 3.7853, East 3.8105.
- A quoted redundant exclusion instruction with a pending intent slot resolves without dropping the ranking contract or duplicating filters.

## Important scope note

The Python regression checks and dataset validation were run in the Linux working environment. A fresh Windows desktop bundle build and end-to-end GUI smoke test were **not** run here. After applying this source update, rebuild on Windows using the existing `.venv` and test startup, the quoted-exclusion review flow, and a lowest-average question before treating the GUI issue as fully resolved.
