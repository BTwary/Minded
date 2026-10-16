# Common Analytical Intent Expansion — 2026-10-14

## Purpose

Expand the deterministic, schema-grounded question compiler for frequently asked analyst questions. This change is implemented in the canonical contract compiler and the approved-contract execution path; it is not a separate text-only answer shortcut.

## Added/expanded intent coverage

- Minimum / smallest / lowest value forms, including “What is the lowest age?”
- Maximum / largest / highest value forms, including “What is the highest age?”
- Distinct/unique-value counts, including “How many unique customer_id?” and “number of distinct customers”.
- Missing-value percentage, including “What is the percentage of missing values in age?”
- Whole-row duplicate counts, including “How many duplicate rows are there?”; duplicates mean complete rows identical across every column, counted after the first occurrence.
- Grouped missing-value percentages and all-missing columns, with percentage defined as `100 * missing cells / rows in the evaluated population`.

The new `MISSING_PERCENT` operation is in the typed aggregation registry and runs through `aggregate_frame`, including filters, grouping and verification. It counts nulls in the selected column, not empty strings or arbitrary sentinel text; string normalization into missing values remains a separate data-quality policy.

## Verification performed

- `43 passed` across:
  - `tests/independent_release/test_common_descriptive_intents_expansion.py`
  - `tests/independent_release/test_analytical_operation_registry.py`
  - `tests/independent_release/test_intent_state_contract.py`
- The broader analytical-contract authority test could not collect in this environment because `polars` is not installed. This is an environment limitation, not a passing result.
- Full release manifest, pinned Windows environment, API preview-to-approval-to-execution path, frontend build and installer were not certified in this pass.

## Explicit boundaries

- “Lowest/highest” patterns are accepted only when the phrase resolves to a unique schema column; ranking and comparative language must not be rewritten as a simple minimum/maximum.
- `COUNT_DISTINCT` ignores null values by definition.
- `MISSING_PERCENT` denominator is the rows remaining after approved filters; for grouped execution, each group's own row count is the denominator.
- `DUPLICATE_ROW_COUNT` compares every column and counts duplicate occurrences beyond the first; it does not infer a business key. Key-based duplicate checks require an explicit key-selection intent.
- This remains a bounded deterministic parser, not universal natural-language understanding. Share-of-total, growth rates, cohort retention, funnel conversion, arbitrary joins and complex multi-clause plans require their own typed semantics and end-to-end tests.

## Follow-on expansion: compound planning breadth (2026-10-14)

Expanded `complex_objective_planner.py` metric-list recognition to cover additional common business measures and aliases, including gross/net profit, gross margin, retention, conversion rate, returns/refunds, expenses, AOV, CLV/LTV, sessions/visits, support tickets, satisfaction, response time, and growth rate. Both metric-list positions remain supported: `revenue and profit by region` and `top segments by lifetime value and returns`.

Added a fail-closed limit for questions that decompose into more than 12 objectives. Instead of passing an unbounded plan downstream, the planner returns a dependency-blocked plan with a reason to split the question. This is a safety ceiling, not a claim that every 12-part question is semantically valid.

Added regression tests for expanded financial metric enumeration, post-grouping metric enumeration, and over-segmentation. Validation in the working source tree: 51 focused tests passed. This does not certify the full release manifest, API-to-execution path, or Windows installer.

## Follow-on expansion: wider metric enumeration

The compound objective planner now recognizes a wider decomposition vocabulary across
financial, marketing, customer lifecycle, operations, inventory, support, quality, and
workforce metrics. It can expand metric lists with four or more items, including lists
placed after a ranking/grouping phrase (for example, "rank products by revenue, profit,
margin and return rate"). This is a planning-level improvement only: the expanded
metric phrases are not proof that a dataset contains the requested measures. The normal
schema/alias resolution and execution verification gates must still ground each metric;
unknown measures must not be replaced by unrelated columns. The planner's 12-objective
safety cap remains in force.

## Follow-on expansion: cross-domain metric vocabulary (continued iteration)

Expanded the deterministic compound-planning lexicon with additional metric phrases spanning:

- Subscription economics: net/gross revenue retention, cohort retention, revenue per user/account, paid/trial/lead conversion, repeat-purchase rate and purchase frequency.
- Commerce and payments: cart abandonment, checkout completion, payment failures, basket value, fraud and chargebacks.
- Service operations: first-response time, first-contact resolution, SLA breaches, escalations, reopened cases and resolution rate.
- Supply chain and manufacturing: fulfillment, order accuracy, on-time-in-full, supplier defects, lead time, yield, scrap, rework and downtime.
- Finance and unit economics: contribution margin, EBITDA, working capital, cash-conversion cycle, DSO/DPO, payback period, break-even and return on investment.
- Data reliability: completeness, duplicate rate, invalid rate, outlier rate, freshness and latency.

The planner now avoids treating `per` as a grouping separator in metric-list suffixes because it can consume part of a legitimate metric such as `revenue per user`. Use an explicit grouping phrase such as `by month` or `by channel` for compound metric lists. This conservative behavior prevents a common decomposition ambiguity; it does not resolve every natural-language ambiguity.

Added five regression tests for subscription metrics, commerce funnels, operations/quality, data-quality metrics and finance metrics. Validation in the current environment: 60 focused tests passed across the planner, descriptive-intent expansion, operation registry and intent-state contract suites. Full API execution, Polars-dependent authority tests, the Windows environment, frontend and installer remain outside this validation result.
