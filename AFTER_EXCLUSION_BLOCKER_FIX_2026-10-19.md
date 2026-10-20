# MindEd fix: false temporal blocker after lifecycle exclusions

Date: 2026-10-19

## User-reported defect

Question: `After excluding returns and cancellations, which region generated the most realized revenue?`

The review page stopped before execution with `time_range: requested time scope not preserved`, even though the question contains no date range. The first word `After` was being treated as a temporal preposition by a broad regex in the material-intent check.

The parser also mapped the plural noun `returns` to only the observed `Returned` status and did not preserve `cancellations` as a second exclusion.

## Changes

1. Temporal scope is now inferred from `after`, `before`, or `since` only when followed by a recognizable date/period expression. The opening phrase `After excluding ...` is no longer classified as a time window.
2. A coordinated lifecycle exclusion such as `excluding returns and cancellations` is resolved only when those terms map unambiguously to observed values in one schema-grounded status/state column. The result is a composite `NOT IN` predicate. It does not invent a filter if the schema/value mapping is ambiguous.
3. Added regression tests proving that the exact question is executable, does not create a time-range requirement, excludes both status categories, and does not force remaining valid status values such as `Shipped` to `Delivered`.

## Verification performed

- `py_compile` passed for the changed Python modules.
- Targeted regression suite: **27 passed, 1 skipped**.
- The exact question against a small fixture produces a `CLEAR` preview and `order_status NOT IN ('Returned', 'Cancelled')`.
- Independently aggregated the supplied Northstar CSV (9,000 rows, no nulls in `region`, `realized_revenue`, or `order_status`). Status counts: Delivered 7,795; Returned 904; Cancelled 301. Excluding Returned and Cancelled leaves 7,795 rows in this dataset.

| Rank | Region | Sum of `realized_revenue` | Rows | Share |
|---:|---|---:|---:|---:|
| 1 | East | 355,091.84 | 2,012 | 25.8% |
| 2 | South | 343,625.00 | 2,000 | 25.0% |
| 3 | North | 339,704.02 | 1,885 | 24.7% |
| 4 | West | 337,192.80 | 1,898 | 24.5% |
| | **Total** | **1,375,613.66** | **7,795** | **100.0%** |

East minus West is **17,899.04** (about **5.3% of West's total**).

In this exact uploaded dataset, the only remaining status after excluding Returned and Cancelled is Delivered, so the screenshot's extra `order_status = 'Delivered'` condition accidentally gives the same totals. It is still too restrictive as a general rule: it would drop statuses such as Shipped in a dataset that contains them. Before rerunning this question, remove the manually added `order_status = Delivered` condition with its trash icon; the question-derived `NOT IN` filter should be sufficient.

## Build caveat

A fresh Next.js build could not be rerun in this working environment because `next` is not installed in its local `node_modules` (`next: not found`). The Python regression tests passed. The Windows build script previously succeeded on the user's machine; rebuild this source package there and verify the live desktop flow. No claim is made that this updated executable has already been rebuilt or end-to-end verified.
