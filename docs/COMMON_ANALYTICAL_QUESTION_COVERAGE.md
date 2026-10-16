# Common analytical question coverage — expansion candidate

## Newly implemented and focused-tested

The following forms compile to a schema-bound `CanonicalQuestionContract` and use the approval-gated descriptive executor. The target must resolve to a real dataset column; unknown/ambiguous targets remain incomplete instead of selecting a plausible substitute.

| Analyst intent | Example questions | Canonical operation | Semantics |
|---|---|---|---|
| Most frequent category | “What is the mode of payment method?”, “What is the most common payment method?”, “Which payment method occurs most often?” | `MODE` | Ignores null values; if multiple values tie, selects a deterministic value by type/string ordering. |
| Numeric spread | “What is the range of age?” | `RANGE` | Maximum minus minimum over non-null numeric values. |
| Lower quartile | “Calculate the first quartile of age”, “What is the 25th percentile of age?” | `Q1` | Linear-interpolated 25th percentile. |
| Upper quartile | “Calculate the third quartile of age”, “What is the 75th percentile of age?” | `Q3` | Linear-interpolated 75th percentile. |
| Interquartile spread | “Calculate the interquartile range of age” | `IQR` | Q3 minus Q1 using linear interpolation. |
| Missingness count | “Count missing values in age”, “How many age values are missing?” | `MISSING_COUNT` | Counts null/NA values in the named column; it does not count empty strings unless the loader has normalized them to null. |

Existing typed operations remain available for row count, count, distinct count, sum, mean, median, min/max, variance, standard deviation, and explicit percentile parameters. Multi-dimensional grouping and the approved aggregate-then-rank operation graph remain separate existing capabilities.

## Safety and limits

- New natural-language forms are intentionally bounded patterns, not a claim of universal English understanding.
- Column binding must be supported by the loaded schema. If the metric cannot be resolved, the question is incomplete/refused rather than guessed.
- `MODE` tie handling is deterministic but is a product policy; users needing all tied modes should be supported by a separate explicit output contract, not by silently changing this one.
- Missingness currently means actual null/NA according to the loaded DataFrame. Blank strings, whitespace-only strings, sentinel values such as `N/A`, and domain-specific missing codes need explicit normalization policy.
- Percent change, year-over-year/month-over-month growth, share-of-total with filters, joins across multiple tables, and broad arbitrary compound questions are not certified by this expansion merely because some related primitives exist elsewhere in the repository.
- Focused tests are not a release certification. The pinned Windows environment, API approval-to-execution lifecycle, full release manifest, and clean installer still need verification.

## 2026-10-14 follow-up expansion

New schema-grounded, approved-contract cases now cover minimum/maximum wording variants, unique/distinct counts, and missing-value percentage. See `COMMON_INTENT_EXPANSION_2026-10-14.md` for exact definitions and verification limits. The current focused regression count is 43 passing tests; this is not the full release manifest.

## Compound planning expansion (2026-10-14 follow-on)

The compound planner now recognizes a broader vocabulary of business measures when splitting enumerated metric requests into separate objectives. It also blocks plans exceeding 12 detected objectives rather than allowing unbounded downstream work. This expands deterministic decomposition only; it does not establish semantic understanding of arbitrary metric synonyms, correct joins, time-series semantics, or causal attribution. Those require schema-grounded resolution and execution-level benchmark tests.
