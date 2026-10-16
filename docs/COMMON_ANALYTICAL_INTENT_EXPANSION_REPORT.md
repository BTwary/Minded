# Common analytical intent expansion — implementation report

**Date:** 2026-10-14  
**Artifact type:** source candidate; not a release-certified installer.

## Implemented

- Added typed aggregation primitives `MODE`, `RANGE`, `Q1`, `Q3`, `IQR`, and `MISSING_COUNT` to the existing deterministic analytical operation registry.
- Added schema-bound natural-language patterns for mode/most-common, numeric range, quartiles, interquartile range, and missing-value count. Common variants include “What is the most common payment method?”, “Which payment method occurs most often?”, “What is the 25th percentile of age?”, and “How many age values are missing?”.
- New intent patterns return a complete canonical contract only when the target resolves to a real column; otherwise they produce an incomplete contract with `target_column` unresolved rather than falling through to an unrelated calculation.
- Routed the new primitives through the existing approved-contract executor and fixed categorical `MODE` headline formatting so text categories do not enter numeric formatting.
- Added operation semantics and limits documentation, regression tests, and a release-manifest-required check.

## Verification performed in this environment

- `39 passed` across the new intent expansion, existing typed-operation registry/approval-graph, and intent-state contract suites.
- `python -m compileall -q packages apps scripts tests` passed.
- Release manifest TOML parsed successfully with 34 unique required checks.
- A broader combined run of the M1 plan-contract suite produced 46 passes and 2 setup failures because `python-jose` (`jose`) is not installed in this environment. These are not counted as passes.

## Scope boundaries

This expansion improves a set of frequent descriptive/statistical questions. It does not claim universal natural-language understanding. Percent change and period-over-period growth, general percentage/share denominators, arbitrary compound questions, cross-table joins, and broad API-to-approval-to-execution behavior remain outside this change unless separately tested. The full release manifest, pinned Windows environment, frontend production build, installer, and clean-machine offline behavior have not been certified here.
