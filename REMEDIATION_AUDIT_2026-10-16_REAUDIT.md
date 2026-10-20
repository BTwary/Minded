# MindEd AA-OS re-audit and remediation report (2026-10-16)

## Scope and baseline

Audited the uploaded `MindEd_AAOS_Filter_Query_Output_Polished_2026-10-16.zip` directly, then compared the prior `..._REMediated.zip`. The prior build changed three source/test files and added an audit report, but it did not implement a general per-token intent ledger or prove full post-execution equivalence. The archive does contain a pre-execution consistency gate and a post-execution metric/group reconciliation path; those are not equivalent to checking every executed filter and denominator in every execution path.

The reported 152-case adjudication set was not present as a runnable fixture in either archive, so no claim is made about a rerun or score improvement on those 152 cases.

## Changes in this re-audited build

1. **Numeric thresholds:** bind each explicit numeric comparison attached to a named numeric schema column, rather than just the first one. A threshold count is compared with typed numeric predicates; missing predicates leave the contract incomplete.
2. **Exclusion values:** inflection recovery (e.g. `refunds` -> `refunded`) now requires one unique best observed categorical value across the dataset. It will not select a dimension just because its fuzzy score is slightly higher than another candidate. Ambiguous or unmatched exclusions are unresolved and block enforced execution.
3. **Filter intent ledger:** populate `intent_ledger` with consumed/leftover entries for detected numeric threshold, categorical exclusion, and filter-intent clauses. This is deliberately **not** described as a complete all-content-word ledger; unsupported lexical content and non-filter clause coverage still need a broader implementation.
4. **Confidence firewall:** retain the planner guard that treats an enforced contract with confidence below 0.70 as refused, so proposal semantics cannot fill missing roles.
5. **Execution consistency:** preserve the archive's existing canonical execution checks that compare aggregation, target, groups, filters, and operation graph before returning canonical descriptive results. The separate investigation reconciliation path still does not verify all filters/denominators across every execution path; that remains open.
6. **Regression tests:** add tests for two thresholds in one question, unique inflection resolution, and ambiguous exclusion dimensions.

## Validation

- `compileall` for the modified canonical contract and planner modules: passed.
- Focused filter-ingestion and intent-state tests: **36 passed**.
- Wider selected regression run: **47 passed, 2 failed due to missing environment dependency `python-jose`** while importing the API app. Another attempted test module could not be collected because `duckdb` is not installed. These are environment-blocked tests, not passes.
- The 152-case benchmark was not rerun because its fixture/runner was not found in the archive.

## Remaining work / limitations

- This is a filter-clause ledger, not the requested complete token-by-token semantic coverage proof.
- Post-execution contract reconciliation must be extended to compare actual filters, aggregation, grouping, numerator/denominator, and time scope on every path, and any mismatch/unverifiable result must downgrade to INCONCLUSIVE.
- Generic hypothesis fallbacks should be blocked whenever an enforced specific calculation is incomplete, not just for the currently covered contract families.
- The 152-case adjudicated benchmark and its runner are needed to measure accuracy, unsafe answers, refusals, and failures against the stated baseline.
- Multi-table, causal, forecasting, and time-series requests need separate typed grammars and validators.
