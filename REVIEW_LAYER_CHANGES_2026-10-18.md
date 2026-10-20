# Review/investigation layer — changes of 2026-10-18

Baseline (original archive, `pytest tests packages`): 8 failed, 1902 passed, 2 skipped.
Final (this tree): 0 failed, 1931 passed, 2 skipped.

| File | Change |
|---|---|
| packages/analytics_core/src/intelligence/intent_state.py | `_question_requirements`: an explicit predicate on an identifier-shaped field absent from the dataset (`where loyalty_tier = Gold`) is an UNRESOLVED `filter_conditions` requirement (`filter_column_unavailable`). Previously the resolver rebuilt the contract and discarded the compiler's INCOMPLETE flag, so the unfiltered aggregate could run. |
| packages/analytics_core/src/governance/guided_investigation.py | (1) preview emits ORDER BY when the contract has an explicit ranking direction; (2) `render_contract_preview_from_contract` returns None for INCOMPLETE contracts; (3) readable reason/prompt for an unavailable filter column; (4) `AMBIGUOUS_METRIC` added to the protected statuses so the specific two-measure diagnosis is not downgraded to MISSING_DATA. |
| packages/analytics_core/src/engines/analyst_answer.py | Grouped ranking answers add "X has the highest/lowest <measure>: value (n=…)" (or a tie statement) to the details, after the verified ranked table. Headline format unchanged (existing tests parse it). |
| packages/analytics_core/src/runtime/approved_execution.py | Same ranking statement for the SQL-bound approved path. |
| packages/analytics_core/src/intelligence/canonical_question_contract.py | Approved alias `refund rate` -> refund_flag/is_refunded/refunded/refund. |
| apps/api/src/api/v1/investigations.py | `/interpret`: when the resolver has not accepted the question, `planned_calculation` is "Not executable yet…" instead of heuristic SQL. |
| tests/independent_release/test_review_layer_2026_10_18.py | 16 new regression tests. |

Known gaps: see the report delivered with this archive (top-N, compound-operation execution, multi-measure comparison, empty "Compare revenue" contract, etc.).

## Second pass: the 7 baseline failures

| Test(s) | Cause | Resolution |
|---|---|---|
| 3 in test_relational_plan_contract.py (+1 sub-assertion), test_calculation_lineage.py | Stale expectations: generated SQL deliberately quotes identifiers (`SUM("revenue")`) | Assertions updated to the quoted form (4 tests). Not a code change. |
| test_p0_security_tenant_isolation::test_allows_legitimate_public_endpoints | Depended on live DNS; the validator correctly fails closed when a name cannot resolve | Test now stubs `getaddrinfo`; added `test_unresolvable_hostname_fails_closed`. SSRF code unchanged. |
| test_session6 plateau stagnation | Test omitted `all_verified`; the engine deliberately treats missing verification as not verified | Test passes `all_verified=True`; added a test that an unverified plateau does NOT stop. Engine unchanged. |
| test_session10 multi-group numeric comparison (end-to-end) | Real defect: the analyst layer promoted a group difference to STATISTICALLY_SIGNIFICANT from its own recomputed tests, but the final P0 gate required loop-linked evidence that this question's loop never produces (it tested "concentration"), so a clear +15 effect at n=1200 was reported INCONCLUSIVE | `verification_gate.analyst_group_comparison_independently_verified`: an analyst-owned GROUP_COMPARISON counts as verified only if ANOVA and Kruskal-Wallis both reject at 0.05, `robust` is True and power is adequate. `all_verifications_passed` is still required. Gate otherwise unchanged. Null-data control and unit tests added. |

Still open (not addressed): top-N, compound-operation execution, multi-measure comparison, empty "Compare revenue" contract, wrong measure/dimension bindings for "Top X by Y" and "average order value per customer", misleading refusal for "Show total revenue where region = North", relative-period resolution.
