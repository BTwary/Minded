# MindEd AA-OS End-to-End Execution Fix Report

Date: 2026-10-16

## Defects addressed

1. **Approved-contract fast path bypassed variable-resolution preflight.** The controller dispatched a locked canonical contract before the central missing-variable gate. A question containing a nonexistent explicit identifier such as `marketing_spend` could therefore reach an approved execution path without the intended fail-closed check. The preflight now runs before locked-contract dispatch.
2. **Human corrections could be silently discarded after contract lock.** The existing guidance replay occurred later in the controller and explicitly cleared guidance when a locked contract existed. The locked-contract path now replays guidance before execution and amends the effective contract for validated `selected_metric`, `selected_aggregation`, and `selected_group_dimension` corrections. Each amendment records old/effective contract hashes and applied fields in `investigation.canonical_contract_amended`.
3. **A late scope-filter correction could be ignored.** Since free-form scope SQL is not represented as typed conditions in the already locked contract, the locked path now fails closed with an explicit message rather than running the broader population silently.
4. **Regression guard added.** Tests assert that the variable preflight and contract amendment logic occur before the approved execution dispatch, and that an explicit missing identifier is detected.

## Validation

Passed:

- `python -m py_compile packages/analytics_core/src/runtime/controller.py tests/independent_release/test_controller_preflight_order.py`
- `python -m pytest tests/test_universal_question_compiler.py tests/independent_release/test_contract_execution_reconciliation.py tests/independent_release/test_typed_investigation_planner.py tests/independent_release/test_controller_preflight_order.py -q`
- Result: **40 passed**

Not verified in this environment:

- The real controller question-matrix integration suite could not be collected because the runtime dependency `duckdb` is absent. Installing it was blocked by unavailable package-network access.
- The Windows executable has not been rebuilt or smoke-tested.

## Safety and compatibility notes

- Unsupported aggregation corrections fail closed instead of being approximated.
- A selected metric must exist in the locked dataset; quantitative metrics must be numeric unless a row-count operation is requested.
- Human corrections do not alter the original lock event. They create an auditable effective contract amendment for this execution.
- A scope filter submitted after lock is rejected until the filter is represented in the canonical typed contract. No unfiltered fallback is allowed.

Additional regression run:

- `tests/independent_release/test_m1_plan_contract_consistency.py` plus the four focused suites above: **47 passed, 2 blocked**.
- The two blocked tests import the API application and fail during collection/setup because `python-jose` (`jose`) is missing.
- The end-to-end controller question matrix remains blocked because `duckdb` is missing. Package installation could not reach the package index in this environment.
