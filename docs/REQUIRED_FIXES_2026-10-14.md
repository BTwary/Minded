# MindEd AA-OS — Required Conditional Runtime Fixes

Date: 2026-10-14

## Changes applied

1. **Connected typed branch resolution to the authoritative investigation controller.** After the universal plan and causal gate have produced runtime facts, the controller calls `resolve_plan_branches(...)` and `conditional_execution_policy(...)` before causal effect estimation and before the recursive experiment loop.
2. **Persisted decisions durably.** The controller writes `investigation.typed_conditional_plan.resolved` and, where a branch changes the allowed execution path, `investigation.typed_conditional_plan.enforced`. Events include plan version/fingerprint, task type, fact values, branch decisions, policy, and attempt number.
3. **Made critical conditional paths fail closed.** Forecasting requires a measured monotonic time index; prediction requires an independent leakage PASS (not inferred from an empty heuristic list); comparisons require at least two observed groups; association requires a compatible validated branch; segmentation interpretation requires measured stability; decision recommendations require resolved objective/constraints; reconciliation requires an explicit identity. Missing branch records are not treated as success.
4. **Kept causal fallback appropriately bounded.** When identification is not established, non-causal investigation may continue, but the typed branch cannot authorize causal-effect estimation and the claim is downgraded to association.
5. **Hardened predicate semantics.** Type-mismatched equality predicates and malformed membership operands now return `UNKNOWN`, not a misleading `FALSE`.
6. **Added regression coverage.** New tests exercise task-level authorization, missing/blocked branch fail-closed behavior, causal non-causal fallback, unknown leakage state, controller wiring, and release-bootstrap safeguards.
7. **Added a Windows clean-environment bootstrap script.** `scripts/bootstrap_release_env.ps1` creates `.venv-release`, installs `requirements-lock.txt` with `--require-hashes`, checks dependency consistency and runs the release gate. It deliberately refuses to remove hashes or loosen pins.

## Validation run

| Check | Result |
|---|---|
| Focused planning, intent, derived-rate, contract, operation registry and conditional wiring tests | **84 passed** |
| `python -m compileall -q packages apps scripts tests` | **PASS** |
| `scripts/audit_release_status.py --root .` | **PASS** |
| `scripts/audit_canonical_serialization.py --root .` | **PASS** — 193 files scanned |
| Dependency lock audit in this runner | **FAIL / environment mismatch** |
| Controller-importing autonomy test | **BLOCKED** — `duckdb` is not installed |
| Full manifest release gate / clean Windows EXE | **NOT RUN TO COMPLETION** |

## Why release certification remains blocked

This runner uses Python 3.13.5 and has many missing or mismatched distributions relative to the project's exact hashed lock. Attempting to install the missing packages failed because package-index DNS/network access is unavailable in this environment. The dependency audit currently reports missing packages such as DuckDB, Polars, PyArrow, python-jose and sqlglot, plus version mismatches including NumPy, pandas, Pydantic, SQLAlchemy, FastAPI and pytest.

These are not safely fixed by changing the lock to match an arbitrary preinstalled environment. The correct next verification step is to run `scripts/bootstrap_release_env.ps1` on a Windows machine with Python 3.12 and package-index access, then resolve any actual locked-install or manifest failures. After that, test the packaged EXE from a clean extraction and fresh user-data directory with real datasets/questions.

## Important limitations

The runtime now **enforces** the new conditional policy. Some specialist paths will intentionally remain blocked until their missing evidence producer exists: in particular, this controller does not yet produce an independent leakage PASS, cluster-stability PASS, or complete decision context automatically. This prevents unsupported model/segment/recommendation claims; it does not mean those specialist capabilities are fully implemented end-to-end.

**Verdict: conditional integration fix implemented and focused tests pass; release certification remains BLOCKED pending exact-lock integration tests and clean-machine EXE verification.**
