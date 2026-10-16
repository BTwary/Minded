# Merge + real-workspace fix (2026-10-07)

Merged `MindEd_fixed (1).zip` ("A") and `MindEd_AAOS_v14_real_workspace_fixes.zip` ("B"). They differed in
24 files; the changes were complementary, so the merge is the union (B as base, A's fixes layered on).

## The reported error
`Investigation Execution Error: Contract <uuid> not found` on the 4-CSV workspace
(products 200r, orders 120,000r, customers 15,000r, sales_orders_flat_120k 120,000r).

Reproduced on zip A with the real controller on a worker thread while another thread polls status
like the UI does. Root cause: file-backed SQLite used `StaticPool` (one shared connection for all
threads), so the UI's status poll `close()` rolled back the worker's uncommitted contract row.
Fixed in B (kept): per-thread connections for file DBs (`database.py`), dataset-quality evaluation moved
before the contract write transaction, and a durability check after commit (`controller.py`).

## Taken from A (fail-closed verification)
- `verification.py`, `epistemic_calibration.py`: no fabricated secondary value (None stays None); calibration
  treats a missing delta as maximal uncertainty instead of raising `TypeError`.
- `execution_provider.py`: primary metric derived from the full result, truncation only bounds the carried frame.
- `controller.py`: `record_experiment_failed` on timeout/exception so executed/failed ids agree.
- `test_verification_none_secondary_failclosed.py`.

## Taken from B
- AOV / measure phrases never bind to identifier columns (`canonical_question_contract.py`).
- Non-numeric target in missingness bounds returns "undefined" instead of crashing.
- Hash-locked desktop toolchain, signing scripts, wheelhouse build, Next.js inert-URL allowlist.
- `tests/test_real_workspace_regressions.py`.

## New
- `tests/test_e2e_threaded_category_revenue.py`: 120k-row file-DB investigation with concurrent status polling.
  Fails on zip A, passes on the merged tree.

## Verified on the real CSVs (merged build)
"Which product category generates the highest total revenue?" -> COMPLETED / OBSERVED, Electronics 24.11M of
61.79M; per-category totals and row counts match a direct pandas groupby exactly.
