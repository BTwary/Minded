# MindEd AA-OS desktop startup diagnostics fix

Date: 2026-10-20

## Why this update exists

The live desktop process owned the loopback listener but `GET /`, `GET /health`, and `GET /openapi.json` timed out with zero bytes. The SQLite investigation queue contained one `COMPLETED` job and no jobs in the checked active statuses. That rules out an active queued investigation as the direct explanation, but does not yet establish the root cause of the server hang.

## Changes

- The native launcher now waits for a real `HTTP 200` response from `/health`, not just a successful TCP connect. This prevents logging "Backend ready" and navigating the UI when the HTTP application is not responsive.
- The launcher writes recurring Python thread stack snapshots during the first five minutes of launch to `%LOCALAPPDATA%\\Minded\\AAOS\\logs\\desktop_thread_dump.log`. This is meant to make an unresponsive thread diagnosable in a subsequent run.
- The launcher logs before and after `window.load_url`, so the startup log can distinguish API readiness from navigation behavior.
- API lifespan startup now writes progress and elapsed time for schema initialization, local file retention, and local worker startup to `desktop_startup.log` when in offline desktop mode.
- Added source-level contract tests for health-based readiness and startup-phase logs.

## Verification performed here

- `python -m py_compile` passed for `packaging/windows/minded_entry.py` and `apps/api/src/main.py`.
- `pytest -q tests/test_offline_desktop_contract.py`: 23 passed.

## What this does not claim

This package is a **diagnostic update**, not a claim that the underlying runtime hang has already been fixed. A fresh Windows production build and a live run are required. If it still hangs, send `desktop_startup.log` and `desktop_thread_dump.log`; their phase markers and Python thread stacks should identify the blocked component. No database job state or user data is changed by this patch.
