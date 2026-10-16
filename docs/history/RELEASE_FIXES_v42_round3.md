# v42 round3 – desktop (.exe) release fixes

1. `process_isolation.run_isolated`: in a frozen build `sys.executable -m <module>` re-launched the GUI, so every
   isolated DuckDB query failed ("Worker exited ... without a result"). Frozen builds now call
   `<exe> --aaos-isolated-worker <req> <resp>`; `minded_entry.py` dispatches that flag before any GUI logic.
2. `minded_entry.py`: forces `ENVIRONMENT=development` and `AAOS_LOCAL_AUTOWORKER=true` (the loopback demo login and
   the in-process investigation worker are gated on them; a stray host `ENVIRONMENT` value would leave the UI unable to log in / jobs queued forever).
3. `MindEd_AAOS_v1.0.0_Windows_x64.spec`: removed hard-coded `C:/Users/bhask/...` paths (now relative to the spec).
4. Added `tests/test_frozen_worker_dispatch.py`.

Not verified in this environment: the Next.js static export (`apps/web/out`) and the actual PyInstaller/Windows build.

## Added after independent verification (locked env: Python 3.12, hash-locked requirements-lock.txt)
5. `canonical_question_contract.py`: (a) symmetric "X correlated with Y" between two numeric columns no longer lets the
   semantic engine's generic target guess flip the user's variable order; (b) in "Why ..." questions an outcome named in
   the question (e.g. "survival") now takes precedence over the semantic guess (which picked `pclass` from the word "class"
   and answered "No, the opposite"). Real-data stress test: 28/32 -> 32/32 (raw and clean).
6. `survival.py`: applied the hardening that `tests/test_phase28_survival_hardening.py` already required (median float
   tolerance, finite/empty guards, unlabeled-group drop, non-computable log-rank, constant-covariate + collinearity
   checks, `n`). Module is not used by the analysis pipeline.
7. `real_stress_results_raw.json` regenerated (the shipped copy claimed 32/32 but the code produced 28/32).

Verification: 1521 tests passed, 0 failed, 0 skipped (locked env + gradio/plotly/lifelines for the HF-space and survival tests).
Release gate: all content checks pass; remaining blockers are environmental (see handoff): clean-room needs
MINDED_CLEAN_ROOM=1 + frontend `npm ci` lock check; the lock audit passes only in an env without the extra test packages.

## Hugging Face Space (fixed)
8. `hf_space/requirements.txt`: the pinned set was unsatisfiable (gradio 6.25.0 needs python-multipart>=0.0.18, pinned 0.0.9). Relaxed
   fastapi/pydantic/httpx/multipart/uvicorn pins; resolves and runs on Python 3.12.
9. `hf_space/sample_data/financials/quarterly_financials_messy.csv`: revenue/opex/net-profit columns were empty (scenario 2 always
   returned INSUFFICIENT_DATA). Regenerated with currency prefixes, thousands separators, accounting parentheses and #N/A tokens.
10. Demo questions for scenarios 2 and 3 changed to ones the engine answers (retention-rate questions are gated as churn, correctly fail-closed).
11. `hf_space/app.py`: ephemeral data root, AI/telemetry/Gradio analytics off by default.
A ready-to-upload Space folder (flat layout, root README with YAML header) is delivered separately as MindEd_HF_Space_upload.zip.
12. HF privacy wording: removed "Local-First, Zero-Cloud / Air-Gapped / Zero data leaves the environment" from the Space README and UI
    banner (true for the desktop app, misleading for a hosted Space). Now states: hosted demo, uploads are processed on the Space,
    do not upload confidential data, no external AI API used.
13. `tests/independent_release/test_hf_space_app.py`: added executed-answer tests for scenarios 2 (net profit by division) and 3 (average
    monthly spend by tier) with expected values computed independently from the CSVs; the scenario-2 test fails on the old empty data.
14. `GET /investigations/{id}` now returns `error_message` (the recorded failure reason) when status is FAILED. The UI already read this
    field but the API never sent it, so every failure showed the generic "AA-OS investigation failed" text with no cause.
15. Slow .exe start-up: the build used PyInstaller `--onefile`, which unpacks the whole scientific stack (~500 MB) to %TEMP% on every launch
    (and Defender scans it). Now `--onedir --noupx` (build script and .spec): app folder = exe + `_internal`; nothing is unpacked per launch.
    This also makes the `_internal` check in offline_desktop_certification.py meaningful.
16. `packaging/windows/minded_entry.py`: the native window now opens immediately with a loading screen; the engine is imported and started in
    the background and the window navigates to it when ready (previously nothing was visible until the backend finished starting).
    Distribute the whole `bundle/MindedAAOS` FOLDER (install.ps1 already copies the folder); do not copy the exe on its own.
    Not testable here (needs Windows + WebView2): verify the first window appears within ~1-2 s and the UI loads after the engine is ready.
