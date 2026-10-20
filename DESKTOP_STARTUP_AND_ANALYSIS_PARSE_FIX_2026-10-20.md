# MindEd AA-OS: startup freeze and analysis parsing fixes

Date: 2026-10-20

## Summary

This patch addresses two user-facing defects reported in the Windows desktop app:

1. Startup stalls for minutes or displays `MindedAAOS (Not Responding)` before the interface loads.
2. Natural-language questions sometimes remain blocked in clarification/preview instead of becoming a runnable analysis, particularly grouped comparisons and ranking questions that include a trailing exclusion clause.

## Startup diagnosis and fix

The user-provided thread dump showed the pywebview GUI thread repeatedly inside `webview.util.get_functions()` (line 206), alongside the WinForms native-window thread. The exposed `DesktopExportAPI` had a native window reference; the native window, in turn, exposes the bridge as `js_api`. This creates a cyclic graph (`window -> js_api -> window`) that pywebview can recursively inspect while building its JavaScript bridge.

The native-window reference was moved to module-level `_desktop_window`, outside the exposed API object's attribute graph. `DesktopExportAPI` no longer holds any window reference; its save method consults the module global only when the user initiates an export. Window-close handling clears that global. The background API boot path now logs how long its imports take, so any residual slow startup can be separated from the bridge recursion.

This diagnosis is strongly supported by the thread dump, but the rebuilt Windows executable still needs a live startup test. Python module imports such as pandas can still take some time in a packaged application; this change targets the observed recursive GUI-thread stall, not every possible source of startup latency.

## Analysis parsing fixes

- Ranking questions with a trailing population/exclusion clause can preserve the metric before the scope clause, allowing the existing filter parser to bind the rest of the sentence.
- `revenue` resolves to a revenue-named schema column only when the match is unique. If both `net_revenue` and `realized_revenue` exist, the contract does not guess; it stays blocked for metric clarification.
- A ranking with no resolved target metric is marked unresolved and is not treated as executable.
- The plain-language pattern `How does revenue compare by region?` now compiles through the canonical contract as a schema-grounded grouped aggregate. The default is `SUM` for this comparative question when no aggregation word is specified.
- The test cases cover single and coordinated status exclusions, ambiguous revenue columns, missing ranking targets, and grouped comparisons.

Examples covered by regression tests:

- `Which region generated the most revenue after excluding cancelled orders?`
- `Which region generated the most revenue after excluding returns and cancellations?`
- `How does revenue compare by region?`

## Verification performed in the source workspace

- Python compilation check passed for `packaging/windows/minded_entry.py` and `packages/analytics_core/src/intelligence/canonical_question_contract.py`.
- Targeted parser/startup suite: **39 passed**.
- Related workspace/contract/regression suite: **34 passed**. This is a separate overlapping suite, not an additive test count.
- A complete suite was not run: DuckDB is not installed in this Linux source-validation environment, and some broader tests require it.
- No fresh Windows PyInstaller build or live desktop run was performed here. The user must rebuild on Windows and check the final offline certification output, then confirm window startup and analysis behavior.

## Windows rebuild / verification

Apply the ZIP over the existing `MindEd_desktop_startup_diagnostic_fix` source folder, preserving `.venv` and local user data. From that folder, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
$runtime = (Resolve-Path .\.venv\Scripts\python.exe).Path
.\packaging\windows\build_exe_scope.ps1 -PythonRuntimePath $runtime
```

Launch only after `OFFLINE DESKTOP CERTIFICATION: PASS`:

```powershell
& ".\packaging\windows\bundle\MindedAAOS\MindedAAOS.exe"
```

On a successful startup, verify that the loading screen completes promptly, `/health` returns HTTP 200, and the question examples above either run or ask a specific schema-grounded clarification rather than blocking on a false date/time requirement.
