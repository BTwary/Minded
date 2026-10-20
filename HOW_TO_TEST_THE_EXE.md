# Build and test the MindEd AA-OS Windows EXE

**Important:** this source archive does not ship a prebuilt `.exe`. Build the Windows application on Windows so PyInstaller collects the correct Windows runtime and native WebView components.

## 1. Build and launch

Prerequisites on the Windows 10/11 x64 build machine:

- 64-bit CPython 3.12 and the Python Launcher (`py.exe`)
- Node.js LTS with npm
- Microsoft Edge WebView2 Runtime (normally present on current Windows 10/11)
- Network access during the initial locked-dependency and frontend build

Extract this entire source package to a writable folder, open PowerShell in its root, then run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\BUILD_AND_TEST_EXE.ps1
```

This creates a separate `.venv-exe-test`, installs the two hash-locked dependency sets, runs focused regressions, builds the current web frontend, creates the PyInstaller desktop bundle, runs offline bundle certification, and launches the EXE for a GUI smoke test. The build may take some time on the first run.

To skip the focused Python tests but still build and launch the EXE:

```powershell
.\BUILD_AND_TEST_EXE.ps1 -SkipFocusedTests
```

To build and certify without launching the window:

```powershell
.\BUILD_AND_TEST_EXE.ps1 -SkipLaunch
```

The EXE is expected at:

```text
packaging\windows\bundle\MindedAAOS\MindedAAOS.exe
```

The helper does **not** sign the EXE or installer. It is for local test builds, not final public distribution.

## 2. Manual GUI acceptance checklist

Use a safe copy of a test dataset, not your only copy of production data.

1. **Launch:** native MindEd window opens; no console window or browser address bar appears.
2. **Startup diagnostics:** confirm `%LOCALAPPDATA%\Minded\AAOS\logs\desktop_startup.log` includes `HTTP health check passed (status=200`.
3. **Dataset load:** import a CSV with headers and confirm column names, row count, data types and missing values are sensible.
4. **Filter before analysis:** choose customer segment = `Small Business`, quarter = `Q3`, delivery completed = `1`; run a question and confirm the same three conditions appear in interpreted and executed filters and in the report.
5. **Question sequence:** ask two or more questions in a single analysis; accept each result and verify each appears as a separate section in the same cumulative report.
6. **Filter correction:** change one filter through the supported correction path; confirm the new run records its new effective filters and prior runs remain traceable.
7. **Visual selection:** ask a ranking, time-trend and relationship question; confirm visualization types differ where the data and method warrant it.
8. **Why/diagnostic:** ask a diagnostic question; confirm the report distinguishes computed evidence, possible explanations and unsupported causal claims.
9. **Export:** export PDF and DOCX before closing; check both include every accepted question, consistent numbers, filters, charts/tables and limitations.
10. **Close analysis:** close it; verify that adding a question or resuming review is blocked or requires the explicitly supported reopen/version flow.
11. **Persistence:** close and relaunch the app, open history, and verify the analysis and question trail still appear.
12. **Offline behavior:** after a successful first setup, disconnect from the network and check that the app still opens and local analysis works in its supported offline mode.

Record pass/fail and attach logs/PDF/DOCX to each failure. A process that remains running is not by itself proof that the UI or analysis workflow works.

## 3. Test artifacts

The helper writes logs, a JSON summary and offline certification output under `artifacts\exe-test\`. Startup logs are in `%LOCALAPPDATA%\Minded\AAOS\logs\`.

A successful build and offline certification are necessary but not sufficient for release readiness: manual GUI tests and analytic correctness checks still need to pass.
