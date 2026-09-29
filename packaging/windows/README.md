# Windows installation and `.exe` scope

The supported zero-friction distribution target is a **local desktop bundle**. The release implementation must package the Python API/analytical engine, the production Next.js frontend, and all required runtime assets behind a self-contained local launcher. The installed machine must not need Python, Node, npm, pip, or internet access.

## Required installer responsibilities

1. Install/use a private Python runtime; do not require the user to manage a global Python environment.
2. Store mutable state under `%LOCALAPPDATA%\\Minded\\AAOS` rather than beside the executable.
3. Create SQLite, dataset, backup, logs, cache, exports, and feedback directories on first launch.
4. Generate/persist a local secret automatically; never ask the user to invent one.
5. Run deterministic analysis with **AI and cloud disabled** until the user explicitly opts in.
6. Provide an optional encrypted backup/restore workflow that can remain entirely local.
7. Keep optional cloud storage/AI credentials in OS-managed/user-controlled configuration rather than the executable or source package.
8. Support offline launch and analysis after installation.

## Recommended packaging direction

Use a signed Windows installer plus a self-contained PyInstaller application and a static/prebuilt frontend served by the local FastAPI process. Node/npm are build-time only; the installed application has no Node/Python/pip runtime dependency exposed to the host machine.

The `.exe` is a self-contained local application that serves the exported frontend from the canonical local analytical process over loopback-only HTTP, and displays it inside a **native OS window** (via `pywebview`, backed by the Microsoft Edge WebView2 runtime on Windows) rather than the user's default browser. There is no visible browser tab, address bar, port number, or console window: `MindedAAOS.exe` opens and closes exactly like an ordinary installed application, with its own icon, title bar, and taskbar entry. Internally it still runs a loopback-only HTTP server (that is how essentially every Electron/webview-style desktop app talks to its own backend), but the user never sees or touches it. For a truly offline installer, the final release builder should vendor the exact Python/Node runtimes or ship their official redistributables and a local wheel/npm cache; the release `install.ps1` now fails closed unless a certified local wheelhouse and bundled Python runtime are present. `-BootstrapOnline` is development-only and is never a release path.

### Native window requirements

- Build machine needs `pywebview` and `pyinstaller` installed from the certified wheelhouse — see `packaging/windows/requirements-desktop.txt`.
- End-user machine needs the **Microsoft Edge WebView2 Runtime**. Windows 10 (22H2+) and Windows 11 ship it pre-installed; only older Windows 10 builds may need the redistributable from Microsoft, which the installer can optionally bundle (`MicrosoftEdgeWebview2Setup.exe`, offline/evergreen bootstrapper) for fully air-gapped installs.
- If the runtime window ever fails to start (e.g. WebView2 missing), the packaged app shows a native message box explaining why instead of crashing to a console.

## Non-goals

The `.exe` must not silently create a Supabase/Postgres account, Hugging Face Space, cloud bucket, AI API key, telemetry channel, or paid service dependency.

## Offline certification

Before shipping a desktop bundle, run:

```powershell
python scripts\offline_desktop_certification.py --bundle <final-bundle-root> --json <certification.json>
```

Certification passes only when the bundle contains its own runtime/assets, declares `offline=true` and `loopback-only`, contains a non-empty local wheelhouse, and contains no external network URL references in packaged runtime/configuration files. A source-only checkout is not considered a certified desktop release.
