"""Self-contained offline desktop entrypoint for MindEd AA-OS.

The packaged executable initializes the local application environment, starts the
deterministic local API in a background thread on an unused loopback port, and displays
it inside a native OS window (via pywebview) rather than the user's web browser.
There is no visible browser tab, no console window, and no port the user ever needs to
know about: the app opens and closes exactly like any other installed desktop software.
Node/npm and external Python runtimes are build-time only and never required on the end-user machine.
"""
from __future__ import annotations

import os
import socket
import sys
import threading
import time
from pathlib import Path

# Release invariants are established before importing the application module.
# Force offline invariants unconditionally; host environment variables cannot override desktop release policy.
os.environ["AAOS_OFFLINE_MODE"] = "1"
os.environ["AI_ENABLED"] = "false"
os.environ["AI_PROVIDER"] = "none"
os.environ["TELEMETRY_ENABLED"] = "false"
os.environ["FEEDBACK_UPLOAD_ENABLED"] = "false"
os.environ["STORAGE_PROVIDER"] = "local"

HOST = "127.0.0.1"
WINDOW_TITLE = "MindEd AA-OS"


def _acquire_single_instance(mutex_name: str = "Local\\MindEd_AAOS_SingleInstance_Mutex") -> bool:
    """Ensure only one desktop instance runs at a time.
    If another instance is active, bring its native window to the foreground and exit."""
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        user32 = ctypes.windll.user32

        # Create or open named mutex
        kernel32.CreateMutexW(None, True, mutex_name)
        last_error = kernel32.GetLastError()
        ERROR_ALREADY_EXISTS = 183

        if last_error == ERROR_ALREADY_EXISTS:
            hwnd = user32.FindWindowW(None, WINDOW_TITLE)
            if hwnd:
                user32.ShowWindow(hwnd, 9)  # SW_RESTORE
                user32.SetForegroundWindow(hwnd)
            return False
        return True
    except Exception:
        return True


def _free_loopback_port() -> int:
    """Pick an unused loopback port dynamically so the app never collides with anything
    else running on the machine, and the user never sees or needs to know a port number exists."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((HOST, 0))
        return probe.getsockname()[1]


def _wait_until_ready(host: str, port: int, timeout_seconds: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.25)
            try:
                probe.connect((host, port))
                return True
            except OSError:
                time.sleep(0.1)
    return False


def _show_fatal_error(message: str) -> None:
    """Best-effort native error dialog. The packaged app must never crash to
    a raw traceback in front of a non-technical end user."""
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(0, message, WINDOW_TITLE, 0x10)  # MB_ICONERROR | MB_OK
    except Exception:
        print(message, file=sys.stderr)


def _resource_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def _icon_path() -> str | None:
    root = _resource_root()
    # Check bundled root first
    bundled = root / "app.ico"
    if bundled.is_file():
        return str(bundled)
    # Check PyInstaller temp MEIPASS if applicable
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        meipass_icon = Path(meipass) / "app.ico"
        if meipass_icon.is_file():
            return str(meipass_icon)
    # Check source development tree
    dev_icon = root / "packaging" / "windows" / "app.ico"
    if dev_icon.is_file():
        return str(dev_icon)
    return None


def _allow_loopback(host: object) -> bool:
    # Never resolve arbitrary hostnames here: DNS is itself an outbound network
    # dependency and must remain impossible in offline desktop mode.
    value = str(host).strip().lower().rstrip(".")
    return value in {"localhost", "::1", "127.0.0.1"}


def _install_offline_network_guard() -> None:
    """Block non-loopback sockets for the offline desktop process."""
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_getaddrinfo = socket.getaddrinfo

    def connect(self: socket.socket, address):  # type: ignore[no-untyped-def]
        host = address[0] if isinstance(address, tuple) and address else address
        if not _allow_loopback(host):
            raise OSError("AAOS offline mode blocks non-loopback network access")
        return original_connect(self, address)

    def connect_ex(self: socket.socket, address):  # type: ignore[no-untyped-def]
        host = address[0] if isinstance(address, tuple) and address else address
        if not _allow_loopback(host):
            return 111
        return original_connect_ex(self, address)

    def getaddrinfo(host, port, *args, **kwargs):  # type: ignore[no-untyped-def]
        if not _allow_loopback(host):
            raise OSError("AAOS offline mode blocks non-loopback name resolution")
        return original_getaddrinfo(host, port, *args, **kwargs)

    socket.socket.connect = connect  # type: ignore[assignment]
    socket.socket.connect_ex = connect_ex  # type: ignore[assignment]
    socket.getaddrinfo = getaddrinfo  # type: ignore[assignment]


_install_offline_network_guard()

# Resolve the bundled static frontend before the application is imported.
# PyInstaller places runtime files beside the executable in the onedir bundle.
_root = _resource_root()
_frontend = _root / "frontend"
if _frontend.is_dir():
    os.environ["AAOS_FRONTEND_ROOT"] = str(_frontend)


def main() -> None:
    # 1. Single instance guard
    if not _acquire_single_instance():
        sys.exit(0)

    # 2. Local loopback backend startup
    try:
        from apps.api.src.main import app
        import uvicorn
    except Exception as exc:
        _show_fatal_error(
            "MindEd could not start its local analytical engine.\n\n"
            "Your data has not been sent to an external service.\n\n"
            "Restart MindEd and try again."
        )
        sys.exit(1)

    port = _free_loopback_port()
    config = uvicorn.Config(app, host=HOST, port=port, reload=False, log_level="warning")
    server = uvicorn.Server(config)

    server_thread = threading.Thread(target=server.run, name="aaos-local-api", daemon=True)
    server_thread.start()

    if not _wait_until_ready(HOST, port):
        server.should_exit = True
        _show_fatal_error(
            "MindEd could not start its local analytical engine.\n\n"
            "Your data has not been sent to an external service.\n\n"
            "Restart MindEd and try again."
        )
        sys.exit(1)

    # 3. Native desktop window via pywebview (no browser fallback)
    try:
        import webview
    except Exception:
        server.should_exit = True
        _show_fatal_error(
            "MindEd could not open its application window.\n\n"
            "This application requires the Microsoft Edge WebView2 Runtime. "
            "Windows 10 (22H2+) and Windows 11 include it by default. On older systems, "
            "install the 'WebView2 Runtime' from Microsoft and reopen MindEd AA-OS."
        )
        sys.exit(1)

    window = webview.create_window(
        WINDOW_TITLE,
        url=f"http://{HOST}:{port}/",
        width=1440,
        height=900,
        min_size=(1024, 700),
        text_select=True,
        confirm_close=False,
    )

    def _on_closed() -> None:
        # Shut the embedded server down cleanly when the user closes the window.
        server.should_exit = True

    window.events.closed += _on_closed

    try:
        webview.start(icon=_icon_path(), private_mode=False)
    except Exception as exc:  # pragma: no cover - native GUI failure path
        server.should_exit = True
        _show_fatal_error(
            "MindEd could not open its application window.\n\n"
            "This application requires the Microsoft Edge WebView2 Runtime. "
            "Windows 10 (22H2+) and Windows 11 include it by default. On older systems, "
            "install the 'WebView2 Runtime' from Microsoft and reopen MindEd AA-OS."
        )
        sys.exit(1)

    server.should_exit = True
    server_thread.join(timeout=5)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        _show_fatal_error(
            "MindEd could not start its local analytical engine.\n\n"
            "Your data has not been sent to an external service.\n\n"
            "Restart MindEd and try again."
        )
        sys.exit(1)
