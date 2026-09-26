"""Self-contained offline desktop entrypoint.

The packaged executable starts the deterministic local API in a background
thread and displays it inside a native OS window (via pywebview) rather than
the user's default web browser. There is no visible browser tab, no console
window, and no port the user ever needs to know about: the app opens and
closes exactly like any other installed desktop program. Node/npm are
build-time only and never required on the end-user machine.
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
WINDOW_TITLE = "Minded AA-OS"


def _free_loopback_port() -> int:
    """Pick an unused loopback port so the app never collides with anything
    else that might be running on the machine, and the user never sees or
    needs to know a port number exists."""
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

        ctypes.windll.user32.MessageBoxW(0, message, WINDOW_TITLE, 0x10)
    except Exception:
        print(message, file=sys.stderr)


def _resource_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


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

from apps.api.src.main import app  # noqa: E402
import uvicorn  # noqa: E402


def _icon_path() -> str | None:
    candidate = _root / "app.ico"
    return str(candidate) if candidate.is_file() else None


def main() -> None:
    port = _free_loopback_port()
    config = uvicorn.Config(app, host=HOST, port=port, reload=False, log_level="warning")
    server = uvicorn.Server(config)

    server_thread = threading.Thread(target=server.run, name="aaos-local-api", daemon=True)
    server_thread.start()

    if not _wait_until_ready(HOST, port):
        server.should_exit = True
        _show_fatal_error(
            "Minded AA-OS could not start its local analytical engine.\n\n"
            "Please restart the application. If this keeps happening, check "
            "that no other program is using a conflicting port or blocking "
            "127.0.0.1, and reinstall if the problem persists."
        )
        sys.exit(1)

    try:
        import webview
    except Exception:
        # pywebview (or its native WebView2 dependency) is unavailable.
        # Fall back to the default browser rather than leaving the user with
        # nothing, but this path should not occur in a properly built release.
        import webbrowser

        webbrowser.open(f"http://{HOST}:{port}/")
        server_thread.join()
        return

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
        # Shut the embedded server down cleanly when the user closes the
        # window, exactly as any other installed desktop application would.
        server.should_exit = True

    window.events.closed += _on_closed

    try:
        webview.start(icon=_icon_path(), private_mode=False)
    except Exception as exc:  # pragma: no cover - native GUI failure path
        server.should_exit = True
        _show_fatal_error(
            "Minded AA-OS could not open its application window.\n\n"
            "This usually means the Microsoft Edge WebView2 Runtime is not "
            "installed. Windows 10 (22H2+) and Windows 11 include it by "
            "default; on older systems, install the 'WebView2 Runtime' "
            "from Microsoft and reopen Minded AA-OS.\n\n"
            f"Details: {exc}"
        )
        sys.exit(1)

    server.should_exit = True
    server_thread.join(timeout=5)


if __name__ == "__main__":
    main()
