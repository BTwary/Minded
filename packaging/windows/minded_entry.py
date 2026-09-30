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

# In a --windowed GUI process on Windows, sys.stdout and sys.stderr are None.
# Provide null streams that safely implement isatty() and write() to prevent
# formatters or libraries (e.g. uvicorn.logging.DefaultFormatter) from failing.
class _NullStream:
    def write(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        pass

    def flush(self):  # type: ignore[no-untyped-def]
        pass

    def isatty(self) -> bool:
        return False


if sys.stdout is None:
    sys.stdout = _NullStream()  # type: ignore[assignment]
if sys.stderr is None:
    sys.stderr = _NullStream()  # type: ignore[assignment]

# Release invariants are established before importing the application module.
# Force offline invariants unconditionally; host environment variables cannot override desktop release policy.
os.environ["AAOS_OFFLINE_MODE"] = "1"
os.environ["AAOS_BUSINESS_TIMEZONE"] = os.getenv("AAOS_BUSINESS_TIMEZONE", "UTC")
os.environ["AI_ENABLED"] = "false"
os.environ["AI_PROVIDER"] = "none"
os.environ["TELEMETRY_ENABLED"] = "false"
os.environ["FEEDBACK_UPLOAD_ENABLED"] = "false"
os.environ["STORAGE_PROVIDER"] = "local"

HOST = "127.0.0.1"
WINDOW_TITLE = "MindEd AA-OS"

_instance_mutex_handle = None


def _log_file() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / "Minded" / "AAOS" / "logs" / "desktop_startup.log"


def _log(msg: str) -> None:
    try:
        log_path = _log_file()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    except Exception as e:
        print(f"Log failure: {e}", file=sys.stderr)


def _acquire_single_instance(mutex_name: str = "Local\\MindEd_AAOS_SingleInstance_Mutex") -> bool:
    """Ensure only one desktop instance runs at a time.
    If another instance is active, bring its native window to the foreground and exit."""
    global _instance_mutex_handle
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        user32 = ctypes.windll.user32

        kernel32.SetLastError(0)
        handle = kernel32.CreateMutexW(None, True, mutex_name)
        last_error = kernel32.GetLastError()
        ERROR_ALREADY_EXISTS = 183

        if last_error == ERROR_ALREADY_EXISTS:
            hwnd = user32.FindWindowW(None, WINDOW_TITLE)
            if hwnd:
                user32.ShowWindow(hwnd, 9)  # SW_RESTORE
                user32.SetForegroundWindow(hwnd)
            return False

        _instance_mutex_handle = handle
        return True
    except Exception as exc:
        _log(f"Warning: Single instance mutex acquisition error: {exc}")
        return True


def _free_loopback_port() -> int:
    """Pick an unused loopback port dynamically so the app never collides with anything
    else running on the machine, and the user never sees or needs to know a port number exists."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((HOST, 0))
        return probe.getsockname()[1]


def _wait_until_ready(host: str, port: int, server_thread: threading.Thread, timeout_seconds: float = 45.0) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not server_thread.is_alive():
            return False
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.25)
            try:
                probe.connect((host, port))
                return True
            except OSError:
                time.sleep(0.1)
    return False


def _show_fatal_error(message: str, exc: BaseException | None = None) -> None:
    """Best-effort native error dialog. The packaged app must never crash to
    a raw traceback in front of a non-technical end user."""
    import traceback

    if exc is not None:
        _log(f"Fatal error: {exc}\n{traceback.format_exc()}")
    else:
        _log(f"Fatal error: {message}")

    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(0, message, WINDOW_TITLE, 0x10)  # MB_ICONERROR | MB_OK
    except Exception:
        print(message, file=sys.stderr)


def _resource_root() -> Path:
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass).resolve()
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
    if host is None or host == "" or host == 0:
        return True
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
_root = _resource_root()
_frontend = _root / "frontend"
if _frontend.is_dir():
    os.environ["AAOS_FRONTEND_ROOT"] = str(_frontend)


def main() -> None:
    _log("MindEd AA-OS starting...")
    # 1. Single instance guard
    if not _acquire_single_instance():
        _log("Another instance is already running; brought to foreground and exiting.")
        sys.exit(0)

    # 2. Local loopback backend startup
    try:
        from apps.api.src.main import app
        import uvicorn
    except Exception as exc:
        _show_fatal_error(
            "MindEd could not start its local analytical engine.\n\n"
            "Your data has not been sent to an external service.\n\n"
            "Restart MindEd and try again.",
            exc=exc,
        )
        sys.exit(1)

    port = _free_loopback_port()
    _log(f"Loopback port allocated: {port}")
    config = uvicorn.Config(app, host=HOST, port=port, reload=False, log_config=None)
    server = uvicorn.Server(config)

    server_exception: list[Exception] = []

    def _run_server() -> None:
        try:
            server.run()
        except Exception as e:
            server_exception.append(e)
            _log(f"Server thread exception: {e}")

    server_thread = threading.Thread(target=_run_server, name="aaos-local-api", daemon=True)
    server_thread.start()

    if not _wait_until_ready(HOST, port, server_thread=server_thread):
        server.should_exit = True
        exc = server_exception[0] if server_exception else None
        _show_fatal_error(
            "MindEd could not start its local analytical engine.\n\n"
            "Your data has not been sent to an external service.\n\n"
            "Restart MindEd and try again.",
            exc=exc,
        )
        sys.exit(1)

    _log("Backend ready. Starting native desktop window...")

    # 3. Native desktop window via pywebview (no browser fallback)
    try:
        import webview
    except Exception as exc:
        server.should_exit = True
        _show_fatal_error(
            "MindEd could not open its application window.\n\n"
            "This application requires the Microsoft Edge WebView2 Runtime. "
            "Windows 10 (22H2+) and Windows 11 include it by default. On older systems, "
            "install the 'WebView2 Runtime' from Microsoft and reopen MindEd AA-OS.",
            exc=exc,
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
        _log("Native window closed by user; shutting down local backend...")
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
            "install the 'WebView2 Runtime' from Microsoft and reopen MindEd AA-OS.",
            exc=exc,
        )
        sys.exit(1)

    server.should_exit = True
    server_thread.join(timeout=5)
    _log("MindEd AA-OS exited cleanly.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        _show_fatal_error(
            "MindEd could not start its local analytical engine.\n\n"
            "Your data has not been sent to an external service.\n\n"
            "Restart MindEd and try again.",
            exc=exc,
        )
        sys.exit(1)
