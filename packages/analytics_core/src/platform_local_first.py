"""Local-first runtime paths for MindEd.

The scientific engine never requires a cloud account, AI provider, telemetry,
or a writable source-tree directory.  All mutable state defaults to a
per-user application-data directory and can be overridden explicitly.
"""
from __future__ import annotations

import os
import secrets
import sys
from pathlib import Path

APP_NAME = "Minded"
PRODUCT_DIR = "AAOS"


def user_data_root() -> Path:
    explicit = os.getenv("AAOS_DATA_ROOT")
    if explicit:
        return Path(explicit).expanduser().resolve()
    if sys.platform.startswith("win"):
        base = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / APP_NAME / PRODUCT_DIR
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME / PRODUCT_DIR
    base = os.getenv("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / APP_NAME / PRODUCT_DIR


def ensure_local_layout(root: Path | None = None) -> dict[str, Path]:
    root = (root or user_data_root()).expanduser().resolve()
    paths = {
        "root": root,
        "storage": root / "data_store",
        "backups": root / "backups",
        "logs": root / "logs",
        "config": root / "config",
        "exports": root / "exports",
        "feedback": root / "feedback",
        "cache": root / "cache",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def default_database_url() -> str:
    path = ensure_local_layout()["root"] / "autonomous_analyst.db"
    return f"sqlite:///{path.as_posix()}"


_AI_PROVIDERS_REQUIRING_OPT_IN = ("gemini", "openai", "claude", "anthropic", "groq", "ollama", "custom")


def resolve_ai_enabled(environ=None) -> bool:
    """Single source of truth for whether the optional AI layer is enabled.

    Mode 1 (zero-AI) is the default and privacy boundary. An EXPLICIT AI_ENABLED value
    always wins (so AI_ENABLED=false can never be overridden by a leftover AI_PROVIDER).
    Only when AI_ENABLED is unset/blank is a named external provider treated as opt-in.
    """
    env = os.environ if environ is None else environ
    raw = (env.get("AI_ENABLED") or "").strip().lower()
    if raw:
        return raw in ("true", "1", "yes", "on")
    return (env.get("AI_PROVIDER") or "none").strip().lower() in _AI_PROVIDERS_REQUIRING_OPT_IN


def persistent_secret_key() -> str:
    env_secret = os.getenv("SECRET_KEY")
    if env_secret and not env_secret.startswith("replace-with-"):
        return env_secret
    config_dir = ensure_local_layout()["config"]
    secret_file = config_dir / "local_secret.key"
    try:
        if secret_file.exists():
            value = secret_file.read_text(encoding="utf-8").strip()
            if len(value) >= 32:
                return value
        value = secrets.token_urlsafe(48)
        secret_file.write_text(value, encoding="utf-8")
        try:
            os.chmod(secret_file, 0o600)
        except OSError:
            pass
        return value
    except OSError:
        # Last-resort process-local secret: still safe enough for a temporary
        # development process, but intentionally not presented as persistent.
        return secrets.token_urlsafe(48)


def local_first_defaults() -> dict[str, str]:
    layout = ensure_local_layout()
    return {
        "AAOS_DATA_ROOT": str(layout["root"]),
        "DATA_STORAGE_DIR": str(layout["storage"]),
        "DATABASE_URL": default_database_url(),
        "AI_ENABLED": "false",
        "AI_PROVIDER": "none",
        "STORAGE_PROVIDER": "local",
        "TELEMETRY_ENABLED": "false",
        "FEEDBACK_UPLOAD_ENABLED": "false",
    }
