"""Application Settings and Configuration."""
import json
import os
import secrets
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Load local configuration before settings or downstream modules read os.environ.
# Explicit override is intentionally disabled so real process environment values win.
load_dotenv(Path(__file__).resolve().parents[4] / ".env", override=False)
from typing_extensions import Annotated
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from packages.analytics_core.src.platform_local_first import default_database_url, persistent_secret_key, ensure_local_layout, resolve_ai_enabled
try:
    from pydantic_settings import NoDecode
except ImportError:
    class NoDecode:
        pass


class Settings(BaseSettings):
    PROJECT_NAME: str = "Autonomous AI Data Analyst"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = os.getenv("SECRET_KEY") or persistent_secret_key()
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    
    # Database: Default SQLite for instant local zero-cost run, or PostgreSQL
    DATABASE_URL: str = os.getenv("DATABASE_URL") or default_database_url()
    
    # Storage Directory
    DATA_STORAGE_DIR: str = os.getenv("DATA_STORAGE_DIR") or str(ensure_local_layout()["storage"])
    # Raw/derived dataset files are retained locally for 6 months by default.
    # Analysis metadata/history is stored separately and is not deleted by this TTL.
    DATA_RETENTION_DAYS: int = int(os.getenv("DATA_RETENTION_DAYS", "180"))
    
    # AI Augmentation Settings: Mode 1 (Free / Zero-API) vs Mode 2 (Optional AI Augmentation)
    AI_ENABLED: bool = resolve_ai_enabled()
    AI_PROVIDER: str = os.getenv("AI_PROVIDER", "none")
    AI_MODEL: str = os.getenv("AI_MODEL", "auto")
    AI_API_KEY: str = os.getenv("AI_API_KEY", "")
    AI_BASE_URL: str = os.getenv("AI_BASE_URL", "")
    
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-3-7-sonnet-20250219")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    DEMO_MODE: bool = os.getenv("DEMO_MODE", "false").lower() in ("true", "1")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    # Arbitrary Python execution is disabled by default. Enable only behind a real OS/container sandbox.
    CODE_EXECUTION_ENABLED: bool = os.getenv("AAOS_CODE_EXECUTION_ENABLED", "false").lower() in ("true", "1")
    REQUIRE_LOCAL_DEMO_TOKEN: bool = os.getenv("AAOS_REQUIRE_LOCAL_DEMO_TOKEN", "true").lower() in ("true", "1")

    # Explicit opt-in boundaries: no telemetry or unsolicited feedback/cloud upload.
    TELEMETRY_ENABLED: bool = os.getenv("TELEMETRY_ENABLED", "false").lower() in ("true", "1")
    FEEDBACK_UPLOAD_ENABLED: bool = os.getenv("FEEDBACK_UPLOAD_ENABLED", "false").lower() in ("true", "1")
    STORAGE_PROVIDER: str = os.getenv("STORAGE_PROVIDER", "local")
    AAOS_DATA_ROOT: str = os.getenv("AAOS_DATA_ROOT", str(ensure_local_layout()["root"]))

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def _default_database_url(cls, v):
        """.env.example documents leaving DATABASE_URL blank to get the computed
        local SQLite default. pydantic-settings loads that literal blank string
        from the .env file and uses it to override the class-level default above
        (which is only evaluated against the *process* environment, not the .env
        file), so the documented quickstart resolved to DATABASE_URL="" and
        create_engine("") crashed the app on startup. Re-apply the computed
        default whenever the loaded value is empty."""
        if v is None or (isinstance(v, str) and not v.strip()):
            return default_database_url()
        return v

    @field_validator("DATA_STORAGE_DIR", mode="before")
    @classmethod
    def _default_data_storage_dir(cls, v):
        if v is None or (isinstance(v, str) and not v.strip()):
            return str(ensure_local_layout()["storage"])
        return v

    @field_validator("AAOS_DATA_ROOT", mode="before")
    @classmethod
    def _default_aaos_data_root(cls, v):
        if v is None or (isinstance(v, str) and not v.strip()):
            return str(ensure_local_layout()["root"])
        return v

    CORS_ORIGINS: Annotated[List[str], NoDecode] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def _parse_cors_origins(cls, v):
        """Accept either a JSON array or the documented comma-separated
        string (see .env.example) for CORS_ORIGINS. pydantic-settings v2
        otherwise attempts strict JSON-decoding of List[str] env vars and
        raises SettingsError on a plain comma-separated value, which is
        exactly what .env.example ships -- so the documented setup crashes
        the app on startup without this."""
        if v is None or isinstance(v, list):
            return v
        if isinstance(v, str):
            stripped = v.strip()
            if not stripped:
                return []
            if stripped.startswith("["):
                return json.loads(stripped)
            return [origin.strip() for origin in stripped.split(",") if origin.strip()]
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="allow",
    )


    def validate_production_security(self) -> None:
        """Fail closed on obviously unsafe production defaults."""
        if self.ENVIRONMENT.lower() == "production":
            _weak_prefixes = ("super-secret-dev-", "replace-with-", "changeme", "change-me")
            if self.SECRET_KEY.lower().startswith(_weak_prefixes) or len(self.SECRET_KEY) < 32:
                raise ValueError(
                    "Production requires a non-default SECRET_KEY of at least 32 characters "
                    "(placeholder values from .env.example are rejected)."
                )
            if "*" in self.CORS_ORIGINS:
                raise ValueError("Production CORS_ORIGINS cannot contain '*'.")
            if self.CODE_EXECUTION_ENABLED:
                raise ValueError("Arbitrary code execution cannot be enabled in production without an external OS/container sandbox contract.")

settings = Settings()
settings.validate_production_security()
