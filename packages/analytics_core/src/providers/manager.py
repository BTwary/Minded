"""Unified Infrastructure Provider Manager for AA-OS BYOI Architecture."""
import os
from typing import Any, Dict, List, Optional
from packages.analytics_core.src.platform_local_first import ensure_local_layout, resolve_ai_enabled
from packages.analytics_core.src.providers.storage import (
    BaseStorageProvider,
    LocalStorageProvider,
    S3StorageProvider,
    GCSStorageProvider,
)
from packages.analytics_core.src.providers.ai import (
    BaseAIProvider,
    MockAIProvider,
    NoneAIProvider,
    OllamaAIProvider,
    OpenAIProvider,
    GeminiProvider,
)


class InfrastructureManager:
    """Central registry and factory for pluggable BYOI providers."""

    _storage_provider: Optional[BaseStorageProvider] = None
    _ai_provider: Optional[BaseAIProvider] = None

    @classmethod
    def get_storage_provider(cls) -> BaseStorageProvider:
        if cls._storage_provider is None:
            storage_type = os.getenv("STORAGE_PROVIDER", "local").lower()
            if storage_type == "s3":
                cls._storage_provider = S3StorageProvider(
                    bucket_name=os.getenv("AWS_S3_BUCKET", "aa-os-datasets"),
                    region_name=os.getenv("AWS_REGION", "us-east-1"),
                )
            elif storage_type == "gcs":
                cls._storage_provider = GCSStorageProvider(
                    bucket_name=os.getenv("GCS_BUCKET", "aa-os-datasets"),
                )
            else:
                # BUGFIX: this previously defaulted to a repo-relative
                # "./data_store" independently of apps/api/src/core/config.py's
                # DATA_STORAGE_DIR setting, which already migrated to the
                # local-first, per-OS user-data directory in
                # platform_local_first.ensure_local_layout(). Two different
                # defaults for the same setting meant this provider (the one
                # that actually writes ingested dataset files) silently kept
                # writing into the checked-out repository tree -- exactly the
                # runtime-data contamination the local-first migration and
                # scripts/release_gate.py's data_store cleanliness check were
                # meant to eliminate. Route through the same helper so there
                # is exactly one source of truth for this path.
                cls._storage_provider = LocalStorageProvider(
                    base_dir=os.getenv("DATA_STORAGE_DIR", str(ensure_local_layout()["storage"])),
                )
        return cls._storage_provider

    @classmethod
    def get_ai_provider(cls) -> BaseAIProvider:
        # SECURITY INVARIANT: AI_ENABLED is the single authoritative on/off switch.
        # It is evaluated on EVERY call — before the cache — so a runtime change
        # from AI_ENABLED=true to AI_ENABLED=false is immediately effective.
        # The cache stores the live provider constructed when AI is enabled, but
        # the cache is NEVER consulted when AI is currently disabled.
        if not resolve_ai_enabled():
            # Always return a fresh NoneAIProvider when AI is disabled; never
            # serve a previously cached live provider.
            return NoneAIProvider()
        # AI is currently enabled — consult / populate the per-process cache.
        if cls._ai_provider is None or isinstance(cls._ai_provider, NoneAIProvider):
            provider_type = os.getenv("AI_PROVIDER", "none").lower()
            if provider_type == "ollama":
                cls._ai_provider = OllamaAIProvider(
                    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                    model_name=os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b"),
                )
            elif provider_type == "openai":
                cls._ai_provider = OpenAIProvider(
                    api_key=os.getenv("OPENAI_API_KEY", ""),
                    model_name=os.getenv("OPENAI_MODEL", "gpt-4o"),
                    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                )
            elif provider_type == "gemini":
                cls._ai_provider = GeminiProvider(
                    api_key=os.getenv("GEMINI_API_KEY", ""),
                    model_name=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
                )
            elif provider_type == "mock":
                cls._ai_provider = MockAIProvider()
            else:
                cls._ai_provider = NoneAIProvider()
        return cls._ai_provider

    @classmethod
    def set_storage_provider(cls, provider: BaseStorageProvider) -> None:
        cls._storage_provider = provider

    @classmethod
    def set_ai_provider(cls, provider: BaseAIProvider) -> None:
        cls._ai_provider = provider

    @classmethod
    def get_infrastructure_matrix(cls) -> Dict[str, Any]:
        """Return real-time diagnostic matrix of all connected infrastructure layers."""
        storage = cls.get_storage_provider()
        ai = cls.get_ai_provider()

        storage_ok, storage_msg = storage.test_connection()
        ai_ok, ai_msg = ai.test_connection()

        db_url = os.getenv("DATABASE_URL", "sqlite:///./autonomous_analyst.db")
        db_type = "PostgreSQL (Supabase/Cloud)" if "postgres" in db_url else "SQLite (Local Free Mode)"

        return {
            "mode": "LOCAL_FREE" if (storage.is_local and ai.is_local) else "CONNECTED_BYOI",
            "layers": [
                {
                    "name": "Database (System of Record)",
                    "category": "database",
                    "provider": "PostgreSQL" if "postgres" in db_url else "SQLite",
                    "status": "connected",
                    "is_local": "sqlite" in db_url,
                    "owner": "You (User-Controlled)",
                    "estimated_cost": "$0.00 (Local / Supabase Free Tier)",
                    "message": f"Connected to {db_type}",
                },
                {
                    "name": "Object Storage (Raw Datasets)",
                    "category": "storage",
                    "provider": storage.provider_type.upper(),
                    "status": "connected" if storage_ok else "error",
                    "is_local": storage.is_local,
                    "owner": "You (User-Controlled)",
                    "estimated_cost": "$0.00 (Local Filesystem / S3 Free Tier)" if storage.is_local else "Direct Provider Cost",
                    "message": storage_msg,
                },
                {
                    "name": "AI Reasoning & Hypotheses",
                    "category": "ai",
                    "provider": ai.provider_type.upper(),
                    "status": "connected" if ai_ok else "warning",
                    "is_local": ai.is_local,
                    "owner": "You (BYOM - Bring Your Own Model)",
                    "estimated_cost": "$0.00 (Local / Free Model)" if ai.is_local else "Direct API Cost",
                    "message": ai_msg,
                },
                {
                    "name": "OLAP Analytical Engine",
                    "category": "analytics",
                    "provider": "DuckDB + Apache Arrow + Polars",
                    "status": "connected",
                    "is_local": True,
                    "owner": "In-Process Embedded (AA-OS Core)",
                    "estimated_cost": "$0.00 (Local SIMD In-Memory Vector Execution)",
                    "message": "DuckDB in-process vector execution active (Zero Cloud Query Costs)",
                },
                {
                    "name": "Python Sandbox Isolation",
                    "category": "sandbox",
                    "provider": "OS Process Isolation + AST Gate",
                    "status": "connected",
                    "is_local": True,
                    "owner": "AA-OS Kernel",
                    "estimated_cost": "$0.00",
                    "message": "Subprocess isolation with AST scanner and filesystem/network disarming",
                },
            ],
        }
