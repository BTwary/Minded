"""FastAPI Main Application Entrypoint."""
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from apps.api.src.api.v1.router import api_router
from apps.api.src.core.config import settings
from apps.api.src.core.database import init_db
from apps.api.src.services.storage_service import StorageService
import os
import threading

is_production = settings.ENVIRONMENT.lower() == "production"

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Production-grade Autonomous AI Data Analyst Platform with Deterministic Engines",
    version="1.0.0",
    openapi_url=None if is_production else f"{settings.API_V1_STR}/openapi.json",
    docs_url=None if is_production else f"{settings.API_V1_STR}/docs",
    redoc_url=None if is_production else f"{settings.API_V1_STR}/redoc",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept", "X-Request-ID"],
)

# Offline desktop CSP prevents a browser-side bundle from contacting external
# origins even if a future UI dependency accidentally introduces one.
if os.getenv("AAOS_OFFLINE_MODE", "0").lower() in ("1", "true", "yes"):
    from starlette.middleware.base import BaseHTTPMiddleware

    class OfflineContentSecurityPolicyMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):  # type: ignore[no-untyped-def]
            response = await call_next(request)
            response.headers[
                "Content-Security-Policy"
            ] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
            return response

    app.add_middleware(OfflineContentSecurityPolicyMiddleware)

# Initialize database schema
@app.on_event("startup")
def on_startup():
    init_db()
    # Enforce local dataset retention on startup. Stored source/derived files
    # older than DATA_RETENTION_DAYS (180 by default) are removed; analytical
    # metadata/history remains in the SQLite/PostgreSQL system of record.
    StorageService().enforce_retention()

    # Local desktop mode: run the already-implemented durable investigation
    # worker inside the API process so queued investigations do not sit idle.
    # Production deployments should set AAOS_LOCAL_AUTOWORKER=false and run
    # the worker as a separate process/service.
    local_worker_enabled = os.getenv("AAOS_LOCAL_AUTOWORKER", "true").lower() in ("1", "true", "yes")
    if local_worker_enabled and settings.ENVIRONMENT.lower() == "development":
        from packages.analytics_core.src.execution.worker import WorkerSupervisor
        supervisor = WorkerSupervisor(poll_interval_seconds=0.5)
        thread = threading.Thread(target=supervisor.run_loop, name="aaos-local-worker", daemon=True)
        thread.start()
        app.state.aaos_worker = supervisor
        app.state.aaos_worker_thread = thread


# Include API v1 router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Packaged offline desktop mode serves the prebuilt static frontend from the
# same loopback origin as the API. This removes Node/npm from the runtime
# dependency graph and keeps browser calls same-origin. The source checkout
# remains unchanged when AAOS_FRONTEND_ROOT is unset or missing.
_frontend_root = os.getenv("AAOS_FRONTEND_ROOT", "")
if _frontend_root and Path(_frontend_root).is_dir():
    app.mount("/", StaticFiles(directory=_frontend_root, html=True), name="frontend")


@app.get("/")
def root():
    return {
        "status": "online",
        "service": settings.PROJECT_NAME,
        "version": "1.0.0",
        "docs_url": None if is_production else f"{settings.API_V1_STR}/docs",
        "api_v1": settings.API_V1_STR,
        "frontend_url": "http://localhost:3000",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": "1.0.0",
        "ai_provider": settings.AI_PROVIDER,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("apps.api.src.main:app", host="0.0.0.0", port=8000, reload=False)
