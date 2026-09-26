"""Main API v1 Router."""
from fastapi import APIRouter
from apps.api.src.api.v1.alerts import router as alerts_router
from apps.api.src.api.v1.analysis import router as analysis_router
from apps.api.src.api.v1.auth import router as auth_router
from apps.api.src.api.v1.chat import router as chat_router
from apps.api.src.api.v1.dashboards import router as dashboards_router
from apps.api.src.api.v1.datasets import router as datasets_router
from apps.api.src.api.v1.glossary import router as glossary_router
from apps.api.src.api.v1.investigations import router as investigations_router
from apps.api.src.api.v1.metrics import router as metrics_router
from apps.api.src.api.v1.projects import router as projects_router
from apps.api.src.api.v1.query import router as query_router
from apps.api.src.api.v1.reports import router as reports_router
from apps.api.src.api.v1.settings import router as settings_router
from apps.api.src.api.v1.backups import router as backups_router

api_router = APIRouter()

api_router.include_router(auth_router)
api_router.include_router(projects_router)
api_router.include_router(datasets_router)
api_router.include_router(investigations_router)
api_router.include_router(analysis_router)
api_router.include_router(chat_router)
api_router.include_router(metrics_router)
api_router.include_router(glossary_router)
api_router.include_router(dashboards_router)
api_router.include_router(reports_router)
api_router.include_router(alerts_router)
api_router.include_router(query_router)
api_router.include_router(settings_router)
api_router.include_router(backups_router)

