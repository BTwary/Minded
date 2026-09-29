"""AI Provider & Platform Settings API Endpoints for Dual-Mode Operations."""
import time
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from apps.api.src.ai.providers.factory import get_ai_config, get_ai_provider, is_ai_enabled, set_ai_config
from apps.api.src.core.security import get_current_user
from apps.api.src.models.entities import User

router = APIRouter(prefix="/settings", tags=["settings"])


class AIConfigRequest(BaseModel):
    enabled: Optional[bool] = None
    provider: Optional[str] = "none"
    model: Optional[str] = "auto"
    api_key: Optional[str] = None
    base_url: Optional[str] = None




class FeedbackRequest(BaseModel):
    category: str = "general"
    message: str
    rating: Optional[int] = None
    include_diagnostics: bool = False


class AITestRequest(BaseModel):
    provider: str
    model: Optional[str] = "auto"
    api_key: Optional[str] = None
    base_url: Optional[str] = None


from packages.analytics_core.src.security.ssrf import assert_safe_endpoint, SSRFSecurityError


@router.get("/ai")
def get_current_ai_settings(current_user: User = Depends(get_current_user)):
    """Retrieve active AI provider and connection metadata for Dual-Mode operations."""
    cfg = get_ai_config(user_id=current_user.id)
    enabled = bool(cfg.get("enabled", False))
    prov = str(cfg.get("provider", "none")).lower()
    raw_key = cfg.get("api_key", "")
    masked_key = f"{raw_key[:4]}...{raw_key[-4:]}" if len(raw_key) > 8 else ("***" if raw_key else "None configured")
    
    active_mode = "AI_AUGMENTED" if (enabled and prov not in ["none", "mock", "offline"]) else "DETERMINISTIC"
    
    if active_mode == "AI_AUGMENTED":
        status_msg = f"AI Augmented Mode Active (Provider: {prov.title()})"
    else:
        status_msg = "Deterministic Mode Active (100% Free / Zero AI API)"

    return {
        "enabled": enabled,
        "active_mode": active_mode,
        "provider": prov,
        "model": cfg.get("model", "auto"),
        "base_url": cfg.get("base_url", ""),
        "api_key_masked": masked_key,
        "is_configured": bool(raw_key or prov in ["ollama", "mock", "none"]),
        "status_message": status_msg,
        "supported_providers": [
            {
                "id": "none",
                "name": "Deterministic Engine (100% Free / Zero AI API)",
                "models": ["deterministic-kernel-v2"],
                "requires_api_key": False,
                "default_base_url": "In-Process DuckDB & Scipy",
                "description": "Zero external API calls. Runs hypotheses, EIG planning, DuckDB execution, and Bayesian updates deterministically.",
            },
            {
                "id": "gemini",
                "name": "Google Gemini",
                "models": ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
                "requires_api_key": True,
                "default_base_url": "Google AI Studio",
                "description": "Fast multimodal reasoning and structured hypothesis proposals.",
            },
            {
                "id": "openai",
                "name": "OpenAI",
                "models": ["gpt-4o", "gpt-4o-mini", "o1", "o3-mini"],
                "requires_api_key": True,
                "default_base_url": "https://api.openai.com/v1",
                "description": "High capability reasoning and semantic natural-language interpretation.",
            },
            {
                "id": "claude",
                "name": "Anthropic Claude",
                "models": ["claude-3-7-sonnet-20250219", "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"],
                "requires_api_key": True,
                "default_base_url": "https://api.anthropic.com/v1",
                "description": "Deep analytical reasoning and nuanced business context synthesis.",
            },
            {
                "id": "groq",
                "name": "Groq Cloud (Fast LPU Inference)",
                "models": ["llama-3.3-70b-versatile", "deepseek-r1-distill-llama-70b", "mixtral-8x7b-32768"],
                "requires_api_key": True,
                "default_base_url": "https://api.groq.com/openai/v1",
                "description": "Ultra-low latency open models.",
            },
            {
                "id": "ollama",
                "name": "Local Ollama (100% Offline / Local GPU / $0)",
                "models": ["qwen2.5-coder:7b", "llama3.2:3b", "deepseek-r1:8b", "mistral:7b"],
                "requires_api_key": False,
                "default_base_url": "http://localhost:11434",
                "description": "Self-hosted local models running directly on your machine.",
            },
            {
                "id": "custom",
                "name": "Custom / OpenRouter / DeepSeek / vLLM",
                "models": ["deepseek-chat", "deepseek-reasoner", "anthropic/claude-3.5-sonnet", "custom-model"],
                "requires_api_key": True,
                "default_base_url": "https://api.deepseek.com/v1",
                "description": "Any OpenAI-compatible API endpoint.",
            },
        ],
    }


@router.post("/ai")
def update_ai_settings(req: AIConfigRequest, current_user: User = Depends(get_current_user)):
    """Update active AI provider and credentials in real-time."""
    try:
        set_ai_config(
            user_id=current_user.id,
            enabled=req.enabled,
            provider=req.provider,
            model=req.model,
            api_key=req.api_key,
            base_url=req.base_url,
        )
    except SSRFSecurityError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err

    mode_name = "AI Augmented" if is_ai_enabled(user_id=current_user.id) else "Deterministic (Free)"
    return {
        "status": "success",
        "message": f"AI settings updated: Active Mode is '{mode_name}' with provider '{req.provider}'.",
        "config": get_current_ai_settings(current_user=current_user),
    }


@router.post("/ai/test")
def test_ai_connection(req: AITestRequest, current_user: User = Depends(get_current_user)):
    """Test live AI provider connection and latency."""
    if req.base_url:
        allow_loop = req.provider.lower() == "ollama" or "ollama" in req.base_url
        try:
            assert_safe_endpoint(req.base_url, allow_loopback=allow_loop, label="AI provider base URL")
        except SSRFSecurityError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err

    start = time.time()
    try:
        prov = req.provider.lower()
        if prov in ["none", "mock", "deterministic"]:
            from apps.api.src.ai.providers.mock import NoneAIProvider
            p = NoneAIProvider()
        elif prov in ["claude", "anthropic"]:
            from apps.api.src.ai.providers.claude import ClaudeProvider
            p = ClaudeProvider(
                api_key=req.api_key or "",
                model=req.model if req.model and req.model != "auto" else "claude-3-7-sonnet-20250219",
                base_url=req.base_url if req.base_url else "https://api.anthropic.com/v1",
            )

        elif prov in ["gemini", "google"]:
            from apps.api.src.ai.providers.gemini import GeminiProvider
            p = GeminiProvider(
                api_key=req.api_key or "",
                model=req.model if req.model and req.model != "auto" else "gemini-2.0-flash",
            )
        elif prov == "ollama":
            from apps.api.src.ai.providers.ollama import OllamaProvider
            p = OllamaProvider(
                base_url=req.base_url or "http://localhost:11434",
                model=req.model if req.model and req.model != "auto" else "qwen2.5-coder:7b",
            )
        else:
            from apps.api.src.ai.providers.openai_compat import OpenAICompatibleProvider
            p = OpenAICompatibleProvider(
                api_key=req.api_key or "test-key",
                base_url=req.base_url or "https://api.openai.com/v1",
                model=req.model if req.model and req.model != "auto" else "gpt-4o",
            )

        resp = p.generate(prompt="Respond with only the word OK.", max_tokens=10)
        latency = round((time.time() - start) * 1000, 1)

        if isinstance(resp, str) and '"error"' in resp:
            return {
                "status": "error",
                "message": resp,
                "latency_ms": latency,
            }

        return {
            "status": "success",
            "message": f"Connection verified successfully in {latency}ms!",
            "latency_ms": latency,
            "sample_response": str(resp).strip()[:50],
        }
    except Exception as e:
        latency = round((time.time() - start) * 1000, 1)
        return {
            "status": "error",
            "message": f"Connection failed: {str(e)}",
            "latency_ms": latency,
        }


@router.get("/infrastructure")
def get_infrastructure_status(current_user: User = Depends(get_current_user)):
    """Retrieve full Bring-Your-Own-Infrastructure (BYOI) status matrix and cost/ownership transparency."""
    from packages.analytics_core.src.providers.manager import InfrastructureManager
    return InfrastructureManager.get_infrastructure_matrix()


@router.post("/feedback")
def submit_local_feedback(req: FeedbackRequest, current_user: User = Depends(get_current_user)):
    """Store feedback locally by default; never uploads it implicitly."""
    if not req.message.strip():
        raise HTTPException(status_code=422, detail="Feedback message cannot be empty.")
    if req.rating is not None and not 1 <= req.rating <= 5:
        raise HTTPException(status_code=422, detail="Rating must be between 1 and 5.")
    from packages.analytics_core.src.platform_local_first import ensure_local_layout
    root = ensure_local_layout()["feedback"]
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "user_id": current_user.id,
        "category": req.category[:64],
        "message": req.message[:5000],
        "rating": req.rating,
        "upload_enabled": settings.FEEDBACK_UPLOAD_ENABLED,
    }
    if req.include_diagnostics:
        payload["diagnostics"] = {
            "environment": settings.ENVIRONMENT,
            "ai_enabled": settings.AI_ENABLED,
            "ai_provider": settings.AI_PROVIDER if settings.AI_ENABLED else "none",
            "storage_provider": settings.STORAGE_PROVIDER,
        }
    path = root / "feedback.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return {
        "status": "stored_locally",
        "upload_performed": False,
        "message": "Feedback saved locally. It is not transmitted unless feedback upload is explicitly enabled.",
        "path": str(path),
    }
