"""Pydantic schemas for deterministic/local AA-OS conversational analysis."""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field
from packages.schemas.src.analysis import AnalysisResponse


class MessageSchema(BaseModel):
    id: str
    conversation_id: str
    sender_role: str
    content: str
    analysis_id: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChatRequest(BaseModel):
    message: str
    project_id: str
    conversation_id: Optional[str] = None
    dataset_ids: Optional[List[str]] = None
    analysis_mode: Literal["DETERMINISTIC", "AI_AUGMENTED"] = "DETERMINISTIC"


class ChatResponse(BaseModel):
    conversation_id: str
    message_id: str
    reply_text: str
    analysis_id: Optional[str] = None
    interpretation: Optional[Dict[str, Any]] = None
    analysis: Optional[AnalysisResponse] = None
    suggested_follow_ups: List[str] = Field(default_factory=list)
    analysis_mode: Literal["DETERMINISTIC", "AI_AUGMENTED"] = "DETERMINISTIC"
    trace_available: bool = False
