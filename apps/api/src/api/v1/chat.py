"""Offline-first conversational analyst endpoint.

Natural-language interpretation is deterministic and local. The canonical
InvestigationController performs the real analysis. Optional AI may only
augment presentation after deterministic analysis is complete.
"""
import uuid
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from apps.api.src.core.database import get_db
from apps.api.src.core.security import get_current_user, verify_project_ownership
from apps.api.src.models.entities import Conversation, Message, User
from apps.api.src.services.analysis_service import AnalysisService
from packages.analytics_core.src.engines.local_conversational_analyst import LocalConversationalAnalyst
from packages.schemas.src.analysis import AnalysisCreate
from packages.schemas.src.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat_with_analyst(
    payload: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Interpret locally, execute the real analyst, and return traceable results."""
    verify_project_ownership(payload.project_id, current_user, db)

    conv_id = payload.conversation_id
    if conv_id:
        conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
        if not conv or conv.project_id != payload.project_id:
            # Do not permit cross-project conversation reuse.
            conv_id = None
    if not conv_id:
        conv_id = str(uuid.uuid4())
        conv = Conversation(
            id=conv_id,
            project_id=payload.project_id,
            title=payload.message[:50] + ("..." if len(payload.message) > 50 else ""),
        )
        db.add(conv)
        db.commit()

    user_msg_id = str(uuid.uuid4())
    db.add(Message(
        id=user_msg_id,
        conversation_id=conv_id,
        sender_role="user",
        content=payload.message,
    ))
    db.commit()

    prior_rows = (
        db.query(Message)
        .filter(Message.conversation_id == conv_id, Message.sender_role == "user")
        .order_by(Message.created_at.desc())
        .limit(6)
        .all()
    )
    prior_questions = [row.content for row in reversed(prior_rows) if row.id != user_msg_id]
    canonical_question, followup_resolved = LocalConversationalAnalyst.resolve_followup(payload.message, prior_questions)
    plan = LocalConversationalAnalyst.plan(canonical_question)
    analysis_service = AnalysisService(db)

    analysis_res = analysis_service.execute_analysis(
        AnalysisCreate(
            question=canonical_question,
            project_id=payload.project_id,
            dataset_ids=payload.dataset_ids,
            conversation_id=conv_id,
            context_notes=(
                "Local conversational routing hint; canonical InvestigationController remains authoritative. "
                f"problem_class={plan.problem_class}; user_intent={plan.user_intent}; horizon={plan.target_horizon or 'unspecified'}; "
                f"followup_resolved={followup_resolved}"
            ),
            analysis_mode=payload.analysis_mode,
        ),
        user_id=current_user.id,
    )

    # The reply is a local deterministic presentation of the canonical result.
    # AI_AUGMENTED may add an explanation rewrite inside AnalysisResponse, but
    # direct_answer, evidence, and traceability remain authoritative.
    parts = []
    if plan.explanation:
        parts.append(plan.explanation)
    if analysis_res.direct_answer:
        parts.append(analysis_res.direct_answer)
    if analysis_res.main_finding and analysis_res.main_finding != analysis_res.direct_answer:
        parts.append(analysis_res.main_finding)
    reply_text = "\n\n".join(p for p in parts if p)
    if not reply_text:
        reply_text = "The analytical investigation completed without a supported textual conclusion. Review the evidence and trace for details."

    assistant_msg_id = str(uuid.uuid4())
    db.add(Message(
        id=assistant_msg_id,
        conversation_id=conv_id,
        sender_role="assistant",
        content=reply_text,
        analysis_id=analysis_res.id,
    ))
    db.commit()

    trace_available = any(getattr(e, "calculation_trace", None) is not None for e in analysis_res.evidence)
    return ChatResponse(
        conversation_id=conv_id,
        message_id=assistant_msg_id,
        reply_text=reply_text,
        analysis_id=analysis_res.id,
        interpretation=plan.to_dict(),
        analysis=analysis_res,
        suggested_follow_ups=plan.follow_ups or analysis_res.suggested_followups,
        analysis_mode=analysis_res.ai_mode if analysis_res.ai_mode in {"DETERMINISTIC", "AI_AUGMENTED"} else "DETERMINISTIC",
        trace_available=trace_available,
    )
