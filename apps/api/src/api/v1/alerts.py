"""Autonomous Monitoring and Alerting API Endpoints with Strict Tenancy & Multi-Project Isolation."""
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from apps.api.src.core.database import get_db
from apps.api.src.core.security import (
    get_current_user,
    get_user_authorized_project_ids,
    verify_project_ownership,
)
from apps.api.src.models.entities import AlertEvent, AlertRule, User
from packages.schemas.src.alert import AlertEventSchema, AlertRuleCreate, AlertRuleResponse

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("/rules", response_model=List[AlertRuleResponse])
def list_alert_rules(
    project_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List monitoring alert rules strictly isolated to authorized user projects."""
    if project_id:
        verify_project_ownership(project_id, current_user, db)
        allowed_ids = [project_id]
    else:
        allowed_ids = get_user_authorized_project_ids(current_user, db)

    rules = db.query(AlertRule).filter(AlertRule.project_id.in_(allowed_ids)).all()
    return [
        AlertRuleResponse(
            id=r.id,
            project_id=r.project_id,
            name=r.name,
            metric_name=r.metric_name,
            dataset_id=r.dataset_id,
            threshold_pct_change=r.threshold_pct_change,
            frequency=r.frequency,
            severity=r.severity,
            auto_investigate=r.auto_investigate,
            is_active=r.is_active,
            last_evaluated_at=r.last_evaluated_at,
            created_at=r.created_at,
        )
        for r in rules
    ]


@router.post("/rules", response_model=AlertRuleResponse)
def create_alert_rule(
    payload: AlertRuleCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new autonomous monitoring rule with project authorization."""
    verify_project_ownership(payload.project_id, current_user, db)
    rule = AlertRule(
        project_id=payload.project_id,
        name=payload.name,
        metric_name=payload.metric_name,
        dataset_id=payload.dataset_id,
        threshold_pct_change=payload.threshold_pct_change,
        frequency=payload.frequency.value,
        severity=payload.severity.value,
        auto_investigate=payload.auto_investigate,
    )
    db.add(rule)
    db.commit()
    db.refresh(rule)
    return AlertRuleResponse(
        id=rule.id,
        project_id=rule.project_id,
        name=rule.name,
        metric_name=rule.metric_name,
        dataset_id=rule.dataset_id,
        threshold_pct_change=rule.threshold_pct_change,
        frequency=rule.frequency,
        severity=rule.severity,
        auto_investigate=rule.auto_investigate,
        is_active=rule.is_active,
        last_evaluated_at=rule.last_evaluated_at,
        created_at=rule.created_at,
    )


@router.get("/events", response_model=List[AlertEventSchema])
def list_alert_events(
    project_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List triggered anomaly alert events strictly isolated to authorized user projects."""
    if project_id:
        verify_project_ownership(project_id, current_user, db)
        allowed_ids = [project_id]
    else:
        allowed_ids = get_user_authorized_project_ids(current_user, db)

    events = (
        db.query(AlertEvent)
        .join(AlertRule, AlertEvent.rule_id == AlertRule.id)
        .filter(AlertRule.project_id.in_(allowed_ids))
        .order_by(AlertEvent.triggered_at.desc())
        .all()
    )
    return [
        AlertEventSchema(
            id=e.id,
            rule_id=e.rule_id,
            title=e.title,
            severity=e.severity,
            description=e.description,
            current_value=e.current_value,
            expected_value=e.expected_value,
            deviation_percentage=e.deviation_percentage,
            affected_dimensions=e.affected_dimensions_json or {},
            investigation_analysis_id=e.investigation_analysis_id,
            evidence_summary=e.evidence_summary,
            is_resolved=e.is_resolved,
            triggered_at=e.triggered_at,
        )
        for e in events
    ]


@router.post("/evaluate")
def evaluate_alerts(
    project_id: str = "proj-default",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Trigger continuous autonomous monitoring evaluation for all active project rules."""
    verify_project_ownership(project_id, current_user, db)
    from packages.analytics_core.src.monitoring.continuous_engine import ContinuousMonitorEngine
    monitor = ContinuousMonitorEngine(db)
    triggered = monitor.evaluate_project_rules(project_id)
    return {
        "status": "success",
        "project_id": project_id,
        "triggered_count": len(triggered),
        "triggered_events": triggered,
    }
