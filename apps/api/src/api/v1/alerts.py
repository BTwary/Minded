"""Local persisted alert rules, event evidence, and lifecycle operations."""
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from apps.api.src.core.database import get_db
from apps.api.src.core.security import (
    get_current_user,
    get_user_authorized_project_ids,
    verify_project_ownership,
)
from apps.api.src.models.entities import AlertEvent, AlertRule, Dataset, DatasetVersion, User
from packages.schemas.src.alert import AlertEventSchema, AlertRuleCreate, AlertRuleResponse

router = APIRouter(prefix="/alerts", tags=["alerts"])


class SnoozeRequest(BaseModel):
    minutes: int = Field(ge=1, le=10080)


def _rule_response(rule: AlertRule) -> AlertRuleResponse:
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
        last_evaluation_status=rule.last_evaluation_status,
        last_evaluation_trigger=rule.last_evaluation_trigger,
        created_at=rule.created_at,
    )


def _event_response(db: Session, event: AlertEvent, rule: AlertRule) -> AlertEventSchema:
    dataset = db.query(Dataset).filter(
        Dataset.id == event.dataset_id,
        Dataset.project_id == event.project_id,
    ).first() if event.dataset_id else None
    version = db.query(DatasetVersion).filter(
        DatasetVersion.id == event.dataset_version_id,
        DatasetVersion.dataset_id == event.dataset_id,
        DatasetVersion.version_number == event.dataset_version,
    ).first() if event.dataset_version_id and event.dataset_id else None

    status = event.lifecycle_status
    if not status:
        status = "resolved" if event.is_resolved else "active"
    dimensions = event.affected_dimensions_json or {}
    return AlertEventSchema(
        id=event.id,
        project_id=event.project_id,
        rule_id=event.rule_id,
        rule_name=rule.name,
        metric_name=event.metric_name or rule.metric_name,
        threshold_pct_change=dimensions.get("threshold_pct_change", rule.threshold_pct_change),
        dataset_id=event.dataset_id or rule.dataset_id,
        dataset_name=dataset.name if dataset else None,
        dataset_version=event.dataset_version,
        dataset_version_id=event.dataset_version_id,
        dataset_fingerprint=event.dataset_fingerprint,
        source_available=bool(
            dataset
            and version
            and (
                not event.dataset_fingerprint
                or version.content_hash == event.dataset_fingerprint
            )
        ),
        lifecycle_status=status,
        acknowledged_at=event.acknowledged_at,
        snoozed_until=event.snoozed_until,
        baseline_period=event.baseline_period or dimensions.get("period_a"),
        comparison_period=event.comparison_period or dimensions.get("period_b"),
        title=event.title,
        severity=event.severity,
        description=event.description,
        current_value=event.current_value,
        expected_value=event.expected_value,
        deviation_percentage=event.deviation_percentage,
        affected_dimensions=dimensions,
        investigation_analysis_id=event.investigation_analysis_id,
        evidence_summary=event.evidence_summary,
        is_resolved=event.is_resolved,
        triggered_at=event.triggered_at,
    )


def _authorized_events_query(
    db: Session,
    user: User,
    project_id: Optional[str],
    rule_id: Optional[str] = None,
):
    if project_id:
        verify_project_ownership(project_id, user, db)
        allowed_ids = [project_id]
    else:
        allowed_ids = get_user_authorized_project_ids(user, db)
    query = (
        db.query(AlertEvent, AlertRule)
        .join(AlertRule, AlertEvent.rule_id == AlertRule.id)
        .filter(AlertRule.project_id.in_(allowed_ids))
    )
    if rule_id:
        query = query.filter(AlertEvent.rule_id == rule_id)
    return query


def _get_authorized_event(db: Session, user: User, event_id: str) -> tuple[AlertEvent, AlertRule]:
    row = (
        _authorized_events_query(db, user, None)
        .filter(AlertEvent.id == event_id)
        .first()
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Alert event not found.")
    return row


@router.get("/rules", response_model=list[AlertRuleResponse])
def list_alert_rules(
    project_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if project_id:
        verify_project_ownership(project_id, current_user, db)
        allowed_ids = [project_id]
    else:
        allowed_ids = get_user_authorized_project_ids(current_user, db)
    rules = db.query(AlertRule).filter(AlertRule.project_id.in_(allowed_ids)).order_by(AlertRule.created_at.desc()).all()
    return [_rule_response(rule) for rule in rules]


@router.post("/rules", response_model=AlertRuleResponse)
def create_alert_rule(
    payload: AlertRuleCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    verify_project_ownership(payload.project_id, current_user, db)
    dataset = db.query(Dataset).filter(
        Dataset.id == payload.dataset_id,
        Dataset.project_id == payload.project_id,
    ).first()
    if dataset is None:
        raise HTTPException(status_code=400, detail="Alert rule dataset must exist in the selected project.")
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
    return _rule_response(rule)


@router.get("/events", response_model=list[AlertEventSchema])
def list_alert_events(
    project_id: Optional[str] = None,
    rule_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rows = _authorized_events_query(db, current_user, project_id, rule_id).order_by(
        AlertEvent.triggered_at.desc()
    ).all()
    return [_event_response(db, event, rule) for event, rule in rows]


@router.get("/events/{event_id}", response_model=AlertEventSchema)
def get_alert_event(
    event_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    event, rule = _get_authorized_event(db, current_user, event_id)
    return _event_response(db, event, rule)


@router.post("/events/{event_id}/acknowledge", response_model=AlertEventSchema)
def acknowledge_alert_event(
    event_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    event, rule = _get_authorized_event(db, current_user, event_id)
    if event.is_resolved:
        raise HTTPException(status_code=409, detail="A resolved alert cannot be acknowledged.")
    event.lifecycle_status = "acknowledged"
    event.acknowledged_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(event)
    return _event_response(db, event, rule)


@router.post("/events/{event_id}/resolve", response_model=AlertEventSchema)
def resolve_alert_event(
    event_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    event, rule = _get_authorized_event(db, current_user, event_id)
    event.is_resolved = True
    event.lifecycle_status = "resolved"
    db.commit()
    db.refresh(event)
    return _event_response(db, event, rule)


@router.post("/events/{event_id}/snooze", response_model=AlertEventSchema)
def snooze_alert_event(
    event_id: str,
    payload: SnoozeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    event, rule = _get_authorized_event(db, current_user, event_id)
    if event.is_resolved:
        raise HTTPException(status_code=409, detail="A resolved alert cannot be snoozed.")
    event.lifecycle_status = "snoozed"
    event.snoozed_until = datetime.now(timezone.utc) + timedelta(minutes=payload.minutes)
    db.commit()
    db.refresh(event)
    return _event_response(db, event, rule)


@router.get("/status")
def get_monitoring_status(
    project_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if project_id:
        verify_project_ownership(project_id, current_user, db)
        allowed_ids = [project_id]
    else:
        allowed_ids = get_user_authorized_project_ids(current_user, db)
    rules = db.query(AlertRule).filter(AlertRule.project_id.in_(allowed_ids)).all()
    last_rule = max(
        (rule for rule in rules if rule.last_evaluated_at),
        key=lambda rule: rule.last_evaluated_at,
        default=None,
    )
    last_success = max(
        (
            rule for rule in rules
            if rule.last_evaluated_at and rule.last_evaluation_status in {"NO_CHANGE", "TRIGGERED", "DUPLICATE"}
        ),
        key=lambda rule: rule.last_evaluated_at,
        default=None,
    )
    return {
        "scheduler_active": False,
        "evaluation_mode": last_rule.last_evaluation_trigger if last_rule else None,
        "last_evaluation_at": last_rule.last_evaluated_at if last_rule else None,
        "last_evaluation_status": last_rule.last_evaluation_status if last_rule else None,
        "last_successful_evaluation_at": last_success.last_evaluated_at if last_success else None,
        "active_rule_count": sum(bool(rule.is_active) for rule in rules),
    }


@router.post("/evaluate")
def evaluate_alerts(
    project_id: str = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    verify_project_ownership(project_id, current_user, db)
    return _evaluate(project_id, "manual", db)


def _evaluate(project_id: str, trigger: str, db: Session, dataset_id: Optional[str] = None) -> dict[str, Any]:
    from packages.analytics_core.src.monitoring.continuous_engine import ContinuousMonitorEngine

    outcomes = ContinuousMonitorEngine(db).evaluate_project_rules(
        project_id, trigger=trigger, dataset_id=dataset_id
    )
    failures = [outcome for outcome in outcomes if outcome["status"] == "FAILED"]
    successful_statuses = {"NO_CHANGE", "TRIGGERED", "DUPLICATE"}
    qualified = any(outcome["status"] not in successful_statuses for outcome in outcomes)
    return {
        "status": "failed" if failures else "qualified" if qualified else "success",
        "project_id": project_id,
        "trigger": trigger,
        "evaluated_rule_count": len(outcomes),
        "triggered_count": sum(outcome["status"] == "TRIGGERED" for outcome in outcomes),
        "outcomes": outcomes,
        "scheduler_active": False,
    }
