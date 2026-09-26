"""Pydantic schemas for Autonomous Monitoring and Alerts."""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from packages.shared.src.enums import AlertSeverity, ScheduleFrequency


class AlertRuleCreate(BaseModel):
    name: str
    project_id: str
    metric_id: Optional[str] = None
    metric_name: str
    dataset_id: str
    condition_sql: Optional[str] = None
    threshold_pct_change: Optional[float] = 15.0
    threshold_absolute_value: Optional[float] = None
    frequency: ScheduleFrequency = ScheduleFrequency.DAILY
    severity: AlertSeverity = AlertSeverity.HIGH
    auto_investigate: bool = True
    notification_channels: List[str] = Field(default_factory=lambda: ["dashboard"])


class AlertRuleResponse(BaseModel):
    id: str
    project_id: str
    name: str
    metric_name: str
    dataset_id: str
    threshold_pct_change: Optional[float]
    frequency: ScheduleFrequency
    severity: AlertSeverity
    auto_investigate: bool
    is_active: bool
    last_evaluated_at: Optional[datetime] = None
    created_at: datetime


class AlertEventSchema(BaseModel):
    id: str
    rule_id: str
    title: str
    severity: AlertSeverity
    description: str
    current_value: float
    expected_value: float
    deviation_percentage: float
    affected_dimensions: Dict[str, Any] = Field(default_factory=dict)
    investigation_analysis_id: Optional[str] = None
    evidence_summary: Optional[str] = None
    is_resolved: bool = False
    triggered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ScheduledJobSchema(BaseModel):
    id: str
    project_id: str
    name: str
    job_type: str  # monitor, dataset_refresh, daily_analysis
    cron_expression: str
    last_run_at: Optional[datetime] = None
    next_run_at: Optional[datetime] = None
    is_active: bool = True
