"""Pydantic schemas for Executive and Technical Reports."""
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from packages.schemas.src.analysis import FindingSchema, EvidenceSchema


class ExecutiveSummarySchema(BaseModel):
    title: str
    period_analyzed: str
    headline_metrics: Dict[str, Any] = Field(default_factory=dict)
    key_findings: List[str] = Field(default_factory=list)
    critical_risks: List[str] = Field(default_factory=list)
    strategic_recommendations: List[str] = Field(default_factory=list)
    executive_summary_markdown: str


class TechnicalReportSchema(BaseModel):
    title: str
    datasets_used: List[Dict[str, Any]] = Field(default_factory=list)
    data_transformations: List[Dict[str, Any]] = Field(default_factory=list)
    sql_queries: List[str] = Field(default_factory=list)
    statistical_tests: List[Dict[str, Any]] = Field(default_factory=list)
    machine_learning_models: List[Dict[str, Any]] = Field(default_factory=list)
    assumptions_and_limitations: List[str] = Field(default_factory=list)
    validation_audit_trail: List[Dict[str, Any]] = Field(default_factory=list)
    reproducibility_manifest_hash: str
    full_markdown: str


class ReportCreate(BaseModel):
    title: str
    project_id: str
    analysis_id: Optional[str] = None
    report_type: str = "executive"  # executive, technical, combined
    custom_notes: Optional[str] = None


class ReportResponse(BaseModel):
    id: str
    project_id: str
    analysis_id: Optional[str] = None
    title: str
    report_type: str
    executive_summary: Optional[ExecutiveSummarySchema] = None
    technical_report: Optional[TechnicalReportSchema] = None
    markdown_content: str
    pdf_artifact_path: Optional[str] = None
    created_at: datetime
