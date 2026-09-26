"""Pydantic schemas for Dashboards and Visualization Widgets."""
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from packages.shared.src.enums import ChartType


class ChartSpecSchema(BaseModel):
    chart_type: ChartType
    title: str
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    color_column: Optional[str] = None
    aggregation: Optional[str] = None
    plotly_figure_json: Optional[Dict[str, Any]] = None
    data_rows: Optional[List[Dict[str, Any]]] = None
    explanation: Optional[str] = None


class WidgetSchema(BaseModel):
    id: str
    title: str
    widget_type: str  # chart, metric_card, table, text
    width: int = 6  # 1 to 12 grid columns
    height: int = 4
    chart_spec: Optional[ChartSpecSchema] = None
    sql_query: Optional[str] = None
    dataset_version: Optional[str] = None
    refresh_rate_seconds: Optional[int] = 0


class DashboardCreate(BaseModel):
    title: str
    description: Optional[str] = None
    project_id: str
    widgets: List[WidgetSchema] = Field(default_factory=list)
    filters: Optional[Dict[str, Any]] = None


class DashboardResponse(BaseModel):
    id: str
    project_id: str
    title: str
    description: Optional[str] = None
    widgets: List[WidgetSchema] = Field(default_factory=list)
    filters: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime
