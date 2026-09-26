"""Pydantic schemas for Semantic Layer and Business Glossary."""
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class BusinessMetricCreate(BaseModel):
    name: str
    display_name: str
    description: str
    sql_formula: str
    project_id: str
    unit: Optional[str] = None
    category: Optional[str] = "financial"
    dimensions: List[str] = Field(default_factory=list)
    filters: Optional[str] = None
    synonyms: List[str] = Field(default_factory=list)


class BusinessMetricResponse(BaseModel):
    id: str
    project_id: str
    name: str
    display_name: str
    description: str
    sql_formula: str
    unit: Optional[str] = None
    category: Optional[str] = None
    dimensions: List[str] = Field(default_factory=list)
    filters: Optional[str] = None
    synonyms: List[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class GlossaryTermCreate(BaseModel):
    term: str
    definition: str
    project_id: str
    category: Optional[str] = "general"
    synonyms: List[str] = Field(default_factory=list)
    related_metrics: List[str] = Field(default_factory=list)
    related_columns: List[str] = Field(default_factory=list)


class GlossaryTermResponse(BaseModel):
    id: str
    project_id: str
    term: str
    definition: str
    category: Optional[str] = None
    synonyms: List[str] = Field(default_factory=list)
    related_metrics: List[str] = Field(default_factory=list)
    related_columns: List[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
