"""Pydantic schemas for Datasets, Versions, and Profiling."""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from packages.shared.src.enums import ColumnDataType, DatasetFormat, SemanticType


class ColumnDistributionSchema(BaseModel):
    histogram_bins: List[float] = Field(default_factory=list)
    histogram_counts: List[int] = Field(default_factory=list)
    top_categories: Dict[str, int] = Field(default_factory=dict)


class ColumnProfileSchema(BaseModel):
    name: str
    data_type: ColumnDataType
    semantic_type: SemanticType = SemanticType.DIMENSION
    null_count: int = 0
    null_percentage: float = 0.0
    unique_count: int = 0
    cardinality_ratio: float = 0.0
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    mean_value: Optional[float] = None
    median_value: Optional[float] = None
    std_dev: Optional[float] = None
    variance: Optional[float] = None
    mode_value: Optional[Any] = None
    mode_frequency: Optional[int] = None
    quantiles: Dict[str, float] = Field(default_factory=dict)
    outlier_count: int = 0
    distribution: ColumnDistributionSchema = Field(default_factory=ColumnDistributionSchema)
    inferred_pattern: Optional[str] = None


class DataQualityBreakdownSchema(BaseModel):
    overall_score: float = Field(ge=0, le=100, description="Composite score 0-100")
    missing_values_score: float = 100.0
    duplicate_rows_score: float = 100.0
    type_consistency_score: float = 100.0
    outlier_risk_score: float = 100.0
    date_validity_score: float = 100.0
    details: List[str] = Field(default_factory=list)


class RelationshipSchema(BaseModel):
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    relationship_type: str = "many_to_one"  # one_to_one, one_to_many, many_to_one
    # Structural overlap is an observed diagnostic, not calibrated epistemic confidence.
    match_ratio: float | None = None
    detection_method: str = "name_and_subset"


class DatasetProfileSchema(BaseModel):
    dataset_name: str
    version: int
    row_count: int
    column_count: int
    columns: List[ColumnProfileSchema] = Field(default_factory=list)
    potential_primary_keys: List[str] = Field(default_factory=list)
    potential_foreign_keys: List[Dict[str, str]] = Field(default_factory=list)
    duplicate_row_count: int = 0
    data_quality: DataQualityBreakdownSchema
    summary_text: Optional[str] = None
    profiled_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DatasetCreate(BaseModel):
    name: str
    description: Optional[str] = None
    project_id: str
    format: DatasetFormat = DatasetFormat.CSV
    source_type: str = "upload"


class DatasetVersionSchema(BaseModel):
    id: str
    dataset_id: str
    version_number: int
    file_path: str
    file_size_bytes: int
    row_count: int
    column_count: int
    transformation_applied: Optional[str] = None
    created_at: datetime


class DatasetResponse(BaseModel):
    id: str
    project_id: str
    name: str
    description: Optional[str] = None
    current_version: int
    format: DatasetFormat
    row_count: int
    column_count: int
    data_quality_score: Optional[float] = None
    created_at: datetime
    updated_at: datetime
    profile: Optional[DatasetProfileSchema] = None
