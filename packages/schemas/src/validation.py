"""Pydantic schemas for Independent Validation Engine."""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from packages.shared.src.enums import ValidationStatus


class ValidationRuleSchema(BaseModel):
    name: str
    rule_type: str  # numerical_tolerance, row_count_check, range_check, logic_check, schema_check
    expected_value: Any
    tolerance: Optional[float] = 0.01  # e.g., 1% relative tolerance
    description: Optional[str] = None


class ValidationResultSchema(BaseModel):
    rule_name: str
    rule_type: str
    status: ValidationStatus
    expected_value: Any
    actual_value: Any
    difference: Optional[float] = None
    tolerance_used: Optional[float] = None
    passed: bool
    explanation: str
    recalculation_details: Optional[Dict[str, Any]] = None
    checked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ValidationSummarySchema(BaseModel):
    total_checks: int
    passed_checks: int
    failed_checks: int
    warning_checks: int
    overall_status: ValidationStatus
    results: List[ValidationResultSchema] = Field(default_factory=list)
    remediation_suggestions: List[str] = Field(default_factory=list)
