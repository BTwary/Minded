"""Deterministic Independent Validation Engine."""
import math
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from packages.schemas.src.validation import (
    ValidationResultSchema,
    ValidationRuleSchema,
    ValidationSummarySchema,
)
from packages.shared.src.enums import ValidationStatus


class IndependentValidator:
    """Production-grade independent validation engine.
    
    Verifies numerical claims, statistical assertions, and business logic
    against raw datasets before presenting results to users.
    """

    def validate_numerical_claim(
        self,
        claim_name: str,
        expected_value: float,
        actual_value: float,
        relative_tolerance: float = 0.01,  # 1% tolerance
        description: Optional[str] = None,
    ) -> ValidationResultSchema:
        """Validate a numerical calculation against deterministic recomputation."""
        if math.isnan(actual_value) or math.isnan(expected_value):
            return ValidationResultSchema(
                rule_name=claim_name,
                rule_type="numerical_tolerance",
                status=ValidationStatus.FAILED,
                expected_value=expected_value,
                actual_value=actual_value,
                difference=None,
                tolerance_used=relative_tolerance,
                passed=False,
                explanation="Numerical claim failed: result is NaN.",
            )

        diff = abs(actual_value - expected_value)
        base = max(abs(actual_value), abs(expected_value), 1e-6)
        rel_diff = diff / base

        passed = rel_diff <= relative_tolerance
        status = ValidationStatus.PASSED if passed else ValidationStatus.FAILED

        explanation = (
            f"PASSED: Recomputed value ({actual_value:.4f}) matches expected ({expected_value:.4f}) within {relative_tolerance*100:.1f}% tolerance (rel_diff={rel_diff:.4f})."
            if passed
            else f"FAILED: Discrepancy detected! Recomputed value ({actual_value:.4f}) differs from expected ({expected_value:.4f}) by {rel_diff*100:.2f}% (tolerance={relative_tolerance*100:.1f}%)."
        )

        return ValidationResultSchema(
            rule_name=claim_name,
            rule_type="numerical_tolerance",
            status=status,
            expected_value=expected_value,
            actual_value=actual_value,
            difference=round(diff, 4),
            tolerance_used=relative_tolerance,
            passed=passed,
            explanation=explanation,
        )

    def validate_proportion_sum(
        self,
        group_shares: Dict[str, float],
        max_allowed_sum: float = 100.0,
        tolerance: float = 0.5,
    ) -> ValidationResultSchema:
        """Validate that component percentage shares do not exceed 100%."""
        total_sum = sum(group_shares.values())
        diff = total_sum - max_allowed_sum
        passed = diff <= tolerance

        status = ValidationStatus.PASSED if passed else ValidationStatus.FAILED
        explanation = (
            f"PASSED: Sum of component shares ({total_sum:.2f}%) is mathematically valid."
            if passed
            else f"FAILED: Component shares sum to {total_sum:.2f}%, which exceeds 100%."
        )

        return ValidationResultSchema(
            rule_name="Component Shares Sum Check",
            rule_type="logic_check",
            status=status,
            expected_value=max_allowed_sum,
            actual_value=round(total_sum, 2),
            difference=round(diff, 2),
            tolerance_used=tolerance,
            passed=passed,
            explanation=explanation,
        )

    def validate_sample_size_sufficiency(
        self,
        sample_size: int,
        minimum_required: int = 30,
    ) -> ValidationResultSchema:
        """Validate that sample size is sufficient for reliable statistical inference."""
        passed = sample_size >= minimum_required
        status = ValidationStatus.PASSED if passed else ValidationStatus.WARNING
        explanation = (
            f"PASSED: Sample size (n={sample_size}) meets statistical reliability threshold (>= {minimum_required})."
            if passed
            else f"WARNING: Low sample size (n={sample_size} < {minimum_required}). Statistical inferences may have wider confidence intervals."
        )

        return ValidationResultSchema(
            rule_name="Sample Size Check",
            rule_type="sample_size_check",
            status=status,
            expected_value=minimum_required,
            actual_value=sample_size,
            difference=sample_size - minimum_required,
            tolerance_used=0.0,
            passed=passed,
            explanation=explanation,
        )

    def compile_validation_summary(
        self, results: List[ValidationResultSchema]
    ) -> ValidationSummarySchema:
        """Aggregate multiple validation checks into a formal summary."""
        total = len(results)
        passed = sum(1 for r in results if r.status == ValidationStatus.PASSED)
        failed = sum(1 for r in results if r.status == ValidationStatus.FAILED)
        warnings = sum(1 for r in results if r.status == ValidationStatus.WARNING)

        overall_status = (
            ValidationStatus.FAILED
            if failed > 0
            else (ValidationStatus.WARNING if warnings > 0 else ValidationStatus.PASSED)
        )

        remediation = []
        if failed > 0:
            remediation.append("Replan analysis: Re-evaluate SQL queries and aggregation filters to resolve numerical discrepancies.")
        if warnings > 0:
            remediation.append("Clarify confidence limitations due to sample size or variance in findings.")

        return ValidationSummarySchema(
            total_checks=total,
            passed_checks=passed,
            failed_checks=failed,
            warning_checks=warnings,
            overall_status=overall_status,
            results=results,
            remediation_suggestions=remediation,
        )
