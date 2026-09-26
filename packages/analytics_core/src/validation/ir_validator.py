"""Deterministic Semantic IR Validator for Typed Analytical Intent against Semantic World Model."""
from typing import Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

from packages.schemas.src.analysis import (
    AggregationType,
    FilterOperator,
    ObjectiveType,
    TypedAnalyticalIntent,
)
from packages.schemas.src.semantic_graph import (
    MetricAdditivity,
    SemanticWorldModelSchema,
)


class IRValidationError(BaseModel):
    """Structured diagnostic for a semantic or structural IR rejection."""
    code: str
    field: str
    message: str
    severity: str = "ERROR"  # ERROR, WARNING


class IRValidationResult(BaseModel):
    """Result of deterministic validation on TypedAnalyticalIntent."""
    is_valid: bool
    intent_id: str
    errors: List[IRValidationError] = Field(default_factory=list)
    warnings: List[IRValidationError] = Field(default_factory=list)
    resolved_tables: List[str] = Field(default_factory=list)
    resolved_columns: List[str] = Field(default_factory=list)


class AnalyticalIRValidator:
    """
    Deterministic validator enforcing semantic admissibility, symbol existence,
    and relational invariants on LLM-proposed Typed Analytical IRs before compilation.
    """

    def validate(
        self,
        intent: TypedAnalyticalIntent,
        world_model: SemanticWorldModelSchema,
    ) -> IRValidationResult:
        errors: List[IRValidationError] = []
        warnings: List[IRValidationError] = []
        resolved_tables: List[str] = []
        resolved_columns: List[str] = []

        # 1. Structural Validation
        if not intent.intent_id or not intent.intent_id.strip():
            errors.append(IRValidationError(
                code="ERR_MISSING_INTENT_ID",
                field="intent_id",
                message="Intent must have a non-empty unique intent_id."
            ))

        # Build symbol catalog from SWM
        swm_tables: Set[str] = set()
        swm_columns: Dict[str, Set[str]] = {}

        for entity in world_model.entities:
            t_name = getattr(entity, "table_name", None) or getattr(entity, "name", None)
            if t_name:
                t_lower = t_name.lower()
                swm_tables.add(t_lower)
                if t_lower not in swm_columns:
                    swm_columns[t_lower] = set()

                if hasattr(entity, "columns") and entity.columns:
                    for c in entity.columns:
                        c_name = getattr(c, "name", str(c))
                        swm_columns[t_lower].add(c_name.lower())
                elif hasattr(entity, "attributes") and entity.attributes:
                    if getattr(entity, "primary_key", None):
                        swm_columns[t_lower].add(entity.primary_key.lower())
                    for a in entity.attributes:
                        swm_columns[t_lower].add(a.lower())

        # Also capture metrics and time dimensions in column catalogs
        for m in world_model.metrics:
            m_tab = getattr(m, "table_name", "").lower()
            m_col = getattr(m, "column_name", None)
            if m_tab and m_col:
                if m_tab not in swm_columns:
                    swm_columns[m_tab] = set()
                swm_columns[m_tab].add(m_col.lower())

        for t in world_model.time_dimensions:
            t_tab = getattr(t, "table_name", "").lower()
            t_col = getattr(t, "column_name", None)
            if t_tab and t_col:
                if t_tab not in swm_columns:
                    swm_columns[t_tab] = set()
                swm_columns[t_tab].add(t_col.lower())

        def check_column_exists(table: Optional[str], col: Optional[str], field_name: str, err_code: str, label: str):
            if not col:
                return
            t_lower = (table or "").lower()
            c_lower = col.lower()
            if t_lower and t_lower in swm_columns:
                if swm_columns[t_lower] and c_lower not in swm_columns[t_lower]:
                    errors.append(IRValidationError(
                        code=err_code,
                        field=field_name,
                        message=f"{label} column '{col}' does not exist in table '{table}'.",
                    ))
            elif not t_lower and swm_columns:
                # Search across all tables
                found = any(c_lower in cols for cols in swm_columns.values())
                if not found and any(len(cols) > 0 for cols in swm_columns.values()):
                    errors.append(IRValidationError(
                        code=err_code,
                        field=field_name,
                        message=f"{label} column '{col}' does not exist in any known table.",
                    ))

        # 2. Target Metric Validation & Additivity Checks
        metric = intent.target_metric
        metric_table = (metric.table or "").lower()
        metric_col = (metric.column or "").lower()

        if metric_table and swm_tables and metric_table not in swm_tables:
            errors.append(IRValidationError(
                code="ERR_UNKNOWN_METRIC_TABLE",
                field="target_metric.table",
                message=f"Target metric table '{metric.table}' does not exist in Semantic World Model."
            ))
        else:
            if metric_table:
                resolved_tables.append(metric.table)

        check_column_exists(
            metric.table,
            metric.column,
            "target_metric.column",
            "ERR_UNKNOWN_METRIC_COLUMN",
            "Target metric",
        )

        # Check metric definition in SWM
        matched_metric = None
        for m in world_model.metrics:
            m_name = getattr(m, "metric_name", None) or getattr(m, "name", None)
            if m_name and m_name.lower() == metric.name.lower():
                matched_metric = m
                break

        if matched_metric:
            additivity = getattr(matched_metric, "additivity", None)
            if additivity in [MetricAdditivity.NON_ADDITIVE, "NON_ADDITIVE"] and metric.aggregation == AggregationType.SUM:
                errors.append(IRValidationError(
                    code="ERR_INVALID_ADDITIVITY_AGGREGATION",
                    field="target_metric.aggregation",
                    message=f"Metric '{metric.name}' is classified as NON_ADDITIVE (e.g. ratio/percentage); SUM aggregation is invalid. Use MEAN or weighted calculation."
                ))

        if metric_col:
            resolved_columns.append(f"{metric.table or 'default'}.{metric.column}")

        # 3. Unit of Analysis / Grain Validation
        grain = intent.unit_of_analysis
        grain_table = grain.table.lower()
        if swm_tables and grain_table not in swm_tables:
            errors.append(IRValidationError(
                code="ERR_UNKNOWN_GRAIN_TABLE",
                field="unit_of_analysis.table",
                message=f"Unit of analysis table '{grain.table}' does not exist in Semantic World Model."
            ))
        else:
            resolved_tables.append(grain.table)

        for idx, key in enumerate(grain.keys):
            check_column_exists(
                grain.table,
                key,
                f"unit_of_analysis.keys[{idx}]",
                "ERR_UNKNOWN_GRAIN_KEY",
                "Grain",
            )

        # 4. Dimensions Validation
        for idx, dim in enumerate(intent.dimensions):
            dim_tab = (dim.table or "").lower()
            if dim_tab and swm_tables and dim_tab not in swm_tables:
                errors.append(IRValidationError(
                    code="ERR_UNKNOWN_DIMENSION_TABLE",
                    field=f"dimensions[{idx}].table",
                    message=f"Dimension table '{dim.table}' does not exist in Semantic World Model."
                ))
            check_column_exists(
                dim.table,
                dim.column,
                f"dimensions[{idx}].column",
                "ERR_UNKNOWN_DIMENSION_COLUMN",
                "Dimension",
            )
            if dim.column:
                resolved_columns.append(f"{dim.table or 'default'}.{dim.column}")

        # 5. Population Filter Validation
        for idx, filter_ast in enumerate(intent.population):
            f_table = filter_ast.variable.table.lower()
            if swm_tables and f_table not in swm_tables:
                errors.append(IRValidationError(
                    code="ERR_UNKNOWN_FILTER_TABLE",
                    field=f"population[{idx}].variable.table",
                    message=f"Filter table '{filter_ast.variable.table}' does not exist in Semantic World Model."
                ))
            check_column_exists(
                filter_ast.variable.table,
                filter_ast.variable.column,
                f"population[{idx}].variable.column",
                "ERR_UNKNOWN_FILTER_COLUMN",
                "Filter",
            )
            if filter_ast.operator in [FilterOperator.IN, FilterOperator.NOT_IN]:
                if not isinstance(filter_ast.value, list):
                    errors.append(IRValidationError(
                        code="ERR_INVALID_FILTER_VALUE_TYPE",
                        field=f"population[{idx}].value",
                        message=f"Operator '{filter_ast.operator.value}' requires a list of values."
                    ))

        # 6. Causal Intent Validation
        if intent.objective == ObjectiveType.ESTIMATE_CAUSAL_EFFECT or intent.causal_intent:
            if not intent.causal_intent:
                errors.append(IRValidationError(
                    code="ERR_MISSING_CAUSAL_INTENT",
                    field="causal_intent",
                    message="Objective ESTIMATE_CAUSAL_EFFECT requires an explicit CausalIntent specifying treatment, outcome, and DAG."
                ))
            else:
                ci = intent.causal_intent
                check_column_exists(
                    ci.treatment.table,
                    ci.treatment.column,
                    "causal_intent.treatment.column",
                    "ERR_UNKNOWN_TREATMENT_COLUMN",
                    "Treatment",
                )
                check_column_exists(
                    ci.outcome.table,
                    ci.outcome.column,
                    "causal_intent.outcome.column",
                    "ERR_UNKNOWN_OUTCOME_COLUMN",
                    "Outcome",
                )
                if ci.treatment.table.lower() == ci.outcome.table.lower() and ci.treatment.column.lower() == ci.outcome.column.lower():
                    errors.append(IRValidationError(
                        code="ERR_IDENTICAL_TREATMENT_OUTCOME",
                        field="causal_intent",
                        message="Treatment variable and outcome variable cannot be identical."
                    ))

        resolved_tables = list(dict.fromkeys(resolved_tables))
        resolved_columns = list(dict.fromkeys(resolved_columns))

        return IRValidationResult(
            is_valid=(len(errors) == 0),
            intent_id=intent.intent_id,
            errors=errors,
            warnings=warnings,
            resolved_tables=resolved_tables,
            resolved_columns=resolved_columns,
        )
