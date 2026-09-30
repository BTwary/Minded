"""Deterministic investigation questions from a dataset profile."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Union

from packages.schemas.src.dataset import ColumnProfileSchema, DatasetProfileSchema
from packages.shared.src.enums import SemanticType

_METRIC_TYPES = {
    SemanticType.METRIC,
    SemanticType.CURRENCY,
    SemanticType.PERCENTAGE,
}
_TIME_TYPES = {SemanticType.TIMESTAMP}
_DIM_TYPES = {
    SemanticType.DIMENSION,
    SemanticType.CATEGORICAL,
    SemanticType.GEOGRAPHIC,
}


def _as_profile(profile: Union[DatasetProfileSchema, Dict[str, Any]]) -> DatasetProfileSchema:
    if isinstance(profile, DatasetProfileSchema):
        return profile
    return DatasetProfileSchema.model_validate(profile)


def _name(col: ColumnProfileSchema) -> str:
    return col.name


def _pick(columns: Iterable[ColumnProfileSchema], types: set) -> List[ColumnProfileSchema]:
    return [c for c in columns if c.semantic_type in types]


def synthesize_questions(
    profile: Union[DatasetProfileSchema, Dict[str, Any]],
    *,
    max_questions: int = 3,
    dataset_label: Optional[str] = None,
) -> List[str]:
    """Build a short, ordered list of investigation questions.

    No LLM. Uses profiled types, quality scores, and time/metric presence so
    ingest-and-run autonomy does not require a human-authored prompt.
    """
    p = _as_profile(profile)
    label = dataset_label or p.dataset_name or "this dataset"
    cols = list(p.columns)
    metrics = _pick(cols, _METRIC_TYPES)
    times = _pick(cols, _TIME_TYPES)
    dims = [c for c in _pick(cols, _DIM_TYPES) if c.cardinality_ratio < 0.5]
    questions: List[str] = []

    metric_name = _name(metrics[0]) if metrics else None
    time_name = _name(times[0]) if times else None
    dim_name = _name(dims[0]) if dims else None

    if metric_name and time_name:
        questions.append(
            f"Why did {metric_name} change over {time_name} in {label}, and which segments explain the movement?"
        )
    elif metric_name and dim_name:
        questions.append(
            f"Which {dim_name} segments drive variation in {metric_name} in {label}?"
        )
    elif metric_name:
        questions.append(
            f"What are the primary drivers, concentrations, and anomalies in {metric_name} within {label}?"
        )
    else:
        questions.append(
            f"What are the most important patterns, anomalies, and data-quality risks in {label}?"
        )

    if dim_name and metric_name and time_name:
        questions.append(
            f"Is the {metric_name} trend in {label} robust across {dim_name}, or does aggregation hide a reversal?"
        )

    quality = p.data_quality
    if quality and quality.overall_score < 85:
        questions.append(
            f"How do missing values, duplicates, or outliers in {label} (quality score {quality.overall_score:.0f}/100) affect the main findings?"
        )

    # Deduplicate while preserving order.
    seen = set()
    unique: List[str] = []
    for q in questions:
        if q not in seen:
            seen.add(q)
            unique.append(q)
    return unique[: max(1, max_questions)]
