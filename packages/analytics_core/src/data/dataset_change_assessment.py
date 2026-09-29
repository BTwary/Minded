"""Autonomous dataset-version change assessment for offline AA-OS investigations.

The assessment is deliberately deterministic and separate from persistence/UI.  It
builds on the canonical DatasetDiff output and answers a narrower question:
"Did the dataset change materially, and could that change be relevant to the
current analytical question?"

It does not claim causality.  A relevant dataset change is a reason to inspect the
refresh/change path, not evidence that the refresh caused the observed outcome.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from packages.analytics_core.src.data.dataset_diff import DatasetDiff, compare_datasets


ROW_CHANGE_RATE_THRESHOLD = 0.05
MISSINGNESS_DELTA_THRESHOLD = 0.05
NUMERIC_MEAN_RELATIVE_CHANGE_THRESHOLD = 0.10


@dataclass(frozen=True)
class DatasetChangeAssessment:
    before_version: int
    after_version: int
    status: str
    material_change: bool = False
    question_relevant: bool = False
    relevance: str = "NONE"  # NONE, LOW, MEDIUM, HIGH
    needs_investigation: bool = False
    identity_mode: str = "UNKNOWN"
    key_columns: List[str] = field(default_factory=list)
    affected_columns: List[str] = field(default_factory=list)
    question_relevant_columns: List[str] = field(default_factory=list)
    material_signals: List[str] = field(default_factory=list)
    analytical_impacts: List[str] = field(default_factory=list)
    time_coverage_change: Optional[Dict[str, Any]] = None
    diff: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normalise_columns(columns: Iterable[str]) -> List[str]:
    return sorted({str(c) for c in columns if c is not None and str(c).strip()})


def _time_coverage(df: pd.DataFrame, time_column: Optional[str]) -> Optional[Dict[str, Any]]:
    if not time_column or time_column not in df.columns:
        return None
    values = pd.to_datetime(df[time_column], errors="coerce").dropna()
    if values.empty:
        return {"available": False}
    return {
        "available": True,
        "min": values.min().isoformat(),
        "max": values.max().isoformat(),
        "valid_count": int(values.size),
    }


def _coverage_changed(
    before: Optional[Dict[str, Any]],
    after: Optional[Dict[str, Any]],
) -> bool:
    if not before or not after:
        return False
    return before.get("min") != after.get("min") or before.get("max") != after.get("max")



def infer_stable_key_columns(semantic: Any, dataset_name: str, df: pd.DataFrame) -> List[str]:
    """Return a stable key only when existing semantic evidence proves one.

    This helper intentionally refuses to guess from arbitrary column order.  It
    reuses the world-model verified grain or explicit dataframe key metadata;
    ambiguous candidates are left unresolved so the diff engine can fall back to
    multiset semantics without inventing record identity.
    """
    world_model = getattr(semantic, "world_model", None)
    verified = getattr(world_model, "verified_grains", {}) if world_model is not None else {}
    candidates = list(verified.get(dataset_name, []) or []) if isinstance(verified, dict) else []
    if len(candidates) == 1 and str(candidates[0]) in df.columns:
        return [str(candidates[0])]

    attrs = getattr(df, "attrs", {}) or {}
    declared = attrs.get("primary_keys") or attrs.get("candidate_keys") or []
    if len(declared) == 1:
        item = declared[0]
        if isinstance(item, (list, tuple)):
            cols = [str(c) for c in item]
        else:
            cols = [str(item)]
        if cols and all(c in df.columns for c in cols):
            return cols
    return []

def assess_dataset_change(
    before: pd.DataFrame,
    after: pd.DataFrame,
    *,
    before_version: int,
    after_version: int,
    question_columns: Optional[Sequence[str]] = None,
    time_column: Optional[str] = None,
    key_columns: Optional[Sequence[str]] = None,
    task: Optional[str] = None,
) -> DatasetChangeAssessment:
    """Assess whether a version change is material and relevant to a question.

    The thresholds are conservative defaults and are explicit in the output so
    callers can audit why a change was classified as material.
    """
    if before_version < 1 or after_version < 1 or before_version == after_version:
        raise ValueError("before_version and after_version must be distinct positive integers")

    diff: DatasetDiff = compare_datasets(before, after, key_columns=key_columns)
    affected: set[str] = set()
    material_signals: List[str] = []
    impacts: List[str] = []
    notes: List[str] = list(diff.notes)

    affected.update(diff.schema_added)
    affected.update(diff.schema_removed)
    affected.update(diff.schema_type_changed.keys())
    affected.update(diff.missingness_delta.keys())
    # DatasetDiff records numeric summary statistics for every common numeric
    # column.  Only treat a column as *affected* here when at least one summary
    # statistic actually changed; otherwise an unchanged target metric could make
    # an unrelated version change appear question-relevant.
    for column, stats in diff.numeric_distribution_delta.items():
        numeric_changed = any(
            abs(float(stats.get(before_key, 0.0)) - float(stats.get(after_key, 0.0))) > 1e-12
            for before_key, after_key in (
                ("mean_before", "mean_after"),
                ("median_before", "median_after"),
                ("std_before", "std_after"),
            )
        )
        if numeric_changed:
            affected.add(str(column))
    affected.update(diff.categorical_changes.keys())

    row_rate = abs(len(after) - len(before)) / max(len(before), 1)
    if row_rate >= ROW_CHANGE_RATE_THRESHOLD:
        material_signals.append(
            f"Row count changed by {row_rate:.1%}, meeting the {ROW_CHANGE_RATE_THRESHOLD:.0%} materiality threshold."
        )
        impacts.append("population_or_exposure")

    for column, delta in diff.missingness_delta.items():
        if abs(float(delta)) >= MISSINGNESS_DELTA_THRESHOLD:
            material_signals.append(
                f"Missingness for '{column}' changed by {float(delta):+.1%}, meeting the "
                f"{MISSINGNESS_DELTA_THRESHOLD:.0%} threshold."
            )
            impacts.append("missingness_bias")

    for column, stats in diff.numeric_distribution_delta.items():
        rel = abs(float(stats.get("mean_relative_change", 0.0)))
        if rel >= NUMERIC_MEAN_RELATIVE_CHANGE_THRESHOLD:
            material_signals.append(
                f"Mean of '{column}' changed materially (relative shift {rel:.1%})."
            )
            impacts.append("metric_or_covariate_distribution")

    if diff.schema_added or diff.schema_removed or diff.schema_type_changed:
        material_signals.append("Schema changed between dataset versions.")
        impacts.append("schema_semantics")

    if diff.categorical_changes:
        material_signals.append("Categorical value support changed between dataset versions.")
        impacts.append("population_composition")

    before_coverage = _time_coverage(before, time_column)
    after_coverage = _time_coverage(after, time_column)
    coverage_change = None
    if _coverage_changed(before_coverage, after_coverage):
        coverage_change = {"before": before_coverage, "after": after_coverage}
        material_signals.append("Temporal coverage changed between dataset versions.")
        impacts.append("temporal_scope")

    relevant_columns = set(_normalise_columns(question_columns or []))
    question_relevant_columns = sorted(affected & relevant_columns)
    question_relevant = bool(question_relevant_columns)

    global_relevance = bool(
        row_rate >= ROW_CHANGE_RATE_THRESHOLD
        or coverage_change is not None
        or "population_or_exposure" in impacts
    )
    structural_relevance = bool(diff.schema_added or diff.schema_removed or diff.schema_type_changed)

    material = bool(material_signals)
    if question_relevant:
        relevance = "HIGH" if material else "LOW"
    elif global_relevance and material:
        relevance = "MEDIUM"
    elif structural_relevance and material:
        relevance = "LOW"
    else:
        relevance = "NONE"

    question_relevant = question_relevant or global_relevance
    needs_investigation = material and question_relevant

    # De-duplicate impacts while keeping deterministic order.
    impacts = sorted(set(impacts))

    if needs_investigation:
        status = "RELEVANT_MATERIAL_CHANGE"
        notes.append(
            "Dataset-version change is potentially relevant to the current question. "
            "This is a review trigger, not a causal claim that the data refresh caused the outcome."
        )
    elif material:
        status = "MATERIAL_CHANGE_NOT_DIRECTLY_RELEVANT"
    else:
        status = "NO_MATERIAL_CHANGE"

    return DatasetChangeAssessment(
        before_version=int(before_version),
        after_version=int(after_version),
        status=status,
        material_change=material,
        question_relevant=bool(question_relevant),
        relevance=relevance,
        needs_investigation=needs_investigation,
        identity_mode=str(diff.row_identity_mode),
        key_columns=list(diff.key_columns),
        affected_columns=sorted(affected),
        question_relevant_columns=question_relevant_columns,
        material_signals=material_signals,
        analytical_impacts=impacts,
        time_coverage_change=coverage_change,
        diff=diff.to_dict(),
        notes=notes,
    )
