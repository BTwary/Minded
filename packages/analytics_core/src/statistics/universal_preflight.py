"""Universal statistical admissibility checks shared by every analytical family."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DesignPreflight:
    allowed: bool
    reasons: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)


def inspect_design(df: pd.DataFrame, *, target: str | None = None, grouping: str | None = None, time_col: str | None = None, weights: str | None = None) -> DesignPreflight:
    reasons: list[str] = []
    d: dict[str, Any] = {"rows": int(len(df)), "columns": int(len(df.columns))}
    if len(df) == 0:
        return DesignPreflight(False, ("empty_analysis_population",), d)
    if target and target in df.columns:
        s = df[target]
        d["target_nonnull"] = int(s.notna().sum())
        if s.notna().sum() == 0:
            reasons.append("target_all_missing")
        if pd.api.types.is_numeric_dtype(s):
            vals = pd.to_numeric(s, errors="coerce").dropna().to_numpy(float)
            if len(vals) and np.ptp(vals) <= max(1.0, np.max(np.abs(vals), initial=0))*np.finfo(float).eps*100:
                reasons.append("target_constant_or_near_constant")
        elif s.dropna().nunique() <= 1:
            reasons.append("target_single_category")
    if grouping and grouping in df.columns:
        g = df[grouping].dropna()
        d["groups"] = int(g.nunique())
        if g.nunique() < 2:
            reasons.append("grouping_has_fewer_than_two_levels")
        sizes = g.value_counts(dropna=True)
        d["group_sizes"] = {str(k): int(v) for k, v in sizes.to_dict().items()}
        if len(sizes) and int(sizes.min()) < 2:
            reasons.append("group_contains_fewer_than_two_observations")
    if weights and weights in df.columns:
        w = pd.to_numeric(df[weights], errors="coerce")
        if w.notna().any() and (w < 0).any():
            reasons.append("negative_weights")
        if w.notna().any() and float(w.sum()) <= 0:
            reasons.append("nonpositive_total_weight")
    if time_col and time_col in df.columns:
        t = pd.to_datetime(df[time_col], errors="coerce")
        d["time_nonnull"] = int(t.notna().sum())
        if t.dropna().nunique() < 2:
            reasons.append("insufficient_time_points")
    # Explicit duplicate-row notice: duplicates may be legitimate, but an
    # identifier-like perfect duplicate is a data-quality condition that must
    # be surfaced before an analysis silently treats repeated records as units.
    d["duplicate_rows"] = int(df.duplicated().sum())
    return DesignPreflight(not bool(reasons), tuple(dict.fromkeys(reasons)), d)
