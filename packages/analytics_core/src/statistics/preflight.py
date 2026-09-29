"""Deterministic statistical-method preflight checks.

These checks run before SciPy/statsmodels calls.  They turn numerical pathologies
(constant ranges, empty groups, sparse tables, degenerate targets, etc.) into
explicit admissibility outcomes instead of warning-driven execution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StatisticalPreflight:
    method: str
    status: str  # ALLOWED / BLOCKED / NOT_APPLICABLE
    reasons: Tuple[str, ...] = ()
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        return self.status == "ALLOWED"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "status": self.status,
            "reasons": list(self.reasons),
            "diagnostics": dict(self.diagnostics),
        }


def _to_clean_float_series(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        s = values
    elif isinstance(values, np.ndarray) and np.issubdtype(values.dtype, np.number):
        return pd.Series(values, dtype=float)
    else:
        s = pd.Series(values)
    if s.empty:
        return pd.Series([], dtype=float)
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    elif pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    else:
        cleaned = s.astype(str).str.strip().str.replace(r"[,$€£¥₹%\s]", "", regex=True)
        return pd.to_numeric(cleaned, errors="coerce")


def _to_clean_float_array(values: Any) -> np.ndarray:
    s = _to_clean_float_series(values)
    v = s.to_numpy(dtype=float)
    return v[np.isfinite(v)]


def _finite(values: Sequence[float]) -> np.ndarray:
    return _to_clean_float_array(values)


def _range_scale(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    scale = max(1.0, float(np.nanmax(np.abs(arr))))
    return float(np.finfo(float).eps * 100.0 * scale)


def numeric_pair(x: Sequence[float], y: Sequence[float], *, method: str, min_n: int = 3) -> StatisticalPreflight:
    sx = _to_clean_float_series(x)
    sy = _to_clean_float_series(y)
    mask = np.isfinite(sx) & np.isfinite(sy)
    xx = sx[mask].to_numpy(dtype=float)
    yy = sy[mask].to_numpy(dtype=float)
    n = len(xx)
    reasons = []
    if n < min_n:
        reasons.append(f"insufficient_finite_pairs:{min_n}")
        return StatisticalPreflight(method, "BLOCKED", tuple(reasons), {"n": int(n)})
    xx = xx[:n]
    yy = yy[:n]
    x_range = float(np.ptp(xx))
    y_range = float(np.ptp(yy))
    x_std = float(np.std(xx))
    y_std = float(np.std(yy))
    if x_range <= _range_scale(xx) or x_std <= _range_scale(xx):
        reasons.append("exposure_constant_or_near_constant")
    if y_range <= _range_scale(yy) or y_std <= _range_scale(yy):
        reasons.append("outcome_constant_or_near_constant")
    return StatisticalPreflight(
        method,
        "BLOCKED" if reasons else "ALLOWED",
        tuple(reasons),
        {"n": int(n), "x_std": x_std, "y_std": y_std, "x_range": x_range, "y_range": y_range},
    )


def independent_groups(groups: Mapping[Any, Iterable[float]], *, method: str, min_group_n: int = 2, min_groups: int = 2) -> StatisticalPreflight:
    cleaned = {k: _finite(list(v)) for k, v in groups.items()}
    usable = {k: v for k, v in cleaned.items() if len(v) >= min_group_n}
    reasons = []
    if len(usable) < min_groups:
        reasons.append(f"insufficient_groups:{min_groups}")
    constant_groups = [str(k) for k, v in usable.items() if np.ptp(v) <= _range_scale(v)]
    if constant_groups and len(constant_groups) == len(usable):
        reasons.append("constant_or_near_constant_group_outcome")
    diagnostics = {
        "group_count_total": len(cleaned),
        "group_count_usable": len(usable),
        "group_sizes": {str(k): int(len(v)) for k, v in usable.items()},
        "constant_groups": constant_groups,
    }
    return StatisticalPreflight(method, "BLOCKED" if reasons else "ALLOWED", tuple(reasons), diagnostics)


def contingency(table: Any, *, method: str, min_shape: Tuple[int, int] = (2, 2)) -> StatisticalPreflight:
    arr = np.asarray(table, dtype=float)
    reasons = []
    if arr.ndim != 2 or arr.shape[0] < min_shape[0] or arr.shape[1] < min_shape[1]:
        reasons.append("contingency_table_below_minimum_shape")
    if arr.ndim == 2:
        if not np.isfinite(arr).all():
            reasons.append("non_finite_contingency_counts")
        if np.any(arr < 0):
            reasons.append("negative_contingency_counts")
        if arr.size and (np.sum(arr, axis=0) == 0).any():
            reasons.append("empty_contingency_column")
        if arr.size and (np.sum(arr, axis=1) == 0).any():
            reasons.append("empty_contingency_row")
    return StatisticalPreflight(
        method,
        "BLOCKED" if reasons else "ALLOWED",
        tuple(reasons),
        {"shape": list(arr.shape) if arr.ndim == 2 else None},
    )


def binary_target(series: pd.Series, *, method: str, min_n: int = 40) -> StatisticalPreflight:
    values = series.dropna()
    reasons = []
    unique = int(values.nunique())
    if len(values) < min_n:
        reasons.append(f"sample_size_below_minimum:{min_n}")
    if unique != 2:
        reasons.append("target_not_binary")
    diagnostics = {"n": int(len(values)), "unique": unique}
    if unique == 2:
        diagnostics["class_counts"] = {str(k): int(v) for k, v in values.value_counts().to_dict().items()}
    return StatisticalPreflight(method, "BLOCKED" if reasons else "ALLOWED", tuple(reasons), diagnostics)


def forecast_target(time_series: pd.Series, *, method: str, min_n: int = 12) -> StatisticalPreflight:
    y = _to_clean_float_array(time_series)
    reasons = []
    if len(y) < min_n:
        reasons.append(f"insufficient_history:{min_n}")
    if len(y) and np.ptp(y) <= _range_scale(y):
        reasons.append("target_constant_or_near_constant")
    return StatisticalPreflight(method, "BLOCKED" if reasons else "ALLOWED", tuple(reasons), {"n": int(len(y)), "range": float(np.ptp(y)) if len(y) else 0.0})
