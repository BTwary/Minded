"""Empirical calibration diagnostics for probability-like predictions."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CalibrationReport:
    sample_size: int
    brier_score: float
    expected_calibration_error: float
    max_calibration_error: float
    bins: int
    status: str
    rationale: str

    def to_dict(self):
        return self.__dict__.copy()


def empirical_calibration(y_true, probabilities, bins: int = 10) -> CalibrationReport:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], np.clip(p[mask], 0.0, 1.0)
    n = len(y)
    if n == 0:
        return CalibrationReport(0, float("nan"), float("nan"), float("nan"), bins, "INSUFFICIENT_EVIDENCE", "No paired outcomes/probabilities were supplied.")
    brier = float(np.mean((p-y)**2))
    edges = np.linspace(0.0, 1.0, bins+1)
    ece = 0.0
    mce = 0.0
    used = 0
    for i in range(bins):
        lo, hi = edges[i], edges[i+1]
        sel = (p >= lo) & ((p < hi) if i < bins-1 else (p <= hi))
        if not np.any(sel):
            continue
        used += 1
        gap = abs(float(np.mean(p[sel])) - float(np.mean(y[sel])))
        ece += float(np.mean(sel)) * gap
        mce = max(mce, gap)
    status = "CALIBRATED" if ece <= 0.05 and mce <= 0.15 else "MIS_CALIBRATED"
    return CalibrationReport(n, brier, float(ece), float(mce), used, status, f"Empirical calibration over {used} non-empty bins; Brier={brier:.4f}, ECE={ece:.4f}, MCE={mce:.4f}.")
