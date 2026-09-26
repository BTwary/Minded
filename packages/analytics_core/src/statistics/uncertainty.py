"""Reproducible uncertainty estimation for statistical effect sizes.

This module deliberately separates effect magnitude from hypothesis-test p-values.
Bootstrap intervals use a fixed seed by default so analyses remain reproducible and
fully traceable. Percentile bootstrap is used because it is transparent and broadly
applicable; its limitations are disclosed rather than presenting it as universally
optimal.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy import stats


from .preflight import _to_clean_float_array


def _clean(x: Sequence[float]) -> np.ndarray:
    return _to_clean_float_array(x)


def _validate(n: int, resamples: int) -> None:
    if n < 3:
        raise ValueError("At least 3 observations are required for bootstrap uncertainty.")
    if resamples < 100:
        raise ValueError("At least 100 bootstrap resamples are required.")


def bootstrap_ci(
    statistic: Callable[..., float],
    *samples: Sequence[float],
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 20260908,
) -> dict[str, float | int | str]:
    """Percentile bootstrap CI with reproducible RNG and explicit metadata."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    arrays = tuple(_clean(s) for s in samples)
    _validate(min(len(a) for a in arrays), resamples)
    observed = float(statistic(*arrays))
    rng = np.random.default_rng(seed)
    values = np.empty(resamples, dtype=float)
    for i in range(resamples):
        boot = tuple(a[rng.integers(0, len(a), len(a))] for a in arrays)
        try:
            values[i] = float(statistic(*boot))
        except (ValueError, FloatingPointError, ZeroDivisionError):
            values[i] = np.nan
    values = values[np.isfinite(values)]
    min_valid = max(100, int(resamples * 0.8))
    if len(values) < min_valid:
        raise ValueError(f"Bootstrap unstable: only {len(values)} of {resamples} resamples were valid.")
    alpha = 1.0 - confidence
    lo, hi = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "estimate": observed,
        "lower": float(lo),
        "upper": float(hi),
        "confidence": float(confidence),
        "resamples": int(resamples),
        "valid_resamples": int(len(values)),
        "seed": int(seed),
        "method": "percentile_bootstrap",
        "interpretation": "Percentile bootstrap interval; use with the documented resampling assumptions and do not interpret as proof of causal identification.",
    }


def hedges_g(a: Sequence[float], b: Sequence[float]) -> float:
    aa, bb = _clean(a), _clean(b)
    n1, n2 = len(aa), len(bb)
    if n1 < 2 or n2 < 2:
        raise ValueError("Hedges g requires at least 2 observations per group.")
    s1, s2 = np.var(aa, ddof=1), np.var(bb, ddof=1)
    df = n1 + n2 - 2
    sp = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / df)
    if sp <= 0:
        return 0.0
    d = (np.mean(aa) - np.mean(bb)) / sp
    correction = 1.0 - 3.0 / max(4.0 * df - 1.0, 1.0)
    return float(d * correction)


def rank_biserial(a: Sequence[float], b: Sequence[float]) -> float:
    aa, bb = _clean(a), _clean(b)
    res = stats.mannwhitneyu(aa, bb, alternative="two-sided", method="auto")
    # Probability-of-superiority orientation: A relative to B.
    return float(2.0 * res.statistic / (len(aa) * len(bb)) - 1.0)


def eta_squared(groups: dict[str, Sequence[float]]) -> float:
    arrays = {k: _clean(v) for k, v in groups.items() if len(_clean(v)) >= 2}
    if len(arrays) < 2:
        raise ValueError("Eta-squared requires at least two groups with >=2 observations.")
    pooled = np.concatenate(list(arrays.values()))
    grand = float(np.mean(pooled))
    ss_between = sum(len(v) * (float(np.mean(v)) - grand) ** 2 for v in arrays.values())
    ss_total = float(np.sum((pooled - grand) ** 2))
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def cramers_v(table: Sequence[Sequence[float]]) -> float:
    arr = np.asarray(table, dtype=float)
    chi2, _, _, _ = stats.chi2_contingency(arr, correction=False)
    n = float(arr.sum())
    dim = min(arr.shape[0] - 1, arr.shape[1] - 1)
    return float(np.sqrt(chi2 / (n * dim))) if n > 0 and dim > 0 else 0.0
