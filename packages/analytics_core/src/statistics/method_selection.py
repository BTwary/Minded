"""Scientific statistical method selection and inference.

Deterministic, data-aware policy layer for inferential statistics. It selects methods
based on the estimand, design, data characteristics, and assumption diagnostics, and
returns an auditable analysis plan without silently switching methods.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from packages.analytics_core.src.statistics.preflight import (
    numeric_pair,
    independent_groups,
    contingency,
    _to_clean_float_array,
    _to_clean_float_series,
)


@dataclass(frozen=True)
class StatisticalMethodDecision:
    problem: str
    method: str
    alternative_methods: Tuple[str, ...] = ()
    estimand: str = ""
    assumptions: Tuple[str, ...] = ()
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    limitations: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem": self.problem,
            "method": self.method,
            "alternative_methods": list(self.alternative_methods),
            "estimand": self.estimand,
            "assumptions": list(self.assumptions),
            "diagnostics": self.diagnostics,
            "rationale": self.rationale,
            "limitations": list(self.limitations),
        }


def _clean(x: Sequence[float]) -> np.ndarray:
    return _to_clean_float_array(x)


def _safe_shapiro(x: np.ndarray) -> Optional[float]:
    if len(x) < 3:
        return None
    if len(x) > 5000:
        # Shapiro's implementation is calibrated for <=5000. For larger samples,
        # use a robust descriptive screen rather than pretending a p-value is exact.
        return None
    try:
        return float(stats.shapiro(x).pvalue)
    except Exception:
        return None


def _normality_screen(x: np.ndarray) -> Dict[str, Any]:
    x = _clean(x)
    if len(x) < 3:
        return {"n": len(x), "shapiro_p": None, "skew": None, "normality_supported": False}
    skew = float(stats.skew(x, bias=False)) if len(x) >= 8 else None
    p = _safe_shapiro(x)
    normal = bool((p is not None and p >= 0.05) or (p is None and skew is not None and abs(skew) < 1.0))
    return {"n": len(x), "shapiro_p": p, "skew": skew, "normality_supported": normal}


def _variance_screen(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    if len(a) < 2 or len(b) < 2:
        return {"levene_p": None, "equal_variance_supported": False}
    try:
        p = float(stats.levene(a, b, center="median").pvalue)
    except Exception:
        p = None
    return {"levene_p": p, "equal_variance_supported": bool(p is not None and p >= 0.05)}


def _effect_cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    denom_df = n1 + n2 - 2
    if denom_df <= 0:
        return float("nan")
    pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / denom_df)
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else 0.0


def _effect_hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    d = _effect_cohens_d(a, b)
    df = len(a) + len(b) - 2
    if not np.isfinite(d) or df <= 1:
        return d
    correction = 1.0 - 3.0 / (4.0 * df - 1.0)
    return float(d * correction)


def select_two_group(a: Sequence[float], b: Sequence[float], *, paired: bool = False, alpha: float = 0.05) -> StatisticalMethodDecision:
    if paired:
        if len(a) != len(b):
            return StatisticalMethodDecision(
                problem="two_group_paired",
                method="INSUFFICIENT_DATA",
                limitations=("paired samples require equal-length aligned observations",),
                rationale="Cannot select a paired method without aligned pairs.",
            )
        aa, bb = _clean_pair(a, b)
        if len(aa) == 0:
            return StatisticalMethodDecision(
                problem="two_group_paired",
                method="INSUFFICIENT_DATA",
                limitations=("paired samples require equal-length aligned observations",),
                rationale="Cannot select a paired method without aligned pairs.",
            )
        d = aa - bb
        # Paired inference is determined by the within-pair differences, not by
        # whether the two marginal series happen to have correlation/other shape
        # diagnostics that flag a constant side. A valid non-zero constant paired
        # difference is still estimable; the only degenerate case is a zero/near-zero
        # difference vector.
        if len(d) and np.ptp(d) <= max(1e-12, np.finfo(float).eps * 100 * max(1.0, float(np.max(np.abs(d))))):
            # A non-zero constant paired shift is estimable, but a t-test has
            # undefined standard error. Use the distribution-free signed-rank
            # framework and surface the degeneracy explicitly rather than
            # silently reporting a fabricated p-value.
            method = "Wilcoxon signed-rank" if not np.allclose(d, 0.0) else "NOT_APPLICABLE"
            if method == "NOT_APPLICABLE":
                return StatisticalMethodDecision(problem="two_group_paired", method=method, diagnostics={"n_pairs": len(d), "difference_range": float(np.ptp(d))}, limitations=("paired_difference_zero",), rationale="All paired differences are zero; no inferential contrast exists.")
            return StatisticalMethodDecision(
                problem="two_group_paired", method=method, alternative_methods=("Paired t-test",),
                estimand="Paired rank/location difference",
                assumptions=("paired observations are correctly aligned", "pairs are independent across subjects"),
                diagnostics={"n_pairs": len(d), "difference_range": float(np.ptp(d)), "constant_nonzero_difference": True, "alpha": alpha},
                rationale="All paired differences have the same non-zero sign/magnitude; use a signed-rank method and do not manufacture a zero-variance t-test.",
                limitations=("Constant paired differences provide no empirical dispersion; inferential interpretation is limited.",),
            )
        normality = _normality_screen(d)
        method = "Paired t-test" if normality["normality_supported"] and len(d) >= 5 else "Wilcoxon signed-rank"
        return StatisticalMethodDecision(
            problem="two_group_paired",
            method=method,
            alternative_methods=("Wilcoxon signed-rank", "Paired t-test"),
            estimand="Mean paired difference" if method == "Paired t-test" else "Paired rank/location difference",
            assumptions=("paired observations are correctly aligned", "pairs are independent across subjects"),
            diagnostics={"difference_normality": normality, "n_pairs": len(d), "alpha": alpha},
            rationale="Paired design was explicitly supplied; method depends on the distribution of within-pair differences.",
            limitations=("Normality tests have limited power at small n; inspect effect size and diagnostic plots.",),
        )

    aa, bb = _clean(a), _clean(b)
    if len(aa) < 3 or len(bb) < 3:
        return StatisticalMethodDecision(
            problem="two_group_independent",
            method="INSUFFICIENT_DATA",
            limitations=("minimum 3 observations per group for inferential screening",),
            rationale="Insufficient observations for a defensible inferential method selection.",
        )
    pf = independent_groups({"a": aa, "b": bb}, method="two_group_independent", min_group_n=3, min_groups=2)
    if "constant_or_near_constant_group_outcome" in pf.reasons:
        return StatisticalMethodDecision(problem="two_group_independent", method="NOT_APPLICABLE", diagnostics=pf.to_dict(), limitations=tuple(pf.reasons), rationale="Both groups are constant; there is no within-group variability for a meaningful inferential comparison, so no SciPy inferential call was made.")
    na, nb = _normality_screen(aa), _normality_screen(bb)
    variance = _variance_screen(aa, bb)
    robust_choice = not (na["normality_supported"] and nb["normality_supported"])
    if robust_choice:
        method = "Mann-Whitney U"
        rationale = "At least one group fails the normality screen; use a rank-based method rather than forcing a t-test."
        estimand = "Stochastic dominance / rank shift"
        alternatives = ("Welch t-test",)
    else:
        # Welch is the safe default for independent means. A non-significant
        # variance test does not prove equal variances, especially at modest n,
        # so AA-OS does not turn a failed Levene test into an equal-variance
        # assumption. Pooled Student t remains available only as an explicit
        # alternative when equal-variance design knowledge exists.
        method = "Welch t-test"
        rationale = "Both groups pass the normality screen; use Welch's unequal-variance t-test as the defensible default rather than inferring equal variances from a non-significant variance test."
        estimand = "Difference in group means"
        alternatives = ("Mann-Whitney U", "Student t-test")
    return StatisticalMethodDecision(
        problem="two_group_independent",
        method=method,
        alternative_methods=alternatives,
        estimand=estimand,
        assumptions=("independent observations", "numeric outcome",),
        diagnostics={"group_a_normality": na, "group_b_normality": nb, "variance": variance, "alpha": alpha, "cohens_d": _effect_cohens_d(aa, bb), "hedges_g": _effect_hedges_g(aa, bb)},
        rationale=rationale,
        limitations=("Group independence is a design assumption and cannot be established from values alone.",),
    )


def select_multi_group(groups: Dict[str, Sequence[float]], *, alpha: float = 0.05) -> StatisticalMethodDecision:
    cleaned = {k: _clean(v) for k, v in groups.items() if len(_clean(v)) >= 2}
    if len(cleaned) < 3:
        return StatisticalMethodDecision(
            problem="multi_group_independent",
            method="TWO_GROUP_OR_INSUFFICIENT",
            limitations=("at least 3 groups with >=2 observations required for omnibus multi-group selection",),
            rationale="Multi-group selection requires at least three estimable groups.",
        )
    preflight = independent_groups(cleaned, method="multi_group_independent", min_group_n=2, min_groups=3)
    if not preflight.allowed:
        return StatisticalMethodDecision(problem="multi_group_independent", method="NOT_APPLICABLE", diagnostics=preflight.to_dict(), limitations=tuple(preflight.reasons), rationale="Omnibus group inference was blocked by data-shape preflight; no SciPy omnibus/variance test was made.")
    normals = {k: _normality_screen(v) for k, v in cleaned.items()}
    try:
        lev_p = float(stats.levene(*cleaned.values(), center="median").pvalue)
    except Exception:
        lev_p = None
    all_normal = all(v["normality_supported"] for v in normals.values())
    equal_var = bool(lev_p is not None and lev_p >= alpha)
    if not all_normal:
        method = "Kruskal-Wallis"
        estimand = "Distributional rank differences across groups"
        rationale = "At least one group fails the normality screen; use a rank-based omnibus test."
    elif equal_var:
        method = "One-way ANOVA"
        estimand = "Between-group mean differences"
        rationale = "All groups pass the normality screen and robust variance screening does not reject homogeneity."
    else:
        method = "Welch ANOVA"
        estimand = "Between-group mean differences without equal-variance assumption"
        rationale = "Normality is acceptable but variance homogeneity is not; use Welch's heteroskedastic omnibus test."
    return StatisticalMethodDecision(
        problem="multi_group_independent",
        method=method,
        alternative_methods=("Kruskal-Wallis", "One-way ANOVA", "Welch ANOVA"),
        estimand=estimand,
        assumptions=("independent observations", "groups are meaningfully defined",),
        diagnostics={"group_count": len(cleaned), "group_normality": normals, "levene_p": lev_p, "equal_variance_supported": equal_var, "alpha": alpha},
        rationale=rationale,
        limitations=("Omnibus significance does not identify which groups differ; post-hoc tests must be multiplicity-controlled.",),
    )


def select_categorical(contingency: pd.DataFrame, *, alpha: float = 0.05) -> StatisticalMethodDecision:
    table = np.asarray(contingency, dtype=float)
    preflight = __import__("packages.analytics_core.src.statistics.preflight", fromlist=["contingency"]).contingency(table, method="categorical_association")
    if not preflight.allowed:
        return StatisticalMethodDecision(problem="categorical_association", method="NOT_APPLICABLE", diagnostics=preflight.to_dict(), limitations=tuple(preflight.reasons), rationale="Contingency inference was blocked by data-shape preflight; no SciPy chi-square call was made.")
    chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
    small_cells = int(np.sum(expected < 5))
    if table.shape == (2, 2) and (small_cells > 0 or np.min(expected) < 1):
        method = "Fisher exact test"
        rationale = "2x2 table contains sparse expected counts; use exact inference instead of asymptotic chi-square."
    else:
        method = "Chi-square test of independence"
        rationale = "Expected counts support asymptotic chi-square inference."
    return StatisticalMethodDecision(
        problem="categorical_association",
        method=method,
        alternative_methods=("Chi-square test of independence", "Fisher exact test"),
        estimand="Association between categorical variables",
        assumptions=("independent observations",),
        diagnostics={"shape": list(table.shape), "expected_min": float(np.min(expected)), "cells_below_5": small_cells, "chi2_p_preview": float(p), "degrees_of_freedom": int(dof), "alpha": alpha},
        rationale=rationale,
    )


def select_correlation(x: Sequence[float], y: Sequence[float], *, alpha: float = 0.05) -> StatisticalMethodDecision:
    xx, yy = _clean_pair(x, y)
    preflight = numeric_pair(xx, yy, method="correlation", min_n=3)
    if not preflight.allowed:
        return StatisticalMethodDecision(problem="association", method="NOT_APPLICABLE", diagnostics=preflight.to_dict(), limitations=tuple(preflight.reasons), rationale="Statistical correlation was blocked by data-shape preflight; no SciPy correlation call was made.")
    pearson = stats.pearsonr(xx, yy)
    spearman = stats.spearmanr(xx, yy)
    outliers_x = _normality_screen(xx)
    outliers_y = _normality_screen(yy)
    # Use a robust monotonic association method when marginal diagnostics are problematic.
    if abs(float(spearman.statistic) - float(pearson.statistic)) > 0.15 or (outliers_x.get("skew") is not None and abs(outliers_x["skew"]) > 1.5) or (outliers_y.get("skew") is not None and abs(outliers_y["skew"]) > 1.5):
        method = "Spearman correlation"
        rationale = "Pearson and rank correlation disagree materially or heavy skew suggests a robust monotonic association is safer."
    else:
        method = "Pearson correlation"
        rationale = "Pearson and Spearman estimates are consistent and diagnostics do not indicate strong nonlinearity/skew requiring rank methods."
    return StatisticalMethodDecision(
        problem="association",
        method=method,
        alternative_methods=("Spearman correlation", "Kendall tau"),
        estimand="Bivariate association coefficient",
        assumptions=("paired observations", "no causal interpretation without identification",),
        diagnostics={"n": len(xx), "pearson_r": float(pearson.statistic), "pearson_p": float(pearson.pvalue), "spearman_rho": float(spearman.statistic), "spearman_p": float(spearman.pvalue), "x_screen": outliers_x, "y_screen": outliers_y, "alpha": alpha},
        rationale=rationale,
        limitations=("Correlation does not establish causation and is sensitive to selection/range restriction.",),
    )


def _clean_pair(x: Sequence[float], y: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    sx = _to_clean_float_series(x)
    sy = _to_clean_float_series(y)
    mask = np.isfinite(sx) & np.isfinite(sy)
    return sx[mask].to_numpy(dtype=float), sy[mask].to_numpy(dtype=float)
