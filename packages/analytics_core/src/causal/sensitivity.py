"""Causal sensitivity analyses for identified treatment-effect analyses."""
from __future__ import annotations
from typing import Any, Dict, Optional, Sequence
import math


def _e_value_rr(rr: float) -> Optional[float]:
    if not math.isfinite(rr) or rr <= 0:
        return None
    rr = max(rr, 1.0 / rr)
    return float(rr + math.sqrt(rr * (rr - 1.0))) if rr >= 1.0 else None


def e_value_report(*, treated_risk: float, control_risk: float, confidence_interval_rr: Optional[list[float]] = None, plausible_strengths: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    if not 0 < treated_risk < 1 or not 0 < control_risk < 1:
        return {"available": False, "reason": "E-value requires strictly positive risks below 1 in both treatment groups."}
    rr = float(treated_risk / control_risk)
    ev = _e_value_rr(rr)
    report: Dict[str, Any] = {
        "available": ev is not None,
        "risk_ratio": rr,
        "e_value": ev,
        "interpretation": "E-value is a robustness summary, not proof against unmeasured confounding.",
        "limitation": "E-values are sensitivity summaries, not a formal proof of absence of unmeasured confounding.",
    }
    if confidence_interval_rr and len(confidence_interval_rr) == 2:
        lo, hi = map(float, confidence_interval_rr)
        if min(lo, hi) > 0 and math.isfinite(lo) and math.isfinite(hi):
            report["confidence_interval_rr"] = [lo, hi]
            report["ci_includes_null"] = lo <= 1.0 <= hi
            report["e_value_lower_bound"] = 1.0 if lo <= 1.0 <= hi else _e_value_rr(lo if lo > 1 else hi)
    strengths = tuple(float(x) for x in (plausible_strengths or ()))
    if strengths:
        report["quantitative_confounding_grid"] = risk_ratio_confounding_grid(rr, strengths=strengths)
    else:
        report["quantitative_confounding_grid"] = {"available": False, "reason": "No plausible confounding-strength grid supplied."}
    return report


def risk_ratio_confounding_grid(risk_ratio: float, *, strengths: Sequence[float] = (1.0, 1.5, 2.0, 3.0, 5.0)) -> Dict[str, Any]:
    rr = float(risk_ratio)
    if not math.isfinite(rr) or rr <= 0:
        return {"available": False, "reason": "risk_ratio must be positive and finite"}
    rows = []
    for strength in strengths:
        s = float(strength)
        if s < 1.0:
            raise ValueError("confounding strengths must be >= 1")
        # Conservative symmetric bias-factor approximation: B=s^2 for equal
        # confounder-treatment and confounder-outcome strengths.
        bias = s * s
        adjusted = rr / bias if rr >= 1 else rr * bias
        crosses_null = adjusted <= 1.0 if rr > 1 else adjusted >= 1.0
        rows.append({"strength": s, "bias_factor": bias, "adjusted_effect": adjusted, "crosses_null": bool(crosses_null)})
    return {"available": True, "risk_ratio": rr, "rows": rows, "method": "symmetric_RR_bias_factor_s_squared"}


def linear_omitted_confounding_sensitivity(*, estimate: float, standard_error: float, df_resid: float, treatment_partial_r2: float, outcome_partial_r2: float) -> Dict[str, Any]:
    r_t, r_y = float(treatment_partial_r2), float(outcome_partial_r2)
    if not (0 <= r_t < 1 and 0 <= r_y < 1):
        raise ValueError("partial R-squared values must be in [0,1)")
    bias_factor = float(standard_error * math.sqrt(max(df_resid, 0) * r_t * r_y / max(1.0 - r_t, 1e-12)))
    toward_null = float(estimate - math.copysign(bias_factor, estimate))
    return {"available": True, "estimate": float(estimate), "standard_error": float(standard_error), "df_resid": float(df_resid), "treatment_partial_r2": r_t, "outcome_partial_r2": r_y, "bias_factor": bias_factor, "worst_case_estimate_toward_null": toward_null, "source_method": "linear_ovb_partial_r2_sensitivity"}


def linear_robustness_value(*, estimate: float, standard_error: float, df_resid: float, alpha: float = 0.05) -> Dict[str, Any]:
    if standard_error <= 0 or df_resid <= 0 or not 0 < alpha < 1:
        raise ValueError("standard_error, df_resid and alpha must be valid")
    # Approximate robustness value: partial R2 required to bring the t statistic to
    # the two-sided critical value. This is a sensitivity metric, not a guarantee.
    from scipy.stats import t as student_t
    t_stat = abs(float(estimate / standard_error))
    crit = float(student_t.ppf(1 - alpha / 2, df_resid))
    rv = (t_stat * t_stat - crit * crit) / (t_stat * t_stat + df_resid) if t_stat > crit else 0.0
    return {"available": True, "robustness_value_partial_r2": float(max(0.0, min(1.0, rv))), "critical_t": crit, "source_method": "approximate_partial_r2_robustness_value"}


def linear_sensitivity_surface(*, estimate: float, standard_error: float, df_resid: float, strengths: Sequence[float] = (0.05, 0.10, 0.20)) -> Dict[str, Any]:
    rows=[]
    for rt in strengths:
        for ry in strengths:
            out=linear_omitted_confounding_sensitivity(estimate=estimate, standard_error=standard_error, df_resid=df_resid, treatment_partial_r2=float(rt), outcome_partial_r2=float(ry))
            rows.append({"treatment_partial_r2":float(rt),"outcome_partial_r2":float(ry),"bias_factor":out["bias_factor"],"worst_case_estimate_toward_null":out["worst_case_estimate_toward_null"]})
    return {"available": True, "rows": rows, "source_method": "linear_ovb_partial_r2_grid"}
