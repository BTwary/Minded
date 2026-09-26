import math

from packages.analytics_core.src.causal.sensitivity import (
    e_value_report,
    linear_omitted_confounding_sensitivity,
    linear_robustness_value,
    linear_sensitivity_surface,
    risk_ratio_confounding_grid,
)


def test_risk_ratio_grid_has_formal_bias_bounds():
    out = risk_ratio_confounding_grid(2.0, strengths=(1.0, 2.0, 3.0))
    assert out["available"] is True
    assert any(row["crosses_null"] is False for row in out["rows"])
    assert all(row["bias_factor"] >= 1.0 for row in out["rows"])


def test_risk_ratio_grid_can_show_tipping_point():
    out = risk_ratio_confounding_grid(1.5, strengths=(1.5, 2.0, 3.0, 5.0))
    assert any(row["crosses_null"] for row in out["rows"])


def test_e_value_includes_quantitative_grid():
    out = e_value_report(treated_risk=0.60, control_risk=0.40, plausible_strengths=(1.5, 2.0))
    assert math.isfinite(out["e_value"])
    assert out["quantitative_confounding_grid"]["available"] is True


def test_linear_ovb_matches_bias_formula():
    out = linear_omitted_confounding_sensitivity(
        estimate=2.0, standard_error=0.5, df_resid=100, treatment_partial_r2=0.10, outcome_partial_r2=0.20
    )
    expected = 0.5 * math.sqrt(100 * 0.10 * 0.20 / 0.90)
    assert math.isclose(out["bias_factor"], expected, rel_tol=1e-12)
    assert out["worst_case_estimate_toward_null"] < 2.0


def test_linear_robustness_value_is_between_zero_and_one():
    out = linear_robustness_value(estimate=2.0, standard_error=0.5, df_resid=100)
    assert 0 < out["robustness_value_partial_r2"] < 1


def test_linear_surface_is_traceable():
    out = linear_sensitivity_surface(estimate=1.0, standard_error=0.25, df_resid=60, strengths=(0.05, 0.10))
    assert out["available"] is True
    assert len(out["rows"]) == 4
    assert "source_method" in out
