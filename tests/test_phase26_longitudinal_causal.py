import numpy as np
import pandas as pd
import pytest

from packages.analytics_core.src.causal.longitudinal import estimate_longitudinal_msm


def make_longitudinal(seed=77, n=80, periods=3):
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(n):
        age = 35 + rng.normal(0, 5)
        prev_a = 0
        latent = rng.normal(0, 0.5)
        for t in range(periods):
            conf = 0.15 * age + 0.8 * prev_a + rng.normal(0, 0.2)
            p = 1 / (1 + np.exp(-(-0.25 + 0.01 * age + 0.25 * prev_a + 0.08 * conf)))
            a = int(rng.random() < p)
            y = 2.0 + 0.4 * age + latent + 0.9 * a + 0.4 * prev_a + rng.normal(0, 0.25)
            rows.append({"unit": unit, "time": t, "age": age, "conf": conf, "treated": a, "outcome": y})
            prev_a = a
    return pd.DataFrame(rows)


def test_longitudinal_msm_returns_traceable_regime_effect():
    df = make_longitudinal()
    res = estimate_longitudinal_msm(
        df,
        unit_col="unit",
        time_col="time",
        treatment_col="treated",
        outcome_col="outcome",
        baseline_covariates=["age"],
        time_varying_covariates=["conf"],
        bootstrap_resamples=20,
    )
    assert np.isfinite(res.effect)
    assert res.periods == 3
    assert res.reproducibility["bootstrap_unit"] == "unit"
    assert "stabilized_weight" in res.formulas
    assert res.weight_diagnostics["weight_cap"] == 50.0


def test_longitudinal_msm_rejects_incomplete_histories():
    df = make_longitudinal(n=30)
    df = df[df["unit"] != 2].copy()
    df = df[df["unit"].ne(3) | df["time"].ne(2)].copy()
    with pytest.raises(ValueError, match="complete treatment histories"):
        estimate_longitudinal_msm(
            df,
            unit_col="unit", time_col="time", treatment_col="treated", outcome_col="outcome",
            baseline_covariates=["age"], time_varying_covariates=["conf"], bootstrap_resamples=0,
        )


def test_longitudinal_msm_rejects_nonbinary_treatment():
    df = make_longitudinal(n=25)
    df.loc[df.index[0], "treated"] = 2
    with pytest.raises(ValueError, match="treated must be binary"):
        estimate_longitudinal_msm(
            df, unit_col="unit", time_col="time", treatment_col="treated", outcome_col="outcome",
            baseline_covariates=["age"], time_varying_covariates=["conf"], bootstrap_resamples=0,
        )


def test_longitudinal_msm_does_not_claim_arbitrary_dynamic_rules():
    df = make_longitudinal(n=30)
    res = estimate_longitudinal_msm(
        df, unit_col="unit", time_col="time", treatment_col="treated", outcome_col="outcome",
        baseline_covariates=["age"], time_varying_covariates=["conf"], bootstrap_resamples=0,
    )
    assert any("always-treated" in x for x in res.limitations)
