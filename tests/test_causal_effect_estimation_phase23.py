import numpy as np
import pandas as pd
import pytest

from packages.analytics_core.src.causal.effect_estimation import estimate_binary_treatment_effect


def make_data(seed=17, n=3000):
    rng = np.random.default_rng(seed)
    age = rng.normal(40, 10, n)
    income = rng.lognormal(10, 0.35, n)
    p = 1 / (1 + np.exp(-(-1.0 + 0.035 * (age - 40) + 0.000002 * (income - np.mean(income)))))
    t = rng.binomial(1, p)
    # True additive treatment effect = 3.
    y = 10 + 0.8 * (age - 40) / 10 + 0.7 * (np.log(income) - 10) + 3.0 * t + rng.normal(0, 1, n)
    return pd.DataFrame({"age": age, "income": income, "treated": t, "outcome": y})


def test_aipw_recovers_known_ate_with_overlap_and_balance():
    df = make_data()
    res = estimate_binary_treatment_effect(
        df,
        treatment_col="treated",
        outcome_col="outcome",
        adjustment_set=["age", "income"],
        estimator="AIPW",
        bootstrap_resamples=250,
    )
    assert res.estimand == "ATE"
    assert 2.5 < res.effect < 3.5
    assert res.positivity["status"] in {"ADEQUATE", "CAUTION"}
    assert "AIPW" in res.formulas["estimand"]
    assert res.reproducibility["bootstrap_seed"] == 20260908


def test_gc_and_ipw_are_available():
    df = make_data(seed=29, n=800)
    for estimator in ("G-COMPUTATION", "IPW"):
        res = estimate_binary_treatment_effect(
            df,
            treatment_col="treated",
            outcome_col="outcome",
            adjustment_set=["age", "income"],
            estimator=estimator,
            bootstrap_resamples=100,
        )
        assert np.isfinite(res.effect)
        assert len(res.confidence_interval) == 2


def test_non_binary_treatment_fails_closed():
    df = make_data(n=100)
    df["treated"] = np.arange(len(df)) % 3
    with pytest.raises(ValueError, match="binary treatment"):
        estimate_binary_treatment_effect(
            df,
            treatment_col="treated",
            outcome_col="outcome",
            adjustment_set=["age"],
            estimator="AIPW",
            bootstrap_resamples=100,
        )


def test_positivity_and_balance_are_traceable():
    df = make_data(seed=33, n=500)
    res = estimate_binary_treatment_effect(
        df,
        treatment_col="treated",
        outcome_col="outcome",
        adjustment_set=["age", "income"],
        estimator="IPW",
        bootstrap_resamples=100,
    )
    for key in ("propensity_min", "propensity_max", "common_support_width"):
        assert key in res.positivity
    for key in ("standardized_mean_difference_before", "standardized_mean_difference_after_ipw", "status"):
        assert key in res.balance
    assert res.assumptions
    assert res.limitations
