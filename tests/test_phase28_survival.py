import numpy as np
import pandas as pd

from packages.analytics_core.src.survival import kaplan_meier, logrank_test, cox_proportional_hazards


def test_kaplan_meier_known_median_and_events():
    df = pd.DataFrame({"t": [1, 2, 3, 4, 5, 6], "e": [1, 1, 0, 1, 0, 1]})
    r = kaplan_meier(df, duration_col="t", event_col="e")
    assert r.events == 4
    assert r.censored == 2
    assert r.estimate["median_survival"] == 4.0
    assert r.formulas["survival"].startswith("S(t)")


def test_logrank_no_difference_known_data():
    df = pd.DataFrame({
        "t": [1, 2, 3, 4, 1, 2, 3, 4],
        "e": [1, 1, 1, 0, 1, 1, 1, 0],
        "g": ["A"] * 4 + ["B"] * 4,
    })
    r = logrank_test(df, duration_col="t", event_col="e", group_col="g")
    assert 0.0 <= r.estimate["p_value"] <= 1.0
    assert r.trace["group_column"] == "g"


def test_cox_returns_hazard_ratio_and_trace():
    rng = np.random.default_rng(123)
    n = 80
    x = rng.normal(size=n)
    base = rng.exponential(scale=3.0, size=n)
    event = (base < 4.0).astype(int)
    t = np.minimum(base, 4.0)
    df = pd.DataFrame({"t": t, "e": event, "x": x})
    r = cox_proportional_hazards(df, duration_col="t", event_col="e", covariates=["x"])
    row = r.estimate["coefficients"][0]
    assert row["hazard_ratio"] > 0
    assert "ph_assumption_status" in r.diagnostics


def test_survival_rejects_invalid_event_and_negative_time():
    bad = pd.DataFrame({"t": [1, -1], "e": [0, 1]})
    try:
        kaplan_meier(bad, duration_col="t", event_col="e")
    except ValueError:
        pass
    else:
        raise AssertionError("Negative duration must be rejected")
