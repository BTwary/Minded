"""Hardening-pass tests for the Phase 28 survival module.

These tests were added after an independent cross-validation against
``lifelines`` (a widely used, separately implemented survival-analysis
library) surfaced a real floating-point boundary bug in the Kaplan-Meier
median calculation. They exist to (a) lock in numerical agreement with an
independent reference implementation rather than only checking internal
consistency, and (b) pin down edge-case behavior that the original Phase 28
test file did not exercise.

``lifelines`` is a test-only dependency: it is not required by
``survival.py`` itself and is skipped gracefully if unavailable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from packages.analytics_core.src.survival import (
    kaplan_meier,
    logrank_test,
    cox_proportional_hazards,
)

lifelines = pytest.importorskip("lifelines", reason="lifelines is a test-only cross-validation dependency")
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test as lifelines_logrank


def _simulated_dataset(seed: int = 7, n: int = 150):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    t = rng.exponential(scale=np.exp(-0.5 * x))
    c = rng.exponential(scale=2.0)
    obs = np.minimum(t, c)
    ev = (t <= c).astype(int)
    return pd.DataFrame({"t": obs, "e": ev, "x": x})


# ---------------------------------------------------------------------------
# Cross-validation against an independent implementation (lifelines)
# ---------------------------------------------------------------------------

def test_km_median_matches_lifelines_reference():
    df = _simulated_dataset()
    ours = kaplan_meier(df, duration_col="t", event_col="e")
    theirs = KaplanMeierFitter().fit(df["t"], df["e"])
    # Before the hardening fix, a survival value landing at ~0.5 + float
    # noise (e.g. 0.5000000000000003) was skipped, deferring the reported
    # median to the *next* event time. This must match the reference to
    # floating-point tolerance, not merely to one event-time step.
    assert ours.estimate["median_survival"] == pytest.approx(theirs.median_survival_time_, rel=1e-9)


def test_logrank_p_value_matches_lifelines_reference():
    df = _simulated_dataset()
    group = (df["x"] > 0)
    labeled = df.assign(g=group.map({True: "A", False: "B"}))
    ours = logrank_test(labeled, duration_col="t", event_col="e", group_col="g")
    theirs = lifelines_logrank(df["t"][group], df["t"][~group], df["e"][group], df["e"][~group])
    assert ours.estimate["computable"] is True
    assert ours.estimate["p_value"] == pytest.approx(theirs.p_value, rel=1e-6)


def test_cox_hazard_ratio_matches_lifelines_reference():
    df = _simulated_dataset()
    ours = cox_proportional_hazards(df, duration_col="t", event_col="e", covariates=["x"])
    theirs = CoxPHFitter().fit(df[["t", "e", "x"]], duration_col="t", event_col="e")
    our_hr = ours.estimate["coefficients"][0]["hazard_ratio"]
    their_hr = float(np.exp(theirs.params_["x"]))
    assert our_hr == pytest.approx(their_hr, rel=1e-4)


# ---------------------------------------------------------------------------
# Edge cases introduced or tightened by the hardening pass
# ---------------------------------------------------------------------------

def test_km_median_exact_half_boundary_regression():
    # Regression test for the specific floating-point boundary bug found via
    # lifelines cross-validation: construct a tiny dataset where survival
    # lands numerically at 0.5 within float noise and assert the median is
    # reported at that crossing time, not the following event time.
    df = pd.DataFrame({"t": [1, 2, 3, 4], "e": [1, 1, 1, 1]})
    r = kaplan_meier(df, duration_col="t", event_col="e")
    # Survival after t=2 is exactly 0.5 (2 of 4 remain); median must be 2.0,
    # not 3.0.
    assert r.estimate["median_survival"] == 2.0


def test_logrank_rejects_unlabeled_rows_from_risk_set():
    # A row with a missing (NaN) group label must be dropped entirely, not
    # left inflating the at-risk pool while belonging to neither group.
    df = pd.DataFrame({
        "t": [1, 2, 3, 4, 1, 2, 3, 4, 5],
        "e": [1, 1, 1, 0, 1, 1, 1, 0, 1],
        "g": ["A"] * 4 + ["B"] * 4 + [None],
    })
    r = logrank_test(df, duration_col="t", event_col="e", group_col="g")
    assert r.diagnostics["rows_dropped_unlabeled_group"] == 1
    assert r.n == 8


def test_logrank_degenerate_variance_is_not_computable():
    # Both subjects fail together at the same single time point: n_total ==
    # d_total at that time, so the hypergeometric variance term is exactly
    # zero. Must be reported as not computable, not as a confident p=1.0.
    df = pd.DataFrame({"t": [1, 1], "e": [1, 1], "g": ["A", "B"]})
    r = logrank_test(df, duration_col="t", event_col="e", group_col="g")
    assert r.estimate["computable"] is False
    assert r.estimate["p_value"] is None


def test_cox_rejects_constant_covariate():
    df = _simulated_dataset()
    df = df.assign(const=1.0)
    with pytest.raises(ValueError, match="constant"):
        cox_proportional_hazards(df, duration_col="t", event_col="e", covariates=["const"])


def test_cox_reports_collinearity_diagnostic():
    df = _simulated_dataset()
    rng = np.random.default_rng(11)
    # Strongly but not perfectly collinear (small independent noise) so the
    # fit still converges to finite estimates and we can inspect the
    # diagnostic rather than hitting the separate non-finite-fit guard.
    df = df.assign(x_dup=df["x"] * 2.0 + rng.normal(scale=0.01, size=len(df)))
    r = cox_proportional_hazards(df, duration_col="t", event_col="e", covariates=["x", "x_dup"])
    assert "design_matrix_condition_number" in r.diagnostics
    assert r.diagnostics["collinearity_flag"] is True

    r_clean = cox_proportional_hazards(df, duration_col="t", event_col="e", covariates=["x"])
    assert r_clean.diagnostics["collinearity_flag"] is False


def test_survival_rejects_infinite_duration():
    df = pd.DataFrame({"t": [1.0, np.inf], "e": [1, 0]})
    with pytest.raises(ValueError, match="finite"):
        kaplan_meier(df, duration_col="t", event_col="e")


def test_survival_rejects_empty_dataset():
    df = pd.DataFrame({"t": pd.Series([], dtype=float), "e": pd.Series([], dtype=float)})
    with pytest.raises(ValueError, match="No valid rows"):
        kaplan_meier(df, duration_col="t", event_col="e")
