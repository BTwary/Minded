"""Focused tests for statistically defensible Bayesian evidence models."""
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from packages.analytics_core.src.engines.belief import BeliefEngine


@dataclass
class H:
    hypothesis_code: str
    is_counter_hypothesis: bool = False
    target_dimension: str = ""
    target_value: object = None


def test_two_group_bayes_factor_is_model_based():
    a = np.array([100, 101, 99, 102, 98, 100, 103, 97, 101, 99], dtype=float)
    b = np.array([130, 131, 129, 132, 128, 130, 133, 127, 131, 129], dtype=float)
    ev = BeliefEngine.bayes_factor_two_groups(a, b)
    assert ev.method == "BIC_GAUSSIAN_TWO_GROUPS"
    assert ev.bayes_factor > 10


def test_null_two_group_evidence_does_not_create_confidence():
    rng = np.random.default_rng(7)
    a = rng.normal(100, 10, 60)
    b = rng.normal(100, 10, 60)
    ev = BeliefEngine.bayes_factor_two_groups(a, b)
    assert ev.bayes_factor < 3


def test_correlation_bayes_factor_uses_model_comparison():
    x = np.arange(40, dtype=float)
    y = 4.0 * x + np.linspace(-2, 2, 40)
    ev = BeliefEngine.bayes_factor_correlation(x, y)
    assert ev.method == "BIC_GAUSSIAN_LINEAR_ASSOCIATION"
    assert ev.bayes_factor > 100


def test_categorical_bayes_factor_penalizes_complexity():
    table = pd.DataFrame([[25, 25], [25, 25]], columns=[0, 1], index=["A", "B"])
    ev = BeliefEngine.bayes_factor_categorical_association(table)
    assert ev.bayes_factor < 1.0


def test_trend_bayes_factor_compares_against_intercept_only_model():
    y = 50.0 + 2.5 * np.arange(24, dtype=float)
    ev = BeliefEngine.bayes_factor_trend(y)
    assert ev.method == "BIC_GAUSSIAN_LINEAR_TREND"
    assert ev.bayes_factor > 10


def test_multihypothesis_evidence_only_updates_attributable_models():
    rng = np.random.default_rng(12)
    df = pd.DataFrame({
        "segment": np.repeat(["A", "B"], 50),
        "metric": np.concatenate([rng.normal(100, 10, 50), rng.normal(130, 10, 50)]),
    })
    h1 = H("HYP-01")
    h2 = H("HYP-02", is_counter_hypothesis=True)
    h3 = H("HYP-03")
    factors, diag = BeliefEngine.compute_model_based_bayes_factors(
        hypotheses=[h1, h2, h3],
        primary_df=df,
        result_df=df,
        target_metric_col="metric",
        group_dimension_col="segment",
        aggregation_type="SUM",
        tested_hypothesis_codes=["HYP-01", "HYP-02"],
    )
    assert factors[0] > 1.0
    assert factors[1] < 1.0
    assert factors[2] == 1.0
    assert diag[0]["method"].startswith("BIC_") or diag[0]["method"].startswith("CONCENTRATION_")
    assert diag[2]["method"] == "NEUTRAL_NO_ATTRIBUTABLE_EVIDENCE"


def test_posterior_update_is_exact_and_uses_evidence_weights():
    post, delta = BeliefEngine.compute_bayesian_posteriors([0.5, 0.5], [20.0, 1.0])
    assert math.isclose(post[0], 20.0 / 21.0, rel_tol=1e-12)
    assert post[1] < 0.05
    assert delta < 0


def test_legacy_api_is_neutral_without_sample_size():
    # Eta-squared alone is not a likelihood model; no sample size means no defensible BF.
    l1, l2 = BeliefEngine.compute_evidence_likelihoods(1, 1, effect_size_eta_sq=90.0)
    assert (l1, l2) == (1.0, 1.0)


if __name__ == "__main__":
    tests = [
        test_two_group_bayes_factor_is_model_based,
        test_null_two_group_evidence_does_not_create_confidence,
        test_correlation_bayes_factor_uses_model_comparison,
        test_categorical_bayes_factor_penalizes_complexity,
        test_trend_bayes_factor_compares_against_intercept_only_model,
        test_multihypothesis_evidence_only_updates_attributable_models,
        test_posterior_update_is_exact_and_uses_evidence_weights,
        test_legacy_api_is_neutral_without_sample_size,
    ]
    for t in tests:
        t()
    print(f"PASSED: {len(tests)}/{len(tests)} Bayesian model-evidence tests")
