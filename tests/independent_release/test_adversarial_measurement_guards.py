import pandas as pd
from packages.analytics_core.src.profiling.data_quality_gate import DataQualityGate
from packages.analytics_core.src.intelligence.adversarial_attacker import AdversarialAttacker
from packages.analytics_core.src.intelligence.predictive_hypothesis import PredictiveHypothesis


def test_leakage_indicator_requires_measured_near_deterministic_relationship():
    df = pd.DataFrame({
        "x": range(20),
        "target": [i % 2 for i in range(20)],
        "future_score": [float(i % 2) for i in range(20)],
    })
    assessment = DataQualityGate.evaluate_fitness(df, metric_col="target")
    assert any("future_score" in x for x in assessment.leakage_indicators)


def test_selection_indicator_records_observed_difference_when_present():
    df = pd.DataFrame({
        "active": [0] * 10 + [1] * 10,
        "metric": [1.0] * 10 + [10.0] * 10,
    })
    assessment = DataQualityGate.evaluate_fitness(df, metric_col="metric")
    assert any("Measured selection-dependent" in x for x in assessment.selection_bias_indicators)


def test_adversarial_attacker_records_measured_selection_and_leakage_findings():
    df = pd.DataFrame({
        "segment": ["a", "b"] * 10,
        "active": [0] * 10 + [1] * 10,
        "metric": [float(i) for i in range(20)],
        "future_metric": [float(i) for i in range(20)],
    })
    h1 = PredictiveHypothesis(id="H1", hypothesis_code="H1", claim="segment effect", mechanism="observed", predicted_observables_if_true=[], predicted_observables_if_false=[], falsification_criteria="none", required_assumptions=[], target_dimension="segment", target_metric="metric", prior_probability=0.5, posterior_probability=0.5)
    h2 = PredictiveHypothesis(id="H2", hypothesis_code="H2", claim="no segment effect", mechanism="null", predicted_observables_if_true=[], predicted_observables_if_false=[], falsification_criteria="none", required_assumptions=[], target_dimension="segment", target_metric="metric", prior_probability=0.5, posterior_probability=0.5, is_counter_hypothesis=True)
    result = AdversarialAttacker.design_attack(h1, [h2], df)
    assert "selection_bias_attack" in result.details or "target_leakage_attack" in result.details
