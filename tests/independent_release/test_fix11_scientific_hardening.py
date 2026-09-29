import random
import pandas as pd

from packages.analytics_core.src.statistics.churn_estimands import analyze_churn_identifiability, ChurnVerdict
from packages.analytics_core.src.intelligence.predictive_hypothesis import HypothesisSynthesizer


def test_tenure_trap_remains_identifiability_limited_with_multiple_confounders():
    from scripts.generate_churn_seed_data import scenario_tenure_trap
    df, _ = scenario_tenure_trap(random.Random(42))
    result = analyze_churn_identifiability(
        df, 'segment', known_confounders=['cohort', 'tenure_days'],
        exposure_col='observation_days', censored_col='censored'
    )
    assert result.verdict == ChurnVerdict.CONFOUNDED_IDENTIFIABILITY_LIMITED


def test_genuine_effect_survives_known_confounders():
    from scripts.generate_churn_seed_data import scenario_genuine_effect
    df, _ = scenario_genuine_effect(random.Random(42))
    result = analyze_churn_identifiability(
        df, 'segment', known_confounders=['cohort', 'tenure_days'],
        exposure_col='observation_days', censored_col='censored'
    )
    assert result.verdict == ChurnVerdict.SUPPORTED_SEGMENT_DIFFERENCE


def test_churn_hypotheses_preserve_multiple_confounders():
    class Sem:
        churn_outcome_available = True
        churn_event_col = 'churn_event'
        group_dimension_col = 'segment'
        available_categorical_cols = ['segment', 'cohort']
        churn_confounder_cols = ['cohort', 'tenure_days']
        churn_exposure_col = 'observation_days'
        target_metric_col = 'churn_event'
        primary_dataset_name = 'customers'
        direction_hint = None
    hyps = HypothesisSynthesizer.synthesize_competing_hypotheses(Sem(), 'Which customer segments have higher churn rates?')
    dims = {h.target_dimension for h in hyps if h.is_counter_hypothesis}
    assert {'cohort', 'tenure_days'}.issubset(dims)
