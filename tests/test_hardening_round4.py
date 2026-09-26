import pandas as pd

from packages.analytics_core.src.cleaning.engine import DataCleaningEngine
from packages.analytics_core.src.engines.belief import BeliefEngine
from packages.analytics_core.src.statistics.multiple_comparisons import pairwise_posthoc


def test_integer_cast_does_not_manufacture_zero_from_invalid_values():
    df = pd.DataFrame({'value': ['1', 'unknown', '3', None]})
    cleaned, logs = DataCleaningEngine().clean_dataset(
        df, [{'action': 'cast_type', 'column': 'value', 'params': {'target_type': 'int'}}]
    )
    assert pd.isna(cleaned.loc[1, 'value'])
    assert cleaned.loc[0, 'value'] == 1
    assert logs[0]['parameters']['rows_failed_conversion'] == 1
    assert 'confidence' not in logs[0]
    assert 'transformation_reliability' in logs[0]


def test_missing_effect_size_is_neutral_evidence_not_zero_effect():
    assert BeliefEngine.compute_evidence_likelihoods(
        1.0, 1.0, effect_size_eta_sq=None, sample_size=20
    ) == (1.0, 1.0)


def test_welch_zero_variance_fails_closed():
    result = pairwise_posthoc(
        {'a': [1, 1, 1], 'b': [2, 2, 2]}, omnibus_method='Welch ANOVA'
    )
    assert result['status'] == 'not_run'
    assert result['omnibus_p_value'] is None


def test_extreme_omnibus_p_is_not_exact_zero():
    result = pairwise_posthoc(
        {'a': list(range(1, 11)), 'b': list(range(1000, 1010))},
        omnibus_method='Welch ANOVA',
        require_significant_omnibus=False,
    )
    assert result['omnibus_p_value'] is not None
    assert result['omnibus_p_value'] > 0.0
