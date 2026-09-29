import os, sys
import pandas as pd
import numpy as np
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from packages.analytics_core.src.statistics.uncertainty import bootstrap_ci, hedges_g, rank_biserial, eta_squared
from packages.analytics_core.src.statistics.inference import execute_two_group, execute_multi_group, execute_correlation, execute_categorical


def test_bootstrap_is_reproducible():
    a = [1,2,3,4,5,6,7,8]
    b = [2,3,4,5,6,7,8,9]
    first = bootstrap_ci(hedges_g, a, b)
    second = bootstrap_ci(hedges_g, a, b)
    assert first == second
    assert first["method"] == "percentile_bootstrap"
    assert first["valid_resamples"] == first["resamples"]


def test_two_group_effect_has_uncertainty():
    result = execute_two_group([1,2,3,4,5,6,7,8,9,10], [3,4,5,6,7,8,9,10,11,12])
    assert "effect_size_confidence_interval" in result
    assert result["uncertainty"]["lower"] <= result["effect_size"] <= result["uncertainty"]["upper"] or result["uncertainty"]["lower"] >= result["effect_size"] >= result["uncertainty"]["upper"]


def test_mann_whitney_effect_has_uncertainty():
    result = execute_two_group([1,1,1,2,3,100], [1,1,2,3,4,200])
    assert result["method"] == "Mann-Whitney U"
    lo, hi = result["effect_size_confidence_interval"]
    assert lo <= result["effect_size"] <= hi


def test_anova_eta_squared_uncertainty():
    result = execute_multi_group({"A":[1,2,3,4,5], "B":[5,6,7,8,9], "C":[9,10,11,12,13]})
    lo, hi = result["eta_squared_confidence_interval"]
    assert lo <= result["eta_squared"] <= hi
    assert 0 <= lo <= hi <= 1


def test_kruskal_reports_epsilon_squared():
    result = execute_multi_group({"A":[1,1,1,2,50], "B":[4,5,5,6,60], "C":[8,9,9,10,70]})
    assert result["method"] == "Kruskal-Wallis"
    assert result["epsilon_squared"] >= 0


def test_spearman_bootstrap_interval_is_present():
    x = np.array([1,2,3,4,5,6,7,8,9,100], dtype=float)
    y = np.array([1,2,3,4,5,6,7,8,9,10], dtype=float)
    result = execute_correlation(x, y)
    assert result["method"] == "Spearman correlation"
    lo, hi = result["confidence_interval"]
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo <= hi


def test_cramers_v_bootstrap_interval_is_present():
    df = pd.DataFrame({"a":[0,0,0,0,1,1,1,1,1,0,0,1], "b":[0,0,1,1,1,1,0,1,0,0,0,1]})
    result = execute_categorical(df, "a", "b")
    if result["method"] == "Fisher exact test":
        lo, hi = result["odds_ratio_confidence_interval"]
        assert 0 <= lo <= hi
        assert result["odds_ratio"] >= 0
    else:
        lo, hi = result["cramers_v_confidence_interval"]
        assert 0 <= lo <= hi <= 1
