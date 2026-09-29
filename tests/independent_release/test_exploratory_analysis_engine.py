import numpy as np
import pandas as pd

from packages.analytics_core.src.engines.exploratory_analysis import ExploratoryAnalysisEngine


def test_reconnaissance_covers_core_statistics_deterministically():
    rng = np.random.default_rng(20260909)
    n = 80
    x = np.arange(n, dtype=float)
    y = 2.0 * x + rng.normal(0, 2, size=n)
    group = np.where(x < 40, "A", "B")
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=n, freq="D"),
        "x": x,
        "y": y,
        "group": group,
    })
    df.loc[[3, 7, 11], "y"] = np.nan
    result1 = ExploratoryAnalysisEngine.analyze(df)
    result2 = ExploratoryAnalysisEngine.analyze(df)
    assert result1 == result2
    assert result1["scope"]["rows"] == 80
    assert result1["columns"]["y"]["missing"] == 3
    assert result1["correlations"]
    assert result1["temporal_trends"]
    assert result1["group_effects"]
    assert result1["missingness"]["rates"]["y"] == 0.0375


def test_screening_layer_does_not_emit_causal_claims():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6], "y": [2, 4, 6, 8, 10, 12]})
    result = ExploratoryAnalysisEngine.analyze(df)
    assert result["interpretation"].startswith("Exploratory screening only")
    assert all("causal" not in str(item).lower() for item in result["top_screening_signals"])


def test_correlation_pair_pvalue_is_not_unadjusted_minimum():
    # Construct a relationship where the two screening tests return very
    # different p-values. The pair-level p-value must not be the raw minimum.
    rng = np.random.default_rng(42)
    x = np.arange(120, dtype=float)
    y = x + rng.normal(0, 18, size=len(x))
    df = pd.DataFrame({"x": x, "y": y})
    item = ExploratoryAnalysisEngine.analyze(df)["correlations"][0]
    expected = min(1.0, 2.0 * min(item["pearson_p"], item["spearman_p"]))
    assert item["p_value_method"] == "simes_two_test"
    assert item["p_value"] >= min(item["pearson_p"], item["spearman_p"])
    assert item["p_value"] >= 0.0
    # The Simes result can equal 2*min p when that is the smaller candidate.
    assert item["p_value"] >= expected or np.isclose(item["p_value"], expected)
