from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_group_concentrated_missingness_becomes_structured_finding():
    import pandas as pd
    from packages.analytics_core.src.profiling.data_quality_gate import DataQualityGate

    df = pd.DataFrame({
        "region": ["A"] * 10 + ["B"] * 10,
        "revenue": [None] * 6 + [10, 11, 12, 13] + list(range(10, 20)),
    })
    result = DataQualityGate.evaluate_fitness(df, metric_col="revenue")
    ids = {f.finding_id for f in result.contextual_findings}
    assert "MISSINGNESS_CONCENTRATED_BY_GROUP" in ids


def test_selection_indicator_is_structured_and_actionable():
    import pandas as pd
    from packages.analytics_core.src.profiling.data_quality_gate import DataQualityGate

    df = pd.DataFrame({
        "active": [True] * 10 + [False] * 10,
        "revenue": list(range(10)) + list(range(100, 110)),
    })
    result = DataQualityGate.evaluate_fitness(df, metric_col="revenue")
    findings = [f for f in result.contextual_findings if f.finding_id == "SELECTION_BIAS_INDICATOR"]
    assert findings
    assert "CAUSAL" in findings[0].affected_methods
    assert findings[0].recommended_action


def test_clean_data_does_not_create_contextual_findings():
    import pandas as pd
    from packages.analytics_core.src.profiling.data_quality_gate import DataQualityGate

    df = pd.DataFrame({
        "region": ["A", "A", "B", "B", "C", "C"],
        "revenue": [10, 11, 12, 13, 14, 15],
    })
    result = DataQualityGate.evaluate_fitness(df, metric_col="revenue")
    assert result.contextual_findings == []
