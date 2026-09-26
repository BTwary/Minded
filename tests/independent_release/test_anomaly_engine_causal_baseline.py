
import pandas as pd
import numpy as np
from packages.analytics_core.src.anomaly.engine import AnomalyDetectionEngine

def test_time_series_baseline_excludes_current_observation():
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=10, freq="D"),
        "value": [10, 10, 10, 10, 10, 100, 10, 10, 10, 10],
    })
    result = AnomalyDetectionEngine().detect_time_series_anomalies(
        df, "date", "value", window=4, sigma=2.0
    )
    dates = [x["date"] for x in result["anomalies"]]
    assert "2026-01-06" in dates

def test_early_points_require_historical_context():
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=8, freq="D"),
        "value": [100, 100, 100, 100, 10, 100, 100, 100],
    })
    result = AnomalyDetectionEngine().detect_time_series_anomalies(
        df, "date", "value", window=4, sigma=2.0
    )
    assert all(x["date"] >= "2026-01-04" for x in result["anomalies"])

def test_flat_history_has_finite_bands():
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=8, freq="D"),
        "value": [5, 5, 5, 5, 5, 5, 5, 20],
    })
    result = AnomalyDetectionEngine().detect_time_series_anomalies(
        df, "date", "value", window=4, sigma=2.0
    )
    assert result["anomalies"]
    for row in result["anomalies"]:
        assert np.isfinite(row["lower_band"])
        assert np.isfinite(row["upper_band"])
