import math
import numpy as np
import pandas as pd

from packages.analytics_core.src.forecasting.engine import ForecastingEngine


def make_trend(n=36):
    dates = pd.date_range("2023-01-01", periods=n, freq="MS")
    y = np.arange(n, dtype=float) * 10 + 100
    return pd.DataFrame({"date": dates, "sales": y})


def make_seasonal(n=48):
    dates = pd.date_range("2021-01-01", periods=n, freq="MS")
    seasonal = np.array([20, 25, 40, 55, 70, 80, 75, 60, 45, 35, 30, 25], dtype=float)
    y = 500 + np.tile(seasonal, n // 12)
    return pd.DataFrame({"date": dates, "sales": y})


def test_rolling_backtest_selects_valid_model_and_emits_trace():
    out = ForecastingEngine().forecast_metric(make_trend(), "date", "sales", horizon_periods=3)
    assert "error" not in out
    assert out["selected_model"] in ForecastingEngine.SUPPORTED_MODELS
    assert len(out["forecast_data"]) == 3
    assert out["calculation_trace"]["selected_model"] == out["selected_model"]
    assert out["selection_metric"] == "out_of_sample_MAE"


def test_seasonal_candidate_available_and_not_hardcoded_sum_formula():
    out = ForecastingEngine().forecast_metric(make_seasonal(), "date", "sales", horizon_periods=3)
    assert "error" not in out
    assert out["seasonal_periods"] == 12
    names = {m["model"] for m in out["candidate_models"]}
    assert "seasonal_naive" in names
    assert out["interval_method"] in {
        "empirical_absolute_backtest_residual_quantile_sqrt_h",
        "normal_residual_approximation_fallback",
    }


def test_negative_series_is_not_silently_clipped():
    dates = pd.date_range("2024-01-01", periods=18, freq="MS")
    df = pd.DataFrame({"date": dates, "metric": -np.arange(18, dtype=float)})
    out = ForecastingEngine().forecast_metric(df, "date", "metric", horizon_periods=2)
    assert "error" not in out
    assert out["next_period_forecast"] < 0


def test_insufficient_history_fails_closed():
    dates = pd.date_range("2024-01-01", periods=5, freq="MS")
    df = pd.DataFrame({"date": dates, "sales": np.arange(5, dtype=float)})
    out = ForecastingEngine().forecast_metric(df, "date", "sales", horizon_periods=3)
    assert "error" in out
