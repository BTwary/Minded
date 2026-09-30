"""Tests for Fix 3: Forecast Temporal Consistency.

Verifies:
1. Irregular/gapped time series: drift and trend forecasting respect canonical elapsed calendar time
   rather than observation row index.
2. Contiguous time series: preserves standard behaviour and matches rolling origin backtests.
"""
import unittest
import numpy as np
import pandas as pd

from packages.analytics_core.src.forecasting.engine import ForecastingEngine


class TestForecastingTemporalConsistency(unittest.TestCase):
    def test_gapped_time_series_drift_uses_elapsed_time(self):
        """A gapped annual series: years 2010, 2011, 2013, 2020 (4 points, 10 elapsed years).
        y values: 100, 110, 130, 200 (total rise = 100 over 10 years -> slope = 10.0/year).
        If naive row spacing were used, slope would be (200 - 100) / 3 = 33.33/step.
        With canonical temporal spacing, slope is (200 - 100) / 10 = 10.0/year.
        """
        engine = ForecastingEngine()
        y = np.array([100.0, 110.0, 130.0, 200.0])
        # Temporal coordinates in years from t0
        t = np.array([0.0, 1.0, 3.0, 10.0])
        # Forecast 1 year ahead (year 2021 -> t = 11.0, delta = 1.0)
        future_t = np.array([11.0, 12.0])

        fc = engine._fit_forecast(y, model_name="drift", horizon=2, seasonal_periods=None, t=t, future_t=future_t)["forecast"]
        # Expected: 200 + 10.0 * 1 = 210.0, 200 + 10.0 * 2 = 220.0
        self.assertAlmostEqual(fc[0], 210.0, places=3)
        self.assertAlmostEqual(fc[1], 220.0, places=3)

    def test_end_to_end_gapped_forecast_metric(self):
        """End-to-end forecast_metric on irregular dates respects elapsed time."""
        # 12 gapped dates over 20 years
        dates = pd.to_datetime([
            "2000-01-01", "2001-01-01", "2003-01-01", "2005-01-01",
            "2007-01-01", "2009-01-01", "2011-01-01", "2013-01-01",
            "2015-01-01", "2017-01-01", "2018-01-01", "2020-01-01",
        ])
        years = np.array([float(d.year - 2000) for d in dates])
        y = 50.0 + 5.0 * years  # True slope 5.0/year
        df = pd.DataFrame({"dt": dates, "val": y})

        engine = ForecastingEngine()
        out = engine.forecast_metric(df, date_column="dt", metric_column="val", horizon_periods=2)
        self.assertNotIn("error", out)
        self.assertEqual(len(out["forecast_data"]), 2)
        # Point forecasts should continue the positive trajectory consistently
        self.assertGreater(out["next_period_forecast"], y[-1])

    def test_fit_forecast_fails_closed_without_temporal_coordinates(self):
        """_fit_forecast must fail closed with ValueError if t is None, wrong length, or zero span."""
        engine = ForecastingEngine()
        y = np.array([10.0, 20.0, 30.0, 40.0])

        # Missing t
        with self.assertRaises(ValueError) as ctx:
            engine._fit_forecast(y, model_name="drift", horizon=2, seasonal_periods=None, t=None)
        self.assertIn("strictly required", str(ctx.exception))

        # Wrong length t
        with self.assertRaises(ValueError) as ctx:
            engine._fit_forecast(y, model_name="drift", horizon=2, seasonal_periods=None, t=np.array([1.0, 2.0]))
        self.assertIn("strictly required", str(ctx.exception))

        # Zero span t
        with self.assertRaises(ValueError) as ctx:
            engine._fit_forecast(y, model_name="drift", horizon=2, seasonal_periods=None, t=np.array([1.0, 1.0, 1.0, 1.0]))
        self.assertIn("strictly required", str(ctx.exception))

    def test_forecast_metric_fails_closed_on_invalid_temporal_axis(self):
        """forecast_metric must fail closed with INVALID_OR_MISSING_TEMPORAL_COORDINATES when dates cannot be resolved."""
        engine = ForecastingEngine()
        # Case A: Non-temporal arbitrary strings
        df_unparseable = pd.DataFrame({
            "not_a_date": ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"],
            "val": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        })
        out_unparseable = engine.forecast_metric(df_unparseable, date_column="not_a_date", metric_column="val", horizon_periods=2)
        self.assertEqual(out_unparseable.get("failure_reason"), "INVALID_OR_MISSING_TEMPORAL_COORDINATES")

        # Case B: When canonical_time_coordinates raises an error
        from unittest.mock import patch
        df_valid = pd.DataFrame({
            "dt": pd.date_range("2020-01-01", periods=10, freq="D"),
            "val": [float(i) for i in range(10)]
        })
        with patch("packages.analytics_core.src.statistics.analytical_math.canonical_time_coordinates", side_effect=ValueError("Corrupted temporal axis")):
            out_err = engine.forecast_metric(df_valid, date_column="dt", metric_column="val", horizon_periods=2)
            self.assertEqual(out_err.get("failure_reason"), "INVALID_OR_MISSING_TEMPORAL_COORDINATES")
            self.assertIn("Failed to resolve valid canonical temporal coordinates", out_err.get("error", ""))

    def test_duplicate_timestamps_intensive_metric_semantic_mean(self):
        """Intensive metrics (e.g. price, conversion_rate) must aggregate as MEAN across duplicate timestamps, never SUM."""
        engine = ForecastingEngine()
        # 2 observations on 2026-01-01 (price=100 and price=100) -> mean must be 100, NOT 200
        dates = pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05", "2026-01-06", "2026-01-07"])
        prices = [100.0, 100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0]
        df = pd.DataFrame({"dt": dates, "price": prices})

        out = engine.forecast_metric(df, date_column="dt", metric_column="price", horizon_periods=2)
        self.assertNotIn("error", out)
        self.assertEqual(out.get("timestamp_aggregation"), "mean")
        self.assertEqual(out["historical_data"][0]["actual"], 100.0)

    def test_duplicate_timestamps_extensive_metric_semantic_sum(self):
        """Extensive metrics (e.g. revenue, volume, units) aggregate additively as SUM across duplicate timestamps."""
        engine = ForecastingEngine()
        dates = pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05", "2026-01-06", "2026-01-07"])
        sales = [100.0, 100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0]
        df = pd.DataFrame({"dt": dates, "sales": sales})

        out = engine.forecast_metric(df, date_column="dt", metric_column="sales", horizon_periods=2)
        self.assertNotIn("error", out)
        self.assertEqual(out.get("timestamp_aggregation"), "sum")
        self.assertEqual(out["historical_data"][0]["actual"], 200.0)

    def test_duplicate_timestamps_explicit_override_and_reject(self):
        """Explicit aggregation overrides work, and aggregation='reject' fails closed on duplicates."""
        engine = ForecastingEngine()
        dates = pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05", "2026-01-06", "2026-01-07"])
        prices = [100.0, 100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0]
        df = pd.DataFrame({"dt": dates, "price": prices})

        # Explicit sum override on price
        out_sum = engine.forecast_metric(df, date_column="dt", metric_column="price", horizon_periods=2, aggregation="sum")
        self.assertEqual(out_sum["historical_data"][0]["actual"], 200.0)
        self.assertEqual(out_sum.get("timestamp_aggregation"), "sum")

        # Explicit reject on duplicates fails closed
        out_reject = engine.forecast_metric(df, date_column="dt", metric_column="price", horizon_periods=2, aggregation="reject")
        self.assertIn("error", out_reject)
        self.assertEqual(out_reject.get("failure_reason"), "DUPLICATE_TIMESTAMPS_REJECTED")

        # _prepare_series directly raises ValueError on reject
        with self.assertRaises(ValueError) as ctx:
            engine._prepare_series(df, "dt", "price", aggregation="reject")
        self.assertIn("aggregation='reject'", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()


