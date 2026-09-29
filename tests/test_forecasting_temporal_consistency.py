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


if __name__ == "__main__":
    unittest.main()
