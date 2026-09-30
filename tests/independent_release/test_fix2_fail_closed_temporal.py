"""Tests for Fix 2: Fail-Closed Behavior for Silent Temporal Fallbacks.

Verifies:
1. BeliefEngine.bayes_factor_trend(x=None) fails closed with UNRESOLVED_TEMPORAL_COORDINATES.
2. BeliefEngine.bayes_factor_trend with mismatched x length fails closed with INVALID_TEMPORAL_COORDINATES_LENGTH.
3. BeliefEngine.bayes_factor_trend with degenerate x (zero variance) fails closed.
4. BeliefEngine.bayes_factor_trend with valid canonical coordinates executes normally.
5. canonical_time_coordinates(None) raises ValueError.
6. canonical_time_coordinates on non-temporal numeric column raises ValueError (rejects non-temporal input).
7. canonical_time_coordinates on unparseable/arbitrary sequence raises ValueError.
8. canonical_time_coordinates with mismatched length raises ValueError.
9. execution_provider._derive_primary_metric(agg="REGRESSION") fails closed without ordinal fallback when time is missing or non-temporal.
"""
import unittest
import numpy as np
import pandas as pd

from packages.analytics_core.src.engines.belief import BeliefEngine
from packages.analytics_core.src.statistics.analytical_math import (
    canonical_time_coordinates,
    resolve_canonical_temporal_axis,
)
from packages.analytics_core.src.engines.execution_provider import _derive_primary_metric


class TestFix2FailClosedTemporal(unittest.TestCase):
    def test_belief_engine_trend_x_none_fails_closed(self):
        y = 100.0 + 2.5 * np.arange(20, dtype=float)
        ev = BeliefEngine.bayes_factor_trend(y, x=None)
        self.assertEqual(ev.method, "UNRESOLVED_TEMPORAL_COORDINATES")
        self.assertEqual(ev.bayes_factor, 1.0)
        self.assertEqual(ev.diagnostic, "missing_temporal_coordinates")
        self.assertIn("Explicit canonical temporal coordinates are required", ev.assumptions[0])

    def test_belief_engine_trend_x_wrong_length_fails_closed(self):
        y = 100.0 + 2.5 * np.arange(20, dtype=float)
        x_short = np.arange(10, dtype=float)
        ev = BeliefEngine.bayes_factor_trend(y, x=x_short)
        self.assertEqual(ev.method, "INVALID_TEMPORAL_COORDINATES_LENGTH")
        self.assertEqual(ev.bayes_factor, 1.0)
        self.assertIn("coordinate_length_mismatch", ev.diagnostic)

    def test_belief_engine_trend_x_degenerate_variance_fails_closed(self):
        y = 100.0 + 2.5 * np.arange(20, dtype=float)
        x_const = np.full(20, 5.0)
        ev = BeliefEngine.bayes_factor_trend(y, x=x_const)
        self.assertEqual(ev.method, "INVALID_TEMPORAL_COORDINATES")
        self.assertEqual(ev.bayes_factor, 1.0)
        self.assertEqual(ev.diagnostic, "degenerate_temporal_coordinates")

    def test_belief_engine_trend_valid_coords_executes(self):
        y = 100.0 + 2.5 * np.arange(20, dtype=float)
        x = np.arange(20, dtype=float)
        ev = BeliefEngine.bayes_factor_trend(y, x=x)
        self.assertEqual(ev.method, "BIC_GAUSSIAN_LINEAR_TREND")
        self.assertGreater(ev.bayes_factor, 10.0)
        self.assertIn("slope=", ev.diagnostic)

    def test_canonical_time_coordinates_none_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            canonical_time_coordinates(None)
        self.assertIn("require a valid time series", str(ctx.exception))

    def test_canonical_time_coordinates_nontemporal_numeric_column_rejected(self):
        s = pd.Series([1500.0, 2000.0, 2500.0, 3200.0, 4000.0], name="weight")
        with self.assertRaises(ValueError) as ctx:
            canonical_time_coordinates(s)
        self.assertIn("non-temporal or unidentifiable grain", str(ctx.exception))

    def test_canonical_time_coordinates_arbitrary_strings_rejected(self):
        s = pd.Series(["apple", "banana", "cherry", "date"], name="fruit")
        with self.assertRaises(ValueError) as ctx:
            canonical_time_coordinates(s)
        self.assertIn("non-temporal or unidentifiable grain", str(ctx.exception))

    def test_canonical_time_coordinates_length_mismatch_raises(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        with self.assertRaises(ValueError) as ctx:
            canonical_time_coordinates(dates, length=12)
        self.assertIn("does not match expected length", str(ctx.exception))

    def test_canonical_time_coordinates_valid_calendar_years(self):
        years = pd.Series([2015, 2016, 2018, 2020], name="year")
        coords = canonical_time_coordinates(years)
        np.testing.assert_array_equal(coords, np.array([0.0, 1.0, 3.0, 5.0]))

    def test_execution_provider_regression_missing_time_column_raises(self):
        df = pd.DataFrame({"revenue": [100.0, 120.0, 140.0, 160.0]})
        with self.assertRaises(ValueError) as ctx:
            _derive_primary_metric(
                df,
                aggregation_type="REGRESSION",
                primary_result_column="revenue",
                sql="SELECT revenue FROM t ORDER BY period",
            )
        self.assertIn("explicit chronological time column", str(ctx.exception))

    def test_execution_provider_regression_nontemporal_column_raises(self):
        df = pd.DataFrame({
            "category": ["A", "B", "C", "D"],
            "revenue": [100.0, 120.0, 140.0, 160.0],
        })
        with self.assertRaises(ValueError) as ctx:
            _derive_primary_metric(
                df,
                aggregation_type="REGRESSION",
                primary_result_column="revenue",
                sql="SELECT category, revenue FROM t ORDER BY category",
            )
        self.assertIn("failed to extract canonical temporal coordinates", str(ctx.exception))

    def test_execution_provider_regression_valid_temporal_executes(self):
        df = pd.DataFrame({
            "year": [2010, 2011, 2013, 2014],
            "revenue": [100.0, 110.0, 130.0, 140.0],
        })
        slope, metric_name = _derive_primary_metric(
            df,
            aggregation_type="REGRESSION",
            primary_result_column="revenue",
            sql="SELECT year, revenue FROM t ORDER BY year",
        )
        self.assertEqual(metric_name, "regression_slope")
        self.assertAlmostEqual(slope, 10.0, places=4)


if __name__ == "__main__":
    unittest.main()
