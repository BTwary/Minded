"""DEFECT-036 independent release test suite.

Found by independently stress-probing the DEFECT-033/034/035 fixes with additional ordinary
real-world phrasings, immediately after those fixes shipped. Same failure shape each time:
a common paraphrase falls through every branch of `parse_period`/`_DIAGNOSTIC_RE` and the
question silently downgrades to the generic RANKING/FORECAST fallback instead of the
diagnostic/period-change answer it asked for.

Two independent gaps, both closed this session:

1. `parse_period` (packages/analytics_core/src/engines/analyst_answer.py) returned None for:
   - "year to date" / "YTD" (calendar-year-to-date, not a fixed rolling window)
   - "since the start/beginning of the <quarter|month|year>" (paraphrase of "this <grain>")
   - a bare "fortnight" with no explicit count ("in the past fortnight")
   - "a year ago" / "compared to a year ago" used as a trailing-window anchor
   - "last/past/trailing <N> quarters" and "... months" where N > 10 (word-number table
     stopped at "ten"; "quarter"/"fortnight"/"year" were not accepted units at all)

2. `_DIAGNOSTIC_RE` (packages/analytics_core/src/engines/intent.py) still missed contractions
   ("what's behind") and gerunds ("causing"), so "What's causing the decline this quarter?"
   was hijacked into FORECAST by the bare "this quarter" match -- the exact DEFECT-033 bug,
   just reopened by a different synonym for the same diagnostic marker.

Run with: python -m unittest tests/independent_release/test_defect036_recency_vocabulary_gaps.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import unittest

import numpy as np
import pandas as pd

from packages.analytics_core.src.engines.analyst_answer import parse_period, classify_question
from packages.analytics_core.src.engines.intent import IntentEngine


def _dates():
    return pd.to_datetime(pd.date_range("2026-01-01", periods=90, freq="D"))


class TestDefect036PeriodParsingGaps(unittest.TestCase):
    def setUp(self):
        self.dates = _dates()

    def test_year_to_date_resolves(self):
        p = parse_period("Why is revenue down year to date?", self.dates)
        self.assertIsNotNone(p)
        self.assertEqual(p.start, pd.Timestamp("2026-01-01"))
        self.assertIn("partial year", p.resolved_year_note)

    def test_ytd_abbreviation_resolves(self):
        p = parse_period("Why did revenue fall YTD?", self.dates)
        self.assertIsNotNone(p)
        self.assertEqual(p.grain, "year")

    def test_since_start_of_quarter_resolves(self):
        p = parse_period("Why did revenue drop since the start of the quarter?", self.dates)
        self.assertIsNotNone(p)
        self.assertEqual(p.start, pd.Timestamp("2026-01-01"))

    def test_since_beginning_of_month_resolves(self):
        p = parse_period("Why did revenue drop since the beginning of the month?", self.dates)
        self.assertIsNotNone(p)
        self.assertEqual(p.start, pd.Timestamp("2026-03-01"))

    def test_bare_fortnight_resolves_to_fourteen_days(self):
        p = parse_period("Why did revenue fall in the past fortnight?", self.dates)
        self.assertIsNotNone(p)
        self.assertEqual((p.end - p.start).days, 14)

    def test_year_ago_comparison_resolves_to_rolling_365_days(self):
        p = parse_period("Why is churn up compared to a year ago?", self.dates)
        self.assertIsNotNone(p)
        self.assertEqual((p.end - p.start).days, 365)

    def test_quantified_quarters_resolves(self):
        p = parse_period("Why did revenue fall over the last two quarters?", self.dates)
        self.assertIsNotNone(p)
        self.assertEqual((p.end - p.start).days, 182)

    def test_quantified_twelve_months_resolves(self):
        p = parse_period("Why did revenue fall over the past twelve months?", self.dates)
        self.assertIsNotNone(p)
        self.assertEqual((p.end - p.start).days, 360)

    def test_ytd_question_classifies_as_period_change_not_ranking(self):
        p = parse_period("Why is revenue down year to date?", self.dates)
        kind = classify_question(
            "Why is revenue down year to date?", has_time=True, has_group=True,
            has_explanatory=False, period=p,
        )
        self.assertEqual(kind, "PERIOD_CHANGE")

    def test_original_defect035_cases_still_pass(self):
        """The new bare/YTD/since-start branches run before the existing quantified-recency
        regex; confirm they don't shadow or break the DEFECT-035 cases they sit next to."""
        p = parse_period("Why did revenue fall in the last 30 days?", self.dates)
        self.assertIsNotNone(p)
        self.assertEqual((p.end - p.start).days, 30)


class TestDefect036DiagnosticVocabularyGaps(unittest.TestCase):
    def test_contraction_whats_causing_is_root_cause(self):
        intent = IntentEngine.parse_intent("What's causing the decline this quarter?")
        self.assertEqual(intent.intent_type, "ROOT_CAUSE")

    def test_contraction_whats_behind_is_root_cause(self):
        intent = IntentEngine.parse_intent("What's behind the drop this quarter?")
        self.assertEqual(intent.intent_type, "ROOT_CAUSE")

    def test_gerund_driving_is_root_cause(self):
        intent = IntentEngine.parse_intent("What is driving the decline this year?")
        self.assertEqual(intent.intent_type, "ROOT_CAUSE")

    def test_genuine_forecast_with_this_quarter_is_unaffected(self):
        intent = IntentEngine.parse_intent("What is our revenue outlook this quarter?")
        self.assertEqual(intent.intent_type, "FORECAST")

    def test_genuine_prediction_is_unaffected(self):
        intent = IntentEngine.parse_intent("What is the probability revenue will grow this quarter?")
        self.assertEqual(intent.intent_type, "PREDICTION")

    def test_genuine_forecast_naming_forecast_directly_is_unaffected(self):
        intent = IntentEngine.parse_intent("Forecast revenue for next quarter.")
        self.assertEqual(intent.intent_type, "FORECAST")


if __name__ == "__main__":
    unittest.main()
