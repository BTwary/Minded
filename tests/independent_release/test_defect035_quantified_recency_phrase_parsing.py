"""DEFECT-035 independent release test suite.

Found while verifying DEFECT-031/034: numeric/word-quantified recency
phrases -- "last 30 days", "past two weeks", "past couple of months",
"trailing 3 months" -- fall through every branch of `parse_period` in
`packages/analytics_core/src/engines/analyst_answer.py` the same way
DEFECT-031's vague recency phrases ("recently", "lately") did before that
fix: `parse_period` returns `None`, `classify_question` never reaches
PERIOD_CHANGE, and the question silently falls back to the generic RANKING
answer, again indistinguishable from an unrelated breakdown question.

Fix: recognise `(?:last|past|trailing) <N> (day|week|month)s?` -- with N as
either a digit or a small set of number words ("two", "a couple of",
"few", ...) -- as a rolling N-day window ending at the latest date in the
data. This reuses the existing "week" grain (never introducing a new grain
value the downstream `{"month","quarter","year","week"}`-keyed `freq` dict
would `KeyError` on) rather than adding a new code path. "Month" in this
phrasing is an approximate 30-day window, not calendar-month arithmetic,
and the resolved-period note says so explicitly.

This also required generalizing `_prior_period` for the "week" grain to
step back by the period's own width rather than assuming a fixed 7 days,
since these rolling windows are not always 7 days wide. For the pre-existing
calendar-week case (width == 7 days) this is behaviorally identical to the
prior fixed-7-day logic.

Run with: python -m unittest tests/independent_release/test_defect035_quantified_recency_phrase_parsing.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import unittest

import numpy as np
import pandas as pd

from packages.analytics_core.src.engines.analyst_answer import (
    build_analyst_result, classify_question, parse_period, _prior_period,
)


def _make_df():
    np.random.seed(7)
    n = 1200
    dates = pd.date_range("2026-01-01", periods=90, freq="D")
    df = pd.DataFrame({
        "order_date": np.random.choice(dates, n),
        "region": np.random.choice(["North", "South", "East", "West"], n),
        "revenue": np.random.gamma(6, 45, n),
    })
    mask = (df["order_date"] >= "2026-03-20") & (df["region"] == "West")
    df.loc[mask, "revenue"] = df.loc[mask, "revenue"] * 0.15
    return df


class TestDefect035QuantifiedRecencyPhraseParsing(unittest.TestCase):
    def setUp(self):
        self.df = _make_df()
        self.dates = pd.to_datetime(self.df["order_date"])

    def test_digit_days_phrase_resolves_to_a_period(self):
        p = parse_period("Why did revenue fall in the last 30 days?", self.dates)
        self.assertIsNotNone(p)
        self.assertEqual((p.end - p.start).days, 30)

    def test_word_number_weeks_phrase_resolves_to_a_period(self):
        p = parse_period("What happened in the past two weeks?", self.dates)
        self.assertIsNotNone(p)
        self.assertEqual((p.end - p.start).days, 14)

    def test_couple_of_months_phrase_resolves_to_a_period(self):
        p = parse_period("Why did churn increase in the past couple of months?", self.dates)
        self.assertIsNotNone(p)
        self.assertEqual((p.end - p.start).days, 60)
        self.assertIn("not a calendar month", p.resolved_year_note)

    def test_trailing_n_months_phrase_resolves_to_a_period(self):
        p = parse_period("Revenue trend over the trailing 3 months", self.dates)
        self.assertIsNotNone(p)
        self.assertEqual((p.end - p.start).days, 90)

    def test_period_ends_at_latest_date_in_data_not_wall_clock(self):
        p = parse_period("Why did revenue fall in the last 5 days?", self.dates)
        self.assertIsNotNone(p)
        self.assertEqual(p.end, self.dates.max().normalize() + pd.Timedelta(days=1))

    def test_prior_period_is_the_same_width_as_the_requested_window(self):
        p = parse_period("Why did revenue fall in the last 30 days?", self.dates)
        p0, p1 = _prior_period(p)
        self.assertEqual(p1, p.start)
        self.assertEqual((p1 - p0).days, (p.end - p.start).days)

    def test_prior_period_for_original_calendar_week_phrasing_is_unchanged(self):
        """The _prior_period generalization must not change behavior for the pre-existing,
        exactly-7-day-wide DEFECT-031 'recently' case."""
        p = parse_period("Why did revenue fall recently?", self.dates)
        p0, p1 = _prior_period(p)
        self.assertEqual(p.grain, "week")
        self.assertEqual((p.end - p.start).days, 7)
        self.assertEqual((p1 - p0).days, 7)
        self.assertEqual(p1, p.start)

    def test_quantified_recency_phrase_classifies_as_period_change_not_ranking(self):
        p = parse_period("Why did revenue fall in the West region in the last 30 days?", self.dates)
        kind = classify_question(
            "Why did revenue fall in the West region in the last 30 days?",
            has_time=True, has_group=True, has_explanatory=False, period=p,
        )
        self.assertEqual(kind, "PERIOD_CHANGE")

    def test_real_answer_scopes_to_the_named_entity_over_the_quantified_window(self):
        """Composes with DEFECT-034: a quantified-recency question naming a specific entity
        must produce a genuine, entity-scoped before/after comparison, not the generic
        ranking fallback and not an unscoped aggregate."""
        result = build_analyst_result(
            "Why did revenue fall in the West region in the last 30 days?",
            self.df, target="revenue", time_col="order_date",
            default_aggregation="sum", candidate_dimensions=["region"],
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.kind, "PERIOD_CHANGE")
        self.assertIn("West", result.headline)

    def test_unrecognised_bare_number_without_unit_word_is_a_documented_remaining_gap(self):
        """Not a fix target this session -- a bare '30d' or unlabelled '30' with no day/week/month
        unit word is not recognised. Documents the boundary so a future change is deliberate."""
        p = parse_period("Why did revenue fall in the last 30d?", self.dates)
        self.assertIsNone(p)


if __name__ == "__main__":
    unittest.main()
