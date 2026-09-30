"""DEFECT-037 independent release test suite.

Adopted from an external research spec's "Minimum Rejection Policy" (reviewed on request):
every prior recency-phrase defect in this line (031/033/034/035/036) shared one root shape --
the question clearly named *some* time window, `parse_period` didn't recognise that specific
phrasing, and `classify_question` then silently fell through to a plain RANKING answer that
never mentions the time window at all, indistinguishable from an unrelated question. Each fix
so far closed one more phrasing, but the phrase space is unbounded.

Rather than only continuing to enumerate phrasings (which the DEFECT-036 session flagged as
architecturally unsustainable), this adds a fail-closed backstop: detect the *general shape*
of "the question references a time window" (a broad, phrasing-agnostic marker) independently
of whether `parse_period`'s specific-phrase table recognises it. When the shape is present but
the specific phrase isn't resolved, the layer now returns an honest full-range answer with a
prominent, disclosed caveat -- "explicit interpretation acceptable" per the reviewed spec's
policy tiers -- instead of a silent misclassification.

This does not replace `parse_period`'s specific-phrase table, which still gives exact,
correctly-scoped periods for everything DEFECT-031/033/034/035/036 already cover; it only
changes what happens for phrasings *outside* that table.

Run with: python -m unittest tests/independent_release/test_defect037_unresolved_timeframe_failclosed.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import unittest

import numpy as np
import pandas as pd

from packages.analytics_core.src.engines.analyst_answer import (
    build_analyst_result, classify_question, parse_period,
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


class TestDefect037UnresolvedTimeframeFailClosed(unittest.TestCase):
    def setUp(self):
        self.df = _make_df()
        self.dates = pd.to_datetime(self.df["order_date"])

    def test_unrecognised_phrase_with_time_unit_word_is_flagged_not_silently_misrouted(self):
        q = "Why did revenue fall in the West region over the last handful of quarters?"
        p = parse_period(q, self.dates)
        self.assertIsNone(p)  # confirms this genuinely isn't in parse_period's table
        kind = classify_question(q, has_time=True, has_group=True, has_explanatory=False, period=p)
        self.assertEqual(kind, "PERIOD_CHANGE_UNRESOLVED_TIMEFRAME")

    def test_bare_ago_phrase_is_flagged(self):
        q = "Why did revenue drop two years ago?"
        p = parse_period(q, self.dates)
        self.assertIsNone(p)
        kind = classify_question(q, has_time=True, has_group=True, has_explanatory=False, period=p)
        self.assertEqual(kind, "PERIOD_CHANGE_UNRESOLVED_TIMEFRAME")

    def test_real_answer_discloses_the_gap_instead_of_pretending_it_is_scoped(self):
        q = "Why did revenue fall in the West region over the last handful of quarters?"
        result = build_analyst_result(
            q, self.df, target="revenue", time_col="order_date",
            default_aggregation="sum", candidate_dimensions=["region"],
        )
        self.assertIsNotNone(result)
        self.assertIn("not recognized", result.headline)
        self.assertTrue(any("could not confidently parse" in c for c in result.caveats))

    def test_question_with_no_time_reference_at_all_is_unaffected(self):
        """A question with a change word but no time reference at all falls through to
        `NONE` (DEFECT-038's cross-sectional diagnostic-question guard), not the pre-DEFECT-038
        RANKING catch-all -- see test_defect038_diagnostic_no_time_reference_is_not_ranking.py
        for that guard's own dedicated coverage. This test only confirms it isn't wrongly
        caught as PERIOD_CHANGE_UNRESOLVED_TIMEFRAME (there's no time reference to be unresolved)."""
        q = "Why did revenue fall in the West region?"
        p = parse_period(q, self.dates)
        kind = classify_question(q, has_time=True, has_group=True, has_explanatory=False, period=p)
        self.assertNotEqual(kind, "PERIOD_CHANGE_UNRESOLVED_TIMEFRAME")

    def test_already_resolved_phrasings_are_unaffected(self):
        """Every phrasing DEFECT-031/033/034/035/036 already fixed must still resolve to a real
        Period and classify as PERIOD_CHANGE, not get caught by the new generic marker."""
        for q in [
            "Why did revenue fall this month?",
            "Why did revenue fall recently?",
            "Why did revenue fall in the last 30 days?",
            "Why is revenue down year to date?",
            "Why did revenue drop since the start of the quarter?",
            "Why did revenue fall in the past fortnight?",
        ]:
            p = parse_period(q, self.dates)
            self.assertIsNotNone(p, f"expected a resolved period for: {q!r}")
            kind = classify_question(q, has_time=True, has_group=True, has_explanatory=False, period=p)
            self.assertEqual(kind, "PERIOD_CHANGE", f"unexpected kind for: {q!r}")


if __name__ == "__main__":
    unittest.main()
