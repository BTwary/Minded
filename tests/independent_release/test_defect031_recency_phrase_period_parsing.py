"""DEFECT-031 independent release test suite.

Regression coverage for a real-run bug found while investigating a report
that natural-language question parsing/analysis "doesn't work in real",
same prompt as DEFECT-030 but a different bug, found by driving real
root-cause "why did X change" questions against real ingested data with a
genuine injected signal (a specific segment's metric collapsing in the most
recent days of the observed date range).

`packages/analytics_core/src/engines/analyst_answer.py`'s `parse_period`
only recognised a fixed vocabulary of explicit relative-period phrases
("last month", "this quarter", "last week", ...), an explicit quarter/
month/year token, or nothing. Vague-but-extremely-common recency phrasing
("recently", "lately", "of late", "this past week", "in the last few
days", ...) matched none of these, so `parse_period` silently returned
`None`.

`classify_question` requires `period is not None` to route to
"PERIOD_CHANGE" even when the question text obviously asks for one (it
matches both `_PERIOD_CHANGE_RE`, e.g. "why", and `_CHANGE_WORDS_RE`, e.g.
"fall"/"drop"). With `period is None`, a root-cause "why did revenue fall
in the West region recently?" question fell through every other branch
(no TREND word, no explicit ASSOCIATION columns passed in, no RANK/
BREAKDOWN/COMPARE keyword) and hit the final unconditional
`if has_group: return "RANKING"` fallback -- silently downgrading a
temporal root-cause question into a plain top-line "who's highest"
breakdown that named no region, no "why", and no change at all, and was
byte-for-byte identical to the answer for an unrelated plain breakdown
question ("what is the total revenue by region?") asked in the same real
run.

Fix: treat "recently" / "lately" / "of late" / "in recent days" /
"in recent weeks" / "this past week" / "the last few days" as recognised
relative-period phrases resolving (via the data's own calendar, never wall
-clock time) to the same "last week" / "this week" semantics already used
for the explicit phrase, rather than silently dropping the temporal framing.

Run with: python -m unittest tests/independent_release/test_defect031_recency_phrase_period_parsing.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import unittest

import numpy as np
import pandas as pd

from packages.analytics_core.src.engines.analyst_answer import build_analyst_result, parse_period, classify_question


def _make_df():
    np.random.seed(7)
    n = 1200
    dates = pd.date_range("2026-01-01", periods=90, freq="D")
    df = pd.DataFrame({
        "order_date": np.random.choice(dates, n),
        "region": np.random.choice(["North", "South", "East", "West"], n),
        "revenue": np.random.gamma(6, 45, n),
    })
    # genuine injected signal: West collapses in the last ~11 days of the range
    mask = (df["order_date"] >= "2026-03-20") & (df["region"] == "West")
    df.loc[mask, "revenue"] = df.loc[mask, "revenue"] * 0.15
    return df


class TestDefect031RecencyPhrasePeriodParsing(unittest.TestCase):
    def setUp(self):
        self.df = _make_df()
        self.dates = pd.to_datetime(self.df["order_date"])

    def test_recently_resolves_to_a_period(self):
        self.assertIsNotNone(parse_period("Why did revenue fall in the West region recently?", self.dates))

    def test_lately_and_of_late_resolve_to_a_period(self):
        self.assertIsNotNone(parse_period("Why did revenue drop lately?", self.dates))
        self.assertIsNotNone(parse_period("Why has revenue dropped of late?", self.dates))

    def test_recency_phrase_classifies_as_period_change_not_ranking(self):
        period = parse_period("Why did revenue fall in the West region recently?", self.dates)
        kind = classify_question(
            "Why did revenue fall in the West region recently?",
            has_time=True, has_group=True, has_explanatory=False, period=period,
        )
        self.assertEqual(kind, "PERIOD_CHANGE")

    def test_real_answer_is_a_genuine_temporal_comparison_not_the_unrelated_ranking_fallback(self):
        result = build_analyst_result(
            "Why did revenue fall in the West region recently?",
            self.df, target="revenue", group="region", time_col="order_date",
            default_aggregation="sum",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.kind, "PERIOD_CHANGE")
        text = result.to_text()
        # Must not be the generic top-line ranking answer (the actual bug: this text used to be
        # byte-identical to the unrelated "what is total revenue by region" breakdown answer).
        self.assertNotIn("Ranking by total revenue", text)

    def test_recently_matches_the_explicit_last_week_phrasing(self):
        """'recently' should behave the same as the already-supported explicit 'last week' phrase,
        since it now resolves via the same relative-period branch."""
        q_recent = "Why did revenue fall in the West region recently?"
        q_explicit = "Why did revenue fall in the West region last week?"
        r_recent = build_analyst_result(q_recent, self.df, target="revenue", group="region",
                                         time_col="order_date", default_aggregation="sum")
        r_explicit = build_analyst_result(q_explicit, self.df, target="revenue", group="region",
                                           time_col="order_date", default_aggregation="sum")
        self.assertIsNotNone(r_recent)
        self.assertIsNotNone(r_explicit)
        self.assertEqual(r_recent.kind, r_explicit.kind)

    def test_unrecognised_recency_phrasing_is_a_documented_remaining_gap_not_a_silent_regression(self):
        """Not a fix target this session -- documents the boundary of what DEFECT-031 covers.
        The adjective form ("the recent decline") still isn't recognised and still falls back
        to RANKING. This test exists so a future fix that changes this behavior updates it
        deliberately rather than it drifting unnoticed."""
        period = parse_period("What's causing the recent revenue decline?", self.dates)
        self.assertIsNone(period)


if __name__ == "__main__":
    unittest.main()
