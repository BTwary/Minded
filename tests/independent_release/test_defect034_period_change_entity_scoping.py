"""DEFECT-034 independent release test suite.

Found by driving DEFECT-031's own real regression dataset (a genuine West-
region revenue collapse injected into an otherwise-normal 90-day date range)
through the real `build_analyst_result` -> `_period_change` path in
`packages/analytics_core/src/engines/analyst_answer.py`.

"Why did revenue fall in the West region recently?" was answering about a
completely different region. `_period_change` never checked whether the
question named a specific entity (unlike the comparison/ranking paths, which
already do this via `_mentioned_levels`) -- so it computed the aggregate
change across *all* regions and attributed it to whichever region happened
to move most in raw dollars. On the injected dataset, that was South (an
ordinary noise fluctuation), not West -- even though West had a genuine 69%
collapse (the actual injected signal) sitting right there in the raw table,
never surfaced in the headline, executive summary, or Strategic Playbook.

Fix: scope the whole before/after comparison to the named entity when the
question unambiguously names exactly one level of exactly one candidate
dimension (falling back to the old aggregate behavior, plus a caveat, when
more than one entity is named and it would be ambiguous which one to scope
to). This also exposed a smaller cosmetic bug: when the change is scoped to
a single named entity and there is no secondary dimension left to decompose
within it, the Strategic Playbook's `driver_lead` fell through to
`_build_strategic_playbook`'s own literal "primary driver" placeholder text
instead of naming the entity that was actually asked about.

Run with: python -m unittest tests/independent_release/test_defect034_period_change_entity_scoping.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import unittest

import numpy as np
import pandas as pd

from packages.analytics_core.src.engines.analyst_answer import build_analyst_result


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


class TestDefect034PeriodChangeEntityScoping(unittest.TestCase):
    def setUp(self):
        self.df = _make_df()

    def test_named_region_is_scoped_to_that_region_not_the_aggregate(self):
        result = build_analyst_result(
            "Why did revenue fall in the West region recently?",
            self.df, target="revenue", time_col="order_date",
            default_aggregation="sum", candidate_dimensions=["region"],
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.kind, "PERIOD_CHANGE")
        self.assertIn("West", result.headline)
        # The real, injected collapse is roughly -69%; the unscoped aggregate bug reported ~-8%
        # attributed to South instead. Assert we're in the right ballpark for the real signal.
        pct = result.numbers.get("pct_change")
        self.assertIsNotNone(pct)
        self.assertLess(pct, -0.5)

    def test_scoped_result_names_the_entity_in_executive_summary(self):
        result = build_analyst_result(
            "Why did revenue fall in the West region recently?",
            self.df, target="revenue", time_col="order_date",
            default_aggregation="sum", candidate_dimensions=["region"],
        )
        self.assertIsNotNone(result)
        self.assertIn("West", result.executive_summary)

    def test_scoped_result_does_not_misname_a_different_segment_as_the_driver(self):
        """The core bug: the old code named 'South' (an unrelated fluctuation) as the driver of
        a question that was explicitly about West. After scoping, nothing in the answer should
        suggest South is the story here."""
        result = build_analyst_result(
            "Why did revenue fall in the West region recently?",
            self.df, target="revenue", time_col="order_date",
            default_aggregation="sum", candidate_dimensions=["region"],
        )
        self.assertIsNotNone(result)
        self.assertNotIn("South", result.executive_summary)

    def test_strategic_playbook_does_not_fall_back_to_literal_primary_driver_text(self):
        """Cosmetic bug exposed by the entity-scoping fix: with no secondary dimension left to
        decompose within the named entity, the playbook's driver text used to fall back to the
        literal, uninformative string "primary driver" instead of naming the entity."""
        result = build_analyst_result(
            "Why did revenue fall in the West region recently?",
            self.df, target="revenue", time_col="order_date",
            default_aggregation="sum", candidate_dimensions=["region"],
        )
        self.assertIsNotNone(result)
        playbook_text = " ".join(
            str(item.get("action", "")) for item in (result.strategic_playbook or [])
        )
        self.assertNotIn("(primary driver)", playbook_text)

    def test_unscoped_question_still_reports_the_aggregate_as_before(self):
        """A question that names no specific region must still behave as before: aggregate
        change across all regions, decomposed to find the biggest mover."""
        result = build_analyst_result(
            "Why did revenue fall recently?",
            self.df, target="revenue", time_col="order_date",
            default_aggregation="sum", candidate_dimensions=["region"],
        )
        self.assertIsNotNone(result)
        self.assertNotIn("scoped_entity", result.numbers)

    def test_ambiguous_multi_entity_question_falls_back_to_aggregate_with_a_caveat(self):
        """Naming more than one segment unambiguously is intentionally left unscoped (picking one
        arbitrarily would be its own silent-misattribution bug) but must be flagged, not silent."""
        result = build_analyst_result(
            "Why did revenue fall in the West and East regions recently?",
            self.df, target="revenue", time_col="order_date",
            default_aggregation="sum", candidate_dimensions=["region"],
        )
        self.assertIsNotNone(result)
        self.assertNotIn("scoped_entity", result.numbers)
        self.assertTrue(any("more than one specific segment" in c for c in result.caveats))


if __name__ == "__main__":
    unittest.main()
