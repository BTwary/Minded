"""DEFECT-038 independent release test suite.

Found while root-causing the scientific-calibration backlog (session 20, priority #1 from
the project's own priority list): `classify_question`'s final catch-all
(`if has_group: return "RANKING"`) fires for ANY group-shaped question that doesn't match a
more specific branch -- including genuinely causal/diagnostic "why did X surge/spike/drop
across Y" questions that have no time dimension at all (a cross-sectional question, not a
period-change one). Those RANKING answers are marked `descriptive=True, supersedes_loop=True`,
and `InvestigationController` (packages/analytics_core/src/runtime/controller.py, ~line 4349)
uses that flag to overwrite the hypothesis loop's own verdict -- including a correct DIAGNOSED
verdict from real Bayesian evidence -- with a shallow "group X has the highest total"
observation that never actually answers the causal "why" the question asked.

This is the root cause of test_session7_o3_simpsons_refutation's and
test_session8_verifier_workflow's DIAGNOSED-expected/OBSERVED-actual failures (confirmed:
both pass again after this fix, with zero other changes).

The fix: `classify_question` now recognises this shape (genuine diagnostic language + genuine
change language + no time window resolved or named) and returns "NONE" instead of "RANKING".
`build_analyst_result` already treats "NONE" as "say nothing rather than guess" (used
elsewhere in the same function), which means the controller's superseding block is skipped
entirely and the hypothesis loop's own verdict is left untouched -- the same safe behavior
this module already uses when a required column is missing.

Run with: python -m unittest tests/independent_release/test_defect038_crosssectional_diagnostic_not_ranking.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import unittest

import pandas as pd

from packages.analytics_core.src.engines.analyst_answer import build_analyst_result, classify_question


class TestDefect038CrossSectionalDiagnosticNotRanking(unittest.TestCase):
    def test_why_surge_across_group_with_no_time_column_is_none_not_ranking(self):
        kind = classify_question(
            "Why did cost_metric surge across datacenter_region?",
            has_time=False, has_group=True, has_explanatory=False, period=None,
        )
        self.assertEqual(kind, "NONE")

    def test_why_surge_across_group_is_none_even_when_dataset_has_an_unreferenced_time_column(self):
        """The bug is about what the *question* references, not what columns the dataset
        happens to have -- a dataset with a date column the question never mentions must not
        change the outcome."""
        kind = classify_question(
            "Why did cost_metric surge across datacenter_region?",
            has_time=True, has_group=True, has_explanatory=False, period=None,
        )
        self.assertEqual(kind, "NONE")

    def test_build_analyst_result_returns_none_for_this_shape(self):
        """None here is correct and important: it means the controller must not overwrite the
        hypothesis loop's own verdict with a shallow ranking recomputation (see controller.py's
        `analyst_result is not None` gate around the OBSERVED-downgrade block)."""
        df = pd.DataFrame({
            "datacenter_region": ["us-east", "us-west", "eu-central", "ap-south"] * 100,
            "cost_metric": [500.0 if (i % 4 == 0 and i < 200) else 100.0 for i in range(400)],
        })
        result = build_analyst_result(
            "Why did cost_metric surge across datacenter_region?", df,
            target="cost_metric", group="datacenter_region", time_col=None,
            default_aggregation="sum",
        )
        self.assertIsNone(result)

    def test_genuine_ranking_question_is_unaffected(self):
        kind = classify_question(
            "Which region has the highest cost_metric?",
            has_time=False, has_group=True, has_explanatory=False, period=None,
        )
        self.assertEqual(kind, "RANKING")

    def test_genuine_group_comparison_question_is_unaffected(self):
        kind = classify_question(
            "How does cost_metric differ between regions?",
            has_time=False, has_group=True, has_explanatory=False, period=None,
        )
        self.assertEqual(kind, "GROUP_COMPARISON")

    def test_genuine_period_change_question_with_time_is_unaffected(self):
        dates = pd.to_datetime(pd.date_range("2026-01-01", periods=90, freq="D"))
        from packages.analytics_core.src.engines.analyst_answer import parse_period
        q = "Why did revenue fall this month?"
        p = parse_period(q, dates)
        kind = classify_question(q, has_time=True, has_group=True, has_explanatory=False, period=p)
        self.assertEqual(kind, "PERIOD_CHANGE")

    def test_genuine_unresolved_timeframe_question_is_unaffected(self):
        """A question that DOES name a time window (just not one `parse_period` recognises)
        must still hit DEFECT-037's fail-closed branch, not the new DEFECT-038 one -- the two
        guards must not shadow each other."""
        dates = pd.to_datetime(pd.date_range("2026-01-01", periods=90, freq="D"))
        from packages.analytics_core.src.engines.analyst_answer import parse_period
        q = "Why did revenue fall over the last handful of quarters?"
        p = parse_period(q, dates)
        kind = classify_question(q, has_time=True, has_group=True, has_explanatory=False, period=p)
        self.assertEqual(kind, "PERIOD_CHANGE_UNRESOLVED_TIMEFRAME")


if __name__ == "__main__":
    unittest.main()
