"""DEFECT-033 independent release test suite.

Found via adversarial stress-testing of `IntentEngine.parse_intent` in
`packages/analytics_core/src/engines/intent.py` (the classifier that decides
which analytical family -- FORECAST, ROOT_CAUSE, PREDICTION, ... -- answers a
natural-language question), while working through the same real
`classify_question`-adjacent surface that DEFECT-031/032 touched.

`_FORECAST_RE` matches bare "this month" / "this quarter" / "this year", not
only forward-looking "next ..." phrasing. Because the `elif forecast` branch
in `parse_intent` is checked *before* `elif _DIAGNOSTIC_RE.search(q_lower)`,
any explicitly backward-looking diagnostic question that happens to contain
"this quarter" / "this month" / "this year" -- e.g. "Why is churn increasing
this quarter?" -- was hijacked into FORECAST and never reached ROOT_CAUSE,
even though it names no "next"/future language at all and carries an
explicit "why" marker.

Fix: `prediction` already takes priority over `forecast` (`forecast = ...
and not prediction`); `diagnostic` needed the same precedence so a
"why ... this quarter" question still reaches the ROOT_CAUSE branch instead
of being silently answered as a time-series projection.

Run with: python -m unittest tests/independent_release/test_defect033_forecast_diagnostic_precedence.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import unittest

from packages.analytics_core.src.engines.intent import IntentEngine


class TestDefect033ForecastDiagnosticPrecedence(unittest.TestCase):
    def test_why_phrased_question_with_this_quarter_is_root_cause_not_forecast(self):
        intent = IntentEngine.parse_intent("Why is churn increasing this quarter?")
        self.assertEqual(intent.intent_type, "ROOT_CAUSE")

    def test_why_phrased_question_with_this_month_is_root_cause_not_forecast(self):
        intent = IntentEngine.parse_intent("Why did revenue drop this month?")
        self.assertEqual(intent.intent_type, "ROOT_CAUSE")

    def test_why_phrased_question_with_this_year_is_root_cause_not_forecast(self):
        intent = IntentEngine.parse_intent("Why did margins fall this year?")
        self.assertEqual(intent.intent_type, "ROOT_CAUSE")

    def test_explain_phrased_question_with_this_quarter_is_root_cause(self):
        intent = IntentEngine.parse_intent("Explain the drop in retention this quarter.")
        self.assertEqual(intent.intent_type, "ROOT_CAUSE")

    def test_genuine_forecast_question_with_next_is_unaffected(self):
        intent = IntentEngine.parse_intent("What will revenue look like next month?")
        self.assertEqual(intent.intent_type, "FORECAST")

    def test_genuine_forecast_question_naming_forecast_directly_is_unaffected(self):
        intent = IntentEngine.parse_intent("Forecast revenue for next quarter.")
        self.assertEqual(intent.intent_type, "FORECAST")

    def test_genuine_forecast_question_with_bare_this_quarter_and_no_diagnostic_marker_is_still_forecast(self):
        """Without a diagnostic marker, bare 'this quarter' forecast phrasing must still route to
        FORECAST -- the fix must not overcorrect into always preferring ROOT_CAUSE."""
        intent = IntentEngine.parse_intent("What is our revenue outlook this quarter?")
        self.assertEqual(intent.intent_type, "FORECAST")

    def test_prediction_still_outranks_forecast_as_before(self):
        """Pre-existing precedence (prediction over forecast) must be unchanged by this fix."""
        intent = IntentEngine.parse_intent("What is the probability revenue will grow this quarter?")
        self.assertEqual(intent.intent_type, "PREDICTION")

    def test_gerund_driving_is_recognised_as_diagnostic(self):
        """DEFECT-036 (found while verifying DEFECT-033): `_DIAGNOSTIC_RE` matched `driver\\w*`
        but not the gerund "driving", so "What is driving the decline this year?" misrouted to
        FORECAST despite naming no forward-looking language at all. Fixed alongside the
        contraction gap below by widening `_DIAGNOSTIC_RE` to `driv(?:er\\w*|ing)`."""
        intent = IntentEngine.parse_intent("What is driving the decline this year?")
        self.assertEqual(intent.intent_type, "ROOT_CAUSE")

    def test_contraction_whats_behind_is_recognised_as_diagnostic(self):
        """DEFECT-036: `_DIAGNOSTIC_RE` matched the literal phrase "what is behind" but not its
        contraction "what's behind", so "What's behind the drop this quarter?" misrouted to
        FORECAST the same way DEFECT-033's original bug did, just under different wording."""
        intent = IntentEngine.parse_intent("What's behind the drop this quarter?")
        self.assertEqual(intent.intent_type, "ROOT_CAUSE")

    def test_gerund_causing_is_recognised_as_diagnostic(self):
        """DEFECT-036: `_DIAGNOSTIC_RE` matched only the fixed phrase "what caused", not the bare
        word "causing"/"caused"/"cause", so "What's causing the decline this quarter?" misrouted
        to FORECAST -- again the same DEFECT-033 bug class under a different phrasing."""
        intent = IntentEngine.parse_intent("What's causing the decline this quarter?")
        self.assertEqual(intent.intent_type, "ROOT_CAUSE")


if __name__ == "__main__":
    unittest.main()
