"""Q6.1 regression suite (v20-C3.1a follow-up): IntentEngine CORRELATION
keyword inflection robustness.

C3.1's Q6 audit (see test_c3_1_q6_real_compiler_symmetry.py) found that
IntentEngine.parse_intent's CORRELATION keyword check was a literal-
substring list containing "association" but not its "-ed" inflection
"associated", so "Is price associated with sales?" and "Are sales and
price associated?" fell through every branch to GENERAL and never reached
CORRELATION at all.

Fixed by replacing the literal list with `_CORRELATION_KEYWORD_RE`, a
regex that matches single-root words by stem instead of listing every
inflection individually (see packages/analytics_core/src/engines/intent.py
for the full rationale). This suite tests the fix directly at the
IntentEngine unit level (the real end-to-end pipeline proof lives in
test_c3_1_q6_real_compiler_symmetry.py::TestQ6RealCompilerPathFixedByQ6_1).

Run with: python -m unittest tests/independent_release/test_q6_1_intent_correlation_inflections.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from packages.analytics_core.src.engines.intent import IntentEngine


class TestPreviouslyMissedInflectionsNowRecognized(unittest.TestCase):
    """The exact gap the Q6 audit found, plus its natural siblings."""

    def test_associated_now_recognized(self):
        self.assertEqual(
            IntentEngine.parse_intent("Is price associated with sales?").intent_type, "CORRELATION",
        )
        self.assertEqual(
            IntentEngine.parse_intent("Are sales and price associated?").intent_type, "CORRELATION",
        )

    def test_other_previously_unlisted_inflections_now_recognized(self):
        cases = [
            "Does price correlate with sales?",       # "correlate" (bare verb, no old entry)
            "What correlates with churn?",             # "correlates"
            "Is revenue dependent on marketing spend?",  # "dependent" (old list only had "depend")
            "Are the two metrics tightly coupled?",     # "coupled" (old list only had "coupling")
            "Is churn linked to plan tier?",            # already worked pre-fix via substring, still must work
            "How is revenue impacted by discounts?",    # "impacted" (old list only had "impact")
        ]
        for q in cases:
            with self.subTest(question=q):
                self.assertEqual(IntentEngine.parse_intent(q).intent_type, "CORRELATION")


class TestPreExistingRecognizedPhrasingsUnaffected(unittest.TestCase):
    """The fix must not change classification for phrasing that already
    worked before Q6.1 -- a pure regex swap, not a behavior expansion for
    already-working cases."""

    def test_originally_listed_words_still_recognized(self):
        cases = [
            "Is price correlated with sales?",
            "What is the relationship between price and sales?",
            "Is revenue tied to marketing spend?",
            "Do the two metrics move together?",
            "What is the connection between churn and tenure?",
            "What is the effect of price on sales?",
        ]
        for q in cases:
            with self.subTest(question=q):
                self.assertEqual(IntentEngine.parse_intent(q).intent_type, "CORRELATION")


class TestOtherIntentsUnaffected(unittest.TestCase):
    """CORRELATION is checked first, so broadening its regex must not
    swallow questions that should still classify as FORECAST / ROOT_CAUSE /
    CHURN -- the exact regression risk of a stem like `depend\\w*` or
    `connect\\w*` over-matching."""

    def test_forecast_unaffected(self):
        self.assertEqual(IntentEngine.parse_intent("Forecast next quarter revenue").intent_type, "FORECAST")
        self.assertEqual(IntentEngine.parse_intent("What will revenue look like next month?").intent_type, "FORECAST")

    def test_root_cause_unaffected(self):
        self.assertEqual(IntentEngine.parse_intent("Why did revenue fall in March?").intent_type, "ROOT_CAUSE")
        self.assertEqual(IntentEngine.parse_intent("What caused the drop in signups?").intent_type, "ROOT_CAUSE")

    def test_churn_unaffected(self):
        self.assertEqual(IntentEngine.parse_intent("Are customers churning?").intent_type, "CHURN")
        self.assertEqual(IntentEngine.parse_intent("What is our retention rate?").intent_type, "CHURN")


if __name__ == "__main__":
    unittest.main()
