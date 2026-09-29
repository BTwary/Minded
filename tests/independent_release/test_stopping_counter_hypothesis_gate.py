import unittest

from packages.analytics_core.src.engines.stopping import StoppingEngine


class StoppingCounterHypothesisGateTests(unittest.TestCase):
    def test_high_posterior_cannot_stop_without_counter_hypothesis(self):
        decision = StoppingEngine.evaluate_stopping(
            leading_posterior=0.95,
            entropy_delta=-0.6,
            experiments_completed=3,
            all_verifications_passed=True,
            counter_hypothesis_evaluated=False,
            has_unresolved_adversarial_issue=False,
        )
        self.assertFalse(decision.should_stop)
        self.assertEqual(decision.reason, "COUNTER_HYPOTHESIS_NOT_EVALUATED")
        self.assertFalse(decision.criteria_breakdown["counter_hypothesis_evaluated"])

    def test_stopping_allowed_after_counter_hypothesis_evaluated(self):
        decision = StoppingEngine.evaluate_stopping(
            leading_posterior=0.95,
            entropy_delta=-0.6,
            experiments_completed=3,
            all_verifications_passed=True,
            counter_hypothesis_evaluated=True,
            has_unresolved_adversarial_issue=False,
        )
        self.assertTrue(decision.should_stop)


if __name__ == "__main__":
    unittest.main()
