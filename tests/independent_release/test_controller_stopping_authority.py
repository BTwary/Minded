import unittest

from packages.analytics_core.src.engines.stopping import StoppingEngine


class ControllerStoppingAuthorityTests(unittest.TestCase):
    def test_normal_stop_requires_competing_evidence(self):
        decision = StoppingEngine.evaluate_stopping(
            leading_posterior=0.90,
            entropy_delta=-0.50,
            experiments_completed=3,
            all_verifications_passed=True,
            counter_hypothesis_evaluated=False,
        )
        self.assertFalse(decision.should_stop)
        self.assertEqual(decision.reason, "COUNTER_HYPOTHESIS_NOT_EVALUATED")

    def test_controller_can_stop_after_direct_counter_evidence(self):
        decision = StoppingEngine.evaluate_stopping(
            leading_posterior=0.90,
            entropy_delta=-0.50,
            experiments_completed=3,
            all_verifications_passed=True,
            counter_hypothesis_evaluated=True,
        )
        self.assertTrue(decision.should_stop)


if __name__ == "__main__":
    unittest.main()
