"""Suite 14: Phase 8 Proactive Drift Daemon & Investigation State Forking Test Suite."""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from packages.analytics_core.src.monitoring.drift_engine import DriftMonitorEngine, DriftAlert
from packages.analytics_core.src.execution.daemon import AutonomousDriftDaemon
from packages.analytics_core.src.runtime.state import InvestigationStateManager


class TestPhase8ProactiveAndForking(unittest.TestCase):
    """Verifies K-S / PSI / CUSUM drift detection, proactive daemon enqueuing, and state DAG forking."""

    def test_01_ks_drift_detection_positive(self):
        """K-S test must fire when distributions genuinely diverge."""
        np.random.seed(42)
        baseline = np.random.normal(loc=100.0, scale=15.0, size=500)
        recent = np.random.normal(loc=85.0, scale=20.0, size=500)

        alert = DriftMonitorEngine.evaluate_continuous_drift(
            baseline, recent, col_name="revenue", table_name="fct_orders"
        )

        self.assertIsNotNone(alert)
        self.assertEqual(alert.drift_type, "CONTINUOUS_KS")
        self.assertLess(alert.p_value, DriftMonitorEngine.KS_P_VALUE_THRESHOLD)
        self.assertIn("revenue", alert.auto_investigation_question)
        self.assertIn("fct_orders", alert.auto_investigation_question)

    def test_02_ks_drift_no_false_positive(self):
        """K-S test must NOT fire when distributions are statistically identical."""
        np.random.seed(42)
        baseline = np.random.normal(loc=100.0, scale=15.0, size=500)
        recent = np.random.normal(loc=100.0, scale=15.0, size=500)

        alert = DriftMonitorEngine.evaluate_continuous_drift(
            baseline, recent, col_name="revenue", table_name="fct_orders"
        )
        self.assertIsNone(alert)

    def test_03_psi_categorical_drift_positive(self):
        """PSI must fire when categorical mix shifts beyond 0.20."""
        baseline_counts = {"EMEA": 500, "NA": 300, "APAC": 200}
        recent_counts = {"EMEA": 100, "NA": 700, "APAC": 200}

        alert = DriftMonitorEngine.evaluate_categorical_drift(
            baseline_counts, recent_counts, col_name="market_zone", table_name="fct_orders"
        )

        self.assertIsNotNone(alert)
        self.assertEqual(alert.drift_type, "CATEGORICAL_PSI")
        self.assertGreater(alert.severity_score, 0.0)
        self.assertIn("market_zone", alert.auto_investigation_question)

    def test_04_psi_no_false_positive(self):
        """PSI must NOT fire on minor proportional variation."""
        baseline_counts = {"EMEA": 500, "NA": 300, "APAC": 200}
        recent_counts = {"EMEA": 490, "NA": 310, "APAC": 200}

        alert = DriftMonitorEngine.evaluate_categorical_drift(
            baseline_counts, recent_counts, col_name="market_zone", table_name="fct_orders"
        )
        self.assertIsNone(alert)

    def test_05_cusum_timeseries_shift(self):
        """CUSUM must detect sustained mean shifts in sequential time-series."""
        np.random.seed(7)
        stable_period = np.random.normal(loc=100.0, scale=5.0, size=30)
        shifted_period = np.random.normal(loc=120.0, scale=5.0, size=10)
        timeseries = np.concatenate([stable_period, shifted_period])

        alert = DriftMonitorEngine.evaluate_timeseries_cusum(
            timeseries, col_name="daily_active_users", table_name="fct_usage"
        )

        self.assertIsNotNone(alert)
        self.assertEqual(alert.drift_type, "TIMESERIES_CUSUM")

    def test_06_daemon_autonomous_enqueue(self):
        """Daemon must enqueue a HIGH priority investigation when drift is detected."""
        mock_queue = MagicMock()
        daemon = AutonomousDriftDaemon(queue_provider=mock_queue)

        mock_table = MagicMock()
        mock_table.name = "fct_orders"
        mock_table.project_id = "proj_001"
        
        col_mock = MagicMock()
        col_mock.name = "revenue"
        col_mock.is_continuous = True
        col_mock.is_categorical = False
        mock_table.monitored_columns = [col_mock]

        np.random.seed(42)
        baseline_data = np.random.normal(100, 15, 500)
        recent_data = np.random.normal(80, 20, 500)

        mock_b_df = MagicMock()
        mock_b_df.__getitem__.return_value.to_numpy.return_value = baseline_data
        mock_r_df = MagicMock()
        mock_r_df.__getitem__.return_value.to_numpy.return_value = recent_data

        with patch.object(daemon, "fetch_baseline_sample", return_value=mock_b_df):
            with patch.object(daemon, "fetch_recent_window", return_value=mock_r_df):
                daemon.run_hourly_cycle([mock_table])

        mock_queue.enqueue_investigation.assert_called_once()
        call_kwargs = mock_queue.enqueue_investigation.call_args.kwargs
        self.assertEqual(call_kwargs["priority"], "HIGH")
        self.assertEqual(call_kwargs["trigger_source"], "AUTONOMOUS_DAEMON")
        self.assertIn("revenue", call_kwargs["question"])

    def test_07_investigation_fork_state_cloning(self):
        """Forking must create an independent child state with parent linkage."""
        parent_state = InvestigationStateManager(investigation_id="inv_parent_001")
        parent_state.register_hypothesis(hyp_id="H1", prior=0.4, description="Test Hyp")
        parent_state.advance_turn()

        child_state = parent_state.fork(
            child_investigation_id="inv_child_001",
            at_turn_index=1,
            new_constraints={"exclude_dates": ["2026-08-18"]},
        )

        self.assertEqual(child_state.parent_investigation_id, "inv_parent_001")
        self.assertEqual(child_state.fork_turn_index, 1)
        self.assertEqual(child_state.fork_constraints, {"exclude_dates": ["2026-08-18"]})

        # Mutating child must not affect parent
        child_state.update_hypothesis_posterior("H1", new_posterior=0.95)
        self.assertNotEqual(parent_state.get_hypothesis("H1").posterior, 0.95)

    def test_08_fork_preserves_evidence_ledger(self):
        """Forked investigations must carry full evidence trail up to fork point."""
        parent_state = InvestigationStateManager(investigation_id="inv_parent_002")
        parent_state.append_evidence(claim="Revenue dropped 4.29%", metric="revenue", value=9571.0)
        parent_state.append_evidence(claim="EMEA is dominant segment", metric="share", value=0.62)
        parent_state.advance_turn()

        child_state = parent_state.fork(
            child_investigation_id="inv_child_002",
            at_turn_index=1,
        )

        self.assertEqual(len(child_state.evidence_ledger.entries), 2)
        self.assertEqual(child_state.evidence_ledger.entries[0].claim, "Revenue dropped 4.29%")


def main():
    print("=" * 80, flush=True)
    print("RUNNING SUITE 14: PHASE 8 PROACTIVE DRIFT & STATE FORKING TESTS", flush=True)
    print("=" * 80, flush=True)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPhase8ProactiveAndForking)
    runner = unittest.TextTestRunner(verbosity=2)
    res = runner.run(suite)
    if res.wasSuccessful():
        print("=" * 80, flush=True)
        print("PHASE 8 PROACTIVE & FORKING: ALL 8 TESTS PASSED (100%)", flush=True)
        print("=" * 80, flush=True)
        return 0
    else:
        print("PHASE 8 PROACTIVE & FORKING: FAILED", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
