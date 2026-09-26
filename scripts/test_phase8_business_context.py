"""Suite 13: Phase 8 Business Context, dbt Manifest & Prescriptive Action Test Suite."""
import json
import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from packages.analytics_core.src.semantic.dbt_manifest_parser import DbtSemanticLayerCompiler, DbtMetricDefinition
from packages.analytics_core.src.intelligence.experiment_designer import ExperimentDesignAgent
from packages.analytics_core.src.intelligence.prescriptive_optimizer import PrescriptiveOptimizer
from packages.analytics_core.src.ml.uplift_engine import UpliftEngine


class TestPhase8BusinessContext(unittest.TestCase):
    """Verifies dbt manifest ingestion, A/B power calculations, and prescriptive optimization."""

    def test_01_dbt_manifest_ingestion_and_filter_injection(self):
        """Verifies that DbtSemanticLayerCompiler extracts metric filters from mock dbt manifest."""
        mock_manifest = {
            "metrics": {
                "metric.mind_ed.active_churn": {
                    "name": "active_churn",
                    "label": "Active Customer Churn",
                    "model": "ref('fct_subscriptions')",
                    "calculation_method": "count",
                    "expression": "subscription_id",
                    "filters": [
                        {"field": "is_churned", "operator": "=", "value": "true"},
                        {"field": "tenure_days", "operator": ">", "value": "30"},
                    ],
                }
            }
        }

        temp_path = os.path.join(PROJECT_ROOT, "mock_manifest_test.json")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(mock_manifest, f)

        try:
            metrics_map = DbtSemanticLayerCompiler.parse_manifest(temp_path)
            self.assertIn("active_churn", metrics_map)
            dbt_def = metrics_map["active_churn"]
            self.assertEqual(dbt_def.base_model, "fct_subscriptions")
            self.assertEqual(len(dbt_def.filters), 2)
            self.assertEqual(dbt_def.filters[0]["field"], "is_churned")
            self.assertEqual(dbt_def.filters[1]["value"], "30")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_02_ab_power_calculation_accuracy(self):
        """Verifies A/B power calculation: mean=100, std=15, MDE=5%, alpha=0.05, power=0.80 -> N per arm ~140-150."""
        spec = ExperimentDesignAgent.calculate_ab_parameters(
            baseline_mean=100.0,
            baseline_std=15.0,
            daily_traffic=1000,
            target_mde_pct=0.05,
            alpha=0.05,
            power=0.80,
        )

        self.assertTrue(140 <= spec.required_n_per_arm <= 150, f"Got N={spec.required_n_per_arm}")
        self.assertEqual(spec.total_sample_size, spec.required_n_per_arm * 2)
        self.assertEqual(spec.estimated_duration_days, 1)
        self.assertEqual(spec.baseline_variance, 225.0)

    def test_03_zero_variance_power_fail_closed(self):
        """Verifies that A/B calculator fails closed if baseline has zero variance."""
        with self.assertRaises(ValueError) as ctx:
            ExperimentDesignAgent.calculate_ab_parameters(
                baseline_mean=50.0,
                baseline_std=0.0,
                daily_traffic=500,
                target_mde_pct=0.05,
            )
        self.assertIn("Cannot calculate power: zero variance", str(ctx.exception))

    def test_04_prescriptive_linprog_budget_allocation(self):
        """Verifies that Linear Programming optimizer allocates $1000 budget to maximize ROI under bounds."""
        channels = ["Search", "Social", "Email"]
        expected_rois = [0.10, 0.25, 0.15]
        total_budget = 1000.0
        min_bounds = [0.0, 0.0, 100.0]
        max_bounds = [500.0, 800.0, 500.0]

        result = PrescriptiveOptimizer.optimize_budget_allocation(
            channels=channels,
            expected_rois=expected_rois,
            total_budget=total_budget,
            min_bounds=min_bounds,
            max_bounds=max_bounds,
        )

        # Optimizer maxes out Social (0.25 ROI) to 800
        self.assertAlmostEqual(result.optimal_allocation["Social"], 800.0, delta=1e-3)
        # Next highest ROI is Email (0.15 ROI > 0.10 Search), so remaining 200 goes to Email (satisfies min bound 100 <= 200 <= 500)
        self.assertAlmostEqual(result.optimal_allocation["Email"], 200.0, delta=1e-3)
        # Search gets 0 since Email has higher return
        self.assertAlmostEqual(result.optimal_allocation["Search"], 0.0, delta=1e-3)
        # Max Expected Return = 800*0.25 + 200*0.15 + 0*0.10 = 200 + 30 = 230.0
        self.assertAlmostEqual(result.max_expected_return, 230.0, delta=1e-3)


def main():
    print("=" * 80, flush=True)
    print("RUNNING SUITE 13: PHASE 8 BUSINESS CONTEXT & PRESCRIPTIVE ACTION TESTS", flush=True)
    print("=" * 80, flush=True)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPhase8BusinessContext)
    runner = unittest.TextTestRunner(verbosity=2)
    res = runner.run(suite)
    if res.wasSuccessful():
        print("=" * 80, flush=True)
        print("PHASE 8 BUSINESS CONTEXT: ALL 4 TESTS PASSED (100%)", flush=True)
        print("=" * 80, flush=True)
        return 0
    else:
        print("PHASE 8 BUSINESS CONTEXT: FAILED", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
