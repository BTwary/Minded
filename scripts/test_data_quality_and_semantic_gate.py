"""
test_data_quality_and_semantic_gate.py: Verifies Data Quality Fitness Gates and Schema-Agnostic World Model.
"""
import os
import sys
import unittest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "packages", "analytics_core", "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "apps", "api", "src"))

from packages.analytics_core.src.profiling.data_quality_gate import DataQualityGate
from packages.analytics_core.src.semantic.world_model import SemanticWorldModelBuilder
from packages.schemas.src.semantic_graph import MetricAdditivity


class TestDataQualityAndSemanticGate(unittest.TestCase):
    """Test suite for data fitness gate and schema-agnostic world model."""

    def test_01_empty_dataset_rejection(self):
        """Empty dataset fails closed with UNFIT verdict."""
        df = pd.DataFrame()
        assessment = DataQualityGate.evaluate_fitness(df, "empty_dataset")
        self.assertEqual(assessment.fitness_verdict, "UNFIT")
        self.assertFalse(assessment.can_proceed)
        self.assertEqual(assessment.overall_quality_score, 0.0)
        print(" -> [TEST 01] Empty dataset fail-closed rejection: PASSED")

    def test_02_severe_missingness_detection(self):
        """100% missing columns flagged as critical issues."""
        df = pd.DataFrame({
            "id": range(20),
            "all_nulls": [np.nan] * 20,
            "metric": [10.0 + i for i in range(20)],
        })
        assessment = DataQualityGate.evaluate_fitness(df, "missing_data")
        self.assertIn("Column 'all_nulls' is 100% missing (all nulls).", assessment.critical_issues)
        self.assertEqual(assessment.fitness_verdict, "UNFIT")
        self.assertFalse(assessment.can_proceed)
        print(" -> [TEST 02] Severe missingness critical issue: PASSED")

    def test_03_constant_column_detection(self):
        """Zero-variance constant columns detected and penalized."""
        df = pd.DataFrame({
            "id": range(30),
            "constant_feature": [42.0] * 30,
            "metric": [float(i * 2) for i in range(30)],
        })
        assessment = DataQualityGate.evaluate_fitness(df, "constant_data")
        self.assertIn("constant_feature", assessment.constant_columns)
        self.assertTrue(any("Zero-variance constant columns detected" in w for w in assessment.warnings))
        print(" -> [TEST 03] Constant column detection: PASSED")

    def test_04_duplicate_row_detection(self):
        """Duplicate rows identified and reported."""
        row = {"id": 1, "name": "Alpha", "val": 100.0}
        df = pd.DataFrame([row] * 10 + [{"id": 2, "name": "Beta", "val": 200.0}] * 10)
        assessment = DataQualityGate.evaluate_fitness(df, "duplicate_data")
        self.assertGreater(assessment.duplicate_rows_percentage, 50.0)
        self.assertTrue(any("duplicate" in w.lower() for w in assessment.warnings))
        print(" -> [TEST 04] Duplicate row detection: PASSED")

    def test_05_insufficient_sample_size(self):
        """Sample size smaller than minimum required threshold fails closed."""
        df = pd.DataFrame({"id": [1, 2], "val": [10.0, 20.0]})
        assessment = DataQualityGate.evaluate_fitness(df, "tiny_data", min_sample_size=5)
        self.assertEqual(assessment.fitness_verdict, "UNFIT")
        self.assertFalse(assessment.can_proceed)
        print(" -> [TEST 05] Insufficient sample size rejection: PASSED")

    def test_06_high_quality_dataset_passes(self):
        """Clean dataset passes data quality gate with FIT verdict and score >= 90."""
        df = pd.DataFrame({
            "customer_key": [f"CUST_{i:04d}" for i in range(100)],
            "segment_name": ["Enterprise", "MidMarket", "SMB", "Growth"] * 25,
            "metric_value": np.random.uniform(50.0, 500.0, 100),
        })
        assessment = DataQualityGate.evaluate_fitness(df, "clean_data")
        self.assertEqual(assessment.fitness_verdict, "FIT")
        self.assertTrue(assessment.can_proceed)
        self.assertGreaterEqual(assessment.overall_quality_score, 90.0)
        print(" -> [TEST 06] High quality dataset FIT status: PASSED")

    def test_07_schema_agnostic_world_model(self):
        """Constructs Semantic World Model on randomized arbitrary column tokens with zero domain heuristics."""
        df = pd.DataFrame({
            "tok_pk_99": [f"ID_{i}" for i in range(50)],
            "tok_cat_partition": ["P_A", "P_B", "P_C", "P_D", "P_E"] * 10,
            "tok_metric_continuous": np.random.normal(100.0, 15.0, 50),
            "tok_ratio_rate": np.random.uniform(0.01, 0.99, 50),
        })
        builder = SemanticWorldModelBuilder()
        world_model = builder.build_world_model({"t_arbitrary": df})

        self.assertGreaterEqual(len(world_model.entities), 1)
        self.assertEqual(len(world_model.metrics), 2)  # tok_metric_continuous, tok_ratio_rate

        # Check rate is inferred as NON_ADDITIVE
        rate_metric = next(m for m in world_model.metrics if m.column_name == "tok_ratio_rate")
        self.assertEqual(rate_metric.additivity, MetricAdditivity.NON_ADDITIVE)

        # Schema-agnostic physical data cannot prove business additivity from
        # an arbitrary numeric column name alone. The semantic resolver uses
        # SUM only as an explicitly marked unresolved default and leaves the
        # additivity claim UNKNOWN until declared or positively inferred.
        cont_metric = next(m for m in world_model.metrics if m.column_name == "tok_metric_continuous")
        self.assertEqual(cont_metric.additivity, MetricAdditivity.UNKNOWN)
        self.assertEqual(getattr(cont_metric, "aggregation_type", None), "sum")

        # Check table grain
        self.assertEqual(world_model.table_grains["t_arbitrary"], "entity_level (tok_pk_99)")
        print(" -> [TEST 07] Schema-agnostic semantic world model: PASSED")


def main():
    print("=" * 80, flush=True)
    print("RUNNING DATA QUALITY & SEMANTIC WORLD MODEL ACCEPTANCE TESTS (7 TESTS)", flush=True)
    print("=" * 80, flush=True)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDataQualityAndSemanticGate)
    runner = unittest.TextTestRunner(verbosity=1)
    res = runner.run(suite)
    if res.wasSuccessful():
        print("=" * 80, flush=True)
        print("ALL 7 DATA QUALITY & SEMANTIC WORLD MODEL TESTS PASSED (100% PROVEN)", flush=True)
        print("=" * 80, flush=True)
        sys.exit(0)
    else:
        print("DATA QUALITY & SEMANTIC WORLD MODEL TESTS FAILED", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
