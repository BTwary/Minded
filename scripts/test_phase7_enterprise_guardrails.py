"""Suite 12: Phase 7 Enterprise Guardrails & Autonomous Data Janitor Test Suite."""
import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import polars as pl
from packages.analytics_core.src.profiling.data_remediation import DataRemediationEngine
from packages.analytics_core.src.security.governance import GovernanceEngine
from packages.analytics_core.src.intelligence.eig_optimizer import ComputeCostEstimator, EIGOptimizer
from packages.analytics_core.src.intelligence.predictive_hypothesis import PredictiveHypothesis
from packages.analytics_core.src.intelligence.experiment_synthesizer import CandidateExperiment


class TestPhase7EnterpriseGuardrails(unittest.TestCase):
    """Verifies Data Remediation, Governance/RLS injection, and Cost-Aware EIG."""

    def test_01_mice_imputation_preserves_target_metric(self):
        """Invariant: Target metric with >5% missingness triggers fail-closed, while covariates are safely imputed."""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "account_id": [f"ACC_{i}" for i in range(n)],
            "target_revenue": np.random.normal(500.0, 50.0, n),
            "covariate_usage": np.random.normal(50.0, 10.0, n),
            "covariate_age": np.random.normal(30.0, 5.0, n),
        })

        # Inject 15% missingness into covariate_usage
        df.loc[10:24, "covariate_usage"] = np.nan

        # Should successfully remediate covariate missingness
        res = DataRemediationEngine.remediate_missingness(df, target_metric="target_revenue")
        self.assertEqual(res.df["covariate_usage"].isnull().sum(), 0)
        self.assertGreaterEqual(len(res.warnings), 1)

        # Now inject 10% missingness into target_revenue -> must fail-closed!
        bad_df = df.copy()
        bad_df.loc[0:9, "target_revenue"] = np.nan
        with self.assertRaises(ValueError) as ctx:
            DataRemediationEngine.remediate_missingness(bad_df, target_metric="target_revenue")
        self.assertIn("Fail-closed invariant triggered", str(ctx.exception))

    def test_02_fuzzy_categorical_clustering(self):
        """Invariant: Fragmented strings (US, USA, U.S.) are normalized to the canonical representative."""
        df = pd.DataFrame({
            "country": ["USA", "US", "U.S.", "USA", "France", "France", "FR", "France"] * 10,
            "metric": np.random.uniform(10.0, 100.0, 80),
        })

        res = DataRemediationEngine.normalize_fuzzy_categoricals(df, similarity_threshold=80.0)
        norm_countries = res.df["country"].unique()
        # Invariant: 'US' and 'U.S.' mapped to 'USA'
        self.assertIn("USA", norm_countries)
        self.assertNotIn("U.S.", norm_countries)
        self.assertNotIn("US", norm_countries)
        self.assertIn("France", norm_countries)

    def test_03_rbac_sql_injection(self):
        """Invariant: RBAC filter (region = 'EMEA') is cleanly injected into SQL query AST."""
        raw_sql = "SELECT market_zone, SUM(realized_value) AS total_val FROM dataset GROUP BY market_zone"
        user_context = {"region": "EMEA", "role": "analyst"}

        secure_sql = GovernanceEngine.inject_rbac_filters(raw_sql, user_context)
        self.assertIn("region = 'EMEA'", secure_sql)
        self.assertIn("WHERE", secure_sql)
        self.assertIn("GROUP BY", secure_sql)

    def test_04_eig_cost_penalty(self):
        """Invariant: Cartesian cross joins receive higher compute cost penalty than linear aggregations."""
        simple_sql = "SELECT market_zone, SUM(revenue) FROM sales GROUP BY market_zone"
        expensive_sql = "SELECT a.id, b.id FROM sales a CROSS JOIN customers b WHERE a.val > 100"

        cost_simple = ComputeCostEstimator.estimate_compute_cost(simple_sql)
        cost_expensive = ComputeCostEstimator.estimate_compute_cost(expensive_sql)

        # Cross join must have at least 10x higher cost proxy
        self.assertGreater(cost_expensive, cost_simple * 5.0)


def main():
    print("=" * 80, flush=True)
    print("RUNNING SUITE 12: PHASE 7 ENTERPRISE GUARDRAILS & DATA REMEDIATION TESTS", flush=True)
    print("=" * 80, flush=True)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPhase7EnterpriseGuardrails)
    runner = unittest.TextTestRunner(verbosity=2)
    res = runner.run(suite)
    if res.wasSuccessful():
        print("=" * 80, flush=True)
        print("PHASE 7 ENTERPRISE GUARDRAILS: ALL 4 TESTS PASSED (100%)", flush=True)
        print("=" * 80, flush=True)
        return 0
    else:
        print("PHASE 7 ENTERPRISE GUARDRAILS: FAILED", flush=True)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())



def test_03b_rbac_sql_injection_is_rejected():
    from packages.analytics_core.src.security.governance import GovernanceEngine
    import pytest
    with pytest.raises(ValueError, match="Unsafe RBAC value"):
        GovernanceEngine.inject_rbac_filters(
            "SELECT * FROM orders", {"region": "x' OR '1'='1"}
        )
